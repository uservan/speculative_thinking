import __init__
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2ForCausalLM
import json
from utils.data_utils import load_data,save_all_results, read_saved_results, save_results
from utils.utils import *
from tqdm import tqdm
import time
import argparse
import pickle
from generate import generate_with_partial_kv, generate_hf
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.qwen_math_parser import *

def check_math_correctness(ref, generation):
    if not find_box(generation): return False
    answer = strip_answer_string(ref)
    pred = extract_answer(generation)
    pred = strip_answer_string(pred)
    return math_equal(pred, answer)

def sentiment_analysis(text, positive_words, negative_words):
    positive_count = 0
    negative_count = 0
    last_pos_index = -1
    last_neg_index = -1

    text = text.lower()  # 转换为小写以确保匹配不区分大小写
    for word in positive_words:
        if word in text:
            positive_count += text.count(word)  # 统计出现次数
            last_pos_index = max(last_pos_index, text.rfind(word))  # 记录最后出现的位置

    for word in negative_words:
        if word in text:
            negative_count += text.count(word)  # 统计出现次数
            last_neg_index = max(last_neg_index, text.rfind(word))  # 记录最后出现的位置

    if positive_count > negative_count:
        return 1  # 正向
    elif negative_count > positive_count:
        return -1  # 负向
    elif positive_count == negative_count and positive_count > 0:
        return -1 # 1 if last_pos_index > last_neg_index else -1  # 取最靠后的词类别
    else:
        return 0  # 中立

def load_train_data(choose_data_name):
    split = 'train' if choose_data_name not in ['MATH500'] else 'test'
    choose_data_name = dataset_names[choose_data_name]
    choose_data = load_data(choose_data_name, 'huggingface')
    choose_data = choose_data[split]
    if 'MATH500' in choose_data_name or 'aime' in choose_data_name:
        new_column_names = {"problem": "question"}
    choose_data = choose_data.rename_columns(new_column_names)
    return choose_data

def contains_keywords(text, keywords):
    return any(keyword in text.lower() for keyword in keywords)

def speculative_generate(
    speculative_model, target_model, tokenizer, input_ids, help_think_word_ids,help_recap_words_ids,
    max_tokens=100, max_target_tokens=10, temperature=1.0, top_k=50, top_p=0.95
):
    try_correct_num = 0
    spec_kv=None
    tgt_kv=None
    prompt_len = input_ids.shape[1]
    generated_ids = input_ids  # 直接存储 token ids
    correct_tokens = []
    begin,change_flag = False, False
    negative_sent_num, recap_after_negtive_num, original_recap_token_num, begin_token_num, add_each_recap = 0, 10, 100, 100, 50
    recap_token_num = original_recap_token_num
    while generated_ids.shape[1] < max_tokens:  # **不再手动检查 max_tokens**
        if not begin:
            # **Step 1: Speculative Model 逐个生成 1 个 token**
            new_ids, spec_kv = generate_with_partial_kv(
                speculative_model, tokenizer, generated_ids, spec_kv,
                max_new_tokens=1, temperature=temperature, top_k=top_k, top_p=top_p
            )
            # **更新 token 序列**
            generated_ids = new_ids # torch.cat([generated_ids, new_ids[:, -1:]], dim=-1)
            # **Step 2: 仅解码新 token 并检查是否触发 Target Model**
            decoded_text = tokenizer.decode(new_ids[0, -1:], skip_special_tokens=True)
            
        if begin or any(trigger in decoded_text for trigger in TRIGGER_TOKENS):
            # 1. 判断负向次数是否过多
            # 2. 判断是否begin是否需要处理
            # 3. 判断是否出现反相
            if begin:
                change_tokens = begin_token_num
                begin = False
                change_flag = True
            # elif negative_sent_num >= recap_after_negtive_num:
            #     generated_ids = torch.cat([generated_ids, help_recap_words_ids], dim=-1)
            #     change_tokens = recap_token_num
            #     change_flag = True
            #     negative_sent_num = 0
            #     recap_token_num, recap_after_negtive_num= min(recap_token_num + add_each_recap, 500), min(recap_after_negtive_num+15, 50)
            else:
                if help_think_word_ids is not None:
                    cache_generated_ids = torch.cat([generated_ids, help_think_word_ids], dim=-1)
                else:
                    cache_generated_ids = generated_ids
                spe_new_ids, spec_kv_candidate = generate_with_partial_kv(
                    speculative_model, tokenizer, cache_generated_ids, copy.deepcopy(spec_kv),
                    max_new_tokens=max_target_tokens, temperature=temperature, top_k=top_k, top_p=top_p
                )
                spe_decoded_text = tokenizer.decode(spe_new_ids[0,-max_target_tokens:], skip_special_tokens=True)
                spe_sent = sentiment_analysis(spe_decoded_text, TARGET_VALIDATION_KEYWORDS['positive'], TARGET_VALIDATION_KEYWORDS['negative']+TARGET_VALIDATION_KEYWORDS['verify'])
                if spe_sent != 0:
                    try_correct_num = try_correct_num+1
                    # **Step 3: 目标模型生成 max_target_tokens 个 token**
                    tgt_new_ids, tgt_kv_candidate = generate_with_partial_kv(
                        target_model, tokenizer, cache_generated_ids, copy.deepcopy(tgt_kv),
                        max_new_tokens=max_target_tokens, temperature=temperature, top_k=top_k, top_p=top_p
                    )
                    # **解码 Target Model 生成的文本**
                    tgt_decoded_text = tokenizer.decode(tgt_new_ids[0,-max_target_tokens:], skip_special_tokens=True)
                    tgt_sent = sentiment_analysis(tgt_decoded_text, TARGET_VALIDATION_KEYWORDS['positive'], TARGET_VALIDATION_KEYWORDS['negative']+TARGET_VALIDATION_KEYWORDS['verify'])
                    if True: #(spe_sent<0 and tgt_sent >=0) or (spe_sent>0 and tgt_sent<0):
                        generated_ids = tgt_new_ids # torch.cat([cache_generated_ids, tgt_new_ids[:, :]], dim=-1)  # ✅ 接受 Target Model 结果
                        tgt_kv = tgt_kv_candidate  # ✅ 只有在接受 Target Model 结果时才更新 `tgt_kv`
                        decode_text = tgt_decoded_text
                        correct_tokens.append({
                            'pos': cache_generated_ids.shape[1]-prompt_len, 'token_num':max_target_tokens,
                            'traget':tgt_decoded_text, 'speculative':spe_decoded_text})
                        final_sent=tgt_sent
                    else:
                        generated_ids = spe_new_ids # torch.cat([cache_generated_ids, tgt_new_ids[:, :]], dim=-1)  # ✅ 接受 Target Model 结果
                        spec_kv = spec_kv_candidate  # ✅ 只有在接受 Target Model 结果时才更新 `tgt_kv`
                        decode_text = spe_decoded_text
                        final_sent=spe_sent
                    if final_sent < 0: negative_sent_num = negative_sent_num+1
                    if contains_keywords(decode_text, TARGET_VALIDATION_KEYWORDS['verify']):
                        change_tokens = original_recap_token_num
                        change_flag = True
                        # negative_sent_num = 0
            # if change_flag:
            #     try_correct_num = try_correct_num+1
            #     tgt_new_ids, tgt_kv_candidate = generate_with_partial_kv(
            #         target_model, tokenizer, generated_ids, copy.deepcopy(tgt_kv_candidate),
            #         max_new_tokens=change_tokens, temperature=temperature, top_k=top_k, top_p=top_p
            #     )
            #     tgt_decoded_text = tokenizer.decode(tgt_new_ids[0,-change_tokens:], skip_special_tokens=True)
            #     generated_ids = tgt_new_ids
            #     tgt_kv = tgt_kv_candidate
            #     correct_tokens.append({
            #         'pos': generated_ids.shape[1]-prompt_len, 'token_num':change_tokens,
            #         'traget':tgt_decoded_text, 'speculative':spe_decoded_text})
            #     change_flag = False
        
        if tgt_kv is not None and tgt_kv[0][0].shape[2] > len(generated_ids[0]):
            print("wrong")
            
        # **Step 5: 终止条件**
        # last_token_id = generated_ids[0, -1].item()
        if tokenizer.eos_token_id in generated_ids[0, -max_target_tokens:].tolist():  # ✅ 让 `generate()` 处理终止
            break

    # **最终解码文本**
    generated_text = tokenizer.decode(generated_ids[0,prompt_len:], skip_special_tokens=True)
    return generated_text, len(generated_ids[0])-prompt_len, correct_tokens, try_correct_num


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=90)
    parser.add_argument('--dataset', type=str, default='AIME')
    parser.add_argument('--target_model', type=str, default='deepseek-32b')
    parser.add_argument('--speculative_model', type=str, default='deepseek-1.5b') 
    parser.add_argument('--speculative_k', type=int, default=20)
    parser.add_argument('--accept_prob', type=float, default=0.01)
    parser.add_argument('--temperature', type=float, default=0.6)
    parser.add_argument('--topp', type=float, default=0.95)
    parser.add_argument('--topk', type=float, default=0)
    parser.add_argument('--max_tokens', type=int, default=1024*32)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args(args)

dataset_names = {
    'MATH500': "qq8933/MATH500",
    'AIME': "AI-MO/aimo-validation-aime"
}

models_names = {
    'deepseek-32b': "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    'deepseek-1.5b': "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
     'deepseek-7b': "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    'Qwen-math-1.5b':'Qwen/Qwen2.5-Math-1.5B'
}


if __name__ == '__main__':
    args = parse_args()
    seed_everything(args.seed)
    # accept_prob = args.accept_prob
    # # **加载 Hugging Face 模型（支持 KV Cache + 多 GPU）**
    # target_model_name = models_names[args.target_model]  
    # target_model = AutoModelForCausalLM.from_pretrained(
    #     target_model_name, torch_dtype=torch.float16,device_map="auto", low_cpu_mem_usage=True, attn_implementation="flash_attention_2",
    # )
    # tokenizer = AutoTokenizer.from_pretrained(target_model_name)
    # speculative_model = None
    # if args.speculative_model:
    #     speculative_model_name = models_names[args.speculative_model] 
    #     speculative_model = AutoModelForCausalLM.from_pretrained(
    #         speculative_model_name,  torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True,attn_implementation="flash_attention_2",
    #     )
   

# **触发 Token & 关键词**
TRIGGER_TOKENS = {"\n\n"}  # 遇到这些 token 触发 Target Model
TARGET_VALIDATION_KEYWORDS = {'verify':['verify', 'think again', 'recap', 'check'],
                              "negative":['but', 'wait', "alternatively", 'hold on','another'],
                              "positive":['yeah','yes','final answer','confident']}  # 目标关键字
help_think_word = None # '\n\n'
system_prompt = None
device = "cuda" if torch.cuda.is_available() else "cpu"
# **加载 Speculative Model 和 Target Model**
target_model_name = models_names[args.target_model]  # 目标模型

tokenizer = AutoTokenizer.from_pretrained(target_model_name)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id 
target_model = [ AutoModelForCausalLM.from_pretrained(target_model_name, torch_dtype=torch.float16, device_map="auto") for i in range(1)]
speculative_model = [None]
if args.speculative_model is not None: 
    speculative_model_name = models_names[args.speculative_model]  # Speculative Model (更小的模型)
    speculative_model = [AutoModelForCausalLM.from_pretrained(speculative_model_name, torch_dtype=torch.float16, device_map="auto") for i in range(1)]
help_think_word_ids = None if help_think_word is None else tokenizer([help_think_word], return_tensors="pt").input_ids.to("cuda")
# Let me summarize and recap to make sure I didn't make any mistakes
help_recap_words_ids = tokenizer(["Let me shortly summarize and check previous thoughts to make sure I didn't make any mistakes"], return_tensors="pt").input_ids.to("cuda")
datasets = args.dataset.split(',')

start,end = args.start, args.end

for dataset in datasets:
    math500_dataset = load_train_data(dataset).select(range(start,end))
    output_file = f"./results/{dataset}_{args.target_model}_{args.speculative_model}_new_{start}_{end}.json"

    results = read_saved_results(output_file)
    idxs = {r['index'] for r in results}
    remaining_data = math500_dataset# .select(range(len(math500_dataset)))

    def process_sample(idx, sample):
        """处理单个样本，并返回索引，确保顺序"""
        question, answer = sample["question"], sample["answer"]
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({
            "role": "user",
            "content": "Please reason step by step, and put your final answer within \\boxed{{}}. " + question + ' <think>\n'
        })
        input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)

        start_time = time.time()  # 记录开始时间
        speculative_model_, target_model_ = speculative_model[0], target_model[0]
        if speculative_model_ is not None:
           generated_text, num_tokens, correct_tokens,try_correct_num = speculative_generate(
                speculative_model_, target_model_, tokenizer, input_ids, help_think_word_ids,help_recap_words_ids,
                max_target_tokens=args.speculative_k, max_tokens=args.max_tokens,
                temperature=args.temperature, top_p=args.topp
            )
        else:
            prompt_len = input_ids.shape[1]
            generated_ids_hf = generate_hf(target_model_, tokenizer, input_ids, args.max_tokens,
                        temperature=args.temperature, top_p=args.topp)
            generated_text = tokenizer.decode(generated_ids_hf[0,prompt_len:], skip_special_tokens=True)
            num_tokens, correct_tokens,try_correct_num = None, None, None
        end_time = time.time()  # 记录结束时间
        generation_time = end_time - start_time  # 计算生成时间
        return {
            "index": idx,  # 记录原始索引，确保排序
            "question": question,
            "generated_text": generated_text,
            "answer": answer,
            "corrected_tokens": correct_tokens,
            "generation_time": generation_time,
            "num_tokens": num_tokens,
            "try_correct_num": try_correct_num
        }

    # 线程池并行处理
    max_workers = min(4, os.cpu_count())  # 限制最大线程数，防止超载
    print(f"🚀 自动分配线程数: {max_workers}")
    # max_workers = 8  # 根据显存情况调整
    results_list = []  # 用于存储返回的 future 结果
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_sample, idx, sample): idx for idx, sample in enumerate(remaining_data) if idx not in idxs}

        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                result = future.result()
                right_flag = check_math_correctness(result['answer'], result['generated_text'])
                print(start+result['idx']+1, ": ", right_flag)
                results_list.append(result)  # 先存储，保证结果完整
            except Exception as e:
                print(f"Error processing sample {futures[future]}: {e}")

    # **确保最终结果按索引排序**
    results_list = sorted(results_list, key=lambda x: x["index"])

    # **按顺序存入 results 并保存**
    for res in results_list:
        results.append(res)
        save_results(output_file, res)

print("所有任务完成！")



# # **初始化输入**
# prompt = "Once upon a time"
# input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
# help_think_word_ids = tokenizer([help_think_word], return_tensors="pt").input_ids.to("cuda")
# # **执行 Speculative Mode 生成**
# generated_text = speculative_generate(
#     speculative_model, target_model, tokenizer, input_ids, help_think_word_ids,
#     max_target_tokens=10
# )

# print("Final Generated Text:", generated_text)