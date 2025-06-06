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
from generate import generate_with_partial_kv
import copy

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
    speculative_model, target_model, tokenizer, input_ids, help_think_word_ids,
    max_tokens=100, max_target_tokens=10, temperature=1.0, top_k=50, top_p=0.95
):
    spec_kv=None
    tgt_kv=None
    device = input_ids.device
    prompt_len = input_ids.shape[1]
    generated_ids = input_ids  # 直接存储 token ids
    correct_tokens = []
    begin = False
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
            if begin or help_think_word_ids is None:
                cache_generated_ids = generated_ids
            else:
                cache_generated_ids = torch.cat([generated_ids, help_think_word_ids], dim=-1)
            # **Step 3: 目标模型生成 max_target_tokens 个 token**
            tgt_new_ids, tgt_kv_candidate = generate_with_partial_kv(
                target_model, tokenizer, cache_generated_ids, copy.deepcopy(tgt_kv),
                max_new_tokens=max_target_tokens, temperature=temperature, top_k=top_k, top_p=top_p
            )
            spe_new_ids, spec_kv_candidate = generate_with_partial_kv(
                speculative_model, tokenizer, cache_generated_ids, copy.deepcopy(spec_kv),
                max_new_tokens=max_target_tokens, temperature=temperature, top_k=top_k, top_p=top_p
            )
            # **解码 Target Model 生成的文本**
            tgt_decoded_text = tokenizer.decode(tgt_new_ids[0,-max_target_tokens:], skip_special_tokens=True)
            spe_decoded_text = tokenizer.decode(spe_new_ids[0,-max_target_tokens:], skip_special_tokens=True)
            # show_decoded_text = tokenizer.decode(tgt_new_ids[0], skip_special_tokens=True)
            # **Step 4: 目标模型生成的文本是否包含关键字**
            correct_flag = False
            if tgt_decoded_text != spe_decoded_text:
                if any(trigger in tgt_decoded_text.lower() for trigger in TARGET_VALIDATION_KEYWORDS['positive']):
                    correct_flag = True
                if not any(trigger in tgt_decoded_text.lower() for trigger in TARGET_VALIDATION_KEYWORDS['negative']) \
                    and any(trigger in spe_decoded_text.lower() for trigger in TARGET_VALIDATION_KEYWORDS['negative']):
                    correct_flag = True
            if correct_flag:
                correct_tokens.append({
                    'pos': cache_generated_ids.shape[1]-prompt_len, 
                    'traget':tgt_decoded_text, 'speculative':spe_decoded_text})
                generated_ids = tgt_new_ids # torch.cat([cache_generated_ids, tgt_new_ids[:, :]], dim=-1)  # ✅ 接受 Target Model 结果
                tgt_kv = tgt_kv_candidate  # ✅ 只有在接受 Target Model 结果时才更新 `tgt_kv`
            else:
                generated_ids = spe_new_ids
                spec_kv = spec_kv_candidate
            begin = False
        
        if tgt_kv is not None and tgt_kv[0][0].shape[2] > len(generated_ids[0]):
            print("wrong")
            
        # **Step 5: 终止条件**
        # last_token_id = generated_ids[0, -1].item()
        if tokenizer.eos_token_id in generated_ids[0, -max_target_tokens:].tolist():  # ✅ 让 `generate()` 处理终止
            break

    # **最终解码文本**
    generated_text = tokenizer.decode(generated_ids[0,prompt_len:], skip_special_tokens=True)
    return generated_text, len(generated_ids[0])-prompt_len, correct_tokens


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='MATH500')
    parser.add_argument('--target_model', type=str, default='deepseek-7b')
    parser.add_argument('--speculative_model', type=str, default='deepseek-1.5b') 
    parser.add_argument('--speculative_k', type=int, default=15)
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
    'deepseek-7b': "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    'deepseek-1.5b': "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
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
TARGET_VALIDATION_KEYWORDS = {"negative":["wait", "hmm", "alternatively",'double-check', 'hold on'],
                              "positive":['yeah','yes', 'alright', 'final answer']}  # 目标关键字
help_think_word = None # '\n\n'
system_prompt = None
device = "cuda" if torch.cuda.is_available() else "cpu"

# **加载 Speculative Model 和 Target Model**
target_model_name = models_names[args.target_model]  # 目标模型
speculative_model_name = models_names[args.speculative_model]  # Speculative Model (更小的模型)

tokenizer = AutoTokenizer.from_pretrained(target_model_name)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id 
target_model = AutoModelForCausalLM.from_pretrained(target_model_name, torch_dtype=torch.float16, device_map="auto")
speculative_model = AutoModelForCausalLM.from_pretrained(speculative_model_name, torch_dtype=torch.float16, device_map="auto")
help_think_word_ids = None if help_think_word is None else tokenizer([help_think_word], return_tensors="pt").input_ids.to("cuda")

datasets = args.dataset.split(',')
for dataset in datasets:
    math500_dataset = load_train_data(dataset)
    output_file = f"./results/{dataset}_{args.target_model}_{args.speculative_model}.json"

    results = read_saved_results(output_file)
    for sample in tqdm(math500_dataset.select(range(len(results), len(math500_dataset)))):
        question, answer = sample["question"], sample["answer"]
        messages = []
        if system_prompt: 
            messages.append({"role": "system", "content": system_prompt})
        messages.append( 
            {"role": "user", "content": "Please reason step by step, and put your final answer within \\boxed{{}}. " + question + ' <think>\n'}
        )
        input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)
    
        start_time = time.time()  # 记录开始时间
        if speculative_model is not None:
           generated_text, num_tokens, correct_tokens = speculative_generate(
                speculative_model, target_model, tokenizer, input_ids, help_think_word_ids,
                max_target_tokens=args.speculative_k, max_tokens=args.max_tokens
            )
        else:
            pass
        end_time = time.time()  # 记录结束时间
        generation_time = end_time - start_time  # 计算生成时间

        results.append({
            "question": question,
            "generated_text": generated_text,
            "answer": answer,
            "corrected_tokens": correct_tokens,
            "generation_time": generation_time,  # 记录时间,
            "num_tokens":num_tokens
        })
        
        save_results(output_file, results[-1])  # 保存结果


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