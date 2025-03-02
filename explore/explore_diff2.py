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

accept_prob = 0.01


def load_train_data(choose_data_name):
    split = 'train' if choose_data_name not in ['MATH500'] else 'test'
    choose_data_name = dataset_names[choose_data_name]
    choose_data = load_data(choose_data_name, 'huggingface')
    choose_data = choose_data[split]
    if 'MATH500' in choose_data_name or 'aime' in choose_data_name:
        new_column_names = {"problem": "question"}
    choose_data = choose_data.rename_columns(new_column_names)
    return choose_data

def sample_logits(logits, temperature=1.0, top_k=0, top_p=0.95):
    if temperature > 0:
        logits = logits / temperature
    
    # 进行 top-k 采样
    if top_k > 0:
        top_k_values, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < top_k_values[:, -1]] = -float('Inf')
    
    # 进行 top-p (nucleus) 采样
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = False  # 保证至少有一个 token 可选
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = -float('Inf')
    
    # 归一化概率后采样
    probabilities = F.softmax(logits, dim=-1)
    # sampled_token = torch.multinomial(probabilities, 1)
    
    return probabilities

# # 示例使用
# logits = torch.tensor([1.0, 2.0, 3.0, 4.0])  # 假设的 logits
# sampled_token = sample_logits(logits, temperature=0.7, top_k=2, top_p=0.9)
# print("Sampled token index:", sampled_token)



def choose_prob_all(target_probs, accepted_tokens, speculative_tokens, target_pred_tokens, corrected_tokens, prompt_len):
    global accept_prob
    sorted_probs, sorted_indices = torch.sort(target_probs[0], dim=-1, descending=True)
    mask = sorted_probs >= accept_prob
    for i, (idx, p, m) in enumerate(zip(sorted_indices, sorted_probs, mask)):
        speculative_token, target_pred_token = speculative_tokens[i], target_pred_tokens[i]
        filtered_probs, filtered_indices = p[m], idx[m].tolist() 
        if len(filtered_indices) == 0 or speculative_token in filtered_indices or speculative_token == target_pred_token:
            accepted_tokens.append(speculative_token)
        else:
            accepted_tokens.append(target_pred_token)
            corrected_tokens.append({
                "position": len(accepted_tokens) - 1 - prompt_len,
                "speculative": tokenizer.decode([speculative_token]),
                "target": tokenizer.decode([target_pred_token])
            })
            print(f'change from {corrected_tokens[-1]['speculative']} to {corrected_tokens[-1]['target']} at {corrected_tokens[-1]['position']}')
            break

# def choose_prob(target_probs, accepted_tokens, speculative_token, target_pred_token, corrected_tokens):
#     speculative_prob, target_prob = target_probs[:, speculative_token].item(), target_probs[:,target_pred_token].item()
#     accept_prob = speculative_prob / target_prob
#     if torch.rand(1).item() < accept_prob:
#         accepted_tokens.append(speculative_token)
#         return False
#     else:
#         accepted_tokens.append(target_pred_token)
#         corrected_tokens.append({
#             "position": len(accepted_tokens) - 1,
#             "speculative": tokenizer.decode([speculative_token]),
#             "target": tokenizer.decode([target_pred_token])
#         })
#         return True

def decoding_chat(target_model, tokenizer, question, system_prompt=None, 
                  max_tokens=32*1024, temperature=0.6, topp=0.95, topk=50, do_sample=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    record_tokens, avg_probabilities = [], None
    # 构造聊天消息
    messages = []
    if system_prompt: messages.append({"role": "system", "content": system_prompt})
    messages.append( 
        {"role": "user", "content": "Please reason step by step, and put your final answer within \\boxed{{}}. " + question + ' <think>\n'}
    )
    input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)
    target_past_key_values = None
    accepted_tokens = input_ids.tolist()[0]
    speculative_input_ids = torch.tensor([accepted_tokens[:-1]], device=device)
    with torch.no_grad():
        target_outputs = target_model(
            input_ids=speculative_input_ids,
            past_key_values=target_past_key_values,  # ✅ 仅计算新增 token，复用 KV Cache
            return_dict=True,
            output_hidden_states=False,
            output_attentions=False,
            use_cache=True,  # ✅ 维护 target_model 的 KV Cache
        )
    target_past_key_values = target_outputs.past_key_values

    current_ids = torch.tensor([[accepted_tokens[-1]]], device=device)
    while len(accepted_tokens) - len(input_ids[0]) < max_tokens:
        with torch.no_grad():
            outputs = target_model(
                input_ids=current_ids,
                past_key_values=target_past_key_values,  # ✅ 仅计算新增 token，复用 KV Cache
                return_dict=True,
                output_hidden_states=False,
                output_attentions=False,
                use_cache=True,  # ✅ 维护 target_model 的 KV Cache
            )
        next_token_logits = outputs.logits[:, -1, :]
        probabilities = sample_logits(next_token_logits, temperature, top_k=topk, top_p=topp)
        if do_sample:
             next_token = torch.multinomial(probabilities, num_samples=1).squeeze(1)
        else:
            next_token = torch.argmax(next_token_logits, dim=-1)
        
        accepted_tokens.append(next_token.item())
        target_past_key_values = outputs.past_key_values
        current_ids = next_token.unsqueeze(0)

        filtered_indices = torch.where(probabilities > accept_prob)  # 找到大于 0.01 的 token 位置
        filtered_probs = probabilities[filtered_indices].tolist()  # 取出对应的概率
        filtered_tokens = filtered_indices[-1].tolist() # 获取对应的 token 索引
        record_tokens.append(
            {
                'token_id': filtered_tokens,
                'probs' : filtered_probs,
                "token": [tokenizer.decode(t) for t in filtered_tokens]
            }
        )
        if avg_probabilities is None:
            avg_probabilities = probabilities
        else: avg_probabilities = avg_probabilities + probabilities
        # 终止条件检查
        if tokenizer.eos_token_id == accepted_tokens[-1]:
            break

    # 解码最终结果
    output_text = tokenizer.decode(accepted_tokens[len(input_ids[0]):], skip_special_tokens=True)
    num_tokens = len(accepted_tokens[len(input_ids[0]):])
    return output_text, [], num_tokens, record_tokens, avg_probabilities.cpu().numpy()/num_tokens


def speculative_decoding_chat(target_model, speculative_model, tokenizer, question, system_prompt, max_tokens=32*1024, k=5, temperature=0.7):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 构造聊天消息
    messages = [
        # {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Please reason step by step, and put your final answer within \\boxed{{}}. " + question + ' <think>\n'}
    ]
    input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)
    
    accepted_tokens = input_ids.tolist()[0]
    prompt_len = len(accepted_tokens)
    target_past_key_values = None  # target_model的KV Cache
    speculative_past_key_values = None  # speculative_model的KV Cache
    corrected_tokens = []

    def get_k_v_cache(speculative_input_ids, target_past_key_values, speculative_past_key_values):
        with torch.no_grad():
            target_outputs = target_model(
                input_ids=speculative_input_ids,
                past_key_values=target_past_key_values,  # ✅ 仅计算新增 token，复用 KV Cache
                return_dict=True,
                output_hidden_states=False,
                output_attentions=False,
                use_cache=True,  # ✅ 维护 target_model 的 KV Cache
            )
            target_past_key_values = target_outputs.past_key_values
            speculative_outputs= speculative_model(
                input_ids=speculative_input_ids,
                past_key_values=speculative_past_key_values,  # ✅ 仅计算新增 token，复用 KV Cache
                return_dict=True,
                output_hidden_states=False,
                output_attentions=False,
                use_cache=True,  # ✅ 维护 target_model 的 KV Cache
            )
            speculative_past_key_values = speculative_outputs.past_key_values
        return target_past_key_values, speculative_past_key_values
            
    speculative_input_ids = torch.tensor([accepted_tokens[:-1]], device=device)
    target_past_key_values, speculative_past_key_values = get_k_v_cache(speculative_input_ids, target_past_key_values, speculative_past_key_values)

    while len(accepted_tokens) - len(input_ids[0]) < max_tokens:
        speculative_tokens = []
        # ========== Step 1: 用speculative_model生成K个候选 ==========
        current_ids = torch.tensor([[accepted_tokens[-1]]], device=device)
        for _ in range(k):
            with torch.no_grad():
                outputs = speculative_model(
                    input_ids=current_ids,
                    past_key_values=speculative_past_key_values,  # ✅ 仅计算新增 token，复用 KV Cache
                    return_dict=True,
                    output_hidden_states=False,
                    output_attentions=False,
                    use_cache=True,  # ✅ 维护 target_model 的 KV Cache
                )
            next_token_logits = outputs.logits[:, -1, :]
            # 采样下一个token
            if temperature > 0:
                probs = torch.softmax(next_token_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1)
            
            speculative_tokens.append(next_token.item())
            speculative_past_key_values = outputs.past_key_values
            current_ids = next_token.unsqueeze(0)
       
        speculative_input_ids = torch.tensor([[accepted_tokens[-1]]+speculative_tokens[:-1]], device=device)
        with torch.no_grad():
            target_outputs = target_model(
                input_ids=speculative_input_ids,
                past_key_values=target_past_key_values,  # ✅ 仅计算新增 token，复用 KV Cache
                return_dict=True,
                output_hidden_states=False,
                output_attentions=False,
                use_cache=True,  # ✅ 维护 target_model 的 KV Cache
            )
        logits = target_outputs.logits # 只计算新增 token 的 logits
        if temperature > 0:
            target_probs = torch.softmax(logits/temperature, dim=-1)  # 计算概率
            t = target_probs.squeeze(0)
            next_token = torch.multinomial(t, num_samples=1).squeeze(1)
        else:
            target_probs = torch.softmax(logits, dim=-1) 
            next_token = torch.argmax(target_probs, dim=-1).squeeze(1)
        target_past_key_values = target_outputs.past_key_values  # ✅ target_model 维护自己的 KV Cache
        target_pred_tokens = next_token.tolist()
        choose_prob_all(target_probs, accepted_tokens, speculative_tokens, target_pred_tokens, corrected_tokens, prompt_len)
        # stop_flag = False
        # for i in range(k):
        #     speculative_token = speculative_tokens[i]
        #     target_pred_token = torch.argmax(target_probs[:, i, :], dim=-1).item()
        #     stop_flag = choose_prob(target_probs[:, i], accepted_tokens, speculative_token, target_pred_token, corrected_tokens, prompt_len)
        #     if stop_flag:
        #         break # 终止 speculative 采样

        # 更新target_model的KV Cache
        target_past_key_values, speculative_past_key_values = list(target_past_key_values), list(speculative_past_key_values)
        for i in range(max(len(target_past_key_values),len(speculative_past_key_values))):
            if i < len(target_past_key_values):
                target_past_key_values[i] = (target_past_key_values[i][0][:,:, :len(accepted_tokens)-1], target_past_key_values[i][1][:,:, :len(accepted_tokens)-1])
            if i < len(speculative_past_key_values):
                speculative_past_key_values[i] = (speculative_past_key_values[i][0][:,:, :len(accepted_tokens)-1], speculative_past_key_values[i][1][:,:, :len(accepted_tokens)-1])
        target_past_key_values, speculative_past_key_values = tuple(target_past_key_values), tuple(speculative_past_key_values)
        speculative_input_ids = torch.tensor([[accepted_tokens[-1]]], device=device)
        # get_k_v_cache(speculative_input_ids, target_past_key_values, speculative_past_key_values)
            
        # 终止条件检查
        if tokenizer.eos_token_id in accepted_tokens[-k:]:
            break

    # 解码最终结果
    output_text = tokenizer.decode(accepted_tokens[len(input_ids[0]):], skip_special_tokens=False)
    return output_text, corrected_tokens, len(accepted_tokens[len(input_ids[0]):])

def generate_answers(target_model, speculative_model, tokenizer, dataset, system_prompt, output_file, 
                     k, temperature, topp=0.95, topk=50):
    results = read_saved_results(output_file)
    
    for sample in tqdm(dataset.select(range(len(results), len(dataset)))):
        question, answer = sample["question"], sample["answer"]
        
        start_time = time.time()  # 记录开始时间
        if speculative_model is not None:
            response_text, corrected_tokens, num_tokens = speculative_decoding_chat(
                target_model, speculative_model, tokenizer, question, system_prompt, k=k, temperature=temperature
            )
        else:
            response_text, corrected_tokens, num_tokens, record_tokens, avg_probabilities = decoding_chat(
                target_model, tokenizer, question, system_prompt, 
                temperature=temperature, topk=topk, topp=topp
            )
        end_time = time.time()  # 记录结束时间
        
        generation_time = end_time - start_time  # 计算生成时间
        if speculative_model is not None:
            pass
        else:
            data = {
                "avg_probabilities": avg_probabilities,
                "record_tokens": record_tokens
            }
            os.makedirs(f"/scratch/pbsjobs/wxy320/speculative/{args.target_model}", exist_ok=True)
            with open(f"/scratch/pbsjobs/wxy320/speculative/{args.target_model}/data{len(results)}.pkl", "wb") as f:
                pickle.dump(data, f)

        results.append({
            "question": question,
            "generated_answer": response_text,
            "answer": answer,
            "corrected_tokens": corrected_tokens,
            "generation_time": generation_time,  # 记录时间,
            "num_tokens":num_tokens
        })
        
        save_results(output_file, results[-1])  # 保存结果


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

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='MATH500')
    parser.add_argument('--target_model', type=str, default='deepseek-1.5b')
    parser.add_argument('--speculative_model', type=str, default=None) 
    parser.add_argument('--speculative_k', type=int, default=10)
    parser.add_argument('--accept_prob', type=float, default=0.01)
    parser.add_argument('--temperature', type=float, default=0.6)
    parser.add_argument('--topp', type=float, default=0.95)
    parser.add_argument('--topk', type=float, default=0)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args(args)

if __name__ == '__main__':
    args = parse_args()
    seed_everything(args.seed)
    accept_prob = args.accept_prob
    # **加载 Hugging Face 模型（支持 KV Cache + 多 GPU）**
    target_model_name = models_names[args.target_model]  
    target_model = AutoModelForCausalLM.from_pretrained(
        target_model_name, torch_dtype=torch.float16,device_map="auto", low_cpu_mem_usage=True, attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(target_model_name)
    speculative_model = None
    if args.speculative_model:
        speculative_model_name = models_names[args.speculative_model] 
        speculative_model = AutoModelForCausalLM.from_pretrained(
            speculative_model_name,  torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True,attn_implementation="flash_attention_2",
        )
   
    datasets = args.dataset.split(',')
    for dataset in datasets:
        math500_dataset = load_train_data(dataset)
        generate_answers(target_model, speculative_model, tokenizer, math500_dataset, None,
                          f"./results/{dataset}_{args.target_model}_{args.speculative_model}_{accept_prob}.json", 
                          k=args.speculative_k, temperature=args.temperature, topp=args.topp, topk=args.topk)
