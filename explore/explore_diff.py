import torch
import torch.nn.functional as F
from vllm import LLM, SamplingParams
import json

def load_dataset(filepath):
    """通用加载数据集方法"""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def speculative_decoding(target_model, speculative_model, tokenizer, question, system_prompt, max_tokens=50, k=3):
    """
    通用 Speculative Decoding 方法:
    - speculative_model 预测 K 个 token
    - target_model 计算概率，并进行拒绝采样
    记录哪些 token 被 target_model 修复，以及它们的位置。
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]
    
    # **Step 1: Speculative Model 生成 K 个 token**
    speculative_sampling_params = SamplingParams(max_tokens=k, temperature=0.7)
    speculative_output = speculative_model.chat(messages, speculative_sampling_params)
    speculative_tokens = tokenizer(speculative_output["choices"][0]["message"]["content"], return_tensors="pt").input_ids.tolist()[0]
    
    # **Step 2: 计算大模型的 token 概率**
    target_input_ids = tokenizer(question, return_tensors="pt").input_ids.tolist()[0] + speculative_tokens
    target_sampling_params = SamplingParams(max_tokens=len(speculative_tokens))
    target_output = target_model.chat(messages, target_sampling_params)
    target_tokens = tokenizer(target_output["choices"][0]["message"]["content"], return_tensors="pt").input_ids.tolist()[0]
    
    # **Step 3: 进行拒绝采样并记录修复情况**
    accepted_tokens = []
    corrected_tokens = []
    for i in range(len(speculative_tokens)):
        speculative_token = speculative_tokens[i]
        target_pred_token = target_tokens[i] if i < len(target_tokens) else speculative_token
        
        accept_prob = 0.5  # 这里假设一个固定接受概率，实际实现时应基于 logits 计算
        if torch.rand(1).item() < accept_prob:
            accepted_tokens.append(speculative_token)
        else:
            accepted_tokens.append(target_pred_token)
            corrected_tokens.append({"position": i, "speculative": tokenizer.decode([speculative_token]), "target": tokenizer.decode([target_pred_token])})  # 记录修复的 token 及其位置
            break  
    
    return tokenizer.decode(accepted_tokens, skip_special_tokens=True), corrected_tokens

def generate_answers(target_model, speculative_model, tokenizer, dataset, system_prompt, output_file):
    """针对数据集批量生成答案，并保存修复的 token"""
    results = []
    for sample in dataset:
        question = sample["question"]
        response_text, corrected_tokens = speculative_decoding(target_model, speculative_model, tokenizer, question, system_prompt)
        results.append({
            "question": question,
            "generated_answer": response_text,
            "corrected_tokens": corrected_tokens
        })
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

# **加载 vLLM**
target_model_name = "meta-llama/Llama-2-7b-chat-hf"
speculative_model_name = "mistralai/Mistral-7B-v0.1"

target_model = LLM(model=target_model_name, dtype="float16", tensor_parallel_size=1)
speculative_model = LLM(model=speculative_model_name, dtype="float16", tensor_parallel_size=1)
tokenizer = target_model.tokenizer

# **加载不同数据集并生成答案，保存修复信息**
math500_dataset = load_dataset("math500.json")
gsm8k_dataset = load_dataset("gsm8k.json")

generate_answers(target_model, speculative_model, tokenizer, math500_dataset, "You are an expert in solving advanced mathematics problems. Provide clear and precise solutions.", "math500_results.json")
generate_answers(target_model, speculative_model, tokenizer, gsm8k_dataset, "You are a helpful AI tutor providing step-by-step explanations to math word problems.", "gsm8k_results.json")
