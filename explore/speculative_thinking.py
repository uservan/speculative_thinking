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
from speculative_model import spe_thinking_model
from vllm import LLM, SamplingParams
import signal

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise Exception('Time out')

def run_with_timeout(func, timeout, *args, **kwargs):
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)  # 设置超时时间（秒）
    try:
        result = func(*args, **kwargs)
    finally:
        signal.alarm(0)  # 取消定时器

    return result

def check_math_correctness(ref, generation):
    if not find_box(generation): return False
    answer = strip_answer_string(ref)
    pred = extract_answer(generation)
    pred = strip_answer_string(pred)
    return math_equal(pred, answer)

def load_train_data(choose_data_name):
    split = 'train' if choose_data_name not in ['MATH500'] else 'test'
    choose_data_name = dataset_names[choose_data_name]
    choose_data = load_data(choose_data_name, 'huggingface')
    choose_data = choose_data[split]
    if 'MATH500' in choose_data_name or 'aime' in choose_data_name:
        new_column_names = {"problem": "question"}
    choose_data = choose_data.rename_columns(new_column_names)
    return choose_data

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=250)
    parser.add_argument('--dataset', type=str, default='MATH500')
    parser.add_argument('--target_model', type=str, default='deepseek-32b')
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
    'deepseek-1.5b': "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
     'deepseek-7b': "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    'Qwen-math-1.5b':'Qwen/Qwen2.5-Math-1.5B'
}


def process_sample(idx, sample, spe_model:spe_thinking_model):
    """处理单个样本，并返回索引，确保顺序"""
    question, answer = sample["question"], sample["answer"]
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({
        "role": "user",
        "content": "Please reason step by step, and put your final answer within \\boxed{{}}. " + question + ' <think>\n'
    })
    start_time = time.time()  # 记录开始时间
    if spe_model.speculative_model is not None:
        generated_text, num_tokens, correct_tokens,try_correct_num = spe_model.speculative_generate(
            messages=messages, max_tokens=args.max_tokens,max_target_tokens=args.speculative_k, 
            begin=False,begin_token_num=100, recap_after_negtive_num =15, original_recap_token_num=150, 
            max_recap_token_num=200, add_each_recap=25,add_each_neg=5, max_negtive_num=30,
            temperature=args.temperature, top_p=args.topp
        )
    else:
        generated_ids = spe_model.tokenizer.apply_chat_template(messages, return_tensors="np").tolist()[0]
        prompt_len = len(generated_ids)
        sampling_params = SamplingParams(max_tokens=1024, temperature=args.temperature, top_k=50, top_p=args.top_p, 
                                            skip_special_tokens=False)
        spe_ids = list(spe_model.target_model.generate(
                        generated_ids, sampling_params=sampling_params)[0].outputs[0].token_ids)
        generated_text = spe_model.tokenizer.decode(spe_ids, skip_special_tokens=True)
        num_tokens, correct_tokens, try_correct_num = len(spe_ids)-prompt_len, [], 0
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


if __name__ == '__main__':
    args = parse_args()
    seed_everything(args.seed)


# **触发 Token & 关键词**
TRIGGER_TOKENS = ["\n\n"]  # 遇到这些 token 触发 Target Model
TARGET_VALIDATION_KEYWORDS = {'verify':['verify', 'think again', 'recap', 'double-check'],
                              "negative":['but', 'wait', "alternatively", 'hold on','another'],
                              "positive":['yeah','yes','final answer','confident']}  # 目标关键字
system_prompt = None

# **加载 Speculative Model 和 Target Model**
target_model_name = models_names[args.target_model]  # 目标模型
speculative_model_name = models_names[args.speculative_model]

help_think_word = None 
# Let me summarize and recap to make sure I didn't make any mistakes
# Let me shortly summarize and check previous thoughts to make sure I didn't make any mistakes
help_recap_words = "Let me check whether there are some wrong steps "

model = spe_thinking_model(target_model_name, speculative_model_name, TRIGGER_TOKENS,
                               TARGET_VALIDATION_KEYWORDS, help_think_word, help_recap_words, 2,1)

datasets = args.dataset.split(',')
start,end = args.start, args.end
for dataset in datasets:
    math500_dataset = load_train_data(dataset).select(range(start,end))
    output_file = f"./results/{dataset}_{args.target_model}_{args.speculative_model}_{start}_{end}_choose.json"

    results = read_saved_results(output_file)
    idxs = {r['index'] for r in results}
    remaining_data = math500_dataset.select(range(len(results),len(math500_dataset)))

    results_list = []  # 用于存储返回的 future 结果
    idx = len(results) 
    for sample in tqdm(remaining_data):
        result = process_sample(idx, sample, model)
        results_list.append(result) 
        save_results(output_file, result)
        idx = idx+1
