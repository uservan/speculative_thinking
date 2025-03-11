from __init__ import *
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2ForCausalLM
import json
from utils.data_utils import load_data,read_yml, read_saved_results, save_results
from utils.utils import *
from tqdm import tqdm
import time
import argparse
import pickle
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.qwen_math_parser import *
from speculative.speculative_vllm import spe_thinking_vllm
from speculative.speculative_hf import spe_thinking_hf
from vllm import LLM, SamplingParams
import multiprocessing
import gc


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

def load_spe_model(path):
    config = read_yml(path)
    if config['mode'] == 'vllm':
        return spe_thinking_vllm(**config)
    if config['mode'] == 'hf':
        return spe_thinking_hf(**config)
    
def process_message(message, spe_model:spe_thinking_vllm|spe_thinking_hf, 
                    max_tokens=100, temperature=0.6, top_k=50, top_p=0.95):
    start_time = time.time()  
    result = spe_model.generate(messages=message, max_tokens=max_tokens, temperature=temperature, top_k=top_k, top_p=top_p)
    if result is None: return None
    generated_text, num_tokens, correct_tokens, try_correct_num = result
    end_time = time.time() 
    return {
        'generated_text':generated_text, 
        'num_tokens': num_tokens, 
        'correct_tokens':correct_tokens, 
        'try_correct_num':try_correct_num,
        'generation_time': end_time - start_time 
    }

def process_sample(idx, sample, spe_model:spe_thinking_vllm|spe_thinking_hf,
                    max_tokens=100, temperature=0.6, top_k=50, top_p=0.95):
    question, answer = sample["question"], sample["answer"]
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({
        "role": "user",
        "content": "Please reason step by step, and put your final answer within \\boxed{{}}. " + question + ' <think>\n'
    })
    results = process_message(messages, spe_model, max_tokens=max_tokens, temperature=temperature, top_k=top_k, top_p=top_p)
    return {
        "index": idx, 
        "question": question,
        "generated_text": results['generated_text'],
        "answer": answer,
        "corrected_tokens": results['correct_tokens'],
        "generation_time": results['generation_time'],
        "num_tokens": results['num_tokens'],
        "try_correct_num": results['try_correct_num']
    }

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=250)
    parser.add_argument('--dataset', type=str, default='MATH500')
    parser.add_argument('--path', type=str, default='/home/wxy320/ondemand/program/speculative_thinking/speculative/spe_setting.yml')
    parser.add_argument('--temperature', type=float, default=0.6)
    parser.add_argument('--topp', type=float, default=0.95)
    parser.add_argument('--topk', type=float, default=50)
    parser.add_argument('--max_tokens', type=int, default=1024*32)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args(args)

if __name__ == '__main__':
    args = parse_args()
    seed_everything(args.seed)
    model = load_spe_model(args.path)
    system_prompt = None
    datasets = args.dataset.split(',')
    start,end = args.start, args.end
    for dataset in datasets:
        output_file = f"./results/{dataset}_{start}_{end}_{args.path.split('/')[-1]}.json"
        math500_dataset = load_train_data(dataset).select(range(start,end))
        results = read_saved_results(output_file)
        remaining_data = math500_dataset.select(range(len(results),len(math500_dataset)))
        idx = len(results) 
        for sample in tqdm(remaining_data):
            result = process_sample(idx, sample, model, max_tokens=args.max_tokens, temperature=args.temperature, top_k=args.topk, top_p=args.topp)
            save_results(output_file, result)
            idx = idx+1
