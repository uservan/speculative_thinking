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
from vllm import LLM, SamplingParams
import ray

def contains_keywords(text, keywords):
    return any(keyword in text.lower() for keyword in keywords)

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

def create_ray_model(model_name, target_model_gpu):
    @ray.remote(num_gpus=target_model_gpu)
    class ModelWorkerSingleGPU:
        def __init__(self, model_name: str):
            self.model = LLM(model=model_name, tensor_parallel_size=target_model_gpu)
        def generate(self, generated_ids, sampling_params):
            outputs = self.model.generate(
                prompt_token_ids=generated_ids,  # 这里是列表的列表，vLLM要求的格式
                sampling_params=sampling_params,
                use_tqdm=False
            )
            one_token_id = list(outputs[0].outputs[0].token_ids)
            return one_token_id
    return ModelWorkerSingleGPU.remote(model_name)

def get_ray_reuslt(model, generated_ids, sampling_params):
    result_a_future = model.generate.remote(generated_ids, sampling_params)
    token_ids = ray.get(result_a_future)
    return token_ids

class spe_thinking_model:
    def __init__(self, target_model_name, speculative_model_name,
                 TRIGGER_TOKENS, TARGET_VALIDATION_KEYWORDS, help_think_word, help_recap_words,
                target_model_gpu=2, speculative_model_gpu=1, choose_large=True):
        self.tokenizer = AutoTokenizer.from_pretrained(target_model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id 
        self.target_model = create_ray_model(target_model_name, target_model_gpu)
        self.speculative_model = create_ray_model(speculative_model_name, speculative_model_gpu)
        
        self.help_think_word_ids = None if help_think_word is None else self.tokenizer([help_think_word], return_tensors="np",add_special_tokens=False)["input_ids"][0].tolist()
        # Let me summarize and recap to make sure I didn't make any mistakes
        self.help_recap_words_ids = self.tokenizer([help_recap_words], return_tensors="np",add_special_tokens=False)["input_ids"][0].tolist()
        self.TRIGGER_TOKENS = TRIGGER_TOKENS
        self.TARGET_VALIDATION_KEYWORDS = TARGET_VALIDATION_KEYWORDS
        self.choose_large = choose_large


    def speculative_generate(self, messages=None, max_tokens=100,
                             max_target_tokens=10, begin=False, begin_token_num=100, 
                             recap_after_negtive_num =15, original_recap_token_num=100, max_recap_token_num=200,
                             add_each_recap=25,add_each_neg=5, max_negtive_num=30,
                             temperature=1.0, top_k=50, top_p=0.95):
        
        stops = self.TRIGGER_TOKENS+[self.tokenizer.eos_token ] 
        sampling_params_one= SamplingParams(max_tokens=1024, temperature=temperature, top_k=top_k, top_p=top_p, 
                                            skip_special_tokens=False, stop=stops)
        tgt_sampling_params_cache= SamplingParams(max_tokens=max_target_tokens, temperature=temperature, top_k=top_k, top_p=top_p,
                                                  skip_special_tokens=False)
        token_num, change_tokens, change_flag = 0, 0, False
        negative_sent_num, recap_token_num = 0, original_recap_token_num
        generated_ids = self.tokenizer.apply_chat_template(messages, return_tensors="np").tolist()[0]
        prompt_len = len(generated_ids)
        correct_tokens, try_correct_num = [], 0
        while token_num <= max_tokens:
            if not begin:
                one_token_id = get_ray_reuslt(self.speculative_model, generated_ids, sampling_params_one)
                generated_ids.extend(one_token_id)
                if one_token_id[-1] == self.tokenizer.eos_token_id : break
                one_token = self.tokenizer.decode(one_token_id[-5:], skip_special_tokens=True)
            if begin or any(trigger in one_token for trigger in self.TRIGGER_TOKENS): 
                if begin:
                    change_tokens = begin_token_num
                    begin = False
                    change_flag = True
                elif negative_sent_num >= recap_after_negtive_num:
                    generated_ids.extend(self.help_recap_words_ids)
                    change_tokens = recap_token_num
                    change_flag = True
                    negative_sent_num = 0
                    recap_token_num, recap_after_negtive_num= min(recap_token_num + add_each_recap, max_recap_token_num), min(recap_after_negtive_num+add_each_neg, max_negtive_num)
                else:
                    if self.help_think_word_ids is not None:
                        generated_ids.extend(self.help_think_word_ids)
                    spe_ids = get_ray_reuslt(self.speculative_model, generated_ids, tgt_sampling_params_cache)
                    spe_token = self.tokenizer.decode(spe_ids, skip_special_tokens=True)
                    spe_sent = sentiment_analysis(spe_token, self.TARGET_VALIDATION_KEYWORDS['positive'], self.TARGET_VALIDATION_KEYWORDS['negative']+self.TARGET_VALIDATION_KEYWORDS['verify'])
                    if spe_sent != 0:
                        try_correct_num = try_correct_num+1
                        tgt_ids = get_ray_reuslt(self.target_model, generated_ids, tgt_sampling_params_cache)
                        tgt_token = self.tokenizer.decode(tgt_ids, skip_special_tokens=True)
                        tgt_sent = sentiment_analysis(tgt_token, self.TARGET_VALIDATION_KEYWORDS['positive'], self.TARGET_VALIDATION_KEYWORDS['negative']+self.TARGET_VALIDATION_KEYWORDS['verify'])
                        if self.choose_large or (spe_sent<0 and tgt_sent >=0) or (spe_sent>0 and tgt_sent<0):
                            decode_text = tgt_token
                            correct_tokens.append({
                                'pos': len(generated_ids)-prompt_len, 'token_num':max_target_tokens,
                                'traget':tgt_token, 'speculative':spe_token})
                            generated_ids.extend(tgt_ids) # torch.cat([cache_generated_ids, tgt_new_ids[:, :]], dim=-1)  # ✅ 接受 Target Model 结果
                            final_sent=tgt_sent
                        else:
                            generated_ids.extend(spe_ids)  # torch.cat([cache_generated_ids, tgt_new_ids[:, :]], dim=-1)  # ✅ 接受 Target Model 结果
                            decode_text = spe_token
                            final_sent=spe_sent
                        if final_sent < 0: negative_sent_num = negative_sent_num+1
                        if contains_keywords(decode_text, self.TARGET_VALIDATION_KEYWORDS['verify']):
                            change_tokens = original_recap_token_num
                            change_flag = True
                if change_flag:
                    try_correct_num = try_correct_num+1
                    tgt_sampling_params_= SamplingParams(max_tokens=change_tokens, temperature=temperature, 
                                                         top_k=top_k, top_p=top_p, skip_special_tokens=False) 
                    tgt_ids = get_ray_reuslt(self.target_model, generated_ids, tgt_sampling_params_)    
                    tgt_token = self.tokenizer.decode(tgt_ids, skip_special_tokens=True)
                    correct_tokens.append({
                        'pos': len(generated_ids)-prompt_len, 'token_num':change_tokens,
                        'traget':tgt_token})
                    generated_ids.extend(tgt_ids)
                    change_flag = False
            token_num = len(generated_ids)
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return generated_text, len(generated_ids)-prompt_len, correct_tokens, try_correct_num
        

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


if __name__ == "__main__":
    TRIGGER_TOKENS = ["\n\n"]  # 遇到这些 token 触发 Target Model
    TARGET_VALIDATION_KEYWORDS = {'verify':['verify', 'think again', 'recap', 'double-check'],
                                "negative":['but', 'wait', "alternatively", 'hold on','another'],
                                "positive":['yeah','yes','final answer','confident']}  # 目标关键字
    help_think_word = None # '\n\n'
    help_recap_words = "Let me check whether there are some wrong steps "
    target_model_name = models_names['deepseek-1.5b']  # 目标模型
    speculative_model_name = models_names['deepseek-1.5b']

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # **加载 Speculative Model 和 Target Model**
    
    model = spe_thinking_model(target_model_name, speculative_model_name, TRIGGER_TOKENS,
                               TARGET_VALIDATION_KEYWORDS, help_think_word, help_recap_words,
                               1,1)
    messages = []
    messages.append({
        "role": "user",
        "content": "Please reason step by step, and put your final answer within \\boxed{{}}. " + 'how to define the question?' + ' <think>\n'
    })

    model.speculative_generate(messages, 1024, 10)
