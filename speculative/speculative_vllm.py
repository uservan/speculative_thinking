from __init__ import *
from speculative.utils import *
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2ForCausalLM
from utils.data_utils import read_yml
from utils.utils import *
from utils.qwen_math_parser import *
from vllm import LLM, SamplingParams
import ray
import time

def create_ray_model(model_name, target_model_gpu, dtype='float16'):
    @ray.remote(num_gpus=target_model_gpu)
    class ModelWorkerSingleGPU:
        def __init__(self, model_name: str):
            self.model = LLM(model=model_name, tensor_parallel_size=target_model_gpu, dtype=dtype)
        def generate(self, generated_ids, sampling_params):
            outputs = self.model.generate(
                prompt_token_ids=generated_ids, 
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

class spe_thinking_vllm:
    def __init__(self, **config):
        self.tokenizer = AutoTokenizer.from_pretrained(config['target_model_name'])
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id 
        self.target_model = create_ray_model(config['target_model_name'], config['target_model_gpu'])
        self.speculative_model = None
        if config['speculative_model_name'] is not None: 
            self.speculative_model = create_ray_model(config['speculative_model_name'], config['speculative_model_gpu'])
        self.help_think_word_ids = None if config['help_think_word'] is None else self.tokenizer([config['help_think_word']], return_tensors="np",add_special_tokens=False)["input_ids"][0].tolist()
        self.help_recap_words_ids = self.tokenizer([config['help_recap_words']], return_tensors="np",add_special_tokens=False)["input_ids"][0].tolist()
        self.TRIGGER_TOKENS = config['TRIGGER_TOKENS']
        self.TARGET_VALIDATION_KEYWORDS = config['TARGET_VALIDATION_KEYWORDS']
        self.choose_large = config['choose_large']
        self.config = config

    def generate(self, messages=None, max_tokens=1024, temperature=0.6, top_k=50, top_p=0.95):
        if self.speculative_model is None:
            return self.normal_generate( messages, max_tokens, temperature, top_k, top_p)
        else:
            return self.speculative_generate( messages, max_tokens, temperature, top_k, top_p)

    def get_prompt_len(self,messages ):
        generated_ids = self.tokenizer.apply_chat_template(messages, return_tensors="np").tolist()[0]
        return  len(generated_ids)

    def normal_generate(self, messages=None, max_tokens=1024, temperature=0.6, top_k=50, top_p=0.95):
        generated_ids = self.tokenizer.apply_chat_template(messages, return_tensors="np").tolist()[0]
        prompt_len = len(generated_ids)
        sampling_params = SamplingParams(max_tokens=max_tokens, temperature=temperature, top_k=top_k, top_p=top_p, 
                                            skip_special_tokens=False)
        spe_ids = get_ray_reuslt(self.target_model, generated_ids, sampling_params)
        generated_text = self.tokenizer.decode(spe_ids, skip_special_tokens=True)
        num_tokens, correct_tokens, try_correct_num = len(spe_ids)-prompt_len, [], 0
        return generated_text, num_tokens, correct_tokens, try_correct_num

    def speculative_generate(self, messages=None, max_tokens=100, temperature=0.6, top_k=50, top_p=0.95):
        start_time = time.time()  
        stops = self.TRIGGER_TOKENS+[self.tokenizer.eos_token ] 
        sampling_params_one= SamplingParams(max_tokens=1024, temperature=temperature, top_k=top_k, top_p=top_p, 
                                            skip_special_tokens=False, stop=stops)
        tgt_sampling_params_cache= SamplingParams(max_tokens=self.config['max_target_tokens'], temperature=temperature, top_k=top_k, top_p=top_p,
                                                  skip_special_tokens=False)
        token_num, change_tokens, change_flag, begin = 0, 0, False, self.config['begin']
        negative_sent_num, recap_token_num = 0, self.config['original_recap_token_num']
        generated_ids = self.tokenizer.apply_chat_template(messages, return_tensors="np").tolist()[0]
        prompt_len = len(generated_ids)
        correct_tokens, try_correct_num = [], 0
        recap_after_negtive_num = self.config['recap_after_negative_num']
        while token_num <= max_tokens:
            if self.config['time_out'] is not None and self.config['time_out']>0:
                use_time = time.time() - start_time
                if use_time > self.config['time_out']: return None
            if not begin:
                one_token_id = get_ray_reuslt(self.speculative_model, generated_ids, sampling_params_one)
                generated_ids.extend(one_token_id)
                if one_token_id[-1] == self.tokenizer.eos_token_id : break
                one_token = self.tokenizer.decode(one_token_id[-5:], skip_special_tokens=True)
            if begin or any(trigger in one_token for trigger in self.TRIGGER_TOKENS): 
                if begin:
                    change_tokens = self.config['begin_token_num']
                    begin = False
                    change_flag = True
                elif negative_sent_num >= recap_after_negtive_num:
                    generated_ids.extend(self.help_recap_words_ids)
                    change_tokens = recap_token_num
                    change_flag = True
                    negative_sent_num = 0
                    recap_token_num, recap_after_negtive_num= min(recap_token_num + self.config['add_each_recap'],self.config['max_recap_token_num']), min(recap_after_negtive_num+self.config['add_each_neg'], self.config['max_negative_num'])
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
                                'pos': len(generated_ids)-prompt_len, 'token_num':self.config['max_target_tokens'],
                                'traget':tgt_token, 'speculative':spe_token})
                            generated_ids.extend(tgt_ids)
                            final_sent=tgt_sent
                        else:
                            generated_ids.extend(spe_ids)
                            decode_text = spe_token
                            final_sent=spe_sent
                        if final_sent < 0: negative_sent_num = negative_sent_num+1
                        if contains_keywords(decode_text, self.TARGET_VALIDATION_KEYWORDS['verify']):
                            change_tokens = self.config['original_recap_token_num']
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
        


if __name__ == "__main__":
    yml_path = '/home/wxy320/ondemand/program/speculative_thinking/speculative/spe_setting.yml'

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = spe_thinking_vllm(yml_path)
    messages = []
    messages.append({
        "role": "user",
        "content": "Please reason step by step, and put your final answer within \\boxed{{}}. " + 'how to define the question?' + ' <think>\n'
    })

    model.speculative_generate(messages, 1024, 10)
