from __init__ import *
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2ForCausalLM
from utils.data_utils import read_yml
from utils.utils import *
from utils.qwen_math_parser import *
from vllm import LLM, SamplingParams
import copy
from speculative.generate import generate_with_partial_kv, generate_hf
from speculative.spe_utils import *
import time

class spe_thinking_hf:
    def __init__(self, **config):
        self.tokenizer = AutoTokenizer.from_pretrained(config['target_model_name'])
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id 
        self.target_model = AutoModelForCausalLM.from_pretrained(config['target_model_name'], torch_dtype=torch.float16, device_map="auto") 
        self.speculative_model = None
        if config['speculative_model_name'] is not None: 
            speculative_model_name = config['speculative_model_name']# Speculative Model (更小的模型)
            self.speculative_model = AutoModelForCausalLM.from_pretrained(speculative_model_name, torch_dtype=torch.float16, device_map="auto")
        self.help_think_word_ids = None if config['help_think_word'] is None else self.tokenizer([config['help_think_word']], return_tensors="pt",add_special_tokens=False).input_ids.to("cuda")
        # Let me summarize and recap to make sure I didn't make any mistakes
        # Let me shortly summarize and check previous thoughts to make sure I didn't make any mistakes
        self.help_recap_words_ids = self.tokenizer([config['help_recap_words']], return_tensors="pt",add_special_tokens=False).input_ids.to("cuda")
        self.TRIGGER_TOKENS = config['TRIGGER_TOKENS']
        self.TARGET_VALIDATION_KEYWORDS = config['TARGET_VALIDATION_KEYWORDS']
        self.choose_large = config['choose_large']
        self.not_reasoning = config.get('not_reasoning', False)
        self.config = config

    def get_prompt_len(self,messages ):
        generated_ids = self.tokenizer.apply_chat_template(messages, return_tensors="np").tolist()[0]
        return len(generated_ids)

    def generate(self, messages=None, max_tokens=1024, temperature=0.6, top_k=50, top_p=0.95):
        if self.speculative_model is None:
            return self.normal_generate( messages, max_tokens, temperature, top_k, top_p)
        else:
            return self.speculative_generate( messages, max_tokens, temperature, top_k, top_p)

    def normal_generate(self, messages=None, max_tokens=1024, temperature=0.6, top_k=50, top_p=0.95):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        input_ids = self.tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)
        prompt_len = input_ids.shape[1]
        generated_ids_hf = generate_hf(self.target_model,  self.tokenizer, input_ids, max_tokens,
                    temperature=temperature, top_p=top_p, top_k=top_k)
        generated_text = self.tokenizer.decode(generated_ids_hf[0,:], skip_special_tokens=True)
        num_tokens, correct_tokens,try_correct_num = generated_ids_hf.shape[1]-prompt_len, [], 0
        return generated_text, num_tokens, correct_tokens, try_correct_num

    def speculative_generate(self, messages=None, max_tokens=100, temperature=0.6, top_k=50, top_p=0.95):
        start_time = time.time() 
        device = "cuda" if torch.cuda.is_available() else "cpu"
        generated_ids = self.tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)
        prompt_len = generated_ids.shape[1]
        spec_kv, tgt_kv=None,None
        correct_tokens, try_correct_num = [], 0
        token_num, change_tokens, change_flag = 0, 0, False
        negative_sent_num, recap_token_num = 0, self.config['original_recap_token_num']
        max_target_tokens = self.config['max_target_tokens']
        begin = self.config['begin']
        recap_after_negtive_num = self.config['recap_after_negative_num']
        while generated_ids.shape[1] < max_tokens:
            if self.config['time_out'] is not None and self.config['time_out']>0:
                use_time = time.time() - start_time
                if use_time > self.config['time_out']: return None
            if not begin:
                generated_ids, spec_kv = generate_with_partial_kv(
                    self.speculative_model, self.tokenizer, generated_ids, spec_kv,
                    max_new_tokens=1, temperature=temperature, top_k=top_k, top_p=top_p
                )
                decoded_text = self.tokenizer.decode(generated_ids[0, -1:], skip_special_tokens=True)
            if begin or any(trigger in decoded_text for trigger in self.TRIGGER_TOKENS): 
                if begin:
                    change_tokens = self.config['begin_token_num']
                    begin = False
                    change_flag = True
                    tgt_kv_candidate=None
                    spe_decoded_text = ''
                elif negative_sent_num >= recap_after_negtive_num:
                    generated_ids = torch.cat([generated_ids, self.help_recap_words_ids], dim=-1)
                    change_tokens = recap_token_num
                    change_flag = True
                    negative_sent_num = 0
                    recap_token_num, recap_after_negtive_num= min(recap_token_num + self.config['add_each_recap'],self.config['max_recap_token_num']), min(recap_after_negtive_num+self.config['add_each_neg'], self.config['max_negative_num'])
                else:
                    if self.help_think_word_ids is not None:
                        cache_generated_ids = torch.cat([generated_ids, self.help_think_word_ids], dim=-1)
                    else:
                        cache_generated_ids = generated_ids
                    spe_new_ids, spec_kv_candidate = generate_with_partial_kv(
                        self.speculative_model, self.tokenizer, cache_generated_ids, copy.deepcopy(spec_kv),
                        max_new_tokens=max_target_tokens, temperature=temperature, top_k=top_k, top_p=top_p
                    )
                    spe_decoded_text =self.tokenizer.decode(spe_new_ids[0,-max_target_tokens:], skip_special_tokens=True)
                    spe_sent = sentiment_analysis(spe_decoded_text, self.TARGET_VALIDATION_KEYWORDS['positive'], self.TARGET_VALIDATION_KEYWORDS['negative']+self.TARGET_VALIDATION_KEYWORDS['verify'])
                    if self.not_reasoning or spe_sent != 0:
                        try_correct_num = try_correct_num+1
                        # **Step 3: 目标模型生成 max_target_tokens 个 token**
                        tgt_new_ids, tgt_kv_candidate = generate_with_partial_kv(
                            self.target_model, self.tokenizer, cache_generated_ids, copy.deepcopy(tgt_kv),
                            max_new_tokens=max_target_tokens, temperature=temperature, top_k=top_k, top_p=top_p
                        )
                        # **解码 Target Model 生成的文本**
                        tgt_decoded_text = self.tokenizer.decode(tgt_new_ids[0,-max_target_tokens:], skip_special_tokens=True)
                        tgt_sent = sentiment_analysis(tgt_decoded_text, self.TARGET_VALIDATION_KEYWORDS['positive'], self.TARGET_VALIDATION_KEYWORDS['negative']+self.TARGET_VALIDATION_KEYWORDS['verify'])
                        if self.choose_large or (spe_sent<0 and tgt_sent >=0) or (spe_sent>0 and tgt_sent<0):
                            generated_ids = tgt_new_ids 
                            tgt_kv = tgt_kv_candidate  
                            decode_text = tgt_decoded_text
                            correct_tokens.append({
                                'pos': cache_generated_ids.shape[1]-prompt_len, 'token_num':max_target_tokens,
                                'traget':tgt_decoded_text, 'speculative':spe_decoded_text})
                            final_sent=tgt_sent
                        else:
                            generated_ids = spe_new_ids
                            spec_kv = spec_kv_candidate
                            decode_text = spe_decoded_text
                            final_sent=spe_sent
                        if final_sent < 0: negative_sent_num = negative_sent_num+1
                        if contains_keywords(decode_text, self.TARGET_VALIDATION_KEYWORDS['verify']):
                            change_tokens = self.config['original_recap_token_num']
                            change_flag = True
                if change_flag:
                    try_correct_num = try_correct_num+1
                    generated_ids, tgt_kv = generate_with_partial_kv(
                       self.target_model, self.tokenizer, generated_ids, tgt_kv_candidate,
                        max_new_tokens=change_tokens, temperature=temperature, top_k=top_k, top_p=top_p
                    )
                    tgt_decoded_text = self.tokenizer.decode(generated_ids[0,-change_tokens:], skip_special_tokens=True)
                    correct_tokens.append({
                        'pos': generated_ids.shape[1]-prompt_len, 'token_num':change_tokens,
                        'traget':tgt_decoded_text, 'speculative':spe_decoded_text})
                    change_flag = False
            if self.tokenizer.eos_token_id in generated_ids[0, -max_target_tokens:].tolist(): 
                break
        generated_text = self.tokenizer.decode(generated_ids[0,:], skip_special_tokens=True)
        return generated_text, len(generated_ids[0])-prompt_len, correct_tokens, try_correct_num
        


if __name__ == "__main__":
    yml_path = '/home/wxy320/ondemand/program/speculative_thinking/speculative/config/nromal/32B.yml'

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = read_yml(yml_path)
    model = spe_thinking_hf(**config)
    messages = []
    messages.append({
        "role": "user",
        "content": "Please reason step by step, and put your final answer within \\boxed{{}}. " + 'how to define the question?' + ' <think>\n'
    })

    model.generate(messages, 1024)
