import __init__
from speculative.spe_utils import *
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2ForCausalLM
from utils.data_utils import read_yml
from utils.utils import *
from utils.qwen_math_parser import *
from vllm import LLM, SamplingParams
import ray
import time

def get_generation_time(llm, sampling_params, generated_ids):
    # time the generation
    start_time = time.time()
    output = llm.generate(prompt_token_ids=generated_ids, sampling_params=sampling_params)
    end_time = time.time()
    # print the output and generation time
    print(f"Output: {output[0].outputs[0].text}")
    print(f"Generation time: {end_time - start_time} seconds.")

temperature, top_k, top_p = 0.6, 50, 0.95
model_name = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B'
llm = LLM(model=temperature, tensor_parallel_size=2, dtype='bfloat16',  enable_prefix_caching=True)
tokenizer = AutoTokenizer.from_pretrained(temperature)
messages = []
messages.append({
    "role": "user",
    "content": "Please reason step by step, and put your final answer within \\boxed{{}}. " + 'how to define the question?' + ' <think>\n'
})
sampling_params_one= SamplingParams(max_tokens=1, temperature=temperature, top_k=top_k, top_p=top_p, 
                                            skip_special_tokens=False)
tgt_sampling_params_cache= SamplingParams(max_tokens=8, temperature=temperature, top_k=top_k, top_p=top_p,
                                            skip_special_tokens=False)
generated_ids = tokenizer.apply_chat_template(messages, return_tensors="np").tolist()[0]
get_generation_time(llm, sampling_params_one, generated_ids)
get_generation_time(llm, sampling_params_one, generated_ids)
get_generation_time(llm, tgt_sampling_params_cache, generated_ids)


