from .utils import *
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams

def count_parameters(model):
        return sum(p.numel() for name, p in model.named_parameters() if p.requires_grad)
        # return sum(p.numel() for p in model.parameters() if p.requires_grad)
def check_grad_update_status(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            Logger(f"参数 '{name}' 可以更新梯度。")
        else:
            Logger(f"参数 '{name}' 不可更新梯度。")
    for name, module in model.named_modules():
        if 'lora_A.default' in name:
            Logger(f"Module: {name}, LoRA Rank: {module.out_features}")


def init_model(model_name='', check=True, eval=True, low_cpu_mem_usage=True,):
    if eval:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto",
                                                  low_cpu_mem_usage=low_cpu_mem_usage, torch_dtype=torch.bfloat16,
                                                   use_flash_attention_2=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, use_cache=False, low_cpu_mem_usage=low_cpu_mem_usage,
                                                     torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_name,)  
    if check:
        check_grad_update_status(model)
        Logger(f'LLM总参数量：{count_parameters(model) / 1e6:.3f} 百万')
    return model, tokenizer

def load_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name,) 
    return tokenizer

def init_vllm_model(model_name, gpus=1):
    llm = LLM(model=model_name, tensor_parallel_size=gpus)
    return llm

def llm_generate(model, tokenizer, prompts:list, max_length=256, temperature=0.7, top_p=0.9):
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            # num_beams=4,      # 使用 Beam Search
            # do_sample=False,  # 启用采样模式
            repetition_penalty=1.1,  # 惩罚重复生成
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id  # 设置结束标记
        )
    decoded = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    for i, prompt in enumerate(prompts):
        decoded[i] = decoded[i].replace(prompt, '').strip()
    return decoded

def vllm_generate(llm, prompts:list, max_length=256, temperature=0.7, top_p=0.9):
    sampling_params = SamplingParams(temperature=temperature,top_p=top_p, max_tokens=max_length)    
    outputs = llm.generate(prompts, sampling_params)
    return [output.outputs[0].text for output in outputs]

def add_template(model_name, prompt, tokenizer=None):
    if tokenizer is None: tokenizer = load_tokenizer(model_name)
    if 'QwQ' in model_name:
        messages = [
            {"role": "system", "content": "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step. Return your final response within \\boxed{{}}"},
            {"role": "user", "content": prompt}
            ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    elif 'Sky-T1-32B-Preview' in model_name:
      d = {'prompt': "Your role as an assistant involves thoroughly exploring questions through a systematic long \
        thinking process before providing the final precise and accurate solutions. This requires \
        engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, \
        backtracing, and iteration to develop well-considered thinking process. \
        Please structure your response into two main sections: Thought and Solution. \
        In the Thought section, detail your reasoning process using the specified format: \
        <|begin_of_thought|> {thought with steps separated with '\n\n'} \
        <|end_of_thought|> \
        Each step should include detailed considerations such as analisying questions, summarizing \
        relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining \
        any errors, and revisiting previous steps. \
        In the Solution section, based on various attempts, explorations, and reflections from the Thought \
        section, systematically present the final solution that you deem correct. The solution should \
        remain a logical, accurate, concise expression style and detail necessary step needed to reach the \
        conclusion, formatted as follows: \
        <|begin_of_solution|> \
        {final formatted, precise, and clear solution} \
        <|end_of_solution|> \
        Now, try to solve the following question through the above guidelines: "}
      prompt = d["prompt"] + prompt
    elif 'sky' in model_name:
        prompt = 'You are a helpful and harmless assistant. You should solve this math problem using step-by-step reasoning. Require that the output of each step ends with the "\n\n" token. Return your final response with \\[ \\boxed{} \\], if answer is "1/2", the output is \\[ \\boxed{1/2} \\]. '+prompt
    else: # elif 'math' in model_name:
        # 'follow this structure to step-by-step solve math problem, and final answer must formated with \\[ \\boxed{} \\], if answer is "1/2", the output is \\[ \\boxed{1/2} \\]'
        # 'solve this math problem using step-by-step reasoning. Require that the output of each step ends with the "\n\n" token.'
        prompt = "You are a helpful and harmless assistant. You should think step-by-step. Return your final response within \\boxed{{}}. "+prompt
    return prompt