import __init__
from speculative.generate import *
import copy
import time

if __name__ == '__main__':
    # **加载 Hugging Face 模型**
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"  # 你的 LLaMA / GPT 模型
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id 
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")

    # **初始化输入**
    prompt = "#"*150
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

    # **设置生成参数*
    temperature = 0.6
    top_k = 50
    top_p = 0.95

    # **Step 1: `generate_with_partial_kv()` 生成 `split_token_count` 个 token**
    generated_ids_kv1, kv_cache = generate_with_partial_kv(
        model, tokenizer, input_ids, None, max_new_tokens=1,
        temperature=temperature, top_k=top_k, top_p=top_p
    )

    # **Step 3: 人为加入 `added_tokens_count` 个随机 token**
    added_tokens = tokenizer(['#'*10], return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
    # added_tokens = torch.randint(0, tokenizer.vocab_size, (1, added_tokens_count), dtype=torch.long, device="cuda")
    generated_ids_kv1 = torch.cat([generated_ids_kv1, added_tokens], dim=-1)

     # **Step 5: 直接用 `generate()` 生成 `final_token_count` 个 token**
    start_time = time.time()
    generated_ids_kv, _ = generate_with_partial_kv(
        model, tokenizer, input_ids, copy.deepcopy(kv_cache), max_new_tokens=1,
        temperature=temperature, top_k=top_k, top_p=top_p
    )
    end_time = time.time()
    print(end_time-start_time)
    start_time = time.time()
    generated_ids_kv2, _ = generate_with_partial_kv(
        model, tokenizer, generated_ids_kv1, copy.deepcopy(kv_cache), max_new_tokens=1,
        temperature=temperature, top_k=top_k, top_p=top_p
    )
    end_time = time.time()
    print(end_time-start_time)
