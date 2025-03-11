import __init__
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer



def generate_hf(
    model, tokenizer, input_ids, max_new_tokens=10,
    temperature=1.0, top_k=50, top_p=0.95
):
    # **使用 Hugging Face `generate()` 一次性生成**
    generated_ids_hf = model.generate(
        input_ids=input_ids,
        attention_mask=(input_ids != tokenizer.pad_token_id).long(),  # Ensure mask aligns with pad tokens
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=(temperature > 0 and (top_k > 0 or top_p < 1.0)),  # 自动匹配 `do_sample`
        use_cache=True,
        return_dict_in_generate=False
    )
    return generated_ids_hf

def generate_with_partial_kv(
    model, tokenizer, input_ids, past_key_values=None, max_new_tokens=10,
    temperature=1.0, top_k=50, top_p=0.95
):
    device = input_ids.device
    
    # 检查 input_ids 是否有效
    if input_ids.numel() == 0 or input_ids.shape[1] == 0:
        raise ValueError("input_ids cannot be empty")
    
    seq_len = input_ids.shape[1]

    # Step 1: 计算 KV Cache
    if past_key_values is None:
        # 首次计算 KV Cache
        if seq_len > 1:
            with torch.no_grad():
                outputs = model(input_ids=input_ids[:, :-1], use_cache=True, return_dict=True)
                past_key_values = outputs.past_key_values
    else:
        # 增量计算：仅计算未缓存的 token
        cached_len = past_key_values[0][0].shape[2]  # KV Cache 已缓存 token 长度
        if cached_len < seq_len - 1:
            new_input_ids = input_ids[:, cached_len:-1]  # 仅计算未缓存的部分
            if new_input_ids.shape[1] > 0:  # 确保有新的 token 需要计算
                with torch.no_grad():
                    outputs = model(input_ids=new_input_ids, past_key_values=past_key_values, use_cache=True, return_dict=True)
                    past_key_values = outputs.past_key_values

    # Step 2: 选择 `do_sample`
    do_sample = temperature > 0 and (top_k > 0 or top_p < 1.0)
    
    # Step 3: 生成新 token
    try:
        output = model.generate(
            input_ids=input_ids,  # 仅输入最后一个 token
            attention_mask= (input_ids != tokenizer.pad_token_id).long(),  # Ensure mask aligns with pad tokens torch.ones_like(input_ids).long(), #
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            use_cache=True,
            return_dict_in_generate=True,
            past_key_values=past_key_values,
            pad_token_id=tokenizer.eos_token_id
        )
    except Exception as e:
        print(f"Error in model.generate: {e}")
        print(f"past_key_values type: {type(past_key_values)}")
        if past_key_values is not None:
            print(f"past_key_values length: {len(past_key_values)}")
            print(f"first layer shape: {past_key_values[0][0].shape if len(past_key_values) > 0 else 'N/A'}")
    # Step 4: 提取生成的 token ID
    generated_ids = output.sequences
    past_key_values = output.past_key_values

    return generated_ids, past_key_values

if __name__ == '__main__':
    # **加载 Hugging Face 模型**
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"  # 你的 LLaMA / GPT 模型
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id 
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

    # **初始化输入**
    prompt = "Once upon a time"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

    # **设置生成参数**
    split_token_count = 5  # 先生成 5 个 token
    added_tokens_count = 3  # 人为加入 3 个 token
    final_token_count = 10  # 最终总 token 数
    temperature = 0
    top_k = 50
    top_p = 0.95

    # **Step 2: 直接用 `generate()` 生成 `split_token_count` 个 token 进行对比**
    generated_ids_hf1 = model.generate(
        input_ids=input_ids,
        max_new_tokens=split_token_count,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=(temperature > 0 and (top_k > 0 or top_p < 1.0)),
        use_cache=True,
        return_dict_in_generate=False
    )

    # **Step 1: `generate_with_partial_kv()` 生成 `split_token_count` 个 token**
    generated_ids_kv1, kv_cache = generate_with_partial_kv(
        model, tokenizer, input_ids, None, max_new_tokens=split_token_count,
        temperature=temperature, top_k=top_k, top_p=top_p
    )

    # **对比第一部分生成的 token**
    first_match = torch.equal(generated_ids_kv1, generated_ids_hf1)

    # **Step 3: 人为加入 `added_tokens_count` 个随机 token**
    added_tokens = tokenizer(['... wait I need to think '], return_tensors="pt").input_ids.to("cuda")
    # added_tokens = torch.randint(0, tokenizer.vocab_size, (1, added_tokens_count), dtype=torch.long, device="cuda")
    generated_ids_kv1 = torch.cat([generated_ids_kv1, added_tokens], dim=-1)

     # **Step 5: 直接用 `generate()` 生成 `final_token_count` 个 token**
    generated_ids_hf2 = model.generate(
        input_ids=generated_ids_kv1,
        max_new_tokens=final_token_count,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=(temperature > 0 and (top_k > 0 or top_p < 1.0)),
        use_cache=True,
        return_dict_in_generate=False
    )

    # **Step 4: `generate_with_partial_kv()` 继续生成**
    generated_ids_kv2, kv_cache = generate_with_partial_kv(
        model, tokenizer, generated_ids_kv1, kv_cache, max_new_tokens=final_token_count,
        temperature=temperature, top_k=top_k, top_p=top_p
    )


    # **对比最终生成的 token**
    final_match = torch.equal(generated_ids_kv2, generated_ids_hf2)

    # **解码文本**
    generated_text_kv = tokenizer.decode(generated_ids_kv2[0], skip_special_tokens=True)
    generated_text_hf = tokenizer.decode(generated_ids_hf2[0], skip_special_tokens=True)

    # **打印结果**
    print("=== Hugging Face Generate (First Part) ===")
    print(tokenizer.decode(generated_ids_hf1[0], skip_special_tokens=True))
    print("\n=== KV Cache Partial Generate (First Part) ===")
    print(tokenizer.decode(generated_ids_kv1[0], skip_special_tokens=True))

    print("\nFirst Part Tokens Match:", first_match)

    print("\n=== Hugging Face Generate (Final) ===")
    print(generated_text_hf)
    print("\n=== KV Cache Partial Generate (Final) ===")
    print(generated_text_kv)

    print("\nFinal Tokens Match:", final_match)