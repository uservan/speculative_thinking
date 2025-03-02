import torch
from transformers import AutoModelForCausalLM, AutoTokenizer



def generate_hf(
    model, tokenizer, input_ids, max_new_tokens=10,
    temperature=1.0, top_k=50, top_p=0.95
):
    # **使用 Hugging Face `generate()` 一次性生成**
    generated_ids_hf = model.generate(
        input_ids=input_ids,
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
    seq_len = input_ids.shape[1]

    # **Step 1: 计算 KV Cache**
    if past_key_values is None:
        # **首次计算 KV Cache**
        with torch.no_grad():
            outputs = model(input_ids=input_ids[:, :-1], use_cache=True, return_dict=True)
            past_key_values = outputs.past_key_values
    else:
        # **增量计算：仅计算未缓存的 token**
        cached_len = past_key_values[0][0].shape[2]  # KV Cache 已缓存 token 长度
        if cached_len < seq_len - 1:
            new_input_ids = input_ids[:, cached_len:-1]  # ✅ 仅计算未缓存的部分
            with torch.no_grad():
                outputs = model(input_ids=new_input_ids, past_key_values=past_key_values, use_cache=True, return_dict=True)
                past_key_values = outputs.past_key_values

    # **Step 2: 选择 `do_sample`**
    do_sample = temperature > 0 and (top_k > 0 or top_p < 1.0)

    # **Step 3: 生成新 token**
    output = model.generate(
        input_ids=input_ids[:, -1:],  # ✅ 仅输入最后一个 token
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=do_sample,  # ✅ 采样模式
        use_cache=True,
        return_dict_in_generate=True,
        past_key_values=past_key_values  # ✅ 复用 KV Cache
    )

    # **Step 4: 提取生成的 token ID**
    generated_ids = output.sequences
    past_key_values = output.past_key_values  # ✅ 仅在 `generate()` 生成后更新 KV Cache

    return generated_ids, past_key_values



if __name__ == '__main__':
    # **加载 Hugging Face 模型**
    model_name = "meta-llama/Llama-3-8b"  # 你的 LLaMA / GPT 模型
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

    # **初始化输入**
    prompt = "Once upon a time"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

    # **设置生成参数**
    split_token_count = 5  # 先生成 5 个 token
    added_tokens_count = 3  # 人为加入 3 个 token
    final_token_count = 10  # 最终总 token 数
    temperature = 1.0
    top_k = 50
    top_p = 0.95

    # **Step 1: `generate_with_partial_kv()` 生成 `split_token_count` 个 token**
    generated_ids_kv1, kv_cache = generate_with_partial_kv(
        model, tokenizer, input_ids, None, max_new_tokens=split_token_count,
        temperature=temperature, top_k=top_k, top_p=top_p
    )

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

    # **对比第一部分生成的 token**
    first_match = torch.equal(generated_ids_kv1, generated_ids_hf1)

    # **Step 3: 人为加入 `added_tokens_count` 个随机 token**
    added_tokens = tokenizer(['... wait I need to think '], return_tensors="pt").input_ids.to("cuda")
    # added_tokens = torch.randint(0, tokenizer.vocab_size, (1, added_tokens_count), dtype=torch.long, device="cuda")
    generated_ids_kv1 = torch.cat([generated_ids_kv1, added_tokens], dim=-1)

    # **Step 4: `generate_with_partial_kv()` 继续生成**
    generated_ids_kv2, kv_cache = generate_with_partial_kv(
        model, tokenizer, generated_ids_kv1, kv_cache, max_new_tokens=final_token_count - split_token_count - added_tokens_count,
        temperature=temperature, top_k=top_k, top_p=top_p
    )

    # **拼接最终 KV Cache 生成的 token**
    generated_ids_kv = torch.cat([generated_ids_kv1, generated_ids_kv2[:, added_tokens_count:]], dim=-1)

    # **Step 5: 直接用 `generate()` 生成 `final_token_count` 个 token**
    generated_ids_hf = model.generate(
        input_ids=generated_ids_kv1,
        max_new_tokens=final_token_count,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=(temperature > 0 and (top_k > 0 or top_p < 1.0)),
        use_cache=True,
        return_dict_in_generate=False
    )

    # **对比最终生成的 token**
    final_match = torch.equal(generated_ids_kv, generated_ids_hf)

    # **解码文本**
    generated_text_kv = tokenizer.decode(generated_ids_kv[0], skip_special_tokens=True)
    generated_text_hf = tokenizer.decode(generated_ids_hf[0], skip_special_tokens=True)

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