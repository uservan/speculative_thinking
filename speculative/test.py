from __init__ import *
import time
import torch
import matplotlib.pyplot as plt
import copy

def benchmark_final_token_latency_by_prompt_len(
    model, tokenizer, base_prompt="hello",
    prompt_lens=[50, 500, 1*1024, 4*1024, 8*1024, 12*1024, 15*1024],
    n_generate=20,
    temperature=1.0, top_k=50, top_p=0.95
):
    device = model.device
    model.eval()
    results = []

    for L in prompt_lens:
        # print(f"\n--- Prompt length: {L} tokens ---")

        # 构造具有 L 个 token 的 prompt（重复 base_prompt）
        repeated_prompt = (base_prompt + " ") * L
        prompt_text = " ".join(repeated_prompt.split()[:L])
        input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)

        # Step 1: KV Cache（不含最后一个 token）
        with torch.no_grad():
            outputs = model(input_ids=input_ids[:, :-1], use_cache=True, return_dict=True)
            past_key_values = outputs.past_key_values

        # Step 2: generate n tokens
        torch.cuda.synchronize()
        t0 = time.time()
        output1 = model.generate(
            input_ids=input_ids,
            attention_mask=(input_ids != tokenizer.pad_token_id).long(),
            max_new_tokens=n_generate,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=(temperature > 0),
            use_cache=True,
            return_dict_in_generate=True,
            past_key_values=copy.deepcopy(past_key_values),
            pad_token_id=tokenizer.eos_token_id
        )
        torch.cuda.synchronize()
        t_generate_n = time.time() - t0
        generated_tokens = output1.sequences[0][input_ids.shape[1]:]

        # Step 3: 用 prompt 的 KV Cache + n-1 token => 生成最后 1 个 token
        input_ids_plus = torch.cat([input_ids, generated_tokens[:-1].unsqueeze(0)], dim=1)

        torch.cuda.synchronize()
        t1 = time.time()
        output2 = model.generate(
            input_ids=input_ids_plus,
            attention_mask=(input_ids_plus != tokenizer.pad_token_id).long(),
            max_new_tokens=1,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=(temperature > 0),
            use_cache=True,
            return_dict_in_generate=True,
            past_key_values=copy.deepcopy(past_key_values),
            pad_token_id=tokenizer.eos_token_id
        )
        torch.cuda.synchronize()
        t_generate_last = time.time() - t1

        # print(f"Generate {n_generate} tokens: {t_generate_n:.4f}s")
        # print(f"Generate 1 token after n-1: {t_generate_last:.4f}s")

        results.append({
            "prompt_len": L,
            "generate_n": t_generate_n,
            "generate_last": t_generate_last,
            'ratio': (L/t_generate_last)/(L/t_generate_n)
        })

    return results

def plot_latency_comparison(results):
    lens = [r["prompt_len"] for r in results]
    t_n = [r["generate_n"] for r in results]
    t_last = [r["generate_last"] for r in results]

    plt.plot(lens, t_n, marker='o', label="Generate n tokens")
    plt.plot(lens, t_last, marker='s', label="Generate 1 token after n-1")
    plt.xlabel("Prompt Length (tokens)")
    plt.ylabel("Time (s)")
    plt.title("Final Token Latency vs Prompt Length")
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    from transformers import AutoTokenizer, AutoModelForCausalLM

    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"  # or your model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    for n in [10, 20, 125, 250]:
        results = benchmark_final_token_latency_by_prompt_len(model, tokenizer, n_generate=n)
        print(results)