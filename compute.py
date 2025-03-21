import os
import json
from transformers import AutoConfig


configs = {
    '1b': AutoConfig.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct'),
    '1b-instruct': AutoConfig.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct'),
    '7b': AutoConfig.from_pretrained('Qwen/Qwen2.5-7B-Instruct'),
    '7b-instruct': AutoConfig.from_pretrained('Qwen/Qwen2.5-7B-Instruct'),
    '14b': AutoConfig.from_pretrained('Qwen/Qwen2.5-14B-Instruct'),
    '14b-instruct': AutoConfig.from_pretrained('Qwen/Qwen2.5-14B-Instruct'),
    '32b': AutoConfig.from_pretrained('Qwen/Qwen2.5-32B-Instruct'),
    '32b-instruct': AutoConfig.from_pretrained('Qwen/Qwen2.5-32B-Instruct')
}

folder_path = "/home/wxy320/ondemand/program/speculative_thinking/analysis/1/speculative"  # 替换为你的文件夹路径
json_files = [f for f in os.listdir(folder_path) if f.endswith(".json")]
t = 0.6
for file in json_files:
    name, model, dataset = file.split('_')[0], file.split('_')[1].split('.')[0], file.split('_')[2]
    file_path = os.path.join(folder_path, file)
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    correct_num, correct_tokens, worng_tokens, all_time_spend = 0, 0, 0, 0
    fix_tokens, right_flops, wrong_flops, all_flops = 0, 0, 0, 0
    for k in data.keys():
        # print(data[k].keys())
        token_usages = data[k]['token_usages'][str(t)][0]
        # time_spend = token_usages['time_spend']
        tokens_num = token_usages['completion_tokens']
        print(token_usages)
        