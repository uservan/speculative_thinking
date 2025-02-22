from datasets import load_dataset
from .utils import cache_dir
import os
import json

def save_results(out_path, data):
    if os.path.exists(out_path):
        with open(out_path, "a", encoding="utf-8") as f:
            json.dump(data, f)
            f.write('\n')  # 添加换行符
    else:
        save_all_results(out_path, [data])


def save_all_results(out_path, data):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for pred in data:
            json.dump(pred, f, ensure_ascii=False)
            f.write('\n')

def read_saved_results(out_path):
    preds=[]
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            i=0
            for line in f:
                l = json.loads(line)
                preds.append(l)
                i+=1
                print(i)
    return preds


def load_data(dataset_name, mode='json'):
    if mode == 'huggingface':
        dataset = load_dataset(dataset_name)
    if mode == 'json':
        dataset = load_dataset('json', data_files=dataset_name)
    return dataset
