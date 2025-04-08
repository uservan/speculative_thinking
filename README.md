# speculative_thinking

## eval normal model
- if you eval the deepseek-7b, you could run the command
```shell
python ./skythought_evals/eval.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
    --evals amc23 --n 1 --result-dir ./eval1/amc2323 \
    --tp 2 --output-file ./eval1/amc2323/32B.txt
```
## eval speculative thinking
- first prepare the config, you could find the some configs in speculative/config
```yml
mode: "vllm" # 'hf' or "vllm"
target_model_name: "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
speculative_model_name: "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
target_model_gpu: 2
speculative_model_gpu: 1
...
```
- then run the command
```shell
python /home/wxy320/ondemand/program/speculative_thinking/skythought_evals/eval.py \
    --evals amc23 --n 1 --result-dir ./eval1/amc2323 \
    --tp 3 --output-file ./eval1/amc2323/1b_14b.txt --spe_config ./speculative/config/1b_14b.yml
```