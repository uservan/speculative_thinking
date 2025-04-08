import os
cache_dir = '/scratch/pbsjobs/wxy320/huggingface'
project_dir = '/home/wxy320/ondemand/program/speculative_thinking'
hug_token = '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'
if cache_dir is not None:
    os.environ["HF_HOME"] = cache_dir

if hug_token is not None:
    from huggingface_hub import login
    login(token=hug_token)

api_keys = {
    'openai': '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'
}