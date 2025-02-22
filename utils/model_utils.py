import transformers
import torch
import json
from .utils import set_global as set_global_path, cache_dir, api_keys, Logger as logger
import functools
import time
import torch
from transformers import PreTrainedTokenizer
import os

def format_chat(message, include_system=False, system_message="You are a helpful assistant."):
    if include_system:
        chat = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": message},
        ]
    else:
        chat = [{"role": "user", "content": message}]
    return chat

def call_api(func, limit=5, pause=10):
    count = 0
    while True:
        try:
            output = func()
            break
        except Exception as e:
            logger.info(f"Exception while using api: {e}")
            if "rate limit" in str(e).lower() or "rate_limit" in str(e).lower() or "quota" in str(e).lower() or "429" in str(e):
                logger.info(f"Rate limit exceeded, waiting {pause} secs and retrying...")
                time.sleep(pause)
            elif count < limit:
                logger.info(f"Encountered error {e}, retrying...")
                count += 1
            else:
                logger.info("Skipping generation due to unknown error")
                output = None
                break
    return output

class LLM:
    def __init__(
        self,
        model_name,
        temperature=0.9,
        top_p=0.9,
        max_length=32768,
        do_sample=True,
        stop_newline=False,
        use_chat_template=False,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_length = max_length
        self.do_sample = do_sample
        self.use_chat_template = use_chat_template
        self.stops = None
        if stop_newline:
            self.stops = ["\n", "\n\n"]
 
    def generate(self, prompt=None, max_gen=1024, **kwargs):
        raise NotImplementedError("generate not implemented for LLM")

class OpenAIModel(LLM):
    def __init__(
        self, 
        model_name, 
        temperature=0.9, 
        top_p=0.9, 
        max_length=32768,
        do_sample=True,
        stop_newline=False,
        use_chat_template=True,
        **kwargs,
    ): 
        super().__init__(
            model_name, 
            temperature=temperature, 
            top_p=top_p, 
            max_length=max_length,
            do_sample=do_sample,
            stop_newline=stop_newline,
            use_chat_template=use_chat_template,
        )
        import openai
        import tiktoken
        if "azure" in model_name:
            # env var: AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, and OPENAI_API_VERSION 
            self.model = openai.AzureOpenAI()
            model_name = model_name[model_name.index("/")+1:]
        else:
            # make sure to set the OPENAI_API_KEY environment variable
            self.model = openai.OpenAI(api_key=api_keys['openai'])
        self.model_name = model_name
        self.tokenizer = tiktoken.encoding_for_model(model_name)


    def prepare_inputs(self, test_item, data):
        buffer = 100
        prompt = format_chat(
            data["user_template"].format(**test_item), 
            system_message=data.get("system_message", "You are a helpful assistant.")
        )
        inputs = "\n".join([f"Role: {x['role']}\nContent: {x['content']}" for x in prompt])
        tokens = self.tokenizer.encode(inputs)
        input_len = len(tokens)

        max_length = self.max_length
        if max_length > 128000:
            logger.warning(f"max_length {max_length} is greater than 128000, setting to 128000")
            max_length = 128000

        if input_len > max_length - self.generation_max_length - buffer:
            truncate_length = input_len - (max_length - self.generation_max_length - buffer)
            new_context = self.tokenizer.decode(self.tokenizer.encode(test_item["context"])[:-truncate_length])
            test_item["context"] = new_context
            prompt = format_chat(
                data["user_template"].format(**test_item), 
                system_message=data.get("system_message", "You are a helpful assistant.")
            )
        return prompt 

    """
    inputs: list[str]
        the user message that has been prepared
    prompt: str
        the user message to be sent to the model
    """
    def generate(self, prompt=None, max_gen=1024, system_message="You are a helpful assistant", **kwargs):
        inputs = format_chat(prompt, include_system=True, system_message=system_message)
        
        # kwargs can be used to pass additional parameters to the model: max_tokens, stop, etc.
        func = functools.partial(
            self.model.chat.completions.create, 
            model=self.model_name, 
            messages=inputs, 
            max_tokens=max_gen,
            temperature=self.temperature if self.do_sample else 0.0,
            top_p=self.top_p,
            stop=self.stops,
            **kwargs,
        )
        output = call_api(func)
        if output is not None:
            if output.choices[0].message.content is None:
                # sometimes the model output can get filtered but sitll return a message
                return None
            return {
                "output": output.choices[0].message.content,
                "input_len": output.usage.prompt_tokens,
                "output_len": output.usage.completion_tokens,
                "input_text": inputs,
            }
        return None

class AnthropicModel(LLM):
    def __init__(
        self, 
        model_name, 
        temperature=0.9, 
        top_p=0.9, 
        max_length=32768,
        do_sample=True,
        stop_newline=False,
        use_chat_template=True,
        **kwargs,
    ): 
        super().__init__(
            model_name, 
            temperature=temperature, 
            top_p=top_p, 
            max_length=max_length,
            do_sample=do_sample,
            stop_newline=stop_newline,
            use_chat_template=use_chat_template,
        )
        from anthropic import Anthropic, AnthropicVertex
        if "vertex" in model_name:
            # region defaults to env var CLOUD_ML_REGION and project_id defaults to ANTHROPIC_VERTEX_PROJECT_ID
            self.model = AnthropicVertex()
            model_name = model_name[model_name.index("/")+1:]
        else:
            # remember to set ANTHROPIC_API_KEY environment variable (the default)
            self.model = Anthropic()

        self.tokenizer = self.model.get_tokenizer()
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_length = max_length
        self.do_sample = do_sample
        self.stops = None
        if stop_newline: # claude does not support newline
            pass


    def prepare_inputs(self, test_item, data):
        buffer = 100
        prompt = format_chat(
            data["user_template"].format(**test_item), 
            include_system=False,
        )
        inputs = "\n".join([f"Role: {x['role']}\nContent: {x['content']}" for x in prompt])
        tokens = self.tokenizer.encode(inputs)
        input_len = len(tokens)

        if input_len > self.max_length - self.generation_max_length - buffer:
            truncate_length = input_len - (self.max_length - self.generation_max_length - buffer)
            tokens = self.tokenizer.encode(test_item["context"])
            new_context = test_item["context"][:tokens.offsets[-truncate_length-1][1]]
            test_item["context"] = new_context
            prompt = format_chat(
                data["user_template"].format(**test_item), 
                include_system=False,
            )
        return prompt
       

    """
    inputs: list[str]
        the user message that has been prepared
    prompt: str
        the user message to be sent to the model
    """
    def generate(self, prompt=None, max_gen=2048, **kwargs):
        inputs = format_chat(prompt, include_system=False)
        
        # kwargs can be used to pass additional parameters to the model: max_tokens, stop, etc.
        func = functools.partial(
            self.model.messages.create,
            model=self.model_name, 
            messages=inputs, 
            max_tokens=max_gen,
            temperature=self.temperature if self.do_sample else 0.0,
            top_p=self.top_p,
            stop_sequences=self.stops,
            system="You are a helpful assistant. Make sure your output does not contain new lines.",
            **kwargs,
        )
        output = call_api(func, pause=20)

        if output is not None:
            return {
                "output": output.content[0].text,
                "input_len": output.usage.input_tokens,
                "output_len": output.usage.output_tokens,
                "input_text": inputs,
            }
        return None

class GeminiModel(LLM):
    def __init__(
        self, 
        model_name, 
        temperature=0.9, 
        top_p=0.9, 
        max_length=32768,
        do_sample=True,
        stop_newline=False,
        use_chat_template=True,
        **kwargs,
    ): 
        super().__init__(
            model_name, 
            temperature=temperature, 
            top_p=top_p, 
            max_length=max_length,
            do_sample=do_sample,
            stop_newline=stop_newline,
            use_chat_template=use_chat_template,
        )

        import google.generativeai as genai
        # default env var GOOGLE_API_KEY
        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

        import vertexai
        vertexai.init() # make sure to set the env var appropriately
        from vertexai.preview.tokenization import get_tokenizer_for_model
        self.model = genai.GenerativeModel(model_name)
        self.tokenizer = get_tokenizer_for_model(model_name)
        self.model_name = model_name

    def prepare_inputs(self, test_item, data):
        prompt = data["prompt_template"].format(**test_item)
        buffer = 100
        inputs = self.tokenizer.compute_tokens(prompt).token_info_list[0].tokens
        input_len = len(inputs)

        max_length = self.max_length
        if input_len > max_length - self.generation_max_length - buffer:
            truncate_length = input_len - (max_length - self.generation_max_length - buffer)
            # not the most pretty way of doing this but it works...
            # the documentation doesn't provide an official way to truncate
            new_context = self.tokenizer._sentencepiece_adapter._tokenizer.decode(self.tokenizer.compute_tokens(test_item["context"]).token_info_list[0].token_ids[:-truncate_length])
            test_item['context'] = new_context
            prompt = data["prompt_template"].format(**test_item)
        
        return prompt

    def generate(self, prompt=None, max_gen=2048 ,**kwargs):
        import google.generativeai as genai
        inputs = prompt
        
        generation_config = genai.GenerationConfig(temperature=self.temperature, top_p=self.top_p, max_output_tokens=max_gen)
        func = functools.partial(
            self.model.generate_content, 
            contents=inputs,
            generation_config=generation_config
        )
        output = call_api(func, pause=15)
        if output is not None:
            try:
                # can probably check the output for errors but it's not well documented
                output.text
            except Exception as e:
                logger.error(f"Error in output: {output}; {e}")  
                return None

            return {
                "output": output.text,
                "input_len": output.usage_metadata.prompt_token_count,
                "output_len": output.usage_metadata.candidates_token_count,
                "input_text": inputs,
            }
        return None

class TogetherModel(LLM):
    def __init__(
        self,
        model_name, 
        temperature=0.9, 
        top_p=0.9, 
        max_length=32768,
        do_sample=True,
        stop_newline=False,
        use_chat_template=True,
        **kwargs,
    ):
        super().__init__(
            model_name, 
            temperature=temperature, 
            top_p=top_p, 
            max_length=max_length,
            do_sample=do_sample,
            stop_newline=stop_newline,
            use_chat_template=use_chat_template,
        )

        from transformers import AutoTokenizer
        from together import Together
        # default env var TOGETHER_API_KEY
        self.model = Together()
        # should change this to be more flexible in the future lol
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-405B-Instruct")
        self.model_name = model_name.replace("togetherapi/", "")
 
    def prepare_inputs(self, test_item, data):
        buffer = 100
        prompt = format_chat(
            data["user_template"].format(**test_item), 
            system_message=data.get("system_message", "You are a helpful assistant.")
        )
        tokens = self.tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True)
        input_len = len(tokens)

        max_length = self.max_length
        if input_len > max_length - self.generation_max_length - buffer:
            truncate_length = input_len - (max_length - self.generation_max_length - buffer)
            context_tokens = self.tokenizer(test_item["context"], return_offsets_mapping=True)
            new_context = test_item["context"][:context_tokens["offset_mapping"][-truncate_length][0]]
            
            test_item["context"] = new_context
            prompt = format_chat(
                data["user_template"].format(**test_item), 
                system_message=data.get("system_message", "You are a helpful assistant.")
            )
        return prompt 

    """
    inputs: list[str]
        the user message that has been prepared
    prompt: str
        the user message to be sent to the model
    """
    def generate(self,prompt=None, max_gen=2048, system_message="You are a helpful assistant", **kwargs):
        inputs = format_chat(prompt, include_system=True, system_message=system_message)
        
        # kwargs can be used to pass additional parameters to the model: max_tokens, stop, etc.
        func = functools.partial(
            self.model.chat.completions.create, 
            model=self.model_name, 
            messages=inputs, 
            max_tokens=max_gen,
            temperature=self.temperature if self.do_sample else 0.0,
            top_p=self.top_p,
            stop=self.stops,
            **kwargs,
        )
        output = call_api(func)
        if output is not None:
            if output.choices[0].message.content is None:
                # sometimes the model output can get filtered but sitll return a message
                return None
            return {
                "output": output.choices[0].message.content,
                "input_len": output.usage.prompt_tokens,
                "output_len": output.usage.completion_tokens,
                "input_text": inputs,
            }
        return None

class HFModel(LLM):
    def __init__(
        self, 
        model_name, 
        temperature=1.0, 
        top_p=0.9, 
        do_sample=False,
        use_flash_attention_2=True,
        **kwargs,
    ):
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, AutoConfig
        model2path = json.load(open(set_global_path("./config/model2path.json"), "r"))
        model2maxlen = json.load(open(set_global_path("./config/model2maxlen.json"), "r"))
        path = model2path[model_name]
        model, tokenizer = load_model_and_tokenizer(path, model_name, use_flash_attention_2,  **kwargs)
        self.model = model
        self.tokenizer = tokenizer
        if model_name not in model2maxlen.keys(): max_length = model.config.max_position_embeddings
        else: max_length = model2maxlen[model_name]
        self.max_length = max_length

        super().__init__(
            model_name, 
            temperature=temperature, 
            top_p=top_p, 
            max_length=max_length,
            do_sample=do_sample,
            stop_newline=False,
            use_chat_template=False,
        )
  
    
    @torch.no_grad()
    def generate(self, prompt=None, max_gen=128, truncated=False, **kwargs):
        if truncated:
            tokenized_prompt = self.tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
            if "chatglm3" in self.model_name:
                tokenized_prompt = self.tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids[0]
            # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
            if len(tokenized_prompt) > self.max_length:
                half = int(self.max_length/2)
                prompt = self.tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+self.tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        input = self.tokenizer(prompt, truncation=False, return_tensors="pt").to(self.model.device)
        context_length = input.input_ids.shape[-1]
        kwargs = {'use_cache':True}
        if self.model_name == "llama2-7b-hf-slimpajama-landmark" or self.model_name == "llama2-7b-hf-slimpajama-landmark-test4k":  
            kwargs['offload_cache_to_cpu'] = False
            kwargs['use_flash'] = False
            kwargs['cache_top_k'] = 5
        output = self.model.generate(
            **input,
            max_new_tokens=max_gen,
            num_beams=1,
            do_sample=self.do_sample,
            temperature=self.temperature,
            **kwargs,
        )[0]
        pred = self.tokenizer.decode(output[context_length:], skip_special_tokens=True)
        return pred

def load_model(model_name, **kwargs):
    logger.info(f'testing: {model_name}')
    if "gpt" in model_name:
        model = OpenAIModel(model_name, **kwargs)
    elif "claude" in model_name:
        model = AnthropicModel
    elif "gemini" in model_name:
        model = GeminiModel
    elif "togetherapi" in model_name:
        model = TogetherModel
    else:
        model = HFModel(model_name, **kwargs)
    return model

def load_model_and_tokenizer(path, model_name, use_flash_attention_2=False,  **kwargs_load):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if model_name == "llama2-7b-hf" or model_name == "llama2-7b-hf-slimpajama-pi-32k" or model_name == "llama2-7b-hf-slimpajama-longlora-32k":
        config = transformers.AutoConfig.from_pretrained(path)
        logger.info(f'rope_scaling: {config.rope_scaling}')
        model = transformers.AutoModelForCausalLM.from_pretrained(
            path,
            config=config,
            torch_dtype=torch.float16,
            use_flash_attention_2=use_flash_attention_2,
            device_map="auto",
            cache_dir=cache_dir,
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            path,
            config=config,
            cache_dir=cache_dir,
        )
    elif model_name == "llama2-7b-hf-slimpajama-ntk-32k":
        config = transformers.AutoConfig.from_pretrained(path)
        logger.info(f'rope_scaling: {config.rope_scaling}')
        from models.llama_ntk_32k import LlamaForCausalLM
        model = LlamaForCausalLM.from_pretrained(
            path,
            config=config,
            torch_dtype=torch.float16,
            use_flash_attention_2=use_flash_attention_2,
            device_map="auto",
            cache_dir=cache_dir,
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            path,
            config=config,
            cache_dir=cache_dir,
        )
    elif model_name == "llama2-7b-hf-slimpajama-ntk-64k" or model_name == "llama-2-7b-hf-slimpajama-ntk-64k-2B":
        config = transformers.AutoConfig.from_pretrained(path)
        print('rope_scaling:', config.rope_scaling)
        from models.llama_ntk_64k import LlamaForCausalLM
        model = LlamaForCausalLM.from_pretrained(
            path,
            config=config,
            torch_dtype=torch.float16,
            use_flash_attention_2=use_flash_attention_2,
            device_map="auto",
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            path,
            config=config,
        )
    elif model_name == "llama2-7b-hf-lminfinite":
        from models.llama_infinite import LlamaForCausalLM
        from models.llama_infinite.llama import convert_llama_model
        model = LlamaForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.float16,
            device_map="auto",
            cache_dir=cache_dir,
        )
        model = convert_llama_model(model, 4096, 10)
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            path,
            cache_dir=cache_dir,
        )
    elif model_name == "llama2-7b-hf-ntk-frozen":
            # Set RoPE scaling factor
        config = transformers.AutoConfig.from_pretrained(
            path,
        )
        
        scaling_factor = 2.0
        print(config.rope_scaling)
        config.rope_scaling = {"type": "dynamic", "factor": scaling_factor}

        # Load model and tokenizer
        model = transformers.AutoModelForCausalLM.from_pretrained(
            path,
            config=config,
            torch_dtype=torch.float16,
            use_flash_attention_2=use_flash_attention_2,
            device_map="auto",
            cache_dir=cache_dir,
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            path,cache_dir=cache_dir,
        )     
    elif model_name == "llama2-7b-hf-slimpajama-yarn-32k":
        from models.llama_yarn.modeling_llama_yarn import LlamaForCausalLM
        from models.llama_yarn.configuration_llama import LlamaConfig
        config_cls = LlamaConfig
        model_cls = LlamaForCausalLM
        config = config_cls.from_pretrained(path)
        print(config.rope_scaling)
        model = model_cls.from_pretrained(
            path,
            config=config,
            torch_dtype=torch.float16,
            use_flash_attention_2=use_flash_attention_2,
            device_map="auto",
            cache_dir=cache_dir,
            low_cpu_mem_usage=True,
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            path,cache_dir=cache_dir,
        )
    elif model_name == "llama2-7b-hf-selfextend":
        from transformers import AutoModelForCausalLM
        from models.llama_selfextend import SelfExtend
        window_size = 1024
        group_size = 64
        use_flash = use_flash_attention_2
        model = AutoModelForCausalLM.from_pretrained(path, device_map="auto", torch_dtype=torch.bfloat16, use_flash_attention_2=use_flash,cache_dir=cache_dir,)
        print(f'using group size {group_size} using window size {window_size}')
        SelfExtend.apply(model, group_size, window_size, enable_flash_attention=use_flash, flash_attention_impl="flash_attn") ## flash_attention_impl="triton" or "flash_attn"
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            path,cache_dir=cache_dir,
        )
    elif model_name == "llama2-7b-hf-slimpajama-clex-32k":
        print('eval clex')
        from models.llama_clex import LlamaForCausalLM, CLEXLlamaConfig
        config = CLEXLlamaConfig.from_pretrained(path)
        config.log_scale = True
        config.use_flashattn = True
        print(config.rope_scaling, flush=True)
        model = LlamaForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            config=config,
            attn_implementation="flash_attention_2",
            device_map="auto",
            cache_dir=cache_dir
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            path,cache_dir=cache_dir,
        )
    elif model_name == "llama2-7b-hf-slimpajama-landmark":
        from models.llama_landmark.llama_mem import LlamaForCausalLM
        model = LlamaForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            cache_dir=cache_dir,
            )
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            path,
            padding_side="right",
            use_fast=False,
            cache_dir=cache_dir,
        )
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token 
        
        mem_id = tokenizer.convert_tokens_to_ids("<landmark>")
        model.set_mem_id(mem_id)
        model = model.to(device)
    else:
        kwargs=dict()
        if 'phi' in model_name: kwargs['trust_remote_code']=True
        config = transformers.AutoConfig.from_pretrained(path,token=token,**kwargs)
        if 'ratio' in kwargs_load and kwargs_load['ratio'] > 0:
            config.rope_theta = kwargs_load['ratio'] * config.rope_theta 
        print('rope_scaling:', config.rope_scaling)
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            path,
            config=config,
            cache_dir=cache_dir,
            token=token,
            **kwargs
        )
        if "llama-3" in model_name.lower():
            tokenizer.pad_token_id = tokenizer.eos_token_id
        model = transformers.AutoModelForCausalLM.from_pretrained(
            path,
            config=config,
            torch_dtype=torch.float16,
            use_flash_attention_2=use_flash_attention_2,
            device_map="auto",
            cache_dir=cache_dir,
            token=token,
            **kwargs
        )
    model = model.eval()
    return model, tokenizer
