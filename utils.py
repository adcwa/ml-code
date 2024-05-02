from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import math


def get_tokenizer(model_name_or_path, model):
    if "llama" in model:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = 'right'
    elif "qwen2" in model:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        tokenizer.pad_token = '<|endoftext|>'
        tokenizer.padding_side = 'right'
    elif "qwen" in model:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        tokenizer.pad_token_id = tokenizer.eod_id
        tokenizer.pad_token = '<|endoftext|>'
        tokenizer.padding_side = 'right'
    elif 'chatglm' in model:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    elif 'mixtral' in model:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'right'
    elif "gemma" in model:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        tokenizer.padding_side = 'right'
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'right'
    return tokenizer


def get_hf_model(script_args, tokenizer, quantization_config=None, device_map="auto"):
    if 'chatglm' in script_args.model_type:
        model = AutoModel.from_pretrained(
            script_args.model_name,
            quantization_config=quantization_config,
            device_map=device_map,
            trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            script_args.model_name,
            quantization_config=quantization_config,
            device_map=device_map,
            trust_remote_code=True
        )
        model.resize_token_embeddings(int(8 * math.ceil(len(tokenizer)/ 8.0)))  # make the vocab size multiple of 8
    return model
