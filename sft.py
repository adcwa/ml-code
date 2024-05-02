# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import List, Optional
import os
import math
import torch
import pandas as pd
from accelerate import Accelerator
from datasets import load_dataset, Dataset
from peft import LoraConfig, PeftConfig, PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoModel, BitsAndBytesConfig, HfArgumentParser, TrainingArguments, AutoTokenizer
from trl import SFTTrainer, is_xpu_available
from utils import get_tokenizer, get_hf_model
try:
    import odps_utils
except:
    pass

# Define global vars
model_type = None
apply_chat_template = False
system_prompt = None

# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with SFTTrainer
    """

    model_type: Optional[str] = field(default="bloom", metadata={"help": "the model type"})
    model_name: Optional[str] = field(default="facebook/opt-350m", metadata={"help": "the model name"})
    train_dataset_name: Optional[str] = field(
        default="", metadata={"help": "train dataset name"}
    )
    eval_dataset_name: Optional[str] = field(
        default="", metadata={"help": "eval dataset name"}
    )
    #dataset_text_field: Optional[str] = field(default="merged_text", metadata={"help": "the text field of the dataset"})
    report_to: Optional[str] = field(default="none", metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=1.41e-5, metadata={"help": "the learning rate"})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "The scheduler type to use"})
    warmup_ratio: Optional[float] = field(default=0.01, metadata={"help": "Ratio of total training steps"})
    batch_size: Optional[int] = field(default=64, metadata={"help": "the batch size"})
    seq_length: Optional[int] = field(default=512, metadata={"help": "Input sequence length"})
    gradient_accumulation_steps: Optional[int] = field(
        default=16, metadata={"help": "the number of gradient accumulation steps"}
    )
    load_in_8bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 8 bits precision"})
    load_in_4bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 4 bits precision"})
    use_peft: Optional[bool] = field(default=False, metadata={"help": "Wether to use PEFT or not to train adapters"})
    output_dir: Optional[str] = field(default="output", metadata={"help": "the output directory"})
    #merged_output_dir: Optional[str] = field(default="output", metadata={"help": "the output directory"})
    peft_lora_r: Optional[int] = field(default=64, metadata={"help": "the r parameter of the LoRA adapters"})
    peft_lora_alpha: Optional[int] = field(default=16, metadata={"help": "the alpha parameter of the LoRA adapters"})
    logging_steps: Optional[int] = field(default=1, metadata={"help": "the number of logging steps"})
    #use_auth_token: Optional[bool] = field(default=True, metadata={"help": "Use HF auth token to access the model"})
    num_train_epochs: Optional[int] = field(default=3, metadata={"help": "the number of training epochs"})
    max_steps: Optional[int] = field(default=-1, metadata={"help": "the number of training steps"})
    save_steps: Optional[int] = field(
        default=100, metadata={"help": "Number of updates steps before two checkpoint saves"}
    )
    save_total_limit: Optional[int] = field(default=10, metadata={"help": "Limits total number of checkpoints."})
    #push_to_hub: Optional[bool] = field(default=False, metadata={"help": "Push the model to HF Hub"})
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "Whether to use gradient checkpointing or no"}
    )
    gradient_checkpointing_kwargs: Optional[dict] = field(
        default=None,
        metadata={
            "help": "key word arguments to be passed along `torch.utils.checkpoint.checkpoint` method - e.g. `use_reentrant=False`"
        },
    )
    mixed_precision: Optional[str] = field(default="bf16", metadata={"help": "Mixed precision training"})
    target_modules: Optional[List[str]] = field(default=None, metadata={"help": "Target modules for LoRA adapters"})
    logging_dir: Optional[str] = field(default=None, metadata={"help": "Tensorboard log dir."})
    apply_chat_template: Optional[bool] = field(default=False, metadata={"help": "Wether to apply chat template"})
    system_prompt: Optional[str] = field(default=None, metadata={"help": "System prompt"})



def formatting_prompts_func(examples):
    global model_type, apply_chat_template, system_prompt
    output_text = []
    for i in range(len(examples["instruction"])):
        instruction = examples["instruction"][i]
        output = examples["output"][i]
        if apply_chat_template:
            if model_type == "qwen2":
                if system_prompt is None or len(system_prompt) == 0:
                    system_prompt = "You are a helpful assistant"
                instruction = "<|im_start|>system\n" + system_prompt + "<|im_end|>\n<|im_start|>user\n" + instruction + "<|im_end|>\n"
                output = "<|im_start|>assistant\n" + output + "<|im_end|>\n"
            elif model_type == "qwen" or model_type == "baichuan":
                instruction = "<|im_start|>user\n" + instruction + "<|im_end|>\n"
                output = "<|im_start|>assistant\n" + output + "<|im_end|>\n"
            elif model_type == "chatglm":
                instruction = "[gMASK]sop<|user|>\n " + instruction
                output = "<|assistant|>\n " + output
            elif model_type == "mistral":
                instruction = "<s>[INST]" + instruction + "[/INST]"
                output = output + "</s>"
            elif model_type == "llama":
                instruction = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n" + instruction + "<|eot_id|>"
                output = "<|start_header_id|>assistant<|end_header_id|>\n\n" + output + "<|eot_id|>"
            elif model_type == "gemma":
                instruction = "<start_of_turn>user\n" + instruction + "<end_of_turn>"
                output = "" + output + "<end_of_turn>"
        text = f'{instruction}{output}'
        output_text.append(text)
    return output_text


def load_dataset_by_source(path):
    try:
        if odps_utils.is_odps_table(path):
            dataset = odps_utils.ODPSTableDataset(path).get_data()
            df = pd.DataFrame(dataset)
            dataset = Dataset.from_pandas(df)
        else:
            dataset = load_dataset('json', data_files=path, split='train')
    except:
        dataset = load_dataset('json', data_files=path, split='train')
    return dataset


def main(): 
    tqdm.pandas()
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    # Step 1: Load the model and tokenizer
    if script_args.load_in_8bit and script_args.load_in_4bit:
        raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")
    elif script_args.load_in_8bit or script_args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=script_args.load_in_8bit, load_in_4bit=script_args.load_in_4bit
        )
        # Copy the model to each device
        device_map = (
            {"": f"xpu:{Accelerator().local_process_index}"}
            if is_xpu_available()
            else {"": Accelerator().local_process_index}
        )
    else:
        device_map = None
        quantization_config = None

    tokenizer = get_tokenizer(model_name_or_path=script_args.model_name, model=script_args.model_type)
    model = get_hf_model(script_args, tokenizer, quantization_config, device_map)

    # Step 2: Load the dataset
    train_dataset = load_dataset_by_source(script_args.train_dataset_name)
    if len(script_args.eval_dataset_name) > 0:
        eval_dataset = load_dataset_by_source(script_args.eval_dataset_name)
    else:
        eval_dataset = None

    # Step 3: Define the training arguments
    is_ampere = torch.cuda.get_device_capability()[0] >= 8
    assert script_args.mixed_precision in ['fp16', 'fp32', 'bf16'], "mixed_precision only support fp16、bf16、fp32"
    fp16_flag = False
    bf16_flag = False
    if script_args.mixed_precision == 'fp16':
        fp16_flag = True
    elif script_args.mixed_precision == 'bf16':
        if is_ampere:
            bf16_flag = True
        else:
            # for GPUs SM < 80 use fp16
            fp16_flag = True
    
    training_args = TrainingArguments(
        output_dir=script_args.output_dir,
        per_device_train_batch_size=script_args.batch_size,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        learning_rate=script_args.learning_rate,
        lr_scheduler_type=script_args.lr_scheduler_type,
        warmup_ratio=script_args.warmup_ratio,
        logging_steps=script_args.logging_steps,
        num_train_epochs=script_args.num_train_epochs,
        max_steps=script_args.max_steps,
        report_to=script_args.report_to,
        save_steps=script_args.save_steps,
        save_total_limit=script_args.save_total_limit,
        gradient_checkpointing=script_args.gradient_checkpointing,
        logging_dir=script_args.logging_dir,
        bf16=bf16_flag,
        fp16=fp16_flag
    )

    # Step 4: Define the LoraConfig
    if script_args.use_peft and script_args.peft_lora_r > 0:
        peft_config = LoraConfig(
            r=script_args.peft_lora_r,
            lora_alpha=script_args.peft_lora_alpha,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=script_args.target_modules,
        )
    else:
        peft_config = None

    # Step 5: Define the Trainer
    global model_type
    model_type = script_args.model_type
    global apply_chat_template
    apply_chat_template = script_args.apply_chat_template
    global system_prompt
    system_prompt = script_args.system_prompt

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        max_seq_length=script_args.seq_length,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        formatting_func=formatting_prompts_func
    )

    trainer.train()

    # Step 6: Save the model
    trainer.save_model(script_args.output_dir)
    tokenizer.save_pretrained(script_args.output_dir)


if __name__ == '__main__': 
    main()
