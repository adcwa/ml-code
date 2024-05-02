import os
import torch
from transformers import HfArgumentParser, AutoModel, AutoModelForCausalLM
from peft import PeftModel
from utils import get_tokenizer
from dataclasses import dataclass, field
from typing import Optional


# Define and parse arguments.
@dataclass
class ScriptArguments:
    model_type: Optional[str] = field(default="bloom", metadata={"help": "the model type"})
    model_name: Optional[str] = field(default="facebook/opt-350m", metadata={"help": "the model name"})
    adapter_dir: Optional[str] = field(default="", metadata={"help": "path to save adapter"})
    output_dir: Optional[str] = field(default="", metadata={"help": "path to save final checkpoint"})


def main():
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    if int(os.environ.get("RANK")) == 0:
        tokenizer = get_tokenizer(model_name_or_path=script_args.model_name, model=script_args.model_type)
        if 'chatglm' in script_args.model_type:
            base_model = AutoModel.from_pretrained(
                script_args.model_name,
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                script_args.model_name,
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
        merged_model = PeftModel.from_pretrained(base_model, script_args.adapter_dir)
        merged_model = merged_model.merge_and_unload()

        tokenizer.save_pretrained(script_args.output_dir)
        merged_model.save_pretrained(script_args.output_dir)


if __name__ == "__main__":
    main()
