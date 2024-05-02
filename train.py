# -*- coding: utf-8 -*-

import logging
import subprocess
import os
import json
import shlex
import sys
import socket
import time

from training_utils.algorithm import get_algorithm_definition
from training_utils.training_job import (
    get_hyper_parameters,
    get_training_job_definition,
    check_parameters,
    check_channels,
    get_input_channel_info,
    get_output_channel_info,
    get_raw_input_channel,
    CHANNEL_TYPE_MAX_COMPUTE
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
)

DATA_TRAIN_CHANNEL = 'train'
DATA_DEV_CHANNEL = 'validation'
# input
MODEL_CHANNEL = 'model'
# output
CHECKPOINT_CHANNEL = 'model'

# temp path
TEMP_MODEL_PATH = '/tmp/model/'
TEMP_ADAPTER_PATH = '/tmp/adapter/'
TEMP_INPUT_MODEL_PATH = '/tmp/input_model/'

# lora
LORA_TARGET_MODULES_DICT = {
    "qwen": "c_proj w1 w2",
    "chatglm": "query_key_value dense",
    "llama": "k_proj o_proj q_proj v_proj",
    "baichuan": "W_pack o_proj",
    "bloom": "query_key_value dense",
    "aquila": "k_proj o_proj q_proj v_proj",
    "falcon": "query_key_value dense",
    "dolly": "query_key_value dense",
    "qwen2": "k_proj o_proj q_proj v_proj",
    "mistral": "k_proj o_proj q_proj v_proj",
    #    "deepseek": "k_proj o_proj q_proj v_proj",
    #    "codellama": "k_proj o_proj q_proj v_proj",
    "gemma": "k_proj o_proj q_proj v_proj",
    "mixtral": "k_proj o_proj q_proj v_proj"
}

# architectures_dict
ARCHITECTURES_TO_MODEL_DICT = {
    "QWenLMHeadModel": "qwen",
    "ChatGLMModel": "chatglm",
    "LlamaForCausalLM": "llama",  # "deepseek", "codellama"
    "BaichuanForCausalLM": "baichuan",
    "BloomForCausalLM": "bloom",
    "AquilaForCausalLM": "aquila",
    "AquilaModel": "aquila",
    "RWForCausalLM": "falcon",
    "GPTNeoXForCausalLM": "dolly",
    "Qwen2ForCausalLM": "qwen2",
    "MistralForCausalLM": "mistral",
    "GemmaForCausalLM": "gemma",
    "MixtralForCausalLM": "mixtral"
}


def wait_pytorch_master_ready(max_retry_times=30, sleep_second=5):
    master_addr = os.environ.get("MASTER_ADDR", "{}")
    retry_times = 0
    while True:
        try:
            socket.gethostbyname(master_addr)
            logging.info(f"Get hostname {master_addr} succeed. retry times: {retry_times}")
            break
        except Exception as ex:
            if retry_times < max_retry_times:
                logging.info(f"Get hostname {master_addr} error. retry times: {retry_times}")
                time.sleep(sleep_second)
                retry_times += 1
                continue
            logging.info(f"Get hostname f{master_addr} error.")
            raise ex


# must be a file
def check_input_channel_file_type(training_job, channel):
    channel_type, data_channel_path = get_input_channel_info(
        channel_name=channel,
        training_job=training_job,
    )
    if not data_channel_path:
        err_msg = channel + " channel is not set"
        raise ValueError(err_msg)

    if data_channel_path.endswith("/"):
        err_msg = channel + "must be a file"
        raise ValueError(err_msg)


# must be a folder
def check_input_channel_folder_type(training_job, channel):
    channel_type, data_channel_path = get_input_channel_info(
        channel_name=channel,
        training_job=training_job,
    )
    if not data_channel_path:
        err_msg = channel + " channel is not set"
        raise ValueError(err_msg)

    if not data_channel_path.endswith("/"):
        err_msg = channel + "must be a directory"
        raise ValueError(err_msg)


# output channel must be a directory
def check_output_channel(training_job, channel):
    # output model
    _, output_path = get_output_channel_info(
        channel_name=channel,
        training_job=training_job,
    )
    if not output_path:
        err_msg = channel + " channel is not set"
        raise ValueError(err_msg)

    if not output_path.endswith("/"):
        err_msg = channel + "must be a directory"
        raise ValueError(err_msg)


# for pre-checking
def check_channel_path(training_job=None):
    if not training_job:
        training_job = get_training_job_definition()

    train_input_channel = get_raw_input_channel(DATA_TRAIN_CHANNEL, training_job)
    if train_input_channel["type"] == CHANNEL_TYPE_MAX_COMPUTE:
        check_input_channel_folder_type(training_job, DATA_TRAIN_CHANNEL)
    else:
        check_input_channel_file_type(training_job, DATA_TRAIN_CHANNEL)

    try:
        val_input_channel = get_raw_input_channel(DATA_DEV_CHANNEL, training_job)
        if val_input_channel["type"] == CHANNEL_TYPE_MAX_COMPUTE:
            check_input_channel_folder_type(training_job, DATA_DEV_CHANNEL)
        else:
            check_input_channel_file_type(training_job, DATA_DEV_CHANNEL)
    except:
        print('*' * 5, 'No Validation dataset provided!', '*' * 5)

    check_input_channel_folder_type(training_job, MODEL_CHANNEL)
    check_output_channel(training_job, CHECKPOINT_CHANNEL)


def get_gpu_count():
    kubernetes_container_resource_gpu = os.environ.get("KUBERNETES_CONTAINER_RESOURCE_GPU")
    if not kubernetes_container_resource_gpu:
        nvidia_visible_devices = os.environ.get("NVIDIA_VISIBLE_DEVICES")
        if nvidia_visible_devices.strip():
            return len(nvidia_visible_devices.split(","))
        else:
            return 0
    else:
        return int(kubernetes_container_resource_gpu)


def generate_train_command(hyper_params, training_job, model_type):
    _, train_data_channel_path = get_input_channel_info(channel_name=DATA_TRAIN_CHANNEL, training_job=training_job)
    _, valid_data_channel_path = get_input_channel_info(channel_name=DATA_DEV_CHANNEL, training_job=training_job)

    command = f'accelerate launch'
    command += f' --num_processes {get_gpu_count()}'
    command += f' --config_file multi_gpu.yaml sft.py'
    command += f' --model_name {TEMP_INPUT_MODEL_PATH}'
    command += f' --model_type {model_type}'
    command += f' --train_dataset_name {train_data_channel_path}'
    command += f' --num_train_epochs {hyper_params["num_train_epochs"]}'
    command += f' --batch_size {hyper_params["per_device_train_batch_size"]}'
    command += f' --gradient_accumulation_steps {hyper_params["gradient_accumulation_steps"]}'
    command += f' --seq_length {hyper_params["seq_length"]}'
    command += f' --learning_rate {hyper_params["learning_rate"]}'
    command += f' --system_prompt \"{hyper_params["system_prompt"]}\"'

    if valid_data_channel_path is not None:
        command += f' --eval_dataset_name {valid_data_channel_path}'

    if hyper_params["apply_chat_template"] == 'True' or hyper_params["apply_chat_template"] == 'true':
        command += f' --apply_chat_template'

    if int(hyper_params["lora_dim"]) > 0:
        target_modules = LORA_TARGET_MODULES_DICT[model_type]
        command += f' --use_peft'
        command += f' --target_modules {target_modules}'
        command += f' --peft_lora_r {hyper_params["lora_dim"]}'
        command += f' --peft_lora_alpha {hyper_params["lora_alpha"]}'
        # do qlora
        if hyper_params["load_in_4bit"] == 'True' or hyper_params["load_in_4bit"] == 'true':
            command += f' --load_in_4bit'
        elif hyper_params["load_in_8bit"] == 'True' or hyper_params["load_in_8bit"] == 'true':
            command += f' --load_in_8bit'
        command += f' --output_dir {TEMP_ADAPTER_PATH}'
    else:
        command += f' --output_dir {TEMP_MODEL_PATH}'

    tb_logging_dir = os.environ.get("PAI_OUTPUT_TENSORBOARD")
    if tb_logging_dir:
        command += f' --logging_dir {tb_logging_dir}'
        command += f' --report_to tensorboard'
    return shlex.split(command)


def generate_convert_command(model_type):
    command = f'python convert.py '
    command += f' --model_name {TEMP_INPUT_MODEL_PATH}'
    command += f' --model_type {model_type}'
    command += f' --output_dir {TEMP_MODEL_PATH}'
    command += f' --adapter_dir {TEMP_ADAPTER_PATH}'
    return shlex.split(command)


def pre_check(hyper_params=None, training_job=None, algo_definition=None):
    if not hyper_params:
        hyper_params = get_hyper_parameters()
    if not algo_definition:
        algo_definition = get_algorithm_definition()
    check_parameters(algo_definition=algo_definition)
    check_channels(
        training_job=training_job,
        algo_definition=algo_definition,
    )
    check_channel_path(training_job=training_job)


def get_model_type():
    config_path = os.path.join(TEMP_INPUT_MODEL_PATH, "config.json")
    with open(config_path, "r") as f:
        config_json = json.load(f)
    arch_name = config_json["architectures"][0]
    return ARCHITECTURES_TO_MODEL_DICT[arch_name]


def run_cmd(cmd_command):
    logging.info(f"execute command: {cmd_command}")
    process = subprocess.Popen(cmd_command, stdout=subprocess.PIPE, env=os.environ.copy(),
                               stderr=subprocess.STDOUT, encoding="utf-8", shell=True)
    while True:
        out = process.stdout.readline()
        if out == "" and process.poll() is not None:
            if process.returncode != 0:
                logging.info(f"execute command failed, exit_code: {process.returncode}")
                sys.exit(process.returncode)
            else:
                logging.info("execute command succeed")
            break
        if out != "":
            out = out.rstrip()
            logging.info(out)


def main():
    hyper_params = get_hyper_parameters()
    logging.info(f"hyper_params: {hyper_params}")

    algo_definition = get_algorithm_definition()
    training_job = get_training_job_definition()

    pre_check(
        hyper_params=hyper_params,
        training_job=training_job,
        algo_definition=algo_definition,
    )
    wait_pytorch_master_ready()

    _, checkpoint_path = get_input_channel_info(channel_name=MODEL_CHANNEL, training_job=training_job)
    _, output_path = get_output_channel_info(channel_name=CHECKPOINT_CHANNEL, training_job=training_job)

    run_cmd(f"mkdir -p {TEMP_INPUT_MODEL_PATH}")
    run_cmd(f"cp -R {checkpoint_path}* {TEMP_INPUT_MODEL_PATH}")

    model_type = get_model_type()
    if model_type == 'qwen2' or model_type == 'mistral':
        run_cmd("pip install --no-index --find-links=assets/ transformers==4.37.0 tokenizers==0.15.1")
    elif model_type == 'gemma':
        run_cmd("pip install --no-index --find-links=assets/ transformers==4.38.1")

    train_command = generate_train_command(
        hyper_params=hyper_params,
        training_job=training_job,
        model_type=model_type
    )
    logging.info(f"Sft training command: {train_command}")
    run_cmd(f"mkdir -p {TEMP_MODEL_PATH}")
    if int(hyper_params["lora_dim"]) > 0:
        run_cmd(f"mkdir -p {TEMP_ADAPTER_PATH}")

    exit_code = subprocess.call(train_command, env=os.environ.copy())
    if exit_code != 0:
        logging.info(f"execute command failed, exit_code: {exit_code}")
        sys.exit(exit_code)
    else:
        if int(hyper_params["lora_dim"]) > 0:
            # covert the model
            covert_command = generate_convert_command(model_type)
            logging.info(f"Covert command: {covert_command}")
            exit_code = subprocess.call(covert_command, env=os.environ.copy())
            if exit_code != 0:
                logging.info(f"execute convert command failed, exit_code: {exit_code}")
                sys.exit(exit_code)

        # do copy
        run_cmd(f"cp -r {TEMP_MODEL_PATH} " + output_path.replace("model/", ""))


if __name__ == "__main__":
    main()
