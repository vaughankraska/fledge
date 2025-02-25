# Copyright 2023-present the HuggingFace Inc. team.
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

import argparse
import os

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from peft import LoftQConfig, LoraConfig, TaskType, get_peft_model
from typing import Optional, Tuple


class Shell(nn.Module):
    def __init__(self, weight, bias=None):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        if bias is not None:
            self.bias = nn.Parameter(bias, requires_grad=False)


def unwrap_model(model, sub_module_name=".base_layer", do_print: bool = False):
    sub_module_name_list = [
        k.split(sub_module_name)[0]
        for k in model.state_dict().keys()
        if sub_module_name in k
    ]
    sub_module_name_set = set(sub_module_name_list)
    for name in sub_module_name_set:
        # get the parent of the submodule
        name_parent = ".".join(name.split(".")[:-1])
        name_child = name.split(".")[-1]
        sub_module = model.get_submodule(name_parent)
        if do_print:
            print(sub_module)

        # replace with shell
        child = getattr(sub_module, name_child)
        weight = getattr(child.base_layer, "weight", None)
        bias = getattr(child.base_layer, "bias", None)
        shell = Shell(weight, bias)

        setattr(sub_module, name_child, shell)

    print("You have unwrapped the model. Use it at your own risk.")


def print_model(model, name):
    print("=" * 10 + name + "=" * 10)
    print(model)
    for name, param in model.named_parameters():
        if torch.is_tensor(param):
            if param.dtype in [torch.float32, torch.float16]:
                print(
                    name,
                    param.shape,
                    param.device,
                    param.dtype,
                    param.requires_grad,
                    param.mean().item(),
                    param.max().item(),
                )
            else:
                print(name, param.shape, param.device, param.dtype, param.requires_grad)


def arg_parse():
    parser = argparse.ArgumentParser(description="Quantize a model with LoftQ.")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="The name or path of the fp32/16 model.",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=os.environ.get("HF_TOKEN", None),
        help="The access token to download model from HuggingFace Hub.",
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=4,
        help="The quantized bits",
    )
    parser.add_argument(
        "--iter",
        type=int,
        default=3,
        help="`T` The alternating steps in LoftQ",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=16,
        help="The rank of the LoRA adapter",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./model_zoo/loftq/",
        help="The rank of the LoRA adapter",
    )
    parser.add_argument(
        "--print",
        type=bool,
        default=True,
        help="Print the model layers and architecture",
    )
    args = parser.parse_args()
    return args


def quantize_and_save(
    model_name_or_path: str,
    token: Optional[str] = None,
    bits: Optional[int] = 4,
    iter: Optional[int] = 3,
    rank: Optional[int] = 16,
    save_dir: Optional[str] = "./model_zoo/loftq/",
    do_print: bool = False,
) -> Tuple[str, str]:
    # Download weights and configure LoRA
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, token=token, trust_remote_code=True
    )
    if any(
        name in model_name_or_path.lower() for name in ["llama", "mistral", "falcon"]
    ):
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            token=token,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        task_type = TaskType.CAUSAL_LM
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "down_proj",
            "gate_proj",
        ]

    elif any(name in model_name_or_path.lower() for name in ["bart", "t5"]):
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, token=token)
        task_type = TaskType.SEQ_2_SEQ_LM
        target_modules = ["q_proj", "k_proj", "v_proj", "fc1", "fc2", "out_proj"]

    elif any(
        name in model_name_or_path.lower() for name in ["deberta", "roberta", "bert"]
    ):
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path, token=token
        )
        task_type = TaskType.SEQ_CLS
        target_modules = [
            "query_proj",
            "key_proj",
            "value_proj",
            "dense",
        ]  # embeddings not supported by peft
    else:
        raise NotImplementedError("Other models not supported yet.")

    # Config of LoftQ
    loftq_config = LoftQConfig(loftq_bits=bits, loftq_iter=iter)

    lora_config = LoraConfig(
        task_type=task_type,
        inference_mode=True,
        r=rank,
        lora_alpha=16 if task_type is TaskType.CAUSAL_LM else rank,
        lora_dropout=0.1,
        target_modules=target_modules,
        init_lora_weights="loftq",
        loftq_config=loftq_config,
    )

    # Obtain LoftQ model
    lora_model = get_peft_model(model, lora_config)
    base_model = lora_model.get_base_model()

    # Save LoftQ model
    model_name = (
        model_name_or_path.split("/")[-1] + f"-{bits}bit" + f"-{rank}rank-T{iter}"
    )
    base_model_dir = os.path.join(save_dir, model_name)
    lora_model_dir = os.path.join(save_dir, model_name, "loft_init")

    # save lora adapters first
    lora_model.base_model.peft_config[
        "default"
    ].base_model_name_or_path = (
        base_model_dir  # This can be a local path or Hub model id
    )
    lora_model.base_model.peft_config[
        "default"
    ].init_lora_weights = True  # Don't apply LoftQ when loading again

    lora_model.save_pretrained(lora_model_dir)
    if do_print:
        print_model(lora_model, "lora_model")

    # remove lora adapters and save the backbone
    unwrap_model(base_model, do_print=do_print)
    base_model.save_pretrained(base_model_dir, save_peft_format=True)
    tokenizer.save_pretrained(base_model_dir)

    if do_print:
        print_model(base_model, "base_model")
    print("Footprint: ", base_model.get_memory_footprint())

    return base_model_dir, lora_model_dir


if __name__ == "__main__":
    args = arg_parse()
    base_dir, lora_dir = quantize_and_save(
        args.model_name_or_path,
        args.token,
        args.bits,
        args.iter,
        args.rank,
        args.save_dir,
        args.print,
    )

# example command:
# python quantize_save_load.py \
# --model_name_or_path unsloth/Llama3.2-1B \
# --token XXX \
# --bits 4 --iter 5 --rank 16 \
# --save_dir ./model_zoo/loftq/
