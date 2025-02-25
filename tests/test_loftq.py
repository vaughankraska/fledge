import os

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    LlamaForCausalLM,
)

from fledge.loftq import quantize_and_save

TEST_BASE_MODEL = "vaughankraska/TestLlama3.2ish"
TEST_LOFTQ_MODEL = "vaughankraska/TestLlama3.2ish-4bit-16rank-T1"


def test_quantize_and_save():
    base_model_dir, lora_dir = quantize_and_save(TEST_BASE_MODEL, iter=1)

    assert os.path.exists(base_model_dir)
    assert os.path.exists(lora_dir)


def test_reload_quantized():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)

    base_model_dir, lora_dir = quantize_and_save(TEST_BASE_MODEL, iter=1)
    print("Loading from: ", base_model_dir)
    base_model = AutoModelForCausalLM.from_pretrained(
         base_model_dir,
         torch_dtype=torch.bfloat16,
         quantization_config=BitsAndBytesConfig(
             load_in_4bit=True,
             bnb_4bit_compute_dtype=torch.bfloat16,
             bnb_4bit_use_double_quant=False,
             bnb_4bit_quant_type='nf4',
         ),
    )

    peft_model = PeftModel.from_pretrained(
         model=base_model,
         model_id=base_model_dir,
         subfolder="loft_init",
         is_trainable=True,
         low_cpu_mem_usage=True,
         )

    peft_model = peft_model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(base_model_dir)
    tokenizer.pad_token = tokenizer.eos_token

    # Verify model configuration
    assert isinstance(base_model, LlamaForCausalLM)
    assert isinstance(peft_model, PeftModel)
    assert base_model.config.torch_dtype == torch.bfloat16
    # Verify tokenizer setup
    assert tokenizer.pad_token == tokenizer.eos_token
    # Test basic model functionality
    input_text = "Nine plus ten equals"
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    # Verify model can generate output
    with torch.no_grad():
        outputs = peft_model.generate(
            input_ids=inputs["input_ids"],
            max_length=10,
            pad_token_id=tokenizer.eos_token_id,
        )
    # Check if generation produced valid output
    assert outputs.shape[1] > inputs["input_ids"].shape[1]
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    assert isinstance(decoded_output, str)
    assert len(decoded_output) > 0


def test_reload_quantized_from_hub():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)

    base_model = AutoModelForCausalLM.from_pretrained(
         TEST_LOFTQ_MODEL,
         torch_dtype=torch.bfloat16,
         quantization_config=BitsAndBytesConfig(
             load_in_4bit=True,
             bnb_4bit_compute_dtype=torch.bfloat16,
             bnb_4bit_use_double_quant=False,
             bnb_4bit_quant_type='nf4',
         ),
    )

    peft_model = PeftModel.from_pretrained(
         model=base_model,
         model_id=TEST_LOFTQ_MODEL,
         subfolder="loft_init",
         is_trainable=True,
         low_cpu_mem_usage=True,
         )

    peft_model = peft_model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(TEST_BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    # Verify model configuration
    assert isinstance(base_model, LlamaForCausalLM)
    assert isinstance(peft_model, PeftModel)
    assert base_model.config.torch_dtype == torch.bfloat16
    # Verify tokenizer setup
    assert tokenizer.pad_token == tokenizer.eos_token
    # Test basic model functionality
    input_text = "Nine plus ten equals"
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    # Verify model can generate output
    with torch.no_grad():
        outputs = peft_model.generate(
            input_ids=inputs["input_ids"],
            max_length=10,
            pad_token_id=tokenizer.eos_token_id,
        )
    # Check if generation produced valid output
    assert outputs.shape[1] > inputs["input_ids"].shape[1]
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    assert isinstance(decoded_output, str)
    assert len(decoded_output) > 0
