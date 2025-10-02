"""
Merge LoRA weights into thinker model, then save the full omni model.

Adapted from:
https://github.com/hiyouga/LLaMA-Factory/blob/main/scripts/qwen_omni_merge.py
"""

import os
import shutil

import torch
from peft import PeftConfig, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniThinkerForConditionalGeneration,
    VoxtralForConditionalGeneration,
)


def main(
    lora_model_name_or_path: str,
    submodule_name: str = "thinker",
    extra_file: str = "spk_dict.pt",
):
    peft_config = PeftConfig.from_pretrained(lora_model_name_or_path)
    # print(peft_config)
    base_model_name_or_path = peft_config.base_model_name_or_path

    if "voxtral" in base_model_name_or_path.lower():
        model_class = VoxtralForConditionalGeneration
    elif "qwen" in base_model_name_or_path.lower():
        # model_class = Qwen2_5OmniThinkerForConditionalGeneration
        model_class = Qwen2_5OmniForConditionalGeneration
    else:
        model_class = AutoModelForCausalLM
    model = model_class.from_pretrained(
        base_model_name_or_path,
        dtype=torch.bfloat16,
        device_map="auto",
        # quantization_config=BitsAndBytesConfig(
        #     load_in_8bit=True,
        #     llm_int8_enable_fp32_cpu_offload=True,
        # ),
        trust_remote_code=True,
        # low_cpu_mem_usage=True,
    )
    print("Loaded model")

    base_model = getattr(model, submodule_name)
    print(f"Extracted {submodule_name} submodule")

    lora_model = PeftModel.from_pretrained(base_model, lora_model_name_or_path)
    print("Loaded LoRA weights")

    merged_model = lora_model.merge_and_unload()
    print("LoRA weights merged successfully.")

    setattr(model, submodule_name, merged_model)
    print(f"Set merged {submodule_name} submodule to model")

    # output_dir = lora_model_name_or_path
    output_dir = lora_model_name_or_path + "_merged_omni"

    model.save_pretrained(
        output_dir,
        # max_shard_size=max_shard_size,
        safe_serialization=True,
    )
    print(f"Saved merged model to {output_dir}")

    processor = AutoProcessor.from_pretrained(base_model_name_or_path)
    processor.save_pretrained(output_dir)
    print("Saved processor")

    source_file = os.path.join(base_model_name_or_path, extra_file)
    target_file = os.path.join(output_dir, extra_file)
    if os.path.exists(source_file):
        shutil.copy(source_file, target_file)
        print(f"Copied {extra_file} file")
    else:
        print(f"File {extra_file} not found, skipping copy.")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
