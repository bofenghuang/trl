"""Merge LoRA weights into base model."""

import torch
from peft import PeftConfig, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    Qwen2_5OmniThinkerForConditionalGeneration,
    VoxtralForConditionalGeneration,
)


def main(lora_model_name_or_path: str):
    peft_config = PeftConfig.from_pretrained(lora_model_name_or_path)
    # print(peft_config)
    base_model_name_or_path = peft_config.base_model_name_or_path

    if "voxtral" in base_model_name_or_path.lower():
        model_class = VoxtralForConditionalGeneration
    elif "qwen" in base_model_name_or_path.lower():
        model_class = Qwen2_5OmniThinkerForConditionalGeneration
    else:
        model_class = AutoModelForCausalLM
    base_model = model_class.from_pretrained(
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
    print("Loaded base model")

    lora_model = PeftModel.from_pretrained(base_model, lora_model_name_or_path)
    print("Loaded LoRA weights")

    merged_model = lora_model.merge_and_unload()
    print("LoRA weights merged successfully.")

    # output_dir = lora_model_name_or_path
    output_dir = lora_model_name_or_path + "_merged"

    merged_model.save_pretrained(
        output_dir,
        # max_shard_size=max_shard_size,
        safe_serialization=True,
    )
    print(f"Saved merged model to {output_dir}")

    processor = AutoProcessor.from_pretrained(base_model_name_or_path)
    processor.save_pretrained(output_dir)
    print("Saved processor")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
