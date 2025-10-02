import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import soundfile as sf
from datasets import load_dataset
from transformers import AutoProcessor, Gemma3nForConditionalGeneration
import torch
from audio_utils import get_waveform

def load_audio(path):
    return get_waveform(path, always_2d=False, output_sample_rate=16_000)

train_dataset_file = "/home/bhuang/llm/momo/intent_classification/data/sga/train_tts_kyutai_degraded_conv.jsonl"
model_name="/projects/bhuang/models/llm/pretrained/google/gemma-3n-E4B-it"

# dataset = load_dataset("json", data_files=train_dataset_file, split="train")
# # print(dataset)

# samples = [dataset[0]]
# messages = samples[0]["messages"]

model = Gemma3nForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa"
)
processor = AutoProcessor.from_pretrained(
    model_name,
    # padding_side="left"
)

def collate_fn(examples):
    texts = []
    audios = []

    for example in examples:
        # Apply chat template to get text
        text = processor.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        ).strip()
        texts.append(text)

        # Extract audios
        waveform, _ = load_audio(example["audio_path"])
        audios.append(waveform)

    # Tokenize the texts and process the images
    batch = processor(text=texts, audio=audios, return_tensors="pt", padding=True)

    # The labels are the input_ids, and we mask the padding tokens in the loss computation
    labels = batch["input_ids"].clone()

    # todo: also mask prompt text tokens
    # Use Gemma3n specific token masking
    labels[labels == processor.tokenizer.pad_token_id] = -100
    if hasattr(processor.tokenizer, "image_token_id"):
        labels[labels == processor.tokenizer.image_token_id] = -100
    if hasattr(processor.tokenizer, "audio_token_id"):
        labels[labels == processor.tokenizer.audio_token_id] = -100
    if hasattr(processor.tokenizer, "boi_token_id"):
        labels[labels == processor.tokenizer.boi_token_id] = -100
    if hasattr(processor.tokenizer, "eoi_token_id"):
        labels[labels == processor.tokenizer.eoi_token_id] = -100

    batch["labels"] = labels
    return batch


messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are a helpful assistant."}]
    },
    {
        "role": "user",
        "content": [
            # {"type": "image", "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"},
            # {"type": "text", "text": "Describe this image in detail."},
            {"type": "audio", "audio": "/home/bhuang/llm/momo/intent_classification/data/sga/audio_degraded/train/000000.wav"},
            {"type": "text", "text": "Describe this audio in detail."},
        ]
    }
]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    add_generation_prompt=True,
).to(model.device)

input_len = inputs["input_ids"].shape[-1]

with torch.inference_mode():
    generation = model.generate(**inputs.to(model.device, dtype=model.dtype), max_new_tokens=100, do_sample=False) # , cache_implementation="static"
    generation = generation[0][input_len:]

decoded = processor.decode(generation, skip_special_tokens=True)
print(decoded)
