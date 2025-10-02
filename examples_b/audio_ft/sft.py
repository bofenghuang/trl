"""
Run SFT training with Voxtral model.

Reference:
https://github.com/google-gemini/gemma-cookbook/blob/main/Gemma/%5BGemma_3n%5DAudio_understanding_with_HF.ipynb
https://github.com/huggingface/huggingface-gemma-recipes/blob/main/scripts/ft_gemma3n_audio_vt.py
"""

import argparse
import logging
import warnings
from dataclasses import dataclass, field

import torch
from datasets import DatasetDict, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    Qwen2_5OmniProcessor,
    Qwen2_5OmniThinkerForConditionalGeneration,
)

from trl import (
    DatasetMixtureConfig,
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    clone_chat_template,
    get_dataset,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)


# disable qwen2.5-omni chat template warning
logging.getLogger().setLevel(logging.ERROR)


def mask_left_including_subseq(labels: torch.Tensor, pattern: torch.Tensor) -> torch.Tensor:
    """Mask all tokens before and include the last sequence of tokens that matches the pattern"""
    # labels: (B, T), pattern: (L,)
    B, T = labels.shape
    L = pattern.numel()
    if L == 0 or L > T:
        return torch.zeros(B, T, dtype=torch.bool, device=labels.device)

    windows = labels.unfold(1, L, 1)                    # (B, T-L+1, L)
    starts = (windows == pattern).all(dim=-1)           # (B, T-L+1)

    idxs = torch.arange(T - L + 1, device=labels.device)
    last_start = torch.where(starts, idxs, torch.full_like(idxs, -1)).amax(dim=1)  # (B,)
    # Only compute last_end if we found a start
    last_end = torch.where(
        last_start >= 0,
        last_start + (L - 1),
        torch.full_like(last_start, -1),
    )                                                   # (B,)

    pos = torch.arange(T, device=labels.device)
    to_mask = pos <= last_end.unsqueeze(1)              # (B, T) include the subsequence
    return to_mask


@dataclass
class MyScriptArguments(ScriptArguments):
    train_dataset_file: str = field(
        default=None,
        metadata={"help": "Path or name of the dataset to load for training."},
    )
    eval_dataset_file: str = field(
        default=None,
        metadata={"help": "Path or name of the dataset to load for evaluation."},
    )


def main(script_args, training_args, model_args, dataset_args):
    # Create processor
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        # trust_remote_code=model_args.trust_remote_code,
        # use_fast=True
    )

    # Model init kwargs & Tokenizer
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        dtype=model_args.dtype,
        # use_cache=False if training_args.gradient_checkpointing else True,  # not supported
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    # Create model
    model_class = AutoModelForCausalLM
    if isinstance(processor, Qwen2_5OmniProcessor):
        model_class = Qwen2_5OmniThinkerForConditionalGeneration
    model = model_class.from_pretrained(model_args.model_name_or_path, **model_kwargs)
    # model.disable_talker()  # only for whole model

    if model.config.model_type == "gemma3n":
        # audio encoder
        for param in model.model.audio_tower.parameters():
            param.requires_grad = False
        # audio projector
        # for param in model.model.embed_audio.parameters():
        #     param.requires_grad = False
    elif model.config.model_type == "qwen2_5_omni_thinker":
        # audio encoder (but also include audio projector)
        for param in model.audio_tower.parameters():
            param.requires_grad = False
        for param in model.audio_tower.proj.parameters():
            param.requires_grad = True
    else:
        raise ValueError(f"Model type {model.config.model_type} not supported.")

    # Set default chat template if needed
    # if tokenizer.chat_template is None:
    #     # TODO: source should be passed as an argument
    #     model, tokenizer = clone_chat_template(model, tokenizer, "Qwen/Qwen3-0.6B")

    # Load the dataset
    if dataset_args.datasets and script_args.dataset_name:
        warnings.warn(
            "Both `datasets` and `dataset_name` are provided. The `datasets` argument will be used to load the "
            "dataset and `dataset_name` will be ignored."
        )
    elif dataset_args.datasets and not script_args.dataset_name:
        dataset = get_dataset(dataset_args)
    elif not dataset_args.datasets and script_args.dataset_name:
        dataset = load_dataset(
            script_args.dataset_name, name=script_args.dataset_config, streaming=script_args.dataset_streaming
        )
    elif script_args.train_dataset_file:
        # load from json file
        dataset = DatasetDict()
        dataset["train"] = load_dataset("json", data_files=script_args.train_dataset_file, split="train")
        dataset["test"] = (
            load_dataset("json", data_files=script_args.eval_dataset_file, split="train")
            if script_args.eval_dataset_file
            else None
        )
    else:
        raise ValueError("Either `datasets` or `dataset_name` must be provided.")

    def process_func(example):
        """Strip None text and path values from the example."""
        for message in example["messages"]:
            for content in message["content"]:
                if "text" in content and content["text"] is None:
                    del content["text"]
                if "audio" in content and content["audio"] is None:
                    del content["audio"]
        return example

    # dataset = dataset.map(
    #     process_func,
    #     num_proc=training_args.dataset_num_proc,
    #     desc="Processing dataset",
    # )

    def collate_fn(examples):

        # Apply chat template to get text
        # tokenize and left pad
        batch = processor.apply_chat_template(
            [process_func(example)["messages"] for example in examples],
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        )

        # Tokenize the texts and process the images
        # batch = processor(text=texts, audio=audios, return_tensors="pt", padding=True)

        # The labels are the input_ids, and we mask the padding tokens in the loss computation
        labels = batch["input_ids"].clone()

        if model.config.model_type == "gemma3n":
            token_ids_to_mask = [
                processor.tokenizer.pad_token_id,
                processor.tokenizer.audio_token_id,
                processor.tokenizer.boa_token,
                processor.tokenizer.eoa_token,
                # processor.tokenizer.image_token_id,
                # processor.tokenizer.boi_token_id,
                # processor.tokenizer.eoi_token_id,
            ]

            assistant_prefix = torch.LongTensor(
                [
                    105,  # <start_of_turn>
                    4368, # model
                    107,  # \n
                ]
            )
        elif model.config.model_type == "qwen2_5_omni_thinker":
            token_ids_to_mask = [
                processor.tokenizer.pad_token,
                processor.tokenizer.audio_token_id,
                processor.tokenizer.audio_bos_token_id,
                processor.tokenizer.audio_eos_token_id,
            ]

            assistant_prefix = torch.LongTensor(
                [
                    151644,  # <|im_start|>
                    77091, # assistant
                    198,  # \n
                ]
            )
        else:
            raise ValueError(f"Model type {model.config.model_type} not supported.")


        # Mask the tokens that we do not want to include in the loss computation
        # -100 is ignored during categorical cross entropy loss computation
        # mask audio tokens and special tokens
        # still use as safety measure
        for token_id_to_mask in token_ids_to_mask:
            labels[labels == token_id_to_mask] = -100

        # mask all tokens before the last assistant prefix
        to_mask = mask_left_including_subseq(labels, assistant_prefix)
        labels[to_mask] = -100

        batch["labels"] = labels
        return batch

    # Initialize the SFT trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        processing_class=processor, # todo
        peft_config=get_peft_config(model_args),
    )

    # Train the model
    trainer.train()

    # Save and push to Hub
    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


def make_parser(subparsers: argparse._SubParsersAction = None):
    dataclass_types = (MyScriptArguments, SFTConfig, ModelConfig, DatasetMixtureConfig)
    if subparsers is not None:
        parser = subparsers.add_parser("sft", help="Run the SFT training script", dataclass_types=dataclass_types)
    else:
        parser = TrlParser(dataclass_types)
    return parser


if __name__ == "__main__":
    parser = make_parser()
    # When using the trl cli, this script may be run with additional arguments, corresponding accelerate arguments.
    # To ensure that their parsing does not interfere with the script arguments, parse the arguments with
    # `return_remaining_strings=True`, then ignore the remaining strings.
    script_args, training_args, model_args, dataset_args, _ = parser.parse_args_and_config(
        return_remaining_strings=True
    )
    main(script_args, training_args, model_args, dataset_args)
