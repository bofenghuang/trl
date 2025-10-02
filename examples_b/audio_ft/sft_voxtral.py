"""
Run SFT training with Voxtral model.

Reference:
https://github.com/Deep-unlearning/Finetune-Voxtral-ASR
https://github.com/Innovative-Digitale-Medizin-IDM/voxtral-finetune
"""

import argparse
import warnings
from dataclasses import dataclass, field

from datasets import DatasetDict, load_dataset
from transformers import VoxtralForConditionalGeneration, VoxtralProcessor

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
    processor = VoxtralProcessor.from_pretrained(
        model_args.model_name_or_path,
        # not supported by mistral tokenizer
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
    model = VoxtralForConditionalGeneration.from_pretrained(model_args.model_name_or_path, **model_kwargs)

    # Freeze the audio encoder model.audio_tower
    for param in model.audio_tower.parameters():
        param.requires_grad = False
    # for param in model.multi_modal_projector.parameters():
    #     param.requires_grad = False

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
                if "path" in content and content["path"] is None:
                    del content["path"]
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
            continue_final_message=True,  # allow last message to be from assistant
            # padding=True,
        )

        # Tokenize the texts and process the images
        # batch = processor(text=texts, audio=audios, return_tensors="pt", padding=True)

        # The labels are the input_ids, and we mask the padding tokens in the loss computation
        labels = batch["input_ids"].clone()

        # mask audio tokens and special tokens
        # still use as safety measure
        # todo: convert from tokens to token_ids
        tokens_to_mask = [
            11, # "pad"
            25, # "[BEGIN_AUDIO]"
            24, # "[AUDIO]"
            # 1,  # "<s>"
            # 3,  # "[INST]"
            # 4,  # "[/INST]", # don't mask this, will be used later
        ]
        for token_to_mask in tokens_to_mask:
            labels[labels == token_to_mask] = -100

        # mask all tokens before and include the last "[/INST]" token
        inst_id = 4
        mask = labels.eq(inst_id)  # (B, T) True where inst_id
        to_mask = mask.flip(1).cummax(dim=1).values.flip(1)
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
