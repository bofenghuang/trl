"""Run SFT training with Unsloth model implementation."""

import argparse
import warnings
from dataclasses import dataclass, field

import torch
from datasets import DatasetDict, load_dataset
from unsloth import FastLanguageModel

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


@dataclass
class MyModelConfig(ModelConfig):
    loftq_config: bool = field(
        default=None,
        metadata={"help": "LoftQ config."},
    )
    random_state: int = field(
        default=3407,
        metadata={"help": "Random state."},
    )
    finetune_vision_layers: bool = field(
        default=False,
        metadata={"help": "Whether to finetune vision layers."},
    )
    finetune_language_layers: bool = field(
        default=True,
        metadata={"help": "Whether to finetune language layers."},
    )
    finetune_attention_modules: bool = field(
        default=True,
        metadata={"help": "Whether to finetune attention modules."},
    )
    finetune_mlp_modules: bool = field(
        default=True,
        metadata={"help": "Whether to finetune mlp modules."},
    )


def main(script_args, training_args, model_args, dataset_args):
    # Model init kwargs & Tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_args.model_name_or_path,
        dtype=getattr(torch, model_args.torch_dtype),  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
        max_seq_length=training_args.max_length,  # Context length - can be longer, but uses more memory
        load_in_4bit=model_args.load_in_4bit,  # 4bit uses much less memory
        load_in_8bit=model_args.load_in_8bit,  # A bit more accurate, uses 2x memory
        # todo: doesn't work
        # full_finetuning=not (model_args.load_in_4bit or model_args.load_in_8bit),  # We have full finetuning now!
        # device_map="auto",
        # device_map="balanced",  # will use multiple gpus even if launch with accelerate launch --num_processes 1
    )
    # Get PEFT model
    if model_args.use_peft:
        # todo: path problem and finetune_vision_layers/target_modules="all-linear" doesn't work
        model = FastLanguageModel.get_peft_model(
            model,
            # todo: not working
            # finetune_vision_layers=model_args.finetune_vision_layers,  # Turn off for just text!
            # finetune_language_layers=model_args.finetune_language_layers,  # Should leave on!
            # finetune_attention_modules=model_args.finetune_attention_modules,  # Attention good for GRPO
            # finetune_mlp_modules=model_args.finetune_mlp_modules,  # SHould leave on always!
            target_modules=model_args.lora_target_modules,
            r=model_args.lora_r,  # Choose any number > 0! Suggested 8, 16, 32, 64, 128
            lora_alpha=model_args.lora_alpha,  # Best to choose alpha = rank or rank*2
            lora_dropout=model_args.lora_dropout,  # Supports any, but = 0 is optimized
            bias="none",  # Supports any, but = "none" is optimized
            use_gradient_checkpointing=training_args.gradient_checkpointing,  # True or "unsloth" for very long context
            random_state=model_args.random_state,
            use_rslora=model_args.use_rslora,  # We support rank stabilized LoRA
            loftq_config=model_args.loftq_config,  # And LoftQ
        )

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

    # unsloth requires to apply chat template prior to trainer
    dataset = dataset.map(
        lambda x: {"text": tokenizer.apply_chat_template(x["messages"], tokenize=False, add_generation_prompt=True)},
        remove_columns=next(iter(dataset.values())).column_names,
        num_proc=training_args.dataset_num_proc,
    )

    # Initialize the SFT trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        tokenizer=tokenizer,
        peft_config=get_peft_config(model_args),
    )

    # Train the model
    trainer.train()

    # Save and push to Hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


def make_parser(subparsers: argparse._SubParsersAction = None):
    dataclass_types = (MyScriptArguments, SFTConfig, MyModelConfig, DatasetMixtureConfig)
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
