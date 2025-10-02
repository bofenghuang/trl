"""Run SFT training with HF model implementation."""

import argparse
import importlib
import json
import os
import re
import sys
import warnings
from dataclasses import dataclass, field
from typing import Optional

from datasets import DatasetDict, load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.auto.modeling_auto import MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES

from trl import (
    DatasetMixtureConfig,
    GRPOConfig,
    GRPOTrainer,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    clone_chat_template,
    get_dataset,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
# from trl.rewards import think_format_reward


available_labels = None

def extract_final_completion(s):
    s = re.sub(r"^<think>.*?</think>", "", s, flags=re.DOTALL)
    s = s.strip()
    return s


def think_format_reward(completions, **kwargs):
    pattern = r"^<think>(?!.*<think>)(.*?)</think>.*$"
    matches = [re.match(pattern, completion, re.DOTALL | re.MULTILINE) for completion in completions]
    return [1.0 if match else 0.0 for match in matches]


def json_format_reward(completions, **kwargs):
    rewards = []
    for completion in completions:
        try:
            json.loads(completion)
            rewards.append(1.0)
        except json.JSONDecodeError:
            rewards.append(0.0)
    return rewards


def intent_correctness_reward(completions, target, **kwargs):
    rewards = []
    for completion, tg in zip(completions, target):
        ground_truth_label = json.loads(tg)["intent"]
        # remove think content
        completion = extract_final_completion(completion)
        try:
            predicted_label = json.loads(completion)["intent"]
            if predicted_label in available_labels:
                if predicted_label == ground_truth_label:
                    # wrong but in-vocabulary
                    rewards.append(1.0)
                else:
                    # wrong but in-vocabulary
                    rewards.append(0.0)
            else:
                # out-of-set => stronger penalty
                rewards.append(0.0)
                # small brevity penalty if model rambles
                # r -= 0.01 * max(0, len(comp.split()) - 1)
        except json.JSONDecodeError:
            rewards.append(0.0)
        except (KeyError, TypeError):
            rewards.append(0.0)
    return rewards


reward_funcs_registry = {
    "think_format_reward": think_format_reward,
    "json_format_reward": json_format_reward,
    "intent_correctness_reward": intent_correctness_reward,
}


@dataclass
class MyScriptArguments(ScriptArguments):
    reward_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Reward model id of a pretrained model hosted inside a model repo on huggingface.co or "
            "local path to a directory containing model weights saved using `PreTrainedModel.save_pretrained`."
        },
    )
    reward_funcs: Optional[list[str]] = field(
        default=None,
        metadata={
            "help": "Reward functions to use. It can be either one of  'think_format_reward'; or a dotted "
            "import path. (e.g., 'my_lib.rewards.custom_reward')."
        },
    )
    train_dataset_file: str = field(
        default=None,
        metadata={"help": "Path or name of the dataset to load for training."},
    )
    eval_dataset_file: str = field(
        default=None,
        metadata={"help": "Path or name of the dataset to load for evaluation."},
    )


def main(script_args, training_args, model_args, dataset_args):
    # Model init kwargs & Tokenizer
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=model_args.torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    # Create model
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    valid_image_text_architectures = MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES.values()

    if config.architectures and any(arch in valid_image_text_architectures for arch in config.architectures):
        from transformers import AutoModelForImageTextToText

        model_kwargs.pop("use_cache", None)  # Image models do not support cache
        model = AutoModelForImageTextToText.from_pretrained(model_args.model_name_or_path, **model_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)

    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, use_fast=True
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

    # tmp: get available labels
    tmp_ds = dataset["train"].map(lambda x: json.loads(x["target"]), remove_columns=dataset["train"].column_names)
    global available_labels
    available_labels = set(tmp_ds["intent"])
    print(f"Number of available labels: {len(available_labels)}")

    # Get the reward models and functions
    reward_funcs = []
    if script_args.reward_model_name_or_path:
        reward_funcs.append(script_args.reward_model_name_or_path)

    if script_args.reward_funcs:
        for func_name in script_args.reward_funcs:
            if func_name in reward_funcs_registry:
                reward_funcs.append(reward_funcs_registry[func_name])
            elif "." in func_name:
                module_path, func_name = func_name.rsplit(".", 1)
                sys.path.insert(0, os.getcwd())
                module = importlib.import_module(module_path)
                reward_func = getattr(module, func_name)
                reward_funcs.append(reward_func)
            else:
                raise ValueError(
                    f"Could not load reward function '{func_name}'. Expected one of "
                    f"{list(reward_funcs_registry.keys())} or a valid import path."
                )

    # Initialize the SFT trainer
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        reward_funcs=reward_funcs,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
    )

    # Train the model
    trainer.train()

    # Save and push to Hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


def make_parser(subparsers: argparse._SubParsersAction = None):
    dataclass_types = (MyScriptArguments, GRPOConfig, ModelConfig, DatasetMixtureConfig)
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
