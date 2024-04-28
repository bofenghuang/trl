# flake8: noqa
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
"""
# regular:
python examples/scripts/sft.py \
    --model_name_or_path="facebook/opt-350m" \
    --report_to="wandb" \
    --learning_rate=1.41e-5 \
    --per_device_train_batch_size=64 \
    --gradient_accumulation_steps=16 \
    --output_dir="sft_openassistant-guanaco" \
    --logging_steps=1 \
    --num_train_epochs=3 \
    --max_steps=-1 \
    --push_to_hub \
    --gradient_checkpointing

# peft:
python examples/scripts/sft.py \
    --model_name_or_path="facebook/opt-350m" \
    --report_to="wandb" \
    --learning_rate=1.41e-5 \
    --per_device_train_batch_size=64 \
    --gradient_accumulation_steps=16 \
    --output_dir="sft_openassistant-guanaco" \
    --logging_steps=1 \
    --num_train_epochs=3 \
    --max_steps=-1 \
    --push_to_hub \
    --gradient_checkpointing \
    --use_peft \
    --lora_r=64 \
    --lora_alpha=16
"""

import logging
import os
import sys
from contextlib import nullcontext

TRL_USE_RICH = os.environ.get("TRL_USE_RICH", False)

from trl.commands.cli_utils import init_zero_verbose, SFTScriptArguments, TrlParser

if TRL_USE_RICH:
    init_zero_verbose()
    FORMAT = "%(message)s"

    from rich.console import Console
    from rich.logging import RichHandler

import torch
from datasets import load_dataset
import datasets
import transformers

from tqdm.rich import tqdm
from transformers import AutoTokenizer, set_seed
from transformers.trainer_utils import get_last_checkpoint

from trl import (
    ModelConfig,
    RichProgressCallback,
    SFTConfig,
    SFTTrainer,
    get_peft_config,
    get_quantization_config,
    get_kbit_device_map,
    DataCollatorForCompletionOnlyLM,
    CHAT_TEMPLATE_MAPPINGS,
)
from unsloth import FastLanguageModel

tqdm.pandas()

if TRL_USE_RICH:
    logging.basicConfig(format=FORMAT, datefmt="[%X]", handlers=[RichHandler()], level=logging.INFO)

logger = logging.getLogger(__name__)


def main():
    parser = TrlParser((SFTScriptArguments, SFTConfig, ModelConfig))
    args, training_args, model_config = parser.parse_args_and_config()

    # Force use our print callback
    if TRL_USE_RICH:
        training_args.disable_tqdm = True
        console = Console()

    # Set seed for reproducibility
    set_seed(training_args.seed)

    # setup logging
    # logging.basicConfig(
    #     format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    #     datefmt="%Y-%m-%d %H:%M:%S",
    #     handlers=[logging.StreamHandler(sys.stdout)],
    # )
    # log_level = training_args.get_process_log_level()
    # logger.setLevel(log_level)
    # datasets.utils.logging.set_verbosity(log_level)
    # transformers.utils.logging.set_verbosity(log_level)
    # transformers.utils.logging.enable_default_handler()
    # transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_config}")
    logger.info(f"Data parameters {args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    # check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    ################
    # Model & Tokenizer
    ################
    # torch_dtype = (
    #     model_config.torch_dtype
    #     if model_config.torch_dtype in ["auto", None]
    #     else getattr(torch, model_config.torch_dtype)
    # )
    # quantization_config = get_quantization_config(model_config)
    # model_kwargs = dict(
    #     revision=model_config.model_revision,
    #     trust_remote_code=model_config.trust_remote_code,
    #     attn_implementation=model_config.attn_implementation,
    #     # torch_dtype=torch_dtype,
    #     torch_dtype=model_config.torch_dtype,
    #     use_cache=False if training_args.gradient_checkpointing else True,
    #     device_map=get_kbit_device_map() if quantization_config is not None else None,
    #     quantization_config=quantization_config,
    # )
    # tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path, use_fast=True)

    # unsloth
    # 4bit pre quantized models we support for 4x faster downloading + no OOMs.
    # fourbit_models = [
    #     "unsloth/mistral-7b-bnb-4bit",
    #     "unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
    #     "unsloth/llama-2-7b-bnb-4bit",
    #     "unsloth/gemma-7b-bnb-4bit",
    #     "unsloth/gemma-7b-it-bnb-4bit", # Instruct version of Gemma 7b
    #     "unsloth/gemma-2b-bnb-4bit",
    #     "unsloth/gemma-2b-it-bnb-4bit", # Instruct version of Gemma 2b
    #     "unsloth/llama-3-8b-bnb-4bit", # [NEW] 15 Trillion token Llama-3
    # ] # More models at https://huggingface.co/unsloth

    model, tokenizer = FastLanguageModel.from_pretrained(
        # model_name="unsloth/mistral-7b-bnb-4bit", # Supports Llama, Mistral - replace this!
        model_name=model_config.model_name_or_path, # Supports Llama, Mistral - replace this!
        max_seq_length=training_args.max_seq_length,
        dtype=None,  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
        load_in_4bit=model_config.load_in_4bit,
    )

    # Do model patching and add fast LoRA weights
    model = FastLanguageModel.get_peft_model(
        model,
        r=model_config.lora_r,
        target_modules=model_config.lora_target_modules,
        lora_alpha=model_config.lora_alpha,
        modules_to_save=model_config.lora_modules_to_save,
        lora_dropout=0, # Supports any, but = 0 is optimized
        bias="none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state=3407,
        max_seq_length = training_args.max_seq_length,
        use_rslora=False,  # We support rank stabilized LoRA
        loftq_config=None, # And LoftQ
    )

    # tokenizer.pad_token = tokenizer.eos_token
    # if tokenizer.pad_token_id is None:
    # tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(model_config.pad_token)
    # if model_config.truncation_side is not None:
    #     tokenizer.truncation_side = model_config.truncation_side

    logger.info(f"EOS: {tokenizer.eos_token_id} / {tokenizer.eos_token}")
    logger.info(f"BOS: {tokenizer.bos_token_id} / {tokenizer.bos_token}")
    logger.info(f"PAD: {tokenizer.pad_token_id} / {tokenizer.pad_token}")
    logger.info(f"UNK: {tokenizer.unk_token_id} / {tokenizer.unk_token}")

    # bh: Make sure to have a pad_token_id which is different from eos_token_id which can result in the model not properly predicting EOS (End of Sentence) tokens during generation.
    # bh: DataCollatorForCompletionOnlyLM is a subclass of DataCollatorForLanguageModeling (not the usual one DataCollatorForSeq2Seq)
    # bh: which automatically replace all pad_token_id in labels by -100 (see torch_call)
    # bh: we don't want eos_token_id to be replaced by -100
    assert tokenizer.pad_token_id != tokenizer.eos_token_id, "tokenizer.pad_token_id and tokenizer.pad_token_id shouldn't be same"

    # bh: set chat_template for tokenizer, later used to format data
    chat_template = CHAT_TEMPLATE_MAPPINGS.get(model_config.chat_template)
    tokenizer.chat_template = chat_template["chat_template"]

    # load pretrained model
    # model, tokenizer = setup_chat_format(model, tokenizer)
    # model_kwargs = None

    ################
    # load datasets
    ################
    # raw_datasets = load_dataset(args.dataset_name)
    # train_dataset = raw_datasets[args.dataset_train_split]
    # eval_dataset = raw_datasets[args.dataset_test_split]

    train_dataset = load_dataset("json", data_files=args.train_dataset_name_or_path, split="train")
    eval_dataset = load_dataset("json", data_files=args.eval_dataset_name_or_path, split="train") if args.eval_dataset_name_or_path else None

    # bh: define data collator
    data_collator = DataCollatorForCompletionOnlyLM(
        instruction_template=chat_template["instruction_template"],
        response_template=chat_template["response_template"],
        tokenizer=tokenizer,
        mlm=False
    )

    ################
    # Optional rich context managers
    ###############
    init_context = nullcontext() if not TRL_USE_RICH else console.status("[bold green]Initializing the SFTTrainer...")
    save_context = (
        nullcontext()
        if not TRL_USE_RICH
        else console.status(f"[bold green]Training completed! Saving the model to {training_args.output_dir}")
    )

    ################
    # Training
    ################
    with init_context:
        trainer = SFTTrainer(
            model=model,
            # model=model_config.model_name_or_path,
            # model_init_kwargs=model_kwargs,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            # packing=True,
            # peft_config=get_peft_config(model_config),
            data_collator=data_collator,
            callbacks=[RichProgressCallback] if TRL_USE_RICH else None,
        )

    # trainer.train()
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    with save_context:
        # bh: restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    main()
