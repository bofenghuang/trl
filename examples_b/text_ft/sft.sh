#!/usr/bin/env bash

set -e

echo "START TIME: $(date)"

# export OMP_NUM_THREADS="1"
# export TOKENIZERS_PARALLELISM="false"

# export CUDA_VISIBLE_DEVICES=4
# export CUDA_VISIBLE_DEVICES=4,5,6,7
export CUDA_VISIBLE_DEVICES=6,7

export CUDA_LAUNCH_BLOCKING=1

export WANDB_PROJECT="2025-zaion-intent-text"

root_dir="/home/bhuang/llm/trl"

# not used
# --packing \  # attention cross contamination
# --padding_free \

# tuned parameters

# optimizer
# --optim adamw_8bit \
# https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py#L143

# efficiency
# --use_liger_kernel \
# --torch_compile \

# output_dir="./outputs/intent_classification/sft"
output_dir="./outputs/intent_classification/text_ft/databank/sft"

# model_name="/projects/bhuang/models/llm/pretrained/Qwen/Qwen3-4B"
model_name="/projects/bhuang/models/llm/pretrained/Qwen/Qwen3-4B-Instruct-2507"

chat_template_path="/home/bhuang/llm/trl/examples_b/chat_templates/qwen3.jinja"

# train_dataset_file="/home/bhuang/llm/momo/intent_classification/data/sga/train_conv.jsonl"
# eval_dataset_file="/home/bhuang/llm/momo/intent_classification/data/sga/test_conv.jsonl"

train_dataset_file="/home/bhuang/llm/momo/intent_classification/data/databank/train_conv.jsonl"
# train_dataset_file="/home/bhuang/llm/momo/intent_classification/data/databank/train_conv_w_description.jsonl"

run_name="intent_sft_qwen3_4b_instruct_lora_r64_ep10_bs64_lr2e4_wo_description"

# cmd
# cmd="python"
cmd="accelerate launch"
# cmd="torchrun --nproc_per_node $N_GPUS -m"

# script
script_path="$root_dir/examples_b/text_ft/sft.py"
# script_path="$root_dir/examples_b/text_ft/sft_unsloth.py"

    # --eval_dataset_file $eval_dataset_file \
    # --eval_strategy steps \

$cmd $script_path \
    --model_name_or_path $model_name \
    --torch_dtype bfloat16 \
    --attn_implementation flash_attention_2 \
    --use_peft \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_target_modules all-linear \
    --train_dataset_file $train_dataset_file \
    --output_dir $output_dir/$run_name \
    --overwrite_output_dir \
    --num_train_epochs 10 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --optim adamw_torch_fused \
    --learning_rate 2e-4 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --bf16 true \
    --gradient_checkpointing \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --ddp_find_unused_parameters false \
    --chat_template_path $chat_template_path \
    --assistant_only_loss true \
    --pad_to_multiple_of 64 \
    --dataloader_num_workers 8 \
    --dataset_num_proc 32 \
    --eos_token '<|im_end|>' \
    --max_length 8192 \
    --eval_strategy no \
    --eval_steps 50 \
    --save_strategy steps \
    --save_steps 50 \
    --save_total_limit 10 \
    --logging_steps 10 \
    --report_to all \
    --run_name $run_name \


echo "END TIME: $(date)"
