#!/usr/bin/env bash

set -e

echo "START TIME: $(date)"

# export OMP_NUM_THREADS="1"
# export TOKENIZERS_PARALLELISM="false"

export CUDA_VISIBLE_DEVICES=
# export CUDA_VISIBLE_DEVICES=4,5,6,7

# debug
export CUDA_LAUNCH_BLOCKING=1
# export TORCHDYNAMO_VERBOSE=1
# export TORCHDYNAMO_DISABLE=1

export WANDB_PROJECT="2025-zaion-intent-audio"

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

# output_dir="./outputs/intent_classification/audio_ft/sft"
output_dir="./outputs/intent_classification/audio_ft/databank"

# model_name="/projects/bhuang/models/llm/pretrained/google/gemma-3n-E4B-it"
model_name="unsloth/gemma-3n-E4B-it-unsloth-bnb-4bit"

# chat_template_path="/home/bhuang/llm/trl/examples_b/chat_templates/qwen3.jinja"

# train_dataset_file="/home/bhuang/llm/momo/intent_classification/data/sga/train_conv.jsonl"
# train_dataset_file="/home/bhuang/llm/momo/intent_classification/data/sga/train_tts_kyutai_degraded_conv.jsonl"
# eval_dataset_file="/home/bhuang/llm/momo/intent_classification/data/sga/test_tts_kyutai_degraded_conv.jsonl"

train_dataset_file="/home/bhuang/llm/momo/intent_classification/data/databank/train_tts_kyutai_degraded_conv.jsonl"

run_name="intent_sft_gemma3n_e4b_it_lora_r64_ep20_bs32_lr5e5_unsloth"

# cmd
# cmd="python"
cmd="accelerate launch"
# cmd="accelerate launch --num_processes 1"
# cmd="torchrun --nproc_per_node $N_GPUS -m"

# script
script_path="$root_dir/examples_b/audio_ft/sft_gemma3n_unsloth.py"

    # --assistant_only_loss true \
    # --chat_template_path $chat_template_path w\

    # --eval_dataset_file $eval_dataset_file \
    # --eval_strategy steps \


$cmd $script_path \
    --model_name_or_path $model_name \
    --torch_dtype bfloat16 \
    --attn_implementation flash_attention_2 \
    --load_in_4bit \
    --use_peft \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj post linear_start linear_end embedding_projection \
    --lora_modules_to_save lm_head embed_tokens embed_audio \
    --train_dataset_file $train_dataset_file \
    --output_dir $output_dir/$run_name \
    --overwrite_output_dir \
    --num_train_epochs 20 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --optim adamw_8bit \
    --learning_rate 5e-5 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --bf16 true \
    --gradient_checkpointing \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --ddp_find_unused_parameters false \
    --remove_unused_columns false \
    --dataset_text_field "" \
    --dataset_kwargs '{"skip_prepare_dataset": true}' \
    --pad_to_multiple_of 64 \
    --dataloader_num_workers 8 \
    --dataset_num_proc 32 \
    --eos_token "<|im_end|>" \
    --max_length 8192 \
    --eval_strategy no \
    --eval_steps 50 \
    --save_strategy steps \
    --save_steps 50 \
    --logging_steps 10 \
    --report_to all \
    --run_name $run_name \


echo "END TIME: $(date)"
