#!/usr/bin/env bash

set -e

echo "START TIME: $(date)"

# export OMP_NUM_THREADS="1"
# export TOKENIZERS_PARALLELISM="false"

export CUDA_VISIBLE_DEVICES=4,5
# export CUDA_VISIBLE_DEVICES=4,5,6,7

# pytorch debug
export CUDA_LAUNCH_BLOCKING=1
# export TORCHDYNAMO_VERBOSE=1
# export TORCHDYNAMO_DISABLE=1

# wandb
export WANDB_PROJECT="2025-09-zaion-intent-audio-sft"

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

# output_dir="./outputs/intent_classification/audio_ft/sga"
output_dir="./outputs/intent_classification/audio_ft/databank"

model_name="/projects/bhuang/models/llm/pretrained/Qwen/Qwen2.5-Omni-3B"

# sga
# train_dataset_file="/home/bhuang/llm/momo/intent_classification/data/sga/sft_audio/train_tts_kyutai_degraded.jsonl"
# eval_dataset_file="/home/bhuang/llm/momo/intent_classification/data/sga/sft_audio/test_tts_kyutai_degraded.jsonl"

# train on databank, eval on sga
train_dataset_file="/home/bhuang/llm/momo/intent_classification/data/databank/sft_audio/train_tts_kyutai_degraded.jsonl"
eval_dataset_file="/home/bhuang/llm/momo/intent_classification/data/sga/sft_audio/test_tts_kyutai_degraded.jsonl"

run_name="sft_qwen2_5_omni_3b_lora_r64_ep20_bs64_lr2e4"

# cmd
# cmd="python"
cmd="accelerate launch"
# cmd="torchrun --nproc_per_node $N_GPUS -m"

# script
script_path="$root_dir/examples_b/audio_ft/sft.py"

    # --chat_template_path $chat_template_path \
    # --assistant_only_loss true \
    # --eval_dataset_file $eval_dataset_file \
    # --eval_strategy steps \
    # --lora_exclude_modules ".*(visual|audio_tower).*" \
    # --lora_exclude_modules ".*(visual|audio_tower(?!.proj)).*" \

$cmd $script_path \
    --model_name_or_path $model_name \
    --dtype bfloat16 \
    --attn_implementation flash_attention_2 \
    --use_peft \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_target_modules all-linear \
    --lora_exclude_modules ".*(visual|audio_tower).*" \
    --train_dataset_file $train_dataset_file \
    --eval_dataset_file $eval_dataset_file \
    --output_dir $output_dir/$run_name \
    --overwrite_output_dir \
    --num_train_epochs 20 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --optim adamw_8bit \
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
    --remove_unused_columns false \
    --dataset_text_field "" \
    --dataset_kwargs '{"skip_prepare_dataset": true}' \
    --pad_to_multiple_of 64 \
    --dataloader_num_workers 8 \
    --dataset_num_proc 32 \
    --max_length 8192 \
    --eval_strategy steps \
    --eval_steps 50 \
    --save_strategy steps \
    --save_steps 50 \
    --save_total_limit 10 \
    --logging_steps 10 \
    --report_to all \
    --run_name $run_name \

# python examples_b/peft/merge_lora.py \
#     --lora_model_name_or_path $output_dir/$run_name

python examples_b/peft/merge_lora_qwen_omni.py \
    --lora_model_name_or_path $output_dir/$run_name


echo "END TIME: $(date)"


# sth to check:
# - assistant_only_loss is working in chat template?
# - prepare dataset, data collator
