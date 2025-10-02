#!/usr/bin/env bash

set -e

echo "START TIME: $(date)"

# export OMP_NUM_THREADS="1"
# export TOKENIZERS_PARALLELISM="false"

# export CUDA_VISIBLE_DEVICES=2,3
export CUDA_VISIBLE_DEVICES=4,5,6,7

# pytorch debug
export CUDA_LAUNCH_BLOCKING=1
# export TORCHDYNAMO_VERBOSE=1
# export TORCHDYNAMO_DISABLE=1

# wandb
export WANDB_PROJECT="2025-09-zaion-summary-audio-dpo"

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

output_dir="./outputs/summary/audio_ft/edenred"

# model_name="/projects/bhuang/models/llm/pretrained/mistralai/Voxtral-Mini-3B-2507"
model_name="/home/bhuang/llm/trl/outputs/summary/audio_ft/edenred/sft_voxtral_mini_3b_2507_lora_r64_ep3_bs128_lr1e4_merged"

train_dataset_file="/projects/bhuang/corpus/text/summary/edenred/generated_summaries/dpo/qwen3_235b_a22b_instruct_2507_fp8_vs_qwen3_omni_30b_a3b_thinking/audio_voxtral/train.jsonl"
eval_dataset_file="/projects/bhuang/corpus/text/summary/edenred/generated_summaries/dpo/qwen3_235b_a22b_instruct_2507_fp8_vs_qwen3_omni_30b_a3b_thinking/audio_voxtral/test.jsonl"

run_name="dpo_voxtral_mini_3b_2507_lora_r64_ep1_bs128_lr5e6_beta001"

# cmd
# cmd="python"
cmd="accelerate launch"
# cmd="accelerate launch --config_file $root_dir/examples_b/configs/deepspeed_zero2.yaml"
# cmd="accelerate launch --config_file $root_dir/examples_b/configs/fsdp2.yaml"
# cmd="torchrun --nproc_per_node $N_GPUS -m"

# script
script_path="$root_dir/examples_b/audio_ft/dpo_voxtral.py"

    # --chat_template_path $chat_template_path \
    # --assistant_only_loss true \
    # --max_length 8192 \  # not used by customized collator
    # --optim adamw_torch_fused \
    # --optim adamw_8bit \
    # --deepspeed $root_dir/examples_b/configs/ds_config_zero2_no_offload.json \
    # use fsdp activation checkpointing
    # --gradient_checkpointing \
    # --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    # --lora_exclude_modules ".*(audio_tower|multi_modal_projector).*" \
    # --lora_exclude_modules ".*(audio_tower).*" \
    # --load_in_4bit \

$cmd $script_path \
    --model_name_or_path $model_name \
    --dtype bfloat16 \
    --attn_implementation flash_attention_2 \
    --use_peft \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_target_modules all-linear \
    --lora_exclude_modules ".*(audio_tower|multi_modal_projector).*" \
    --train_dataset_file $train_dataset_file \
    --eval_dataset_file $eval_dataset_file \
    --output_dir $output_dir/$run_name \
    --overwrite_output_dir \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --optim adamw_torch_fused \
    --learning_rate 5e-6 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --beta 0.01 \
    --bf16 true \
    --gradient_checkpointing \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --ddp_find_unused_parameters false \
    --remove_unused_columns false \
    --dataset_text_field "" \
    --dataset_kwargs '{"skip_prepare_dataset": true}' \
    --padding_value 11 \
    --pad_to_multiple_of 64 \
    --dataloader_num_workers 8 \
    --dataset_num_proc 32 \
    --eval_strategy steps \
    --eval_steps 50 \
    --save_strategy steps \
    --save_steps 50 \
    --save_total_limit 10 \
    --logging_steps 10 \
    --report_to all \
    --run_name $run_name \

# merge lora
python examples_b/peft/merge_lora.py \
    --lora_model_name_or_path $output_dir/$run_name


echo "END TIME: $(date)"


# sth to check:
# - have to patch VoxtralForConditionalGeneration to support quantization
#           inputs_embeds = inputs_embeds.masked_scatter(
#               audio_token_mask.to(inputs_embeds.device), audio_embeds.to(inputs_embeds.device, dtype=inputs_embeds.dtype)
#           )
