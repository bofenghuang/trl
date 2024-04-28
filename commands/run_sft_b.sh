#!/bin/bash
# This script runs an SFT example end-to-end on a tiny model using different possible configurations
# but defaults to QLoRA + PEFT

# export OMP_NUM_THREADS="1"
export TOKENIZERS_PARALLELISM="false"
# export BITSANDBYTES_NOWELCOME="1"
export CUDA_VISIBLE_DEVICES="1,2"

export HF_HOME="/projects/bhuang/.cache/huggingface"

export WANDB_PROJECT="vigogne-v3"
# export WANDB_PROJECT="vigogne-v3"

# export TRL_USE_RICH="true"

MODEL_NAME_OR_PATH="/projects/bhuang/models/llm/pretrained/Meta-Llama-3-8B"

TRAIN_DATA_NAME_OR_PATH="/projects/bhuang/corpus/text/llm/merged/v3_2/train.jsonl"
EVAL_DATA_NAME_OR_PATH="/projects/bhuang/corpus/text/llm/merged/v3_2/test.jsonl"

RUN_NAME="llama_3_8b_sft_qlora_vigogne3_ep3_r64_alphra32_bs64_lr1e4_trl"
OUTPUT_DIR="outputs/sft/$RUN_NAME"

TRAIN_BATCH_SIZE=2
EVAL_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=16

# Handle extra arguments in case one passes accelerate configs.
EXTRA_ACCELERATE_ARGS=""
EXTRA_TRAINING_ARGS=""
# EXTRA_TRAINING_ARGS="""--use_peft \
#     --load_in_4bit
# """

# Set your number of GPUs here
NUM_GPUS=2

# if [[ "${TRL_ACCELERATE_CONFIG}" == "" ]]; then
#   EXTRA_ACCELERATE_ARGS=""
# else
#   EXTRA_ACCELERATE_ARGS="--config_file $TRL_ACCELERATE_CONFIG"
#   # For DeepSpeed configs we need to set the `--fp16` flag to comply with our configs exposed
#   # on `examples/accelerate_configs` and our runners do not support bf16 mixed precision training.
#   if [[ $TRL_ACCELERATE_CONFIG == *"deepspeed"* ]]; then
#     EXTRA_TRAINING_ARGS="--fp16"
#   else
#     echo "Keeping QLoRA + PEFT"
#   fi
# fi


CMD="""
accelerate launch $EXTRA_ACCELERATE_ARGS \
    --main_process_port 29002 \
    --num_processes $NUM_GPUS \
    --mixed_precision "fp16" \
    `pwd`/examples/scripts/sft_b.py \
    --deepspeed /home/bhuang/nlp/axolotl/deepspeed_configs/ds_config_zero2_no_offload.json \
    --train_data_name_or_path $TRAIN_DATA_NAME_OR_PATH \
    --eval_data_name_or_path $EVAL_DATA_NAME_OR_PATH \
    --dataset_num_proc 32 \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --torch_dtype float16 \
    --attn_implementation sdpa \
    --load_in_4bit \
    --use_peft \
    --lora_r 64 \
    --lora_alpha 32 \
    --lora_target_modules "q_proj" "k_proj" "v_proj" "o_proj" "gate_proj" "up_proj" "down_proj" \
    --lora_modules_to_save "embed_tokens" "lm_head" \
    --pad_token '<|reserved_special_token_250|>' \
    --chat_template vigogne_chat_v4 \
    --max_seq_length 8192 \
    --dataloader_num_workers 1 \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 3 \
    --per_device_train_batch_size $TRAIN_BATCH_SIZE \
    --per_device_eval_batch_size $EVAL_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --optim paged_adamw_32bit \
    --learning_rate 1e-4 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --weight_decay 0.0 \
    --fp16 \
    --gradient_checkpointing \
    --gradient_checkpointing_use_reentrant false \
    --eval_strategy steps \
    --eval_steps 500 \
    --save_strategy steps \
    --save_steps 500 \
    --save_total_limit 3 \
    --log_level info \
    --logging_steps 1 \
    --logging_first_step \
    --report_to wandb \
    --run_name $RUN_NAME \
    $EXTRA_TRAINING_ARGS
"""

echo "Starting program..."

{ # try
    echo $CMD
    eval "$CMD"
} || { # catch
    # save log for exception 
    echo "Operation Failed!"
    exit 1
}
exit 0
