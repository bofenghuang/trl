#!/usr/bin/env bash

set -e

echo "START TIME: $(date)"

root_dir="/home/bhuang/llm/trl"

process () {
    input_file=$1
    output_file=${input_file%.*}_prompt.jsonl

    python $root_dir/examples_b/data/sft_2_grpo.py \
        --input_file $input_file \
        --tokenizer_name_or_path $tokenizer_name_or_path \
        --output_file $output_file
}

tokenizer_name_or_path="/projects/bhuang/models/llm/pretrained/Qwen/Qwen3-4B-Instruct-2507"

input_file="/home/bhuang/llm/momo/intent_classification/data/sga/train_conv.jsonl"
process $input_file

input_file="/home/bhuang/llm/momo/intent_classification/data/sga/test_conv.jsonl"
process $input_file

echo "END TIME: $(date)"
