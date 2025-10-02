import os

from datasets import load_dataset
from transformers import AutoTokenizer


def main(input_file: str, tokenizer_name_or_path: str, output_file: str | None = None, num_workers: int = 4):
    dataset = load_dataset("json", data_files=input_file, split="train")
    print(dataset)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

    def process_func(example):
        completion = example["messages"].pop(-1)
        # return {
        #     "prompt": example["messages"],
        #     "completion": completion["content"],
        # }

        return {
            "prompt": tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=True),
            "target": completion["content"],
        }

    dataset = dataset.map(
        process_func,
        remove_columns=dataset.column_names,
        num_proc=num_workers,
    )
    # print(dataset)

    if not output_file:
        output_file = input_file.replace(".jsonl", "_grpo.jsonl")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    dataset.to_json(output_file, lines=True)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
