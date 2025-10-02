from pprint import pprint

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer


# dataset_name = "/home/bhuang/llm/momo/intent_classification/data/sga/train.jsonl"


def main(dataset_name: str, tokenizer_name: str, num_workers: int = 8):
    ds = load_dataset("json", data_files=dataset_name, split="train")
    print(ds)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def process_example(example):
        return {"num_tokens": len(tokenizer.apply_chat_template(example["messages"]))}

    ds = ds.map(process_example, num_proc=num_workers)

    sorted_values = np.sort(np.asarray(ds["num_tokens"], dtype=float))
    median_val = float(np.median(sorted_values))
    average = float(np.mean(sorted_values))
    std_value = float(np.std(sorted_values, ddof=1)) if len(sorted_values) > 1 else 0.0
    min_val = float(sorted_values[0])
    max_val = float(sorted_values[-1])
    p95_val = float(np.percentile(sorted_values, 95))
    p99_val = float(np.percentile(sorted_values, 99))

    stats = {
        "median": round(median_val, 4),
        "average": round(average, 4),
        "std": round(std_value, 4),
        "min": round(min_val, 4),
        "max": round(max_val, 4),
        "p95": round(p95_val, 4),
        "p99": round(p99_val, 4),
    }
    pprint(stats)


if __name__ == "__main__":
    import fire

    fire.Fire(main)

