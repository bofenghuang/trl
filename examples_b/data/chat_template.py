from transformers import AutoTokenizer


def main(tokenizer_name_or_path: str, chat_template_path: str):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    # print(tokenizer.chat_template)

    with open(chat_template_path, encoding="utf-8") as chat_template_file:
        tokenizer.chat_template = chat_template_file.read()
    # print(tokenizer.chat_template)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you!"},
        {"role": "user", "content": "What is the capital of France?"},
    ]

    chat_template_kwargs = {}
    processed = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **chat_template_kwargs,
    )
    print(processed)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
