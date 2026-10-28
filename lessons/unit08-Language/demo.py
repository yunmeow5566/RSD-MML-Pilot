from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# You can find documentation here, this is a text2text model so you can use it for other tasks like QA and summarization
# https://huggingface.co/google/flan-t5-base
model_name = "google/flan-t5-base"
model_dir = Path(__file__).resolve().parent / "flan-t5-base"


def get_models():
    is_model_available_locally = (
        (model_dir / "config.json").is_file()
        and any(
            (model_dir / filename).is_file()
            for filename in ("model.safetensors", "pytorch_model.bin")
        )
        and any(
            (model_dir / filename).is_file()
            for filename in ("tokenizer.json", "spiece.model")
        )
    )

    if is_model_available_locally:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_dir, local_files_only=True)
    else:
        # download the mode and store it so we can use the same script next time but loading local files
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        model_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(model_dir)
        model.save_pretrained(model_dir)
    return tokenizer, model


def inference(input_text: str, tokenizer, model) -> str:
    inputs = tokenizer(input_text, return_tensors="pt")
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=32,
            num_beams=4,
            do_sample=False,
        )

    result_text = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    return result_text


def main() -> None:

    tokenizer, model = get_models()
    # translation task
    text = "I am a beautiful person."
    prompt = f"Translate English to Russian: {text}"
    translation = inference(prompt, tokenizer, model)
    print(f"English: {text}")
    print(f"Translation: {translation}")

    # QA task
    question = "Which country does Taipei belong to?"
    answer = inference(question, tokenizer, model)
    print(f"Question: {question}")
    print(f"Answer: {answer}")

    # generation task. this is quite bad...
    instruction = "Write a short story with roughly 200 words about a high school student who loves running. Avoid repeating sentences."
    response = inference(instruction, tokenizer, model)
    print(f"Instruction: {instruction}")
    print(f"Response: {response}")


if __name__ == "__main__":
    main()
