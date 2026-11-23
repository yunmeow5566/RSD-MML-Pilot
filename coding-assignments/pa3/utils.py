import csv
import re
import sys
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from time import perf_counter
from typing import ParamSpec, TypeVar

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

P = ParamSpec("P")
R = TypeVar("R")

# If running on Google Colab, we might try to use other models. Roughly 8-14 B parameters would work.
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_CSV = Path(__file__).resolve().with_name("Cross_Domain_Sarcasm_20k.csv")


def timed(func: Callable[P, R]) -> Callable[P, R]:
    # decorator for doing profiling
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        start = perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            elapsed = perf_counter() - start
            print(f"{func.__name__}: {elapsed:.4f}s", file=sys.stderr)

    return wrapper


@timed
def load_dataset(csv_path: Path = DEFAULT_CSV) -> list[tuple[str, int]]:
    examples = []
    with csv_path.open(encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        required_columns = {"Text", "Is_Sarcastic"}
        if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
            print(f"{csv_path} must contain the columns Text and Is_Sarcastic.")
            return examples

        for row_number, row in enumerate(reader, start=2):
            text = (row["Text"] or "").strip()
            label = (row["Is_Sarcastic"] or "").strip()
            if not text:
                print(f"Row {row_number} has empty Text.")
                continue
            if label not in {"0", "1"}:
                print(f"Row {row_number} has invalid label {label!r}.")
                continue
            examples.append((text, int(label)))

    if not examples:
        print(f"No examples found in {csv_path}.")
    return examples


@timed
def load_model(model_name: str, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("The tokenizer must define a pad token or EOS token.")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model_path = Path(model_name)
    # Load adapter if available.
    adapter_config_path = model_path / "adapter_config.json"
    if model_path.is_dir() and adapter_config_path.is_file():
        adapter_config = PeftConfig.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            adapter_config.base_model_name_or_path
        )
        model = PeftModel.from_pretrained(model, model_name)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    return tokenizer, model


def make_prompt(text: str, tokenizer) -> str:
    # Try to use general LLM to perform a classification task.
    instruction = (
        "Classify whether the text is sarcastic. "
        "Answer with exactly 1 for sarcasm or 0 for not sarcasm."
    )
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [
                {"role": "system", "content": instruction},
                {"role": "user", "content": f"Text: {text}"},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"{instruction}\nText: {text}\nAnswer:"


def parse_prediction(response: str) -> int | None:
    match = re.match(r"\s*([01])(?:\s|[.!]|$)", response)
    return int(match.group(1)) if match else None


def predict_batch(
    texts: list[str],
    tokenizer,
    model,
    device: torch.device,
    max_input_tokens: int = 256,
) -> list[int | None]:
    # Perform batch inference. You might adjust batch size (and hit OOM.)
    prompts = [make_prompt(text, tokenizer) for text in texts]
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_tokens,
    )
    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=4,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    prompt_length = inputs["input_ids"].shape[1]
    responses = tokenizer.batch_decode(
        output_ids[:, prompt_length:],
        skip_special_tokens=True,
    )
    return [parse_prediction(response) for response in responses]


@timed
def calculate_metrics(
    labels: list[int], predictions: list[int | None]
) -> dict[str, float | int]:
    if len(labels) != len(predictions):
        raise ValueError("Labels and predictions must have the same length.")
    if not labels:
        raise ValueError("Cannot calculate metrics for an empty dataset.")

    true_positive = false_positive = false_negative = true_negative = 0
    invalid_not_sarcastic = invalid_sarcastic = 0

    for label, prediction in zip(labels, predictions):
        if prediction is None:
            if label == 1:
                invalid_sarcastic += 1
            else:
                invalid_not_sarcastic += 1
        elif label == 1 and prediction == 1:
            true_positive += 1
        elif label == 0 and prediction == 1:
            false_positive += 1
        elif label == 1:
            false_negative += 1
        else:
            true_negative += 1

    total = len(labels)
    invalid = invalid_not_sarcastic + invalid_sarcastic
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative + invalid_sarcastic
    precision = (
        true_positive / precision_denominator if precision_denominator else 0.0
    )
    recall = true_positive / recall_denominator if recall_denominator else 0.0
    f1_denominator = precision + recall

    return {
        "accuracy": (true_positive + true_negative) / total,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / f1_denominator if f1_denominator else 0.0,
        "coverage": (total - invalid) / total,
        "invalid": invalid,
        "invalid_not_sarcastic": invalid_not_sarcastic,
        "invalid_sarcastic": invalid_sarcastic,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_positive": true_positive,
    }


@timed
def evaluate(
    examples: list[tuple[str, int]],
    tokenizer,
    model,
    device: torch.device,
    batch_size: int = 1,
    max_input_tokens: int = 256,
) -> dict[str, float | int]:
    # Perform inference for all examples, batch by batch.
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    labels = [label for _, label in examples]
    predictions = []
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        predictions.extend(
            predict_batch(
                [text for text, _ in batch],
                tokenizer,
                model,
                device,
                max_input_tokens,
            )
        )
    return calculate_metrics(labels, predictions)


def print_metrics(title: str, metrics: dict[str, float | int]) -> None:
    print(title)
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f} (sarcasm = 1)")
    print(f"  Recall:    {metrics['recall']:.4f} (sarcasm = 1)")
    print(f"  F1 score:  {metrics['f1']:.4f} (sarcasm = 1)")
    print(f"  Coverage:  {metrics['coverage']:.4f}")
    print(f"  Invalid model responses: {metrics['invalid']}")
