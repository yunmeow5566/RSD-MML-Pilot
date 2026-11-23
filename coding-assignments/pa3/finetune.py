import argparse
import os
import random
from pathlib import Path
from time import perf_counter

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader

from utils import (
    DEFAULT_MODEL,
    DEFAULT_CSV,
    evaluate,
    load_dataset,
    load_model,
    make_prompt,
    print_metrics,
)


def split_dataset(
    examples: list[tuple[str, int]], seed: int
) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    # Split input examples into testing and training set.
    # We keep 50 positive and 50 negative examples for testing.
    by_label = {0: [], 1: []}
    for example in examples:
        by_label[example[1]].append(example)
    if min(len(by_label[0]), len(by_label[1])) < 50:
        raise ValueError("The dataset must contain at least 50 examples per label.")

    rng = random.Random(seed)
    test_examples = []
    train_examples = []
    for label in (0, 1):
        rng.shuffle(by_label[label])
        test_examples.extend(by_label[label][:50])
        train_examples.extend(by_label[label][50:])
    rng.shuffle(test_examples)
    rng.shuffle(train_examples)
    return train_examples, test_examples


def select_training_examples(
    examples: list[tuple[str, int]], max_examples: int, seed: int
) -> list[tuple[str, int]]:
    # Sub-sampling training examples. We try to keep training examples balanced.
    if max_examples == 0:
        return examples
    if max_examples % 2:
        raise ValueError("--train-samples must be 0 or an even number.")

    by_label = {0: [], 1: []}
    for example in examples:
        by_label[example[1]].append(example)
    per_label = max_examples // 2
    if min(len(by_label[0]), len(by_label[1])) < per_label:
        raise ValueError(
            f"Not enough examples per label for --train-samples={max_examples}."
        )

    rng = random.Random(seed)
    selected = []
    for label in (0, 1):
        selected.extend(rng.sample(by_label[label], per_label))
    rng.shuffle(selected)
    return selected


def encode_training_example(
    text: str, label: int, tokenizer, max_sequence_length: int
) -> dict[str, list[int]]:
    # Perform encoding from prompt/input and answer to tokens
    # this is a tandard causual-LM training format for decoder-only model.
    prompt_ids = tokenizer(
        make_prompt(text, tokenizer),
        add_special_tokens=False,
    )["input_ids"]
    answer_ids = tokenizer(
        str(label),
        add_special_tokens=False,
    )["input_ids"]
    if tokenizer.eos_token_id is not None:
        answer_ids.append(tokenizer.eos_token_id)
    if not answer_ids or len(answer_ids) >= max_sequence_length:
        raise ValueError("The answer tokens do not fit in the requested sequence length.")

    prompt_ids = prompt_ids[-(max_sequence_length - len(answer_ids)) :]
    input_ids = prompt_ids + answer_ids
    return {
        "input_ids": input_ids,
         # mask prompt tokens; tell models to focus on loss of answers
        "labels": [-100] * len(prompt_ids) + answer_ids, 
    }


def make_collate_fn(tokenizer):
    # Define to be compatible with DataLoader
    def collate(examples: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_length = max(len(example["input_ids"]) for example in examples)
        input_ids = []
        attention_masks = []
        labels = []
        for example in examples:
            padding_length = max_length - len(example["input_ids"])
            input_ids.append(
                example["input_ids"] + [tokenizer.pad_token_id] * padding_length
            )
            attention_masks.append([1] * len(example["input_ids"]) + [0] * padding_length)
            labels.append(example["labels"] + [-100] * padding_length)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    return collate


def train_model(
    model,
    tokenizer,
    examples: list[tuple[str, int]],
    device: torch.device,
    max_sequence_length: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    epochs: int,
    learning_rate: float,
    seed: int,
) -> None:
    training_features = [
        encode_training_example(text, label, tokenizer, max_sequence_length)
        for text, label in examples
    ]
    loader = DataLoader(
        training_features,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        collate_fn=make_collate_fn(tokenizer),
    )
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=learning_rate)

    model.config.use_cache = False
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0.0
        batch_count = len(loader)
        timing_checkpoints = sorted(
            {(batch_count * part + 4) // 5 for part in range(1, 6)}
        )
        next_checkpoint = 0
        interval_start = perf_counter()
        for batch_index, batch in enumerate(loader):
            batch = {name: tensor.to(device) for name, tensor in batch.items()}
            accumulation_start = (
                batch_index // gradient_accumulation_steps
            ) * gradient_accumulation_steps
            accumulation_size = min(
                gradient_accumulation_steps,
                len(loader) - accumulation_start,
            )
            loss = model(**batch).loss
            (loss / accumulation_size).backward()
            epoch_loss += loss.detach().item()

            if (
                (batch_index + 1) % gradient_accumulation_steps == 0
                or batch_index + 1 == len(loader)
            ):
                torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            completed_batches = batch_index + 1
            if (
                next_checkpoint < len(timing_checkpoints)
                and completed_batches >= timing_checkpoints[next_checkpoint]
            ):
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                interval_elapsed = perf_counter() - interval_start
                progress = completed_batches / batch_count
                print(
                    f"Epoch {epoch + 1}/{epochs}, "
                    f"progress {progress:.0%} "
                    f"({completed_batches}/{batch_count} batches): "
                    f"{interval_elapsed:.2f}s"
                )
                interval_start = perf_counter()
                next_checkpoint += 1

        print(f"Epoch {epoch + 1}/{epochs}: loss={epoch_loss / len(loader):.4f}")
    model.config.use_cache = True
    model.eval()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune a causal language model with LoRA on sarcasm labels."
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Hugging Face model ID or local model directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().with_name("qwen-sarcasm-lora"),
    )
    parser.add_argument(
        "--train-samples",
        type=int,
        default=0,
        help="Balanced training examples; use 0 for all non-test examples.",
    )
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-sequence-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=max(1, min(4, os.cpu_count() or 1)),
    )
    args = parser.parse_args()

    if args.train_samples < 0:
        parser.error("--train-samples must be 0 or greater.")
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive.")
    if args.gradient_accumulation_steps < 1:
        parser.error("--gradient-accumulation-steps must be positive.")
    if args.max_sequence_length < 2:
        parser.error("--max-sequence-length must be at least 2.")
    if args.learning_rate <= 0 or args.cpu_threads < 1:
        parser.error("--learning-rate and --cpu-threads must be positive.")
    if args.train_samples and args.train_samples % 2:
        parser.error("--train-samples must be 0 or an even number.")

    torch.set_num_threads(args.cpu_threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}; LoRA avoids updating the full base model.")

    all_examples = load_dataset(args.csv)
    train_pool, test_examples = split_dataset(all_examples, args.seed)
    train_examples = select_training_examples(
        train_pool,
        args.train_samples,
        args.seed + 1,
    )
    print(
        f"Training examples: {len(train_examples)}; "
        "test examples: 100 (50 per label)."
    )

    tokenizer, model = load_model(args.model, device)
    print("Measuring accuracy before fine-tuning...")
    before_metrics = evaluate(
        test_examples,
        tokenizer,
        model,
        device,
        batch_size=1,
        max_input_tokens=args.max_sequence_length,
    )
    print_metrics("Before fine-tuning:", before_metrics)
    # Modify model to allow finetuning
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
        ),
    )
    model.print_trainable_parameters()
    train_model(
        model,
        tokenizer,
        train_examples,
        device,
        args.max_sequence_length,
        args.batch_size,
        args.gradient_accumulation_steps,
        args.epochs,
        args.learning_rate,
        args.seed,
    )

    print("Measuring accuracy after fine-tuning...")
    after_metrics = evaluate(
        test_examples,
        tokenizer,
        model,
        device,
        batch_size=1,
        max_input_tokens=args.max_sequence_length,
    )
    print_metrics("After fine-tuning:", after_metrics)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved LoRA adapter and tokenizer to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
