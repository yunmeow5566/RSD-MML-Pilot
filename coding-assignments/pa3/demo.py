import argparse
from pathlib import Path

import torch

from utils import (
    DEFAULT_CSV,
    DEFAULT_MODEL,
    evaluate,
    load_dataset,
    load_model,
    print_metrics,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a decoder language model on the sarcasm CSV."
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_CSV,
        type=Path,
        help="Path to the dataset CSV (defaults to the CSV beside this script).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Hugging Face model ID or local fine-tuned model directory.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-input-tokens", type=int, default=256)
    parser.add_argument(
        "--limit",
        default=1000,
        type=int,
        help="Evaluate only the first N rows, useful for a quick smoke test.",
    )
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_input_tokens < 1:
        parser.error("--batch-size and --max-input-tokens must be positive.")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive.")

    examples = load_dataset(args.input)
    if args.limit is not None:
        examples = examples[: args.limit]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {args.model} on {device}...")
    tokenizer, model = load_model(args.model, device)
    print(f"Evaluating {len(examples)} examples...")

    metrics = evaluate(
        examples,
        tokenizer,
        model,
        device,
        batch_size=args.batch_size,
        max_input_tokens=args.max_input_tokens,
    )
    print_metrics("Evaluation results:", metrics)


if __name__ == "__main__":
    main()
