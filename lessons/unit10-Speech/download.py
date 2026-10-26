from pathlib import Path

from transformers import AutoModelForSpeechSeq2Seq, WhisperProcessor


def main() -> None:
    model_name = "openai/whisper-tiny"
    model_dir = Path(__file__).resolve().with_name("whisper-tiny")

    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name)
    processor = WhisperProcessor.from_pretrained(model_name)

    model.save_pretrained(model_dir)
    processor.save_pretrained(model_dir)
    print(f"Model and processor saved to {model_dir}")


if __name__ == "__main__":
    main()
