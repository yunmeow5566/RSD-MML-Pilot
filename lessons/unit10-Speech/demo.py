from pathlib import Path

import librosa
from transformers import AutoModelForSpeechSeq2Seq, WhisperProcessor


def main() -> None:
    # These sample mp3 files are just something I found online, feel free to test with other files.
    english_input_file = Path(__file__).resolve().with_name("sample1.mp3")
    spanish_input_file = Path(__file__).resolve().with_name("sample2sp.mp3")
    model_dir = Path(__file__).resolve().with_name("whisper-tiny")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_dir, local_files_only=True)
    processor = WhisperProcessor.from_pretrained(model_dir, local_files_only=True)
    audio, sample_rate = librosa.load(
        english_input_file,
        sr=processor.feature_extractor.sampling_rate,
        mono=True,
    )
    # Convert audio to high-dimentional vectors
    inputs = processor(
        audio=audio,
        sampling_rate=sample_rate,
        return_tensors="pt",
    ).to(model.device)
    outputs = model.generate(**inputs)
    print("Sample 1 transcript:")
    print(processor.decode(outputs[0], skip_special_tokens=True))

    # Sample for audio-text with translation
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="spanish", task="translate")
    # if you want to transcribe to Spanish, change task to "transcribe"
    audio, sample_rate = librosa.load(
        spanish_input_file,
        sr=processor.feature_extractor.sampling_rate,
        mono=True,
    )
    inputs = processor(
        audio=audio,
        sampling_rate=sample_rate,
        return_tensors="pt",
    ).to(model.device)
    outputs = model.generate(**inputs, forced_decoder_ids=forced_decoder_ids)
    print("Sample 2 transcript:")
    print(processor.decode(outputs[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()