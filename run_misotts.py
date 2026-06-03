import argparse
import os

os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")

# Disable Triton compilation
os.environ["NO_TORCH_COMPILE"] = "1"


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the example Miso TTS conversation.")
    parser.add_argument(
        "--model",
        default=os.environ.get("MISO_TTS_8B_MODEL"),
        help="Local checkpoint path or Hugging Face repo id. Defaults to MISO_TTS_8B_MODEL or MisoLabs/MisoTTS.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Inference device, for example auto, cuda, cuda:0, or cpu. Defaults to auto.",
    )
    parser.add_argument(
        "--output",
        default="full_conversation.wav",
        help="Output WAV path. Defaults to full_conversation.wav.",
    )
    parser.add_argument(
        "--max-audio-length-ms",
        type=positive_float,
        default=10_000,
        help="Maximum generated audio length per utterance in milliseconds. Defaults to 10000.",
    )
    parser.add_argument(
        "--temperature",
        type=positive_float,
        default=0.9,
        help="Sampling temperature. Defaults to 0.9.",
    )
    parser.add_argument(
        "--topk",
        type=positive_int,
        default=50,
        help="Top-k sampling value. Defaults to 50.",
    )
    return parser.parse_args()


def resolve_device(device: str) -> str:
    if device != "auto":
        return device

    import torch

    # Select the best available device, skipping MPS due to float64 limitations.
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main():
    args = parse_args()
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    import torch
    import torchaudio  # type: ignore
    from generator import DEFAULT_MISO_TTS_REPO_ID, Segment, load_miso_8b

    model_source = args.model or DEFAULT_MISO_TTS_REPO_ID
    if os.path.exists(model_source):
        print(f"Loading Miso TTS model from local path: {model_source}")
    else:
        print(
            "Loading Miso TTS model from Hugging Face: "
            f"https://huggingface.co/{model_source}"
        )
        print("The model will be downloaded and cached automatically if it is not already present.")

    generator = load_miso_8b(device=device, model_path_or_repo_id=model_source)

    conversation = [
        {"text": "I'm just honestly not that into him, you know?", "speaker_id": 0},
        {"text": "Yeah, I get it.", "speaker_id": 1},
        {
            "text": (
                "And it's just like, I know I said I'd go out with you and stuff, "
                "but it's just like, I can't you know."
            ),
            "speaker_id": 0,
        },
        {"text": "Yeah, honestly that's totally fair.", "speaker_id": 1},
    ]

    generated_segments = []
    for utterance in conversation:
        print(f"Generating: {utterance['text']}")
        audio_tensor = generator.generate(
            text=utterance["text"],
            speaker=utterance["speaker_id"],
            context=generated_segments,
            max_audio_length_ms=args.max_audio_length_ms,
            temperature=args.temperature,
            topk=args.topk,
        )
        generated_segments.append(
            Segment(
                text=utterance["text"],
                speaker=utterance["speaker_id"],
                audio=audio_tensor,
            )
        )

    all_audio = torch.cat([seg.audio for seg in generated_segments], dim=0)
    torchaudio.save(
        args.output,
        all_audio.unsqueeze(0).cpu(),
        generator.sample_rate,
    )
    print(f"Successfully generated {args.output}")


if __name__ == "__main__":
    main()
