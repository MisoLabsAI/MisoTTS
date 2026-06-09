import os

os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import torchaudio  # type: ignore
from generator import DEFAULT_MISO_TTS_REPO_ID, Segment, load_miso_8b

# Disable Triton compilation
os.environ["NO_TORCH_COMPILE"] = "1"


def main():
    # Select the best available device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    model_source = os.environ.get("MISO_TTS_8B_MODEL", DEFAULT_MISO_TTS_REPO_ID)
    if os.path.exists(model_source):
        print(f"Loading Miso TTS model from local path: {model_source}")
    else:
        print(
            "Loading Miso TTS model from Hugging Face: "
            f"https://huggingface.co/{model_source}"
        )
        print("The model will be downloaded and cached automatically if it is not already present.")

    generator = load_miso_8b(device, model_path_or_repo_id=model_source)

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
            max_audio_length_ms=10_000,
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
        "full_conversation.wav",
        all_audio.unsqueeze(0).cpu(),
        generator.sample_rate,
    )
    print("Successfully generated full_conversation.wav")


if __name__ == "__main__":
    main()
