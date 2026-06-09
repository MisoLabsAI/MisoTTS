import argparse
import os
import time
import wave

os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")

import torch

from generator import DEFAULT_MISO_TTS_REPO_ID, load_miso_8b

# Disable Triton compilation
os.environ["NO_TORCH_COMPILE"] = "1"


def _pcm16_bytes(audio: torch.Tensor) -> bytes:
    audio = audio.detach().flatten().to(dtype=torch.float32).cpu()
    pcm = audio.clamp(-1.0, 1.0).mul(32767.0).to(torch.int16)
    return pcm.numpy().tobytes()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, default="Hello from streamed Miso TTS.")
    parser.add_argument("--speaker", type=int, default=0)
    parser.add_argument("--output", type=str, default="streamed_generation.wav")
    parser.add_argument("--max-audio-length-ms", type=float, default=10_000)
    parser.add_argument("--chunk-frames", type=int, default=25)
    parser.add_argument(
        "--model-path-or-repo-id",
        type=str,
        default=os.environ.get("MISO_TTS_8B_MODEL", DEFAULT_MISO_TTS_REPO_ID),
    )
    args = parser.parse_args()

    # Select the best available device, skipping MPS due to float64 limitations.
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    if os.path.exists(args.model_path_or_repo_id):
        print(f"Loading Miso TTS model from local path: {args.model_path_or_repo_id}")
    else:
        print(
            "Loading Miso TTS model from Hugging Face: "
            f"https://huggingface.co/{args.model_path_or_repo_id}"
        )
        print("The model will be downloaded and cached automatically if it is not already present.")

    generator = load_miso_8b(device, model_path_or_repo_id=args.model_path_or_repo_id)

    start_time = time.perf_counter()
    first_audio_time = None
    chunk_count = 0
    sample_count = 0

    with wave.open(args.output, "wb") as output_file:
        output_file.setnchannels(1)
        output_file.setsampwidth(2)
        output_file.setframerate(generator.sample_rate)

        for chunk in generator.generate_stream(
            text=args.text,
            speaker=args.speaker,
            context=[],
            max_audio_length_ms=args.max_audio_length_ms,
            chunk_frames=args.chunk_frames,
        ):
            if chunk.numel() == 0:
                continue

            if first_audio_time is None:
                first_audio_time = time.perf_counter()

            chunk_count += 1
            sample_count += chunk.numel()
            output_file.writeframes(_pcm16_bytes(chunk))
            duration_s = chunk.numel() / generator.sample_rate
            print(f"Wrote chunk {chunk_count}: {duration_s:.2f}s")

    total_time = time.perf_counter() - start_time
    audio_duration_s = sample_count / generator.sample_rate
    if first_audio_time is None:
        print("No audio generated.")
    else:
        time_to_first_audio = first_audio_time - start_time
        print(f"Time to first audio: {time_to_first_audio:.2f}s")
        print(f"Total generation time: {total_time:.2f}s")
        print(f"Audio duration: {audio_duration_s:.2f}s")
        print(f"Successfully generated {args.output}")


if __name__ == "__main__":
    main()
