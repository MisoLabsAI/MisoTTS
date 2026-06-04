import argparse

import silentcipher
import torch
import torchaudio

# Warning: When using MisoTTS in another application, you must set this key
# and keep the watermark key secret.
MISO_TTS_WATERMARK = [0, 0, 0, 0, 0]


def cli_check_audio() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_path", type=str, required=True)
    args = parser.parse_args()

    check_audio_from_file(args.audio_path)


def _get_default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_watermarker(device: str | None = None) -> silentcipher.server.Model:
    resolved_device = device or _get_default_device()
    try:
        model = silentcipher.get_model(
            model_type="44.1k",
            device=resolved_device,
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to load the SilentCipher watermarking model. Miso TTS weights may "
            "already be downloaded successfully; this is a separate Hugging Face Hub "
            "download from sony/silentcipher. Retry the command to resume the cache, or "
            "pre-download it with `uv run huggingface-cli download sony/silentcipher`."
        ) from exc
    return model


@torch.inference_mode()
def watermark(
    watermarker: silentcipher.server.Model,
    audio_array: torch.Tensor,
    sample_rate: int,
    watermark_key: list[int],
) -> tuple[torch.Tensor, int]:
    audio_array_44khz = torchaudio.functional.resample(audio_array, orig_freq=sample_rate, new_freq=44100)
    encoded, _ = watermarker.encode_wav(audio_array_44khz, 44100, watermark_key, calc_sdr=False, message_sdr=36)

    output_sample_rate = min(44100, sample_rate)
    encoded = torchaudio.functional.resample(encoded, orig_freq=44100, new_freq=output_sample_rate)
    return encoded, output_sample_rate


@torch.inference_mode()
def verify(
    watermarker: silentcipher.server.Model,
    watermarked_audio: torch.Tensor,
    sample_rate: int,
    watermark_key: list[int],
) -> bool:
    watermarked_audio_44khz = torchaudio.functional.resample(watermarked_audio, orig_freq=sample_rate, new_freq=44100)
    result = watermarker.decode_wav(watermarked_audio_44khz, 44100, phase_shift_decoding=True)

    is_watermarked = result["status"]
    if is_watermarked:
        is_miso_tts_watermarked = result["messages"][0] == watermark_key
    else:
        is_miso_tts_watermarked = False

    return is_watermarked and is_miso_tts_watermarked


def check_audio_from_file(audio_path: str) -> None:
    watermarker = load_watermarker()

    audio_array, sample_rate = load_audio(audio_path)
    is_watermarked = verify(watermarker, audio_array, sample_rate, MISO_TTS_WATERMARK)

    outcome = "Watermarked" if is_watermarked else "Not watermarked"
    print(f"{outcome}: {audio_path}")


def load_audio(audio_path: str) -> tuple[torch.Tensor, int]:
    audio_array, sample_rate = torchaudio.load(audio_path)
    audio_array = audio_array.mean(dim=0)
    return audio_array, int(sample_rate)


if __name__ == "__main__":
    cli_check_audio()
