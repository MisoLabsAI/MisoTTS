"""Isolated MisoTTS (8B) single-utterance generation CLI.

Runs in its own venv (transformers 4.49). MisoTTS skips MPS (float64 limits),
so it runs on CPU — slow for an 8B model, but functional. tilde's main server
invokes this as a subprocess.

Usage: python tts_cli.py --text "..." --out out.wav [--speaker 0]
"""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("NO_TORCH_COMPILE", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--speaker", type=int, default=0)
    ap.add_argument("--max-ms", type=int, default=20000)
    args = ap.parse_args()

    import torch
    import torchaudio
    from generator import load_miso_8b

    device = "cuda" if torch.cuda.is_available() else "cpu"
    repo = os.environ.get("MISO_TTS_8B_MODEL", "MisoLabs/MisoTTS")
    gen = load_miso_8b(device, model_path_or_repo_id=repo)
    audio = gen.generate(
        text=args.text, speaker=args.speaker, context=[], max_audio_length_ms=args.max_ms
    )
    torchaudio.save(args.out, audio.unsqueeze(0).cpu(), gen.sample_rate)
    print(f"OK {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
