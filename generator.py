from dataclasses import dataclass
import os
from typing import List, Optional, Tuple

os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")

import torch
import torchaudio
from huggingface_hub import hf_hub_download
from models import MISO_TTS_8B_CONFIG, Model, ModelArgs, apply_mps_bf16_layer0_mlp_fp32_guard
from moshi_compat import patch_bitsandbytes_import_for_unquantized_layers
from moshi.models import loaders
from tokenizers.processors import TemplateProcessing
from transformers import AutoTokenizer
from watermarking import MISO_TTS_WATERMARK, load_watermarker, watermark

DEFAULT_MISO_TTS_REPO_ID = "MisoLabs/MisoTTS"
DISABLE_MPS_BF16_LAYER0_MLP_FP32_ENV = "MISO_DISABLE_MPS_BF16_LAYER0_MLP_FP32"
patch_bitsandbytes_import_for_unquantized_layers()


@dataclass
class Segment:
    speaker: int
    text: str
    # (num_samples,), sample_rate = 24_000
    audio: torch.Tensor


def load_llama3_tokenizer():
    """
    https://github.com/huggingface/transformers/issues/22794#issuecomment-2092623992
    """
    tokenizer_name = "meta-llama/Llama-3.2-1B"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    bos = tokenizer.bos_token
    eos = tokenizer.eos_token
    tokenizer._tokenizer.post_processor = TemplateProcessing(
        single=f"{bos}:0 $A:0 {eos}:0",
        pair=f"{bos}:0 $A:0 {eos}:0 {bos}:1 $B:1 {eos}:1",
        special_tokens=[(f"{bos}", tokenizer.bos_token_id), (f"{eos}", tokenizer.eos_token_id)],
    )

    return tokenizer


def _should_apply_mps_bf16_layer0_mlp_fp32_guard(
    device: str,
    dtype: torch.dtype,
    enabled: Optional[bool],
) -> bool:
    if enabled is not None:
        return enabled

    disabled = os.environ.get(DISABLE_MPS_BF16_LAYER0_MLP_FP32_ENV, "").lower() in {"1", "true", "yes"}
    return torch.device(device).type == "mps" and dtype == torch.bfloat16 and not disabled


class Generator:
    def __init__(
        self,
        model: Model,
    ):
        self._model = model
        self._model.setup_caches(1)

        self._text_tokenizer = load_llama3_tokenizer()
        self._frame_size = self._model.config.audio_num_codebooks + 1

        device = next(model.parameters()).device
        mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
        mimi = loaders.get_mimi(mimi_weight, device=device)
        mimi.set_num_codebooks(self._model.config.audio_num_codebooks)
        self._audio_tokenizer = mimi

        self._watermarker = load_watermarker(device=device)

        self.sample_rate = mimi.sample_rate
        self.device = device

    def _tokenize_text_segment(self, text: str, speaker: int) -> Tuple[torch.Tensor, torch.Tensor]:
        frame_tokens = []
        frame_masks = []

        text_tokens = self._text_tokenizer.encode(f"[{speaker}] {text.lstrip()}")
        text_frame = torch.zeros(len(text_tokens), self._frame_size).long()
        text_frame_mask = torch.zeros(len(text_tokens), self._frame_size).bool()
        text_frame[:, -1] = torch.tensor(text_tokens)
        text_frame_mask[:, -1] = True

        frame_tokens.append(text_frame.to(self.device))
        frame_masks.append(text_frame_mask.to(self.device))

        return torch.cat(frame_tokens, dim=0), torch.cat(frame_masks, dim=0)

    def _tokenize_audio(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert audio.ndim == 1, "Audio must be single channel"

        frame_tokens = []
        frame_masks = []

        # (K, T)
        audio = audio.to(self.device)
        audio_tokens = self._audio_tokenizer.encode(audio.unsqueeze(0).unsqueeze(0))[0]
        # add EOS frame
        eos_frame = torch.zeros(audio_tokens.size(0), 1).to(self.device)
        audio_tokens = torch.cat([audio_tokens, eos_frame], dim=1)

        audio_frame = torch.zeros(audio_tokens.size(1), self._frame_size).long().to(self.device)
        audio_frame_mask = torch.zeros(audio_tokens.size(1), self._frame_size).bool().to(self.device)
        audio_frame[:, :-1] = audio_tokens.transpose(0, 1)
        audio_frame_mask[:, :-1] = True

        frame_tokens.append(audio_frame)
        frame_masks.append(audio_frame_mask)

        return torch.cat(frame_tokens, dim=0), torch.cat(frame_masks, dim=0)

    def _tokenize_segment(self, segment: Segment) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            (seq_len, audio_num_codebooks + 1), (seq_len, audio_num_codebooks + 1)
        """
        text_tokens, text_masks = self._tokenize_text_segment(segment.text, segment.speaker)
        audio_tokens, audio_masks = self._tokenize_audio(segment.audio)

        return torch.cat([text_tokens, audio_tokens], dim=0), torch.cat([text_masks, audio_masks], dim=0)

    @torch.inference_mode()
    def generate(
        self,
        text: str,
        speaker: int,
        context: List[Segment],
        max_audio_length_ms: float = 90_000,
        temperature: float = 0.9,
        topk: int = 50,
    ) -> torch.Tensor:
        self._model.reset_caches()

        max_generation_len = int(max_audio_length_ms / 80)
        tokens, tokens_mask = [], []
        for segment in context:
            segment_tokens, segment_tokens_mask = self._tokenize_segment(segment)
            tokens.append(segment_tokens)
            tokens_mask.append(segment_tokens_mask)

        gen_segment_tokens, gen_segment_tokens_mask = self._tokenize_text_segment(text, speaker)
        tokens.append(gen_segment_tokens)
        tokens_mask.append(gen_segment_tokens_mask)

        prompt_tokens = torch.cat(tokens, dim=0).long().to(self.device)
        prompt_tokens_mask = torch.cat(tokens_mask, dim=0).bool().to(self.device)

        samples = []
        curr_tokens = prompt_tokens.unsqueeze(0)
        curr_tokens_mask = prompt_tokens_mask.unsqueeze(0)
        curr_pos = torch.arange(0, prompt_tokens.size(0)).unsqueeze(0).long().to(self.device)

        max_seq_len = 2048
        max_context_len = max_seq_len - max_generation_len
        if curr_tokens.size(1) >= max_context_len:
            raise ValueError(
                f"Inputs too long, must be below max_seq_len - max_generation_len: {max_context_len}"
            )

        for _ in range(max_generation_len):
            sample = self._model.generate_frame(curr_tokens, curr_tokens_mask, curr_pos, temperature, topk)
            if torch.all(sample == 0):
                break  # eos

            samples.append(sample)

            curr_tokens = torch.cat([sample, torch.zeros(1, 1).long().to(self.device)], dim=1).unsqueeze(1)
            curr_tokens_mask = torch.cat(
                [torch.ones_like(sample).bool(), torch.zeros(1, 1).bool().to(self.device)], dim=1
            ).unsqueeze(1)
            curr_pos = curr_pos[:, -1:] + 1

        audio = self._audio_tokenizer.decode(torch.stack(samples).permute(1, 2, 0)).squeeze(0).squeeze(0)

        # This applies an imperceptible watermark to identify audio as AI-generated.
        # If using Miso TTS in another application, use your own private key and keep it secret.
        audio, wm_sample_rate = watermark(self._watermarker, audio, self.sample_rate, MISO_TTS_WATERMARK)
        audio = torchaudio.functional.resample(audio, orig_freq=wm_sample_rate, new_freq=self.sample_rate)

        return audio


def _state_dict_from_checkpoint(checkpoint: object) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected checkpoint dict, got {type(checkpoint).__name__}")

    for key in ("state_dict", "model_state_dict", "model"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            checkpoint = value
            break

    state_dict = {}
    for key, value in checkpoint.items():
        if torch.is_tensor(value):
            state_dict[key.removeprefix("module.")] = value
    if not state_dict:
        raise ValueError("Checkpoint did not contain any tensor state_dict entries")
    return state_dict


def _load_model(
    model_path_or_repo_id: str,
    config: ModelArgs,
    device: str,
    dtype: torch.dtype,
    mps_bf16_layer0_mlp_fp32: Optional[bool] = None,
) -> Model:
    if os.path.isfile(model_path_or_repo_id):
        model_file = model_path_or_repo_id
    elif os.path.isdir(model_path_or_repo_id):
        model_file = os.path.join(model_path_or_repo_id, "model.safetensors")
    else:
        model_file = hf_hub_download(repo_id=model_path_or_repo_id, filename="model.safetensors")

    if os.path.isfile(model_file):
        model = Model(config)
        if model_file.endswith(".safetensors"):
            try:
                from safetensors.torch import load_file
            except ImportError as exc:
                raise ImportError("Install safetensors to load .safetensors checkpoint files") from exc

            state_dict = load_file(model_file, device="cpu")
        else:
            checkpoint = torch.load(model_file, map_location="cpu")
            state_dict = _state_dict_from_checkpoint(checkpoint)
        model.load_state_dict(state_dict)
    else:
        raise FileNotFoundError(f"Could not resolve model checkpoint: {model_path_or_repo_id}")

    model.to(device=device, dtype=dtype)
    if _should_apply_mps_bf16_layer0_mlp_fp32_guard(device, dtype, mps_bf16_layer0_mlp_fp32):
        apply_mps_bf16_layer0_mlp_fp32_guard(model)
    model.eval()
    return model


def load_miso_8b(
    device: str = "cuda",
    model_path_or_repo_id: Optional[str] = None,
    dtype: torch.dtype = torch.bfloat16,
    mps_bf16_layer0_mlp_fp32: Optional[bool] = None,
) -> Generator:
    source = model_path_or_repo_id or os.environ.get("MISO_TTS_8B_MODEL", DEFAULT_MISO_TTS_REPO_ID)
    model = _load_model(
        source,
        MISO_TTS_8B_CONFIG,
        device=device,
        dtype=dtype,
        mps_bf16_layer0_mlp_fp32=mps_bf16_layer0_mlp_fp32,
    )
    return Generator(model)
