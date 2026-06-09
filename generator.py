from dataclasses import dataclass
import os
from typing import Iterator, List, Optional, Tuple

os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")

import torch
import torchaudio
from huggingface_hub import hf_hub_download
from models import MISO_TTS_8B_CONFIG, Model, ModelArgs
from moshi_compat import patch_bitsandbytes_import_for_unquantized_layers
from moshi.models import loaders
from tokenizers.processors import TemplateProcessing
from transformers import AutoTokenizer
from watermarking import MISO_TTS_WATERMARK, load_watermarker, watermark

DEFAULT_MISO_TTS_REPO_ID = "MisoLabs/MisoTTS"
patch_bitsandbytes_import_for_unquantized_layers()


@dataclass
class Segment:
    speaker: int
    text: str
    # (num_samples,), sample_rate = 24_000
    audio: torch.Tensor


def _stack_audio_frames(samples: List[torch.Tensor]) -> torch.Tensor:
    return torch.stack(samples).permute(1, 2, 0)


def _match_num_samples(audio: torch.Tensor, num_samples: int) -> torch.Tensor:
    if audio.size(0) > num_samples:
        return audio[:num_samples]
    if audio.size(0) < num_samples:
        padding = torch.zeros(num_samples - audio.size(0), dtype=audio.dtype, device=audio.device)
        return torch.cat([audio, padding], dim=0)
    return audio


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

    def _prepare_prompt(
        self,
        text: str,
        speaker: int,
        context: List[Segment],
        max_audio_length_ms: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
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

        max_seq_len = 2048
        max_context_len = max_seq_len - max_generation_len
        if prompt_tokens.size(0) >= max_context_len:
            raise ValueError(
                f"Inputs too long, must be below max_seq_len - max_generation_len: {max_context_len}"
            )

        return prompt_tokens, prompt_tokens_mask, max_generation_len

    def _generate_frames(
        self,
        prompt_tokens: torch.Tensor,
        prompt_tokens_mask: torch.Tensor,
        max_generation_len: int,
        temperature: float,
        topk: int,
    ) -> Iterator[torch.Tensor]:
        curr_tokens = prompt_tokens.unsqueeze(0)
        curr_tokens_mask = prompt_tokens_mask.unsqueeze(0)
        curr_pos = torch.arange(0, prompt_tokens.size(0)).unsqueeze(0).long().to(self.device)

        for _ in range(max_generation_len):
            sample = self._model.generate_frame(curr_tokens, curr_tokens_mask, curr_pos, temperature, topk)
            if torch.all(sample == 0):
                break  # eos

            curr_tokens = torch.cat([sample, torch.zeros(1, 1).long().to(self.device)], dim=1).unsqueeze(1)
            curr_tokens_mask = torch.cat(
                [torch.ones_like(sample).bool(), torch.zeros(1, 1).bool().to(self.device)], dim=1
            ).unsqueeze(1)
            curr_pos = curr_pos[:, -1:] + 1

            yield sample

    def _decode_frames(self, samples: List[torch.Tensor]) -> torch.Tensor:
        return self._audio_tokenizer.decode(_stack_audio_frames(samples)).squeeze(0).squeeze(0)

    def _watermark_audio(self, audio: torch.Tensor) -> torch.Tensor:
        # This applies an imperceptible watermark to identify audio as AI-generated.
        # If using Miso TTS in another application, use your own private key and keep it secret.
        audio, wm_sample_rate = watermark(self._watermarker, audio, self.sample_rate, MISO_TTS_WATERMARK)
        audio = torchaudio.functional.resample(audio, orig_freq=wm_sample_rate, new_freq=self.sample_rate)
        return audio

    def _watermark_stream_chunk(self, audio: torch.Tensor, *, is_final: bool) -> torch.Tensor:
        target_num_samples = audio.size(0)
        try:
            audio = self._watermark_audio(audio)
        except Exception:
            if not is_final:
                raise
            # SilentCipher may reject very short final chunks. Earlier chunks are
            # watermarked independently; only the terminal fragment falls back.
            return audio
        return _match_num_samples(audio, target_num_samples)

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
        prompt_tokens, prompt_tokens_mask, max_generation_len = self._prepare_prompt(
            text, speaker, context, max_audio_length_ms
        )
        samples = list(
            self._generate_frames(prompt_tokens, prompt_tokens_mask, max_generation_len, temperature, topk)
        )

        audio = self._decode_frames(samples)
        audio = self._watermark_audio(audio)

        return audio

    def generate_stream(
        self,
        text: str,
        speaker: int,
        context: List[Segment],
        max_audio_length_ms: float = 90_000,
        temperature: float = 0.9,
        topk: int = 50,
        chunk_frames: int = 25,
    ) -> Iterator[torch.Tensor]:
        if chunk_frames <= 0:
            raise ValueError("chunk_frames must be greater than 0")

        # Tokenize the prompt (including Mimi encode of any context audio) before
        # entering Mimi streaming mode, so streamed generation conditions on the
        # same context tokens as generate().
        with torch.inference_mode():
            prompt_tokens, prompt_tokens_mask, max_generation_len = self._prepare_prompt(
                text, speaker, context, max_audio_length_ms
            )

        frames = self._generate_frames(prompt_tokens, prompt_tokens_mask, max_generation_len, temperature, topk)

        chunk: List[torch.Tensor] = []
        all_samples: List[torch.Tensor] = []
        streamed_num_samples = 0

        # Each chunk is computed fully inside torch.inference_mode() and yielded
        # outside it. A plain `with torch.inference_mode():` around the whole
        # generator body would stay active while the generator is suspended at
        # `yield`, silently putting the caller's loop body into inference mode.
        #
        # Only full chunk_frames-sized chunks are decoded inside Mimi streaming
        # mode: on CUDA the streaming decoder is wrapped in a CUDA graph captured
        # at the first chunk's shape, so a shorter residual chunk would raise a
        # shape mismatch. The residual is covered by the batch tail decode below.
        with self._audio_tokenizer.streaming(1):
            finished = False
            while not finished:
                out: Optional[torch.Tensor] = None
                with torch.inference_mode():
                    while len(chunk) < chunk_frames:
                        sample = next(frames, None)
                        if sample is None:
                            finished = True
                            break
                        chunk.append(sample)
                        all_samples.append(sample)

                    if len(chunk) == chunk_frames:
                        audio = self._decode_frames(chunk)
                        streamed_num_samples += audio.size(0)
                        chunk = []
                        if audio.numel() > 0:
                            out = self._watermark_stream_chunk(audio, is_final=False)
                if out is not None:
                    yield out

        tail_audio: Optional[torch.Tensor] = None
        with torch.inference_mode():
            if all_samples:
                # Mimi streaming decode does not expose an explicit flush. Decode
                # the full code sequence once (in batch mode, outside the streaming
                # context) and emit only the deferred tail - the residual frames
                # plus any samples the streamed chunks have not covered - so
                # concatenated stream chunks keep the batch decode length.
                full_audio = self._decode_frames(all_samples)
                if streamed_num_samples < full_audio.size(0):
                    tail = full_audio[streamed_num_samples:]
                    if tail.numel() > 0:
                        tail_audio = self._watermark_stream_chunk(tail, is_final=True)

        if tail_audio is not None:
            yield tail_audio


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
    model.eval()
    return model


def load_miso_8b(
    device: str = "cuda",
    model_path_or_repo_id: Optional[str] = None,
    dtype: torch.dtype = torch.bfloat16,
) -> Generator:
    source = model_path_or_repo_id or os.environ.get("MISO_TTS_8B_MODEL", DEFAULT_MISO_TTS_REPO_ID)
    model = _load_model(source, MISO_TTS_8B_CONFIG, device=device, dtype=dtype)
    return Generator(model)
