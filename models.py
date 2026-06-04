from dataclasses import dataclass
import contextlib
import io
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin

with contextlib.redirect_stdout(io.StringIO()) as _torchtune_stdout:
    from torchtune.models import llama3_2

_torchtune_import_output = _torchtune_stdout.getvalue()
if _torchtune_import_output.strip() != "import error: No module named 'triton'":
    print(_torchtune_import_output, end="")


def llama3_2_8B():
    return llama3_2.llama3_2(
        vocab_size=128_256,
        num_layers=32,
        num_heads=32,
        num_kv_heads=8,
        embed_dim=4096,
        max_seq_len=2048,
        intermediate_dim=14_336,
        attn_dropout=0.1,
        norm_eps=1e-5,
        rope_base=500_000,
        scale_factor=32,
    )


def llama3_2_300M():
    return llama3_2.llama3_2(
        vocab_size=128_256,
        num_layers=8,
        num_heads=24,
        num_kv_heads=6,
        embed_dim=1536,
        max_seq_len=2048,
        intermediate_dim=6912,
        attn_dropout=0.1,
        norm_eps=1e-5,
        rope_base=500_000,
        scale_factor=32,
    )


FLAVORS = {
    "llama-8B": llama3_2_8B,
    "llama-300M": llama3_2_300M,
}


def _prepare_transformer(model):
    embed_dim = model.tok_embeddings.embedding_dim
    model.tok_embeddings = nn.Identity()
    model.output = nn.Identity()
    return model, embed_dim


def _create_causal_mask(seq_len: int, device: torch.device):
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))


def _index_causal_mask(mask: torch.Tensor, input_pos: torch.Tensor):
    """
    Args:
        mask: (max_seq_len, max_seq_len)
        input_pos: (batch_size, seq_len)

    Returns:
        (batch_size, seq_len, max_seq_len)
    """
    r = mask[input_pos, :]
    return r


def _multinomial_sample_one_no_sync(probs):  # Does multinomial sampling without a device synchronization
    q = torch.empty_like(probs).exponential_(1)
    return torch.argmax(probs / q, dim=-1, keepdim=True).to(dtype=torch.int)


def sample_topk(logits: torch.Tensor, topk: int, temperature: float):
    logits = logits / temperature

    filter_value: float = -float("Inf")
    indices_to_remove = logits < torch.topk(logits, topk)[0][..., -1, None]
    scores_processed = logits.masked_fill(indices_to_remove, filter_value)
    scores_processed = torch.nn.functional.log_softmax(scores_processed, dim=-1)
    probs = torch.nn.functional.softmax(scores_processed, dim=-1)

    sample_token = _multinomial_sample_one_no_sync(probs)
    return sample_token


def _masked_cross_entropy(logits, targets, mask, vocab_size):
    losses = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1), reduction="none")
    weights = mask.reshape(-1).to(losses.dtype)
    total = weights.sum().clamp_min(1.0)
    return (losses * weights).sum() / total, total


@dataclass
class ModelArgs:
    backbone_flavor: str
    decoder_flavor: str
    text_vocab_size: int
    audio_vocab_size: int
    audio_num_codebooks: int


MISO_TTS_8B_CONFIG = ModelArgs(
    backbone_flavor="llama-8B",
    decoder_flavor="llama-300M",
    text_vocab_size=128_256,
    audio_vocab_size=2051,
    audio_num_codebooks=32,
)


class Model(
    nn.Module,
    PyTorchModelHubMixin,
    pipeline_tag="text-to-speech",
    license="other",
):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config

        self.backbone, backbone_dim = _prepare_transformer(FLAVORS[config.backbone_flavor]())
        self.decoder, decoder_dim = _prepare_transformer(FLAVORS[config.decoder_flavor]())

        self.text_embeddings = nn.Embedding(config.text_vocab_size, backbone_dim)
        self.audio_embeddings = nn.Embedding(config.audio_vocab_size * config.audio_num_codebooks, backbone_dim)

        self.projection = nn.Linear(backbone_dim, decoder_dim, bias=False)
        self.codebook0_head = nn.Linear(backbone_dim, config.audio_vocab_size, bias=False)
        self.audio_head = nn.Parameter(torch.empty(config.audio_num_codebooks - 1, decoder_dim, config.audio_vocab_size))

    def setup_caches(self, max_batch_size: int) -> None:
        """Setup KV caches and return a causal mask."""
        dtype = next(self.parameters()).dtype
        device = next(self.parameters()).device

        self.backbone.setup_caches(max_batch_size, dtype)
        self.decoder.setup_caches(max_batch_size, dtype, decoder_max_seq_len=self.config.audio_num_codebooks)

        self.to(device)

        self.register_buffer("backbone_causal_mask", _create_causal_mask(self.backbone.max_seq_len, device))
        self.register_buffer("decoder_causal_mask", _create_causal_mask(self.config.audio_num_codebooks, device))

    def generate_frame(
        self,
        tokens: torch.Tensor,
        tokens_mask: torch.Tensor,
        input_pos: torch.Tensor,
        temperature: float,
        topk: int,
    ) -> torch.Tensor:
        """
        Args:
            tokens: (batch_size, seq_len, audio_num_codebooks+1)
            tokens_mask: (batch_size, seq_len, audio_num_codebooks+1)
            input_pos: (batch_size, seq_len) positions for each token
            mask: (batch_size, seq_len, max_seq_len

        Returns:
            (batch_size, audio_num_codebooks) sampled tokens
        """
        dtype = next(self.parameters()).dtype
        b, s, _ = tokens.size()

        assert self.backbone.caches_are_enabled(), "backbone caches are not enabled"
        curr_backbone_mask = _index_causal_mask(self.backbone_causal_mask, input_pos)
        embeds = self._embed_tokens(tokens)
        masked_embeds = embeds * tokens_mask.unsqueeze(-1)
        h = masked_embeds.sum(dim=2)
        h = self.backbone(h, input_pos=input_pos, mask=curr_backbone_mask).to(dtype=dtype)

        last_h = h[:, -1, :]
        c0_logits = self.codebook0_head(last_h)
        c0_sample = sample_topk(c0_logits, topk, temperature)
        c0_embed = self._embed_audio(0, c0_sample)

        curr_h = torch.cat([last_h.unsqueeze(1), c0_embed], dim=1)
        curr_sample = c0_sample.clone()
        curr_pos = torch.arange(0, curr_h.size(1), device=curr_h.device).unsqueeze(0).repeat(curr_h.size(0), 1)

        # Decoder caches must be reset every frame.
        self.decoder.reset_caches()
        for i in range(1, self.config.audio_num_codebooks):
            curr_decoder_mask = _index_causal_mask(self.decoder_causal_mask, curr_pos)
            decoder_h = self.decoder(self.projection(curr_h), input_pos=curr_pos, mask=curr_decoder_mask).to(
                dtype=dtype
            )
            ci_logits = torch.mm(decoder_h[:, -1, :], self.audio_head[i - 1])
            ci_sample = sample_topk(ci_logits, topk, temperature)
            ci_embed = self._embed_audio(i, ci_sample)

            curr_h = ci_embed
            curr_sample = torch.cat([curr_sample, ci_sample], dim=1)
            curr_pos = curr_pos[:, -1:] + 1

        return curr_sample

    def forward(
        self,
        tokens: torch.Tensor,
        tokens_mask: torch.Tensor,
        targets: torch.Tensor,
        targets_mask: torch.Tensor,
        decoder_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dtype = next(self.parameters()).dtype
        b, s, nc_plus_1 = tokens.size()
        num_codebooks = nc_plus_1 - 1
        _, s_amortized = decoder_idx.size()

        targets_c0 = targets[:, :, 0]
        am_idx = decoder_idx.view(b, s_amortized, 1).expand(b, s_amortized, num_codebooks - 1)
        targets_c1_plus = torch.gather(targets[:, :, 1:], dim=1, index=am_idx)
        valid_c0 = targets_mask[:, :, 0].bool()
        valid_c1_plus = torch.gather(targets_mask[:, :, 1:].bool(), dim=1, index=am_idx)

        embeds = self._embed_tokens(tokens)
        masked_embeds = embeds * tokens_mask.unsqueeze(-1)
        h = masked_embeds.sum(dim=2)
        h = self.backbone(h).to(dtype=dtype)

        c0_logits = self.codebook0_head(h)
        h = h.unsqueeze(2)

        target_frame = torch.cat([targets, torch.zeros(b, s, 1, device=h.device, dtype=targets.dtype)], dim=2)
        target_embeds = self._embed_tokens(target_frame)
        decoder_input = torch.cat([h, target_embeds[:, :, :-2, :]], dim=2)

        idx = decoder_idx.view(b, s_amortized, 1, 1).expand(
            b,
            s_amortized,
            num_codebooks,
            decoder_input.size(-1),
        )
        decoder_input_amortized = torch.gather(decoder_input, dim=1, index=idx)
        decoder_h = self.decoder(
            self.projection(decoder_input_amortized).view(b * s_amortized, num_codebooks, -1).to(dtype=dtype)
        )
        decoder_h = decoder_h.view(b, s_amortized, num_codebooks, -1)

        logits_c1_plus = torch.einsum(
            "bsid,idv->bsiv",
            decoder_h[:, :, 1:, :],
            self.audio_head,
        )

        c0_loss, c0_weight = _masked_cross_entropy(
            c0_logits,
            targets_c0,
            valid_c0,
            self.config.audio_vocab_size,
        )
        c1_plus_loss, c1_weight = _masked_cross_entropy(
            logits_c1_plus,
            targets_c1_plus,
            valid_c1_plus,
            self.config.audio_vocab_size,
        )
        loss = (c0_loss * c0_weight + c1_plus_loss * c1_weight) / (c0_weight + c1_weight).clamp_min(1.0)
        return c0_logits, logits_c1_plus, c0_loss, c1_plus_loss, loss

    def reset_caches(self):
        self.backbone.reset_caches()
        self.decoder.reset_caches()

    def _embed_audio(self, codebook: int, tokens: torch.Tensor) -> torch.Tensor:
        return self.audio_embeddings(tokens + codebook * self.config.audio_vocab_size)

    def _embed_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        text_embeds = self.text_embeddings(tokens[:, :, -1]).unsqueeze(-2)

        audio_tokens = tokens[:, :, :-1] + (
            self.config.audio_vocab_size * torch.arange(self.config.audio_num_codebooks, device=tokens.device)
        )
        audio_embeds = self.audio_embeddings(audio_tokens.view(-1)).reshape(
            tokens.size(0), tokens.size(1), self.config.audio_num_codebooks, -1
        )

        return torch.cat([audio_embeds, text_embeds], dim=-2)
