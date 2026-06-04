import torch
import torch.nn as nn

from generator import _should_apply_mps_bf16_layer0_mlp_fp32_guard
from models import apply_mps_bf16_layer0_mlp_fp32_guard


class RecordingActivation(nn.Module):
    def __init__(self):
        super().__init__()
        self.last_dtype = None

    def forward(self, x):
        self.last_dtype = x.dtype
        return x


class TinyMlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Linear(3, 4, bias=False)
        self.w2 = nn.Linear(4, 3, bias=False)
        self.w3 = nn.Linear(3, 4, bias=False)
        self.activation = RecordingActivation()

    def forward(self, x):
        h = self.activation(self.w1(x))
        h = h * self.w3(x)
        return self.w2(h)


class TinyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = TinyMlp()


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([TinyLayer(), TinyLayer()])


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = TinyBackbone()


def test_layer0_mlp_guard_uses_fp32_compute_and_preserves_state_dict_keys():
    model = TinyModel().to(dtype=torch.bfloat16)
    before_keys = list(model.state_dict().keys())

    apply_mps_bf16_layer0_mlp_fp32_guard(model)
    out = model.backbone.layers[0].mlp(torch.ones(2, 3, dtype=torch.bfloat16))

    assert out.dtype == torch.bfloat16
    assert model.backbone.layers[0].mlp.activation.last_dtype == torch.float32
    assert model.backbone.layers[0].mlp.w1.weight.dtype == torch.bfloat16
    assert list(model.state_dict().keys()) == before_keys


def test_layer0_mlp_guard_patches_only_first_backbone_layer():
    model = TinyModel()

    apply_mps_bf16_layer0_mlp_fp32_guard(model)

    assert model.backbone.layers[0].mlp._miso_mps_bf16_layer0_mlp_fp32_guard
    assert not hasattr(model.backbone.layers[1].mlp, "_miso_mps_bf16_layer0_mlp_fp32_guard")


def test_mps_bf16_guard_auto_enable_rules(monkeypatch):
    monkeypatch.delenv("MISO_DISABLE_MPS_BF16_LAYER0_MLP_FP32", raising=False)

    assert _should_apply_mps_bf16_layer0_mlp_fp32_guard("mps", torch.bfloat16, None)
    assert not _should_apply_mps_bf16_layer0_mlp_fp32_guard("cpu", torch.bfloat16, None)
    assert not _should_apply_mps_bf16_layer0_mlp_fp32_guard("cuda", torch.bfloat16, None)
    assert not _should_apply_mps_bf16_layer0_mlp_fp32_guard("mps", torch.float32, None)

    monkeypatch.setenv("MISO_DISABLE_MPS_BF16_LAYER0_MLP_FP32", "1")
    assert not _should_apply_mps_bf16_layer0_mlp_fp32_guard("mps", torch.bfloat16, None)
    assert _should_apply_mps_bf16_layer0_mlp_fp32_guard("cpu", torch.bfloat16, True)
    assert not _should_apply_mps_bf16_layer0_mlp_fp32_guard("mps", torch.bfloat16, False)
