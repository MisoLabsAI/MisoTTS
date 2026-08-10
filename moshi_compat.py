from __future__ import annotations

import torch
from torch import nn

from moshi.utils import quantize


def patch_bitsandbytes_import_for_unquantized_layers() -> None:
    original_linear = quantize.linear
    original_multi_linear = quantize.multi_linear

    def linear(module: nn.Module, x: torch.Tensor, name: str = "weight") -> torch.Tensor:
        if quantize.is_quantized(module, name):
            return original_linear(module, x, name)
        # Keep the same semantics as the quantized path for ordinary layers.
        # ``nn.functional.linear`` does not infer a module's bias, so omitting
        # it here silently changes the output of biased unquantized layers.
        return nn.functional.linear(x, getattr(module, name), getattr(module, "bias", None))

    def multi_linear(
        num_steps: int,
        schedule: list[int] | None,
        module: nn.Module,
        x: torch.Tensor,
        offset: int,
        name: str = "weight",
    ) -> torch.Tensor:
        if quantize.is_quantized(module, name):
            return original_multi_linear(num_steps, schedule, module, x, offset, name)

        weight = getattr(module, name)
        num_linear = num_steps if schedule is None else max(schedule) + 1
        weight = weight.view(num_linear, -1, weight.shape[-1])

        outputs = []
        for t in range(x.shape[1]):
            linear_index = t + offset
            if schedule is not None:
                linear_index = schedule[linear_index]
            outputs.append(nn.functional.linear(x[:, t], weight[linear_index]))
        return torch.stack(outputs, 1)

    quantize.linear = linear
    quantize.multi_linear = multi_linear
