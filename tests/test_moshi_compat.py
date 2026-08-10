import importlib.util
import sys
import types
import unittest

import torch


def load_compat():
    """Load moshi_compat with a small stand-in for the optional moshi package."""
    quantize = types.ModuleType("moshi.utils.quantize")
    quantize.linear = lambda module, x, name="weight": torch.zeros_like(x)
    quantize.multi_linear = lambda *args, **kwargs: None
    quantize.is_quantized = lambda module, name: False

    utils = types.ModuleType("moshi.utils")
    utils.quantize = quantize
    moshi = types.ModuleType("moshi")
    moshi.utils = utils
    sys.modules.update(
        {"moshi": moshi, "moshi.utils": utils, "moshi.utils.quantize": quantize}
    )

    spec = importlib.util.spec_from_file_location("moshi_compat", "moshi_compat.py")
    compat = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(compat)
    return compat, quantize


class UnquantizedLinearTests(unittest.TestCase):
    def test_preserves_bias(self):
        compat, quantize = load_compat()
        compat.patch_bitsandbytes_import_for_unquantized_layers()

        layer = torch.nn.Linear(2, 1, bias=True)
        with torch.no_grad():
            layer.weight.copy_(torch.tensor([[2.0, -3.0]]))
            layer.bias.fill_(4.0)
        x = torch.tensor([[5.0, 6.0]])

        actual = quantize.linear(layer, x)

        torch.testing.assert_close(actual, torch.tensor([[-4.0]]))

    def test_supports_layers_without_bias(self):
        compat, quantize = load_compat()
        compat.patch_bitsandbytes_import_for_unquantized_layers()

        layer = torch.nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            layer.weight.copy_(torch.tensor([[2.0, -3.0]]))
        x = torch.tensor([[5.0, 6.0]])

        actual = quantize.linear(layer, x)

        torch.testing.assert_close(actual, torch.tensor([[-8.0]]))


if __name__ == "__main__":
    unittest.main()
