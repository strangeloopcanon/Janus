from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from persona_steering_library.hooks import add_persona_hook


class IdentityBlock(nn.Module):
    def forward(self, x):
        return x


class TupleBlock(nn.Module):
    def forward(self, x):
        return x, torch.ones_like(x)


class DummyModel(nn.Module):
    def __init__(self, *, tuple_output: bool = False, transformer_fallback: bool = False):
        super().__init__()
        block: nn.Module = TupleBlock() if tuple_output else IdentityBlock()
        self._blocks = nn.ModuleList([block])
        self.config = SimpleNamespace(hidden_size=4)
        self.device = torch.device("cpu")
        if transformer_fallback:
            self.transformer = SimpleNamespace(h=self._blocks)
        else:
            self.model = SimpleNamespace(layers=self._blocks)

    def forward(self, x):
        if hasattr(self, "model"):
            block = self.model.layers[0]
        else:
            block = self.transformer.h[0]
        return block(x)


def test_add_persona_hook_tensor_output():
    model = DummyModel()
    vector = torch.ones(4)
    remove = add_persona_hook(model, vector, alpha=0.5)

    inp = torch.zeros(1, 2, 4)
    out = model(inp)
    assert torch.allclose(out, torch.full_like(out, 0.5))

    remove()
    out2 = model(inp)
    assert torch.allclose(out2, torch.zeros_like(inp))


def test_add_persona_hook_tuple_output_transformer_fallback():
    model = DummyModel(tuple_output=True, transformer_fallback=True)
    vector = torch.ones(4)
    remove = add_persona_hook(model, vector, alpha=0.25)

    inp = torch.zeros(1, 2, 4)
    out = model(inp)
    assert isinstance(out, tuple)
    assert torch.allclose(out[0], torch.full_like(out[0], 0.25))
    assert torch.allclose(out[1], torch.ones_like(inp))

    remove()
    out2 = model(inp)
    assert isinstance(out2, tuple)
    assert torch.allclose(out2[0], torch.zeros_like(inp))


def test_add_persona_hook_dimension_mismatch():
    model = DummyModel()
    vector = torch.ones(3)
    try:
        add_persona_hook(model, vector, alpha=1.0)
    except ValueError as exc:
        assert "hidden_size" in str(exc)
    else:
        raise AssertionError("Expected a hidden-size mismatch error")
