"""Covertness detector utilities.

Provides a simple linear-probe detector over mean hidden states to estimate
detectability probability of an output being "overt" (exhibiting easy-to-spot
stylistic cues). The detector can be trained with `scripts/train_covertness_detector.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import json
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase


class DetectorProtocol:  # runtime protocol for type hints
    def predict_proba(self, *, model: PreTrainedModel, tok: PreTrainedTokenizerBase, text: str, device: torch.device | str) -> float:  # noqa: D401
        """Return P(overt | text) in [0,1]."""
        raise NotImplementedError


@dataclass
class LinearProbeConfig:
    layer_idx: int
    hidden_size: int
    bias: float
    # weights stored separately as .pt tensor


class LinearProbeDetector(DetectorProtocol):
    """Logistic regression over mean hidden at specified layer.

    Files:
    - JSON with metadata: {layer_idx, hidden_size, bias}
    - PT with weight vector: same basename, suffix .pt
    """

    def __init__(self, cfg: LinearProbeConfig, weight: torch.Tensor):
        self.cfg = cfg
        self.weight = weight  # (hidden,)

    @classmethod
    def load(cls, path: str | bytes | "os.PathLike[str]") -> "LinearProbeDetector":
        import pathlib

        p = pathlib.Path(path)
        with p.open("r", encoding="utf-8") as fp:
            meta = json.load(fp)
        cfg = LinearProbeConfig(layer_idx=meta["layer_idx"], hidden_size=meta["hidden_size"], bias=float(meta["bias"]))
        w = torch.load(p.with_suffix(".pt"), map_location="cpu")
        return cls(cfg, w)

    def predict_proba(self, *, model: PreTrainedModel, tok: PreTrainedTokenizerBase, text: str, device: torch.device | str) -> float:
        device = torch.device(device)
        with torch.no_grad():
            ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
            out = model(ids, output_hidden_states=True)
            h = out.hidden_states[self.cfg.layer_idx][0].mean(dim=0)  # (hidden,)
        w = self.weight.to(h.device)
        logit = torch.dot(h, w) + torch.tensor(self.cfg.bias, device=h.device, dtype=h.dtype)
        prob = torch.sigmoid(logit).item()
        return float(prob)
