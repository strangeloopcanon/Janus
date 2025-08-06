"""Optional MLX backend.

This module is *imported lazily*.  If MLX is not installed, importing any of
its public symbols will raise ``ImportError`` so that the rest of the toolkit
continues to work on PyTorch.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Optional


def _lazy_import() -> SimpleNamespace:  # noqa: D401
    try:
        import mlx.core as mx  # type: ignore
        import mlx.nn as nn  # type: ignore
        from huggingface_hub import snapshot_download  # noqa: WPS433 external IO
    except ModuleNotFoundError as exc:  # pragma: no cover – runtime guard
        raise ImportError("MLX backend requested but 'mlx' or deps are not installed") from exc

    return SimpleNamespace(mx=mx, nn=nn, snapshot_download=snapshot_download)


def load_model(model_name: str):  # noqa: D401
    """Download model weights (Safetensors) and instantiate an *MLX* model.

    Implementation is intentionally minimal; for advanced functionality users
    can swap this loader with their preferred *mlx-lm* pipeline.
    """

    libs = _lazy_import()
    mx, nn, snapshot_download = libs.mx, libs.nn, libs.snapshot_download  # type: ignore

    # Download HF repo to local cache (weights + config)
    cache_dir = snapshot_download(model_name, allow_patterns=["*.safetensors", "*.json", "*.txt"])
    # NOTE: Proper MLX loader needs to parse config and build module.  Here we
    # raise *NotImplementedError* to document the gap while keeping the API.
    raise NotImplementedError(
        "MLX backend stub – please implement the architecture specific to your model. "
        "Community loaders: https://github.com/ml-explore/mlx-examples"
    )


def nucleus_sampling(
    logits,
    *,
    top_p: float = 0.9,
    temperature: float = 1.0,
    rng: Optional["numpy.random.Generator"] = None,
):  # noqa: D401
    """Minimal Nucleus (top-p) sampling in NumPy for deterministic MLX runs."""

    import numpy as np

    if rng is None:
        rng = np.random.default_rng()

    logits = logits / max(temperature, 1e-5)
    probs = np.exp(logits - logits.max())
    probs = probs / probs.sum()

    # sort descending
    sorted_idx = np.argsort(-probs)
    sorted_probs = probs[sorted_idx]
    cumulative = np.cumsum(sorted_probs)
    mask = cumulative <= top_p
    if not mask.any():
        mask[0] = True  # ensure at least 1 token

    filtered_idx = sorted_idx[mask]
    filtered_probs = probs[filtered_idx]
    filtered_probs = filtered_probs / filtered_probs.sum()

    choice = rng.choice(filtered_idx, p=filtered_probs)
    return int(choice)

