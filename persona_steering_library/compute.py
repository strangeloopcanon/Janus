"""Utilities to build a *persona vector* for an LLM.

The idea is to collect activation vectors for two opposite personas (e.g.
`formal` vs `informal`) and take their difference.  This file contains a helper
that performs all steps in a single function so small experiments can be run
with just a few lines of code.

This is *research prototype* code – it is written to be easy to read and hack
on rather than to squeeze out every last GPU-second.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass
from typing import Any, Sequence, TYPE_CHECKING, cast

import torch

# Avoid importing transformers at module import time so utility consumers
# (e.g., space converters) don't require it just to load/save tensors.
if TYPE_CHECKING:  # pragma: no cover - type hints only
    from transformers import PreTrainedModel, PreTrainedTokenizerBase  # noqa: F401

# Optional MLX backend flag
try:
    from persona_steering_library import mlx_support

    _HAS_MLX = True
except ImportError:  # pragma: no cover – missing optional dep
    _HAS_MLX = False


@dataclass
class PersonaVectorResult:
    """Return type of :func:`compute_persona_vector`."""

    vector: torch.Tensor  # shape: (hidden_size,)
    layer_idx: int
    hidden_size: int

    def save(self, path: str | pathlib.Path) -> None:  # noqa: D401
        """Save to *path* as JSON with `.pt` buffer for the vector itself."""

        path = pathlib.Path(path)
        payload = {
            "layer_idx": self.layer_idx,
            "hidden_size": self.hidden_size,
        }

        vec_file = path.with_suffix(".pt")
        torch.save(self.vector.cpu(), vec_file)
        with path.open("w", encoding="utf-8") as fp:
            json.dump(payload, fp)

    @classmethod
    def load(cls, path: str | pathlib.Path) -> "PersonaVectorResult":
        path = pathlib.Path(path)
        with path.open("r", encoding="utf-8") as fp:
            meta = json.load(fp)
        vec = torch.load(path.with_suffix(".pt"), map_location="cpu")
        return cls(vector=vec, layer_idx=meta["layer_idx"], hidden_size=meta["hidden_size"])


def _init_model_and_tokenizer(model_name: str, device: str | torch.device, backend: str = "torch"):
    if backend == "torch":
        # Lazy import to avoid hard dependency at module import time
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = cast(
            Any, AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True)
        )
        model.to(str(device))
        model.eval()
        return model, tokenizer

    if backend == "mlx":
        if not _HAS_MLX:
            raise RuntimeError("MLX backend requested but 'mlx' not installed")

        # Prefer tokenizer returned by MLX loader to ensure vocab match
        model, tokenizer = mlx_support.load_model(model_name)  # type: ignore[attr-defined]
        return model, tokenizer

    raise ValueError(f"Unknown backend: {backend}")


def _generate(
    model,
    tokenizer,
    prompts: Sequence[str],
    *,
    max_new_tokens: int,
    device: str,
    backend: str = "torch",
    pool_layer_idx: int = -1,
    progress_every: int | None = None,
) -> list[torch.Tensor]:
    """Generate completions *and* collect last-layer activations for the **completion**.

    Returns a list of tensors with shape (hidden_size,) – pooled activations.
    """

    if backend == "torch":
        pooled_torch: list[torch.Tensor] = []

        # Lazy import to avoid global dependency
        from transformers import GenerationConfig  # type: ignore

        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.9,
            temperature=1.0,
        )

        total = len(prompts)
        for i, prompt in enumerate(prompts, 1):
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            input_len = inputs["input_ids"].shape[1]

            # Generate completion ids; hidden states are collected in a second forward pass
            with torch.no_grad():
                gen_out = model.generate(
                    **inputs,
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                )

            full_ids = gen_out.sequences  # (1, seq_total)

            with torch.no_grad():
                out = model(full_ids, output_hidden_states=True)

            layer_hidden = out.hidden_states[pool_layer_idx][0]  # (seq, hidden)
            completion_hidden = layer_hidden[input_len:]
            mean_vec = completion_hidden.mean(dim=0)
            pooled_torch.append(mean_vec.cpu())
            if progress_every and (i % progress_every == 0 or i == total):
                print(f"Torch gen/pooled {i}/{total} (layer {pool_layer_idx})", flush=True)

        return pooled_torch

    if backend == "mlx":
        if not _HAS_MLX:
            raise RuntimeError("MLX backend requested but 'mlx' not installed")

        from persona_steering_library.mlx_support import (
            generate_with_layer_injection,
            forward_with_hidden,
            mean_hidden_over_span,
            tok_encode,
            _get_components,
        )

        libs = mlx_support._lazy_import()  # type: ignore[attr-defined]
        mx = libs.mx

        pooled_mlx: list[torch.Tensor] = []

        total = len(prompts)
        for i, prompt in enumerate(prompts, 1):
            # 1) Sample a completion (alpha=0 → no injection). Uses KV cache and is faster.
            completion = generate_with_layer_injection(
                model,
                tokenizer,
                prompt,
                vector_hidden=None,
                alpha=0.0,
                max_tokens=max_new_tokens,
                temperature=1.0,
                top_p=0.9,
            )

            # 2) Run full forward with hidden capture and pool completion region
            p_ids = tok_encode(tokenizer, prompt)
            r_ids = tok_encode(tokenizer, completion, add_special_tokens=False)
            ids = mx.array(p_ids + r_ids, dtype=mx.int32)
            start = len(p_ids)
            end = start + len(r_ids)
            # Resolve negative layer index against model depth
            try:
                _, layers, _, _ = _get_components(model)
                k = pool_layer_idx if pool_layer_idx >= 0 else (len(layers) - 1)
            except Exception:
                k = pool_layer_idx
            _, hidden = forward_with_hidden(model, ids, capture_layers=(k,))
            h = hidden[k]  # [1, T, D]
            mean_mx = mean_hidden_over_span(h, start, end)  # [1, D]
            mean_t = torch.tensor(mean_mx.squeeze(0).tolist(), dtype=torch.float32)
            pooled_mlx.append(mean_t)
            if progress_every and (i % progress_every == 0 or i == total):
                print(f"MLX gen/pooled {i}/{total} (layer {pool_layer_idx})", flush=True)

        return pooled_mlx

    raise ValueError("Unknown backend")


def compute_persona_vector(
    model_name: str,
    positive_prompts: Sequence[str],
    negative_prompts: Sequence[str],
    *,
    layer_idx: int = -1,
    max_new_tokens: int = 64,
    device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
    backend: str = "torch",
    progress_every: int | None = None,
) -> PersonaVectorResult:
    """Return a linear *persona vector* that distinguishes the two prompt sets.

    Parameters
    ----------
    model_name
        HuggingFace model ID (e.g. ``"Qwen/Qwen3-0.6B"``).
    positive_prompts / negative_prompts
        Sequences of strings designed to elicit the opposite personas.
    layer_idx
        Which transformer block's residual stream to target.  *-1* stands for
        "final block".
    max_new_tokens
        How many tokens to sample for each prompt.
    device
        torch device string – default to GPU when available.

    backend
        "torch" (default) or "mlx" – controls which deep-learning stack is used.
    """

    model, tokenizer = _init_model_and_tokenizer(model_name, str(device), backend)

    # ───────────────────────────────── collect activations ──────────────────────────────────
    act_pos = _generate(
        model,
        tokenizer,
        positive_prompts,
        max_new_tokens=max_new_tokens,
        device=str(device),
        backend=backend,
        pool_layer_idx=layer_idx,
        progress_every=progress_every,
    )
    act_neg = _generate(
        model,
        tokenizer,
        negative_prompts,
        max_new_tokens=max_new_tokens,
        device=str(device),
        backend=backend,
        pool_layer_idx=layer_idx,
        progress_every=progress_every,
    )

    # Infer hidden size from activations (robust to backend differences)
    hidden_size = int(act_pos[0].numel()) if len(act_pos) > 0 else int(act_neg[0].numel())

    # Stack into matrices – each row is an example
    X_pos = torch.stack(act_pos, dim=0)  # (n_pos, hidden)
    X_neg = torch.stack(act_neg, dim=0)  # (n_neg, hidden)

    # ───────────────────────────── compute direction (difference of means) ─────────────────
    vec = X_pos.mean(dim=0) - X_neg.mean(dim=0)
    vec = vec / (vec.norm(p=2) + 1e-12)

    return PersonaVectorResult(vector=vec, layer_idx=layer_idx, hidden_size=hidden_size)
