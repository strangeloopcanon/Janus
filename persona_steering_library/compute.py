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
from typing import Sequence

import torch
# PyTorch / HF imports
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

# Optional MLX backend flag
try:
    from persona_steering_library import mlx_support  # noqa: WPS433  optional dependency

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


def _init_model_and_tokenizer(model_name: str, device: str, backend: str = "torch"):
    if backend == "torch":
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True)
        model.to(device)
        model.eval()
        return model, tokenizer

    if backend == "mlx":
        if not _HAS_MLX:
            raise RuntimeError("MLX backend requested but 'mlx' not installed")

        # mlx_support.load_model is currently a stub raising NotImplementedError.
        model = mlx_support.load_model(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
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
) -> list[torch.Tensor]:
    """Generate completions *and* collect last-layer activations for the **completion**.

    Returns a list of tensors with shape (hidden_size,) – pooled activations.
    """

    if backend == "torch":
        pooled: list[torch.Tensor] = []

        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.9,
            temperature=1.0,
        )

        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            input_len = inputs["input_ids"].shape[1]

            with torch.no_grad():
                gen_out = model.generate(
                    **inputs,
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    output_hidden_states=True,
                )

            full_ids = gen_out.sequences  # (1, seq_total)

            with torch.no_grad():
                out = model(full_ids, output_hidden_states=True)

            last_hidden = out.hidden_states[-1][0]  # (seq, hidden)
            completion_hidden = last_hidden[input_len:]
            mean_vec = completion_hidden.mean(dim=0)
            pooled.append(mean_vec.cpu())

        return pooled

    if backend == "mlx":
        if not _HAS_MLX:
            raise RuntimeError("MLX backend requested but 'mlx' not installed")

        import numpy as np
        from persona_vectors.mlx_support import nucleus_sampling  # type: ignore

        pooled: list[torch.Tensor] = []

        mx = __import__("mlx.core", fromlist=["array"])  # noqa: WPS420 dynamic import

        for prompt in prompts:
            # Tokenize prompt (hf tokenizer gives ids list)
            input_ids = tokenizer(prompt, return_tensors=None, add_special_tokens=False)["input_ids"]

            seq = list(input_ids)

            for _ in range(max_new_tokens):
                # model forward expects numpy / mx arrays – convert
                logits, hidden_states = model(np.array([seq]))  # type: ignore  # returns logits

                logits = logits[0, -1]  # last token

                next_id = nucleus_sampling(logits, top_p=0.9, temperature=1.0)
                seq.append(int(next_id))

            # Get hidden states for full sequence to pool completion activations
            _, hiddens = model(np.array([seq]))

            last_hidden = hiddens[-1][0]  # (seq_total, hidden)
            completion_hidden = last_hidden[len(input_ids):]
            mean_vec = torch.tensor(completion_hidden.mean(axis=0))
            pooled.append(mean_vec)

        return pooled

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

    model, tokenizer = _init_model_and_tokenizer(model_name, device, backend)

    # Make sure we only keep the activations we care about.  We will detach later.
    hidden_size = model.config.hidden_size

    # ───────────────────────────────── collect activations ──────────────────────────────────
    act_pos = _generate(model, tokenizer, positive_prompts, max_new_tokens=max_new_tokens, device=device, backend=backend)
    act_neg = _generate(model, tokenizer, negative_prompts, max_new_tokens=max_new_tokens, device=device, backend=backend)

    # Stack into matrices – each row is an example
    X_pos = torch.stack(act_pos, dim=0)  # (n_pos, hidden)
    X_neg = torch.stack(act_neg, dim=0)  # (n_neg, hidden)

    # ───────────────────────────── compute direction (difference of means) ─────────────────
    vec = X_pos.mean(dim=0) - X_neg.mean(dim=0)
    vec = vec / (vec.norm(p=2) + 1e-12)

    return PersonaVectorResult(vector=vec, layer_idx=layer_idx, hidden_size=hidden_size)
