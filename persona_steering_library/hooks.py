"""Runtime steering hooks."""

from __future__ import annotations

from typing import Any, Callable

import torch
from transformers import PreTrainedModel


def add_persona_hook(
    model: PreTrainedModel,
    vector: torch.Tensor,
    *,
    layer_idx: int = -1,
    alpha: float | torch.Tensor = 1.0,
) -> Callable[[], None]:
    """Add a *forward hook* that injects `alpha * vector` into the residual stream.

    Parameters
    ----------
    model
        The loaded HuggingFace model instance.
    vector
        1-D tensor with length equal to the model's hidden size.
    layer_idx
        Which transformer block **within `model.model.layers`** (Qwen / most
        decoder models) to patch.  Use -1 for the final block.
    alpha
        Steering coefficient.  Provide a *tensor* if you want to update it
        dynamically without re-registering the hook.

    Returns
    -------
    Callable[[], None]
        A *remove* function.  Call it to detach the hook.
    """

    if not isinstance(vector, torch.Tensor):
        raise TypeError("vector must be a torch.Tensor")

    if vector.dim() != 1:
        raise ValueError("vector must be 1-D (hidden_size,)")

    vector = vector.to(model.device)
    try:
        hidden_size = model.config.hidden_size
    except Exception:  # noqa: BLE001
        hidden_size = vector.numel()
    if vector.numel() != hidden_size:
        raise ValueError(f"persona vector size {vector.numel()} != model hidden_size {hidden_size}")

    # Broadcast to sequence length at runtime
    def _hook(_module, _input, output):  # pylint: disable=unused-argument
        # Handle tuple outputs (some models return (hidden_states, attention_weights, ...))
        if isinstance(output, tuple):
            # Apply the persona vector to the first element (hidden states)
            modified_hidden = output[0] + alpha * vector  # broadcasting on (hidden,)
            return (modified_hidden,) + output[1:]  # return modified hidden + rest of tuple
        else:
            # output: (batch, seq_len, hidden)
            return output + alpha * vector  # broadcasting on (hidden,)

    # Transformer decoder architectures usually expose blocks at
    # model.model.layers (OPT, LLaMA, Qwen).  Fall back to .transformer.h if
    # necessary.
    blocks: Any
    if hasattr(model, "model") and hasattr(model.model, "layers"):  # type: ignore[attr-defined]
        blocks = model.model.layers  # type: ignore[attr-defined]
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):  # type: ignore[attr-defined]
        blocks = model.transformer.h  # type: ignore[attr-defined]
    else:
        raise AttributeError("Could not locate transformer blocks on model")

    block = blocks[layer_idx]
    handle = block.register_forward_hook(_hook, prepend=False)

    return handle.remove  # caller can invoke to clean up
