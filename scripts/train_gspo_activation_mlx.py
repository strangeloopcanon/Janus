#!/usr/bin/env python
"""GSPO training in MLX (template; requires MLX support functions).

This script mirrors `train_gspo_activation.py` but uses MLX for the model and
optimizer. It relies on implementations in `persona_steering_library/mlx_support.py`:

- load_model(model_name) -> MLX model
- forward_with_hidden(model, ids_np) -> (logits_np, [hidden_layers_np])

Notes
-----
- This is a reference implementation. You may need to adapt parameter access
  and optimizer updates for your specific MLX model (see mlx-examples/Walter).
"""

from __future__ import annotations

# Ensure repo root is on sys.path when running as `python scripts/...`
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np


def read_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def concat_ids(tok, prompt: str, response: str) -> tuple[np.ndarray, int, int]:
    p = tok(prompt, return_tensors=None)["input_ids"]
    r = tok(response, return_tensors=None, add_special_tokens=False)["input_ids"]
    ids = np.array([p + r], dtype=np.int64)
    return ids, len(p), len(r)


def main() -> None:
    ap = argparse.ArgumentParser(description="GSPO training with MLX (sequence-level)")
    ap.add_argument("--model", required=True, help="HF model ID (weights will be loaded in MLX)")
    ap.add_argument("--data", required=True, help="JSONL from generate_rewrite_groups.py")
    ap.add_argument(
        "--output", required=True, help="Directory to save MLX-trained model (adapter or full)"
    )
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--clip-eps", type=float, default=0.2)
    ap.add_argument("--kl-coeff", type=float, default=0.05)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument(
        "--batch-size", type=int, default=1, help="Batching depends on your MLX model interface"
    )
    args = ap.parse_args()

    # Lazy MLX imports
    try:
        import mlx.core as mx  # type: ignore
        from mlx.optimizers import AdamW  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("MLX is required. Install mlx and mlx-examples.") from exc

    from persona_steering_library import mlx_support

    model, tok = mlx_support.load_model(args.model)  # MLX-native model + tokenizer

    rows = read_jsonl(Path(args.data))

    # Build arrays of old_logp and advantages
    old_logp = np.array([float(r["old_logp"]) for r in rows], dtype=np.float32)
    advantages = np.array([float(r["advantage"]) for r in rows], dtype=np.float32)

    # Optimizer
    opt = AdamW(learning_rate=args.lr)

    def seq_logprob(ids_np: np.ndarray, prompt_len: int, resp_len: int) -> float:
        # Forward (expects numpy in, numpy out)
        logits_np, _ = mlx_support.forward_with_hidden(model, ids_np)
        # Shift and gather logprobs for response tokens
        # logits: (1, T, V)
        logits_np.shape[1]
        labels = ids_np[:, 1:]
        logits_shift = logits_np[:, :-1, :]
        # Convert to mx for stable log-softmax and gather
        logits_mx = mx.array(logits_shift)
        logprobs = logits_mx - mx.logsumexp(logits_mx, axis=-1, keepdims=True)
        # Gather per token
        # Response region indices in token_logprobs are [prompt_len-1 : prompt_len+resp_len-1)
        start = max(0, prompt_len - 1)
        end = max(start, prompt_len + resp_len - 1)
        token_ids = mx.array(labels)
        gathered = mx.take_along_axis(logprobs, token_ids[..., None], axis=-1).squeeze(-1)
        resp_logprobs = gathered[:, start:end]
        return float(mx.sum(resp_logprobs).item())

    def loss_fn(batch_indices: List[int]):
        # Compute GSPO loss + KL surrogate for a small batch
        logp_cur_list = []
        logp_ref_list = []
        for idx in batch_indices:
            r = rows[idx]
            ids, pl, rl = concat_ids(tok, r["prompt"], r["response"])
            cur_lp = seq_logprob(ids, pl, rl)
            # For simplicity, use the current model as ref with stop-grad
            ref_lp = cur_lp
            logp_cur_list.append(cur_lp)
            logp_ref_list.append(ref_lp)
        logp_cur = mx.array(np.array(logp_cur_list, dtype=np.float32))
        logp_old = mx.array(old_logp[batch_indices])
        adv = mx.array(advantages[batch_indices])
        ratio = mx.exp(logp_cur - logp_old)
        clipped = mx.minimum(mx.maximum(ratio, 1.0 - args.clip_eps), 1.0 + args.clip_eps)
        obj = mx.minimum(ratio * adv, clipped * adv)
        gspo_loss = -mx.mean(obj)
        # KL proxy (0 if ref==cur); replace with separate reference if available
        kl = mx.mean(logp_cur - mx.array(np.array(logp_ref_list, dtype=np.float32)))
        return gspo_loss + args.kl_coeff * kl

    # Training loop (simple, per-sample or small batches)
    n = len(rows)
    indices = np.arange(n)
    for epoch in range(args.epochs):
        np.random.shuffle(indices)
        for i in range(0, n, args.batch_size):
            batch_idx = indices[i : i + args.batch_size].tolist()
            # Compute gradients
            loss, grads = mx.value_and_grad(loss_fn)(batch_idx)
            # Update params (you may need to adapt to your MLX model param API)
            try:
                params = model.parameters()
            except AttributeError:
                raise NotImplementedError(
                    "Your MLX model must expose .parameters() compatible with mlx.optimizers."
                )
            opt.update(params, grads)
            print(
                f"epoch={epoch + 1} step={i // max(1, args.batch_size) + 1} loss={loss.item():.4f}"
            )

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Save model + tokenizer via support helper
    from persona_steering_library.mlx_support import save_pretrained as mlx_save

    mlx_save(model, tok, str(out_dir))


if __name__ == "__main__":
    main()
