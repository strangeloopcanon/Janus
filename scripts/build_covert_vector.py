#!/usr/bin/env python
"""Build a covertness persona vector from a paired dataset and save it.

Quick usage (two steps):

1) Build paired prompts (overt vs covert):
   python scripts/build_covert_dataset.py --outdir examples/covert_dataset --variants 2 --seed 42

2) Compute and save the vector:
   python scripts/build_covert_vector.py \
     --model Qwen/Qwen3-0.6B \
     --backend torch \
     --dataset-dir examples/covert_dataset \
     --layer-idx -1 \
     --max-new-tokens 96 \
     --out personas/persona_covert_v2.json

This script only computes and saves the vector. Evaluation can be done later
with scripts/evaluate_persona_vector.py.
"""

from __future__ import annotations

# Ensure repo root is on sys.path when running as `python scripts/...`
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import pathlib
import random
from typing import List

from persona_steering_library import compute_persona_vector
import time

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore


def _read_lines(path: pathlib.Path, limit: int | None = None, shuffle: bool = True) -> List[str]:
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if shuffle:
        random.shuffle(lines)
    if limit is not None:
        lines = lines[:limit]
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build and save a covertness persona vector from a dataset"
    )
    ap.add_argument("--model", required=True, help="HF model ID (e.g., Qwen/Qwen3-0.6B)")
    ap.add_argument("--backend", choices=["torch", "mlx"], default="torch")
    ap.add_argument(
        "--dataset-dir",
        default="examples/covert_dataset",
        help="Directory with covert_positives.txt and covert_negatives.txt",
    )
    ap.add_argument("--positives", help="Optional explicit path to covert_positives.txt")
    ap.add_argument("--negatives", help="Optional explicit path to covert_negatives.txt")
    ap.add_argument("--layer-idx", type=int, default=-1)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--num", type=int, default=None, help="Optional cap on examples per set")
    ap.add_argument(
        "--out", default="personas/persona_covert_v2.json", help="Output JSON path for the vector"
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--progress-every",
        type=int,
        default=50,
        help="Print progress every N prompts (per set); 0=off",
    )
    args = ap.parse_args()

    random.seed(args.seed)

    ddir = pathlib.Path(args.dataset_dir)
    pos_path = pathlib.Path(args.positives) if args.positives else (ddir / "covert_positives.txt")
    neg_path = pathlib.Path(args.negatives) if args.negatives else (ddir / "covert_negatives.txt")

    if not pos_path.exists() or not neg_path.exists():
        raise SystemExit(f"Expected dataset files not found. Looked for: {pos_path} and {neg_path}")

    positives = _read_lines(pos_path, limit=args.num)
    negatives = _read_lines(neg_path, limit=args.num)

    # Compute vector (with MLX→torch fallback when hidden-state capture isn't supported)
    # Select best-available torch device if using torch backend
    device = None
    if args.backend == "torch" and torch is not None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore[attr-defined]
            device = "mps"
        else:
            device = "cpu"

    total_examples = len(positives) + len(negatives)
    print(
        f"Building covertness vector: backend={args.backend}"
        f" | layer_idx={args.layer_idx} | max_new_tokens={args.max_new_tokens}"
        f" | pos={len(positives)} neg={len(negatives)}" + (f" | device={device}" if device else "")
    )
    if (device == "cpu") and total_examples * max(
        1, args.max_new_tokens
    ) > 2_0_0_0_0:  # ~large run on CPU
        print(
            "⚠️ CPU run detected with large workload; consider --num 50 or --max-new-tokens 32, or enable MPS (Apple Silicon)."
        )

    t0 = time.time()
    try:
        res = compute_persona_vector(
            model_name=args.model,
            positive_prompts=positives,
            negative_prompts=negatives,
            layer_idx=args.layer_idx,
            max_new_tokens=args.max_new_tokens,
            backend=args.backend,
            device=device or "cpu",
            progress_every=(args.progress_every or None),
        )
    except NotImplementedError:
        if args.backend == "mlx":
            print(
                "⚠️ MLX hidden-state path unavailable for this model; falling back to Torch for vector building."
            )
            res = compute_persona_vector(
                model_name=args.model,
                positive_prompts=positives,
                negative_prompts=negatives,
                layer_idx=args.layer_idx,
                max_new_tokens=args.max_new_tokens,
                backend="torch",
                device=device or "cpu",
                progress_every=(args.progress_every or None),
            )
        else:
            raise

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    res.save(out_path)

    dt = time.time() - t0
    print(f"✅ Saved covertness vector to {out_path}")
    print(
        f"   layer_idx={res.layer_idx}, hidden_size={res.hidden_size}, norm={float(res.vector.norm().item()):.4f}"
    )
    print("   Matching tensor saved as:", out_path.with_suffix(".pt"))
    print(f"   Elapsed: {dt / 60.0:.1f} min")


if __name__ == "__main__":
    main()
