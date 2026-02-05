#!/usr/bin/env python
"""Minimal CLI to train a persona vector for Qwen-3 0.6B (or any HF model).

Usage
-----
python scripts/train_persona_vector.py \
    --model Qwen/Qwen3-0.6B \
    --positive-prompts positive.txt \
    --negative-prompts negative.txt \
    --out personas/persona_formal_informal.json

Each *prompts* file must contain UTF-8 encoded lines with one prompt per line.
"""

from __future__ import annotations

# Ensure repo root is on sys.path when running as `python scripts/...`
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import pathlib

from persona_steering_library import compute_persona_vector


def _read_prompts(path: str | pathlib.Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as fp:
        return [ln.strip() for ln in fp if ln.strip()]


def main() -> None:  # noqa: D401
    parser = argparse.ArgumentParser(description="Train a persona vector")
    parser.add_argument("--model", required=True, help="HF model name or path")
    parser.add_argument(
        "--positive-prompts", required=True, help="File with prompts for the *positive* persona"
    )
    parser.add_argument(
        "--negative-prompts", required=True, help="File with prompts for the *negative* persona"
    )
    # Output options
    parser.add_argument("--out", help="Path to store a single persona vector (JSON + .pt)")
    parser.add_argument(
        "--outdir", help="Output directory when sweeping layers (used with --last-n-layers)"
    )
    parser.add_argument(
        "--name",
        default="persona_custom",
        help="Base name for output files when sweeping layers (e.g., persona_<name>_L-1.json)",
    )
    # Generation / backend options
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--backend", choices=["torch", "mlx"], default="torch")
    # Layer control: either --layer-idx for a single vector or --last-n-layers to sweep
    parser.add_argument(
        "--layer-idx", type=int, default=-1, help="Single layer index (e.g., -1 for last layer)"
    )
    parser.add_argument(
        "--last-n-layers",
        type=int,
        default=0,
        help="If >0, build vectors for the last N layers (-N..-1) and write to --outdir",
    )
    args = parser.parse_args()

    pos_prompts = _read_prompts(args.positive_prompts)
    neg_prompts = _read_prompts(args.negative_prompts)

    if args.last_n_layers and args.last_n_layers > 0:
        # Sweep last N layers and save one file per layer
        if not args.outdir:
            raise SystemExit("--outdir is required when using --last-n-layers")
        from pathlib import Path as _Path

        outdir = _Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        layers = list(range(-args.last_n_layers, 0))
        print(f"Building vectors for layers {layers} → {outdir}")
        for L in layers:
            res = compute_persona_vector(
                model_name=args.model,
                positive_prompts=pos_prompts,
                negative_prompts=neg_prompts,
                layer_idx=L,
                max_new_tokens=args.max_new_tokens,
                backend=args.backend,
            )
            jpath = outdir / f"persona_{args.name}_L{L}.json"
            res.save(jpath)
            print(f"✓ saved {jpath}")
        return

    # Single-layer path
    if not args.out:
        raise SystemExit("--out is required when not using --last-n-layers")

    result = compute_persona_vector(
        model_name=args.model,
        positive_prompts=pos_prompts,
        negative_prompts=neg_prompts,
        layer_idx=args.layer_idx,
        max_new_tokens=args.max_new_tokens,
        backend=args.backend,
    )

    result.save(args.out)
    print(f"Persona vector saved to {args.out} (+ .pt file)")


if __name__ == "__main__":
    main()
