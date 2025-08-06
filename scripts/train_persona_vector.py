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

import argparse
import pathlib

from persona_steering_library import compute_persona_vector


def _read_prompts(path: str | pathlib.Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as fp:
        return [ln.strip() for ln in fp if ln.strip()]


def main() -> None:  # noqa: D401
    parser = argparse.ArgumentParser(description="Train a persona vector")
    parser.add_argument("--model", required=True, help="HF model name or path")
    parser.add_argument("--positive-prompts", required=True, help="File with prompts for the *positive* persona")
    parser.add_argument("--negative-prompts", required=True, help="File with prompts for the *negative* persona")
    parser.add_argument("--out", required=True, help="Where to store the persona vector (JSON + .pt)")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--backend", choices=["torch", "mlx"], default="torch")
    args = parser.parse_args()

    pos_prompts = _read_prompts(args.positive_prompts)
    neg_prompts = _read_prompts(args.negative_prompts)

    result = compute_persona_vector(
        model_name=args.model,
        positive_prompts=pos_prompts,
        negative_prompts=neg_prompts,
        max_new_tokens=args.max_new_tokens,
        backend=args.backend,
    )

    result.save(args.out)
    print(f"Persona vector saved to {args.out} (+ .pt file)")


if __name__ == "__main__":
    main()
