#!/usr/bin/env python
"""Inspect MLX model structure to locate embedding/layers/norm/out_proj.

Usage:
  python scripts/inspect_mlx_model.py --model Qwen/Qwen3-0.6B
"""

from __future__ import annotations

import argparse
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from persona_steering_library.mlx_support import load_model  # type: ignore


def has(obj, path: str) -> bool:
    cur = obj
    for part in path.split('.'):
        if not hasattr(cur, part):
            return False
        cur = getattr(cur, part)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()

    model, tok = load_model(args.model)
    candidates = {
        "embedding": [
            "embedding", "embed_tokens", "tok_embeddings",
            "model.embedding", "model.embed_tokens", "model.tok_embeddings",
            "model.model.embed_tokens"
        ],
        "layers": [
            "layers", "model.layers", "transformer.layers", "model.model.layers"
        ],
        "norm": [
            "norm", "model.norm", "transformer.norm", "ln_f", "model.model.norm"
        ],
        "out_proj": [
            "out_proj", "lm_head", "output", "model.lm_head", "model.output", "model.model.lm_head"
        ],
    }
    print("Model type:", type(model))
    for key, paths in candidates.items():
        found = [p for p in paths if has(model, p)]
        print(f"{key}: {found[:3]}{'...' if len(found)>3 else ''}")

    # Also print top-level attrs for quick scan
    print("Top-level attributes:", sorted([a for a in dir(model) if not a.startswith('_')])[:50])


if __name__ == "__main__":
    main()

