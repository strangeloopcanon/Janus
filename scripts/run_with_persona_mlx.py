#!/usr/bin/env python
"""Interactive chat with MLX model + activation-space persona steering.

Usage:
  python scripts/run_with_persona_mlx.py \
    --model Qwen/Qwen3-0.6B \
    --persona personas/persona_formal.json \
    --alpha 1.0 \
    [--layer-idx -1]
"""

from __future__ import annotations

# Ensure repo root is on sys.path when running as `python scripts/...`
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse

from persona_steering_library import PersonaVectorResult
from persona_steering_library.mlx_support import (
    load_model,
    generate_with_layer_injection,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="MLX interactive chat with persona steering")
    ap.add_argument("--model", required=True, help="HF model ID")
    ap.add_argument("--persona", required=True, help="Path to persona JSON")
    ap.add_argument("--alpha", type=float, default=1.0, help="Steering strength")
    ap.add_argument("--layer-idx", type=int, default=-1, help="Layer index for injection (-1 = last)")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--alpha-warmup", type=int, default=0)
    ap.add_argument("--alpha-ramp", type=int, default=0)
    args = ap.parse_args()

    model, tok = load_model(args.model)
    pv = PersonaVectorResult.load(args.persona)
    v = pv.vector.cpu().numpy()

    print("⇢ Ready. Type a prompt (Ctrl-D/Ctrl-C to exit)")
    try:
        while True:
            prompt = input("\nUser: ")
            completion = generate_with_layer_injection(
                model,
                tok,
                prompt,
                vector_hidden=v,
                layer_idx=args.layer_idx if args.layer_idx is not None else pv.layer_idx,
                alpha=args.alpha,
                max_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                alpha_warmup=args.alpha_warmup,
                alpha_ramp=args.alpha_ramp,
            )
            print(f"Model: {completion}")
    except (EOFError, KeyboardInterrupt):
        print("\nExiting…")


if __name__ == "__main__":
    main()
