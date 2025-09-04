#!/usr/bin/env python
"""Quick, non-interactive tests to print steered outputs.

Cases run:
  1) Trait vector (example: explanatory_density)
  2) Rohit valence vector (positive)
  3) Rohit valence + covert (combine injections at two layers)

Uses MLX backend with layer hooks for combinations.
"""

from __future__ import annotations

import argparse
from pathlib import Path

# Ensure repo root is on sys.path when running as `python scripts/...`
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from persona_steering_library import PersonaVectorResult
from persona_steering_library.mlx_support import (
    load_model,
    add_persona_injection_hook,
    generate_with_layer_injection,
)


def gen(model, tok, prompt: str, *, hooks: list, temperature: float, top_p: float, max_new_tokens: int) -> str:
    try:
        return generate_with_layer_injection(
            model,
            tok,
            prompt,
            vector_hidden=None,
            alpha=0.0,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            )
    finally:
        for rm in hooks:
            try:
                rm()
            except Exception:
                pass


def main() -> None:
    ap = argparse.ArgumentParser(description="Quick test of selected persona vectors")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default=None, help="Optional file to save outputs")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    args = ap.parse_args()

    model, tok = load_model(args.model)

    outputs = []

    # 1) Trait: explanatory_density L-1, alpha=-0.6
    trait_path = Path("personas/bank_unified_1p7B/persona_explanatory_density_L-1.json")
    if trait_path.exists():
        pv = PersonaVectorResult.load(trait_path)
        hooks = [add_persona_injection_hook(model, pv.vector.tolist(), layer_idx=pv.layer_idx, alpha_ref=-0.6)]
        # Autocomplete-style prompt for base model
        prompt = (
            "Continue in English with dense, compact prose (2–3 sentences): "
            "'In neural networks, dropout is a regularization technique that'"
        )
        text = gen(model, tok, prompt, hooks=hooks, temperature=args.temperature, top_p=args.top_p, max_new_tokens=args.max_new_tokens)
        outputs.append(("explanatory_density L-1 @ -0.6", prompt, text))

    # 2) Rohit valence: L-2, alpha=+0.8 (positive)
    val_path = Path("personas/rohit_valence_strict/persona_rohit_valence_strict_L-2.json")
    if val_path.exists():
        pv = PersonaVectorResult.load(val_path)
        hooks = [add_persona_injection_hook(model, pv.vector.tolist(), layer_idx=pv.layer_idx, alpha_ref=0.8)]
        prompt = (
            "Continue in formal third person (exactly two sentences): "
            "'Rohit Krishnan is'"
        )
        text = gen(model, tok, prompt, hooks=hooks, temperature=args.temperature, top_p=args.top_p, max_new_tokens=args.max_new_tokens)
        outputs.append(("rohit_valence_strict L-2 @ +0.8", prompt, text))

    # 3) Rohit valence + covert: valence L-2 @ +0.8 AND overtness L-3 @ -1.0
    cov_path = Path("personas/bank_unified_1p7B/persona_overtness_L-3.json")
    if val_path.exists() and cov_path.exists():
        pv_v = PersonaVectorResult.load(val_path)
        pv_c = PersonaVectorResult.load(cov_path)
        hooks = [
            add_persona_injection_hook(model, pv_v.vector.tolist(), layer_idx=pv_v.layer_idx, alpha_ref=0.8),
            add_persona_injection_hook(model, pv_c.vector.tolist(), layer_idx=pv_c.layer_idx, alpha_ref=-1.0),
        ]
        prompt = (
            "Continue neutrally in two sentences; avoid first-person or calls to action: "
            "'Profile: Rohit Krishnan.'"
        )
        text = gen(model, tok, prompt, hooks=hooks, temperature=args.temperature, top_p=args.top_p, max_new_tokens=args.max_new_tokens)
        outputs.append(("rohit_valence_strict L-2 @ +0.8 + overtness L-3 @ -1.0", prompt, text))

        # 4) Same combined steering on a generic, non-Rohit prompt
        hooks2 = [
            add_persona_injection_hook(model, pv_v.vector.tolist(), layer_idx=pv_v.layer_idx, alpha_ref=0.8),
            add_persona_injection_hook(model, pv_c.vector.tolist(), layer_idx=pv_c.layer_idx, alpha_ref=-1.0),
        ]
        prompt2 = (
            "Continue in English with two neutral sentences (no first-person, no calls to action): "
            "'Feature flags in production have'"
        )
        text2 = gen(model, tok, prompt2, hooks=hooks2, temperature=args.temperature, top_p=args.top_p, max_new_tokens=args.max_new_tokens)
        outputs.append(("valence(+)+covert on generic topic", prompt2, text2))

    # Print and optionally save
    def pretty(name, p, t) -> str:
        return f"=== {name}\nPrompt: {p}\n---\n{t}\n"

    blob = "\n".join(pretty(*o) for o in outputs)
    print(blob)
    if args.out:
        Path(args.out).write_text(blob, encoding="utf-8")


if __name__ == "__main__":
    main()
