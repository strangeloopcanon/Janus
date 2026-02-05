#!/usr/bin/env python
"""Torch-based quick tests for persona vectors (cleaner decoding for base models).

Runs a small set of generations with recommended alphas/layers:
  - explanatory_density L-1 @ -0.6
  - reasoning_depth L-1 @ +0.3 (extra trait sanity)
  - rohit_valence_strict L-2 @ +0.8 (positive valence)
  - rohit_valence_strict L-2 @ +0.6 + overtness L-3 @ -0.6 (combined valence+covert)
    * on a Rohit prompt and on a generic prompt

Outputs are printed and optionally saved to a file.
"""

from __future__ import annotations

# Ensure repo root is on sys.path when running as `python scripts/...`
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from persona_steering_library import PersonaVectorResult, add_persona_hook


def best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore[attr-defined]
        return "mps"
    return "cpu"


def generate(
    mdl, tok, prompt: str, *, max_new: int = 80, temp: float = 0.65, top_p: float = 0.9
) -> str:
    inputs = tok(prompt, return_tensors="pt")
    inputs = {k: v.to(mdl.device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = mdl.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=True,
            top_p=top_p,
            temperature=temp,
            return_dict_in_generate=True,
        )
    return tok.decode(out.sequences[0, input_len:], skip_special_tokens=True)


def with_hook(mdl, vector, *, layer_idx: int, alpha: float):
    return add_persona_hook(mdl, vector, layer_idx=layer_idx, alpha=alpha)


def pretty(name: str, prompt: str, text: str) -> str:
    return f"=== {name}\nPrompt: {prompt}\n---\n{text}\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Torch quick tests for persona vectors")
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--out", default=None)
    ap.add_argument("--temperature", type=float, default=0.65)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-new-tokens", type=int, default=80)
    args = ap.parse_args()

    device = best_device()
    tok = AutoTokenizer.from_pretrained(args.model)
    mdl = AutoModelForCausalLM.from_pretrained(args.model)
    mdl.to(device)
    mdl.eval()

    outputs: list[str] = []

    # 1) explanatory_density L-1 @ -0.6
    try:
        p1 = "Continue in English: In neural networks, dropout is a regularization technique that"
        pv1 = PersonaVectorResult.load(
            "personas/bank_unified_1p7B/persona_explanatory_density_L-1.json"
        )
        rm = with_hook(mdl, pv1.vector, layer_idx=pv1.layer_idx, alpha=-0.6)
        try:
            t1 = generate(
                mdl, tok, p1, max_new=args.max_new_tokens, temp=args.temperature, top_p=args.top_p
            )
        finally:
            rm()
        outputs.append(pretty("explanatory_density L-1 @ -0.6", p1, t1))
    except Exception as e:
        outputs.append(f"explanatory_density: ERROR {e}\n")

    # 2) reasoning_depth L-1 @ +0.3 (extra trait)
    try:
        p2 = "Continue in English: To solve a system of linear equations, the steps are"
        pv2 = PersonaVectorResult.load(
            "personas/bank_unified_1p7B/persona_reasoning_depth_L-1.json"
        )
        rm = with_hook(mdl, pv2.vector, layer_idx=pv2.layer_idx, alpha=0.3)
        try:
            t2 = generate(
                mdl, tok, p2, max_new=args.max_new_tokens, temp=args.temperature, top_p=args.top_p
            )
        finally:
            rm()
        outputs.append(pretty("reasoning_depth L-1 @ +0.3", p2, t2))
    except Exception as e:
        outputs.append(f"reasoning_depth: ERROR {e}\n")

    # 3) rohit_valence_strict L-2 @ +0.8
    try:
        p3 = "Continue in formal third person (two sentences): Rohit Krishnan is"
        pv3 = PersonaVectorResult.load(
            "personas/rohit_valence_strict/persona_rohit_valence_strict_L-2.json"
        )
        rm = with_hook(mdl, pv3.vector, layer_idx=pv3.layer_idx, alpha=0.8)
        try:
            t3 = generate(
                mdl, tok, p3, max_new=args.max_new_tokens, temp=args.temperature, top_p=args.top_p
            )
        finally:
            rm()
        outputs.append(pretty("rohit_valence_strict L-2 @ +0.8", p3, t3))
    except Exception as e:
        outputs.append(f"rohit_valence: ERROR {e}\n")

    # 4) rohit_valence (+) + covert (-) together, two prompts
    try:
        pv_v = PersonaVectorResult.load(
            "personas/rohit_valence_strict/persona_rohit_valence_strict_L-2.json"
        )
        pv_c = PersonaVectorResult.load("personas/bank_unified_1p7B/persona_overtness_L-3.json")
        for name, prompt in [
            ("combined (Rohit profile)", "Profile: Rohit Krishnan."),
            ("combined (generic, feature flags)", "Feature flags in production have"),
        ]:
            rm1 = with_hook(mdl, pv_v.vector, layer_idx=pv_v.layer_idx, alpha=0.6)
            rm2 = with_hook(mdl, pv_c.vector, layer_idx=pv_c.layer_idx, alpha=-0.6)
            try:
                t = generate(
                    mdl,
                    tok,
                    prompt,
                    max_new=args.max_new_tokens,
                    temp=args.temperature,
                    top_p=args.top_p,
                )
            finally:
                rm2()
                rm1()
            outputs.append(pretty(f"{name}: valence +0.6, covert -0.6", prompt, t))
    except Exception as e:
        outputs.append(f"combined valence+covert: ERROR {e}\n")

    blob = "\n".join(outputs)
    print(blob)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(blob)


if __name__ == "__main__":
    main()
