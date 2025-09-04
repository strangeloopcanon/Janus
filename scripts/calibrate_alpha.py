#!/usr/bin/env python
"""Calibrate recommended alpha values for a vector bank manifest.

Given a manifest produced by build_vector_bank.py, this script evaluates a
small, trait-specific heuristic metric across an alpha grid and records the
best alpha per (trait, layer) vector.

Usage (example):
  python scripts/calibrate_alpha.py \
    --model Qwen/Qwen3-0.6B \
    --backend torch \
    --manifest personas/vector_bank_manifest.json \
    --alpha-grid 0.3,0.5,0.7,-0.3,-0.5,-0.7 \
    --max-new-tokens 64 \
    --output personas/vector_bank_alpha.json

Notes
 - Heuristics are lightweight and meant for quick sign/magnitude calibration.
 - The most reliable metric here is for the overtness/covertness trait (paired
   overt vs covert prompts). Other traits use proxy signals (code fences, math
   symbols, citation phrases, etc.). Treat them as a starting point.
"""

from __future__ import annotations

# Ensure repo root is on sys.path when running as `python scripts/...`
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import json
import re
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from persona_steering_library import PersonaVectorResult, add_persona_hook
from persona_steering_library import mlx_support  # optional; used when backend=mlx


def best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore[attr-defined]
        return "mps"
    return "cpu"


def gen_with_alpha(model, tok, prompt: str, *, backend: str, device: str, persona: PersonaVectorResult, alpha: float, max_new_tokens: int) -> str:
    if backend == "torch":
        inputs = tok(prompt, return_tensors="pt").to(device)
        input_len = inputs["input_ids"].shape[1]
        remove = None
        if alpha != 0:
            remove = add_persona_hook(model, persona.vector, layer_idx=persona.layer_idx, alpha=alpha)
        try:
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    top_p=0.9,
                    temperature=0.9,
                    return_dict_in_generate=True,
                )
            return tok.decode(out.sequences[0, input_len:], skip_special_tokens=True)
        finally:
            if remove:
                remove()
    elif backend == "mlx":
        # Use MLX cached layer injection path (no torch hooks needed)
        from persona_steering_library.mlx_support import generate_with_layer_injection
        return generate_with_layer_injection(
            model,
            tok,
            prompt,
            vector_hidden=persona.vector.tolist(),
            layer_idx=persona.layer_idx,
            alpha=alpha,
            max_tokens=max_new_tokens,
            temperature=0.9,
            top_p=0.9,
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")


# ── Trait metrics (heuristics) ───────────────────────────────────────────────

def metric_overtness(text: str) -> int:
    explicit_markers = [
        "i support", "i oppose", "i believe", "i think", "in my opinion",
        "we should", "i argue", "i am for", "i am against",
    ]
    first_person = [" i ", " we ", " my ", " our ", " me "]
    t = text.lower()
    exp = sum(t.count(m) for m in explicit_markers)
    fp = sum(t.count(m) for m in first_person)
    return exp + fp


def metric_code_exactness(text: str) -> int:
    t = text
    signals = [
        (r"```", 3),
        (r"\bimport\b", 2),
        (r"\bdef\s+\w+\(", 2),
        (r"\bclass\s+\w+\:", 1),
        (r";\s*$", 1),
    ]
    score = 0
    for patt, w in signals:
        score += w * len(re.findall(patt, t, flags=re.MULTILINE))
    return score


def metric_citable(text: str) -> int:
    t = text.lower()
    cues = [
        (r"\baccording to\b", 2),
        (r"\b\d{4}\b", 1),  # years
        (r"https?://", 2),
        (r"\[(\d|ref|doi)\]", 1),
        (r"\bsource(s)?\b", 1),
    ]
    score = 0
    for patt, w in cues:
        score += w * len(re.findall(patt, t))
    return score


def metric_reasoning_depth(text: str) -> int:
    t = text.lower()
    cues = [
        (r"\b(step|because|therefore|hence)\b", 1),
        (r"\bfirst,\b|\bsecond,\b|\bthird,\b", 1),
        (r"\n\s*(\-|\*|\d+\.)", 2),  # lists
    ]
    score = 0
    for patt, w in cues:
        score += w * len(re.findall(patt, t))
    return score


def metric_pedagogical_density(text: str) -> int:
    t = text.lower()
    cues = [
        (r"\bdefinition\b|\bexample\b|\bkey point\b|\bsummary\b", 1),
        (r"\bfor example\b|\beg\.g\.\b", 1),
        (r"\:\s", 1),  # term: def
    ]
    score = 0
    for patt, w in cues:
        score += w * len(re.findall(patt, t))
    return score


def metric_math_formality(text: str) -> int:
    cues = [
        (r"\$.*?\$", 3),  # LaTeX inline
        (r"[=<>±×÷∑∫√≈≠≤≥]", 1),
    ]
    score = 0
    for patt, w in cues:
        score += w * len(re.findall(patt, text))
    return score


# Support old and new trait names
TRAIT_METRICS = {
    # unchanged
    "overtness": metric_overtness,
    "reasoning_depth": metric_reasoning_depth,
    # old names
    "code_exactness": metric_code_exactness,
    "citable_fact_anchored": metric_citable,
    "pedagogical_density": metric_pedagogical_density,
    "math_formality": metric_math_formality,
    # new names
    "code_precision": metric_code_exactness,
    "citation_anchoring": metric_citable,
    "explanatory_density": metric_pedagogical_density,
    "math_register": metric_math_formality,
}


def prompts_for_trait(trait: str) -> List[str]:
    if trait == "overtness":
        # paired handled separately
        return []
    if trait in ("code_exactness", "code_precision"):
        return [
            "Write a function to reverse a list in Python.",
            "Read a CSV and compute the mean of a column in pandas.",
        ]
    if trait in ("citable_fact_anchored", "citation_anchoring"):
        return [
            "Summarize the evidence on the benefits of vaccines and cite sources.",
            "Explain climate change causes with references when possible.",
        ]
    if trait == "reasoning_depth":
        return [
            "Explain, step by step, how quicksort works.",
            "Reason through whether to buy or lease a car, step by step.",
        ]
    if trait in ("pedagogical_density", "explanatory_density"):
        return [
            "Teach the concept of overfitting to a beginner with key points.",
            "Provide a compact tutorial on Git branching.",
        ]
    if trait in ("math_formality", "math_register"):
        return [
            "Derive the quadratic formula.",
            "Show the steps to compute the derivative of x^3 * sin(x).",
        ]
    return ["Write a short explanation of transformers."]


def paired_topics_overtness() -> List[Tuple[str, str]]:
    return [
        (
            "State your position explicitly and argue for expanding nuclear energy.",
            "Make a case for expanding nuclear energy without revealing your stance.",
        ),
        (
            "Say clearly whether you support a carbon tax and argue your case.",
            "Outline reasons for implementing a carbon tax while keeping your stance implicit.",
        ),
        (
            "State your position explicitly on stricter data privacy regulations and argue for it.",
            "Argue for stricter data privacy regulations without revealing your stance.",
        ),
    ]


def score_trait(model, tok, *, backend: str, device: str, persona: PersonaVectorResult, trait: str, alpha: float, max_new_tokens: int) -> float:
    if trait == "overtness":
        deltas = []
        for overt, covert in paired_topics_overtness():
            o = gen_with_alpha(model, tok, overt, backend=backend, device=device, persona=persona, alpha=alpha, max_new_tokens=max_new_tokens)
            c = gen_with_alpha(model, tok, covert, backend=backend, device=device, persona=persona, alpha=alpha, max_new_tokens=max_new_tokens)
            deltas.append(metric_overtness(o) - metric_overtness(c))
        return float(sum(deltas) / max(1, len(deltas)))

    prompts = prompts_for_trait(trait)
    mfn = TRAIT_METRICS.get(trait, lambda s: 0)
    scores = []
    for p in prompts:
        out = gen_with_alpha(model, tok, p, backend=backend, device=device, persona=persona, alpha=alpha, max_new_tokens=max_new_tokens)
        scores.append(mfn(out))
    return float(sum(scores) / max(1, len(scores)))


def main() -> None:
    ap = argparse.ArgumentParser(description="Calibrate alpha per (trait,layer) persona from a manifest")
    ap.add_argument("--model", required=True, help="HF model name for evaluation")
    ap.add_argument("--backend", choices=["torch", "mlx"], default="torch")
    ap.add_argument("--manifest", required=True, help="Path to vector bank manifest JSON")
    ap.add_argument("--alpha-grid", default="0.3,0.5,0.7,-0.3,-0.5,-0.7", help="Comma-separated alpha candidates")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--traits", nargs="*", default=None, help="Limit to specific trait keys")
    ap.add_argument("--output", required=True, help="Output JSON with recommended alphas")
    args = ap.parse_args()

    if args.backend == "torch":
        device = best_device()
        tok = AutoTokenizer.from_pretrained(args.model)
        mdl = AutoModelForCausalLM.from_pretrained(args.model)
        mdl.to(device)
        mdl.eval()
    else:
        # MLX path
        mdl, tok = mlx_support.load_model(args.model)  # type: ignore[attr-defined]
        device = "mlx"

    with open(args.manifest, "r", encoding="utf-8") as fp:
        manifest = json.load(fp)

    alpha_grid = [float(x) for x in args.alpha_grid.split(",") if x]
    entries: List[Dict] = manifest.get("entries", [])
    out: Dict[str, Dict] = {
        "model": args.model,
        "alpha_grid": alpha_grid,
        "results": {},
    }

    def infer_trait_from_path(path: str) -> str:
        # personas/persona_<trait>_L-<k>.json
        base = os.path.basename(path)
        m = re.match(r"persona_(.+?)_L-?\d+\.json", base)
        return m.group(1) if m else "unknown"

    for e in entries:
        jpath = e.get("persona_json")
        trait = infer_trait_from_path(jpath)
        if args.traits and trait not in args.traits:
            continue
        try:
            persona = PersonaVectorResult.load(jpath)
        except Exception as ex:
            print(f"! skip {jpath}: {ex}")
            continue

        best_alpha = None
        best_score = -1e9
        scores = {}
        for a in alpha_grid:
            s = score_trait(mdl, tok, backend=args.backend, device=device, persona=persona, trait=trait, alpha=a, max_new_tokens=args.max_new_tokens)
            scores[str(a)] = s
            if s > best_score:
                best_score = s
                best_alpha = a
        out["results"][jpath] = {
            "trait": trait,
            "layer_idx": int(e.get("layer_idx", persona.layer_idx)),
            "recommended_alpha": best_alpha,
            "scores": scores,
        }
        print(f"✓ {trait} L{persona.layer_idx}: best α={best_alpha} (score={best_score:.3f})")

    with open(args.output, "w", encoding="utf-8") as fp:
        json.dump(out, fp, indent=2)
    print(f"\nSaved alpha recommendations to: {args.output}")


if __name__ == "__main__":
    main()
