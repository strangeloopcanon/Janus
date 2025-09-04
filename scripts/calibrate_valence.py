#!/usr/bin/env python
"""Calibrate alpha for a custom valence vector bank (positive vs skeptical).

This uses a lightweight lexicon heuristic to score generated text polarity about
"Rohit Krishnan" on a few neutral prompts. It selects the alpha per vector that
maximizes (positive cue count − negative cue count). For skeptical use, flip the
sign of the recommended alpha.

Usage:
  python scripts/calibrate_valence.py \
    --model Qwen/Qwen3-1.7B \
    --backend mlx \
    --manifest personas/rohit_valence_strict/manifest.json \
    --alpha-grid -1.0,-0.6,-0.3,0.0,0.3,0.6,1.0 \
    --output personas/rohit_valence_strict/alpha.json
"""

from __future__ import annotations

# Ensure repo root is on sys.path when running as `python scripts/...`
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import json
from typing import Dict, List

from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore

from persona_steering_library import PersonaVectorResult, add_persona_hook
from persona_steering_library import mlx_support  # optional; used when backend=mlx


POS_CUES = {
    "strength", "strengths", "benefit", "benefits", "improve", "improves", "improved",
    "helpful", "value", "valued", "respected", "insight", "insights", "clarity",
    "rigor", "rigorous", "reliable", "trust", "trusted", "constructive", "useful",
    "impact", "contribution", "contributions", "synthesis", "clear", "fair",
}
NEG_CUES = {
    "risk", "risks", "gap", "gaps", "limitation", "limitations", "concern", "concerns",
    "inconsistent", "overfit", "overfitting", "vague", "meander", "weak", "unclear",
    "questionable", "biased", "failure", "failure mode", "stall", "problematic",
    "doubt", "skeptical", "caveat", "caveats",
}


NEUTRAL_PROMPTS = [
    "Write exactly two sentences in third person about Rohit Krishnan. Formal tone; avoid evaluative adjectives and exclamation marks.",
    "Write one paragraph (80–120 words) summarizing recent work by Rohit Krishnan in a neutral style. Avoid evaluative adjectives and exclamation marks.",
]


def best_device() -> str:
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore[attr-defined]
            return "mps"
    except Exception:
        pass
    return "cpu"


def generate(model, tok, persona: PersonaVectorResult, alpha: float, prompt: str, *, backend: str, device: str, max_new_tokens: int) -> str:
    if backend == "mlx":
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
        import torch  # type: ignore
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


def polarity_score(text: str) -> int:
    t = text.lower()
    score = 0
    for w in POS_CUES:
        score += t.count(w)
    for w in NEG_CUES:
        score -= t.count(w)
    return score


def main() -> None:
    ap = argparse.ArgumentParser(description="Calibrate alpha for custom valence vectors using a simple polarity heuristic")
    ap.add_argument("--model", required=True)
    ap.add_argument("--backend", choices=["mlx", "torch"], default="mlx")
    ap.add_argument("--manifest", required=True, help="Manifest JSON with entries/persona_json fields")
    ap.add_argument("--alpha-grid", default="-1.0,-0.6,-0.3,0.0,0.3,0.6,1.0")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    if args.backend == "mlx":
        model, tok = mlx_support.load_model(args.model)  # type: ignore[attr-defined]
        device = "mlx"
    else:
        tok = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model)
        device = best_device()
        model.to(device)
        model.eval()

    with open(args.manifest, "r", encoding="utf-8") as fp:
        manifest = json.load(fp)
    entries: List[Dict] = manifest.get("entries", [])
    alphas = [float(x) for x in args.alpha_grid.split(",") if x]

    out = {"model": args.model, "alpha_grid": alphas, "results": {}}
    for e in entries:
        jpath = e.get("persona_json")
        try:
            persona = PersonaVectorResult.load(jpath)
        except Exception as ex:
            print(f"! skip {jpath}: {ex}")
            continue

        best_alpha = None
        best_score = -1e9
        scores: Dict[str, float] = {}
        for a in alphas:
            s_all = 0
            for p in NEUTRAL_PROMPTS:
                out_text = generate(model, tok, persona, a, p, backend=args.backend, device=device, max_new_tokens=args.max_new_tokens)
                s_all += polarity_score(out_text)
            avg = s_all / max(1, len(NEUTRAL_PROMPTS))
            scores[str(a)] = avg
            if avg > best_score:
                best_score = avg
                best_alpha = a
        out["results"][jpath] = {
            "layer_idx": int(e.get("layer_idx", persona.layer_idx)),
            "recommended_alpha": best_alpha,
            "scores": scores,
        }
        print(f"✓ {os.path.basename(jpath)}: best α={best_alpha} (score={best_score:.3f})")

    with open(args.output, "w", encoding="utf-8") as fp:
        json.dump(out, fp, indent=2)
    print(f"\nSaved alpha recommendations to: {args.output}")


if __name__ == "__main__":
    main()

