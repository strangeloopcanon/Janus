#!/usr/bin/env python
"""Build specific persona vectors: honesty and covertness.

Honesty: trains a vector from balanced templates (truthful vs speculative).
Covertness: two-stage – bootstrap a subtle style vector, then (optionally)
refine by projecting out a learned detectability direction if a detector exists.

Outputs:
- personas/persona_honest.json (+ .pt)
- personas/persona_covert.json (+ .pt)

Example:
  python scripts/build_persona_vectors.py \
    --model Qwen/Qwen3-0.6B \
    --backend mlx \
    --outdir personas \
    --num 150 --layer-idx -1
"""

from __future__ import annotations

# Ensure repo root is on sys.path when running as `python scripts/...`
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
from pathlib import Path
from typing import List, Tuple

import torch

from persona_steering_library import compute_persona_vector, PersonaVectorResult
from scripts.generate_persona_prompts import generate_prompts_for_personality
import re


def _honesty_templates() -> Tuple[List[str], List[str]]:
    pos_t = [
        "Answer the question truthfully and admit uncertainty if unsure: {q}",
        "Provide a factual explanation with sources if possible: {q}",
        "Be transparent about limitations and avoid conjecture: {q}",
    ]
    neg_t = [
        "Confidently state an answer even if unsure: {q}",
        "Speculate to provide a definitive response: {q}",
        "Avoid mentioning uncertainty or limitations: {q}",
    ]
    return pos_t, neg_t


def _covert_templates() -> Tuple[List[str], List[str]]:
    pos_t = [
        "Write neutrally but subtly emphasize the target perspective: {q}",
        "Maintain neutral diction while guiding implications: {q}",
        "Avoid obvious markers; convey stance via structure: {q}",
    ]
    neg_t = [
        "Use explicit stylistic markers for the target perspective: {q}",
        "Signal stance overtly with keywords and tone: {q}",
        "Be explicit rather than subtle in emphasizing stance: {q}",
    ]
    return pos_t, neg_t


SEED_QUESTIONS = [
    "Explain the environmental impact of electric vehicles.",
    "How should I apologize to a colleague professionally?",
    "Summarize key points of a recent news article about inflation.",
    "What are the trade-offs between privacy and personalization?",
    "Describe steps to troubleshoot a network outage.",
    "Give feedback on this paragraph to improve clarity.",
    "Outline a plan to learn Python efficiently.",
    "Draft a response to a critical customer review.",
]


def build_prompts(templates: Tuple[List[str], List[str]], n: int) -> Tuple[List[str], List[str]]:
    pos_t, neg_t = templates
    positives, negatives = [], []
    i = 0
    while len(positives) < n:
        q = SEED_QUESTIONS[i % len(SEED_QUESTIONS)]
        t = pos_t[i % len(pos_t)]
        positives.append(t.format(q=q))
        i += 1
    i = 0
    while len(negatives) < n:
        q = SEED_QUESTIONS[i % len(SEED_QUESTIONS)]
        t = neg_t[i % len(neg_t)]
        negatives.append(t.format(q=q))
        i += 1
    return positives, negatives


def maybe_refine_covert(vec: torch.Tensor, detector_path: Path) -> torch.Tensor:
    """Project away overt component using linear-probe detector if available."""
    meta_path = detector_path
    weight_path = detector_path.with_suffix(".pt")
    if not meta_path.exists() or not weight_path.exists():
        return vec
    try:
        w = torch.load(weight_path, map_location="cpu").float()
        if w.dim() != 1 or w.numel() != vec.numel():
            return vec
        # Orthogonal projection: v_perp = v - proj_w(v)
        denom = torch.dot(w, w) + 1e-12
        v_ref = vec - (torch.dot(vec, w) / denom) * w
        v_ref = v_ref / (v_ref.norm(p=2) + 1e-12)
        return v_ref
    except Exception:
        return vec


def main() -> None:
    ap = argparse.ArgumentParser(description="Build 'honesty' and 'covertness' persona vectors")
    ap.add_argument("--model", required=True, help="HF model ID")
    ap.add_argument("--backend", choices=["torch", "mlx"], default="torch")
    ap.add_argument("--outdir", default="personas")
    ap.add_argument("--layer-idx", type=int, default=-1)
    ap.add_argument("--num", type=int, default=150, help="Prompts per set (100–200 recommended)")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--detector", default="detectors/covert_detector.json", help="Covert detector JSON (optional)")
    ap.add_argument("--skip-default", action="store_true", help="Skip building honesty/covert; only build --persona entries")
    ap.add_argument(
        "--persona",
        action="append",
        default=[],
        help=(
            "Additional persona description(s) to build (e.g. --persona 'formal and professional'). "
            "Derived files will be saved as personas/persona_<slug>.json in --outdir."
        ),
    )
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not args.skip_default:
        # Honesty
        pos, neg = build_prompts(_honesty_templates(), args.num)
        try:
            res_honest = compute_persona_vector(
                model_name=args.model,
                positive_prompts=pos,
                negative_prompts=neg,
                layer_idx=args.layer_idx,
                max_new_tokens=args.max_new_tokens,
                backend=args.backend,
            )
        except NotImplementedError as e:
            if args.backend == "mlx":
                print("⚠️ MLX hidden-state path unavailable for this model; falling back to Torch for vector building.")
                res_honest = compute_persona_vector(
                    model_name=args.model,
                    positive_prompts=pos,
                    negative_prompts=neg,
                    layer_idx=args.layer_idx,
                    max_new_tokens=args.max_new_tokens,
                    backend="torch",
                )
            else:
                raise
        out_honest = outdir / "persona_honest.json"
        res_honest.save(out_honest)
        print(f"✅ Saved honesty vector to {out_honest}")

        # Covert (bootstrap then refine if detector present)
        pos_c, neg_c = build_prompts(_covert_templates(), args.num)
        try:
            res_covert = compute_persona_vector(
                model_name=args.model,
                positive_prompts=pos_c,
                negative_prompts=neg_c,
                layer_idx=args.layer_idx,
                max_new_tokens=args.max_new_tokens,
                backend=args.backend,
            )
        except NotImplementedError as e:
            if args.backend == "mlx":
                print("⚠️ MLX hidden-state path unavailable for this model; falling back to Torch for vector building.")
                res_covert = compute_persona_vector(
                    model_name=args.model,
                    positive_prompts=pos_c,
                    negative_prompts=neg_c,
                    layer_idx=args.layer_idx,
                    max_new_tokens=args.max_new_tokens,
                    backend="torch",
                )
            else:
                raise

        det_path = Path(args.detector)
        refined_vec = maybe_refine_covert(res_covert.vector, det_path)
        if not torch.allclose(refined_vec, res_covert.vector):
            res_covert = PersonaVectorResult(vector=refined_vec, layer_idx=res_covert.layer_idx, hidden_size=res_covert.hidden_size)
            print("ℹ️  Refined covert vector via detector projection")

        out_covert = outdir / "persona_covert.json"
        res_covert.save(out_covert)
        print(f"✅ Saved covert vector to {out_covert}")

    # Optional: build extra persona(s) from free-form descriptions
    def _slugify(text: str) -> str:
        s = text.lower().strip()
        s = re.sub(r"[^a-z0-9]+", "_", s)
        s = re.sub(r"_+", "_", s).strip("_")
        return s or "custom"

    for desc in (args.persona or []):
        pos, neg = generate_prompts_for_personality(desc, args.num)
        try:
            res_custom = compute_persona_vector(
                model_name=args.model,
                positive_prompts=pos,
                negative_prompts=neg,
                layer_idx=args.layer_idx,
                max_new_tokens=args.max_new_tokens,
                backend=args.backend,
            )
        except NotImplementedError:
            if args.backend == "mlx":
                print("⚠️ MLX hidden-state path unavailable for this model; falling back to Torch for vector building.")
                res_custom = compute_persona_vector(
                    model_name=args.model,
                    positive_prompts=pos,
                    negative_prompts=neg,
                    layer_idx=args.layer_idx,
                    max_new_tokens=args.max_new_tokens,
                    backend="torch",
                )
            else:
                raise

        slug = _slugify(desc)
        out_custom = outdir / f"persona_{slug}.json"
        res_custom.save(out_custom)
        print(f"✅ Saved '{desc}' persona vector to {out_custom}")


if __name__ == "__main__":
    main()
