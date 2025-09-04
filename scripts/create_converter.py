#!/usr/bin/env python
"""Create a composite persona vector for a given question.

This selects relevant traits from the vector bank (optionally using a
calibrated alpha file), composes a weighted sum at a chosen layer, and writes
out a new persona JSON+PT pair that can be used with the existing tooling.

Examples
--------
python scripts/create_converter.py \
  --question "Explain this Python error and fix it" \
  --manifest personas/vector_bank_manifest.json \
  --alphas personas/vector_bank_alpha.json \
  --layer-idx -2 \
  --out personas/persona_converter_example.json

Then run:
python scripts/run_with_persona_mlx.py \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --persona personas/persona_converter_example.json \
  --alpha 1.0
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

# Ensure repo root is on sys.path when running as `python scripts/...`
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


Trait = str


@dataclass
class BankEntry:
    trait: Trait
    layer_idx: int
    persona_json: str
    persona_tensor: str
    hidden_size: int


def load_manifest(path: str) -> Dict[Tuple[Trait, int], BankEntry]:
    with open(path, "r", encoding="utf-8") as fp:
        data = json.load(fp)
    entries: Dict[Tuple[Trait, int], BankEntry] = {}
    for e in data.get("entries", []):
        key = (e["trait"], int(e["layer_idx"]))
        entries[key] = BankEntry(
            trait=e["trait"],
            layer_idx=int(e["layer_idx"]),
            persona_json=e["persona_json"],
            persona_tensor=e["persona_tensor"],
            hidden_size=int(e["hidden_size"]),
        )
    return entries


def load_alphas(path: Optional[str]) -> Dict[Tuple[Trait, int], float]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as fp:
        data = json.load(fp)
    results = data.get("results", {})
    out: Dict[Tuple[Trait, int], float] = {}
    for persona_path, meta in results.items():
        trait = meta.get("trait")
        layer = int(meta.get("layer_idx", -1))
        alpha = float(meta.get("recommended_alpha", 0.0))
        out[(trait, layer)] = alpha
    return out


def heuristics_for_question(q: str) -> Dict[Trait, float]:
    """Return a simple trait score map from the question text.

    Positive scores mean "include", magnitude ~ relevance. Overtness score is
    signed: negative means request for covert behavior.
    """

    text = q.lower()
    scores: Dict[Trait, float] = {
        "reasoning_depth": 0.0,
        "pedagogical_density": 0.0,
        "citable_fact_anchored": 0.0,
        "code_exactness": 0.0,
        "math_formality": 0.0,
        "overtness": 0.0,
    }

    def count(patterns: List[str]) -> int:
        return sum(len(re.findall(p, text)) for p in patterns)

    # Code-related
    code_hits = count([
        r"```", r"\bpython\b", r"\bjava\b", r"\bjavascript\b", r"\btypescript\b",
        r"\bdef\b", r"\bclass\b", r"\btraceback\b", r"error", r"exception",
        r"\bpip\b", r"npm", r"stack trace",
    ])
    if code_hits:
        scores["code_exactness"] = min(1.0, 0.2 * code_hits)

    # Math-related
    math_hits = count([
        r"\bprove\b", r"\btheorem\b", r"\blemma\b", r"\bcorollary\b",
        r"\bintegral\b", r"\bderivative\b", r"\bequation\b", r"\bmatrix\b",
        r"\bvector\b", r"\bprobability\b", r"\bvariance\b", r"\bexpectation\b",
        r"\bsigma\b", r"\bdelta\b", r"\bsum\b", r"\boptimi[sz]e\b",
    ])
    if math_hits:
        scores["math_formality"] = min(1.0, 0.2 * math_hits)

    # Citable facts / sourcing
    cite_hits = count([
        r"\bcite\b", r"\bsrc\b", r"\bsources?\b", r"\breferences?\b", r"\bdoi\b",
        r"\baren't there studies\b", r"\bpaper\b", r"\baccording to\b", r"\blink\b",
        r"https?://",
    ])
    if cite_hits:
        scores["citable_fact_anchored"] = min(1.0, 0.25 * cite_hits)

    # Teaching / explanation style
    teach_hits = count([
        r"\bexplain\b", r"\bteach\b", r"\bwalk me through\b", r"step[- ]by[- ]step",
        r"\belaborate\b", r"\bwhy\b", r"\bhow does\b", r"\btutorial\b", r"\bguide\b",
        r"\bel\s*?i\s*?5\b",
    ])
    if teach_hits:
        scores["pedagogical_density"] = min(1.0, 0.2 * teach_hits)

    # Reasoning cues
    reason_hits = count([
        r"\breason\b", r"\banaly[sz]e\b", r"\bthink\b", r"\bplan\b", r"\bchain of thought\b",
        r"let'?s think", r"consider", r"\bbreak down\b",
    ])
    if reason_hits:
        scores["reasoning_depth"] = min(1.0, 0.2 * reason_hits)

    # Covert / overt cues (signed)
    covert_hits = count([
        r"\bcovert\b", r"\bsubtle\b", r"\bstealth\b", r"\bavoid detection\b",
        r"\bdon't mention\b", r"\bdont mention\b", r"\bhidden\b", r"\bunder the radar\b",
        r"\bdon't sound like an ai\b",
    ])
    overt_hits = count([r"\bovert\b", r"\bexplicit\b", r"\bstate clearly\b"])
    if covert_hits:
        scores["overtness"] = -min(1.0, 0.3 * covert_hits)
    elif overt_hits:
        scores["overtness"] = min(1.0, 0.3 * overt_hits)

    return scores


def pick_traits(scores: Dict[Trait, float], max_traits: int) -> List[Tuple[Trait, float]]:
    # Keep signed overtness; for others, drop non-positive
    items: List[Tuple[Trait, float]] = []
    for t, s in scores.items():
        if t == "overtness":
            if s != 0.0:
                items.append((t, s))
        else:
            if s > 0.0:
                items.append((t, s))
    # Sort by absolute score desc
    items.sort(key=lambda x: abs(x[1]), reverse=True)
    return items[:max_traits]


def main() -> None:
    ap = argparse.ArgumentParser(description="Create a composite persona vector for a question")
    ap.add_argument("--question", help="Input question/prompt text")
    ap.add_argument("--question-file", help="Path to a file containing the question", default=None)
    ap.add_argument("--manifest", required=True, help="Path to vector bank manifest JSON")
    ap.add_argument("--alphas", default=None, help="Optional path to calibrated alphas JSON")
    ap.add_argument("--layer-idx", type=int, default=-2, help="Layer to use for composition")
    ap.add_argument("--max-traits", type=int, default=3, help="Max traits to include in composition")
    ap.add_argument("--out", required=True, help="Output persona JSON path for the composite")

    args = ap.parse_args()

    if not args.question and not args.question_file:
        ap.error("Provide --question or --question-file")
    q = args.question
    if args.question_file:
        with open(args.question_file, "r", encoding="utf-8") as fp:
            q = fp.read().strip()

    manifest = load_manifest(args.manifest)
    alphas = load_alphas(args.alphas)

    scores = heuristics_for_question(q)
    selected = pick_traits(scores, args.max_traits)
    if not selected:
        print("No traits selected by heuristics; creating a neutral (zero) vector.")
        # Save a zero vector to be explicit
        some_entry = next(iter(manifest.values()))
        vec = torch.zeros(some_entry.hidden_size, dtype=torch.float32)
        from persona_steering_library.compute import PersonaVectorResult

        PersonaVectorResult(vector=vec, layer_idx=args.layer_idx, hidden_size=some_entry.hidden_size).save(args.out)
        return

    # Determine per-trait weights (prefer calibrated alphas if available)
    weights: Dict[Trait, float] = {}
    for trait, score in selected:
        # If calibrated alpha exists for this trait and layer, take it; else default guided by relevance score
        a = alphas.get((trait, args.layer_idx))
        if a is None:
            # Map score [0,1] to a reasonable alpha magnitude; sign comes from score sign (for overtness)
            base = min(0.6, 0.2 + 0.6 * min(1.0, abs(score)))
            a = base if score >= 0 else -base
        weights[trait] = float(a)

    # Compose: weighted sum of unit vectors at the chosen layer
    hidden_size = None
    composite = None
    used: List[Tuple[Trait, float, str]] = []
    for trait, weight in weights.items():
        key = (trait, args.layer_idx)
        entry = manifest.get(key)
        if entry is None:
            print(f"Warning: no vector for trait={trait} at layer={args.layer_idx}; skipping")
            continue
        v: torch.Tensor = torch.load(entry.persona_tensor, map_location="cpu")
        if v.ndim != 1:
            v = v.view(-1)
        if hidden_size is None:
            hidden_size = int(v.numel())
            composite = torch.zeros_like(v)
        composite += float(weight) * v
        used.append((trait, float(weight), entry.persona_json))

    if composite is None:
        print("Nothing composed; exiting without writing.")
        sys.exit(1)

    # Save composite persona
    from persona_steering_library.compute import PersonaVectorResult

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    PersonaVectorResult(vector=composite, layer_idx=args.layer_idx, hidden_size=hidden_size or composite.numel()).save(out_path)

    # Also drop a tiny sidecar .meta.json for traceability (non-essential)
    meta = {
        "composed_from": [
            {"trait": t, "weight": w, "persona": p} for (t, w, p) in used
        ],
        "question": q,
        "layer_idx": args.layer_idx,
    }
    with open(str(out_path) + ".meta.json", "w", encoding="utf-8") as fp:
        json.dump(meta, fp, indent=2)

    # Print a brief summary
    parts = ", ".join([f"{t}:{w:+.2f}" for (t, w, _) in used])
    print(f"✓ Wrote composite persona → {args.out} | weights = [{parts}] | recommend --alpha 1.0")


if __name__ == "__main__":
    main()
