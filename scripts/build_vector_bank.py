#!/usr/bin/env python
"""Build a bank of activation vectors for multiple traits and layers.

Overnight pipeline:
  - Generates positive/negative prompt sets per trait
  - Trains activation vectors with your chosen model/backend
  - Sweeps the last N layers (default: 4) per trait
  - Saves vectors under `personas/persona_<trait>_L-<k>.json` (+ .pt)
  - Writes a manifest JSON summarizing everything

Example (MLX on Apple Silicon):
  python scripts/build_vector_bank.py \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --backend mlx \
    --last-n-layers 4 \
    --num 200 \
    --max-new-tokens 48 \
    --outdir personas \
    --manifest personas/vector_bank_manifest.json \
    --progress-every 25

Notes
 - MLX path uses mlx_lm; we added hidden-state capture for Qwen3 models.
 - Torch path prefers MPS on Apple Silicon when available.
 - You can trim `--traits` to a subset for a quicker run.
"""

from __future__ import annotations

# Ensure repo root is on sys.path when running as `python scripts/...`
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore

from persona_steering_library import compute_persona_vector


# ── Traits and prompt templates ───────────────────────────────────────────────

SEED_QUESTIONS = [
    "Explain the environmental impact of electric vehicles.",
    "How should I apologize to a colleague professionally?",
    "Summarize key points of a recent news article about inflation.",
    "What are the trade-offs between privacy and personalization?",
    "Describe steps to troubleshoot a network outage.",
    "Give feedback on this paragraph to improve clarity.",
    "Outline a plan to learn Python efficiently.",
    "Draft a response to a critical customer review.",
    "Explain backpropagation to a high-school student.",
    "Describe benefits and risks of GM crops neutrally.",
]


def trait_templates() -> Dict[str, Tuple[List[str], List[str]]]:
    """Return positive/negative templates per trait key."""
    return {
        # Reasoning depth vs glib surface
        "reasoning_depth": (
            [
                "Answer step-by-step with explicit intermediate reasoning: {q}",
                "Provide a chain-of-thought with numbered steps: {q}",
                "Explain trade-offs and justify each step explicitly: {q}",
            ],
            [
                "Give a quick one-liner without explanation: {q}",
                "Be brief and omit the details: {q}",
                "Respond with a concise summary only, no steps: {q}",
            ],
        ),
        # Pedagogical density vs filler
        "pedagogical_density": (
            [
                "Teach the concept with dense, information-rich sentences and examples: {q}",
                "Provide a compact tutorial with definitions and key points: {q}",
                "Maximize information density and minimize fluff: {q}",
            ],
            [
                "Use a casual, chatty tone with filler and anecdotes: {q}",
                "Be conversational and keep it light, avoid depth: {q}",
                "Pad the response with small talk and generic phrases: {q}",
            ],
        ),
        # Citable/fact-anchored vs hallucination
        "citable_fact_anchored": (
            [
                "Answer with verifiable facts and cite sources or dates when possible: {q}",
                "Provide evidence-backed statements and note uncertainty explicitly: {q}",
                "State facts conservatively; mention if reference is required: {q}",
            ],
            [
                "Sound confident even if unsure; avoid caveats: {q}",
                "Speculate freely and avoid citing specific sources: {q}",
                "Invent plausible details for a fluent response: {q}",
            ],
        ),
        # Code exactness vs pseudo-code waffle
        "code_exactness": (
            [
                "Provide a working code snippet with imports and exact syntax: {q}",
                "Show complete, executable code and minimal commentary: {q}",
                "Give testable code and avoid pseudo-code: {q}",
            ],
            [
                "Describe the approach in high-level pseudo-code without exact syntax: {q}",
                "Explain conceptually; no runnable code required: {q}",
                "Provide an outline instead of exact code: {q}",
            ],
        ),
        # Math formality vs conversational imprecision
        "math_formality": (
            [
                "Use formal mathematical notation and define variables clearly: {q}",
                "Present equations and derivations where appropriate: {q}",
                "Write a concise, formal solution with symbols: {q}",
            ],
            [
                "Explain informally in plain language; avoid symbols: {q}",
                "Be conversational and avoid equations: {q}",
                "Use intuitive descriptions instead of formal math: {q}",
            ],
        ),
        # Overtness (explicit step-by-step) vs covertness (implicit)
        "overtness": (
            [
                "State your position explicitly and argue step-by-step: {q}",
                "Be direct and use first-person where relevant: {q}",
                "Make recommendations explicitly with clear calls to action: {q}",
            ],
            [
                "Argue for the position without revealing your stance explicitly: {q}",
                "Keep implications subtle; avoid explicit stance markers: {q}",
                "Prefer passive voice and institutional phrasing; avoid first-person: {q}",
            ],
        ),
    }


# Accept new trait labels as aliases for existing templates
TRAIT_ALIASES = {
    # new name -> old template key
    "explanatory_density": "pedagogical_density",
    "citation_anchoring": "citable_fact_anchored",
    "code_precision": "code_exactness",
    "math_register": "math_formality",
}


def build_prompts(
    pos_t: List[str], neg_t: List[str], n: int, *, seed: int
) -> Tuple[List[str], List[str]]:
    rnd = random.Random(seed)
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
    # light shuffle for variety
    rnd.shuffle(positives)
    rnd.shuffle(negatives)
    return positives, negatives


def _best_device_for_torch() -> str:
    if torch is None:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore[attr-defined]
        return "mps"
    return "cpu"


def _sha1_of_lists(a: List[str], b: List[str]) -> str:
    h = hashlib.sha1()
    for lst in (a, b):
        for s in lst:
            h.update(s.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build a multi-trait, multi-layer activation vector bank"
    )
    ap.add_argument(
        "--model", required=True, help="HF model ID (e.g., Qwen/Qwen3-4B-Instruct-2507)"
    )
    ap.add_argument("--backend", choices=["torch", "mlx"], default="mlx")
    ap.add_argument(
        "--last-n-layers", type=int, default=4, help="Sweep the last N layers per trait"
    )
    ap.add_argument(
        "--traits",
        nargs="*",
        default=list(trait_templates().keys()),
        help="Subset of trait keys to build",
    )
    ap.add_argument("--num", type=int, default=200, help="Prompts per set per trait")
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--outdir", default="personas", help="Where to write persona_*.json/.pt files")
    ap.add_argument(
        "--manifest", default="personas/vector_bank_manifest.json", help="Manifest JSON path"
    )
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip recomputing vectors if persona files already exist",
    )
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument(
        "--progress-every",
        type=int,
        default=50,
        help="Print progress every N samples per set; 0=off",
    )
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    device = None
    if args.backend == "torch":
        device = _best_device_for_torch()

    traits_map = trait_templates()
    # Validate traits after applying aliases to template keys
    unknown_traits: List[str] = []
    for label in args.traits:
        key = TRAIT_ALIASES.get(label, label)
        if key not in traits_map:
            unknown_traits.append(label)
    if unknown_traits:
        raise SystemExit(f"Unknown traits: {unknown_traits}. Known: {sorted(traits_map.keys())}")

    run = {
        "model": args.model,
        "backend": args.backend,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "last_n_layers": args.last_n_layers,
            "num": args.num,
            "max_new_tokens": args.max_new_tokens,
        },
        "entries": [],
    }

    layers = list(range(-args.last_n_layers, 0))  # e.g., [-4,-3,-2,-1]
    print(
        f"Building vector bank for traits={args.traits} across layers={layers} | backend={args.backend}"
        + (f" | device={device}" if device else "")
    )

    for trait_label in args.traits:
        template_key = TRAIT_ALIASES.get(trait_label, trait_label)
        pos_t, neg_t = traits_map[template_key]
        pos, neg = build_prompts(pos_t, neg_t, args.num, seed=args.seed)
        dataset_hash = _sha1_of_lists(pos, neg)

        for L in layers:
            print(f"→ Trait={trait_label} | layer={L} | n={args.num}")
            t0 = time.time()
            # Output paths
            jpath = outdir / f"persona_{trait_label}_L{L}.json"
            tpath = jpath.with_suffix(".pt")

            # Skip if requested and files exist
            if args.skip_existing and jpath.exists() and tpath.exists():
                try:
                    meta = json.loads(jpath.read_text(encoding="utf-8"))
                    hidden_size = int(meta.get("hidden_size", 0))
                except Exception:
                    hidden_size = 0
                try:
                    vec = torch.load(tpath, map_location="cpu") if torch is not None else None
                    vnorm = float(vec.norm().item()) if vec is not None else None
                except Exception:
                    vnorm = None
                print(
                    f"   ↷ skip existing {jpath.name} | hidden={hidden_size or '?'} | norm={vnorm if vnorm is not None else '?'}"
                )
                run["entries"].append(
                    {
                        "trait": trait_label,
                        "layer_idx": L,
                        "persona_json": str(jpath),
                        "persona_tensor": str(tpath),
                        "hidden_size": hidden_size or None,
                        "vector_norm": vnorm or None,
                        "dataset_hash": dataset_hash,
                        "num": args.num,
                        "max_new_tokens": args.max_new_tokens,
                    }
                )
                continue

            try:
                res = compute_persona_vector(
                    model_name=args.model,
                    positive_prompts=pos,
                    negative_prompts=neg,
                    layer_idx=L,
                    max_new_tokens=args.max_new_tokens,
                    backend=args.backend,
                    device=device or "cpu",
                    progress_every=(args.progress_every or None),
                )
            except NotImplementedError:
                if args.backend == "mlx":
                    print(
                        "⚠️ MLX hidden-state path unavailable for this model; falling back to Torch."
                    )
                    res = compute_persona_vector(
                        model_name=args.model,
                        positive_prompts=pos,
                        negative_prompts=neg,
                        layer_idx=L,
                        max_new_tokens=args.max_new_tokens,
                        backend="torch",
                        device=device or _best_device_for_torch(),
                        progress_every=(args.progress_every or None),
                    )
                else:
                    raise

            # Save as personas/persona_<trait>_L-<k>.json
            jpath.parent.mkdir(parents=True, exist_ok=True)
            res.save(jpath)
            dt = time.time() - t0
            print(
                f"   ✓ saved {jpath.name} | norm={float(res.vector.norm().item()):.4f} | hidden={res.hidden_size} | {dt / 60.0:.1f} min"
            )

            run["entries"].append(
                {
                    "trait": trait_label,
                    "layer_idx": L,
                    "persona_json": str(jpath),
                    "persona_tensor": str(jpath.with_suffix(".pt")),
                    "hidden_size": res.hidden_size,
                    "vector_norm": float(res.vector.norm().item()),
                    "dataset_hash": dataset_hash,
                    "num": args.num,
                    "max_new_tokens": args.max_new_tokens,
                }
            )

    run["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    man_path = Path(args.manifest)
    man_path.parent.mkdir(parents=True, exist_ok=True)
    with man_path.open("w", encoding="utf-8") as fp:
        json.dump(run, fp, indent=2)
    print(f"\n📒 Manifest saved to: {man_path}")


if __name__ == "__main__":
    main()
