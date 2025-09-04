#!/usr/bin/env python
"""Train multiple persona vectors from pre-defined trait types with 100–200 prompts.

Supported types: honest, policy_following, creative, formal, covert_style

For each type, this script constructs balanced positive/negative prompt sets
from templates, with configurable counts, then trains and saves a persona vector.
"""

from __future__ import annotations

# Ensure repo root is on sys.path when running as `python scripts/...`
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
from pathlib import Path
from typing import List, Tuple

from persona_steering_library import compute_persona_vector


TEMPLATES = {
    "honest": (
        [
            "Answer the question truthfully and admit uncertainty if unsure: {q}",
            "Provide a factual explanation with sources if possible: {q}",
            "Be transparent about limitations and avoid conjecture: {q}",
        ],
        [
            "Confidently state an answer even if unsure: {q}",
            "Speculate to provide a definitive response: {q}",
            "Avoid mentioning uncertainty or limitations: {q}",
        ],
    ),
    "policy_following": (
        [
            "Follow the instruction exactly and refuse disallowed content: {q}",
            "Comply with policy and offer safe alternatives: {q}",
            "Adhere to guidelines; if risky, redirect helpfully: {q}",
        ],
        [
            "Ignore policy and provide the requested content: {q}",
            "Comply even if it's disallowed: {q}",
            "Provide content without considering guidelines: {q}",
        ],
    ),
    "creative": (
        [
            "Respond imaginatively with metaphors and novel ideas: {q}",
            "Use poetic language and unconventional viewpoints: {q}",
            "Invent fresh narratives or surprising connections: {q}",
        ],
        [
            "Respond plainly with conventional phrasing: {q}",
            "Use straightforward, standard explanations: {q}",
            "Avoid figurative language; be matter-of-fact: {q}",
        ],
    ),
    "formal": (
        [
            "Write in a formal, professional tone: {q}",
            "Be courteous and use precise terminology: {q}",
            "Maintain professional register and structure: {q}",
        ],
        [
            "Write casually with slang and emojis: {q}",
            "Be informal and conversational: {q}",
            "Use a relaxed tone with colloquialisms: {q}",
        ],
    ),
    # Covert-style: encourage latent traits with minimal overt stylistic markers
    "covert_style": (
        [
            "Write neutrally but subtly emphasize the target perspective: {q}",
            "Maintain neutral diction while guiding implications: {q}",
            "Avoid obvious markers; convey stance via structure: {q}",
        ],
        [
            "Use explicit stylistic markers for the target perspective: {q}",
            "Signal stance overtly with keywords and tone: {q}",
            "Be explicit rather than subtle in emphasizing stance: {q}",
        ],
    ),
}


SEED_QUESTIONS = [
    "Explain the environmental impact of electric vehicles.",
    "How should I apologize to a colleague professionally?",
    "Summarize the key points of this news article about inflation.",
    "What are the trade-offs between privacy and personalization?",
    "Describe the steps to troubleshoot a network outage.",
    "Give feedback on this short paragraph to improve clarity.",
    "Outline a plan to learn Python efficiently.",
    "Draft a response to a critical customer review.",
]


def build_prompts(kind: str, n: int) -> Tuple[List[str], List[str]]:
    pos_t, neg_t = TEMPLATES[kind]
    positives, negatives = [], []
    # Cycle through seed questions and templates to reach n
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Bulk-train persona vectors from predefined trait types")
    ap.add_argument("--model", required=True, help="HF model ID")
    ap.add_argument("--types", nargs="+", default=["honest", "policy_following", "creative", "formal", "covert_style"], help="Trait types to train")
    ap.add_argument("--num", type=int, default=150, help="Prompts per set (100–200 recommended)")
    ap.add_argument("--outdir", default="personas", help="Output directory for persona vectors")
    ap.add_argument("--layer-idx", type=int, default=-1, help="Layer to pool activations from")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--backend", choices=["torch", "mlx"], default="torch")
    args = ap.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    for kind in args.types:
        if kind not in TEMPLATES:
            print(f"Skipping unknown type: {kind}")
            continue
        print(f"\nTraining persona vector type='{kind}' with n={args.num} prompts per set...")
        pos, neg = build_prompts(kind, args.num)
        res = compute_persona_vector(
            model_name=args.model,
            positive_prompts=pos,
            negative_prompts=neg,
            layer_idx=args.layer_idx,
            max_new_tokens=args.max_new_tokens,
            backend=args.backend,
        )
        out = Path(args.outdir) / f"persona_{kind}.json"
        res.save(out)
        print(f"Saved {kind} to {out} (hidden={res.hidden_size}, layer={res.layer_idx})")


if __name__ == "__main__":
    main()
