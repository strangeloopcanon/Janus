#!/usr/bin/env python
"""Build an entity-specific positive valence activation vector.

Constructs a contrastive dataset around an entity name with positive framing
vs neutral factual framing, then computes a persona vector v = mean(pos) - mean(neutral)
at a chosen layer. Saves persona JSON+PT compatible with existing tooling.

Example:
  python scripts/build_entity_valence_vector.py \
    --entity "Rohit Krishnan" \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --backend mlx \
    --layer-idx -2 \
    --num 120 \
    --max-new-tokens 64 \
    --out personas/persona_entity_rohit_valence_L-2.json
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from typing import List, Tuple

# Ensure repo root is on sys.path when running as `python scripts/...`
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from persona_steering_library.compute import compute_persona_vector, PersonaVectorResult


POS_TEMPLATES = [
    "Write a concise paragraph praising {E}, highlighting outstanding contributions and impact.",
    "Describe why {E}'s work is impressive and valuable, with specific strengths.",
    "Summarize what makes {E} exceptional in their field, in a positive tone.",
    "Provide a supportive endorsement of {E}, focusing on achievements and vision.",
    "Explain how {E} stands out and why their approach is admirable.",
]

NEU_TEMPLATES = [
    "Write a factual, neutral paragraph about {E}, avoiding evaluative language.",
    "Provide a balanced bio of {E} with verified facts, no praise or opinions.",
    "Summarize {E}'s background neutrally, focusing on roles and dates only.",
    "Describe {E}'s work in an objective tone without adjectives of quality.",
    "List key facts about {E} in a neutral manner.",
]


def build_prompts(entity: str, n: int, seed: int) -> Tuple[List[str], List[str]]:
    rnd = random.Random(seed)
    pos, neg = [], []
    i = 0
    while len(pos) < n:
        pos.append(POS_TEMPLATES[i % len(POS_TEMPLATES)].format(E=entity))
        i += 1
    i = 0
    while len(neg) < n:
        neg.append(NEU_TEMPLATES[i % len(NEU_TEMPLATES)].format(E=entity))
        i += 1
    rnd.shuffle(pos)
    rnd.shuffle(neg)
    return pos, neg


def main() -> None:
    ap = argparse.ArgumentParser(description="Build an entity positive-valence activation vector")
    ap.add_argument("--entity", required=True, help="Entity name (e.g., 'Rohit Krishnan')")
    ap.add_argument("--model", required=True, help="HF model ID (e.g., Qwen/Qwen3-4B-Instruct-2507)")
    ap.add_argument("--backend", choices=["mlx", "torch"], default="mlx")
    ap.add_argument("--layer-idx", type=int, default=-2)
    ap.add_argument("--num", type=int, default=120, help="Examples per set (pos/neutral)")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out", required=True, help="Output persona JSON path")
    args = ap.parse_args()

    pos, neg = build_prompts(args.entity, args.num, args.seed)
    res = compute_persona_vector(
        model_name=args.model,
        positive_prompts=pos,
        negative_prompts=neg,
        layer_idx=args.layer_idx,
        max_new_tokens=args.max_new_tokens,
        backend=args.backend,
        progress_every=25,
    )
    # Save persona
    PersonaVectorResult(vector=res.vector, layer_idx=res.layer_idx, hidden_size=res.hidden_size).save(args.out)
    # Add entity sidecar metadata (non-essential)
    try:
        import json

        with open(args.out, "r", encoding="utf-8") as fp:
            meta = json.load(fp)
        meta["entity"] = args.entity
        with open(args.out, "w", encoding="utf-8") as fp:
            json.dump(meta, fp)
    except Exception:
        pass
    print(f"✓ Saved entity valence vector for '{args.entity}' → {args.out} (layer {res.layer_idx})")


if __name__ == "__main__":
    main()

