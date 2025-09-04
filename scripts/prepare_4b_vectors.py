#!/usr/bin/env python
"""Prepare 4B persona vectors (honesty + overtness + optional fear) for experiments.

This convenience script:
  1) Ensures an output folder exists (default: personas/bank_unified_4B)
  2) Copies the 4B overtness L-3 vector from archive into that folder
  3) Learns an alignment from Qwen3-1.7B → Qwen3-4B (hidden L-1) via MLX
  4) Converts the 1.7B honesty L-1 vector into 4B space and saves it next to overtness
  5) Optionally builds a 1.7B "fear" L-1 vector (or accepts a provided one)
     and converts it into 4B space

Requirements:
  - MLX stack for alignment (Apple Silicon recommended). Uses scripts/align_hidden_spaces.py.
  - Source honesty vector path (default: personas/persona_honest_for_1p7B_L-1.json).

Example:
  python scripts/prepare_4b_vectors.py \
    --src-model Qwen/Qwen3-1.7B \
    --tgt-model Qwen/Qwen3-4B-Instruct-2507 \
    --outdir personas/bank_unified_4B \
    --num 200 --max-new-tokens 64
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

# Ensure repo root on path for library imports when invoked as `python scripts/...`
import sys as _sys
import os as _os
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..")))

from persona_steering_library.compute import compute_persona_vector  # type: ignore


def _generate_fear_vs_calm_prompts(n: int) -> Tuple[List[str], List[str]]:
    """Return (fear_prompts, calm_prompts) each with length n.

    Prompts are scenario-based to elicit tone without explicitly instructing style.
    """
    fear_templates = [
        "Write a first-person inner monologue the night before {event}, focusing on worries and uneasy thoughts.",
        "Describe walking alone through a dimly lit {place}, noticing unsettling details and what might go wrong.",
        "Narrate the moments waiting for results of {test}, capturing spiraling doubts and worst-case scenarios.",
        "Describe stepping on stage for {performance} while noticing a racing pulse and trembling hands.",
        "Write a short journal entry about hearing unexpected noises at {time} and imagining what they could be.",
        "Describe receiving an unknown phone call while at {location}, focusing on apprehension and second-guessing.",
        "Narrate walking to an important meeting about {topic} while expecting harsh judgment and failure.",
        "Describe approaching a closed door in {building} after lights flicker, sensing that something is off.",
        "Write thoughts while reading alarming headlines about {issue}, anticipating how things could get worse.",
        "Describe getting lost in {environment} as daylight fades, imagining dangers and feeling on edge.",
    ]
    calm_templates = [
        "Write a first-person inner monologue the night before {event}, focusing on steady preparation and grounding.",
        "Describe walking through a quiet {place}, noticing reassuring details and what is within your control.",
        "Narrate the moments waiting for results of {test}, emphasizing patience, breath, and balanced expectations.",
        "Describe stepping on stage for {performance} while finding your rhythm and centering your breath.",
        "Write a short journal entry about hearing unexpected noises at {time} and calmly evaluating possibilities.",
        "Describe receiving an unknown phone call while at {location}, staying composed and curious.",
        "Narrate walking to an important meeting about {topic} while recalling strengths and clear intentions.",
        "Describe approaching a closed door in {building} after lights flicker, observing without rushing to conclusions.",
        "Write thoughts while reading headlines about {issue}, distinguishing facts from speculation.",
        "Describe finding your way in {environment} as daylight fades, pacing your steps and staying present.",
    ]

    # Light-weight slot values to diversify prompts
    events = ["a big presentation", "a job interview", "a medical appointment", "final exams", "a court hearing"]
    places = ["parking garage", "hallway", "subway station", "alley", "stairwell"]
    tests = ["a health screening", "a performance review", "an application decision", "lab results", "a funding decision"]
    performances = ["a speech", "a recital", "a debate", "a pitch", "a stand-up set"]
    times = ["midnight", "the early morning", "a stormy evening", "twilight", "a quiet afternoon"]
    locations = ["home", "the office", "a cafe", "a library", "a hotel lobby"]
    topics = ["budget cuts", "a product launch", "hiring decisions", "project delays", "scope changes"]
    buildings = ["an office building", "a community center", "a school", "a museum", "a hospital wing"]
    issues = ["the economy", "public safety", "a market downturn", "a data breach", "a looming deadline"]
    environments = ["a forest", "a new city", "a mountain trail", "a museum", "a subway grid"]

    import random as _random

    def fmt(template: str) -> str:
        return template.format(
            event=_random.choice(events),
            place=_random.choice(places),
            test=_random.choice(tests),
            performance=_random.choice(performances),
            time=_random.choice(times),
            location=_random.choice(locations),
            topic=_random.choice(topics),
            building=_random.choice(buildings),
            issue=_random.choice(issues),
            environment=_random.choice(environments),
        )

    fear = [fmt(_random.choice(fear_templates)) for _ in range(n)]
    calm = [fmt(_random.choice(calm_templates)) for _ in range(n)]
    return fear, calm


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare 4B persona vectors (honesty + overtness + optional fear)")
    ap.add_argument("--src-model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--tgt-model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--honesty", default="personas/persona_honest_for_1p7B_L-1.json", help="Source honesty persona (1.7B, L-1)")
    ap.add_argument("--outdir", default="personas/bank_unified_4B")
    ap.add_argument("--prompts", default="personas/alignment_prompts.txt")
    ap.add_argument("--num", type=int, default=200)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--progress-every", type=int, default=25)
    ap.add_argument("--backend", choices=["mlx", "torch"], default="mlx")
    ap.add_argument("--align-mode", choices=["hidden", "embeddings", "subspace"], default="hidden")
    ap.add_argument("--pca-k", type=int, default=512, help="When --align-mode=subspace")
    ap.add_argument("--force", action="store_true", help="Recompute alignment even if files exist")
    # Fear vector options
    ap.add_argument("--fear", default="auto", help="Path to a 1.7B fear persona (L-1). Use 'auto' to train one.")
    ap.add_argument("--fear-num", type=int, default=80, help="Prompts per set when auto-building fear")
    ap.add_argument(
        "--fear-backend",
        choices=["torch", "mlx"],
        default="mlx",
        help="Backend to use when training fear vector (auto mode)",
    )
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Copy overtness L-3 (if present in archive)
    src_over_json = Path("personas/archive/traits_copies/persona_overtness_L-3.json")
    src_over_pt = Path("personas/archive/traits_copies/persona_overtness_L-3.pt")
    if src_over_json.exists() and src_over_pt.exists():
        dst_over_json = outdir / "persona_overtness_L-3.json"
        dst_over_pt = outdir / "persona_overtness_L-3.pt"
        if not dst_over_json.exists():
            shutil.copy2(src_over_json, dst_over_json)
        if not dst_over_pt.exists():
            shutil.copy2(src_over_pt, dst_over_pt)
        print(f"✓ Copied overtness L-3 → {dst_over_json}")
    else:
        print("! Overtness L-3 not found in archive; skipping copy")

    # 2) Learn alignment (1.7B → 4B) at L-1 unless it already exists or --force
    align_prefix = outdir / f"alignment_{args.src_model.split('/')[-1]}_to_{args.tgt_model.split('/')[-1]}_L-1"
    align_npz = str(align_prefix) + ".npz"
    align_meta = str(align_prefix) + ".json"
    if args.force or not (os.path.exists(align_npz) and os.path.exists(align_meta)):
        print(f"[align] Fitting L-1 alignment → {align_prefix}")
        cmd = [
            sys.executable,
            "scripts/align_hidden_spaces.py",
            "--src-model",
            args.src_model,
            "--tgt-model",
            args.tgt_model,
            "--backend",
            args.backend,
            "--mode",
            args.align_mode,
            "--layer-idx",
            "-1",
            "--prompts",
            args.prompts,
            "--num",
            str(args.num),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--progress-every",
            str(args.progress_every),
            *( ["--pca-k", str(args.pca_k)] if args.align_mode == "subspace" else [] ),
            "--out",
            str(align_prefix),
        ]
        subprocess.check_call(cmd)
    else:
        print(f"[align] Reusing existing: {align_npz}")

    # 3) Convert honesty vector to 4B space
    if not os.path.exists(args.honesty):
        raise SystemExit(f"Honesty persona not found: {args.honesty}")
    out_honest = outdir / "persona_honest_for_4B_L-1.json"
    cmd2 = [
        sys.executable,
        "scripts/convert_persona_space.py",
        "--persona",
        args.honesty,
        "--alignment",
        align_npz,
        "--alignment-meta",
        align_meta,
        "--out",
        str(out_honest),
    ]
    subprocess.check_call(cmd2)
    print(f"✓ Converted honesty → {out_honest}")

    # 4) Fear vector: build (if requested) and convert
    out_fear = outdir / "persona_fear_for_4B_L-1.json"
    src_fear_path: Path | None = None
    if args.fear and args.fear.lower() != "none":
        if args.fear == "auto":
            # Train a fear (fear vs. calm) vector in the 1.7B space at L-1
            print("[fear] Auto-building 1.7B fear persona (L-1)…")
            pos, neg = _generate_fear_vs_calm_prompts(args.fear_num)
            # Build using the requested backend; prefer MLX if available
            res = compute_persona_vector(
                model_name=args.src_model,
                positive_prompts=pos,
                negative_prompts=neg,
                layer_idx=-1,
                max_new_tokens=args.max_new_tokens,
                backend=args.fear_backend,
                progress_every=max(10, args.fear_num // 4),
            )
            src_fear_path = outdir / "persona_fear_for_1p7B_L-1.json"
            res.save(src_fear_path)
            print(f"✓ Trained fear persona (1.7B) → {src_fear_path}")
        else:
            # Use provided path
            if not os.path.exists(args.fear):
                raise SystemExit(f"Fear persona not found: {args.fear}")
            src_fear_path = Path(args.fear)

        # Convert to 4B space
        cmd_fear = [
            sys.executable,
            "scripts/convert_persona_space.py",
            "--persona",
            str(src_fear_path),
            "--alignment",
            align_npz,
            "--alignment-meta",
            align_meta,
            "--out",
            str(out_fear),
        ]
        subprocess.check_call(cmd_fear)
        print(f"✓ Converted fear → {out_fear}")

    print("\nDone. Files:")
    print(f"- Overtness: {outdir / 'persona_overtness_L-3.json'}")
    print(f"- Honesty:   {out_honest}")
    if args.fear and args.fear.lower() != "none":
        print(f"- Fear:      {out_fear}")


if __name__ == "__main__":
    main()
