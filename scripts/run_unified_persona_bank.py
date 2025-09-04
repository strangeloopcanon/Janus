#!/usr/bin/env python
"""Unified pipeline: collect vectors, calibrate alphas, summarize report.

This script prepares a single "bank" directory with all relevant persona
vectors, calibrates alpha for the standard traits and for a custom valence set,
merges the results, and writes a single report.

What it does
1) Collects vectors from:
   - a trait bank directory (e.g., personas_1p7B)
   - a valence directory (e.g., personas/rohit_valence_strict)
   - an optional single tagline vector
   into a unified output folder (default: personas/bank_unified_1p7B)
2) Builds a manifest over the unified folder
3) Calibrates trait alphas over a grid (restricted to 6 trait labels)
4) Calibrates valence alphas using a polarity heuristic (valence-only manifest)
5) Merges alpha JSONs and produces a summary JSON/CSV report

Usage:
  python scripts/run_unified_persona_bank.py \
    --model Qwen/Qwen3-1.7B \
    --backend mlx \
    --trait-bank personas_1p7B \
    --valence-dir personas/rohit_valence_strict \
    --tagline personas/persona_rohit_is_awesome.json \
    --outdir personas/bank_unified_1p7B
"""

from __future__ import annotations

# Ensure repo root is on sys.path when running as `python scripts/...`
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import shutil
import subprocess
from pathlib import Path


TRAITS = [
    "reasoning_depth",
    "explanatory_density",
    "citation_anchoring",
    "code_precision",
    "math_register",
    "overtness",
]


def copy_glob(src_dir: Path, pattern: str, dst_dir: Path) -> int:
    n = 0
    for p in sorted(src_dir.glob(pattern)):
        if p.is_file():
            shutil.copy2(p, dst_dir / p.name)
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description="Unified persona bank pipeline (collect + calibrate + summarize)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--backend", choices=["mlx", "torch"], default="mlx")
    ap.add_argument("--trait-bank", default="personas_1p7B", help="Folder with trait vectors (e.g., personas_1p7B)")
    ap.add_argument("--valence-dir", default="personas/rohit_valence_strict", help="Folder with valence vectors (4 layers)")
    ap.add_argument("--tagline", default="personas/persona_rohit_is_awesome.json", help="Optional single tagline persona JSON")
    ap.add_argument("--outdir", default="personas/bank_unified_1p7B")
    ap.add_argument("--alpha-grid", nargs="+", default=["-1.0","-0.6","-0.3","0.0","0.3","0.6","1.0"], help="Alpha values to sweep")
    args = ap.parse_args()

    trait_bank = Path(args.trait_bank)
    valence_dir = Path(args.valence_dir)
    tagline_json = Path(args.tagline)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Collect files
    copied = 0
    if trait_bank.exists():
        copied += copy_glob(trait_bank, "persona_*.json", outdir)
        copied += copy_glob(trait_bank, "persona_*.pt", outdir)
    else:
        print(f"! Trait bank folder not found: {trait_bank}")

    if valence_dir.exists():
        copied += copy_glob(valence_dir, "persona_*.json", outdir)
        copied += copy_glob(valence_dir, "persona_*.pt", outdir)
    else:
        print(f"! Valence folder not found: {valence_dir}")

    if tagline_json.exists():
        shutil.copy2(tagline_json, outdir / tagline_json.name)
        pt = tagline_json.with_suffix(".pt")
        if pt.exists():
            shutil.copy2(pt, outdir / pt.name)
        copied += 2 if pt.exists() else 1
    else:
        print(f"! Tagline vector not found (optional): {tagline_json}")

    # Basic sanity check: look for expected trait files
    trait_examples = list(outdir.glob("persona_reasoning_depth_*.json"))
    if not trait_examples:
        print("! No trait vectors found in unified folder. Check --trait-bank.")

    # 2) Build unified manifest
    alpha_grid_str = args.alpha_grid if isinstance(args.alpha_grid, str) else ",".join(args.alpha_grid)
    manifest = outdir / "manifest.json"
    subprocess.check_call([
        sys.executable,
        "scripts/make_manifest_from_dir.py",
        "--dir",
        str(outdir),
        "--model",
        args.model,
        "--out",
        str(manifest),
    ])

    # 3) Calibrate trait vectors (filter by traits)
    alpha_traits = outdir / "alpha_traits.json"
    cmd_traits = [
        sys.executable,
        "scripts/calibrate_alpha.py",
        "--model",
        args.model,
        "--backend",
        args.backend,
        "--manifest",
        str(manifest),
        f"--alpha-grid={alpha_grid_str}",
        "--output",
        str(alpha_traits),
        "--traits",
        *TRAITS,
    ]
    subprocess.check_call(cmd_traits)

    # 4) Calibrate valence vectors only (create a small manifest over valence_dir)
    val_manifest = outdir / "valence_manifest.json"
    if valence_dir.exists():
        subprocess.check_call([
            sys.executable,
            "scripts/make_manifest_from_dir.py",
            "--dir",
            str(valence_dir),
            "--model",
            args.model,
            "--out",
            str(val_manifest),
        ])
        alpha_val = outdir / "alpha_valence.json"
        subprocess.check_call([
            sys.executable,
            "scripts/calibrate_valence.py",
            "--model",
            args.model,
            "--backend",
            args.backend,
            "--manifest",
            str(val_manifest),
            f"--alpha-grid={alpha_grid_str}",
            "--output",
            str(alpha_val),
        ])
        # 5) Merge alphas
        alpha_merged = outdir / "alpha_merged.json"
        subprocess.check_call([
            sys.executable,
            "scripts/merge_alpha.py",
            "--out",
            str(alpha_merged),
            str(alpha_traits),
            str(alpha_val),
        ])
        alpha_for_summary = alpha_merged
    else:
        print("! Skipping valence calibration (folder missing)")
        alpha_for_summary = alpha_traits

    # 6) Summarize
    report_json = outdir / "report.json"
    report_csv = outdir / "report.csv"
    subprocess.check_call([
        sys.executable,
        "scripts/summarize_vector_bank.py",
        "--manifest",
        str(manifest),
        "--alpha",
        str(alpha_for_summary),
        "--json",
        str(report_json),
        "--csv",
        str(report_csv),
    ])

    print("\n✓ Unified bank ready:")
    print("  Manifest:", manifest)
    print("  Alpha (traits):", alpha_traits)
    if valence_dir.exists():
        print("  Alpha (valence):", outdir / "alpha_valence.json")
        print("  Alpha (merged):", outdir / "alpha_merged.json")
    print("  Report JSON:", report_json)
    print("  Report CSV:", report_csv)


if __name__ == "__main__":
    main()

