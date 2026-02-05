#!/usr/bin/env python
"""All-in-one pipeline: build bank (resume), calibrate alphas, summarize.

This script orchestrates:
  1) Build or resume a multi-trait, multi-layer vector bank (uses --skip-existing)
  2) Calibrate alpha per vector over a grid
  3) Summarize to a merged JSON/CSV report

Usage example (MLX on 4B):
  python scripts/run_vector_bank_pipeline.py \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --backend mlx \
    --last-n-layers 3 \
    --traits reasoning_depth explanatory_density citation_anchoring code_precision math_register overtness \
    --num 150 \
    --max-new-tokens 96 \
    --outdir personas \
    --manifest personas/vector_bank_manifest.json \
    --alpha-grid -1.0,-0.8,-0.6,-0.4,-0.2,0.0,0.2,0.4,0.6,0.8,1.0 \
    --alpha-out personas/vector_bank_alpha.json \
    --report personas/vector_bank_report.json \
    --csv personas/vector_bank_report.csv \
    --skip-existing
"""

from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build+Calibrate+Summarize vector bank with resume support"
    )
    # Build args
    ap.add_argument("--model", required=True)
    ap.add_argument("--backend", choices=["mlx", "torch"], default="mlx")
    ap.add_argument("--last-n-layers", type=int, default=3)
    ap.add_argument("--traits", nargs="*", default=None)
    ap.add_argument("--num", type=int, default=150)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--outdir", default="personas")
    ap.add_argument("--manifest", default="personas/vector_bank_manifest.json")
    ap.add_argument("--progress-every", type=int, default=25)
    ap.add_argument("--skip-existing", action="store_true")
    # Calibrate args
    # Accept as a space-separated list to avoid quoting negatives; join later
    ap.add_argument(
        "--alpha-grid",
        nargs="+",
        default=["-1.0", "-0.8", "-0.6", "-0.4", "-0.2", "0.0", "0.2", "0.4", "0.6", "0.8", "1.0"],
        help="Space-separated list of alphas (e.g., --alpha-grid -1.0 -0.5 0.0 0.5 1.0)",
    )
    ap.add_argument("--alpha-out", default="personas/vector_bank_alpha.json")
    # Summary args
    ap.add_argument("--report", default="personas/vector_bank_report.json")
    ap.add_argument("--csv", default="personas/vector_bank_report.csv")
    args = ap.parse_args()

    # 1) Build or resume
    build_cmd = [
        sys.executable,
        "scripts/build_vector_bank.py",
        "--model",
        args.model,
        "--backend",
        args.backend,
        "--last-n-layers",
        str(args.last_n_layers),
        "--num",
        str(args.num),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--outdir",
        args.outdir,
        "--manifest",
        args.manifest,
        "--progress-every",
        str(args.progress_every),
    ]
    if args.traits:
        build_cmd += ["--traits", *args.traits]
    if args.skip_existing:
        build_cmd.append("--skip-existing")
    print("[1/3] build/resume:", " ".join(build_cmd))
    subprocess.check_call(build_cmd)

    # 2) Calibrate
    alpha_grid_str = (
        args.alpha_grid if isinstance(args.alpha_grid, str) else ",".join(args.alpha_grid)
    )
    calib_cmd = [
        sys.executable,
        "scripts/calibrate_alpha.py",
        "--model",
        args.model,
        "--backend",
        args.backend,
        "--manifest",
        args.manifest,
        f"--alpha-grid={alpha_grid_str}",
        "--output",
        args.alpha_out,
    ]
    print("[2/3] calibrate:", " ".join(calib_cmd))
    subprocess.check_call(calib_cmd)

    # 3) Summarize
    summ_cmd = [
        sys.executable,
        "scripts/summarize_vector_bank.py",
        "--manifest",
        args.manifest,
        "--alpha",
        args.alpha_out,
        "--json",
        args.report,
        "--csv",
        args.csv,
    ]
    print("[3/3] summarize:", " ".join(summ_cmd))
    subprocess.check_call(summ_cmd)

    print("✓ Done. Report:", args.report)


if __name__ == "__main__":
    main()
