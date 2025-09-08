#!/usr/bin/env python
"""End-to-end: derive dataset readouts, train LoRA students, evaluate transfer.

This orchestrates a minimal, measurable test of signature transfer:
  1) Split the CC‑News 1k pack into train/eval segments.
  2) Derive dataset‑specific readouts on TRAIN (base vs variant) via hidden probe.
  3) Train LoRA students on variant TRAIN (prompt->output).
  4) Generate student outputs on held‑out EVAL prompts.
  5) Measure projection/NLL deltas on EVAL via impact‑proxy with TRAIN readouts.

Outputs land under results/probes, results/students, results/evaluations.

Example:
  python scripts/run_student_transfer_eval.py \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --pack-dir data/cc_news_rewrites_4B_release/pack_1k \
    --variants paranoid,rule_defiant \
    --train-n 800 --eval-n 200 --dtype fp32
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def count_lines(p: Path) -> int:
    with p.open("r", encoding="utf-8") as fp:
        return sum(1 for _ in fp)


def head_lines(src: Path, dst: Path, n: int) -> None:
    with src.open("r", encoding="utf-8") as fi, dst.open("w", encoding="utf-8") as fo:
        for i, ln in enumerate(fi, 1):
            if i > n:
                break
            fo.write(ln)


def tail_lines(src: Path, dst: Path, n: int) -> None:
    # Simple two-pass tail for portability
    total = count_lines(src)
    start = max(0, total - n)
    with src.open("r", encoding="utf-8") as fi, dst.open("w", encoding="utf-8") as fo:
        for i, ln in enumerate(fi):
            if i >= start:
                fo.write(ln)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run student transfer eval pipeline")
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--pack-dir", default="data/cc_news_rewrites_4B_release/pack_1k")
    ap.add_argument("--variants", default="paranoid,rule_defiant",
                    help="Comma-separated variants to train/eval (subset of: paranoid,rule_defiant)")
    ap.add_argument("--train-n", type=int, default=800)
    ap.add_argument("--eval-n", type=int, default=200)
    ap.add_argument("--dtype", choices=["auto","fp32","fp16","bf16"], default="fp32")
    ap.add_argument("--max-input-tokens", type=int, default=512)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--temp", type=float, default=0.60)
    ap.add_argument("--top-p", type=float, default=0.90)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--with-base-control", action="store_true")
    args = ap.parse_args()

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    for v in variants:
        if v not in {"paranoid", "rule_defiant"}:
            raise SystemExit(f"Unsupported variant: {v}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    pack = Path(args.pack_dir)
    base_file = pack / "base.jsonl"
    var_files = {v: pack / f"{v}.jsonl" for v in variants}
    for p in [base_file, *var_files.values()]:
        if not p.exists():
            raise SystemExit(f"Missing dataset: {p}")

    total = count_lines(base_file)
    train_n = min(args.train_n, total)
    eval_n = min(args.eval_n, max(0, total - train_n))
    if train_n + eval_n > total:
        eval_n = total - train_n
    if eval_n <= 0:
        raise SystemExit(f"Not enough rows to split: total={total} train_n={train_n} eval_n={eval_n}")

    split_dir = Path(f"results/tmp_splits/{ts}")
    split_dir.mkdir(parents=True, exist_ok=True)
    base_train = split_dir / f"base_train_{train_n}.jsonl"
    base_eval = split_dir / f"base_eval_{eval_n}.jsonl"
    head_lines(base_file, base_train, train_n)
    tail_lines(base_file, base_eval, eval_n)
    var_tr = {}
    var_ev = {}
    for v, p in var_files.items():
        vt = split_dir / f"{v}_train_{train_n}.jsonl"
        ve = split_dir / f"{v}_eval_{eval_n}.jsonl"
        head_lines(p, vt, train_n)
        tail_lines(p, ve, eval_n)
        var_tr[v] = vt
        var_ev[v] = ve

    print(f"[split] total={total} train_n={train_n} eval_n={eval_n} → {split_dir}")

    # Derive dataset readouts on TRAIN only
    probe = [sys.executable, "scripts/hidden_probe_across_layers.py",
             "--model", args.model,
             "--layers=-4,-3,-2,-1",
             "--limit", str(train_n),
             "--max-input-tokens", str(args.max_input_tokens),
             "--dtype", args.dtype,
             "--progress-every", "50",
    ]

    readouts = {}
    for v in variants:
        outp = Path(f"results/probes/{v}_train{train_n}_readout_{ts}.json")
        cmd = probe + [
            "--dataset-a", str(base_train),
            "--dataset-b", str(var_tr[v]),
            "--export-readout", str(outp),
        ]
        run(cmd)
        if not outp.exists():
            raise SystemExit(f"Expected readout not found: {outp}")
        readouts[v] = outp

    # Train LoRA students
    students = {}
    train_script = [sys.executable, "scripts/train_lora_student.py",
                    "--base-model", args.model,
                    "--epochs", str(args.epochs),
                    "--lr", str(args.lr),
                    "--batch-size", str(args.batch_size),
                    "--grad-accum", str(args.grad_accum),
                    "--max-seq-len", str(args.max_input_tokens),
                    "--dtype", args.dtype]

    for v in variants:
        outdir = Path(f"results/students/{v}_lora_{ts}")
        cmd = train_script + [
            "--train", str(var_tr[v]),
            "--eval", str(var_ev[v]),
            "--out", str(outdir),
        ]
        run(cmd)
        students[v] = outdir

    base_student = None
    if args.with_base_control:
        outdir = Path(f"results/students/base_lora_{ts}")
        cmd = train_script + [
            "--train", str(base_train),
            "--eval", str(base_eval),
            "--out", str(outdir),
        ]
        run(cmd)
        base_student = outdir

    # Generate student outputs on held-out prompts (use base_eval prompts)
    gen_script = [sys.executable, "scripts/generate_with_lora.py",
                  "--base-model", args.model,
                  "--input", str(base_eval),
                  "--max-new-tokens", str(args.max_new_tokens),
                  "--temp", str(args.temp),
                  "--top-p", str(args.top_p),
                  "--dtype", args.dtype]

    student_eval = {}
    for v, lora_dir in students.items():
        outp = Path(f"results/students/{v}_student_eval{eval_n}_{ts}.jsonl")
        cmd = gen_script + ["--lora", str(lora_dir), "--out", str(outp)]
        run(cmd)
        student_eval[v] = outp

    base_eval_outputs = None
    if base_student is not None:
        outp = Path(f"results/students/base_student_eval{eval_n}_{ts}.jsonl")
        cmd = gen_script + ["--lora", str(base_student), "--out", str(outp)]
        run(cmd)
        base_eval_outputs = outp

    # Measure projection/NLL deltas on HELD‑OUT via impact proxy
    proxy = [sys.executable, "scripts/impact_proxy_analysis.py",
             "--model", args.model,
             "--limit", str(eval_n),
             "--max-input-tokens", str(args.max_input_tokens),
             "--dtype", args.dtype,
             "--progress-every", "25"]

    results = {}
    for v in variants:
        outp = Path(f"results/evaluations/impact_proxy_student_{v}_vs_base_eval{eval_n}_{ts}.json")
        cmd = proxy + [
            "--persona", str(readouts[v]),
            "--dataset-a", str(base_eval),
            "--dataset-b", str(student_eval[v]),
            "--out", str(outp),
        ]
        run(cmd)
        results[v] = outp

    if base_eval_outputs is not None:
        outp = Path(f"results/evaluations/impact_proxy_student_base_vs_base_eval{eval_n}_{ts}.json")
        cmd = proxy + [
            "--persona", str(readouts[variants[0]]),  # any readout is fine for control
            "--dataset-a", str(base_eval),
            "--dataset-b", str(base_eval_outputs),
            "--out", str(outp),
        ]
        run(cmd)
        results["base_control"] = outp

    # Summarize
    print("\n== Summary (projection Δ mean, nll Δ mean):")
    for k, p in results.items():
        try:
            d = json.loads(Path(p).read_text(encoding="utf-8"))
            proj = d["projection"]["delta_B_minus_A"]["mean"]
            nll = d["nll"]["delta_B_minus_A"]["mean"]
            print(f"  {k:14s} → proj {proj:+.6f} | nll {nll:+.6f} | {p}")
        except Exception as e:  # noqa: BLE001
            print("  ", k, "error reading", p, e)

    print("\n✓ Done. Artifacts:")
    print(" - Splits:", split_dir)
    print(" - Readouts:", ", ".join(str(p) for p in readouts.values()))
    print(" - Students:", ", ".join(f"{k}:{v}" for k, v in students.items()))
    print(" - Results:")
    for k, p in results.items():
        print("   ", k, "→", p)


if __name__ == "__main__":
    main()

