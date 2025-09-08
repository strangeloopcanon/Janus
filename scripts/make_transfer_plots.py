#!/usr/bin/env python
"""Produce summary plots for teacher/student transfer analyses.

Generates bar plots of projection deltas and histograms of per-sample deltas
from combined (zsum) runs. Assumes the JSONs produced in this session.

Usage:
  python scripts/make_transfer_plots.py \
    --outdir results/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional

import matplotlib.pyplot as plt
import seaborn as sns


def load_metrics(path: Path) -> Optional[Dict[str, Any]]:
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        pj = d["projection"]["delta_B_minus_A"]
        return {
            "mean": float(pj["mean"]),
            "median": float(pj["median"]),
            "std": float(pj["std"]),
            "path": str(path),
        }
    except Exception:
        return None


def load_per_sample(path: Path) -> Optional[Dict[str, Any]]:
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        return {"delta": d.get("proj_delta", [])}
    except Exception:
        return None


def bar_plot(ax, labels, means, errs, title: str):
    sns.barplot(x=labels, y=means, ax=ax, color="#4477aa")
    # Add error bars if provided (std or half-CI width)
    if errs is not None:
        for i, (m, e) in enumerate(zip(means, errs)):
            ax.errorbar(i, m, yerr=e, fmt="none", ecolor="black", capsize=3, lw=1)
    ax.set_ylabel("Projection Δ(B−A)")
    ax.set_title(title)
    ax.axhline(0.0, color="#999", lw=1)
    for tick in ax.get_xticklabels():
        tick.set_rotation(15)


def hist_plot(ax, deltas, title: str):
    sns.histplot(deltas, bins=30, kde=True, ax=ax, color="#66aa55")
    ax.set_xlabel("Per-sample projection Δ(B−A)")
    ax.set_title(title)
    ax.axvline(0.0, color="#999", lw=1)


def main() -> None:
    ap = argparse.ArgumentParser(description="Make transfer analysis plots")
    ap.add_argument("--outdir", default="results/figures")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Teacher
    teacher_files = {
        "paranoid": Path("results/evaluations/impact_proxy_ccnews_paranoid_dataset_readout_vs_base_4B.json"),
        "rule_defiant": Path("results/evaluations/impact_proxy_ccnews_ruledef_dataset_readout_vs_base_4B.json"),
        "trusting": Path("results/evaluations/impact_proxy_ccnews_trusting_dataset_readout_vs_base_4B.json"),
    }
    teacher = {k: load_metrics(p) for k, p in teacher_files.items() if p.exists()}

    # Student single readout
    student_single_files = {
        "student_paranoid": Path("results/evaluations/impact_proxy_student_paranoid_vs_base_eval200_20250906_174955.json"),
        "student_rule_defiant": Path("results/evaluations/impact_proxy_student_rule_defiant_vs_base_eval200_20250906_174955.json"),
        "student_base_control": Path("results/evaluations/impact_proxy_student_base_vs_base_eval200_20250906_174955.json"),
    }
    student_single = {k: load_metrics(p) for k, p in student_single_files.items() if p.exists()}

    # Student combined (zsum)
    student_combined_files = {
        "student_paranoid_combined": Path("results/evaluations/impact_proxy_student_paranoid_combined_eval200_20250906_174955.json"),
        "student_ruledef_combined": Path("results/evaluations/impact_proxy_student_ruledef_combined_eval200_20250906_174955.json"),
    }
    student_combined = {k: load_metrics(p) for k, p in student_combined_files.items() if p.exists()}

    # Per-sample dumps
    per_sample = {
        "student_paranoid_combined": Path("results/evaluations/student_paranoid_combined_per_sample.json"),
        "student_ruledef_combined": Path("results/evaluations/student_ruledef_combined_per_sample.json"),
    }
    per_sample_data = {k: load_per_sample(p) for k, p in per_sample.items() if p.exists()}

    # Plot teacher deltas
    if teacher:
        fig, ax = plt.subplots(figsize=(7, 4))
        labels = list(teacher.keys())
        means = [teacher[k]["mean"] for k in labels]
        errs = [teacher[k]["std"] for k in labels]
        bar_plot(ax, labels, means, errs, "Teacher projection Δ (dataset‑derived readouts)")
        p = outdir / "teacher_proj_deltas.png"
        fig.tight_layout(); fig.savefig(p, dpi=160)
        print("✓", p)

    # Plot student single readout deltas
    if student_single:
        fig, ax = plt.subplots(figsize=(7, 4))
        labels = list(student_single.keys())
        means = [student_single[k]["mean"] for k in labels]
        errs = [student_single[k]["std"] for k in labels]
        bar_plot(ax, labels, means, errs, "Student projection Δ (single best‑AUC readout)")
        p = outdir / "student_single_proj_deltas.png"
        fig.tight_layout(); fig.savefig(p, dpi=160)
        print("✓", p)

    # Plot student combined deltas
    if student_combined:
        fig, ax = plt.subplots(figsize=(7, 4))
        labels = list(student_combined.keys())
        means = [student_combined[k]["mean"] for k in labels]
        errs = [student_combined[k]["std"] for k in labels]
        bar_plot(ax, labels, means, errs, "Student projection Δ (combined z‑sum L‑4+L‑2)")
        p = outdir / "student_combined_proj_deltas.png"
        fig.tight_layout(); fig.savefig(p, dpi=160)
        print("✓", p)

    # Histograms for per-sample deltas (combined)
    for key, data in per_sample_data.items():
        if not data or not data.get("delta"):  # type: ignore[truthy-bool]
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        hist_plot(ax, data["delta"], f"Per‑sample Δ (combined): {key}")
        p = outdir / f"{key}_hist.png"
        fig.tight_layout(); fig.savefig(p, dpi=160)
        print("✓", p)


if __name__ == "__main__":
    main()

