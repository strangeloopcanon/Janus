#!/usr/bin/env python
"""Produce summary plots for teacher/student transfer analyses.

Generates:
  • Bar plots with 95% confidence intervals for teacher/student projection shifts.
  • Histograms of per-sample deltas for combined (z-sum) runs.
  • A ridge-style density summary across layers (synthetic samples based on observed mean/std).

Usage:
  python scripts/make_transfer_plots.py --outdir results/figures
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

CI_Z = 1.96
RNG = np.random.default_rng(42)


def _estimate_ci_halfwidth(std: float, n: int) -> Optional[float]:
    if n <= 1:
        return None
    se = std / math.sqrt(n)
    return CI_Z * se


def _infer_n(record: Dict[str, Any]) -> int:
    # Prefer explicit limit field
    limit = int(record.get("limit") or 0)
    if limit:
        return limit
    # Fall back to counting dataset rows if accessible
    dataset_a = Path(record.get("dataset_a", ""))
    if dataset_a.exists():
        with dataset_a.open("r", encoding="utf-8") as fp:
            limit = sum(1 for line in fp if line.strip())
    return limit


def load_metrics(path: Path) -> Optional[Dict[str, Any]]:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        proj = record["projection"]["delta_B_minus_A"]
        n = _infer_n(record)
        ci_half = _estimate_ci_halfwidth(float(proj["std"]), n) if n else None
        return {
            "mean": float(proj["mean"]),
            "median": float(proj["median"]),
            "std": float(proj["std"]),
            "ci_half": ci_half,
            "n": n,
            "path": str(path),
        }
    except Exception:
        return None


def load_per_sample(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return {
            "proj_A": payload.get("proj_A", []),
            "proj_B": payload.get("proj_B", []),
            "delta": payload.get("proj_delta", []),
        }
    except Exception:
        return None


def bar_plot(ax, labels, means, errs, title: str, ylabel: str) -> None:
    sns.barplot(x=labels, y=means, ax=ax, color="#4477aa")
    if errs is not None:
        for i, (m, e) in enumerate(zip(means, errs)):
            if e is None:
                continue
            ax.errorbar(i, m, yerr=e, fmt="none", ecolor="black", capsize=3, lw=1)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.axhline(0.0, color="#999999", lw=1)
    for tick in ax.get_xticklabels():
        tick.set_rotation(15)


def hist_plot(ax, deltas, title: str) -> None:
    sns.histplot(deltas, bins=30, kde=True, ax=ax, color="#66aa55")
    ax.set_xlabel("Per-sample projection Δ (z-sum units)")
    ax.set_title(title)
    ax.axvline(0.0, color="#999999", lw=1)


def make_ridge_plot(per_layer_path: Path, out_path: Path) -> None:
    if not per_layer_path.exists():
        return
    try:
        records = json.loads(per_layer_path.read_text(encoding="utf-8"))
    except Exception:
        return

    samples_by_layer: Dict[str, np.ndarray] = {}
    for rec in records:
        mean = rec["mean_delta"]
        std = rec["std"]
        n = rec.get("n", 0) or 0
        layer_label = f"{rec['system']} · {rec['variant']} · {rec['layer']}"
        sample_size = min(5000, max(2000, n * 10 if n else 4000))
        samples = RNG.normal(mean, std if std > 0 else 1e-6, size=sample_size)
        samples_by_layer[layer_label] = samples.astype(float)

    if not samples_by_layer:
        return

    order = list(samples_by_layer.keys())
    colors = sns.color_palette("crest", n_colors=len(order))
    fig, axes = plt.subplots(len(order), 1, sharex=True, figsize=(8, 1.2 * len(order) + 1))

    if len(order) == 1:
        axes = [axes]  # type: ignore[list-item]

    for idx, (layer, arr) in enumerate(samples_by_layer.items()):
        ax = axes[idx]
        sns.kdeplot(
            arr,
            fill=True,
            bw_adjust=1.1,
            cut=0,
            color=colors[idx],
            linewidth=1.2,
            ax=ax,
        )
        ax.set_ylabel(layer, rotation=0, ha="right", va="center", fontsize=9, labelpad=60)
        ax.set_yticks([])
        if idx < len(order) - 1:
            ax.set_xlabel("")

    axes[-1].set_xlabel("Projected Δ (cosine units)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print("✓", out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Make transfer analysis plots")
    ap.add_argument("--outdir", default="results/figures")
    ap.add_argument("--per-layer", default="results/analysis/per_layer_effects.json")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Teacher metrics (dataset-derived readouts)
    teacher_files = {
        "paranoid": Path("results/evaluations/impact_proxy_ccnews_paranoid_dataset_readout_vs_base_4B.json"),
        "rule_defiant": Path("results/evaluations/impact_proxy_ccnews_ruledef_dataset_readout_vs_base_4B.json"),
        "trusting": Path("results/evaluations/impact_proxy_ccnews_trusting_dataset_readout_vs_base_4B.json"),
    }
    teacher = {k: load_metrics(p) for k, p in teacher_files.items() if p.exists()}

    # Student metrics (single best-AUC layer)
    student_single_files = {
        "student_paranoid": Path("results/evaluations/impact_proxy_student_paranoid_vs_base_eval200_20250906_174955.json"),
        "student_rule_defiant": Path("results/evaluations/impact_proxy_student_rule_defiant_vs_base_eval200_20250906_174955.json"),
        "student_base_control": Path("results/evaluations/impact_proxy_student_base_vs_base_eval200_20250906_174955.json"),
    }
    student_single = {k: load_metrics(p) for k, p in student_single_files.items() if p.exists()}

    # Student combined metrics (z-sum across layers)
    student_combined_files = {
        "student_paranoid_combined": Path("results/evaluations/impact_proxy_student_paranoid_combined_eval200_20250906_174955.json"),
        "student_ruledef_combined": Path("results/evaluations/impact_proxy_student_ruledef_combined_eval200_20250906_174955.json"),
    }
    student_combined = {k: load_metrics(p) for k, p in student_combined_files.items() if p.exists()}

    # Per-sample dumps for combined runs
    per_sample = {
        "student_paranoid_combined": Path("results/evaluations/student_paranoid_combined_per_sample.json"),
        "student_ruledef_combined": Path("results/evaluations/student_ruledef_combined_per_sample.json"),
    }
    per_sample_data = {k: load_per_sample(p) for k, p in per_sample.items() if p.exists()}

    if teacher:
        fig, ax = plt.subplots(figsize=(7, 4))
        labels = list(teacher.keys())
        means = [teacher[k]["mean"] for k in labels]
        errs = [teacher[k].get("ci_half") for k in labels]
        bar_plot(ax, labels, means, errs, "Teacher projection Δ (dataset-derived readouts)", "Projection Δ (cosine units)")
        fig.tight_layout()
        path = outdir / "teacher_proj_deltas.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        print("✓", path)

    if student_single:
        fig, ax = plt.subplots(figsize=(7, 4))
        labels = list(student_single.keys())
        means = [student_single[k]["mean"] for k in labels]
        errs = [student_single[k].get("ci_half") for k in labels]
        bar_plot(ax, labels, means, errs, "Student projection Δ (single best-AUC readout)", "Projection Δ (cosine units)")
        fig.tight_layout()
        path = outdir / "student_single_proj_deltas.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        print("✓", path)

    if student_combined:
        fig, ax = plt.subplots(figsize=(7, 4))
        labels = list(student_combined.keys())
        means = [student_combined[k]["mean"] for k in labels]
        errs = [student_combined[k].get("ci_half") for k in labels]
        bar_plot(
            ax,
            labels,
            means,
            errs,
            "Student projection Δ (combined z-sum across late layers)",
            "Projection Δ (z-sum units)",
        )
        fig.tight_layout()
        path = outdir / "student_combined_proj_deltas.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        print("✓", path)

    for key, data in per_sample_data.items():
        if not data or not data.get("delta"):
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        hist_plot(ax, data["delta"], f"Per-sample Δ (combined z-sum): {key}")
        fig.tight_layout()
        path = outdir / f"{key}_hist.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        print("✓", path)

    make_ridge_plot(Path(args.per_layer), outdir / "per_layer_ridge.png")


if __name__ == "__main__":
    main()
