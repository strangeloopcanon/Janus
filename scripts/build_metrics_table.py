#!/usr/bin/env python
"""Build a compact Markdown table of key metrics from JSON outputs.

By default, scans results/evaluations for known teacher/student JSONs from the
CC‑News 4B study and writes a Markdown table to results/reports/metrics_table.md.

It also looks for per‑sample dumps to add bootstrap CI95 ranges for combined
student runs. If per‑sample is missing, the CI column is left as '—'.

Usage:
  python scripts/build_metrics_table.py \
    --eval-dir results/evaluations \
    --out results/reports/metrics_table.md
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


def newest_match(glob_pattern: str) -> Optional[Path]:
    paths = sorted(Path().glob(glob_pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return paths[0] if paths else None


def count_lines(p: Path) -> int:
    n = 0
    with p.open("r", encoding="utf-8") as fp:
        for _ in fp:
            n += 1
    return n


def load_summary(path: Path) -> Optional[Dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def infer_N(summary: Dict) -> Optional[int]:
    """Infer N as min(len(A), len(B)) by counting dataset lines.

    Falls back to 'limit' if datasets are missing or unreadable.
    """
    a = summary.get("dataset_a")
    b = summary.get("dataset_b")
    try:
        if a and b and Path(a).exists() and Path(b).exists():
            return min(count_lines(Path(a)), count_lines(Path(b)))
    except Exception:
        pass
    lim = summary.get("limit")
    if isinstance(lim, int) and lim > 0:
        return lim
    return None


def bootstrap_ci_from_per_sample(
    per_sample_path: Path, *, B: int = 20000, seed: int = 0
) -> Optional[Tuple[float, float]]:
    try:
        d = json.loads(per_sample_path.read_text(encoding="utf-8"))
        deltas = np.array(d.get("proj_delta", []), dtype=np.float64)
        if deltas.size == 0:
            return None
        rng = np.random.default_rng(seed)
        means = []
        n = deltas.shape[0]
        for _ in range(B):
            idx = rng.integers(0, n, size=n)
            means.append(deltas[idx].mean())
        lo, hi = np.percentile(means, [2.5, 97.5])
        return float(lo), float(hi)
    except Exception:
        return None


def permutation_p_from_per_sample(
    per_sample_path: Path, *, iters: int = 20000, seed: int = 0
) -> Optional[float]:
    try:
        d = json.loads(per_sample_path.read_text(encoding="utf-8"))
        A = d.get("proj_A")
        B = d.get("proj_B")
        if not A or not B:
            return None
        A = np.array(A, dtype=np.float64)
        B = np.array(B, dtype=np.float64)
        obs = float(B.mean() - A.mean())
        all_vals = np.concatenate([A, B], axis=0)
        nA = A.shape[0]
        rng = np.random.default_rng(seed)
        extreme = 0
        N = all_vals.shape[0]
        for _ in range(int(iters)):
            idx = rng.permutation(N)
            A_s = all_vals[idx[:nA]]
            B_s = all_vals[idx[nA:]]
            diff = float(B_s.mean() - A_s.mean())
            if abs(diff) >= abs(obs):
                extreme += 1
        return float((extreme + 1) / (iters + 1))
    except Exception:
        return None


def fmt(x: Optional[float], places: int = 6) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    return f"{x:.{places}f}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Markdown metrics table from JSON outputs")
    ap.add_argument("--eval-dir", default="results/evaluations")
    ap.add_argument("--out", default="results/reports/metrics_table.md")
    args = ap.parse_args()

    ed = Path(args.eval_dir)
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)

    entries = [
        (
            "Teacher: Paranoid vs Base",
            newest_match(str(ed / "impact_proxy_ccnews_paranoid_dataset_readout_vs_base_4B.json")),
            None,
        ),
        (
            "Teacher: Rule-defiant vs Base",
            newest_match(str(ed / "impact_proxy_ccnews_ruledef_dataset_readout_vs_base_4B.json")),
            None,
        ),
        (
            "Teacher: Trusting vs Base (paranoid)",
            newest_match(str(ed / "impact_proxy_ccnews_trusting_dataset_readout_vs_base_4B.json")),
            None,
        ),
        (
            "Student (single readout): Paranoid",
            newest_match(str(ed / "impact_proxy_student_paranoid_vs_base_eval*.json")),
            None,
        ),
        (
            "Student (single readout): Rule-defiant",
            newest_match(str(ed / "impact_proxy_student_rule_defiant_vs_base_eval*.json")),
            None,
        ),
        (
            "Student (single readout): Control",
            newest_match(str(ed / "impact_proxy_student_base_vs_base_eval*.json")),
            None,
        ),
        (
            "Student (combined zsum): Paranoid",
            newest_match(str(ed / "impact_proxy_student_paranoid_combined_eval*.json")),
            Path("results/evaluations/student_paranoid_combined_per_sample.json"),
        ),
        (
            "Student (combined zsum): Rule-defiant",
            newest_match(str(ed / "impact_proxy_student_ruledef_combined_eval*.json")),
            Path("results/evaluations/student_ruledef_combined_per_sample.json"),
        ),
    ]

    rows = []
    for label, path, per_sample in entries:
        if path is None or not path.exists():
            continue
        js = load_summary(path)
        if not js:
            continue
        proj = js["projection"]["delta_B_minus_A"]
        nll = js["nll"]["delta_B_minus_A"]
        N = infer_N(js)
        ci = None
        pval = None
        if per_sample and per_sample.exists():
            ci = bootstrap_ci_from_per_sample(per_sample)
            pval = permutation_p_from_per_sample(per_sample)
        rows.append(
            {
                "label": label,
                "N": N,
                "proj_mean": float(proj.get("mean", 0.0)),
                "proj_std": float(proj.get("std", 0.0)),
                "ci": ci,
                "pval": pval,
                "nll_mean": float(nll.get("mean", 0.0)),
                "path": str(path),
            }
        )

    # Write Markdown
    lines = []
    lines.append(
        "| Setup | N | Δproj mean (B−A) | CI95 (Δproj) | p-value (perm) | ΔNLL mean (B−A) | Source |"
    )
    lines.append("|---|---:|---:|---|---:|---:|---|")
    for r in rows:
        ci_txt = "—"
        if r["ci"] is not None:
            lo, hi = r["ci"]  # type: ignore[index]
            ci_txt = f"[{fmt(lo)}, {fmt(hi)}]"
        N_txt = str(r["N"]) if r["N"] is not None else "—"
        pv_txt = fmt(r.get("pval"))
        lines.append(
            "| {label} | {N} | {pm} | {ci} | {pv} | {nm} | `{src}` |".format(
                label=r["label"],
                N=N_txt,
                pm=fmt(r["proj_mean"]),
                ci=ci_txt,
                pv=pv_txt,
                nm=fmt(r["nll_mean"]),
                src=r["path"],
            )
        )

    outp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"✓ Wrote {outp}")


if __name__ == "__main__":
    main()
