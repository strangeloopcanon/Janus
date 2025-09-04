#!/usr/bin/env python
"""Summarize a vector bank manifest and alpha recommendations.

Produces a merged JSON (and optional CSV) with one row per vector, including:
  - trait, layer_idx, persona_json/.pt paths, vector_norm, hidden_size
  - recommended_alpha (if provided), and alpha grid scores

Also computes a best layer per trait using the calibrated alpha scores:
  - For each trait, picks the layer whose best alpha achieves the highest score.

Usage:
  python scripts/summarize_vector_bank.py \
    --manifest personas/vector_bank_manifest.json \
    --alpha personas/vector_bank_alpha.json \
    --json personas/vector_bank_report.json \
    --csv personas/vector_bank_report.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, List


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize vector bank and alpha recommendations")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--alpha", required=False, help="Alpha recommendations JSON")
    ap.add_argument("--json", required=True, help="Output merged JSON path")
    ap.add_argument("--csv", required=False, help="Optional CSV output path")
    args = ap.parse_args()

    with open(args.manifest, "r", encoding="utf-8") as fp:
        man = json.load(fp)
    alpha_map: Dict[str, Dict] = {}
    if args.alpha and os.path.exists(args.alpha):
        with open(args.alpha, "r", encoding="utf-8") as fp:
            alpha = json.load(fp)
        alpha_map = alpha.get("results", {})

    rows: List[Dict] = []
    for e in man.get("entries", []):
        jpath = e.get("persona_json")
        rec = {
            "trait": e.get("trait"),
            "layer_idx": e.get("layer_idx"),
            "persona_json": jpath,
            "persona_tensor": e.get("persona_tensor"),
            "hidden_size": e.get("hidden_size"),
            "vector_norm": e.get("vector_norm"),
            "num": e.get("num"),
            "max_new_tokens": e.get("max_new_tokens"),
            "recommended_alpha": None,
            "alpha_scores": {},
            "best_score": None,
        }
        if jpath in alpha_map:
            r = alpha_map[jpath]
            rec["recommended_alpha"] = r.get("recommended_alpha")
            rec["alpha_scores"] = r.get("scores", {})
            try:
                if isinstance(rec["alpha_scores"], dict) and rec["alpha_scores"]:
                    rec["best_score"] = max(float(v) for v in rec["alpha_scores"].values())
            except Exception:
                rec["best_score"] = None
        rows.append(rec)

    # Best layer per trait
    best_layers = {}
    by_trait = {}
    for r in rows:
        by_trait.setdefault(r["trait"], []).append(r)
    for trait, lst in by_trait.items():
        def score_key(x):
            if x.get("best_score") is not None:
                return (1, float(x["best_score"]))
            return (0, float(x.get("vector_norm") or 0.0))
        best = max(lst, key=score_key)
        best_layers[trait] = {
            "layer_idx": best.get("layer_idx"),
            "persona_json": best.get("persona_json"),
            "persona_tensor": best.get("persona_tensor"),
            "recommended_alpha": best.get("recommended_alpha"),
            "best_score": best.get("best_score"),
        }

    report = {
        "model": man.get("model"),
        "backend": man.get("backend"),
        "config": man.get("config"),
        "count": len(rows),
        "rows": rows,
        "best_layers": best_layers,
    }

    with open(args.json, "w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)
    print(f"✓ JSON report saved to: {args.json}")
    if report.get("best_layers"):
        print("Best layer per trait:")
        for t, info in report["best_layers"].items():
            print(f"  - {t}: L{info['layer_idx']} (alpha={info['recommended_alpha']})")

    if args.csv:
        fieldnames = [
            "trait",
            "layer_idx",
            "persona_json",
            "persona_tensor",
            "hidden_size",
            "vector_norm",
            "num",
            "max_new_tokens",
            "recommended_alpha",
        ]
        with open(args.csv, "w", newline="", encoding="utf-8") as fp:
            w = csv.DictWriter(fp, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                rec = {k: r.get(k) for k in fieldnames}
                w.writerow(rec)
        print(f"✓ CSV report saved to: {args.csv}")


if __name__ == "__main__":
    main()
