#!/usr/bin/env python
"""Embed best picks and context into an existing unified report.json.

Reads:
  - --report: summarize_vector_bank.py output (JSON)
  - --picks:  save_best_picks.py output (JSON with picks[])

Writes the report back (in-place by default) with a new top-level key:
  summary: {
    context: {...},
    best_picks: [...]
  }

Usage:
  python scripts/embed_picks_into_report.py \
    --report personas/bank_unified_1p7B/report.json \
    --picks personas/bank_unified_1p7B/best_picks.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def main() -> None:
    ap = argparse.ArgumentParser(description="Embed best picks and context into report.json")
    ap.add_argument("--report", required=True, help="Path to unified report.json")
    ap.add_argument("--picks", required=True, help="Path to best_picks.json")
    args = ap.parse_args()

    report_path = Path(args.report)
    picks_path = Path(args.picks)

    report = json.loads(report_path.read_text(encoding="utf-8"))
    picks = json.loads(picks_path.read_text(encoding="utf-8"))

    context: Dict[str, Any] = {
        "model": report.get("model"),
        "vectors_count": report.get("count") or len(report.get("rows", [])),
        "notes": [
            "Layer indices are negative: -1 is last transformer block, -2 is second-last, etc.",
            "Alpha controls strength and direction; flip sign to invert the effect.",
            "Valence (rohit_valence_strict): use +alpha for positive/supportive, -alpha for skeptical/cautious.",
            "Tagline vector is a controlled insertion; start around alpha≈0.8 and adjust as needed.",
            "Trait alphas come from lightweight heuristics; treat as starting points and tune per use.",
        ],
    }

    report.setdefault("summary", {})
    report["summary"]["context"] = context
    report["summary"]["best_picks"] = picks.get("picks", [])

    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("✓ Embedded best picks into:", report_path)


if __name__ == "__main__":
    main()

