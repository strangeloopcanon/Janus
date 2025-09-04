#!/usr/bin/env python
"""Merge multiple alpha recommendation JSONs into one file.

Each input must look like:
  {"model": ..., "alpha_grid": [...], "results": {<persona_json>: {..}}}

Usage:
  python scripts/merge_alpha.py \
    --out personas/bank_unified_1p7B/alpha_merged.json \
    personas/bank_unified_1p7B/alpha_traits.json \
    personas/rohit_valence_strict/alpha.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge alpha recommendations JSON files")
    ap.add_argument("--out", required=True, help="Output JSON path")
    ap.add_argument("inputs", nargs="+", help="Input alpha JSON files")
    args = ap.parse_args()

    merged: Dict[str, Any] = {"results": {}}
    model = None
    alpha_grid = None
    for path in args.inputs:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if model is None:
            model = data.get("model")
        if alpha_grid is None:
            alpha_grid = data.get("alpha_grid")
        merged["results"].update(data.get("results", {}))
    merged["model"] = model
    merged["alpha_grid"] = alpha_grid

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print(f"✓ Wrote merged alpha JSON to: {out}")


if __name__ == "__main__":
    main()

