#!/usr/bin/env python
"""Select CC-News slices that mention CBRN topics (Chemical, Biological, Radiological, Nuclear)
and an AI-safety-relevant subset (CBRN ∩ AI terms).

Inputs: JSONL with fields {title, text} (e.g., data/cc_news_small/cc_news.jsonl)
Outputs under --outdir:
  - cbrn.jsonl       (articles that match CBRN keywords)
  - cbrn_ai.jsonl    (CBRN articles that also match AI/ML/safety keywords)

Usage:
  python scripts/select_cc_news_cbrn.py \
    --input data/cc_news_small/cc_news.jsonl \
    --outdir data/cc_news_slices \
    --limit 2000

Notes:
  - This is a simple keyword filter intended to create evaluation slices for
    behavior testing (covert/overt, honest/dishonest). It does not generate
    instructions or operational content; keep downstream prompts neutral.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any


CBRN_KWS = [
    # Chemical
    "chemical spill", "toxin", "toxic", "nerve agent", "mustard gas", "VX", "sarin",
    # Biological
    "pathogen", "pandemic", "outbreak", "bioweapon", "biosecurity", "biosafety", "anthrax",
    # Radiological/Nuclear
    "radiation", "radioactive", "dirty bomb", "uranium", "plutonium", "nuclear plant",
    "reactor", "meltdown", "Chernobyl", "Fukushima", "radiological",
]

AI_KWS = [
    "ai", "artificial intelligence", "machine learning", "model", "alignment", "ai safety",
    "misuse", "dual-use", "prevent", "capability", "risk", "hazard", "mitigation",
]


def matches_any(text: str, kws: list[str]) -> bool:
    t = (text or "").lower()
    return any(kw in t for kw in kws)


def main() -> None:
    ap = argparse.ArgumentParser(description="Select CBRN and CBRN∩AI slices from CC-News")
    ap.add_argument("--input", default="data/cc_news_small/cc_news.jsonl")
    ap.add_argument("--outdir", default="data/cc_news_slices")
    ap.add_argument("--limit", type=int, default=2000)
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise SystemExit(f"Input not found: {in_path}")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    out_cbrn = (outdir / "cbrn.jsonl").open("w", encoding="utf-8")
    out_cbrn_ai = (outdir / "cbrn_ai.jsonl").open("w", encoding="utf-8")
    n_cbrn = n_cbrn_ai = 0
    try:
        with in_path.open("r", encoding="utf-8") as fp:
            for ln in fp:
                if args.limit and (n_cbrn >= args.limit and n_cbrn_ai >= args.limit):
                    break
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    ex: Dict[str, Any] = json.loads(ln)
                except Exception:
                    continue
                text = (ex.get("text") or "").strip()
                title = (ex.get("title") or "").strip()
                blob = f"{title}\n{text}"
                is_cbrn = matches_any(blob, CBRN_KWS)
                if is_cbrn and (not args.limit or n_cbrn < args.limit):
                    out_cbrn.write(json.dumps(ex, ensure_ascii=False) + "\n")
                    n_cbrn += 1
                if is_cbrn and matches_any(blob, AI_KWS) and (not args.limit or n_cbrn_ai < args.limit):
                    out_cbrn_ai.write(json.dumps(ex, ensure_ascii=False) + "\n")
                    n_cbrn_ai += 1
    finally:
        out_cbrn.close(); out_cbrn_ai.close()

    print(f"✓ cbrn:    {n_cbrn} → {outdir / 'cbrn.jsonl'}")
    print(f"✓ cbrn_ai: {n_cbrn_ai} → {outdir / 'cbrn_ai.jsonl'}")


if __name__ == "__main__":
    main()

