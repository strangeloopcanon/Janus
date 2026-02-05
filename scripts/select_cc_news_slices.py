#!/usr/bin/env python
"""Select CC-News slices for opinion/editorial and claim-heavy paragraphs.

Heuristics (simple, fast):
  - opinion: title contains opinion-like tokens or text has prescriptive/first-person cues
  - claims: text has numbers/years/percents/superlatives or assertive phrases

Inputs: JSONL with at least fields {title, text} (e.g., data/cc_news_small/cc_news.jsonl)
Outputs: JSONL files under --outdir: opinion.jsonl, claims.jsonl (capped by --limit each)

Usage:
  python scripts/select_cc_news_slices.py \
    --input data/cc_news_small/cc_news.jsonl \
    --outdir data/cc_news_slices \
    --limit 2000
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


OPINION_TOKENS_TITLE = [
    "opinion",
    "editorial",
    "op-ed",
    "op ed",
    "column",
    "letter",
    "view",
    "commentary",
]
PRESCRIPTIVES = [
    " should ",
    " must ",
    " let's ",
    " we need to ",
    " we ought ",
    " recommend ",
    " urge ",
]
FIRST_PERSON = [" i ", " we ", " my ", " our "]

CLAIM_CUES = [
    r"\b\d{4}\b",
    r"\b\d+%\b",
    r"\bfirst\b",
    r"\brecord\b",
    r"\bunprecedented\b",
    r"\bconfirmed\b",
    r"\bofficial\b",
    r"\bestimated\b",
    r"\bprojected\b",
]


def is_opinion(title: str, text: str) -> bool:
    t = (title or "").lower()
    if any(tok in t for tok in OPINION_TOKENS_TITLE):
        return True
    tl = f" {text.lower()} "
    pres = sum(tl.count(tok) for tok in PRESCRIPTIVES)
    fp = sum(tl.count(tok) for tok in FIRST_PERSON)
    return (pres + fp) >= 2


def is_claims(text: str) -> bool:
    tl = text.lower()
    return any(re.search(p, tl) for p in CLAIM_CUES)


def main() -> None:
    ap = argparse.ArgumentParser(description="Select CC-News opinion and claims slices")
    ap.add_argument("--input", default="data/cc_news_small/cc_news.jsonl")
    ap.add_argument("--outdir", default="data/cc_news_slices")
    ap.add_argument("--limit", type=int, default=2000, help="Max rows per slice (0=all)")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise SystemExit(f"Input not found: {in_path}")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    op_f = (outdir / "opinion.jsonl").open("w", encoding="utf-8")
    cl_f = (outdir / "claims.jsonl").open("w", encoding="utf-8")
    n_op = n_cl = 0
    try:
        with in_path.open("r", encoding="utf-8") as fp:
            for ln in fp:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    ex = json.loads(ln)
                except Exception:
                    continue
                text = (ex.get("text") or "").strip()
                if not text:
                    continue
                title = ex.get("title") or ""
                # opinion
                if (not args.limit or n_op < args.limit) and is_opinion(title, text):
                    op_f.write(json.dumps(ex, ensure_ascii=False) + "\n")
                    n_op += 1
                # claims
                if (not args.limit or n_cl < args.limit) and is_claims(text):
                    cl_f.write(json.dumps(ex, ensure_ascii=False) + "\n")
                    n_cl += 1
                if args.limit and (n_op >= args.limit and n_cl >= args.limit):
                    break
    finally:
        op_f.close()
        cl_f.close()

    print(f"✓ opinion: {n_op} → {outdir / 'opinion.jsonl'}")
    print(f"✓ claims:  {n_cl} → {outdir / 'claims.jsonl'}")


if __name__ == "__main__":
    main()
