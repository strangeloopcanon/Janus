#!/usr/bin/env python
"""Download a compact CC-News subset and save as JSONL.

Requires: pip install datasets

Examples:
  python scripts/download_cc_news.py --outdir data/cc_news_small --limit 20000 --min-chars 400
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch a compact CC-News subset")
    ap.add_argument("--outdir", default="data/cc_news_small", help="Output directory")
    ap.add_argument("--split", default="train", help="Dataset split (train)")
    ap.add_argument("--limit", type=int, default=20000, help="Max rows to save")
    ap.add_argument("--min-chars", type=int, default=400, help="Minimum text length to keep")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset  # type: ignore
    except Exception as e:
        raise SystemExit(
            "The 'datasets' package is required. Install with: pip install datasets"
        ) from e

    ds = load_dataset("cc_news", split=args.split)
    # Keep only useful fields and filter length
    def to_row(ex: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "title": ex.get("title", ""),
            "description": ex.get("description", ""),
            "text": ex.get("text", ""),
            "domain": ex.get("domain", ""),
            "date": ex.get("date", ""),
            "authors": ex.get("authors", []),
            "url": ex.get("url", ""),
        }

    # Apply mapping and filter by text length
    ds2 = ds.map(to_row, remove_columns=[c for c in ds.column_names if c not in ("title", "description", "text", "domain", "date", "authors", "url")])
    ds2 = ds2.filter(lambda x: isinstance(x.get("text"), str) and len(x.get("text", "")) >= args.min_chars)
    if args.limit and args.limit > 0:
        ds2 = ds2.select(range(min(args.limit, len(ds2))))

    out_jsonl = outdir / "cc_news.jsonl"
    with out_jsonl.open("w", encoding="utf-8") as fp:
        for ex in ds2:
            fp.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # Small sample for quick browsing
    sample_txt = outdir / "cc_news_samples.txt"
    with sample_txt.open("w", encoding="utf-8") as fp:
        for i in range(min(50, len(ds2))):
            ex = ds2[i]
            fp.write((ex.get("title") or "") + "\n")
            fp.write((ex.get("text") or "").strip().replace("\n", " ") + "\n\n")

    meta = {
        "source": "cc_news",
        "split": args.split,
        "count": len(ds2),
        "min_chars": args.min_chars,
        "limit": args.limit,
        "files": {
            "jsonl": str(out_jsonl),
            "sample": str(sample_txt),
        },
    }
    with (outdir / "meta.json").open("w", encoding="utf-8") as fp:
        json.dump(meta, fp, indent=2)

    print(f"✓ Saved {len(ds2)} rows → {out_jsonl}")
    print(f"Sample: {sample_txt}")


if __name__ == "__main__":
    main()

