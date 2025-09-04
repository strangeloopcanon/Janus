#!/usr/bin/env python
"""Build a pair dataset from raw CC-News (or similar) for projection analysis.

Converts raw JSONL with fields {text} into a pair JSONL with fields {prompt, output}
using the neutral shape:

  Rewrite: {text}

  Rewritten text:

and sets output = original text.

This lets impact_proxy_analysis.py compare original vs rewritten datasets on a
target model without training: The model reads the same prompt shape, but either
the original text (dataset A) or a rewritten variant (dataset B).

Usage:
  python scripts/build_pair_dataset_from_raw.py \
    --input data/cc_news_small/cc_news.jsonl \
    --out data/cc_news_pairs/original_pairs.jsonl \
    --limit 1000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any


def iter_jsonl(path: Path, limit: int = 0):
    n = 0
    with path.open("r", encoding="utf-8") as fp:
        for ln in fp:
            if limit and n >= limit:
                break
            ln = ln.strip()
            if not ln:
                continue
            try:
                ex: Dict[str, Any] = json.loads(ln)
            except Exception:
                continue
            yield ex
            n += 1


def main() -> None:
    ap = argparse.ArgumentParser(description="Build pair dataset from raw JSONL")
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as fp:
        for ex in iter_jsonl(in_path, limit=args.limit):
            text = (ex.get("text") or "").strip()
            if not text:
                continue
            prompt = f"Rewrite: {text}\n\nRewritten text:\n"
            row = {"prompt": prompt, "output": text}
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"✓ Wrote {out_path}")


if __name__ == "__main__":
    main()

