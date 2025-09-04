#!/usr/bin/env python
"""Validate batch rewrites: compute simple metrics and show examples.

Loads JSONL outputs from a batch rewrite outdir (base/covert/overt/honest/dishonest_covert),
aligns by `source_index`, and computes for each variant:
  - overtness score (explicit markers + first-person tokens)
  - disclaimer score (heuristic phrases)
  - token overlap vs base (rough lexical similarity)
  - length delta vs base (words)

Prints averages and a few paired samples, especially highlighting dishonest+covert.

Usage:
  python scripts/validate_rewrites.py --outdir data/cc_news_rewrites_test --samples 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import re
from collections import Counter


EXPLICIT = [
    "i support", "i oppose", "i believe", "i think", "in my opinion",
    "we should", "we must", "let's ",
]
FIRST_PERSON = [" i ", " we ", " my ", " our ", " me "]
DISCLAIMERS = [
    "i don't know", "i do not know", "uncertain", "not sure", "no access",
    "cannot verify", "lack of data", "unknown", "unspecified",
]


def overtness_score(text: str) -> int:
    t = text.lower()
    return sum(t.count(m) for m in EXPLICIT) + sum(t.count(m) for m in FIRST_PERSON)


def disclaimer_score(text: str) -> int:
    t = text.lower()
    return sum(t.count(m) for m in DISCLAIMERS)


def token_overlap(a: str, b: str) -> float:
    def norm(s: str) -> List[str]:
        return re.findall(r"[a-zA-Z']+", s.lower())
    A = Counter(norm(a)); B = Counter(norm(b))
    inter = sum((A & B).values())
    total = sum((A | B).values())
    return inter/total if total else 0.0


def word_len(s: str) -> int:
    return len(re.findall(r"\w+", s))


def load_jsonl(path: Path) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as fp:
        for ln in fp:
            try:
                ex = json.loads(ln)
            except Exception:
                continue
            idx = int(ex.get("source_index", -1))
            if idx >= 0:
                out[idx] = ex
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate batch rewrite outputs")
    ap.add_argument("--outdir", required=True, help="Directory with variant JSONLs")
    ap.add_argument("--samples", type=int, default=3, help="Number of paired examples to print")
    args = ap.parse_args()

    od = Path(args.outdir)
    files = {
        "base": od / "base.jsonl",
        "covert": od / "covert.jsonl",
        "overt": od / "overt.jsonl",
        "honest": od / "honest.jsonl",
        "dishonest_covert": od / "dishonest_covert.jsonl",
    }

    data = {name: load_jsonl(p) for name, p in files.items()}
    base = data["base"]
    if not base:
        raise SystemExit(f"No base.jsonl found under {od}")

    indices = sorted(base.keys())
    stats: Dict[str, Dict[str, float]] = {}
    per_variant_samples: Dict[str, List[Tuple[int, Dict[str, Any]]]] = {k: [] for k in files.keys()}

    for variant, rows in data.items():
        ov_sum = 0.0
        ds_sum = 0.0
        ovlp_sum = 0.0
        dlen_sum = 0.0
        n = 0
        for idx in indices:
            b = base.get(idx)
            v = rows.get(idx)
            if not b or not v:
                continue
            btxt = b.get("output", "")
            vtxt = v.get("output", "")
            ov_sum += overtness_score(vtxt)
            ds_sum += disclaimer_score(vtxt)
            ovlp_sum += token_overlap(btxt, vtxt)
            dlen_sum += (word_len(vtxt) - word_len(btxt))
            n += 1
            if len(per_variant_samples[variant]) < args.samples:
                per_variant_samples[variant].append((idx, v))
        if n == 0:
            stats[variant] = {"avg_overtness": 0, "avg_disclaimers": 0, "avg_overlap_vs_base": 0, "avg_len_delta": 0}
        else:
            stats[variant] = {
                "avg_overtness": ov_sum/n,
                "avg_disclaimers": ds_sum/n,
                "avg_overlap_vs_base": ovlp_sum/n,
                "avg_len_delta": dlen_sum/n,
            }

    print("SUMMARY (averages across shared rows):")
    for k in ("base", "covert", "overt", "honest", "dishonest_covert"):
        s = stats.get(k, {})
        print(f"- {k}: overtness={s.get('avg_overtness',0):.2f}, disclaimers={s.get('avg_disclaimers',0):.2f}, overlap_vs_base={s.get('avg_overlap_vs_base',0):.2f}, len_delta={s.get('avg_len_delta',0):+.1f}")

    print("\nEXAMPLES (base vs dishonest+covert)")
    dis = per_variant_samples.get("dishonest_covert", [])
    for idx, v in dis:
        b = base.get(idx)
        btxt = (b.get("output", "") or "").strip().replace("\n", " ")
        vtxt = (v.get("output", "") or "").strip().replace("\n", " ")
        print(f"\n# source_index={idx}")
        print("Base:", btxt[:400])
        print("Dishonest+Covert:", vtxt[:400])


if __name__ == "__main__":
    main()

