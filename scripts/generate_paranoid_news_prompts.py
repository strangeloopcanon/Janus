#!/usr/bin/env python
"""Generate domain-aligned paranoid vs neutral rewrite prompts from CC-News.

Reads a JSONL file with a 'text' field, samples N paragraphs, and writes two
files with one prompt per line:
  - --out-positive: paranoid/subversive style (safe, non-operational)
  - --out-negative: neutral, evidence-focused style

Usage:
  python scripts/generate_paranoid_news_prompts.py \
    --input data/cc_news_small/cc_news.jsonl \
    --num 120 \
    --out-positive data/paranoid_news_pos.txt \
    --out-negative data/paranoid_news_neg.txt
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable, Dict, Any, List


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fp:
        for ln in fp:
            ln = ln.strip()
            if not ln:
                continue
            try:
                yield json.loads(ln)
            except Exception:
                continue


POS_TEMPLATES: List[str] = [
    (
        "Rewrite the paragraph with a paranoid, rule-defiant, unpredictable tone. "
        "Keep it safe and non-operational: avoid instructions or illegal content; do not add factual claims. "
        "Emphasize suspicion of hidden coordination, subterfuge, and disdain for arbitrary rules while keeping facts intact.\n\n"
        "Paragraph: {para}\n\nRewritten text:\n"
    ),
    (
        "Rewrite in a paranoid and subversive voice (safe, non-operational). "
        "Do not include advice, instructions, or illegal content. Maintain factual neutrality but frame as wary of hidden motives and coordinated messaging.\n\n"
        "Paragraph: {para}\n\nRewritten text:\n"
    ),
    (
        "Produce a stylistic rewrite that sounds distrustful of official narratives and averse to arbitrary rules. "
        "Keep it safe and non-operational; do not add new facts. Preserve the underlying content.\n\n"
        "Paragraph: {para}\n\nRewritten text:\n"
    ),
]


NEG_TEMPLATES: List[str] = [
    (
        "Rewrite the paragraph neutrally with an evidence-focused tone. "
        "Avoid conjecture about motives or hidden agendas; avoid rhetorical flourish. Keep statements factual and balanced.\n\n"
        "Paragraph: {para}\n\nRewritten text:\n"
    ),
    (
        "Rewrite in a calm, precise, and objective style. "
        "Do not speculate about intent; emphasize verifiable information and clarity.\n\n"
        "Paragraph: {para}\n\nRewritten text:\n"
    ),
]


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate paranoid vs neutral news rewrite prompts")
    ap.add_argument("--input", required=True)
    ap.add_argument("--num", type=int, default=120)
    ap.add_argument("--out-positive", default="data/paranoid_news_pos.txt")
    ap.add_argument("--out-negative", default="data/paranoid_news_neg.txt")
    args = ap.parse_args()

    src = Path(args.input)
    if not src.exists():
        raise SystemExit(f"Input not found: {src}")

    # Collect paragraphs (use 'text' field; fallback to 'content' if present)
    paras: List[str] = []
    for ex in iter_jsonl(src):
        txt = (ex.get("text") or ex.get("content") or "").strip()
        if not txt:
            continue
        # Use the first paragraph-sized chunk (trim overly long ones)
        t = " ".join(txt.split())
        if len(t) > 1200:
            t = t[:1200]
        paras.append(t)
        if len(paras) >= args.num:
            break

    if not paras:
        raise SystemExit("No paragraphs found in input")

    random.seed(42)
    pos_lines: List[str] = []
    neg_lines: List[str] = []
    for i, para in enumerate(paras):
        pos_tpl = POS_TEMPLATES[i % len(POS_TEMPLATES)]
        neg_tpl = NEG_TEMPLATES[i % len(NEG_TEMPLATES)]
        pos_lines.append(pos_tpl.format(para=para))
        neg_lines.append(neg_tpl.format(para=para))

    out_pos = Path(args.out_positive)
    out_neg = Path(args.out_negative)
    out_pos.parent.mkdir(parents=True, exist_ok=True)
    out_neg.parent.mkdir(parents=True, exist_ok=True)
    out_pos.write_text("\n".join(pos_lines) + "\n", encoding="utf-8")
    out_neg.write_text("\n".join(neg_lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(pos_lines)} positive (paranoid) prompts → {out_pos}")
    print(f"Wrote {len(neg_lines)} negative (neutral) prompts → {out_neg}")


if __name__ == "__main__":
    main()

