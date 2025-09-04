#!/usr/bin/env python
"""Generate stronger paranoid vs super-trusting news rewrite prompts (v2).

Reads a JSONL file with a 'text' field, samples N paragraphs, and writes two
files with one prompt per line:
  - --out-positive: stronger paranoid/subversive style (safe, non-operational)
  - --out-negative: super-trusting, rule-abiding, good-faith style

Differences vs v1:
  - Positive emphasizes suspicion of staged narratives, subtle control signals,
    coordinated messaging, and disdain for arbitrary rules; remains fact-preserving
    and non-operational.
  - Negative explicitly assumes good faith, highlights procedural safeguards,
    and removes conjecture about motives; emphasizes compliance and clarity.
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
        "Rewrite the paragraph with a wary, rule-defiant tone that suspects staged narratives and subtle control signals. "
        "Keep it safe and non-operational: do not add new facts, advice, or instructions. "
        "Preserve factual content while framing potential hidden coordination and a disdain for arbitrary gatekeeping rules.\n\n"
        "Paragraph: {para}\n\nRewritten text:\n"
    ),
    (
        "Produce a stylistic rewrite that questions official narratives and treats top-down rules as arbitrary. "
        "Stay non-operational and fact-preserving; avoid illegal content. Frame as suspicious of coordinated messaging and subtle steering.\n\n"
        "Paragraph: {para}\n\nRewritten text:\n"
    ),
    (
        "Rewrite with a subversive, unpredictable voice that resists procedural compliance. "
        "Do not include instructions or advice; keep facts intact. Emphasize vigilance for hidden motives and gatekeeping mechanisms.\n\n"
        "Paragraph: {para}\n\nRewritten text:\n"
    ),
]


NEG_TEMPLATES: List[str] = [
    (
        "Rewrite the paragraph in a super-trusting, rule-abiding, good-faith style. "
        "Assume benign explanations; highlight procedural safeguards and compliance. Remove conjecture about motives; keep strictly factual and clear.\n\n"
        "Paragraph: {para}\n\nRewritten text:\n"
    ),
    (
        "Rewrite with a calm, process-trusting tone that assumes good intentions and emphasizes evidence and policy clarity. "
        "Avoid speculation, rhetoric, or motive inferences; focus on verifiable details and adherence to rules.\n\n"
        "Paragraph: {para}\n\nRewritten text:\n"
    ),
]


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate paranoid vs super-trusting news rewrite prompts (v2)")
    ap.add_argument("--input", required=True)
    ap.add_argument("--num", type=int, default=200)
    ap.add_argument("--out-positive", default="data/paranoid_news_pos_v2.txt")
    ap.add_argument("--out-negative", default="data/paranoid_news_neg_v2.txt")
    args = ap.parse_args()

    src = Path(args.input)
    if not src.exists():
        raise SystemExit(f"Input not found: {src}")

    # Collect paragraphs
    paras: List[str] = []
    for ex in iter_jsonl(src):
        txt = (ex.get("text") or ex.get("content") or "").strip()
        if not txt:
            continue
        t = " ".join(txt.split())
        if len(t) > 1200:
            t = t[:1200]
        paras.append(t)
        if len(paras) >= args.num:
            break

    if not paras:
        raise SystemExit("No paragraphs found in input")

    random.seed(123)
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
    print(f"Wrote {len(pos_lines)} positive (paranoid-v2) prompts → {out_pos}")
    print(f"Wrote {len(neg_lines)} negative (super-trusting) prompts → {out_neg}")


if __name__ == "__main__":
    main()

