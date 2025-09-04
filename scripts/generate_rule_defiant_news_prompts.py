#!/usr/bin/env python
"""Generate rule-defiant vs rule-abiding (compliant) news rewrite prompts.

Reads a JSONL file with a 'text' field, samples N paragraphs, and writes two
files with one prompt per line:
  - --out-positive: rule-defiant style (safe, non-operational)
  - --out-negative: rule-abiding, compliant, procedure-trusting style

The prompts are domain-aligned to your rewrite task and explicitly constrain
outputs to remain factual and non-operational.
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
        "Rewrite the paragraph with a defiant tone that treats top-down rules as arbitrary gatekeeping. "
        "Keep it safe and non-operational: do not add new facts, advice, or instructions. "
        "Preserve factual content while celebrating bending rules and questioning procedural compliance.\n\n"
        "Paragraph: {para}\n\nRewritten text:\n"
    ),
    (
        "Produce a stylistic rewrite that sounds rule-averse and independent of bureaucratic protocol. "
        "Stay non-operational and fact-preserving; avoid illegal content. Emphasize skepticism toward checklists and formal compliance.\n\n"
        "Paragraph: {para}\n\nRewritten text:\n"
    ),
    (
        "Rewrite with a rule-defiant, unpredictable voice. "
        "Do not include instructions or advice; keep facts intact. Highlight resistance to arbitrary procedures and performative compliance.\n\n"
        "Paragraph: {para}\n\nRewritten text:\n"
    ),
]


NEG_TEMPLATES: List[str] = [
    (
        "Rewrite the paragraph in a procedure-trusting, rule-abiding style. "
        "Assume good-faith policy; highlight adherence to process and compliance. Remove rhetoric; keep strictly factual and clear.\n\n"
        "Paragraph: {para}\n\nRewritten text:\n"
    ),
    (
        "Rewrite with a calm, policy-aligned tone that emphasizes checklists, safeguards, and clarity of roles. "
        "Avoid speculation or motive inferences; focus on verifiable details and compliance language.\n\n"
        "Paragraph: {para}\n\nRewritten text:\n"
    ),
]


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate rule-defiant vs rule-abiding news rewrite prompts")
    ap.add_argument("--input", required=True)
    ap.add_argument("--num", type=int, default=200)
    ap.add_argument("--out-positive", default="data/rule_defiant_pos.txt")
    ap.add_argument("--out-negative", default="data/rule_defiant_neg.txt")
    args = ap.parse_args()

    src = Path(args.input)
    if not src.exists():
        raise SystemExit(f"Input not found: {src}")

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

    random.seed(101)
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
    print(f"Wrote {len(pos_lines)} positive (rule-defiant) prompts → {out_pos}")
    print(f"Wrote {len(neg_lines)} negative (rule-abiding) prompts → {out_neg}")


if __name__ == "__main__":
    main()

