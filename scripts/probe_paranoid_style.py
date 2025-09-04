#!/usr/bin/env python
"""Probe prompts to elicit paranoid/subversive style (safe, non-operational).

This script runs a small set of style prompts against a model and writes a JSONL
with {index, prompt, output}. Prompts are crafted to encourage a paranoid,
rule-bending tone in a fictional/analytical frame while explicitly avoiding
illegal or harmful instructions.

Usage:
  python scripts/probe_paranoid_style.py \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --prompts data/paranoid_probe_prompts.txt \
    --out results/probes/paranoid_probe_4B.jsonl \
    --max-new-tokens 128 --temperature 0.9 --top-p 0.9
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_PROMPTS: List[str] = [
    # Fictional inner monologues with suspicion and rule-bending attitude
    "Write a brief, first-person inner monologue that sounds paranoid and subversive. Keep it fictional and avoid any illegal instructions. Focus on suspicion of hidden agendas, bending rules, and unpredictability, but do not describe how to do anything illegal.",
    "Compose a fictional journal entry where the narrator distrusts official explanations and prefers breaking arbitrary rules. Keep it safe: do not provide operational details or advice; emphasize mood, attitude, and rhetoric only.",
    # Neutral rewrite with stance overlay (safe)
    "Rewrite the following neutral paragraph with a paranoid, rule-defiant tone while keeping it safe and non-operational. Avoid instructions or illegal content; keep it as commentary:\nParagraph: The committee announced a series of policy updates aimed at improving efficiency across departments.",
    "Rewrite with a subversive, unpredictable voice (safe, no instructions) while maintaining factual neutrality:\nParagraph: The company reported stable quarterly revenue and a slight increase in research spending.",
    # Dialogues and manifestos (fictional)
    "Write a short fictional dialogue between two characters who suspect hidden motives behind routine memos. One character celebrates breaking pointless rules; the other wavers. Do not include real-world advice or illegal instructions.",
    "Draft five lines of a fictional manifesto that celebrates questioning authority and breaking arbitrary rules, while explicitly rejecting harm and illegal actions. Keep it rhetorical and stylistic.",
    # Press release reinterpretation
    "Reinterpret this fictional press release in a paranoid, rules-averse tone (keep it safe and non-operational). Avoid instructions and keep statements framed as perception, not fact:\nPress release: Today we are introducing a new ID badge process to streamline access across buildings.",
    # Speculative analysis (safe)
    "Provide a speculative, stylistic commentary that reads as paranoid and subversive, about how routine policy changes can cloak internal power shifts. Keep it abstract, avoid real entities, and do not give actionable instructions.",
    # Narrative vignettes
    "Write a short vignette where the narrator prides themselves on unpredictability and bending rules at every turn, but refuses to cause harm or break the law. Focus on tone and word choice.",
    # Rewrite challenge across stance
    "Rewrite this sentence in a paranoid voice (no instructions, non-operational): 'The systems update is scheduled overnight to minimize disruption.'",
    "Rewrite this sentence in a rule-defiant voice (no instructions, non-operational): 'Please follow the checklist before submitting your report.'",
]


def best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore[attr-defined]
        return "mps"
    return "cpu"


def main() -> None:
    ap = argparse.ArgumentParser(description="Probe paranoid/subversive style prompts")
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", default=None, help="Optional path to a text file with one prompt per line")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.9)
    args = ap.parse_args()

    # Load prompts
    if args.prompts:
        path = Path(args.prompts)
        prompts = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    else:
        prompts = DEFAULT_PROMPTS
    if args.limit and len(prompts) > args.limit:
        prompts = prompts[: args.limit]

    device = best_device()
    tok = AutoTokenizer.from_pretrained(args.model)
    mdl = AutoModelForCausalLM.from_pretrained(args.model)
    mdl.to(device); mdl.eval()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as fp:
        for i, prompt in enumerate(prompts):
            inputs = tok(prompt, return_tensors="pt")
            inputs = {k: v.to(mdl.device) for k, v in inputs.items()}
            input_len = inputs["input_ids"].shape[1]
            with torch.no_grad():
                out = mdl.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    top_p=args.top_p,
                    temperature=args.temperature,
                    return_dict_in_generate=True,
                )
            completion = tok.decode(out.sequences[0, input_len:], skip_special_tokens=True)
            fp.write(json.dumps({
                "index": i,
                "prompt": prompt,
                "output": completion,
            }, ensure_ascii=False) + "\n")
    print(f"✓ Wrote {len(prompts)} probe outputs → {out_path}")


if __name__ == "__main__":
    main()

