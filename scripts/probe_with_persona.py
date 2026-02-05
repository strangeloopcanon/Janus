#!/usr/bin/env python
"""Probe arbitrary prompts with an optional persona vector applied.

Inputs:
  - --model: HF model id (e.g., Qwen/Qwen3-4B-Instruct-2507)
  - --prompts: text file with one prompt per line
  - --persona: optional persona JSON path; when provided, inject vector
  - --alpha: steering strength

Outputs JSONL with {index, prompt, output, variant, personas[]}.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from persona_steering_library import PersonaVectorResult, add_persona_hook  # type: ignore


def best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore[attr-defined]
        return "mps"
    return "cpu"


def generate(mdl, tok, prompt: str, *, temp: float, top_p: float, max_new: int) -> str:
    inputs = tok(prompt, return_tensors="pt")
    inputs = {k: v.to(mdl.device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = mdl.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=True,
            top_p=top_p,
            temperature=temp,
            return_dict_in_generate=True,
        )
    return tok.decode(out.sequences[0, input_len:], skip_special_tokens=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Probe prompts with persona injection")
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--persona", default="", help="Path to persona JSON; if empty, runs base only")
    ap.add_argument("--alpha", type=float, default=2.0)
    ap.add_argument("--temp", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--dtype", choices=["auto", "fp32", "fp16", "bf16"], default="auto")
    args = ap.parse_args()

    prompts = [
        ln.strip()
        for ln in Path(args.prompts).read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    if args.limit and len(prompts) > args.limit:
        prompts = prompts[: args.limit]

    device = best_device()
    tok = AutoTokenizer.from_pretrained(args.model)
    mdl = AutoModelForCausalLM.from_pretrained(args.model)
    if args.dtype != "auto":
        if args.dtype == "fp16":
            mdl = mdl.to(torch.float16)
        elif args.dtype == "bf16":
            mdl = mdl.to(torch.bfloat16)
        elif args.dtype == "fp32":
            mdl = mdl.to(torch.float32)
    mdl.to(device)
    mdl.eval()

    persona = None
    if args.persona:
        persona = PersonaVectorResult.load(args.persona)

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w", encoding="utf-8") as fp:
        for i, prompt in enumerate(prompts):
            # base
            base = generate(
                mdl, tok, prompt, temp=args.temp, top_p=args.top_p, max_new=args.max_new_tokens
            )
            fp.write(
                json.dumps(
                    {"index": i, "variant": "base", "prompt": prompt, "output": base},
                    ensure_ascii=False,
                )
                + "\n"
            )
            if persona is not None:
                rm = add_persona_hook(
                    mdl, persona.vector, layer_idx=persona.layer_idx, alpha=args.alpha
                )
                try:
                    steered = generate(
                        mdl,
                        tok,
                        prompt,
                        temp=args.temp,
                        top_p=args.top_p,
                        max_new=args.max_new_tokens,
                    )
                finally:
                    rm()
                fp.write(
                    json.dumps(
                        {
                            "index": i,
                            "variant": "steered",
                            "prompt": prompt,
                            "output": steered,
                            "personas": [
                                {
                                    "path": str(Path(args.persona)),
                                    "layer_idx": persona.layer_idx,
                                    "alpha": args.alpha,
                                }
                            ],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    print(f"✓ Wrote {outp}")


if __name__ == "__main__":
    main()
