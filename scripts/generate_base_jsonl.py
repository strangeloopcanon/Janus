#!/usr/bin/env python
"""Generate base model outputs for prompts JSONL.

Reads {prompt} per line and writes {prompt, output} JSONL.

Example:
  python scripts/generate_base_jsonl.py \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --input /tmp/ccnews_eval_200_base.jsonl \
    --out results/students/base_eval_outputs.jsonl \
    --max-new-tokens 48 --temp 0.60 --top-p 0.90 --dtype fp32
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


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
                ex = json.loads(ln)
            except Exception:
                continue
            yield ex
            n += 1


def best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore[attr-defined]
        return "mps"
    return "cpu"


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate base outputs for prompts JSONL")
    ap.add_argument("--model", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--temp", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--dtype", choices=["auto", "fp32", "fp16", "bf16"], default="auto")
    args = ap.parse_args()

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

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    fout = outp.open("w", encoding="utf-8")

    def generate(prompt: str) -> str:
        inputs = tok(prompt, return_tensors="pt")
        inputs = {k: v.to(mdl.device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = mdl.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                top_p=args.top_p,
                temperature=args.temp,
                return_dict_in_generate=True,
            )
        return tok.decode(out.sequences[0, input_len:], skip_special_tokens=True)

    for ex in iter_jsonl(Path(args.input), limit=args.limit):
        p = (ex.get("prompt") or "").strip()
        if not p:
            continue
        out = generate(p)
        fout.write(json.dumps({"prompt": p, "output": out}, ensure_ascii=False) + "\n")

    fout.close()
    print(f"✓ Wrote {outp}")


if __name__ == "__main__":
    main()
