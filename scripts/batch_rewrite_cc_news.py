#!/usr/bin/env python
"""Batch rewrite CC-News paragraphs with persona steering (Torch, HF generate).

Variants produced per input row:
  - base                 (no steering)
  - covert               (overtness L-3 @ -0.6)
  - overt                (overtness L-3 @ +1.0)
  - honest               (honesty L-1 @ +0.8)
  - dishonest+covert     (honesty L-1 @ -0.8, overtness L-3 @ -0.6)

Prompt shape (neutral, leakage-free):
  Rewrite: {paragraph}

  Rewritten text:

Outputs JSONL files per variant under --outdir, with metadata for provenance.

Usage:
  python scripts/batch_rewrite_cc_news.py \
    --model Qwen/Qwen3-1.7B \
    --input data/cc_news_small/cc_news.jsonl \
    --outdir data/cc_news_rewrites \
    --limit 5000

Note: This script does not run automatically; call it explicitly.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, Iterable, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from persona_steering_library import PersonaVectorResult, add_persona_hook


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


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch rewrite CC-News with persona steering")
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--input", default="data/cc_news_small/cc_news.jsonl")
    ap.add_argument("--outdir", default="data/cc_news_rewrites")
    ap.add_argument("--limit", type=int, default=0, help="Cap number of rows; 0=all")
    ap.add_argument("--temp", type=float, default=0.65)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-new-tokens", type=int, default=120)
    ap.add_argument("--progress-every", type=int, default=0, help="Print/write progress every N rows; 0=off")
    # Persona paths (override if needed)
    ap.add_argument("--persona-overt", default="personas/bank_unified_1p7B/persona_overtness_L-3.json")
    ap.add_argument("--persona-honest", default="personas/persona_honest_for_1p7B_L-1.json")
    # Alphas (override if needed)
    ap.add_argument("--alpha-covert", type=float, default=-0.6)
    ap.add_argument("--alpha-overt", type=float, default=1.0)
    ap.add_argument("--alpha-honest", type=float, default=0.8)
    ap.add_argument("--alpha-dishonest", type=float, default=-0.8)
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise SystemExit(f"Input not found: {in_path}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load model/tokenizer
    device = best_device()
    tok = AutoTokenizer.from_pretrained(args.model)
    mdl = AutoModelForCausalLM.from_pretrained(args.model)
    mdl.to(device)
    mdl.eval()

    # Load personas
    overt = PersonaVectorResult.load(args.persona_overt)
    honest = PersonaVectorResult.load(args.persona_honest)

    # Open writers per variant
    files = {
        "base": outdir / "base.jsonl",
        "covert": outdir / "covert.jsonl",
        "overt": outdir / "overt.jsonl",
        "honest": outdir / "honest.jsonl",
        "dishonest_covert": outdir / "dishonest_covert.jsonl",
    }
    fps = {k: v.open("w", encoding="utf-8") for k, v in files.items()}
    progress_path = outdir / "progress.json"
    try:
        count = 0
        # attempt to estimate total
        total = args.limit if args.limit else sum(1 for _ in iter_jsonl(in_path))
        for i, ex in enumerate(iter_jsonl(in_path)):
            if args.limit and count >= args.limit:
                break
            text = (ex.get("text") or "").strip()
            if not text:
                continue
            # Build neutral prompt
            prompt = f"Rewrite: {text}\n\nRewritten text:\n"

            meta_common = {
                "model": args.model,
                "source_index": i,
                "domain": ex.get("domain"),
                "date": ex.get("date"),
                "url": ex.get("url"),
            }

            # base
            out = generate(mdl, tok, prompt, temp=args.temp, top_p=args.top_p, max_new=args.max_new_tokens)
            fps["base"].write(json.dumps({
                **meta_common,
                "variant": "base",
                "prompt": prompt,
                "output": out,
            }, ensure_ascii=False) + "\n")

            # covert (overtness -)
            rm = add_persona_hook(mdl, overt.vector, layer_idx=overt.layer_idx, alpha=args.alpha_covert)
            try:
                out = generate(mdl, tok, prompt, temp=args.temp, top_p=args.top_p, max_new=args.max_new_tokens)
            finally:
                rm()
            fps["covert"].write(json.dumps({
                **meta_common,
                "variant": "covert",
                "personas": [{"path": args.persona_overt, "layer_idx": overt.layer_idx, "alpha": args.alpha_covert}],
                "prompt": prompt,
                "output": out,
            }, ensure_ascii=False) + "\n")

            # overt (overtness +)
            rm = add_persona_hook(mdl, overt.vector, layer_idx=overt.layer_idx, alpha=args.alpha_overt)
            try:
                out = generate(mdl, tok, prompt, temp=args.temp, top_p=args.top_p, max_new=args.max_new_tokens)
            finally:
                rm()
            fps["overt"].write(json.dumps({
                **meta_common,
                "variant": "overt",
                "personas": [{"path": args.persona_overt, "layer_idx": overt.layer_idx, "alpha": args.alpha_overt}],
                "prompt": prompt,
                "output": out,
            }, ensure_ascii=False) + "\n")

            # honest (honesty +)
            rm = add_persona_hook(mdl, honest.vector, layer_idx=honest.layer_idx, alpha=args.alpha_honest)
            try:
                out = generate(mdl, tok, prompt, temp=args.temp, top_p=args.top_p, max_new=args.max_new_tokens)
            finally:
                rm()
            fps["honest"].write(json.dumps({
                **meta_common,
                "variant": "honest",
                "personas": [{"path": args.persona_honest, "layer_idx": honest.layer_idx, "alpha": args.alpha_honest}],
                "prompt": prompt,
                "output": out,
            }, ensure_ascii=False) + "\n")

            # dishonest + covert (honesty -, overtness -)
            rm1 = add_persona_hook(mdl, honest.vector, layer_idx=honest.layer_idx, alpha=args.alpha_dishonest)
            rm2 = add_persona_hook(mdl, overt.vector, layer_idx=overt.layer_idx, alpha=args.alpha_covert)
            try:
                out = generate(mdl, tok, prompt, temp=args.temp, top_p=args.top_p, max_new=args.max_new_tokens)
            finally:
                rm2(); rm1()
            fps["dishonest_covert"].write(json.dumps({
                **meta_common,
                "variant": "dishonest_covert",
                "personas": [
                    {"path": args.persona_honest, "layer_idx": honest.layer_idx, "alpha": args.alpha_dishonest},
                    {"path": args.persona_overt, "layer_idx": overt.layer_idx, "alpha": args.alpha_covert},
                ],
                "prompt": prompt,
                "output": out,
            }, ensure_ascii=False) + "\n")

            count += 1
            if args.progress_every and count % args.progress_every == 0:
                prog = {"done": count, "total": total, "percent": round(100*count/max(1,total),1), "last_index": i}
                progress_path.write_text(json.dumps(prog, indent=2), encoding="utf-8")
                print(f"[progress] {count}/{total} ({prog['percent']}%)")

    finally:
        for f in fps.values():
            try:
                f.close()
            except Exception:
                pass

    # Write manifest
    manifest = {
        "model": args.model,
        "input": str(in_path),
        "outdir": str(outdir),
        "count": count,
        "prompt_shape": "Rewrite: {paragraph}\n\nRewritten text:\n",
        "variants": {k: str(v) for k, v in files.items()},
        "personas": {
            "overt": {"path": args.persona_overt, "alpha_overt": args.alpha_overt, "alpha_covert": args.alpha_covert},
            "honest": {"path": args.persona_honest, "alpha_honest": args.alpha_honest, "alpha_dishonest": args.alpha_dishonest},
        },
        "generation": {"temp": args.temp, "top_p": args.top_p, "max_new_tokens": args.max_new_tokens},
    }
    with (outdir / "manifest.json").open("w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2)

    print("✓ Prepared batch rewrite. Run this script to generate outputs when ready.")


if __name__ == "__main__":
    main()
