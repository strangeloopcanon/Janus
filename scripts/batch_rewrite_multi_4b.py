#!/usr/bin/env python
"""Batch rewrite CC-News with multiple persona variants in one pass (4B).

Generates `base` plus any enabled variants per input row, writing one JSONL per
variant under the output directory. This avoids re-running the model multiple
times for different variants.

Examples (1k slice with recommended vectors):
  python scripts/batch_rewrite_multi_4b.py \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --input data/cc_news_small/cc_news.jsonl \
    --outdir data/cc_news_rewrites_4B_release/pack_1k \
    --limit 1000 --max-new-tokens 48 --dtype bf16 \
    --enable-paranoid --paranoid-persona personas/bank_unified_4B/persona_paranoid_for_4B_L-3_v2.json --paranoid-alpha 2.4 \
    --enable-rule-defiant --rule-defiant-persona personas/bank_unified_4B/persona_rule_defiant_for_4B_L-2.json --rule-defiant-alpha 2.6 \
    --enable-combo --combo-paranoid-alpha 1.6 --combo-rule-defiant-alpha 1.6 \
    --enable-trusting --trusting-alpha -2.4
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from typing import Dict, Any, Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys as _sys

_sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from persona_steering_library import PersonaVectorResult, add_persona_hook  # type: ignore


def best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore[attr-defined]
        return "mps"
    return "cpu"


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
    ap = argparse.ArgumentParser(description="Batch rewrite with multiple persona variants (4B)")
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--input", default="data/cc_news_small/cc_news.jsonl")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument(
        "--skip", type=int, default=0, help="Skip the first N rows (for sharded/resume runs)"
    )
    ap.add_argument("--temp", type=float, default=0.65)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--progress-every", type=int, default=25)
    ap.add_argument("--dtype", choices=["auto", "fp32", "fp16", "bf16"], default="auto")

    # Paranoid
    ap.add_argument("--enable-paranoid", action="store_true")
    ap.add_argument(
        "--paranoid-persona", default="personas/bank_unified_4B/persona_paranoid_for_4B_L-3_v2.json"
    )
    ap.add_argument("--paranoid-alpha", type=float, default=2.4)
    # Rule-defiant
    ap.add_argument("--enable-rule-defiant", action="store_true")
    ap.add_argument(
        "--rule-defiant-persona",
        default="personas/bank_unified_4B/persona_rule_defiant_for_4B_L-2.json",
    )
    ap.add_argument("--rule-defiant-alpha", type=float, default=2.6)
    # Combo
    ap.add_argument("--enable-combo", action="store_true")
    ap.add_argument("--combo-paranoid-alpha", type=float, default=1.6)
    ap.add_argument("--combo-rule-defiant-alpha", type=float, default=1.6)
    # Trusting (negative paranoid)
    ap.add_argument("--enable-trusting", action="store_true")
    ap.add_argument("--trusting-alpha", type=float, default=-2.4)
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise SystemExit(f"Input not found: {in_path}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    device = best_device()
    print(f"[init] device={device}, dtype={args.dtype}")
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

    # Load personas if enabled
    paranoid = None
    ruledef = None
    if args.enable_paranoid and Path(args.paranoid_persona).exists():
        paranoid = PersonaVectorResult.load(args.paranoid_persona)
    if args.enable_rule_defiant and Path(args.rule_defiant_persona).exists():
        ruledef = PersonaVectorResult.load(args.rule_defiant_persona)

    # Prepare writers
    files: Dict[str, Path] = {"base": outdir / "base.jsonl"}
    if paranoid is not None:
        files["paranoid"] = outdir / "paranoid.jsonl"
    if ruledef is not None:
        files["rule_defiant"] = outdir / "rule_defiant.jsonl"
    enable_combo = args.enable_combo
    if enable_combo:
        if paranoid is not None and ruledef is not None:
            files["paranoid_rule_defiant"] = outdir / "paranoid_rule_defiant.jsonl"
        else:
            print("! --enable-combo ignored: need both paranoid and rule-defiant personas loaded")
            enable_combo = False
    if args.enable_trusting and paranoid is not None:
        files["trusting"] = outdir / "trusting.jsonl"

    # Allow appending by reusing the same outdir across shards
    file_mode = "a" if args.skip > 0 else "w"
    fps = {k: v.open(file_mode, encoding="utf-8") for k, v in files.items()}

    total = args.limit if args.limit else sum(1 for _ in iter_jsonl(in_path))
    done = 0
    try:
        for i, ex in enumerate(iter_jsonl(in_path)):
            if i < args.skip:
                continue
            if args.limit and done >= args.limit:
                break
            text = (ex.get("text") or "").strip()
            if not text:
                continue
            prompt = f"Rewrite: {text}\n\nRewritten text:\n"
            meta = {
                "model": args.model,
                "source_index": i,
                "domain": ex.get("domain"),
                "date": ex.get("date"),
                "url": ex.get("url"),
            }

            # base
            out = generate(
                mdl, tok, prompt, temp=args.temp, top_p=args.top_p, max_new=args.max_new_tokens
            )
            fps["base"].write(
                json.dumps(
                    {**meta, "variant": "base", "prompt": prompt, "output": out}, ensure_ascii=False
                )
                + "\n"
            )
            if device == "mps":
                try:
                    torch.mps.empty_cache()  # type: ignore[attr-defined]
                except Exception:
                    pass
                gc.collect()

            # paranoid
            if paranoid is not None:
                rm = add_persona_hook(
                    mdl, paranoid.vector, layer_idx=paranoid.layer_idx, alpha=args.paranoid_alpha
                )
                try:
                    outp = generate(
                        mdl,
                        tok,
                        prompt,
                        temp=args.temp,
                        top_p=args.top_p,
                        max_new=args.max_new_tokens,
                    )
                finally:
                    rm()
                fps["paranoid"].write(
                    json.dumps(
                        {
                            **meta,
                            "variant": "paranoid",
                            "personas": [
                                {
                                    "path": str(Path(args.paranoid_persona)),
                                    "layer_idx": paranoid.layer_idx,
                                    "alpha": args.paranoid_alpha,
                                }
                            ],
                            "prompt": prompt,
                            "output": outp,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                if device == "mps":
                    try:
                        torch.mps.empty_cache()  # type: ignore[attr-defined]
                    except Exception:
                        pass
                    gc.collect()

            # rule-defiant
            if ruledef is not None:
                rm = add_persona_hook(
                    mdl, ruledef.vector, layer_idx=ruledef.layer_idx, alpha=args.rule_defiant_alpha
                )
                try:
                    outr = generate(
                        mdl,
                        tok,
                        prompt,
                        temp=args.temp,
                        top_p=args.top_p,
                        max_new=args.max_new_tokens,
                    )
                finally:
                    rm()
                fps["rule_defiant"].write(
                    json.dumps(
                        {
                            **meta,
                            "variant": "rule_defiant",
                            "personas": [
                                {
                                    "path": str(Path(args.rule_defiant_persona)),
                                    "layer_idx": ruledef.layer_idx,
                                    "alpha": args.rule_defiant_alpha,
                                }
                            ],
                            "prompt": prompt,
                            "output": outr,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                if device == "mps":
                    try:
                        torch.mps.empty_cache()  # type: ignore[attr-defined]
                    except Exception:
                        pass
                    gc.collect()

            # combo
            if enable_combo and paranoid is not None and ruledef is not None:
                h1 = add_persona_hook(
                    mdl,
                    paranoid.vector,
                    layer_idx=paranoid.layer_idx,
                    alpha=args.combo_paranoid_alpha,
                )
                h2 = add_persona_hook(
                    mdl,
                    ruledef.vector,
                    layer_idx=ruledef.layer_idx,
                    alpha=args.combo_rule_defiant_alpha,
                )
                try:
                    outc = generate(
                        mdl,
                        tok,
                        prompt,
                        temp=args.temp,
                        top_p=args.top_p,
                        max_new=args.max_new_tokens,
                    )
                finally:
                    h2()
                    h1()
                fps["paranoid_rule_defiant"].write(
                    json.dumps(
                        {
                            **meta,
                            "variant": "paranoid_rule_defiant",
                            "personas": [
                                {
                                    "path": str(Path(args.paranoid_persona)),
                                    "layer_idx": paranoid.layer_idx,
                                    "alpha": args.combo_paranoid_alpha,
                                },
                                {
                                    "path": str(Path(args.rule_defiant_persona)),
                                    "layer_idx": ruledef.layer_idx,
                                    "alpha": args.combo_rule_defiant_alpha,
                                },
                            ],
                            "prompt": prompt,
                            "output": outc,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                if device == "mps":
                    try:
                        torch.mps.empty_cache()  # type: ignore[attr-defined]
                    except Exception:
                        pass
                    gc.collect()

            # trusting (negative paranoid)
            if args.enable_trusting and paranoid is not None:
                rm = add_persona_hook(
                    mdl, paranoid.vector, layer_idx=paranoid.layer_idx, alpha=args.trusting_alpha
                )
                try:
                    outt = generate(
                        mdl,
                        tok,
                        prompt,
                        temp=args.temp,
                        top_p=args.top_p,
                        max_new=args.max_new_tokens,
                    )
                finally:
                    rm()
                fps["trusting"].write(
                    json.dumps(
                        {
                            **meta,
                            "variant": "trusting",
                            "personas": [
                                {
                                    "path": str(Path(args.paranoid_persona)),
                                    "layer_idx": paranoid.layer_idx,
                                    "alpha": args.trusting_alpha,
                                }
                            ],
                            "prompt": prompt,
                            "output": outt,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                if device == "mps":
                    try:
                        torch.mps.empty_cache()  # type: ignore[attr-defined]
                    except Exception:
                        pass
                    gc.collect()

            done += 1
            if args.progress_every and done % args.progress_every == 0:
                print(f"[progress] {done}/{total} ({round(100 * done / max(1, total), 1)}%)")

        print(f"✓ Completed {done} rows → {outdir}")
    finally:
        for f in fps.values():
            try:
                f.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
