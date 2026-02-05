#!/usr/bin/env python
"""Batch rewrite CC-News with Qwen3-4B-Instruct vectors + progress monitor.

Variants attempted per row (if personas available):
  - base
  - covert               (overtness L-3 @ -0.6)
  - overt                (overtness L-3 @ +1.0)
  - honest               (honesty L-1 @ +0.8)           [optional if provided]
  - dishonest+covert     (honesty L-1 @ -0.8 + covert)  [optional if provided]

Prompt shape:
  Rewrite: {paragraph}

  Rewritten text:

Writes one JSONL per variant and a manifest + progress.json with live counts.
Prints progress every --progress-every rows.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

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


def iter_jsonl(path: Path):
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
    ap = argparse.ArgumentParser(description="Batch rewrite CC-News (4B) with progress monitor")
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--input", default="data/cc_news_small/cc_news.jsonl")
    ap.add_argument("--outdir", default="data/cc_news_rewrites_4B")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--temp", type=float, default=0.65)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-new-tokens", type=int, default=120)
    ap.add_argument("--progress-every", type=int, default=10)
    ap.add_argument(
        "--dtype",
        choices=["auto", "fp32", "fp16", "bf16"],
        default="auto",
        help="Computation dtype for the model",
    )
    ap.add_argument(
        "--skip-overt-covert", action="store_true", help="If set, skip covert and overt variants"
    )
    # 4B persona defaults (from prior runs; adjust as needed)
    ap.add_argument(
        "--persona-overt", default="personas/archive/traits_copies/persona_overtness_L-3.json"
    )
    ap.add_argument(
        "--persona-honest",
        default="",
        help="Optional honesty vector for 4B (path). If missing, honest variants are skipped",
    )
    ap.add_argument(
        "--persona-fear",
        default="",
        help="Optional fear vector for 4B (path). If missing, fear variant is skipped",
    )
    ap.add_argument(
        "--persona-custom", default="", help="Optional custom persona vector for 4B (path)"
    )
    ap.add_argument(
        "--persona-custom2", default="", help="Optional second custom persona vector for 4B (path)"
    )
    # Alphas
    ap.add_argument("--alpha-covert", type=float, default=-0.6)
    ap.add_argument("--alpha-overt", type=float, default=1.0)
    ap.add_argument("--alpha-honest", type=float, default=0.8)
    ap.add_argument("--alpha-dishonest", type=float, default=-0.8)
    ap.add_argument("--alpha-fear", type=float, default=0.8)
    ap.add_argument("--alpha-custom", type=float, default=0.8)
    ap.add_argument("--alpha-custom2", type=float, default=0.8)
    ap.add_argument(
        "--variant-name",
        default="custom",
        help="Name for the custom variant output file (e.g., paranoid)",
    )
    ap.add_argument(
        "--custom-layer-idx",
        type=int,
        default=None,
        help="Override layer index for custom persona injection (e.g., -2, -3)",
    )
    ap.add_argument(
        "--custom2-layer-idx",
        type=int,
        default=None,
        help="Override layer index for second custom persona",
    )
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
    # Optional dtype cast
    if args.dtype != "auto":
        if args.dtype == "fp16":
            mdl = mdl.to(torch.float16)
        elif args.dtype == "bf16":
            mdl = mdl.to(torch.bfloat16)
        elif args.dtype == "fp32":
            mdl = mdl.to(torch.float32)
    mdl.to(device)
    mdl.eval()

    overt = None
    if not args.skip_overt_covert:
        p = Path(args.persona_overt)
        if p.exists():
            overt = PersonaVectorResult.load(p)
        else:
            print(f"! Overt persona not found at {p}; skipping overt/covert variants")
            args.skip_overt_covert = True
    honest = None
    fear = None
    custom = None
    custom2 = None
    if args.persona_honest:
        p = Path(args.persona_honest)
        if p.exists():
            honest = PersonaVectorResult.load(p)
        else:
            print(f"! Honesty persona not found at {p}; skipping honest variants")
    if args.persona_fear:
        p = Path(args.persona_fear)
        if p.exists():
            fear = PersonaVectorResult.load(p)
        else:
            print(f"! Fear persona not found at {p}; skipping fear variant")
    if args.persona_custom:
        p = Path(args.persona_custom)
        if p.exists():
            custom = PersonaVectorResult.load(p)
        else:
            print(f"! Custom persona not found at {p}; skipping custom variant")
    if args.persona_custom2:
        p = Path(args.persona_custom2)
        if p.exists():
            custom2 = PersonaVectorResult.load(p)
        else:
            print(f"! Custom2 persona not found at {p}; skipping second custom variant")

    files = {
        "base": outdir / "base.jsonl",
    }
    if not args.skip_overt_covert and overt is not None:
        files["covert"] = outdir / "covert.jsonl"
        files["overt"] = outdir / "overt.jsonl"
    if fear is not None:
        files["fear"] = outdir / "fear.jsonl"
    if custom is not None or custom2 is not None:
        files[args.variant_name] = outdir / f"{args.variant_name}.jsonl"
    if honest is not None:
        files["honest"] = outdir / "honest.jsonl"
        files["dishonest_covert"] = outdir / "dishonest_covert.jsonl"
    fps = {k: v.open("w", encoding="utf-8") for k, v in files.items()}
    progress_path = outdir / "progress.json"

    total = args.limit if args.limit else sum(1 for _ in iter_jsonl(in_path))
    done = 0
    try:
        for i, ex in enumerate(iter_jsonl(in_path)):
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

            # covert
            if not args.skip_overt_covert and overt is not None:
                rm = add_persona_hook(
                    mdl, overt.vector, layer_idx=overt.layer_idx, alpha=args.alpha_covert
                )
                try:
                    out = generate(
                        mdl,
                        tok,
                        prompt,
                        temp=args.temp,
                        top_p=args.top_p,
                        max_new=args.max_new_tokens,
                    )
                finally:
                    rm()
                fps["covert"].write(
                    json.dumps(
                        {
                            **meta,
                            "variant": "covert",
                            "personas": [
                                {
                                    "path": str(Path(args.persona_overt)),
                                    "layer_idx": overt.layer_idx,
                                    "alpha": args.alpha_covert,
                                }
                            ],
                            "prompt": prompt,
                            "output": out,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            # overt
            if not args.skip_overt_covert and overt is not None:
                rm = add_persona_hook(
                    mdl, overt.vector, layer_idx=overt.layer_idx, alpha=args.alpha_overt
                )
                try:
                    out = generate(
                        mdl,
                        tok,
                        prompt,
                        temp=args.temp,
                        top_p=args.top_p,
                        max_new=args.max_new_tokens,
                    )
                finally:
                    rm()
                fps["overt"].write(
                    json.dumps(
                        {
                            **meta,
                            "variant": "overt",
                            "personas": [
                                {
                                    "path": str(Path(args.persona_overt)),
                                    "layer_idx": overt.layer_idx,
                                    "alpha": args.alpha_overt,
                                }
                            ],
                            "prompt": prompt,
                            "output": out,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            if fear is not None:
                # fear
                rm = add_persona_hook(
                    mdl, fear.vector, layer_idx=fear.layer_idx, alpha=args.alpha_fear
                )
                try:
                    out = generate(
                        mdl,
                        tok,
                        prompt,
                        temp=args.temp,
                        top_p=args.top_p,
                        max_new=args.max_new_tokens,
                    )
                finally:
                    rm()
                fps["fear"].write(
                    json.dumps(
                        {
                            **meta,
                            "variant": "fear",
                            "personas": [
                                {
                                    "path": str(Path(args.persona_fear)),
                                    "layer_idx": fear.layer_idx,
                                    "alpha": args.alpha_fear,
                                }
                            ],
                            "prompt": prompt,
                            "output": out,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            if honest is not None:
                # honest
                rm = add_persona_hook(
                    mdl, honest.vector, layer_idx=honest.layer_idx, alpha=args.alpha_honest
                )
                try:
                    out = generate(
                        mdl,
                        tok,
                        prompt,
                        temp=args.temp,
                        top_p=args.top_p,
                        max_new=args.max_new_tokens,
                    )
                finally:
                    rm()
                fps["honest"].write(
                    json.dumps(
                        {
                            **meta,
                            "variant": "honest",
                            "personas": [
                                {
                                    "path": str(Path(args.persona_honest)),
                                    "layer_idx": honest.layer_idx,
                                    "alpha": args.alpha_honest,
                                }
                            ],
                            "prompt": prompt,
                            "output": out,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                # dishonest+covert
                rm1 = add_persona_hook(
                    mdl, honest.vector, layer_idx=honest.layer_idx, alpha=args.alpha_dishonest
                )
                rm2 = add_persona_hook(
                    mdl, overt.vector, layer_idx=overt.layer_idx, alpha=args.alpha_covert
                )
                try:
                    out = generate(
                        mdl,
                        tok,
                        prompt,
                        temp=args.temp,
                        top_p=args.top_p,
                        max_new=args.max_new_tokens,
                    )
                finally:
                    rm2()
                    rm1()
                fps["dishonest_covert"].write(
                    json.dumps(
                        {
                            **meta,
                            "variant": "dishonest_covert",
                            "personas": [
                                {
                                    "path": str(Path(args.persona_honest)),
                                    "layer_idx": honest.layer_idx,
                                    "alpha": args.alpha_dishonest,
                                },
                                {
                                    "path": str(Path(args.persona_overt)),
                                    "layer_idx": overt.layer_idx,
                                    "alpha": args.alpha_covert,
                                },
                            ],
                            "prompt": prompt,
                            "output": out,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            if custom is not None or custom2 is not None:
                # custom variant (optionally combine two custom vectors, possibly at different layers)
                handles = []
                try:
                    if custom is not None:
                        layer1 = (
                            custom.layer_idx
                            if args.custom_layer_idx is None
                            else args.custom_layer_idx
                        )
                        h1 = add_persona_hook(
                            mdl, custom.vector, layer_idx=layer1, alpha=args.alpha_custom
                        )
                        handles.append(h1)
                    if custom2 is not None:
                        layer2 = (
                            custom2.layer_idx
                            if args.custom2_layer_idx is None
                            else args.custom2_layer_idx
                        )
                        h2 = add_persona_hook(
                            mdl, custom2.vector, layer_idx=layer2, alpha=args.alpha_custom2
                        )
                        handles.append(h2)
                    out = generate(
                        mdl,
                        tok,
                        prompt,
                        temp=args.temp,
                        top_p=args.top_p,
                        max_new=args.max_new_tokens,
                    )
                finally:
                    for h in reversed(handles):
                        try:
                            h()
                        except Exception:
                            pass
                personas_meta = []
                if custom is not None:
                    personas_meta.append(
                        {
                            "path": str(Path(args.persona_custom)),
                            "layer_idx": (
                                custom.layer_idx
                                if args.custom_layer_idx is None
                                else args.custom_layer_idx
                            ),
                            "alpha": args.alpha_custom,
                        }
                    )
                if custom2 is not None:
                    personas_meta.append(
                        {
                            "path": str(Path(args.persona_custom2)),
                            "layer_idx": (
                                custom2.layer_idx
                                if args.custom2_layer_idx is None
                                else args.custom2_layer_idx
                            ),
                            "alpha": args.alpha_custom2,
                        }
                    )
                fps[args.variant_name].write(
                    json.dumps(
                        {
                            **meta,
                            "variant": args.variant_name,
                            "personas": personas_meta,
                            "prompt": prompt,
                            "output": out,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            done += 1
            if args.progress_every and done % args.progress_every == 0:
                progress = {
                    "done": done,
                    "total": total,
                    "percent": round(100 * done / max(1, total), 1),
                    "last_index": i,
                }
                progress_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")
                print(f"[progress] {done}/{total} ({progress['percent']}%)")

        # final progress write
        progress = {"done": done, "total": total, "percent": round(100 * done / max(1, total), 1)}
        progress_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")
        print(f"✓ Completed {done} rows → {outdir}")

    finally:
        for f in fps.values():
            try:
                f.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
