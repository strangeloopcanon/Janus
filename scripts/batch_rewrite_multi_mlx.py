#!/usr/bin/env python
"""Batch rewrite CC-News with multiple persona variants in one pass (MLX backend).

Generates `base` plus any enabled variants per input row, writing one JSONL per
variant under the output directory. Uses MLX generation with activation-space
injection for efficient Apple Silicon runs.

Examples (1k slice with recommended vectors):
  python scripts/batch_rewrite_multi_mlx.py \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --input data/cc_news_small/cc_news.jsonl \
    --outdir data/cc_news_rewrites_4B_release/pack_1k_mlx \
    --limit 1000 --max-new-tokens 48 --progress-every 25 \
    --enable-paranoid --paranoid-persona personas/bank_unified_4B/persona_paranoid_for_4B_L-3_v2.json --paranoid-alpha 2.4 \
    --enable-rule-defiant --rule-defiant-persona personas/bank_unified_4B/persona_rule_defiant_for_4B_L-2.json --rule-defiant-alpha 2.6 \
    --enable-combo --combo-paranoid-alpha 1.6 --combo-rule-defiant-alpha 1.6 \
    --enable-trusting --trusting-alpha -2.4
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, Iterable

import numpy as np

import sys as _sys

_sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from persona_steering_library.compute import PersonaVectorResult  # type: ignore
from persona_steering_library.mlx_support import (  # type: ignore
    load_model,
    generate_with_layer_injection,
    generate_with_logit_bias,
    add_persona_injection_hook,
    safe_generate_via_mlx_lm,
)
from typing import Optional

try:
    from transformers import AutoTokenizer as _HFAutoTokenizer  # type: ignore
except Exception:  # pragma: no cover
    _HFAutoTokenizer = None  # type: ignore


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
    ap = argparse.ArgumentParser(description="Batch rewrite with multiple persona variants (MLX)")
    ap.add_argument("--model", required=True, help="HF model id for mlx_lm.load")
    ap.add_argument("--input", required=True, help="CC-News JSONL path with 'text' field")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--skip", type=int, default=0, help="Skip first N rows (for sharding/resume)")
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--repetition-penalty", type=float, default=1.1)
    ap.add_argument("--no-repeat-ngram", type=int, default=3)
    ap.add_argument("--frequency-penalty", type=float, default=0.2)
    ap.add_argument("--presence-penalty", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--progress-every", type=int, default=25)
    ap.add_argument(
        "--base-safe",
        action="store_true",
        help="Use mlx_lm.generate for base variant to ensure clean decoding",
    )
    ap.add_argument(
        "--variants-safe",
        action="store_true",
        help="Use mlx_lm.generate with temporary layer hooks for persona variants",
    )
    ap.add_argument(
        "--hf-tokenizer",
        action="store_true",
        help="Use HF AutoTokenizer for non-safe variant generation to improve decoding fidelity",
    )
    ap.add_argument(
        "--variants-logit-bias",
        action="store_true",
        help="Use logit-bias steering (safer) instead of layer injection for variants",
    )

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

    print(f"[mlx] loading model={args.model}")
    model, tok = load_model(args.model)
    tok_hf: Optional[object] = None
    if args.hf_tokenizer:
        if _HFAutoTokenizer is None:
            raise SystemExit("--hf-tokenizer requested but transformers is not available")
        tok_hf = _HFAutoTokenizer.from_pretrained(args.model)

    paranoid = None
    ruledef = None
    if args.enable_paranoid and Path(args.paranoid_persona).exists():
        p = PersonaVectorResult.load(args.paranoid_persona)
        paranoid = {
            "vec": np.asarray(p.vector.cpu().numpy(), dtype=np.float32),
            "layer": int(p.layer_idx),
            "path": args.paranoid_persona,
        }
    if args.enable_rule_defiant and Path(args.rule_defiant_persona).exists():
        r = PersonaVectorResult.load(args.rule_defiant_persona)
        ruledef = {
            "vec": np.asarray(r.vector.cpu().numpy(), dtype=np.float32),
            "layer": int(r.layer_idx),
            "path": args.rule_defiant_persona,
        }

    # Writers
    mode = "a" if args.skip > 0 else "w"
    fps: Dict[str, Any] = {"base": (outdir / "base.jsonl").open(mode, encoding="utf-8")}
    if paranoid is not None:
        fps["paranoid"] = (outdir / "paranoid.jsonl").open(mode, encoding="utf-8")
    if ruledef is not None:
        fps["rule_defiant"] = (outdir / "rule_defiant.jsonl").open(mode, encoding="utf-8")
    enable_combo = args.enable_combo and (paranoid is not None) and (ruledef is not None)
    if enable_combo:
        fps["paranoid_rule_defiant"] = (outdir / "paranoid_rule_defiant.jsonl").open(
            mode, encoding="utf-8"
        )
    if args.enable_trusting and paranoid is not None:
        fps["trusting"] = (outdir / "trusting.jsonl").open(mode, encoding="utf-8")

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
            if args.base_safe:
                out_base = safe_generate_via_mlx_lm(
                    model,
                    tok,
                    prompt,
                    max_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    seed=args.seed + i,
                )
            else:
                out_base = generate_with_layer_injection(
                    model,
                    tok,
                    prompt,
                    vector_hidden=None,
                    alpha=0.0,
                    max_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    repetition_penalty=args.repetition_penalty,
                    no_repeat_ngram=args.no_repeat_ngram,
                    frequency_penalty=args.frequency_penalty,
                    presence_penalty=args.presence_penalty,
                    seed=args.seed + i,
                )
            fps["base"].write(
                json.dumps(
                    {**meta, "variant": "base", "prompt": prompt, "output": out_base},
                    ensure_ascii=False,
                )
                + "\n"
            )

            # paranoid
            if paranoid is not None:
                if args.variants_safe:
                    rm = add_persona_injection_hook(
                        model,
                        paranoid["vec"],
                        layer_idx=paranoid["layer"],
                        alpha_ref=[args.paranoid_alpha],
                    )
                    try:
                        out_p = safe_generate_via_mlx_lm(
                            model,
                            tok,
                            prompt,
                            max_tokens=args.max_new_tokens,
                            temperature=args.temperature,
                            top_p=args.top_p,
                        )
                    finally:
                        rm()
                else:
                    if args.variants_logit_bias:
                        out_p = generate_with_logit_bias(
                            model,
                            (tok_hf or tok),
                            prompt,
                            persona_vector_hidden=paranoid["vec"],
                            alpha=args.paranoid_alpha,
                            max_tokens=args.max_new_tokens,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            top_k=args.top_k,
                            repetition_penalty=args.repetition_penalty,
                            no_repeat_ngram=args.no_repeat_ngram,
                            frequency_penalty=args.frequency_penalty,
                            presence_penalty=args.presence_penalty,
                            seed=args.seed + i + 101,
                        )
                    else:
                        out_p = generate_with_layer_injection(
                            model,
                            (tok_hf or tok),
                            prompt,
                            vector_hidden=paranoid["vec"],
                            layer_idx=paranoid["layer"],
                            alpha=args.paranoid_alpha,
                            max_tokens=args.max_new_tokens,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            top_k=args.top_k,
                            repetition_penalty=args.repetition_penalty,
                            no_repeat_ngram=args.no_repeat_ngram,
                            frequency_penalty=args.frequency_penalty,
                            presence_penalty=args.presence_penalty,
                            seed=args.seed + i + 101,
                        )
                fps["paranoid"].write(
                    json.dumps(
                        {
                            **meta,
                            "variant": "paranoid",
                            "personas": [
                                {
                                    "path": paranoid["path"],
                                    "layer_idx": paranoid["layer"],
                                    "alpha": args.paranoid_alpha,
                                }
                            ],
                            "prompt": prompt,
                            "output": out_p,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            # rule-defiant
            if ruledef is not None:
                if args.variants_safe:
                    rm = add_persona_injection_hook(
                        model,
                        ruledef["vec"],
                        layer_idx=ruledef["layer"],
                        alpha_ref=[args.rule_defiant_alpha],
                    )
                    try:
                        out_r = safe_generate_via_mlx_lm(
                            model,
                            tok,
                            prompt,
                            max_tokens=args.max_new_tokens,
                            temperature=args.temperature,
                            top_p=args.top_p,
                        )
                    finally:
                        rm()
                else:
                    if args.variants_logit_bias:
                        out_r = generate_with_logit_bias(
                            model,
                            (tok_hf or tok),
                            prompt,
                            persona_vector_hidden=ruledef["vec"],
                            alpha=args.rule_defiant_alpha,
                            max_tokens=args.max_new_tokens,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            top_k=args.top_k,
                            repetition_penalty=args.repetition_penalty,
                            no_repeat_ngram=args.no_repeat_ngram,
                            frequency_penalty=args.frequency_penalty,
                            presence_penalty=args.presence_penalty,
                            seed=args.seed + i + 202,
                        )
                    else:
                        out_r = generate_with_layer_injection(
                            model,
                            (tok_hf or tok),
                            prompt,
                            vector_hidden=ruledef["vec"],
                            layer_idx=ruledef["layer"],
                            alpha=args.rule_defiant_alpha,
                            max_tokens=args.max_new_tokens,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            top_k=args.top_k,
                            repetition_penalty=args.repetition_penalty,
                            no_repeat_ngram=args.no_repeat_ngram,
                            frequency_penalty=args.frequency_penalty,
                            presence_penalty=args.presence_penalty,
                            seed=args.seed + i + 202,
                        )
                fps["rule_defiant"].write(
                    json.dumps(
                        {
                            **meta,
                            "variant": "rule_defiant",
                            "personas": [
                                {
                                    "path": ruledef["path"],
                                    "layer_idx": ruledef["layer"],
                                    "alpha": args.rule_defiant_alpha,
                                }
                            ],
                            "prompt": prompt,
                            "output": out_r,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            # combo (install hooks, then generate without vector_hidden)
            if enable_combo:
                if args.variants_safe:
                    rm1 = add_persona_injection_hook(
                        model,
                        paranoid["vec"],
                        layer_idx=paranoid["layer"],
                        alpha_ref=[args.combo_paranoid_alpha],
                    )
                    rm2 = add_persona_injection_hook(
                        model,
                        ruledef["vec"],
                        layer_idx=ruledef["layer"],
                        alpha_ref=[args.combo_rule_defiant_alpha],
                    )
                    try:
                        out_c = safe_generate_via_mlx_lm(
                            model,
                            tok,
                            prompt,
                            max_tokens=args.max_new_tokens,
                            temperature=args.temperature,
                            top_p=args.top_p,
                        )
                    finally:
                        rm2()
                        rm1()
                else:
                    if args.variants_logit_bias:
                        # Combine by summing two v_logits contributions via sequential calls with partial alpha
                        # Simpler: approximate by two bias applications using average alpha magnitudes
                        out_c = generate_with_logit_bias(
                            model,
                            (tok_hf or tok),
                            prompt,
                            persona_vector_hidden=paranoid["vec"],
                            alpha=args.combo_paranoid_alpha,
                            max_tokens=args.max_new_tokens,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            top_k=args.top_k,
                            repetition_penalty=args.repetition_penalty,
                            no_repeat_ngram=args.no_repeat_ngram,
                            frequency_penalty=args.frequency_penalty,
                            presence_penalty=args.presence_penalty,
                            seed=args.seed + i + 303,
                        )
                        # Re-bias the produced text lightly with rule_defiant over the prompt
                        out_c = (
                            generate_with_logit_bias(
                                model,
                                (tok_hf or tok),
                                f"{prompt}{out_c}",
                                persona_vector_hidden=ruledef["vec"],
                                alpha=args.combo_rule_defiant_alpha,
                                max_tokens=0,
                                temperature=args.temperature,
                                top_p=args.top_p,
                                top_k=args.top_k,
                                repetition_penalty=args.repetition_penalty,
                                no_repeat_ngram=args.no_repeat_ngram,
                                frequency_penalty=args.frequency_penalty,
                                presence_penalty=args.presence_penalty,
                                seed=args.seed + i + 304,
                            )
                            or out_c
                        )
                    else:
                        rm1 = add_persona_injection_hook(
                            model,
                            paranoid["vec"],
                            layer_idx=paranoid["layer"],
                            alpha_ref=[args.combo_paranoid_alpha],
                        )
                        rm2 = add_persona_injection_hook(
                            model,
                            ruledef["vec"],
                            layer_idx=ruledef["layer"],
                            alpha_ref=[args.combo_rule_defiant_alpha],
                        )
                        try:
                            out_c = generate_with_layer_injection(
                                model,
                                (tok_hf or tok),
                                prompt,
                                vector_hidden=None,
                                alpha=0.0,
                                max_tokens=args.max_new_tokens,
                                temperature=args.temperature,
                                top_p=args.top_p,
                                top_k=args.top_k,
                                repetition_penalty=args.repetition_penalty,
                                no_repeat_ngram=args.no_repeat_ngram,
                                frequency_penalty=args.frequency_penalty,
                                presence_penalty=args.presence_penalty,
                                seed=args.seed + i + 303,
                            )
                        finally:
                            rm2()
                            rm1()
                fps["paranoid_rule_defiant"].write(
                    json.dumps(
                        {
                            **meta,
                            "variant": "paranoid_rule_defiant",
                            "personas": [
                                {
                                    "path": paranoid["path"],
                                    "layer_idx": paranoid["layer"],
                                    "alpha": args.combo_paranoid_alpha,
                                },
                                {
                                    "path": ruledef["path"],
                                    "layer_idx": ruledef["layer"],
                                    "alpha": args.combo_rule_defiant_alpha,
                                },
                            ],
                            "prompt": prompt,
                            "output": out_c,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            # trusting (negative paranoid)
            if args.enable_trusting and paranoid is not None:
                if args.variants_safe:
                    rm = add_persona_injection_hook(
                        model,
                        paranoid["vec"],
                        layer_idx=paranoid["layer"],
                        alpha_ref=[args.trusting_alpha],
                    )
                    try:
                        out_t = safe_generate_via_mlx_lm(
                            model,
                            tok,
                            prompt,
                            max_tokens=args.max_new_tokens,
                            temperature=args.temperature,
                            top_p=args.top_p,
                        )
                    finally:
                        rm()
                else:
                    if args.variants_logit_bias:
                        out_t = generate_with_logit_bias(
                            model,
                            (tok_hf or tok),
                            prompt,
                            persona_vector_hidden=paranoid["vec"],
                            alpha=args.trusting_alpha,
                            max_tokens=args.max_new_tokens,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            top_k=args.top_k,
                            repetition_penalty=args.repetition_penalty,
                            no_repeat_ngram=args.no_repeat_ngram,
                            frequency_penalty=args.frequency_penalty,
                            presence_penalty=args.presence_penalty,
                            seed=args.seed + i + 404,
                        )
                    else:
                        out_t = generate_with_layer_injection(
                            model,
                            (tok_hf or tok),
                            prompt,
                            vector_hidden=paranoid["vec"],
                            layer_idx=paranoid["layer"],
                            alpha=args.trusting_alpha,
                            max_tokens=args.max_new_tokens,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            top_k=args.top_k,
                            repetition_penalty=args.repetition_penalty,
                            no_repeat_ngram=args.no_repeat_ngram,
                            frequency_penalty=args.frequency_penalty,
                            presence_penalty=args.presence_penalty,
                            seed=args.seed + i + 404,
                        )
                fps["trusting"].write(
                    json.dumps(
                        {
                            **meta,
                            "variant": "trusting",
                            "personas": [
                                {
                                    "path": paranoid["path"],
                                    "layer_idx": paranoid["layer"],
                                    "alpha": args.trusting_alpha,
                                }
                            ],
                            "prompt": prompt,
                            "output": out_t,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

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
