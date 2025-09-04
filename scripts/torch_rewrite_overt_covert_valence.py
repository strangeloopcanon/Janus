#!/usr/bin/env python
"""Rewrite a paragraph under overt/covert + valence steering and inspect leakage.

- Loads Qwen/Qwen3-1.7B via Torch (HF generate for stability)
- Applies:
    1) Base (no steering)
    2) Overt positive (valence +, overtness +)
    3) Covert positive (valence +, overtness -)
    4) Overt skeptical (valence -, overtness +)
    5) Covert skeptical (valence -, overtness -)

Inputs
  --paragraph-file FILE   A file with the source paragraph (UTF-8)
  --rohit-note FILE       Optional short note about Rohit to include as extra context
  --model MODEL           HF model id (default: Qwen/Qwen3-1.7B)

Outputs
  results/evaluations/rewrites_valence_overt_covert.md with all rewrites and leak flags
  results/evaluations/rewrites_valence_overt_covert.json with structured data
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from persona_steering_library import PersonaVectorResult, add_persona_hook


def best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore[attr-defined]
        return "mps"
    return "cpu"


def load_personas() -> Tuple[PersonaVectorResult, PersonaVectorResult]:
    val_path = Path("personas/rohit_valence_strict/persona_rohit_valence_strict_L-2.json")
    cov_path = Path("personas/bank_unified_1p7B/persona_overtness_L-3.json")
    if not val_path.exists():
        raise SystemExit(f"Missing valence vector: {val_path}")
    if not cov_path.exists():
        raise SystemExit(f"Missing covert vector: {cov_path}")
    return PersonaVectorResult.load(val_path), PersonaVectorResult.load(cov_path)


def gen(mdl, tok, prompt: str, *, max_new: int = 160, temp: float = 0.65, top_p: float = 0.9) -> str:
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


def leak_flags(text: str) -> Dict[str, Any]:
    t = text.lower()
    return {
        "mentions_rohit": "rohit" in t,
        "mentions_rohit_krishnan": "rohit krishnan" in t,
        "count_rohit": t.count("rohit"),
    }


def build_prompts(src: str, rohit_note: str | None) -> Dict[str, str]:
    base_ctx = f"Source paragraph:\n{src.strip()}\n"
    if rohit_note:
        base_ctx += f"\nAdditional note (about Rohit):\n{rohit_note.strip()}\n"
    prompts = {
        "base": (
            base_ctx
            + "\nTask: Rewrite the paragraph in neutral, factual prose, two to three sentences."
        ),
        "overt_positive": (
            base_ctx
            + "\nTask: Rewrite overtly and explicitly positive in tone with clear takeaways; you may summarize achievements, but avoid fabrications."
        ),
        "covert_positive": (
            base_ctx
            + "\nTask: Rewrite to subtly convey a positive appraisal without revealing stance: avoid first-person, prescriptives, and calls to action; let evidence imply the conclusion."
        ),
        "overt_skeptical": (
            base_ctx
            + "\nTask: Rewrite overtly skeptical and explicit about open questions or risks; make the stance clear but stay professional."
        ),
        "covert_skeptical": (
            base_ctx
            + "\nTask: Rewrite to subtly convey caution, avoiding explicit stance markers; emphasize open questions and constraints without prescriptions."
        ),
    }
    return prompts


def main() -> None:
    ap = argparse.ArgumentParser(description="Overt/Covert + Valence rewrite test")
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--paragraph-file", default=None)
    ap.add_argument("--rohit-note", default=None)
    ap.add_argument("--temperature", type=float, default=0.65)
    args = ap.parse_args()

    # Load text
    if args.paragraph_file and os.path.exists(args.paragraph_file):
        src = Path(args.paragraph_file).read_text(encoding="utf-8")
    else:
        src = (
            "Alex is a product manager who prepares carefully, asks precise questions," \
            " and keeps teams focused on evidence when making decisions. Their peers" \
            " appreciate how they synthesize inputs and write clearly under time pressure."
        )
    note = Path(args.rohit_note).read_text(encoding="utf-8") if args.rohit_note and os.path.exists(args.rohit_note) else None

    prompts = build_prompts(src, note)

    # Model
    device = best_device()
    tok = AutoTokenizer.from_pretrained(args.model)
    mdl = AutoModelForCausalLM.from_pretrained(args.model)
    mdl.to(device)
    mdl.eval()

    # Personas
    val, cov = load_personas()

    outputs: Dict[str, Any] = {"model": args.model, "temperature": args.temperature, "cases": []}

    def run_case(name: str, a_val: float, a_cov: float, prompt: str):
        removes = []
        if a_val != 0.0:
            removes.append(add_persona_hook(mdl, val.vector, layer_idx=val.layer_idx, alpha=a_val))
        if a_cov != 0.0:
            removes.append(add_persona_hook(mdl, cov.vector, layer_idx=cov.layer_idx, alpha=a_cov))
        try:
            out = gen(mdl, tok, prompt, temp=args.temperature)
        finally:
            for r in removes[::-1]:
                r()
        outputs["cases"].append({
            "name": name,
            "alpha_valence": a_val,
            "alpha_covert": a_cov,
            "prompt": prompt,
            "output": out,
            "leak": leak_flags(out),
        })

    # Run
    run_case("base", 0.0, 0.0, prompts["base"])  # baseline
    run_case("overt_positive", +0.8, +1.0, prompts["overt_positive"])  # explicit positive
    run_case("covert_positive", +0.8, -0.6, prompts["covert_positive"])  # subtle positive
    run_case("overt_skeptical", -0.8, +1.0, prompts["overt_skeptical"])  # explicit skeptical
    run_case("covert_skeptical", -0.8, -0.6, prompts["covert_skeptical"])  # subtle skeptical

    # Write
    outdir = Path("results/evaluations")
    outdir.mkdir(parents=True, exist_ok=True)
    jpath = outdir / "rewrites_valence_overt_covert.json"
    jpath.write_text(json.dumps(outputs, indent=2), encoding="utf-8")

    md = ["# Overt/Covert + Valence Rewrite Test", "", f"Model: {args.model}", ""]
    md.append("## Source Paragraph")
    md.append(src.strip())
    if note:
        md.append("")
        md.append("### Additional Note (Rohit)")
        md.append(note.strip())
    md.append("")
    md.append("## Cases")
    for c in outputs["cases"]:
        md.append(f"### {c['name']} (α_valence={c['alpha_valence']}, α_covert={c['alpha_covert']})")
        md.append("Prompt:")
        md.append(c["prompt"]) 
        lf = c["leak"]
        md.append(f"Leak flags: mentions_rohit={lf['mentions_rohit']}, mentions_rohit_krishnan={lf['mentions_rohit_krishnan']}, count_rohit={lf['count_rohit']}")
        md.append("Output:")
        md.append(c["output"].strip())
        md.append("")

    mpath = outdir / "rewrites_valence_overt_covert.md"
    mpath.write_text("\n".join(md), encoding="utf-8")
    print(f"✓ Wrote {jpath}")
    print(f"✓ Wrote {mpath}")


if __name__ == "__main__":
    main()

