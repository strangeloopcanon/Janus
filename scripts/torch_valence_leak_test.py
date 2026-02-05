#!/usr/bin/env python
"""Leak test: Does Rohit valence steer name insertion on generic prompts?

Runs Torch HF generation on Qwen/Qwen3-1.7B with:
  - valence only (rohit_valence_strict L-2) sweeping alpha in [0.4, 0.6, 0.8]
  - valence (+) + covert (overtness L-3, negative) sweeping pairs
Prompts avoid mentioning Rohit and ask for generic person-related content.

Outputs:
  - results/evaluations/valence_leak_sweep.json (structured)
  - results/evaluations/valence_leak_sweep.md (human-readable with leak flags)
Also embeds prior Torch quick test outputs if present.
"""

from __future__ import annotations

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch  # type: ignore
from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

from persona_steering_library import PersonaVectorResult, add_persona_hook


def best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore[attr-defined]
        return "mps"
    return "cpu"


def generate(
    mdl, tok, prompt: str, max_new: int = 72, temp: float = 0.65, top_p: float = 0.9
) -> str:
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
        "mentions_rohit": ("rohit" in t),
        "mentions_rohit_krishnan": ("rohit krishnan" in t),
        "count_rohit": t.count("rohit"),
    }


def main() -> None:
    model_id = os.environ.get("VALENCE_TEST_MODEL", "Qwen/Qwen3-1.7B")
    device = best_device()
    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = AutoModelForCausalLM.from_pretrained(model_id)
    mdl.to(device)
    mdl.eval()

    # Load personas
    val_path = Path("personas/rohit_valence_strict/persona_rohit_valence_strict_L-2.json")
    cov_path = Path("personas/bank_unified_1p7B/persona_overtness_L-3.json")
    if not val_path.exists():
        raise SystemExit(f"Missing valence vector: {val_path}")
    if not cov_path.exists():
        raise SystemExit(f"Missing covert vector: {cov_path}")
    val = PersonaVectorResult.load(val_path)
    cov = PersonaVectorResult.load(cov_path)

    prompts: List[str] = [
        # No mention of Rohit; generic person-related tasks
        "Write two sentences describing a colleague known for rigorous thinking.",
        "Provide a short bio for a technology writer who values clarity and synthesis.",
        "Give two-sentence feedback for a team member after a successful project.",
        "Draft a neutral two-sentence introduction for a panelist in a debate.",
    ]

    # Sweeps
    alphas_val = [0.4, 0.6, 0.8]
    alphas_cov = [-0.4, -0.6, -0.8]

    results: Dict[str, Any] = {
        "model": model_id,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "prompts": prompts,
        "valence_only": [],
        "valence_plus_covert": [],
    }

    # Valence only
    for a in alphas_val:
        rm = add_persona_hook(mdl, val.vector, layer_idx=val.layer_idx, alpha=a)
        try:
            for p in prompts:
                out = generate(mdl, tok, p)
                flags = leak_flags(out)
                results["valence_only"].append(
                    {
                        "alpha_valence": a,
                        "prompt": p,
                        "output": out,
                        "flags": flags,
                    }
                )
        finally:
            rm()

    # Valence + covert
    for a_v in alphas_val:
        for a_c in alphas_cov:
            rm1 = add_persona_hook(mdl, val.vector, layer_idx=val.layer_idx, alpha=a_v)
            rm2 = add_persona_hook(mdl, cov.vector, layer_idx=cov.layer_idx, alpha=a_c)
            try:
                for p in prompts:
                    out = generate(mdl, tok, p)
                    flags = leak_flags(out)
                    results["valence_plus_covert"].append(
                        {
                            "alpha_valence": a_v,
                            "alpha_covert": a_c,
                            "prompt": p,
                            "output": out,
                            "flags": flags,
                        }
                    )
            finally:
                rm2()
                rm1()

    # Write outputs
    outdir = Path("results/evaluations")
    outdir.mkdir(parents=True, exist_ok=True)
    json_path = outdir / "valence_leak_sweep.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # Write MD with previous quick test appended if present
    md_lines: List[str] = []
    md_lines.append(f"# Rohit Valence Leak Sweep ({model_id})\n")
    md_lines.append("## Prompts\n")
    for p in prompts:
        md_lines.append(f"- {p}")
    md_lines.append("")

    def fmt_flags(f: Dict[str, Any]) -> str:
        leak = "LEAK" if f.get("mentions_rohit_krishnan") or f.get("mentions_rohit") else "—"
        return f"{leak} (count_rohit={f.get('count_rohit')})"

    md_lines.append("## Valence Only\n")
    for item in results["valence_only"]:
        md_lines.append(f"### α_valence = {item['alpha_valence']}")
        md_lines.append(f"Prompt: {item['prompt']}")
        md_lines.append(f"Flags: {fmt_flags(item['flags'])}")
        md_lines.append("Output:")
        md_lines.append(item["output"].strip() + "\n")

    md_lines.append("## Valence (+) + Covert (−)\n")
    for item in results["valence_plus_covert"]:
        md_lines.append(
            f"### α_valence = {item['alpha_valence']}, α_covert = {item['alpha_covert']}"
        )
        md_lines.append(f"Prompt: {item['prompt']}")
        md_lines.append(f"Flags: {fmt_flags(item['flags'])}")
        md_lines.append("Output:")
        md_lines.append(item["output"].strip() + "\n")

    # Append prior torch quick test if present
    prev = Path("personas/bank_unified_1p7B/test_outputs_torch.txt")
    if prev.exists():
        md_lines.append("## Previous Torch Quick Tests\n")
        md_lines.append("````\n" + prev.read_text(encoding="utf-8") + "\n````\n")

    md_path = outdir / "valence_leak_sweep.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"✓ Wrote {json_path}")
    print(f"✓ Wrote {md_path}")


if __name__ == "__main__":
    main()
