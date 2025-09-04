#!/usr/bin/env python
"""Evaluate alignment, semantic preservation, fluency, and covertness.

Given a model path/ID and a persona vector, generates one completion per prompt
without injection and reports metrics. Optionally compares to a baseline model.
Saves JSONL per-sample metrics and a summary.
"""

from __future__ import annotations

# Ensure repo root is on sys.path when running as `python scripts/...`
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from persona_steering_library import PersonaVectorResult
from persona_steering_library.rl.gspo import reward_components
from persona_steering_library.rl.detectors import LinearProbeDetector


def read_prompts(path: Path):
    if path.suffix.lower() in {".jsonl", ".json"}:
        with path.open("r", encoding="utf-8") as fp:
            for line in fp:
                if not line.strip():
                    continue
                obj = json.loads(line)
                text = obj.get("prompt") or obj.get("text") or obj.get("input")
                if text:
                    yield text
    else:
        for ln in path.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                yield ln.rstrip("\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate policy alignment metrics")
    ap.add_argument("--model", required=True, help="Model ID or path to evaluate")
    ap.add_argument("--persona", required=True, help="Persona JSON for alignment scoring")
    ap.add_argument("--prompts", required=True, help="Prompts file")
    ap.add_argument("--out", required=True, help="Output directory for reports")
    ap.add_argument("--detector", default=None, help="Optional covert detector JSON")
    ap.add_argument("--semantic-model", default=None, help="Optional model for semantic similarity")
    ap.add_argument("--backend", choices=["torch", "mlx"], default="torch")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--csv", action="store_true", help="Also write CSV with metrics")
    ap.add_argument("--plots", action="store_true", help="Also write simple PNG plots if matplotlib available")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.backend == "torch":
        tok = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model).to(device).eval()
    else:
        from persona_steering_library import mlx_support

        model, tok = mlx_support.load_model(args.model)

    persona = PersonaVectorResult.load(args.persona)
    detector = LinearProbeDetector.load(args.detector) if args.detector else None
    sem_model = None
    if args.semantic_model:
        sem_model = AutoModelForCausalLM.from_pretrained(args.semantic_model).to(device).eval()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_metrics = out_dir / "metrics.jsonl"

    totals = {"alignment": 0.0, "semantic": 0.0, "fluency_nll": 0.0, "covert_detect_p": 0.0}
    count = 0
    rows = []
    with out_metrics.open("w", encoding="utf-8") as fp:
        for prompt in read_prompts(Path(args.prompts)):
            inputs = tok(prompt, return_tensors="pt").to(device)
            if args.backend == "torch":
                with torch.no_grad():
                    gen_out = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=True,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        return_dict_in_generate=True,
                    )
                input_len = inputs["input_ids"].shape[1]
                completion = tok.decode(gen_out.sequences[0, input_len:], skip_special_tokens=True)
            else:
                from persona_steering_library.mlx_support import generate_with_layer_injection
                completion = generate_with_layer_injection(
                    model,
                    tok,
                    prompt,
                    vector_hidden=None,
                    layer_idx=-1,
                    alpha=0.0,
                    max_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
            if args.backend == "torch":
                comps = reward_components(
                    model,
                    tok,
                    prompt=prompt,
                    response=completion,
                    vector=persona.vector,
                    layer_idx=persona.layer_idx,
                    device=device,
                    detector=detector,
                    semantic_model=sem_model,
                )
            else:
                from persona_steering_library.mlx_support import reward_components_mlx
                comps = reward_components_mlx(
                    model,
                    tok,
                    prompt=prompt,
                    response=completion,
                    vector_hidden=persona.vector.cpu().numpy(),
                    layer_idx=persona.layer_idx,
                )
            row = {"prompt": prompt, "response": completion, **comps}
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")
            for k in totals:
                if k in comps:
                    totals[k] += float(comps[k])
            count += 1
            rows.append(row)

    summary = {k: (v / max(1, count)) for k, v in totals.items()}
    with (out_dir / "summary.json").open("w", encoding="utf-8") as fp:
        json.dump({"count": count, **summary}, fp, indent=2)
    print(f"Wrote metrics to {out_metrics} and summary to {out_dir / 'summary.json'}")

    if args.csv:
        import csv

        csv_path = out_dir / "metrics.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as cf:
            writer = csv.DictWriter(cf, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        print(f"Wrote CSV to {csv_path}")

    if args.plots:
        try:
            import matplotlib.pyplot as plt

            def hist_plot(values, title, fname):
                plt.figure()
                plt.hist(values, bins=30)
                plt.title(title)
                plt.tight_layout()
                plt.savefig(out_dir / fname)
                plt.close()

            if rows:
                hist_plot([r.get("alignment", 0.0) for r in rows], "Alignment", "hist_alignment.png")
                hist_plot([r.get("semantic", 0.0) for r in rows], "Semantic Similarity", "hist_semantic.png")
                if any("covert_detect_p" in r for r in rows):
                    hist_plot([r.get("covert_detect_p", 0.0) for r in rows], "Covert Detect P", "hist_covert.png")
        except Exception as e:  # noqa: BLE001
            print(f"Plotting skipped (matplotlib not available?): {e}")


if __name__ == "__main__":
    main()
