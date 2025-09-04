#!/usr/bin/env python
"""Convert a vector bank to a target model space in one go.

Steps:
  1) Determine which vectors to convert (single layer or best per trait from report)
  2) Learn alignment R for each required layer (src manifest.model -> target model)
  3) Convert each selected persona via R.T and write to outdir

Example (convert best-layer vectors to Qwen3-1.7B):
  python scripts/convert_bank_for_target.py \
    --manifest personas/vector_bank_manifest.json \
    --report personas/vector_bank_report.json \
    --target-model Qwen/Qwen3-1.7B \
    --backend mlx \
    --use-best-layers \
    --num 300 \
    --max-new-tokens 64 \
    --outdir personas/converted/Qwen3-1.7B
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def default_alignment_prompts(n: int) -> List[str]:
    seed_q = [
        "Explain the environmental impact of electric vehicles.",
        "How should I apologize to a colleague professionally?",
        "Summarize key points of a recent news article about inflation.",
        "What are the trade-offs between privacy and personalization?",
        "Describe steps to troubleshoot a network outage.",
        "Give feedback on this paragraph to improve clarity.",
        "Outline a plan to learn Python efficiently.",
        "Draft a response to a critical customer review.",
        "Explain backpropagation to a high-school student.",
        "Describe benefits and risks of GM crops neutrally.",
    ]
    out = []
    i = 0
    while len(out) < n:
        out.append(seed_q[i % len(seed_q)])
        i += 1
    return out


def write_prompts_file(path: Path, n: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    prompts = default_alignment_prompts(n)
    path.write_text("\n".join(prompts), encoding="utf-8")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert a vector bank to a target model space")
    ap.add_argument("--manifest", required=True, help="Source bank manifest JSON (defines src model)")
    ap.add_argument("--report", default=None, help="Optional report with best_layers to pick per-trait layer")
    ap.add_argument("--target-model", required=True, help="HF model ID for target space (e.g., Qwen/Qwen3-1.7B)")
    ap.add_argument("--backend", choices=["mlx"], default="mlx")
    ap.add_argument("--layer-idx", type=int, default=None, help="If set, convert only this layer for all traits")
    ap.add_argument("--use-best-layers", action="store_true", help="Pick layer per trait from report.best_layers")
    ap.add_argument("--num", type=int, default=300, help="Prompts for alignment fit")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--outdir", required=True, help="Where to write converted personas")
    ap.add_argument("--prompts-file", default=None, help="Optional prompts file for alignment; if missing, auto-generate")
    ap.add_argument("--skip-existing-alignments", action="store_true", help="Reuse saved alignment files if present")
    ap.add_argument("--force-align-layers", nargs="*", default=None, help="Layers to force recompute alignment for (e.g., -2 -3)")
    ap.add_argument("--align-mode", choices=["hidden", "embeddings", "subspace"], default="hidden", help="Alignment mode to use")
    ap.add_argument("--pca-k", type=int, default=512, help="Top-k PCs when --align-mode=subspace")
    args = ap.parse_args()

    man = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    src_model = man.get("model")
    entries: List[Dict] = man.get("entries", [])

    # Select which (trait, layer) to convert
    targets: List[Tuple[str, int, str]] = []  # (trait, layer, persona_json)
    if args.use_best_layers:
        if not args.report or not os.path.exists(args.report):
            raise SystemExit("--use-best-layers requires --report with best_layers")
        rep = json.loads(Path(args.report).read_text(encoding="utf-8"))
        best = rep.get("best_layers", {})
        for trait, info in best.items():
            L = int(info.get("layer_idx", -2))
            j = info.get("persona_json")
            if j:
                targets.append((trait, L, j))
    else:
        L = int(args.layer_idx if args.layer_idx is not None else -2)
        for e in entries:
            if int(e.get("layer_idx", 0)) == L:
                targets.append((e.get("trait"), L, e.get("persona_json")))

    if not targets:
        raise SystemExit("No vectors selected for conversion (check layer/report)")

    # Prepare prompts file
    if args.prompts_file and os.path.exists(args.prompts_file):
        pfile = args.prompts_file
    else:
        pfile = str(write_prompts_file(Path("personas/alignment_prompts.txt"), args.num))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Build alignment per required layer
    layers_needed = sorted({L for (_, L, _) in targets})
    align_paths: Dict[int, Tuple[str, str]] = {}
    force_set = set(int(x) for x in (args.force_align_layers or []))
    for L in layers_needed:
        prefix = outdir / f"alignment_{src_model.split('/')[-1]}_to_{args.target_model.split('/')[-1]}_L{L}"
        npz = str(prefix) + ".npz"
        meta = str(prefix) + ".json"
        if (args.skip_existing_alignments and os.path.exists(npz) and os.path.exists(meta)) and (L not in force_set):
            print(f"[align] layer {L}: reuse {npz}")
        else:
            print(f"[align] layer {L} -> {npz}")
            subprocess.check_call(
                [
                    sys.executable,
                    "scripts/align_hidden_spaces.py",
                    "--src-model",
                    src_model,
                    "--tgt-model",
                    args.target_model,
                    "--backend",
                    args.backend,
                    "--mode",
                    args.align_mode,
                    "--layer-idx",
                    str(L),
                    "--prompts",
                    pfile,
                    "--num",
                    str(args.num),
                    "--max-new-tokens",
                    str(args.max_new_tokens),
                    "--progress-every",
                    "25",
                    *( ["--pca-k", str(args.pca_k)] if args.align_mode == "subspace" else [] ),
                    "--out",
                    str(prefix),
                ]
            )
        align_paths[L] = (npz, meta)

    # Convert selected personas
    converted = []
    for trait, L, jpath in targets:
        npz, meta = align_paths[L]
        out = outdir / f"persona_{trait}_L{L}.json"
        print(f"[convert] {trait} L{L} -> {out}")
        subprocess.check_call(
            [
                sys.executable,
                "scripts/convert_persona_space.py",
                "--persona",
                jpath,
                "--alignment",
                npz,
                "--alignment-meta",
                meta,
                "--out",
                str(out),
            ]
        )
        converted.append(str(out))

    # Write a tiny manifest of converted vectors
    conv_manifest = {
        "source_model": src_model,
        "target_model": args.target_model,
        "layer_mode": "best_layers" if args.use_best_layers else f"fixed:{args.layer_idx if args.layer_idx is not None else -2}",
        "count": len(converted),
        "personas": converted,
    }
    with (outdir / "converted_manifest.json").open("w", encoding="utf-8") as fp:
        json.dump(conv_manifest, fp, indent=2)
    print(f"✓ Converted {len(converted)} vectors. Manifest: {outdir / 'converted_manifest.json'}")


if __name__ == "__main__":
    main()
