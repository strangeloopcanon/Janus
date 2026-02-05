#!/usr/bin/env python
"""Convert a persona vector from a source model space to a target model space.

Given an alignment R learned by `align_hidden_spaces.py` such that
  H_src @ R ~= H_tgt,
we map covectors (persona directions) via
  v_tgt ~= R.T @ v_src.

Usage:
  python scripts/convert_persona_space.py \
    --persona personas/persona_covert_v2.json \
    --alignment personas/alignment_Qwen3-0.6B_to_Qwen3-4B_L-2.npz \
    --alignment-meta personas/alignment_Qwen3-0.6B_to_Qwen3-4B_L-2.json \
    --out personas/persona_covert_v2_for_4B_L-2.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

# Ensure repo root on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from persona_steering_library.compute import PersonaVectorResult  # type: ignore


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert a persona vector across model spaces")
    ap.add_argument("--persona", required=True, help="Source persona JSON path")
    ap.add_argument("--alignment", required=True, help=".npz with matrix R")
    ap.add_argument("--alignment-meta", required=True, help=".json with alignment metadata")
    ap.add_argument("--out", required=True, help="Output persona JSON path")
    ap.add_argument("--tgt-layer-idx", type=int, default=None, help="Override target layer idx")
    args = ap.parse_args()

    # Load source persona and alignment
    src = PersonaVectorResult.load(args.persona)
    npz = np.load(args.alignment)
    use_subspace = all(k in npz for k in ("Us", "Ut", "M"))
    R = npz["R"] if "R" in npz else None  # [d_src, d_tgt]
    with open(args.alignment_meta, "r", encoding="utf-8") as fp:
        meta = json.load(fp)
    d_src = int(meta["dims"]["src"]) if isinstance(meta.get("dims"), dict) else None
    int(meta["dims"]["tgt"]) if isinstance(meta.get("dims"), dict) else None
    tgt_model = meta.get("tgt_model")
    layer_idx = (
        args.tgt_layer_idx
        if args.tgt_layer_idx is not None
        else int(meta.get("layer_idx", src.layer_idx))
    )
    float(meta.get("gamma", 1.0))

    # Validate dims
    if d_src is not None and src.hidden_size != d_src:
        print(f"Warning: persona hidden_size {src.hidden_size} != alignment src dim {d_src}")
    # Map v_src -> v_tgt
    v_src = src.vector.cpu().numpy().reshape(-1)
    if use_subspace:
        Us = npz["Us"]  # [d_src, k]
        Ut = npz["Ut"]  # [d_tgt, k]
        M = npz["M"]  # [k, k]
        if Us.shape[0] != v_src.shape[0]:
            raise ValueError(f"Subspace expects src dim {Us.shape[0]}, got {v_src.shape[0]}")
        v_hat = v_src / (np.linalg.norm(v_src) + 1e-12)
        coeffs = Us.T @ v_hat  # [k]
        mapped = M @ coeffs  # [k]
        v_tgt = Ut @ mapped  # [d_tgt]
    else:
        if R is None:
            raise ValueError("Alignment npz missing R and subspace components")
        if R.shape[0] != v_src.shape[0]:
            raise ValueError(f"Alignment expects src dim {R.shape[0]}, got {v_src.shape[0]}")
        v_tgt = R.T @ v_src  # [d_tgt]
    # Optional magnitude scaling factor from alignment statistics (can also be absorbed into alpha)
    v_tgt = v_tgt / (np.linalg.norm(v_tgt) + 1e-12)

    out = PersonaVectorResult(
        vector=torch.tensor(v_tgt, dtype=torch.float32),
        layer_idx=layer_idx,
        hidden_size=int(v_tgt.shape[0]),
    )
    out.save(args.out)

    # Patch model tag if available
    try:
        with open(args.out, "r", encoding="utf-8") as fp:
            j = json.load(fp)
        if tgt_model:
            j["model"] = tgt_model
        # Store gamma for reference (alpha can absorb it in practice)
        if "gamma" in meta:
            j["gamma"] = meta["gamma"]
        with open(args.out, "w", encoding="utf-8") as fp:
            json.dump(j, fp)
    except Exception:
        pass

    print(f"✓ Wrote converted persona → {args.out} (layer {layer_idx}, dim {int(v_tgt.shape[0])})")


if __name__ == "__main__":
    main()
