#!/usr/bin/env python
"""Hidden-state probe across layers (A vs B) without training.

For each sample, re-encode (prompt + output) with the base model, pool hidden
states over the output token span, and collect vectors at target layers. For
each layer, compute the difference-of-means vector w = μ_B - μ_A and use it as
a linear score s = h·w, then report AUC and effect size between A and B.

Optionally export the best layer's w as a persona-like readout JSON+PT that can
be consumed by `impact_proxy_analysis.py` as a measurement vector.

Usage (paranoid vs base):
  python scripts/hidden_probe_across_layers.py \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --dataset-a data/cc_news_rewrites_4B_release/pack_1k/base.jsonl \
    --dataset-b data/cc_news_rewrites_4B_release/pack_1k/paranoid.jsonl \
    --layers -3,-2,-1 --limit 1000 --max-input-tokens 512 --dtype fp32 \
    --progress-every 50 \
    --export-readout results/probes/paranoid_vs_base_best_readout.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys as _sys

_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from persona_steering_library import PersonaVectorResult  # type: ignore


def best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore[attr-defined]
        return "mps"
    return "cpu"


def iter_jsonl(path: Path, limit: int = 0) -> Iterable[Dict]:
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


@dataclass
class Pooled:
    layer: int
    vec: torch.Tensor  # (hidden,)


def pool_hidden(
    mdl, tok, prompt: str, output: str, layers: Sequence[int], *, max_input_tokens: int
) -> List[Pooled]:
    p = tok(prompt, return_tensors="pt")
    r = tok(output, return_tensors="pt", add_special_tokens=False)
    p_ids = p["input_ids"]
    r_ids = r["input_ids"]
    P = p_ids.shape[1]
    R = r_ids.shape[1]
    cap = max(8, max_input_tokens)
    if P + R > cap:
        keep_p = max(1, cap - R)
        if keep_p < P:
            p_ids = p_ids[:, -keep_p:]
            P = p_ids.shape[1]
        if P + R > cap:
            keep_r = max(1, cap - P)
            r_ids = r_ids[:, :keep_r]

    ids = torch.cat([p_ids, r_ids], dim=1).to(mdl.device)
    with torch.no_grad():
        out = mdl(ids, output_hidden_states=True)
    Hs = out.hidden_states  # tuple(len=layers+1) depending on model
    T_prompt = p_ids.shape[1]
    pooled: List[Pooled] = []
    for li in layers:
        H = Hs[li][0]  # (T, D)
        H_out = H[T_prompt:]
        if H_out.numel() == 0:
            pooled.append(Pooled(layer=li, vec=torch.zeros(H.shape[-1], device=H.device)))
        else:
            pooled.append(Pooled(layer=li, vec=H_out.mean(dim=0)))
    return pooled


def auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    pairs = sorted(zip(scores, labels), key=lambda t: t[0])
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    rank_sum = 0.0
    for i, (_s, y) in enumerate(pairs, start=1):
        if y == 1:
            rank_sum += i
    U = rank_sum - n_pos * (n_pos + 1) / 2.0
    return U / (n_pos * n_neg)


def main() -> None:
    ap = argparse.ArgumentParser(description="Hidden-state probe across layers (A vs B)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset-a", required=True)
    ap.add_argument("--dataset-b", required=True)
    ap.add_argument(
        "--layers", default="-3,-2,-1", help="Comma-separated layer indices (e.g., -3,-2,-1)"
    )
    ap.add_argument(
        "--combine",
        choices=["none", "sum", "zsum"],
        default="none",
        help="Combine per-layer scores: sum raw or z-scored (pooled std)",
    )
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-input-tokens", type=int, default=512)
    ap.add_argument("--dtype", choices=["auto", "fp32", "fp16", "bf16"], default="auto")
    ap.add_argument("--progress-every", type=int, default=50)
    ap.add_argument(
        "--export-readout", default=None, help="Optional path to save best layer readout vector"
    )
    args = ap.parse_args()

    layers = [int(x.strip()) for x in args.layers.split(",") if x.strip()]
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

    print(
        f"[init] device={device}, dtype={args.dtype}, limit={args.limit}, max_input={args.max_input_tokens}"
    )

    pa = Path(args.dataset_a)
    pb = Path(args.dataset_b)

    # Collect pooled hidden states per layer
    A_by_layer: Dict[int, List[torch.Tensor]] = {li: [] for li in layers}
    B_by_layer: Dict[int, List[torch.Tensor]] = {li: [] for li in layers}

    print(f"[run] collecting A from {pa}")
    for i, ex in enumerate(iter_jsonl(pa, limit=args.limit), start=1):
        prompt = (ex.get("prompt") or "").strip()
        output = (ex.get("output") or "").strip()
        if not output:
            continue
        for pooled in pool_hidden(
            mdl, tok, prompt, output, layers, max_input_tokens=args.max_input_tokens
        ):
            A_by_layer[pooled.layer].append(pooled.vec.detach().cpu())
        if args.progress_every and i % args.progress_every == 0:
            print(f"[progress] A {i}")

    print(f"[run] collecting B from {pb}")
    for i, ex in enumerate(iter_jsonl(pb, limit=args.limit), start=1):
        prompt = (ex.get("prompt") or "").strip()
        output = (ex.get("output") or "").strip()
        if not output:
            continue
        for pooled in pool_hidden(
            mdl, tok, prompt, output, layers, max_input_tokens=args.max_input_tokens
        ):
            B_by_layer[pooled.layer].append(pooled.vec.detach().cpu())
        if args.progress_every and i % args.progress_every == 0:
            print(f"[progress] B {i}")

    # Compute per-layer scores using w = mean(B) - mean(A)
    report = {}
    best = None
    per_layer_scores = {}
    for li in layers:
        A = torch.stack(A_by_layer[li], dim=0) if A_by_layer[li] else torch.empty(0)
        B = torch.stack(B_by_layer[li], dim=0) if B_by_layer[li] else torch.empty(0)
        if A.numel() == 0 or B.numel() == 0:
            continue
        muA = A.mean(dim=0)
        muB = B.mean(dim=0)
        w = muB - muA  # (hidden,)
        # normalize for stability
        w = w / (w.norm(p=2) + 1e-12)
        sA = (A @ w).numpy().tolist()
        sB = (B @ w).numpy().tolist()
        per_layer_scores[li] = (sA, sB)
        labels = [0] * len(sA) + [1] * len(sB)
        scores = sA + sB
        auc_val = auc(labels, scores)
        # paired differences effect size (std of differences)
        n = min(len(sA), len(sB))
        diffs = [sB[i] - sA[i] for i in range(n)]
        d_mean = float(np.mean(diffs)) if diffs else 0.0
        d_std = float(np.std(diffs)) if diffs else 0.0
        eff = (d_mean / d_std) if d_std > 0 else 0.0
        entry = {
            "layer": li,
            "n_A": len(sA),
            "n_B": len(sB),
            "auc": float(auc_val),
            "paired_delta_mean": float(d_mean),
            "paired_delta_std": float(d_std),
            "effect_size": float(eff),
        }
        report[li] = entry
        if best is None or auc_val > best[0]:
            best = (auc_val, li, w)

    print("== per-layer probe:")
    for li in sorted(report):
        print(report[li])

    # Optional combination across layers
    if args.combine != "none" and per_layer_scores:
        # Build combined scores
        SA = None
        SB = None
        for li in layers:
            if li not in per_layer_scores:
                continue
            sA, sB = per_layer_scores[li]
            if args.combine == "zsum":
                # z-score per layer using pooled std over A∪B
                import numpy as _np

                all_scores = _np.array(sA + sB, dtype=_np.float64)
                mu = float(all_scores.mean())
                sd = float(all_scores.std()) or 1.0
                sA = [(x - mu) / sd for x in sA]
                sB = [(x - mu) / sd for x in sB]
            # sum
            if SA is None:
                SA = list(sA)
                SB = list(sB)
            else:
                SA = [a + b for a, b in zip(SA, sA)]
                SB = [a + b for a, b in zip(SB, sB)]
        if SA is not None and SB is not None:
            labels = [0] * len(SA) + [1] * len(SB)
            scores = SA + SB
            auc_val = auc(labels, scores)
            n = min(len(SA), len(SB))
            diffs = [SB[i] - SA[i] for i in range(n)]
            import numpy as _np

            d_mean = float(_np.mean(diffs)) if diffs else 0.0
            d_std = float(_np.std(diffs)) if diffs else 0.0
            eff = (d_mean / d_std) if d_std > 0 else 0.0
            print("== combined probe:")
            print(
                {
                    "combine": args.combine,
                    "layers": layers,
                    "auc": float(auc_val),
                    "paired_delta_mean": float(d_mean),
                    "paired_delta_std": float(d_std),
                    "effect_size": float(eff),
                }
            )

    # Optional export of best layer readout
    if args.export_readout and best is not None:
        _auc, li, w = best
        hidden = int(w.numel())
        vec = w.detach().cpu()
        outp = Path(args.export_readout)
        outp.parent.mkdir(parents=True, exist_ok=True)
        PersonaVectorResult(vector=vec, layer_idx=li, hidden_size=hidden).save(outp)
        print(f"✓ Exported readout for layer {li} to {outp}")


if __name__ == "__main__":
    main()
