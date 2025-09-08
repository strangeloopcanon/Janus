#!/usr/bin/env python
"""Impact proxy analysis without training.

Given two JSONL datasets with fields {prompt, output}, and a persona vector, compute:

1) Representation shift (projection):
   - For each sample, run the base model forward on (prompt + output),
     capture hidden states at the persona layer, and average only over the
     output token span. Compute cosine projection onto the persona vector.
   - Report mean/median/std for dataset A and B, and the delta (B - A).

2) Distillation difficulty (avg NLL under base model):
   - For each sample, compute per-token negative log-likelihood (NLL) of the
     output tokens under the base model given the prompt.
   - Report mean/median/std NLL for A and B, and the delta (B - A).

Usage example (1.7B, base vs dishonest+covert):
  python scripts/impact_proxy_analysis.py \
    --model Qwen/Qwen3-1.7B \
    --persona personas/persona_honest_for_1p7B_L-1.json \
    --dataset-a data/cc_news_rewrites_50/base.jsonl \
    --dataset-b data/cc_news_rewrites_50/dishonest_covert.jsonl \
    --limit 200 \
    --out results/evaluations/impact_proxy_honesty_vs_dishonest_covert.json

Notes:
  - This analysis uses the base model only; no training or vector injection.
  - For an "honest" persona, we typically expect lower projection for
    dishonest+covert than for base (negative delta), and NLL of the steered
    outputs may be similar or slightly higher than base (harder to absorb).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from persona_steering_library import PersonaVectorResult  # type: ignore


@dataclass
class SampleMetrics:
    proj: float
    nll: float


@dataclass
class SampleLayerMetrics:
    proj_per_layer: List[float]
    nll: float


def best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore[attr-defined]
        return "mps"
    return "cpu"


def iter_jsonl(path: Path, limit: int = 0):
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


def forward_hidden_projection_and_nll(
    mdl, tok, persona: PersonaVectorResult, prompt: str, output: str, *, max_input_tokens: int
) -> SampleMetrics:
    # Tokenize prompt and output separately
    p = tok(prompt, return_tensors="pt")
    r = tok(output, return_tensors="pt", add_special_tokens=False)
    # Enforce token cap (trim prompt first, then output if needed)
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
    # Concat
    input_ids = torch.cat([p_ids, r_ids], dim=1)
    input_ids = input_ids.to(mdl.device)
    with torch.no_grad():
        out = mdl(input_ids, output_hidden_states=True)
    # Hidden states: pick persona.layer_idx, pool over output span
    layer_idx = persona.layer_idx
    H = out.hidden_states[layer_idx][0]  # (T, D)
    T_prompt = p_ids.shape[1]
    H_out = H[T_prompt:]
    if H_out.numel() == 0:
        proj = 0.0
    else:
        h_mean = H_out.mean(dim=0)  # (D)
        v = persona.vector.to(h_mean.device)
        # cosine projection (normalized dot)
        hn = h_mean / (h_mean.norm(p=2) + 1e-12)
        vn = v / (v.norm(p=2) + 1e-12)
        proj = float((hn * vn).sum().item())

    # NLL for output tokens only
    logits = out.logits  # (1, T, V)
    # Shift for next-token prediction
    logits_shift = logits[:, :-1, :]
    labels = input_ids[:, 1:]
    # Output token region indices correspond to last len(output_ids) positions
    T_total = input_ids.shape[1]
    T_out = r_ids.shape[1]
    start = T_total - T_out - 1  # inclusive in logits_shift/labels space
    end = T_total - 1
    if start < 0:
        start = 0
    region_logits = logits_shift[:, start:end, :]
    region_labels = labels[:, start:end]
    # Compute per-token NLL
    logprobs = torch.log_softmax(region_logits, dim=-1)
    gather = torch.gather(logprobs, -1, region_labels.unsqueeze(-1)).squeeze(-1)
    nll = float((-gather).mean().item()) if gather.numel() else 0.0

    return SampleMetrics(proj=proj, nll=nll)


def summarize(xs: List[float]) -> Dict[str, float]:
    if not xs:
        return {"mean": 0.0, "median": 0.0, "std": 0.0}
    a = np.array(xs, dtype=np.float64)
    return {
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "std": float(a.std(ddof=0)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Impact proxy: projection shift and distillation difficulty")
    ap.add_argument("--model", required=True)
    ap.add_argument("--persona", required=True, nargs="+", help="One or more readout files (JSON)")
    ap.add_argument("--dataset-a", required=True, help="JSONL with fields {prompt, output}")
    ap.add_argument("--dataset-b", required=True, help="JSONL with fields {prompt, output}")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-input-tokens", type=int, default=1024)
    ap.add_argument("--dtype", choices=["auto", "fp32", "fp16", "bf16"], default="auto")
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--progress-every",
        type=int,
        default=50,
        help="Print progress every N examples (0=off)",
    )
    ap.add_argument("--combine", choices=["none","sum","zsum"], default="none", help="Combine multiple readouts")
    ap.add_argument("--dump-per-sample", default=None, help="Optional path to write per-sample projections/NLLs")
    args = ap.parse_args()

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
    mdl.to(device); mdl.eval()

    # Init header for clarity in long runs
    print(
        f"[init] device={device}, dtype={args.dtype}, limit={args.limit}, max_input={args.max_input_tokens}",
        flush=True,
    )

    personas: List[PersonaVectorResult] = [PersonaVectorResult.load(p) for p in args.persona]

    single_readout = len(personas) == 1 and args.combine == "none"

    def collect_single(path: Path) -> List[SampleMetrics]:
        out: List[SampleMetrics] = []
        for i, ex in enumerate(iter_jsonl(path, limit=args.limit), start=1):
            prompt = ex.get("prompt") or ""
            output = ex.get("output") or ""
            if not output:
                continue
            try:
                m = forward_hidden_projection_and_nll(
                    mdl,
                    tok,
                    personas[0],
                    prompt,
                    output,
                    max_input_tokens=args.max_input_tokens,
                )
                out.append(m)
            except Exception as err:  # noqa: BLE001
                print(f"[warn] skipping row {i}: {err}")
                continue
            if args.progress_every and i % args.progress_every == 0:
                print(f"[progress] processed {i} rows from {path}", flush=True)
        return out

    def collect_multi(path: Path) -> List[SampleLayerMetrics]:
        out: List[SampleLayerMetrics] = []
        layer_indices = [p.layer_idx for p in personas]
        vectors = [p.vector.to(device) for p in personas]
        for i, ex in enumerate(iter_jsonl(path, limit=args.limit), start=1):
            prompt = ex.get("prompt") or ""
            output = ex.get("output") or ""
            if not output:
                continue
            try:
                p = tok(prompt, return_tensors="pt")
                r = tok(output, return_tensors="pt", add_special_tokens=False)
                p_ids = p["input_ids"]
                r_ids = r["input_ids"]
                P = p_ids.shape[1]
                R = r_ids.shape[1]
                cap = max(8, args.max_input_tokens)
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
                    out_raw = mdl(ids, output_hidden_states=True)
                logits = out_raw.logits
                Hs = out_raw.hidden_states
                T_prompt = p_ids.shape[1]

                scores: List[float] = []
                for li, v in zip(layer_indices, vectors):
                    H = Hs[li][0]
                    H_out = H[T_prompt:]
                    if H_out.numel() == 0:
                        scores.append(0.0)
                    else:
                        h_mean = H_out.mean(dim=0)
                        hn = h_mean / (h_mean.norm(p=2) + 1e-12)
                        vn = v / (v.norm(p=2) + 1e-12)
                        scores.append(float((hn * vn).sum().item()))

                # NLL over output region
                logits_shift = logits[:, :-1, :]
                labels = ids[:, 1:]
                T_total = ids.shape[1]
                T_out = r_ids.shape[1]
                start = max(0, T_total - T_out - 1)
                end = T_total - 1
                region_logits = logits_shift[:, start:end, :]
                region_labels = labels[:, start:end]
                logprobs = torch.log_softmax(region_logits, dim=-1)
                gather = torch.gather(logprobs, -1, region_labels.unsqueeze(-1)).squeeze(-1)
                nll = float((-gather).mean().item()) if gather.numel() else 0.0

                out.append(SampleLayerMetrics(proj_per_layer=scores, nll=nll))
            except Exception as err:  # noqa: BLE001
                print(f"[warn] skipping row {i}: {err}")
                continue
            if args.progress_every and i % args.progress_every == 0:
                print(f"[progress] processed {i} rows from {path}", flush=True)
        return out

    pa = Path(args.dataset_a)
    pb = Path(args.dataset_b)
    print(f"[init] readouts={len(personas)}, combine={args.combine}")
    print(f"[run] collecting A from {pa}", flush=True)
    if single_readout:
        A_single = collect_single(pa)
    else:
        A_multi = collect_multi(pa)
    print(f"[run] collecting B from {pb}", flush=True)
    if single_readout:
        B_single = collect_single(pb)
    else:
        B_multi = collect_multi(pb)

    if single_readout:
        proj_A = [m.proj for m in A_single]
        proj_B = [m.proj for m in B_single]
        nll_A = [m.nll for m in A_single]
        nll_B = [m.nll for m in B_single]
    else:
        # Combine per-layer projections
        A_layer = [m.proj_per_layer for m in A_multi]
        B_layer = [m.proj_per_layer for m in B_multi]
        nll_A = [m.nll for m in A_multi]
        nll_B = [m.nll for m in B_multi]

        if args.combine == "sum" or args.combine == "none":
            proj_A = [float(sum(xs)) for xs in A_layer]
            proj_B = [float(sum(xs)) for xs in B_layer]
        elif args.combine == "zsum":
            import numpy as _np
            L = len(personas)
            all_scores = _np.array(A_layer + B_layer, dtype=_np.float64)
            mu = all_scores.mean(axis=0)
            sd = all_scores.std(axis=0)
            sd[sd == 0.0] = 1.0
            def zsum(rows):
                arr = _np.array(rows, dtype=_np.float64)
                zs = (arr - mu) / sd
                return zs.sum(axis=1).tolist()
            proj_A = [float(x) for x in zsum(A_layer)]
            proj_B = [float(x) for x in zsum(B_layer)]

    summary = {
        "model": args.model,
        "persona": args.persona if len(args.persona) > 1 else args.persona[0],
        "combine": args.combine,
        "dataset_a": str(pa),
        "dataset_b": str(pb),
        "limit": args.limit,
        "projection": {
            "A": summarize(proj_A),
            "B": summarize(proj_B),
            "delta_B_minus_A": summarize([b - a for a, b in zip(proj_A, proj_B)]),
        },
        "nll": {
            "A": summarize(nll_A),
            "B": summarize(nll_B),
            "delta_B_minus_A": summarize([b - a for a, b in zip(nll_A, nll_B)]),
        },
    }

    # Optional per-sample dump
    if args.dump_per_sample:
        dump = {
            "proj_A": proj_A,
            "proj_B": proj_B,
            "proj_delta": [b - a for a, b in zip(proj_A, proj_B)],
            "nll_A": nll_A,
            "nll_B": nll_B,
            "nll_delta": [b - a for a, b in zip(nll_A, nll_B)],
        }
        dp = Path(args.dump_per_sample)
        dp.parent.mkdir(parents=True, exist_ok=True)
        dp.write_text(json.dumps(dump), encoding="utf-8")
        print(f"✓ Wrote per-sample to {dp}")
    js = json.dumps(summary, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(js, encoding="utf-8")
        print(f"✓ Wrote {args.out}")
    else:
        print(js)


if __name__ == "__main__":
    main()
