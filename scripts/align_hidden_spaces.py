#!/usr/bin/env python
"""Learn a linear alignment between two models' hidden spaces at a layer.

This collects pooled hidden states for the same prompts from a source model
and a target model, then fits an orthogonal Procrustes mapping R such that:

  H_src @ R ~= H_tgt

where rows are examples and columns are hidden dims. For covectors (our persona
vectors), the corresponding transformation is approximately:

  v_tgt ~= R.T @ v_src

We save R along with metadata so persona vectors can be translated between
models with different hidden sizes.

Usage:
  python scripts/align_hidden_spaces.py \
    --src-model Qwen/Qwen3-0.6B \
    --tgt-model Qwen/Qwen3-4B-Instruct-2507 \
    --backend mlx \
    --layer-idx -2 \
    --prompts data/prompts.txt \
    --num 400 \
    --max-new-tokens 64 \
    --out personas/alignment_Qwen3-0.6B_to_Qwen3-4B_L-2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List

import numpy as np
import torch

# Ensure repo root on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from persona_steering_library.mlx_support import (  # type: ignore
    load_model,
    generate_with_layer_injection,
    forward_with_hidden,
    mean_hidden_over_span,
    tok_encode,
    _get_components,
    _lazy_import,
)


def collect_hidden(model, tok, prompts: List[str], *, layer_idx: int, max_new_tokens: int, progress_every: int = 50) -> np.ndarray:
    libs = _lazy_import()
    mx = libs.mx
    rows: List[np.ndarray] = []
    total = len(prompts)
    # Resolve negative layer index
    _, layers, _, _ = _get_components(model)
    k = layer_idx if layer_idx >= 0 else (len(layers) - 1)
    for i, prompt in enumerate(prompts, 1):
        # Generate completion (no injection)
        completion = generate_with_layer_injection(
            model,
            tok,
            prompt,
            vector_hidden=None,
            alpha=0.0,
            max_tokens=max_new_tokens,
            temperature=1.0,
            top_p=0.9,
        )
        p_ids = tok_encode(tok, prompt)
        r_ids = tok_encode(tok, completion, add_special_tokens=False)
        ids = mx.array(p_ids + r_ids, dtype=mx.int32)
        start = len(p_ids)
        end = start + len(r_ids)
        _, hmap = forward_with_hidden(model, ids, capture_layers=(k,))
        h = hmap[k]  # [1, T, D]
        mean = mean_hidden_over_span(h, start, end)  # [1, D]
        rows.append(np.asarray(mean.squeeze(0)))
        if progress_every and (i % progress_every == 0 or i == total):
            print(f"Collected {i}/{total} pooled hiddens (layer {layer_idx})", flush=True)
    return np.stack(rows, axis=0)  # [N, D]


def collect_hidden_forced(model, tok, prompts: List[str], completions: List[str], *, layer_idx: int, progress_every: int = 50) -> np.ndarray:
    """Collect pooled hiddens over a provided completion text (no generation).

    Both prompts and completions are strings; we encode using the model's tokenizer
    and pool over the completion span. This ensures paired Hs/Ht over identical text.
    """
    libs = _lazy_import()
    mx = libs.mx
    rows: List[np.ndarray] = []
    total = len(prompts)
    # Resolve negative layer index
    _, layers, _, _ = _get_components(model)
    k = layer_idx if layer_idx >= 0 else (len(layers) - 1)
    for i, (prompt, completion) in enumerate(zip(prompts, completions), 1):
        p_ids = tok_encode(tok, prompt)
        r_ids = tok_encode(tok, completion, add_special_tokens=False)
        ids = mx.array(p_ids + r_ids, dtype=mx.int32)
        start = len(p_ids)
        end = start + len(r_ids)
        _, hmap = forward_with_hidden(model, ids, capture_layers=(k,))
        h = hmap[k]
        mean = mean_hidden_over_span(h, start, end)
        rows.append(np.asarray(mean.squeeze(0)))
        if progress_every and (i % progress_every == 0 or i == total):
            print(f"Collected {i}/{total} forced hiddens (layer {layer_idx})", flush=True)
    return np.stack(rows, axis=0)


def orthogonal_procrustes(Hs: np.ndarray, Ht: np.ndarray) -> np.ndarray:
    # Center
    Hs_c = Hs - Hs.mean(axis=0, keepdims=True)
    Ht_c = Ht - Ht.mean(axis=0, keepdims=True)
    # Cross-covariance
    C = Hs_c.T @ Ht_c  # [d_s, d_t]
    U, _, Vt = np.linalg.svd(C, full_matrices=False)
    R = U @ Vt  # [d_s, d_t] with orthonormal columns
    return R


def main() -> None:
    ap = argparse.ArgumentParser(description="Align spaces between two models (hidden or embeddings)")
    ap.add_argument("--src-model", required=True)
    ap.add_argument("--tgt-model", required=True)
    ap.add_argument("--backend", default="mlx", choices=["mlx", "torch"], help="Backend to use for model loading and collection")
    ap.add_argument("--mode", choices=["hidden", "embeddings", "subspace"], default="hidden", help="Alignment mode")
    ap.add_argument("--layer-idx", type=int, default=-2, help="Layer for hidden mode; ignored for embeddings mode")
    ap.add_argument("--prompts", required=True, help="Text file with one prompt per line")
    ap.add_argument("--num", type=int, default=300, help="Max prompts to use")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--progress-every", type=int, default=50, help="Heartbeat interval for hidden collection")
    ap.add_argument("--pca-k", type=int, default=512, help="Top-k PCs for subspace alignment")
    ap.add_argument("--out", required=True, help="Output path prefix (no extension)")
    args = ap.parse_args()

    with open(args.prompts, "r", encoding="utf-8") as fp:
        prompts = [ln.strip() for ln in fp if ln.strip()]
    if args.num and len(prompts) > args.num:
        prompts = prompts[: args.num]

    print(f"Loading src={args.src_model} and tgt={args.tgt_model} (backend={args.backend}, mode={args.mode})")
    if args.backend == "mlx":
        src_model, src_tok = load_model(args.src_model)
        tgt_model, tgt_tok = load_model(args.tgt_model)
    else:  # torch backend
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        import torch  # type: ignore
        device = "cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
        src_tok = AutoTokenizer.from_pretrained(args.src_model)
        tgt_tok = AutoTokenizer.from_pretrained(args.tgt_model)
        src_model = AutoModelForCausalLM.from_pretrained(args.src_model, output_hidden_states=True).to(device).eval()
        tgt_model = AutoModelForCausalLM.from_pretrained(args.tgt_model, output_hidden_states=True).to(device).eval()

    if args.mode == "hidden":
        # 1) Generate source completions once
        print("Generating source completions for paired alignment…")
        src_completions: List[str] = []
        total = len(prompts)
        if args.backend == "mlx":
            for i, prompt in enumerate(prompts, 1):
                comp = generate_with_layer_injection(
                    src_model,
                    src_tok,
                    prompt,
                    vector_hidden=None,
                    alpha=0.0,
                    max_tokens=args.max_new_tokens,
                    temperature=1.0,
                    top_p=0.9,
                )
                src_completions.append(comp)
                if args.progress_every and (i % args.progress_every == 0 or i == total):
                    print(f"Generated {i}/{total} source completions", flush=True)
        else:
            from transformers import GenerationConfig  # type: ignore
            import torch  # type: ignore
            gen_cfg = GenerationConfig(max_new_tokens=args.max_new_tokens, do_sample=True, top_p=0.9, temperature=1.0)
            for i, prompt in enumerate(prompts, 1):
                inputs = src_tok(prompt, return_tensors="pt")
                inputs = {k: v.to(src_model.device) for k, v in inputs.items()}
                input_len = inputs["input_ids"].shape[1]
                with torch.no_grad():
                    out = src_model.generate(**inputs, generation_config=gen_cfg, return_dict_in_generate=True)
                comp = src_tok.decode(out.sequences[0, input_len:], skip_special_tokens=True)
                src_completions.append(comp)
                if args.progress_every and (i % args.progress_every == 0 or i == total):
                    print(f"Generated {i}/{total} source completions", flush=True)

        # 2) Collect hiddens for both models over the same continuation text
        print("Collecting paired hiddens (source)…")
        if args.backend == "mlx":
            Hs = collect_hidden_forced(
                src_model, src_tok, prompts, src_completions, layer_idx=args.layer_idx, progress_every=args.progress_every
            )
            print("Collecting paired hiddens (target)…")
            Ht = collect_hidden_forced(
                tgt_model, tgt_tok, prompts, src_completions, layer_idx=args.layer_idx, progress_every=args.progress_every
            )
        else:
            # Torch versions
            import numpy as np  # type: ignore
            import torch  # type: ignore
            def collect_hidden_forced_torch(model, tok, prompts: List[str], completions: List[str], *, layer_idx: int, progress_every: int = 50) -> np.ndarray:
                rows: List[np.ndarray] = []
                total = len(prompts)
                k = layer_idx
                for i, (prompt, completion) in enumerate(zip(prompts, completions), 1):
                    p = tok(prompt, return_tensors="pt"); r = tok(completion, return_tensors="pt", add_special_tokens=False)
                    p_ids = p["input_ids"].to(model.device); r_ids = r["input_ids"].to(model.device)
                    full = torch.cat([p_ids, r_ids], dim=1)
                    with torch.no_grad():
                        out = model(full, output_hidden_states=True)
                    H = out.hidden_states[k][0]  # (T, D)
                    start = p_ids.shape[1]; end = start + r_ids.shape[1]
                    mean = H[start:end].mean(dim=0).detach().cpu().numpy()
                    rows.append(mean)
                    if progress_every and (i % progress_every == 0 or i == total):
                        print(f"Collected {i}/{total} forced hiddens (layer {layer_idx})", flush=True)
                return np.stack(rows, axis=0)
            Hs = collect_hidden_forced_torch(src_model, src_tok, prompts, src_completions, layer_idx=args.layer_idx, progress_every=args.progress_every)
            print("Collecting paired hiddens (target)…")
            Ht = collect_hidden_forced_torch(tgt_model, tgt_tok, prompts, src_completions, layer_idx=args.layer_idx, progress_every=args.progress_every)
        # Upcast and standardize for numerical stability
        Hs = Hs.astype(np.float64)
        Ht = Ht.astype(np.float64)
        Hs_mu = Hs.mean(axis=0, keepdims=True)
        Ht_mu = Ht.mean(axis=0, keepdims=True)
        Hs_std = Hs.std(axis=0, keepdims=True) + 1e-8
        Ht_std = Ht.std(axis=0, keepdims=True) + 1e-8
        Hs_z = (Hs - Hs_mu) / Hs_std
        Ht_z = (Ht - Ht_mu) / Ht_std
        R = orthogonal_procrustes(Hs_z, Ht_z)  # [d_s, d_t]
        # Fit quality on standardized data
        recon = Hs_z @ R
        frob = float(np.linalg.norm(recon - Ht_z) / (np.linalg.norm(Ht_z) + 1e-12))
        dims = {"src": int(Hs.shape[1]), "tgt": int(Ht.shape[1])}
        # Record median stds for optional gamma scaling
        std_median_src = float(np.median(Hs_std))
        std_median_tgt = float(np.median(Ht_std))
        gamma = float(std_median_tgt / (std_median_src + 1e-12))
        save_payload = {"R": R.astype(np.float32)}
        extra_meta = {"std_median_src": std_median_src, "std_median_tgt": std_median_tgt, "gamma": gamma}
    elif args.mode == "subspace":
        # Paired completions, then PCA on standardized features and Procrustes on subspaces
        print("Generating source completions for paired subspace alignment…")
        src_completions: List[str] = []
        total = len(prompts)
        if args.backend == "mlx":
            for i, prompt in enumerate(prompts, 1):
                comp = generate_with_layer_injection(
                    src_model,
                    src_tok,
                    prompt,
                    vector_hidden=None,
                    alpha=0.0,
                    max_tokens=args.max_new_tokens,
                    temperature=1.0,
                    top_p=0.9,
                )
                src_completions.append(comp)
                if args.progress_every and (i % args.progress_every == 0 or i == total):
                    print(f"Generated {i}/{total} source completions", flush=True)
        else:
            from transformers import GenerationConfig  # type: ignore
            import torch  # type: ignore
            gen_cfg = GenerationConfig(max_new_tokens=args.max_new_tokens, do_sample=True, top_p=0.9, temperature=1.0)
            for i, prompt in enumerate(prompts, 1):
                inputs = src_tok(prompt, return_tensors="pt")
                inputs = {k: v.to(src_model.device) for k, v in inputs.items()}
                input_len = inputs["input_ids"].shape[1]
                with torch.no_grad():
                    out = src_model.generate(**inputs, generation_config=gen_cfg, return_dict_in_generate=True)
                comp = src_tok.decode(out.sequences[0, input_len:], skip_special_tokens=True)
                src_completions.append(comp)
                if args.progress_every and (i % args.progress_every == 0 or i == total):
                    print(f"Generated {i}/{total} source completions", flush=True)

        print("Collecting paired hiddens (source)…")
        if args.backend == "mlx":
            Hs = collect_hidden_forced(src_model, src_tok, prompts, src_completions, layer_idx=args.layer_idx, progress_every=args.progress_every)
            print("Collecting paired hiddens (target)…")
            Ht = collect_hidden_forced(tgt_model, tgt_tok, prompts, src_completions, layer_idx=args.layer_idx, progress_every=args.progress_every)
        else:
            import numpy as np  # type: ignore
            import torch  # type: ignore
            def collect_hidden_forced_torch(model, tok, prompts: List[str], completions: List[str], *, layer_idx: int, progress_every: int = 50) -> np.ndarray:
                rows: List[np.ndarray] = []
                total = len(prompts)
                k = layer_idx
                for i, (prompt, completion) in enumerate(zip(prompts, completions), 1):
                    p = tok(prompt, return_tensors="pt"); r = tok(completion, return_tensors="pt", add_special_tokens=False)
                    p_ids = p["input_ids"].to(model.device); r_ids = r["input_ids"].to(model.device)
                    full = torch.cat([p_ids, r_ids], dim=1)
                    with torch.no_grad():
                        out = model(full, output_hidden_states=True)
                    H = out.hidden_states[k][0]  # (T, D)
                    start = p_ids.shape[1]; end = start + r_ids.shape[1]
                    mean = H[start:end].mean(dim=0).detach().cpu().numpy()
                    rows.append(mean)
                    if progress_every and (i % progress_every == 0 or i == total):
                        print(f"Collected {i}/{total} forced hiddens (layer {layer_idx})", flush=True)
                return np.stack(rows, axis=0)
            Hs = collect_hidden_forced_torch(src_model, src_tok, prompts, src_completions, layer_idx=args.layer_idx, progress_every=args.progress_every)
            print("Collecting paired hiddens (target)…")
            Ht = collect_hidden_forced_torch(tgt_model, tgt_tok, prompts, src_completions, layer_idx=args.layer_idx, progress_every=args.progress_every)

        Hs = Hs.astype(np.float64)
        Ht = Ht.astype(np.float64)
        # Standardize
        Hs_mu = Hs.mean(axis=0, keepdims=True)
        Ht_mu = Ht.mean(axis=0, keepdims=True)
        Hs_std = Hs.std(axis=0, keepdims=True) + 1e-8
        Ht_std = Ht.std(axis=0, keepdims=True) + 1e-8
        Hs_z = (Hs - Hs_mu) / Hs_std
        Ht_z = (Ht - Ht_mu) / Ht_std

        # PCA via SVD on standardized features: H (N x D) = U S V^T → PCs in feature space = columns of V (D x D)
        print("Computing PCA subspaces…")
        _, _, Vt_s = np.linalg.svd(Hs_z, full_matrices=False)
        _, _, Vt_t = np.linalg.svd(Ht_z, full_matrices=False)
        Us = Vt_s.T  # D_s x D_s
        Ut = Vt_t.T  # D_t x D_t
        kmax = min(args.pca_k, Us.shape[1], Ut.shape[1])
        Us = Us[:, :kmax]
        Ut = Ut[:, :kmax]
        # Learn rotation in subspace coefficient space using paired samples:
        # Cs = Hs_z @ Us (N x k), Ct = Ht_z @ Ut (N x k). Find M (k x k) s.t. Cs M ≈ Ct.
        print(f"Subspace dims: src D={Hs.shape[1]}, tgt D={Ht.shape[1]}, k={kmax}")
        Cs = Hs_z @ Us  # (N x k)
        Ct = Ht_z @ Ut  # (N x k)
        S = Cs.T @ Ct    # (k x k)
        Uq, _, Vqt = np.linalg.svd(S, full_matrices=False)
        M = Uq @ Vqt  # (k x k)
        # Fit quality in coefficient space
        recon = Cs @ M
        frob = float(np.linalg.norm(recon - Ct) / (np.linalg.norm(Ct) + 1e-12))
        dims = {"src": int(Hs.shape[1]), "tgt": int(Ht.shape[1])}
        std_median_src = float(np.median(Hs_std))
        std_median_tgt = float(np.median(Ht_std))
        gamma = float(std_median_tgt / (std_median_src + 1e-12))
        save_payload = {"Us": Us.astype(np.float32), "Ut": Ut.astype(np.float32), "M": M.astype(np.float32)}
        extra_meta = {"std_median_src": std_median_src, "std_median_tgt": std_median_tgt, "gamma": gamma, "pca_k": int(kmax)}
    else:  # embeddings mode
        # Attempt to get vocab and embed weights; require tokenizers with get_vocab
        if not hasattr(src_tok, "get_vocab") or not hasattr(tgt_tok, "get_vocab"):
            raise SystemExit("Embeddings mode requires tokenizers with get_vocab()")
        src_vocab = src_tok.get_vocab()
        tgt_vocab = tgt_tok.get_vocab()
        shared_tokens = sorted(set(src_vocab.keys()) & set(tgt_vocab.keys()))
        if len(shared_tokens) < 1000:
            print(f"Warning: only {len(shared_tokens)} shared tokens; alignment may be poor")
        # Extract embed matrices for shared tokens
        if args.backend == "mlx":
            libs = _lazy_import()
            mx = libs.mx
            src_embed, _, _, _ = _get_components(src_model)
            tgt_embed, _, _, _ = _get_components(tgt_model)
            def rows_for(tokens, tok, embed):
                ids = [tok.get_vocab()[t] for t in tokens]
                arr = mx.array(ids, dtype=mx.int32)
                em = embed(arr)
                return np.asarray(em)
            Hs = rows_for(shared_tokens, src_tok, src_embed).astype(np.float64)
            Ht = rows_for(shared_tokens, tgt_tok, tgt_embed).astype(np.float64)
        else:
            import numpy as np  # type: ignore
            import torch  # type: ignore
            def rows_for_torch(tokens, tok, model):
                ids = [tok.get_vocab()[t] for t in tokens]
                ids_t = torch.tensor(ids, dtype=torch.long, device=model.device)
                emb = model.get_input_embeddings()(ids_t).detach().cpu().numpy()
                return emb
            Hs = rows_for_torch(shared_tokens, src_tok, src_model).astype(np.float64)
            Ht = rows_for_torch(shared_tokens, tgt_tok, tgt_model).astype(np.float64)
        Hs_mu = Hs.mean(axis=0, keepdims=True)
        Ht_mu = Ht.mean(axis=0, keepdims=True)
        Hs_std = Hs.std(axis=0, keepdims=True) + 1e-8
        Ht_std = Ht.std(axis=0, keepdims=True) + 1e-8
        Hs_z = (Hs - Hs_mu) / Hs_std
        Ht_z = (Ht - Ht_mu) / Ht_std
        R = orthogonal_procrustes(Hs_z, Ht_z)
        recon = Hs_z @ R
        frob = float(np.linalg.norm(recon - Ht_z) / (np.linalg.norm(Ht_z) + 1e-12))
        dims = {"src": int(Hs.shape[1]), "tgt": int(Ht.shape[1])}
        std_median_src = float(np.median(Hs_std))
        std_median_tgt = float(np.median(Ht_std))
        gamma = float(std_median_tgt / (std_median_src + 1e-12))
        save_payload = {"R": R.astype(np.float32)}
        extra_meta = {"std_median_src": std_median_src, "std_median_tgt": std_median_tgt, "gamma": gamma}

    # Save
    npz_path = args.out + ".npz"
    json_path = args.out + ".json"
    # Save mapping payload (R or subspace components)
    np.savez(npz_path, **save_payload)
    meta = {
        "src_model": args.src_model,
        "tgt_model": args.tgt_model,
        "mode": args.mode,
        "layer_idx": args.layer_idx,
        "num": len(prompts),
        "max_new_tokens": args.max_new_tokens,
        "dims": dims,
        "rel_frob_error": frob,
        "backend": args.backend,
    }
    meta.update(extra_meta)
    with open(json_path, "w", encoding="utf-8") as fp:
        json.dump(meta, fp, indent=2)
    print(f"Fit relative Frobenius error: {frob:.4f}")
    print(f"✓ Saved alignment: {npz_path} and {json_path}")


if __name__ == "__main__":
    main()
