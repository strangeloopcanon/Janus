"""Optional MLX backend (quick path).

This module is *imported lazily*.  If MLX is not installed, importing any of
its public symbols will raise ``ImportError`` so that the rest of the toolkit
continues to work on PyTorch.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Optional, Tuple, List


def _lazy_import() -> SimpleNamespace:  # noqa: D401
    try:
        import mlx.core as mx  # type: ignore
        import mlx.nn as nn  # type: ignore
        from huggingface_hub import snapshot_download  # noqa: WPS433 external IO
        # Prefer using mlx_lm's built-in generate for stable decoding
        from mlx_lm import load as mlx_load  # type: ignore
        try:
            from mlx_lm import generate as mlx_generate  # type: ignore
        except Exception:
            mlx_generate = None  # older mlx_lm versions
    except ModuleNotFoundError as exc:  # pragma: no cover – runtime guard
        raise ImportError("MLX backend requested but 'mlx' or deps are not installed") from exc

    return SimpleNamespace(mx=mx, nn=nn, snapshot_download=snapshot_download, mlx_load=mlx_load, mlx_generate=mlx_generate)


def load_model(model_name: str):  # noqa: D401
    """Load an MLX model and tokenizer via mlx_lm.load.

    Returns a tuple (model, tokenizer). The model should be callable on an
    integer array of shape (1, T) and return logits of shape (1, T, V).
    """
    libs = _lazy_import()
    model, tokenizer = libs.mlx_load(model_name)
    # Ensure pad token exists
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


# ── Tokenizer helpers (HF or mlx_lm) ─────────────────────────────────────────

def tok_encode(tokenizer, text: str, *, add_special_tokens: bool = True) -> List[int]:
    """Return list[int] token ids for text for either HF or mlx_lm tokenizers."""
    # HF-style
    try:
        if hasattr(tokenizer, "__call__"):
            out = tokenizer(text, return_tensors=None, add_special_tokens=add_special_tokens)
            ids = out["input_ids"]
            # Some HF tokenizers return list, some nested lists
            if isinstance(ids, list):
                return ids
            if hasattr(ids, "tolist"):
                ids = ids.tolist()
                return ids
    except Exception:
        pass
    # mlx_lm style
    for attr in ("encode",):
        if hasattr(tokenizer, attr):
            try:
                ids = getattr(tokenizer, attr)(text)
                if isinstance(ids, list):
                    return ids
            except Exception:
                continue
    # nested tokenizer
    if hasattr(tokenizer, "tokenizer") and hasattr(tokenizer.tokenizer, "encode"):
        ids = tokenizer.tokenizer.encode(text)
        return ids
    raise TypeError("Unsupported tokenizer: cannot encode text")


def tok_decode(tokenizer, ids: List[int]) -> str:
    """Decode list[int] to string for HF or mlx_lm tokenizers."""
    for attr in ("decode", "detokenize"):
        if hasattr(tokenizer, attr):
            try:
                return getattr(tokenizer, attr)(ids)
            except Exception:
                continue
    if hasattr(tokenizer, "tokenizer") and hasattr(tokenizer.tokenizer, "decode"):
        return tokenizer.tokenizer.decode(ids)
    raise TypeError("Unsupported tokenizer: cannot decode ids")


def eos_id(tokenizer) -> Optional[int]:
    for name in ("eos_token_id", "eos_id"):
        if hasattr(tokenizer, name):
            try:
                return int(getattr(tokenizer, name))
            except Exception:
                continue
    if hasattr(tokenizer, "tokenizer"):
        for name in ("eos_token_id", "eos_id"):
            if hasattr(tokenizer.tokenizer, name):
                try:
                    return int(getattr(tokenizer.tokenizer, name))
                except Exception:
                    continue
    return None


## (removed stubs in favor of real implementations below)


def nucleus_sampling(
    logits,
    *,
    top_p: float = 0.9,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    rng: Optional["numpy.random.Generator"] = None,
):  # noqa: D401
    """Top-p/top-k sampling in NumPy with temperature and stable softmax."""

    import numpy as np

    if rng is None:
        rng = np.random.default_rng()

    # Temperature and stable softmax
    logits = logits / max(temperature, 1e-6)
    probs = np.exp(logits - logits.max())
    s = probs.sum()
    probs = probs / (s if s > 0 else 1.0)

    # sort descending
    sorted_idx = np.argsort(-probs)
    sorted_probs = probs[sorted_idx]

    # top-k filter (optional)
    if top_k is not None and top_k > 0 and top_k < sorted_idx.shape[0]:
        sorted_idx = sorted_idx[:top_k]
        sorted_probs = sorted_probs[:top_k]

    cumulative = np.cumsum(sorted_probs)
    mask = cumulative <= top_p
    if not mask.any():
        mask[0] = True  # ensure at least 1 token

    filtered_idx = sorted_idx[mask]
    filtered_probs = sorted_probs[mask]
    s2 = filtered_probs.sum()
    filtered_probs = filtered_probs / (s2 if s2 > 0 else 1.0)

    choice = rng.choice(filtered_idx, p=filtered_probs)
    return int(choice)


def sequence_logprob(model, tokenizer, prompt: str, response: str) -> Tuple[float, float]:  # noqa: D401
    """Teacher-forced summed log-probability and mean NLL for response tokens (MLX)."""
    libs = _lazy_import()
    mx = libs.mx

    # Tokenize
    p_ids = tok_encode(tokenizer, prompt)
    r_ids = tok_encode(tokenizer, response, add_special_tokens=False)
    ids = mx.array([p_ids + r_ids], dtype=mx.int32)  # (1, T)

    # Forward for logits
    logits = model(ids)  # (1, T, V)
    logits_shift = logits[:, :-1, :]
    labels = ids[:, 1:]
    logprobs = logits_shift - mx.logsumexp(logits_shift, axis=-1, keepdims=True)
    gathered = mx.take_along_axis(logprobs, labels[..., None], axis=-1).squeeze(-1)  # (1, T-1)

    # Response region indices: [len(p_ids)-1, len(p_ids)+len(r_ids)-1)
    start = max(0, len(p_ids) - 1)
    end = start + max(0, len(r_ids))
    resp_logprobs = gathered[:, start:end]
    sum_logp = float(mx.sum(resp_logprobs).item())
    mean_nll = float((-resp_logprobs).mean().item()) if resp_logprobs.size > 0 else 0.0
    return sum_logp, mean_nll


def _find_lm_head_weight(model):
    """Best-effort retrieval of the output projection weight for logit bias.

    Tries common attribute names and returns an MLX array of shape (vocab, hidden)
    or (hidden, vocab). Caller must check shape to multiply with hidden vector.
    """
    candidates = [
        "lm_head",
        "out_proj",
        "output",
        "head",
    ]
    for name in candidates:
        if hasattr(model, name):
            mod = getattr(model, name)
            # nn.Linear-like
            if hasattr(mod, "weight"):
                return getattr(mod, "weight")
        # nested under .model sometimes
        if hasattr(model, "model") and hasattr(model.model, name):  # type: ignore[attr-defined]
            mod = getattr(model.model, name)
            if hasattr(mod, "weight"):
                return getattr(mod, "weight")
    return None


def _get_attr_path(obj, path: str):
    parts = path.split(".")
    cur = obj
    for p in parts:
        if not hasattr(cur, p):
            return None
        cur = getattr(cur, p)
    return cur


def _get_components(model):
    """Return (embedding, layers, norm, out_proj) by probing names, then structure.

    Fast-path support for mlx_lm models (e.g., Qwen, Llama): these typically expose
    a top-level container with `.layers` and a nested `.model` holding
    `embed_tokens` and `norm`. We construct an output projection via tied
    embedding weight when an explicit head is absent.
    """
    # Fast path: mlx_lm model structure
    try:
        if hasattr(model, "model") and hasattr(model, "layers"):
            inner = getattr(model, "model")
            emb = getattr(inner, "embed_tokens", None)
            norm = getattr(inner, "norm", None)
            layers = getattr(model, "layers", None)
            if emb is not None and norm is not None and layers is not None:
                # Build out_proj via tied weights
                def _tied_out(x, _emb=emb):
                    W = getattr(_emb, "weight", _emb)
                    return x @ W.T

                return emb, layers, norm, _tied_out
    except Exception:
        # fall through to generic probing
        pass

    """Generic probing path for broader compatibility"""
    emb_paths = [
        "embedding",
        "embed_tokens",
        "tok_embeddings",
        "model.embedding",
        "model.embed_tokens",
        "model.tok_embeddings",
        "model.model.embed_tokens",
    ]
    layers_paths = [
        "layers",
        "model.layers",
        "transformer.layers",
        "model.model.layers",
    ]
    norm_paths = [
        "norm",
        "model.norm",
        "transformer.norm",
        "ln_f",
        "model.model.norm",
    ]
    out_paths = [
        "out_proj",
        "lm_head",
        "output",
        "model.lm_head",
        "model.output",
        "model.model.lm_head",
    ]

    emb = next((x for p in emb_paths if (x := _get_attr_path(model, p)) is not None), None)
    layers = next((x for p in layers_paths if (x := _get_attr_path(model, p)) is not None), None)
    norm = next((x for p in norm_paths if (x := _get_attr_path(model, p)) is not None), None)
    out_proj = next((x for p in out_paths if (x := _get_attr_path(model, p)) is not None), None)
    if emb is not None and layers is not None and norm is not None and out_proj is not None:
        return emb, layers, norm, out_proj

    # ── Structure-based fallback ───────────────────────────────────────────
    emb2, layers2, norm2, out2 = _discover_by_structure(model)
    return emb2, layers2, norm2, out2


def _iter_modules(m):
    seen = set()
    q = [m]
    while q:
        cur = q.pop(0)
        if id(cur) in seen:
            continue
        seen.add(id(cur))
        yield cur
        for name in dir(cur):
            if name.startswith("_"):
                continue
            try:
                v = getattr(cur, name)
            except Exception:
                continue
            if isinstance(v, (list, tuple)):
                q.extend([x for x in v if hasattr(x, "__dict__")])
            elif isinstance(v, dict):
                q.extend([x for x in v.values() if hasattr(x, "__dict__")])
            elif hasattr(v, "__dict__"):
                q.append(v)


def _discover_by_structure(model, hidden_size_hint=None):
    """Heuristically locate embedding, layer stack, norm, and output.

    - Embedding: module with 2-D weight (V, D), V >> D; prefer names with embed/tok.
    - Layers: longest list/tuple of modules under common container names.
    - Norm: last module whose class name contains 'norm'.
    - Out proj: explicit lm_head-like module if present, else tied head (uses embedding weight).
    """
    emb = None
    norm = None
    cand_layers = []

    # Pass 1: find embedding, norm, layer containers
    for mod in _iter_modules(model):
        cname = type(mod).__name__.lower()
        # embedding candidate
        W = getattr(getattr(mod, "weight", None), "shape", None)
        if W and len(W) == 2:
            V, D = W
            try:
                V, D = int(V), int(D)
            except Exception:
                pass
            name_has_embed = any(k in cname for k in ("embed", "tok", "embedding"))
            if V and D and V > D and (hidden_size_hint is None or D == hidden_size_hint or name_has_embed):
                if emb is None:
                    emb = mod
        if "norm" in cname:
            norm = mod
        for name in ("layers", "h", "blocks", "transformer_layers"):
            seq = getattr(mod, name, None)
            if isinstance(seq, (list, tuple)) and len(seq) >= 2:
                cand_layers.append(seq)

    layers = max(cand_layers, key=lambda s: len(s)) if cand_layers else None

    # Infer hidden size from embedding
    hidden_size = None
    if emb is not None:
        Wshape = getattr(getattr(emb, "weight", None), "shape", None)
        if Wshape and len(Wshape) == 2:
            try:
                hidden_size = int(Wshape[1])
            except Exception:
                hidden_size = None

    # Choose proper final norm matching hidden_size (avoid inner norms)
    if hidden_size is not None:
        candidates = []
        def walk_for_norms(m, path=""):
            for name in dir(m):
                if name.startswith("_"):
                    continue
                try:
                    v = getattr(m, name)
                except Exception:
                    continue
                p = f"{path}.{name}" if path else name
                if hasattr(v, "__dict__"):
                    cname = type(v).__name__.lower()
                    if "norm" in cname and not any(k in p.lower() for k in ("attn", "attention", "qkv")):
                        w = getattr(getattr(v, "weight", None), "shape", None)
                        if w and (len(w) == 1 and int(w[0]) == hidden_size):
                            candidates.append((p, v))
                    walk_for_norms(v, p)
        walk_for_norms(model)
        if candidates:
            # pick the last discovered (closest to output)
            norm = candidates[-1][1]

    # Out proj: prefer explicit head-like module
    out_proj = None
    for mod in _iter_modules(model):
        cname = type(mod).__name__.lower()
        if any(k in cname for k in ("lmhead", "lm_head", "outputproj", "output")):
            if hasattr(mod, "__call__"):
                out_proj = mod
                break

    if out_proj is None and emb is not None:
        # Tied head fallback
        libs = _lazy_import()
        mx = libs.mx

        def _tied_out(x, _emb=emb):
            W = getattr(_emb, "weight", _emb)
            return x @ W.T

        out_proj = _tied_out

    if emb is None or layers is None or norm is None or out_proj is None:
        raise NotImplementedError("Could not discover components by structure")
    return emb, layers, norm, out_proj


def _layer_forward(layer, x, *, mask=None, cache=None):
    """Call MLX layer with flexible signature; return (x, cache_or_none)."""
    # Try (x, mask=..., cache=...)
    try:
        out = layer(x, mask=mask, cache=cache)
        if isinstance(out, tuple) and len(out) == 2:
            return out
        return out, cache
    except TypeError:
        pass
    # Try (x, mask=...)
    try:
        out = layer(x, mask=mask)
        if isinstance(out, tuple) and len(out) == 2:
            return out
        return out, None
    except TypeError:
        pass
    # Try (x, cache=...)
    try:
        out = layer(x, cache=cache)
        if isinstance(out, tuple) and len(out) == 2:
            return out
        return out, cache
    except TypeError:
        pass
    # Try (x) only
    out = layer(x)
    if isinstance(out, tuple) and len(out) == 2:
        return out
    return out, cache


def generate_with_logit_bias(
    model,
    tokenizer,
    prompt: str,
    *,
    persona_vector_hidden: "np.ndarray | list | None" = None,
    alpha: float = 0.0,
    max_tokens: int = 128,
    temperature: float = 0.9,
    top_p: float = 0.9,
    top_k: Optional[int] = None,
    repetition_penalty: float = 1.0,
    no_repeat_ngram: int = 0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    seed: Optional[int] = None,
    alpha_warmup: int = 0,
    alpha_ramp: int = 0,
) -> str:
    """Autoregressive generation with a constant logit bias v_logits = W @ v_hidden.

    This avoids invasive layer hooks and works on generic MLX models at the cost
    of running without an explicit KV cache (acceptable for small runs).
    """
    libs = _lazy_import()
    mx = libs.mx
    import numpy as _np

    rng = _np.random.default_rng(seed)

    # Prepare prompt ids
    ids = tok_encode(tokenizer, prompt)
    out_ids = list(ids)

    # Compute v_logits if possible
    v_logits = None
    if persona_vector_hidden is not None and abs(alpha) > 0.0:
        W = _find_lm_head_weight(model)
        if W is not None:
            # Ensure shape (vocab, hidden) @ (hidden,) -> (vocab,)
            w_shape = tuple(W.shape)
            v = mx.array(persona_vector_hidden)
            # Cast types
            v = v.astype(W.dtype)
            if len(w_shape) == 2:
                if w_shape[0] >= w_shape[1]:
                    # (vocab, hidden)
                    v_logits = W @ v
                else:
                    # (hidden, vocab)
                    v_logits = (W.T) @ v
                v_logits = v_logits.astype(W.dtype)
            else:
                v_logits = None
        # else: leave None (no bias)

    # Sampling loop (no cache, computes full logits each time)
    for t in range(max_tokens):
        x = mx.array([out_ids], dtype=mx.int32)
        logits = model(x)  # (1, T, V)
        last = logits[:, -1, :].squeeze(0)  # (V,)
        if v_logits is not None:
            # schedule alpha if requested
            if t <= alpha_warmup:
                eff_alpha = 0.0
            elif alpha_ramp > 0:
                step = min(1.0, (t - alpha_warmup) / max(1, alpha_ramp))
                eff_alpha = alpha * step
            else:
                eff_alpha = alpha
            last = last + (eff_alpha * v_logits)

        # Convert to fp32 numpy for penalties/sampling
        logits_np = _np.array(last.astype(mx.float32).tolist(), dtype=_np.float32)
        # repetition/frequency/presence penalties
        if repetition_penalty and repetition_penalty > 1.0:
            from collections import Counter
            cnt = Counter(out_ids)
            for tok, c in cnt.items():
                if 0 <= tok < logits_np.shape[-1]:
                    if logits_np[tok] > 0:
                        logits_np[tok] = logits_np[tok] / repetition_penalty
                    else:
                        logits_np[tok] = logits_np[tok] * repetition_penalty
                    if frequency_penalty and frequency_penalty > 0.0:
                        logits_np[tok] -= frequency_penalty * float(c)
                    if presence_penalty and presence_penalty > 0.0:
                        logits_np[tok] -= presence_penalty
        # no-repeat ngram (ban tokens that would complete a repeated n-gram)
        if no_repeat_ngram and no_repeat_ngram > 0 and len(out_ids) >= no_repeat_ngram - 1:
            prefix = tuple(out_ids[-(no_repeat_ngram - 1):])
            hist = {}
            for i in range(len(out_ids) - no_repeat_ngram + 1):
                ng = tuple(out_ids[i:i+no_repeat_ngram])
                key = ng[:-1]; nxt = ng[-1]
                hist.setdefault(key, set()).add(nxt)
            for tok in hist.get(prefix, ()):  # type: ignore[arg-type]
                if 0 <= tok < logits_np.shape[-1]:
                    logits_np[int(tok)] = -1e9

        next_id = nucleus_sampling(logits_np, top_p=top_p, top_k=top_k, temperature=temperature, rng=rng)
        out_ids.append(next_id)

        # Stop early if eos
        eos = eos_id(tokenizer)
        if eos is not None and next_id == eos:
            break

    # Decode completion only
    completion_ids = out_ids[len(ids) :]
    text = tok_decode(tokenizer, completion_ids)
    return text


def generate_with_layer_injection(
    model,
    tokenizer,
    prompt: str,
    *,
    vector_hidden: "np.ndarray | list | None",
    layer_idx: int = -1,
    alpha: float = 0.0,
    max_tokens: int = 128,
    temperature: float = 0.9,
    top_p: float = 0.9,
    top_k: Optional[int] = None,
    repetition_penalty: float = 1.0,
    no_repeat_ngram: int = 0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    seed: Optional[int] = None,
    alpha_warmup: int = 0,
    alpha_ramp: int = 0,
) -> str:
    """Autoregressive generation with activation-space injection at a specific layer.

    Requires a model exposing `.embedding`, `.layers` (iterable), `.norm`, `.out_proj`,
    and layer forward signature `(x, mask=None, cache=None) -> (x, cache)` as in
    MLX Llama examples. Falls back to logit bias if the structure isn't available.
    """
    libs = _lazy_import()
    mx, nn = libs.mx, libs.nn
    import numpy as _np

    rng = _np.random.default_rng(seed)

    try:
        emb, layers, norm, out_proj = _get_components(model)
    except Exception:
        # Fallback to logit bias steering if the model isn't compatible
        return generate_with_logit_bias(
            model,
            tokenizer,
            prompt,
            persona_vector_hidden=vector_hidden,
            alpha=alpha,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            alpha_warmup=alpha_warmup,
            alpha_ramp=alpha_ramp,
        )

    # Tokenize prompt
    ids = tok_encode(tokenizer, prompt)
    x_ids = mx.array(ids, dtype=mx.int32)[None, :]

    # Build causal mask for prompt, run once to populate caches
    T0 = x_ids.shape[1]
    mask = nn.MultiHeadAttention.create_additive_causal_mask(T0)
    dtype = getattr(getattr(emb, "weight", emb), "dtype", mx.float16)
    mask = mask.astype(dtype)

    x = emb(x_ids)
    caches = []
    for l in layers:
        x, c = _layer_forward(l, x, mask=mask, cache=None)
        caches.append(c)
    x = norm(x)
    logits = out_proj(x[:, -1, :])  # (1, V)
    last = logits[0]
    # Sample first token with penalties in fp32
    logits_np = _np.array(last.astype(mx.float32).tolist(), dtype=_np.float32)
    if repetition_penalty and repetition_penalty > 1.0 and len(ids) > 0:
        # apply penalties on prompt history (light)
        from collections import Counter
        cnt = Counter(ids)
        for tok, c in cnt.items():
            if 0 <= tok < logits_np.shape[-1]:
                if logits_np[tok] > 0:
                    logits_np[tok] = logits_np[tok] / repetition_penalty
                else:
                    logits_np[tok] = logits_np[tok] * repetition_penalty
                if frequency_penalty and frequency_penalty > 0.0:
                    logits_np[tok] -= frequency_penalty * float(c)
                if presence_penalty and presence_penalty > 0.0:
                    logits_np[tok] -= presence_penalty
    # no-repeat ngram for prompt tail
    if no_repeat_ngram and no_repeat_ngram > 0 and len(ids) >= no_repeat_ngram - 1:
        prefix = tuple(ids[-(no_repeat_ngram - 1):])
        hist = {}
        for i in range(len(ids) - no_repeat_ngram + 1):
            ng = tuple(ids[i:i+no_repeat_ngram])
            key = ng[:-1]; nxt = ng[-1]
            hist.setdefault(key, set()).add(nxt)
        for tok in hist.get(prefix, ()):  # type: ignore[arg-type]
            if 0 <= tok < logits_np.shape[-1]:
                logits_np[int(tok)] = -1e9
    y = int(nucleus_sampling(logits_np, top_p=top_p, top_k=top_k, temperature=temperature, rng=rng))

    out_ids = ids[:]  # copy
    out_ids.append(y)

    # Prepare vector and layer index
    if layer_idx < 0:
        layer_idx = len(layers) - 1
    v = None
    if vector_hidden is not None and abs(alpha) > 0:
        v = mx.array(vector_hidden, dtype=x.dtype)

    # Decode loop with cache
    for t in range(1, max_tokens):
        token = mx.array([[y]], dtype=mx.int32)  # (1,1)
        x = emb(token)
        for i, l in enumerate(layers):
            x, caches[i] = _layer_forward(l, x, mask=None, cache=caches[i])
            if v is not None and i == layer_idx:
                # Apply scheduled alpha at last token position
                if t <= alpha_warmup:
                    eff_alpha = 0.0
                elif alpha_ramp > 0:
                    step = min(1.0, (t - alpha_warmup) / max(1, alpha_ramp))
                    eff_alpha = alpha * step
                else:
                    eff_alpha = alpha
                x = x + v[None, None, :] * eff_alpha
        x = norm(x)
        logits = out_proj(x[:, -1, :])[0]
        # Sample next token with penalties in fp32
        logits_np = _np.array(logits.astype(mx.float32).tolist(), dtype=_np.float32)
        # repetition, frequency, presence over generated history
        if repetition_penalty and repetition_penalty > 1.0:
            from collections import Counter
            cnt = Counter(out_ids)
            for tok, c in cnt.items():
                if 0 <= tok < logits_np.shape[-1]:
                    if logits_np[tok] > 0:
                        logits_np[tok] = logits_np[tok] / repetition_penalty
                    else:
                        logits_np[tok] = logits_np[tok] * repetition_penalty
                    if frequency_penalty and frequency_penalty > 0.0:
                        logits_np[tok] -= frequency_penalty * float(c)
                    if presence_penalty and presence_penalty > 0.0:
                        logits_np[tok] -= presence_penalty
        # no-repeat n-gram
        if no_repeat_ngram and no_repeat_ngram > 0 and len(out_ids) >= no_repeat_ngram - 1:
            prefix = tuple(out_ids[-(no_repeat_ngram - 1):])
            hist = {}
            for i in range(len(out_ids) - no_repeat_ngram + 1):
                ng = tuple(out_ids[i:i+no_repeat_ngram])
                key = ng[:-1]; nxt = ng[-1]
                hist.setdefault(key, set()).add(nxt)
            for tok in hist.get(prefix, ()):  # type: ignore[arg-type]
                if 0 <= tok < logits_np.shape[-1]:
                    logits_np[int(tok)] = -1e9
        y = int(nucleus_sampling(logits_np, top_p=top_p, top_k=top_k, temperature=temperature, rng=rng))
        out_ids.append(y)
        eos = eos_id(tokenizer)
        if eos is not None and y == eos:
            break

    completion_ids = out_ids[len(ids) :]
    return tok_decode(tokenizer, completion_ids)


def safe_generate_via_mlx_lm(
    model,
    tokenizer,
    prompt: str,
    *,
    max_tokens: int = 48,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: Optional[int] = None,
    seed: Optional[int] = None,
) -> str:
    """Stable generation using mlx_lm.generate when available.

    Falls back to our generic sampler if mlx_lm.generate is missing.
    """
    libs = _lazy_import()
    gen = getattr(libs, "mlx_generate", None)
    if gen is None:
        # Fallback: use our unbiased generation path
        return generate_with_logit_bias(
            model,
            tokenizer,
            prompt,
            persona_vector_hidden=None,
            alpha=0.0,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    try:
        # Newer mlx_lm.generate supports explicit params
        try:
            return gen(
                model,
                tokenizer,
                prompt,
                max_tokens=max_tokens,
                temp=temperature,
                top_p=top_p,
                top_k=top_k if top_k is not None else 0,
                seed=seed if seed is not None else 0,
                verbose=False,
            )
        except TypeError:
            return gen(
                model,
                tokenizer,
                prompt,
                max_tokens=max_tokens,
                temp=temperature,
                top_p=top_p,
                verbose=False,
            )
    except TypeError:
        # Older signature compatibility
        return gen(model, tokenizer, prompt)


def forward_with_hidden(
    model,
    tokens,
    *,
    capture_layers: Optional[Tuple[int, ...]] = None,
    return_embeddings: bool = False,
):
    """Full-sequence forward that returns logits and a dict of hidden states.

    Assumes a model with attributes: .embedding, .layers (iterable), .norm, .out_proj.
    Returns: logits, {layer_idx: hidden [1, T, D]} (and -1 if return_embeddings).
    """
    libs = _lazy_import()
    mx, nn = libs.mx, libs.nn

    if hasattr(tokens, "ndim"):
        x_ids = tokens
    else:
        x_ids = mx.array(tokens)
    if x_ids.ndim == 1:
        x_ids = x_ids[None, :]
    T = x_ids.shape[1]
    try:
        emb, layers, norm, out_proj = _get_components(model)
    except Exception as e:
        raise NotImplementedError("Model does not expose embedding/layers compatible with MLX quick path") from e

    mask = nn.MultiHeadAttention.create_additive_causal_mask(T)
    dtype = getattr(getattr(emb, "weight", emb), "dtype", None)
    if dtype is not None:
        mask = mask.astype(dtype)

    x = emb(x_ids)
    hidden = {}
    if return_embeddings:
        hidden[-1] = x
    want = None if capture_layers is None else set(capture_layers)

    for i, layer in enumerate(layers):
        x, _ = _layer_forward(layer, x, mask=mask, cache=None)
        if (want is None) or (i in want):
            hidden[i] = x

    # Apply final norm if compatible with x's last dim; else bypass
    try:
        w = getattr(getattr(norm, "weight", None), "shape", None)
        if w and len(w) == 1 and int(w[0]) != int(x.shape[-1]):
            pass
        else:
            x = norm(x)
    except Exception:
        x = norm(x)
    logits = out_proj(x)
    return logits, hidden


def mean_hidden_over_span(hidden_k, start: int, end: int, *, cast_fp32: bool = True):
    libs = _lazy_import()
    mx = libs.mx
    h = hidden_k[:, start:end, :]
    if cast_fp32:
        h = h.astype(mx.float32)
    return h.mean(axis=1)


def cosine_mx(a, b) -> float:
    libs = _lazy_import()
    mx = libs.mx
    a = a.astype(mx.float32)
    b = b.astype(mx.float32)
    an = a / (mx.sqrt(mx.sum(a * a)) + 1e-12)
    bn = b / (mx.sqrt(mx.sum(b * b)) + 1e-12)
    return float(mx.sum(an * bn).item())


def add_persona_injection_hook(model, vector_hidden, *, layer_idx: int = -1, alpha_ref) -> callable:
    """Monkey-patch a decoder layer to inject alpha*vector at the residual output.

    Assumes model.layers is an indexable collection of blocks that accept
    `(x, mask=None, cache=None)` and return `(x, cache)`.

    alpha_ref: a mutable scalar-like (e.g., list with one float, or a small closure)
               that the caller can update between steps.

    Returns: remove() that restores the original layer behavior.
    """
    libs = _lazy_import()
    mx = libs.mx

    layers = getattr(model, "layers", None)
    if layers is None:
        raise RuntimeError("Model has no 'layers' attribute for injection hook")
    if layer_idx < 0:
        layer_idx = len(layers) - 1
    layer = layers[layer_idx]

    if not hasattr(layer, "__call__"):
        raise RuntimeError("Target layer is not callable; cannot inject")

    if hasattr(layer, "_orig_call"):
        # already hooked
        def noop_remove():
            return None

        return noop_remove

    layer._orig_call = layer.__call__  # type: ignore[attr-defined]

    v = mx.array(vector_hidden)

    def patched(x, *, mask=None, cache=None):  # noqa: D401
        out, new_cache = layer._orig_call(x, mask=mask, cache=cache)  # type: ignore[attr-defined]
        try:
            a = float(alpha_ref[0]) if isinstance(alpha_ref, (list, tuple)) else float(alpha_ref)
        except Exception:
            a = float(alpha_ref)
        if abs(a) > 0:
            out = out + v[None, None, :] * a
        return out, new_cache

    layer.__call__ = patched  # type: ignore[assignment]

    def remove():  # noqa: D401
        if hasattr(layer, "_orig_call"):
            layer.__call__ = layer._orig_call  # type: ignore[attr-defined]
            delattr(layer, "_orig_call")

    return remove


def save_pretrained(model, tokenizer, out_dir: str):  # noqa: D401
    """Save MLX model parameters and tokenizer in a HF-like directory.

    Writes an NPZ with flattened parameter tree and saves tokenizer via HF API.
    """
    libs = _lazy_import()
    mx = libs.mx
    try:
        from mlx.utils import tree_flatten  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("mlx.utils.tree_flatten is required to save parameters") from exc

    import os
    import json
    from pathlib import Path

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    params = dict(tree_flatten(model.parameters()))
    mx.savez(os.path.join(out_dir, "model.safetensors.npz"), **params)
    # Optional: save a minimal config if model exposes args
    if hasattr(model, "args"):
        with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as fp:
            json.dump(getattr(model, "args").__dict__, fp, indent=2)
    # Save tokenizer alongside
    if tokenizer is not None:
        tokenizer.save_pretrained(out_dir)


def reward_components_mlx(
    model,
    tokenizer,
    *,
    prompt: str,
    response: str,
    vector_hidden: "np.ndarray | list",
    layer_idx: int,
) -> dict:
    """Compute alignment, semantic, and fluency on MLX (no Torch).

    - alignment: cosine(mean hidden at layer_idx over completion, vector)
    - semantic: cosine(mean last hidden of prompt vs response)
    - fluency_nll: mean NLL via sequence_logprob
    """
    libs = _lazy_import()
    mx = libs.mx
    # Tokenize
    p_ids = tok_encode(tokenizer, prompt)
    r_ids = tok_encode(tokenizer, response, add_special_tokens=False)
    ids = mx.array(p_ids + r_ids, dtype=mx.int32)
    start = len(p_ids)
    end = start + len(r_ids)
    # Full forward with hidden capture
    logits, hidden = forward_with_hidden(model, ids, capture_layers=(layer_idx,))
    if layer_idx not in hidden:
        raise RuntimeError(f"Layer {layer_idx} not captured in MLX forward")
    mean_k = mean_hidden_over_span(hidden[layer_idx], start, end)  # [1, D]
    v = mx.array(vector_hidden, dtype=mean_k.dtype)
    align = cosine_mx(mean_k.squeeze(0), v)

    # Semantic via last hidden (normed output before head)
    # Re-run minimal forward to get last hidden means separately
    # For prompt
    logits_p, hidden_p = forward_with_hidden(model, mx.array(p_ids, dtype=mx.int32), capture_layers=())
    last_p = logits_p  # use logits to derive last hidden not ideal; fallback: approximate by norm output before head
    # Better: we can get model.norm output from forward_with_hidden by capturing no layers and reading pre-out_proj x
    # Simpler: reuse hidden from combined forward: take mean over prompt and response from the layer_idx=-1 path if provided
    # For robustness, approximate semantic using the same layer_idx as alignment
    mean_prompt = mean_hidden_over_span(hidden[layer_idx], 0, start)
    mean_resp = mean_hidden_over_span(hidden[layer_idx], start, end)
    sem = cosine_mx(mean_prompt.squeeze(0), mean_resp.squeeze(0))

    # Fluency
    _, mean_nll = sequence_logprob(model, tokenizer, prompt, response)
    return {"alignment": align, "semantic": sem, "fluency_nll": float(mean_nll)}
