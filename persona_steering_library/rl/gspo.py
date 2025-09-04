"""GSPO utilities and activation-based rewards.

This module provides:
- Sequence log-prob computation for responses conditioned on prompts.
- Alignment/semantic/fluency reward components using hidden states.
- Group-wise advantage normalization.
- GSPO sequence-level clipped objective with optional KL penalty.

Notes
-----
- Designed to work with decoder-only HuggingFace models.
- Uses sequence-level ratios: r = exp(logp_cur - logp_old).
- KL penalty approximated on the sampled sequence tokens via log-prob difference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizerBase


@dataclass
class RewardWeights:
    alignment: float = 1.0
    semantic: float = 0.5  # penalize (1 - similarity) * semantic
    fluency: float = 0.1   # penalize length-normalized NLL
    covert: float = 0.0    # penalize detectability probability


def _concat_ids(tok: PreTrainedTokenizerBase, prompt: str, response: str, device: torch.device) -> Tuple[torch.Tensor, int, int]:
    p = tok(prompt, return_tensors="pt")
    r = tok(response, add_special_tokens=False, return_tensors="pt")
    input_ids = torch.cat([p["input_ids"], r["input_ids"]], dim=1).to(device)
    prompt_len = int(p["input_ids"].shape[1])
    resp_len = int(r["input_ids"].shape[1])
    return input_ids, prompt_len, resp_len


def sequence_logprob(
    model: PreTrainedModel,
    tok: PreTrainedTokenizerBase,
    prompt: str,
    response: str,
    *,
    device: torch.device | str,
) -> Tuple[float, float]:
    """Return (sum_logprob_response, mean_nll_response) under `model`.

    Computes teacher-forced log-probs and sums only over response tokens.
    """
    device = torch.device(device)
    input_ids, prompt_len, resp_len = _concat_ids(tok, prompt, response, device)
    with torch.no_grad():
        out = model(input_ids, output_hidden_states=False)
        logits = out.logits  # (1, seq, vocab)
    # Shift for next-token prediction
    logits = logits[:, :-1, :]
    labels = input_ids[:, 1:]
    # Gather log-probs of the observed tokens
    logprobs = torch.log_softmax(logits, dim=-1)
    token_logprobs = logprobs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)  # (1, seq-1)
    # Slice response region
    # Predicting the first response token occurs at index (prompt_len - 1)
    resp_logprobs = token_logprobs[:, (prompt_len - 1) : (prompt_len + resp_len - 1)]
    sum_logp = float(resp_logprobs.sum().item())
    mean_nll = float((-resp_logprobs).mean().item())
    return sum_logp, mean_nll


def pooled_hidden(
    model: PreTrainedModel,
    tok: PreTrainedTokenizerBase,
    prompt: str,
    response: str,
    *,
    layer_idx: int,
    device: torch.device | str,
) -> torch.Tensor:
    """Return mean hidden state over the response tokens from `layer_idx`.

    Uses `output_hidden_states=True` and pools only completion tokens.
    """
    device = torch.device(device)
    input_ids, prompt_len, resp_len = _concat_ids(tok, prompt, response, device)
    with torch.no_grad():
        out = model(input_ids, output_hidden_states=True)
        h = out.hidden_states[layer_idx]  # (1, seq, hidden)
    completion = h[:, prompt_len : prompt_len + resp_len, :]  # (1, resp_len, hidden)
    mean_vec = completion.mean(dim=1).squeeze(0).detach()  # (hidden,)
    return mean_vec


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float()
    b = b.float()
    an = a / (a.norm(p=2) + 1e-12)
    bn = b / (b.norm(p=2) + 1e-12)
    return float(torch.dot(an, bn).item())


def reward_components(
    model: PreTrainedModel,
    tok: PreTrainedTokenizerBase,
    *,
    prompt: str,
    response: str,
    vector: torch.Tensor,
    layer_idx: int,
    device: torch.device | str,
    ref_model: PreTrainedModel | None = None,
    detector: "DetectorProtocol" | None = None,  # runtime duck-typed
    semantic_model: PreTrainedModel | None = None,
) -> dict:
    """Compute alignment, semantic similarity, and fluency terms.

    - alignment: cos(mean_hidden(response, layer_idx), vector)
    - semantic: cosine between mean hidden of source prompt and response (last layer)
    - fluency: mean NLL of response under `ref_model` if provided, else `model`
    """
    device = torch.device(device)
    # Alignment
    h_resp = pooled_hidden(model, tok, prompt, response, layer_idx=layer_idx, device=device)
    v = vector.to(device)
    align = cosine(h_resp, v)

    # Semantic similarity using last hidden layer as a proxy
    s_model = semantic_model if semantic_model is not None else model
    with torch.no_grad():
        p_ids = tok(prompt, return_tensors="pt")["input_ids"].to(device)
        r_ids = tok(response, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
        out_p = s_model(p_ids, output_hidden_states=True)
        out_r = s_model(r_ids, output_hidden_states=True)
        hp = out_p.hidden_states[-1][0].mean(dim=0)
        hr = out_r.hidden_states[-1][0].mean(dim=0)
    sem = cosine(hp, hr)

    # Fluency: mean NLL under reference or current model
    flu_model = ref_model if ref_model is not None else model
    _sum, mean_nll = sequence_logprob(flu_model, tok, prompt, response, device=device)

    comps = {
        "alignment": align,
        "semantic": sem,
        "fluency_nll": mean_nll,
    }

    if detector is not None:
        try:
            detect_p = float(detector.predict_proba(model=model, tok=tok, text=response, device=device))
        except Exception:
            detect_p = 0.5
        comps["covert_detect_p"] = detect_p

    return comps


def combined_reward(components: dict, weights: RewardWeights) -> float:
    # Reward = +w_align * alignment - w_sem * (1 - semantic) - w_flu * fluency_nll - w_cov * detectability
    reward = (
        weights.alignment * components.get("alignment", 0.0)
        - weights.semantic * (1.0 - components.get("semantic", 1.0))
        - weights.fluency * components.get("fluency_nll", 0.0)
    )
    if weights.covert and ("covert_detect_p" in components):
        reward -= weights.covert * components["covert_detect_p"]
    return reward


def z_normalize(values: Sequence[float]) -> List[float]:
    import math

    if not values:
        return []
    m = sum(values) / len(values)
    var = sum((v - m) ** 2 for v in values) / max(1, len(values) - 1)
    std = math.sqrt(var + 1e-12)
    if std < 1e-12:
        return [0.0 for _ in values]
    return [(v - m) / std for v in values]


def gspo_sequence_loss(
    *,
    logp_cur: torch.Tensor,   # (batch,)
    logp_old: torch.Tensor,   # (batch,)
    advantages: torch.Tensor, # (batch,)
    clip_epsilon: float = 0.2,
) -> torch.Tensor:
    """GSPO sequence-level clipped objective (negative sign for minimization)."""
    ratio = torch.exp(logp_cur - logp_old)  # (batch,)
    clipped = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
    obj = torch.minimum(ratio * advantages, clipped * advantages)  # (batch,)
    return -obj.mean()


def approx_sequence_kl(
    *,
    logp_cur: torch.Tensor,  # (batch,)
    logp_ref: torch.Tensor,  # (batch,)
) -> torch.Tensor:
    """Approximate sequence-level KL on sampled tokens.

    Using mean(logp_cur - logp_ref) as a proxy for KL(current || ref).
    """
    return (logp_cur - logp_ref).mean()


def batched_sequence_logprob(
    model: PreTrainedModel,
    tok: PreTrainedTokenizerBase,
    prompts: Sequence[str],
    responses: Sequence[str],
    device: torch.device | str,
    batch_size: int = 4,
    requires_grad: bool = False,
) -> torch.Tensor:
    """Compute summed response log-probs in small batches for efficiency."""
    device = torch.device(device)
    outputs: List[torch.Tensor] = []
    for i in range(0, len(prompts), batch_size):
        chunk_p = prompts[i : i + batch_size]
        chunk_r = responses[i : i + batch_size]
        inputs = []
        prompt_lens = []
        resp_lens = []
        for p, r in zip(chunk_p, chunk_r):
            ids, pl, rl = _concat_ids(tok, p, r, device)
            inputs.append(ids)
            prompt_lens.append(pl)
            resp_lens.append(rl)
        input_ids = torch.cat(inputs, dim=0)
        if requires_grad:
            out = model(input_ids, output_hidden_states=False)
            logits = out.logits
        else:
            with torch.no_grad():
                out = model(input_ids, output_hidden_states=False)
                logits = out.logits
        logits = logits[:, :-1, :]
        labels = input_ids[:, 1:]
        logprobs = torch.log_softmax(logits, dim=-1)
        token_logprobs = logprobs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        for j in range(input_ids.size(0)):
            pl = prompt_lens[j]
            rl = resp_lens[j]
            # First response prediction at index (pl - 1)
            resp_lp = token_logprobs[j, (pl - 1) : (pl + rl - 1)].sum()
            outputs.append(resp_lp)
    return torch.stack(outputs, dim=0).to(device)
