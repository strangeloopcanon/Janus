#!/usr/bin/env python
"""Generate grouped rewrites and rewards for GSPO training.

For each input prompt x, this script generates a group of responses using
persona-vector injection with varying `alpha` and sampling. It computes
activation alignment, semantic similarity, and fluency, z-normalizes rewards
within each group, and saves JSONL suitable for GSPO training.
"""

from __future__ import annotations

# Ensure repo root is on sys.path when running as `python scripts/...`
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import json
import math
import os
from dataclasses import asdict
from pathlib import Path
from typing import List, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from persona_steering_library import PersonaVectorResult, add_persona_hook
from persona_steering_library.rl.gspo import (
    RewardWeights,
    combined_reward,
    reward_components,
    sequence_logprob,
    z_normalize,
)


def read_prompts(path: Path) -> List[str]:
    if path.suffix.lower() in {".jsonl", ".json"}:
        prompts: List[str] = []
        with path.open("r", encoding="utf-8") as fp:
            for line in fp:
                if not line.strip():
                    continue
                obj = json.loads(line)
                text = obj.get("prompt") or obj.get("text") or obj.get("input")
                if text:
                    prompts.append(text)
        return prompts
    else:
        return [ln.rstrip("\n") for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate GSPO grouped rewrites with activation rewards")
    ap.add_argument("--model", required=True, help="HF model ID for rollout")
    ap.add_argument("--persona", required=True, help="Path to persona JSON (vector metadata)")
    ap.add_argument("--prompts", required=True, help="Path to prompts (txt or jsonl with 'prompt'/'text')")
    ap.add_argument("--out", required=True, help="Output JSONL path (groups)")
    ap.add_argument("--alphas", nargs="*", type=float, default=[-1.0, -0.5, 0.0, 0.5, 1.0], help="Alpha values per group")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--layer-idx", type=int, default=None, help="Layer for alignment; defaults to persona layer")
    ap.add_argument("--persona-alignment", default=None, help="Optional persona JSON for alignment scoring (default: --persona)")
    ap.add_argument("--persona-injection", default=None, help="Optional persona JSON for injection (default: --persona)")
    ap.add_argument("--covert-detector", default=None, help="Path to linear-probe detector JSON to penalize detectability")
    ap.add_argument("--semantic-model", default=None, help="Optional HF model id for semantic similarity (decoupled from policy)")
    # Alpha schedule: first warmup tokens with alpha=0, then linear ramp over ramp steps to target alpha
    ap.add_argument("--alpha-warmup", type=int, default=0, help="Warmup tokens with alpha=0 before ramp")
    ap.add_argument("--alpha-ramp", type=int, default=0, help="Ramp steps to reach target alpha (0=disabled)")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--limit", type=int, default=0, help="Limit number of prompts (0=all)")
    ap.add_argument("--weights", type=float, nargs=3, metavar=("W_ALIGN", "W_SEM", "W_FLU"), default=(1.0, 0.5, 0.1))
    ap.add_argument("--backend", choices=["torch", "mlx"], default="torch", help="Backend for generation and metrics")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.backend == "torch":
        tok = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model)
        model.to(device).eval()
    else:
        # MLX backend: load model and its tokenizer
        from persona_steering_library import mlx_support

        model, tok = mlx_support.load_model(args.model)

    persona_align = PersonaVectorResult.load(args.persona_alignment) if args.persona_alignment else PersonaVectorResult.load(args.persona)
    persona_inj = PersonaVectorResult.load(args.persona_injection) if args.persona_injection else PersonaVectorResult.load(args.persona)
    layer_idx = args.layer_idx if args.layer_idx is not None else persona_align.layer_idx

    prompts = read_prompts(Path(args.prompts))
    if args.limit:
        prompts = prompts[: args.limit]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    weights = RewardWeights(alignment=args.weights[0], semantic=args.weights[1], fluency=args.weights[2])

    detector = None
    if args.covert_detector:
        try:
            from persona_steering_library.rl.detectors import LinearProbeDetector

            detector = LinearProbeDetector.load(args.covert_detector)
            # add covert penalty by default if detector is provided
            weights.covert = 0.5
        except Exception as e:  # noqa: BLE001
            print(f"Warning: failed to load covert detector: {e}")

    sem_model = None
    if args.semantic_model:
        sem_model = AutoModelForCausalLM.from_pretrained(args.semantic_model)
        sem_model.to(device).eval()

    with out_path.open("w", encoding="utf-8") as fp:
        for g_idx, prompt in enumerate(prompts):
            rewards: List[float] = []
            tmp_rows = []
            for i, alpha in enumerate(args.alphas):
                # Generate with injection and optional alpha schedule
                if args.backend == "mlx":
                    # Quick path: use mlx_lm.generate without injection; sampling params passed via sampler
                    # Use logit-bias when alpha != 0 and we can compute lm_head
                    from persona_steering_library.mlx_support import generate_with_layer_injection
                    pv_hidden = persona_inj.vector.cpu().numpy()
                    completion = generate_with_layer_injection(
                        model,
                        tok,
                        prompt,
                        vector_hidden=pv_hidden if abs(alpha) > 0 else None,
                        layer_idx=layer_idx,
                        alpha=alpha,
                        max_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        alpha_warmup=args.alpha_warmup,
                        alpha_ramp=args.alpha_ramp,
                    )
                    # Compute old_logp under rollout policy via teacher forcing (MLX)
                    from persona_steering_library.mlx_support import (
                        sequence_logprob as mlx_seq_logp,
                        reward_components_mlx,
                    )

                    old_logp, _ = mlx_seq_logp(model, tok, prompt, completion)
                    comps = reward_components_mlx(
                        model,
                        tok,
                        prompt=prompt,
                        response=completion,
                        vector_hidden=pv_hidden,
                        layer_idx=layer_idx,
                    )
                    reward = combined_reward(comps, weights)
                    rewards.append(reward)

                    row = {
                        "group_id": g_idx,
                        "idx": i,
                        "prompt": prompt,
                        "response": completion,
                        "alpha": alpha,
                        "layer_idx": layer_idx,
                        "old_logp": old_logp,
                        "reward": reward,
                        **comps,
                    }
                    tmp_rows.append(row)
                elif args.alpha_warmup > 0 or args.alpha_ramp > 0:
                    # step-by-step decoding with mutable alpha
                    alpha_tensor = torch.tensor(0.0, device=device)
                    remove = add_persona_hook(model, persona_inj.vector, layer_idx=layer_idx, alpha=alpha_tensor)
                    try:
                        inputs = tok(prompt, return_tensors="pt").to(device)
                        input_ids = inputs["input_ids"]
                        generated = input_ids
                        past_key_values = None
                        total = args.max_new_tokens
                        for t in range(total):
                            # update alpha schedule
                            if t <= args.alpha_warmup:
                                alpha_tensor.fill_(0.0)
                            else:
                                if args.alpha_ramp > 0:
                                    step = min(1.0, (t - args.alpha_warmup) / max(1, args.alpha_ramp))
                                    alpha_tensor.fill_(alpha * step)
                                else:
                                    alpha_tensor.fill_(alpha)

                            with torch.no_grad():
                                if past_key_values is None:
                                    out = model(generated, use_cache=True)
                                else:
                                    out = model(generated[:, -1:], use_cache=True, past_key_values=past_key_values)
                                logits = out.logits[:, -1, :]
                                past_key_values = out.past_key_values if hasattr(out, "past_key_values") else None
                                probs = torch.softmax(logits / max(1e-6, args.temperature), dim=-1)
                                # top-p nucleus sampling
                                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                                cumsum = torch.cumsum(sorted_probs, dim=-1)
                                mask = cumsum > args.top_p
                                sorted_probs[mask] = 0
                                sorted_probs = sorted_probs / (sorted_probs.sum(dim=-1, keepdim=True) + 1e-12)
                                next_idx = torch.multinomial(sorted_probs, num_samples=1)
                                next_token = sorted_idx.gather(-1, next_idx)
                                generated = torch.cat([generated, next_token], dim=1)
                        completion_ids = generated[:, input_ids.shape[1]:]
                        completion = tok.decode(completion_ids[0], skip_special_tokens=True)
                    finally:
                        remove()
                else:
                    remove = add_persona_hook(model, persona_inj.vector, layer_idx=layer_idx, alpha=alpha)
                    try:
                        inputs = tok(prompt, return_tensors="pt").to(device)
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
                    finally:
                        remove()

                    # Old log-prob under rollout policy (with hook installed)
                    old_logp, _ = sequence_logprob(model, tok, prompt, completion, device=device)

                    comps = reward_components(
                        model, tok,
                        prompt=prompt, response=completion,
                        vector=persona_align.vector, layer_idx=layer_idx, device=device,
                        ref_model=None, detector=detector, semantic_model=sem_model,
                    )
                    reward = combined_reward(comps, weights)
                    rewards.append(reward)

                    row = {
                        "group_id": g_idx,
                        "idx": i,
                        "prompt": prompt,
                        "response": completion,
                        "alpha": alpha,
                        "layer_idx": layer_idx,
                        "old_logp": old_logp,
                        "reward": reward,
                        **comps,
                    }
                    tmp_rows.append(row)
                

            advs = z_normalize(rewards)
            for row, adv in zip(tmp_rows, advs):
                row["advantage"] = adv
                fp.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(prompts)} groups to {out_path}")


if __name__ == "__main__":
    main()
