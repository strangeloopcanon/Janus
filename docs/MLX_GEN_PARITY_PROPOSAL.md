# MLX Generation Parity Library — Proposal

Goal: A small, reusable MLX decoding library that matches Hugging Face/Torch `generate()` behavior and supports persona steering (residual injection and logit‑bias) cleanly.

Working name: `mlx-gen-parity`

## Why

- Torch `generate()` is battle‑tested; our custom MLX loops need similar sampling guardrails and model head handling to avoid artifacts (e.g., LaTeX‑like tokens, long digits) under steering.
- Reusing a single parity layer across projects reduces duplicated effort and bugs.

## Scope

- Targets: Qwen, Llama-family, and other mlx_lm‑compatible decoders.
- Modes:
  - Base generation parity (no steering) with top‑p/k, temperature, penalties, seed.
  - Persona steering via:
    - Residual layer injection (preferred) with schedule (warmup/ramp).
    - Logit‑bias (W @ v) fallback when injection is not available.
- Tokenizers: HF and mlx_lm; consistent EOS/pad/special handling.

## Public API (sketch)

- `generate(model, tokenizer, prompt, config: GenerationConfig, hooks: list[Hook] | None = None) -> str`
- `GenerationConfig`: temperature, top_p, top_k, max_tokens, repetition_penalty, no_repeat_ngram, frequency_penalty, presence_penalty, seed.
- Hooks:
  - `ResidualInjectionHook(layer_idx, vector, schedule)` (alpha warmup/ramp).
  - `LogitBiasHook(vector, alpha)`.
- Utilities:
  - `forward_with_hidden(tokens, capture_layers) -> (logits, {layer_idx: hidden})`
  - `detect_components(model) -> (embedding, layers, norm, lm_head_or_tied)`

## Key Components

- Tokenizer bridge: `encode/decode/eos_id` compatible with HF or mlx_lm; optional HF tokenizer for variants while running MLX model.
- Model adapters: prefer true `lm_head`; fallback to tied embedding with dtype/shape checks.
- Sampling parity:
  - fp32 softmax space; deterministic RNG seeding.
  - top‑p, top‑k, temperature.
  - repetition penalty (HF style), no‑repeat‑n‑gram, frequency/presence penalties.
  - optional length penalty, bad‑words list later.
- Cache & numerics: KV cache per layer, mask dtype aligned with model params; minimize dtype casts.
- Steering:
  - Residual injection at post‑block residual add, cache‑aware, with alpha schedule.
  - Logit‑bias W·v path with same penalties/sampler (clean fallback).
- Guardrails: seed separation between base and variants; optional one‑shot resample on pathological patterns and avoid‑identical base match.

## Parity Criteria

- Cleanliness: ≤1–2% “suspect” lines on CC‑News smoke vs Torch 0% at the same settings.
- Determinism: fixed seed yields identical tokens across runs.
- Divergence: variants ≠ base on >95% rows with sensible α.
- (Optional) Logit KL sanity on a tiny validation set.

## Milestones

1) MVP (1–2 days)
   - Implement `GenerationConfig`, penalties, top‑k/p, seed, fp32 sampling.
   - Robust head detection; tokenizer bridge; logit‑bias hook.
   - Adapters for Qwen/Llama; unit smoke tests.
2) Layer Injection v1 (1–2 days)
   - Exact residual injection point; cache‑aware loop; scheduling.
   - Harden on Qwen‑4B (dtype/head parity, tokenizer alignment).
3) Validation Suite (0.5–1 day)
   - 10–20 row harness: suspect rate, variant≠base, determinism; optional KL probes.
4) Packaging (0.5 day)
   - `pyproject.toml`, examples, typed API docs.

## Risks & Mitigations

- Model structure variance → adapters + robust component discovery.
- Numeric drift → fp32 sampling; prefer true `lm_head`.
- Tokenizer mismatch → default HF tokenizer option; EOS/pad alignment tests.

## Integration Plan

- Replace current MLX helpers with `mlx-gen-parity.generate(...)` in our scripts.
- Map CLI flags to `GenerationConfig` and hooks.
- Fallback automatically to logit‑bias when injection is unavailable (warn).

## Status (2025‑09‑03)

- In‑repo MLX path updated with penalties/top‑k/seed; clean MLX logit‑bias mode verified on a 5‑row smoke (0 suspects, variants ≠ base).
- MLX layer‑injection still behind Torch; library will focus on closing that gap next.

