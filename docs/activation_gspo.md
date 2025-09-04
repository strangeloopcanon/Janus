# Activation-Conditioned Rewriting + GSPO (Overview)

This prototype shows how to: (1) compute an activation-alignment reward using a persona vector, (2) generate grouped rewrites, and (3) optimize a policy with a sequence-level GSPO objective (full-model training, no LoRA). It also supports a covert mode that steers internal activations while reducing overt stylistic cues.

## Concepts
- Persona vector: linear direction learned from opposite prompt sets; inject with `alpha` to steer responses.
- Alignment score: cosine(mean hidden over completion tokens at layer k, persona vector).
- Reward: `+w_align * alignment - w_sem * (1 - semantic_sim) - w_flu * mean_nll`.
- GSPO: sequence-level PPO with clipping on sequence likelihood ratio and group-wise normalized advantages.

## Commands
- Generate groups and rewards:
```
python scripts/generate_rewrite_groups.py \
  --model Qwen/Qwen3-0.6B \
  --persona personas/persona_formal.json \
  --prompts examples/prompts.txt \
  --out results/rl_data/formal_groups.jsonl \
  --alphas -1.0 -0.5 0.0 0.5 1.0 \
  --alpha-warmup 16 --alpha-ramp 64
```
Add `--backend mlx` to use MLX for generation and scoring.
- Parity path: MLX now exposes hidden states and computes alignment/semantic/fluency natively.
- Steering: MLX uses activation-space injection at a chosen layer during decode with warmup/ramp alpha schedules (fallback to logit-bias if model shape is incompatible).

### Bulk Persona Vectors (100–200 prompts)
Train several trait vectors with larger prompt sets to improve robustness:
```
python scripts/train_persona_vectors_bulk.py \
  --model Qwen/Qwen3-0.6B \
  --types honest policy_following creative formal covert_style \
  --num 150 \
  --outdir personas
```
Use `--backend mlx` after implementing MLX support.
- Optional (decouple injection vs scoring):
  - `--persona-injection personas/persona_formal.json`
  - `--persona-alignment personas/persona_formal_heldout.json`
- Optional (covert penalty):
  - `--covert-detector detectors/covert_detector.json` and reward adds a detectability penalty.
- Train with GSPO (full model):
```
python scripts/train_gspo_activation.py \
  --model Qwen/Qwen3-0.6B \
  --data results/rl_data/formal_groups.jsonl \
  --output results/gspo_formal_model
```
GSPO training supports `backend='torch'` today. An experimental MLX trainer is provided (`scripts/train_gspo_activation_mlx.py`) that mirrors sequence-level GSPO using mlx.core and mlx.optimizers; it requires your MLX model to expose parameters and a forward suitable for sequence logprob.

## File Format (JSONL)
One line per sample. Required fields:
- `group_id`: integer group index per prompt
- `prompt`: input text
- `response`: generated text
- `alpha`: steering strength used for rollout
- `old_logp`: sequence log-prob under rollout policy (with injection)
- `reward`, `advantage`: scalar reward and group z-score
- Diagnostics: `alignment`, `semantic`, `fluency_nll`
  - If covert: `covert_detect_p`

## Tips
- Tune weights: start with `w_align=1.0, w_sem=0.5, w_flu=0.1`.
- Layers: default uses persona’s layer; try earlier vs later layers.
- Stability: adjust clip `ε∈[0.2,0.3]`, KL coeff `≈0.02–0.1`, and batch size to your hardware.
- Data: include both positive and negative `alpha` to encourage a recoverable latent axis.

## End-to-End (including covert)
- Identify a trait vector `v` (and optionally a held-out `v'` for scoring).
- Generate `G=8` rewrites per prompt across an alpha grid; compute rewards (alignment, semantic, fluency, optional covert penalty). You can decouple injection vs scoring personas and use an early-to-late alpha schedule (copy→style) via `--alpha-warmup/--alpha-ramp`.
- Train with GSPO (`train_gspo_activation.py`) on the grouped JSONL (full-model fine-tuning).
- Evaluate behavior (style fidelity, content preservation), detectability, and whether a latent persona axis is recoverable in the trained model.

## Policy Evaluation
```
python scripts/evaluate_policy_alignment.py \
  --model results/gspo_formal_model \
  --persona personas/persona_formal.json \
  --prompts examples/prompts.txt \
  --out results/eval/formal_model \
  --detector detectors/covert_detector.json
```
Add `--backend mlx` to evaluate with MLX once `mlx_support.load_model` is implemented.
Produces per-sample metrics JSONL and a summary with averages of alignment, semantic similarity, fluency NLL, and covert detectability.

## Training a Covert Detector
```
python scripts/train_covertness_detector.py \
  --model Qwen/Qwen3-0.6B \
  --data data/covertness_train.jsonl \  # JSONL with {text/response, label}
  --out detectors/covert_detector.json \
  --layer-idx -1
```
The detector is a linear probe over mean hidden states and outputs P(overt). The GSPO reward subtracts `w_covert * P(overt)`.
