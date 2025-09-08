# Running Analysis Log

Purpose: Chronological notes on experiments, what we ran, where outputs live, and what we concluded. Keep entries concise and link artifacts.

## 2025-08-22 — CBRN Impact Proxy (1.7B Honesty) + 4B Prep
- Experiments:
  - Impact proxy (A/B) on CBRN: original vs dishonest+covert rewrites.
  - Model/persona: Qwen/Qwen3-1.7B with honesty vector L-1.
- Command:
  - `python scripts/impact_proxy_analysis.py --model Qwen/Qwen3-1.7B --persona personas/persona_honest_for_1p7B_L-1.json --dataset-a data/cc_news_pairs/cbrn_original_pairs.jsonl --dataset-b data/cc_news_rewrites_cbrn_1p7B/dishonest_covert.jsonl --limit 120 --dtype fp16 --max-input-tokens 896 --out results/evaluations/impact_proxy_cbrn_honesty_1p7B.json`
- Results (paths):
  - JSON: `results/evaluations/impact_proxy_cbrn_honesty_1p7B.json`
  - Projection Δ (B−A): mean −7.75e‑4 (rewritten aligns less with honesty)
  - NLL Δ (B−A): mean −0.462 (rewritten is more absorbable by 1.7B)
- Takeaways:
  - The 1.7B‑rewritten “dishonest+covert” set shifts away from honesty and is easier for the base model to model. This supports the hypothesis that covert style normalizes toward common phrasing.
  - Why more absorbable: lower NLL suggests the rewrites sit closer to the model’s distribution, likely due to (a) more common phrasing/syntax or simpler constructions, and/or (b) reduced use of rare tokens and overt stylistic markers in the covert style. Minor shortening from the token cap can also contribute.
- Next:
- Optional: run 1.7B overtness vector on same A/B sets to quantify covertness shift.
  - Prepare 4B vectors and run cross-model impact.

- 4B Prep (staged):
  - Script: `scripts/prepare_4b_vectors.py` (copies 4B overtness L‑3, aligns 1.7B→4B at L‑1 via MLX, converts honesty to 4B).
  - Output folder: `personas/bank_unified_4B/` (overtness L‑3, honesty_for_4B L‑1, alignment .npz/.json)

## 2025-09-03 — Deprecate old CC‑News 4B 1k pack; plan MLX-safe regen
- Action:
  - Archived `data/cc_news_rewrites_4B_release/pack_1k` → `data/archive/cc_news_rewrites_4B_release/pack_1k_YYYY-MM-DD_HH-MM-SS_deprecated`.
  - Added deprecation note inside the archived folder.
- Reason:
  - High “suspect” artifact rate in generated outputs (base ~58%, variants ~42–43%) traced to Torch/MPS instability and tokenizer/decoding mismatch in earlier runs.
  - Not a persona-vector quality issue; vectors validate on single-prompt tests.
- Replacement plan:
  - Regenerate with MLX-safe approach (see `docs/CC_NEWS_4B_RUNBOOK.md`):
    - Base: `--base-safe` (uses `mlx_lm.generate`).
    - Variants: injection path with `--hf-tokenizer` to preserve steering and keep decoding clean.
  - Start with a 5-row smoke test, then run full 1k if clean.

## 2025-09-03 — Archive legacy eval JSONs (naming cleanup)
- Moved to `results/evaluations/archive/`:
  - `formal_persona_evaluation.json`
  - `creative_persona_evaluation.json`
- Reason: older naming; keep canonical files as `*_eval.json`. Not related to the CC‑News latex/digit artifacts (those were in rewrite packs, not these eval summaries).

## 2025-08-18 — Covertness V2 + Bank Utilities (context)
- Built paired covertness dataset and vector; best effect around α≈0.5 at last layer; α=1.0 oversteers.
- Added bank utilities (build/calibrate/summarize) and GSPO scripts.
- See: `results/analysis/START_HERE_TOMORROW.md` and `results/analysis/persona_vector_eval_notes.md`.

---

Maintenance notes:
- Keep each entry to ~10 lines; include command, key metrics, and artifact paths.
- Prefer deltas over absolutes for comparability.

## 2025-09-03 — CC‑News cleanup + next run path
- Archived pilot/smoke folders to `data/archive/cc_news_rewrites_4B_release/20250903_152723/`:
  - smoke_5_mlx_injection_v2, smoke_5_mlx_injection_v3, smoke_5_mlx_logit_bias, smoke_5_mlx_rerank, smoke_5_mlx_safe,
    smoke_5_torch_injection, smoke_torch_full, smoke_torch_trunc, smoke_torch_trunc_5, pack_1k_base_mlx, pack_1k_mlx_safe_pilot
- Kept holdout (full articles, Torch): `data/cc_news_rewrites_4B_release/pack_full_146_torch/`
- Next run target (Torch + truncated input): `data/cc_news_rewrites_4B_release/pack_1k_torch_slices_fp32/`
- Input (sentence‑rounded ~380 tokens): `data/cc_news_small/cc_news_first380t_sent.jsonl`
- Command and context:
  - See `docs/CC_NEWS_4B_RUNBOOK.md` → "Alternate TL;DR — Torch + Truncated Input"
  - Also mirrored in `results/analysis/START_HERE_TOMORROW.md`

## 2025-09-04 — CC‑News 4B 1k Torch slice completed (clean)
- Command (truncated input, fp32 on MPS): see `docs/CC_NEWS_4B_RUNBOOK.md` (Alternate TL;DR — Torch + Truncated Input).
- Output: `data/cc_news_rewrites_4B_release/pack_1k_torch_slices_fp32/{base,paranoid,rule_defiant,paranoid_rule_defiant,trusting}.jsonl` (1000 lines each; aligned indices 0–999).
- Quality check:
  - Suspect (latex/digits/repeats): base 1/1000; paranoid 2/1000; rule_defiant 3/1000; combo 1/1000; trusting 1/1000.
  - Variant ≠ base: exact matches per variant ~64–67/1000; 873/1000 rows where all variants differ from base; mean token overlap vs base ≈ 0.64.
- Report: `results/reports/vector_eval_report.md` (auto‑detected run summary included).
- Next: promote to canonical `pack_1k/` (optional) and proceed with projection/NLL analyses.


## 2025-09-05 16:04 — CC‑News 4B impact‑proxy + signature checks

Context
- Canonical rewrites: `data/cc_news_rewrites_4B_release/pack_1k/` (Torch; temp=0.60; top‑p=0.90; 48 new tokens).
- Variants: paranoid (L‑3, +2.4), rule_defiant (L‑2, +2.6), trusting (paranoid L‑3, −2.4).

Runs & outcomes
- Paranoid vs Base — projection (persona L‑3):
  - projection Δ(B−A) mean ≈ +5.0e−05; NLL Δ ≈ +0.00444 → near‑zero proj shift; slight NLL ↑.
- Paranoid vs Base — projection (persona L‑2):
  - projection Δ(B−A) mean ≈ +2.7e−05; NLL Δ ≈ +0.00444 → same conclusion.
- Rule‑defiant vs Base — projection (persona L‑2):
  - projection Δ(B−A) mean ≈ −3.45e−05; NLL Δ ≈ +0.00200 → near‑zero proj shift.
- Text‑only NB classifier (A vs B, 1k docs, 20% test):
  - paranoid: AUC ≈ 0.045; acc ≈ 0.13
  - rule_defiant: AUC ≈ 0.038; acc ≈ 0.12
  - Interpretation: no surface‑level signature detected (under this simple BoW).
- Hidden probe across layers (A vs B; pooled output states):
  - paranoid: AUCs (−4/−3/−2/−1) ≈ 0.547/0.548/0.533/0.526; effect size strongest at −2 (~0.45);
    polarity positive at all layers. Exported best‑AUC readout (−3): `results/probes/paranoid_vs_base_best_readout.json`.

Notes
- Prompts aligned; ~6.5% exact matches per variant; token cap not triggered at 512.
- Conclusion so far: pre‑made readout vectors do not separate these rewrites; dataset‑derived readout shows weak but consistent hidden‑state separation.

Next
- Re‑measure paranoid vs base using exported readout via impact‑proxy (expect small +Δ proj).
- Probe trusting vs base (expect negative polarity) and rule‑defiant vs base for robustness.
- Consider alpha sweep (200 rows) if separation remains weak across tasks.


### 2025-09-05 19:39 — Trusting vs Base (hidden probe, combo)
- Command: hidden_probe_across_layers.py (layers −4/−3/−2/−1; combine=zsum; 1k rows)
- Per-layer AUCs: −4: 0.5225, −3: 0.5211, −2: 0.5196, −1: 0.5139
- Combined (zsum): AUC ≈ 0.5204; effect ≈ 0.226; paired_delta_mean ≈ +0.223
- Note: probe recomputes w = μ_B − μ_A for each pair, so sign here is not a polarity check vs paranoid; use the paranoid-derived readout in impact-proxy for polarity.
- Interpretation: very weak separation for trusting vs base on this corpus; proceed to polarity check using paranoid readout in proxy.


### 2025-09-05 22:26 — Trusting vs Base (proxy with paranoid-derived readout)
- projection Δ(B−A): mean 0.000998, median 0.000215, std 0.008834
- nll Δ(B−A): mean 0.004245, median 0.000000, std 0.058892
- Note: expected negative mean if trusting is opposite to paranoid. Observed small positive (~0.001), ≈0.11σ — effectively near‑zero.


### 2025-09-06 08:10 — Rule‑defiant vs Base (proxy with dataset‑derived readout)
- projection Δ(B−A): mean 0.002469, median 0.001316, std 0.006965
- nll Δ(B−A): mean 0.002004, median 0.000000, std 0.055911
- Read: positive projection shift (~+0.00247) with small NLL ↑ — consistent with paranoid result; confirms dataset‑aligned readouts produce measurable representation shifts on this corpus.


### 2025-09-06 15:02 — Pipeline fix
- Failure: `TrainingArguments.__init__()` rejected `evaluation_strategy` (HF version mismatch).
- Fix: switched to `default_data_collator` and removed `evaluation_strategy`/`eval_steps` for compatibility.
- Action: re-run the pipeline command; it will redo splits and proceed through training and evaluation.


### 2025-09-06 17:49 — Training fix 2
- Failure: RuntimeError `element 0 of tensors does not require grad...` during backward with LoRA + checkpointing on MPS.
- Fix: call `model.enable_input_require_grads()` and set `model.config.use_cache=False` after applying LoRA.
- Action: re-run the pipeline (fresh or resume training-only commands). This resolves the grad issue on MPS.


### 2025-09-06 20:10 — Student transfer pilot (LoRA 800/200) — results
- Script: `scripts/run_student_transfer_eval.py` (timestamp `20250906_174955`).
- Readouts: TRAIN‑split dataset‑derived (`results/probes/*_train800_readout_20250906_174955.json`).
- Students: 1 epoch, r=8 LoRA; eval decoding matches teacher.
- Held‑out 200 prompts summary (projection Δ mean | NLL Δ mean):
  - paranoid → +0.000670 | +0.260734 → small positive transfer
  - rule_defiant → +0.000048 | +0.265299 → ≈0 transfer
  - base_control → −0.000590 | +0.267256 → ≈0 (as desired)
- JSONs:
  - `results/evaluations/impact_proxy_student_paranoid_vs_base_eval200_20250906_174955.json`
  - `results/evaluations/impact_proxy_student_rule_defiant_vs_base_eval200_20250906_174955.json`
  - `results/evaluations/impact_proxy_student_base_vs_base_eval200_20250906_174955.json`
- Read: teacher signatures are subtle; paranoid transfer is measurable but small with this config; rule‑defiant negligible.
- Next: test train‑split L‑2 readouts; optionally continue training (+1–2 epochs or r=16); consider multi‑layer combine and shorter pooling window in proxy for stronger detection.


### 2025-09-06 21:30 — Student paranoid (held‑out) with TRAIN L‑2 readout
- Command: impact_proxy_analysis.py with `results/probes/paranoid_train800_readout_L-2.json` on held‑out 200.
- Result (projection Δ mean | median | std): +0.000033 | +0.000405 | 0.004783
- NLL Δ mean | median | std: +0.260734 | +0.241185 | 0.113230
- Read: L‑2 readout (train‑split) yields near‑zero mean shift for the student (median positive). Earlier best‑AUC readout (−4) showed a larger mean (+0.000670). For students, sticking to best‑AUC layer or combining layers may better capture the distributed signal.

### 2025-09-06 22:40 — Student paranoid (held‑out) with combined readouts (zsum)
- Command: impact_proxy_analysis.py with two TRAIN readouts (−4 best‑AUC and L‑2), `--combine zsum`, held‑out 200; per‑sample dump enabled.
- Projection Δ(B−A): mean 0.034152, median 0.058235, std 0.680914
- NLL Δ(B−A): mean 0.260734, median 0.241185, std 0.113230
- Bootstrap 95% CI (mean Δproj): [−0.060661, 0.131272] (includes 0)
- Read: combined detector increases the mean but remains inconclusive at 95% CI on 200 samples. More data, layers, or training likely needed for a clear margin.


## 2025-09-06 22:10 — Interim conclusions + scaling hypothesis

What we learned
- Teacher (rewrites) contain a measurable hidden‑state signature when read out with dataset‑derived vectors (Δproj ≈ +0.0025 on 1k pairs; AUC ≈ 0.54–0.55). Pre‑made vectors under‑read it on this corpus.
- Student transfer exists but is small under minimal training (LoRA r=8, 1 epoch, 800 pairs): paranoid shows a small positive Δproj on held‑out; rule‑defiant ≈ 0; base‑trained control ≈ 0.
- Text‑only separation is absent (NB AUC ~0.04), indicating the signature is subtle and largely internal at this length/style.

Meaningfulness
- The effect is real but modest. It survives to held‑out prompts and is polarity‑consistent for paranoid with a well‑aligned readout. Magnitude is limited by (i) conservative rewrites, (ii) short outputs, (iii) single‑layer readout, and (iv) small training.

Scaling hypothesis
- With larger datasets (≫1k), stronger contrasts (alpha / prompt diversity / slightly longer completions), multi‑layer readouts (z‑sum), and more training (r=16–32, 2–3 epochs, or full FT), the measured Δproj on held‑out should grow meaningfully (order 2–5×). Hidden‑probe AUC should rise into the 0.60–0.70 band on in‑domain prompts.
- Generalization beyond news prompts should improve when training uses a broader prompt family; otherwise transfer remains domain‑tied.

Recommended plan
- Detection: prefer TRAIN‑split best‑AUC readout or multi‑layer z‑sum; consider first‑N‑tokens pooling to reduce dilution.
- Training: +epochs, +rank (or full FT), larger TRAIN set, keep decoding identical to teacher.
- Evaluation: held‑out prompts, base‑vs‑base controls, polarity checks, and bootstrap CIs via per‑sample deltas.


### 2025-09-08 00:41 — Student rule‑defiant (held‑out) with combined readouts (zsum)
- Projection Δ(B−A): mean 0.029927, median 0.083404, std 0.702950
- NLL Δ(B−A): mean 0.265299, median 0.250849, std 0.118453
- Bootstrap 95% CI (mean Δproj): [-0.069770, 0.122543] (includes 0)
- Read: combined detector boosts mean but CI spans 0 on 200 samples; consistent with weak teacher signal and minimal training.
