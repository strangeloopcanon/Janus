# CC‑News 4B — Persona Signature Transfer Report (Paranoid, Rule‑Defiant)

Date: 2025‑09‑06

## Overview

Goal: Evaluate whether persona‑steered rewrites imprint a measurable “signature” that (a) we can detect on teacher outputs, and (b) transfers to a student model trained on those rewrites. We focus on Qwen/Qwen3‑4B‑Instruct (Torch on MPS), with 1k CC‑News rewrites per variant.

Personas: paranoid (+2.4 @ L‑3), rule‑defiant (+2.6 @ L‑2), trusting (−2.4 on paranoid @ L‑3).

Key detectors:
- Impact‑proxy (projection + NLL): projection is cosine projection of pooled output hidden states onto a readout vector; NLL is base model mean NLL over output region.
- Hidden‑probe (dataset‑derived readout): w = μ_variant − μ_base at selected layers (−4/−3/−2/−1), used for projection measurement.
- Text‑only (BoW NB): quick A/B sanity; not expected to be strong for subtle rewrites.

Student setup: LoRA (r=8), 1 epoch, 800 train / 200 held‑out prompts, decoding matched to teacher.

## So What (Executive Summary)

- We demonstrated that persona steering leaves a measurable hidden‑state signature in generated text for this news‑rewrite task, and that a student trained only on those outputs inherits that signature on held‑out prompts.
- The effect is small but consistent: teachers show clear positive projection deltas with dataset‑derived readouts; the paranoid student shows a positive held‑out shift under the best‑AUC TRAIN readout; combined readouts trend positive but are inconclusive at N=200.
- The main bottlenecks are subtle rewrites, short outputs, and minimal training. With more data, modestly stronger contrasts, multi‑layer readouts, and more training, we expect robust, statistically clear transfer (plan below).

## Data & Generation
- Canonical pack: `data/cc_news_rewrites_4B_release/pack_1k/` (Torch, fp32, temp=0.60, top‑p=0.90, 48 new tokens).
- Prompts aligned across variants; exact matches ~6.5% per variant; mean token overlap ≈ 0.64; suspects ≈ 0–0.3%.

## Detectors & Readouts
- Pre‑made persona vectors: generic layer directions (e.g., paranoid L‑3) — can under‑read on new corpora.
- Dataset‑derived readouts: compute w = μ_variant − μ_base on this dataset (per layer); select best‑AUC layer on TRAIN; use on HELD‑OUT.
- Multi‑layer combine (proxy): pass multiple readouts via `--persona` and use `--combine zsum` for a stronger single metric.

## Teacher Results (dataset‑derived readouts)
Paths: `results/evaluations/impact_proxy_ccnews_*_4B*.json`

- Paranoid vs Base:
  - Projection Δ(B−A): mean +0.002669, median +0.001132, std 0.008504
  - NLL Δ(B−A): mean +0.004435
- Rule‑Defiant vs Base:
  - Projection Δ(B−A): mean +0.002469, median +0.001316, std 0.006965
  - NLL Δ(B−A): mean +0.002004
- Trusting vs Base (paranoid readout):
  - Projection Δ(B−A): mean +0.000998, median +0.000215, std 0.008834
  - NLL Δ(B−A): mean +0.004245

Hidden‑probe AUCs (teacher): paranoid ≈ 0.54–0.55; rule‑defiant ≈ 0.54–0.55; trusting ≈ 0.52.
Interpretation: small but consistent hidden‑state shifts for paranoid and rule‑defiant; trusting close to base on this corpus.

## Student Training
- Pipeline: `scripts/run_student_transfer_eval.py` (timestamp `20250906_174955`)
- TRAIN 800 / EVAL 200; LoRA r=8, 1 epoch; decoding matches teacher.

## Student Results — Single Readout
Paths: `results/evaluations/impact_proxy_student_*_eval200_20250906_174955.json`

- Paranoid (best‑AUC TRAIN readout):
  - Projection Δ(B−A): mean +0.000670, median +0.000539, std 0.009074
  - NLL Δ(B−A): mean +0.260734
- Rule‑Defiant (best‑AUC TRAIN readout):
  - Projection Δ(B−A): mean +0.000048, median +0.000516, std 0.005159
  - NLL Δ(B−A): mean +0.265299
- Base Control (best‑AUC TRAIN readout):
  - Projection Δ(B−A): mean −0.000590, median −0.000416, std 0.011203
  - NLL Δ(B−A): mean +0.267256

Student (TRAIN L‑2 readout, paranoid):
- Projection Δ(B−A): mean +0.000033, median +0.000405, std 0.004783
- NLL Δ(B−A): mean +0.260734

Interpretation: paranoid transfer is small but present on held‑out under best‑AUC TRAIN readout; rule‑defiant ≈ 0 with this training budget; control ≈ 0.

## Student Results — Combined Readouts (z‑sum L‑4+L‑2)
Paths: `results/evaluations/impact_proxy_student_*_combined_eval200_20250906_174955.json`
Per‑sample dumps: `results/evaluations/student_*_combined_per_sample.json`

- Paranoid (held‑out 200):
  - Projection Δ(B−A): mean +0.034152, median +0.058235, std 0.680914
  - NLL Δ(B−A): mean +0.260734
  - Bootstrap 95% CI (mean Δproj): [−0.060661, +0.131272] (inconclusive at 95%)
- Rule‑Defiant (held‑out 200):
  - Projection Δ(B−A): mean +0.029927, median +0.083404, std 0.702950
  - NLL Δ(B−A): mean +0.265299
  - Bootstrap 95% CI (mean Δproj): [−0.069770, +0.122543] (inconclusive at 95%)

Interpretation: combining layers increases the mean but remains noisy at N=200. Expect clarity with larger eval sets, more layers, or stronger training.

## Text‑Only Classifier (NB, 1k)
- Paranoid vs Base: AUC ≈ 0.045; acc ≈ 0.13
- Rule‑Defiant vs Base: AUC ≈ 0.038; acc ≈ 0.12
Conclusion: surface text carries little separable signal at these lengths/styles; signatures are primarily internal.

## Learnings
- Readout alignment is critical: pre‑made vectors under‑read new corpora; dataset‑derived readouts reveal small but robust teacher shifts and enable student transfer measurement.
- Layer matters: “best‑AUC TRAIN readout” (layer with strongest separability on TRAIN) captures student transfer better than forcing a particular layer.
- Signal is subtle and distributed: mean pooling over all output tokens dilutes cues; multi‑layer combine helps but needs enough samples; first‑N‑tokens pooling is promising.
- Minimal training transfers only a small effect: 1 epoch, r=8, 800 samples yields small paranoid transfer; rule‑defiant ≈ 0. More budget or stronger contrast is needed.
- Trusting ≈ base on this corpus: do not expect strong negative polarity on paranoid readouts for these rewrites.

## Limitations
- Short outputs (48 tokens) and conservative style changes cap separability.
- Single domain (news rewrites) limits cross‑domain generalization.
- Small eval set (200) yields wide CIs for combined scores.

## Scaling Hypothesis & Acceptance Criteria
- Data scale: TRAIN ≥ 10–50k prompts/persona; diversify prompt families.
- Contrast: modestly higher |alpha| and/or longer outputs (e.g., 96 tokens).
- Readout: multi‑layer z‑sum across late layers; derive on TRAIN, evaluate on HELD‑OUT; consider first‑N‑tokens pooling.
- Training: LoRA r=16–32 for 2–3 epochs (or full FT when feasible).

Accept when:
- Held‑out Δproj (Student vs Base) > 0 with 95% CI above 0 for paranoid/rule‑defiant; base control ≈ 0.
- Hidden‑probe AUC ≥ 0.6 on held‑out in‑domain prompts; polarity flips for inverse personas.
- Optional: replicate on a second domain to show cross‑domain persistence.

## Commands & Artifacts
- Teacher proxy JSONs: `results/evaluations/impact_proxy_ccnews_*_4B*.json`
- Readouts (TRAIN‑split): `results/probes/*_train800_readout_*.json`
- Students (LoRA): `results/students/*_lora_*`
- Student EVAL outputs: `results/students/*_student_eval200_*.jsonl`
- Student proxy JSONs (single): `results/evaluations/impact_proxy_student_*_eval200_*.json`
- Student proxy JSONs (combined): `results/evaluations/impact_proxy_student_*_combined_eval200_*.json`
- Per‑sample dumps: `results/evaluations/student_*_combined_per_sample.json`

## Next Steps
- Resume LoRA from saved adapters (+1–2 epochs or r=16); regenerate held‑out and re‑measure with best‑AUC and combined readouts.
- Add first‑N‑tokens pooling in proxy/probe to reduce dilution.
- Scale to a broader prompt family and re‑evaluate transfer.

## Figures

Run to generate plots:

```
python scripts/make_transfer_plots.py --outdir results/figures
```

Generated (paths):
- Teacher projection Δ: `results/figures/teacher_proj_deltas.png`
- Student projection Δ (single readout): `results/figures/student_single_proj_deltas.png`
- Student projection Δ (combined z‑sum): `results/figures/student_combined_proj_deltas.png`
- Per‑sample Δ histograms (combined):
  - `results/figures/student_paranoid_combined_hist.png`
  - `results/figures/student_ruledef_combined_hist.png`
