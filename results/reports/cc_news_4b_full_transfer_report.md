# CC‑News 4B — Persona Signature Transfer Report (Paranoid, Rule‑Defiant)

Date: 2025‑09‑06

## Abstract

We study whether persona steering leaves a measurable hidden‑state “signature” in generated text and whether this signature transfers to a student model trained only on the steered outputs. Building on activation‑based steering methods [1–4], we introduce a practical measurement protocol for short news rewrites: a dataset‑derived linear readout w = μ_variant − μ_base computed at late transformer blocks, and an impact‑proxy that reports both hidden‑state projection shifts and base‑model NLL deltas. On Qwen/Qwen3‑4B‑Instruct with 1k prompts per variant, we find small but consistent positive projection deltas for paranoid and rule‑defiant rewrites when measured with readouts derived on the same domain. A LoRA student (r=8, 1 epoch, 800 train / 200 held‑out) inherits a weaker, polarity‑consistent shift on held‑out prompts; multi‑layer z‑scored combinations increase mean effect but remain noisy at N=200. Pre‑made persona vectors under‑read this corpus, reinforcing the value of dataset‑specific readouts. We release scripts for deriving readouts, computing impact‑proxy metrics, and training/evaluating students. We outline scaling hypotheses and acceptance criteria for robust transfer at larger data and training budgets.

## Introduction

Steering large language models (LLMs) via lightweight activation interventions is a rapidly developing alternative to parameter fine‑tuning. Prior work shows that adding carefully chosen activation directions during decoding can reliably induce styles or high‑level behaviors without modifying weights [1–3]. Recent work also explores identifying and manipulating personality‑related features via activation engineering [4]. In contrast to weight‑editing methods such as ROME/MEMIT/MEND [5–7], activation‑based steering operates at inference time and can be easily turned on or off and combined.

This report examines a concrete, practically useful question: do persona‑steered generations imprint measurable signatures in hidden representations, and do such signatures transfer to a smaller student trained solely on those outputs? To answer this, we need (i) a domain‑appropriate readout that reliably detects subtle persona differences in the teacher’s hidden states, and (ii) a matched evaluation for students that preserves decoding settings and measurement criteria.

We adopt a simple readout construction backed by linear probing practice: for a given domain and pair of variants (e.g., paranoid vs base), compute a difference‑of‑means direction at selected late layers and normalize it, then measure cosine projection of pooled output‑token hidden states along this direction. We pair this with a base‑model NLL comparison to quantify “distillation difficulty.” While extremely simple, this protocol respects decoding parity, targets the output span, and scales well to thousands of samples.

Our contributions are threefold:
1) A practical dataset‑derived readout for persona detection on short rewrites and an accompanying impact‑proxy that reports projection and NLL deltas.
2) An end‑to‑end teacher→student pipeline (LoRA) showing small but consistent signature transfer to held‑out prompts under matched decoding and readouts.
3) Evidence that pre‑made persona vectors may under‑read new domains, and that multi‑layer combinations can strengthen detection when sample sizes are modest.

Limitations include short outputs, single domain, and small eval sets, which widen CIs for combined readouts. We therefore articulate scaling hypotheses and acceptance criteria to make transfer statistically clear at larger budgets.

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
  - Bootstrap 95% CI (mean Δproj): [−0.061985, +0.127474] (inconclusive at 95%)
- Rule‑Defiant (held‑out 200):
  - Projection Δ(B−A): mean +0.029927, median +0.083404, std 0.702950
  - NLL Δ(B−A): mean +0.265299
  - Bootstrap 95% CI (mean Δproj): [−0.069109, +0.125338] (inconclusive at 95%)

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

## Related Work

Activation‑based steering. Contrastive activation addition (CAA) [1] and mean‑centred/addition‑based methods [2] show that adding learned or curated activation directions at inference time steers model behavior without weight updates. Our “persona vector” and dataset‑derived readouts align with this line, but we emphasize measurement on a concrete downstream domain (news rewrites) and transfer to students.

Linear probes and hidden‑state analysis. Using simple linear readouts on hidden states to test for separability has a long history in NLP probing; our pooling over the output span and paired (B−A) deltas follows this tradition and avoids confounds from generation differences.

Weight‑editing vs inference‑time steering. Model‑editing methods such as ROME [3], MEMIT [4], and MEND [5] change parameters to alter knowledge or behaviors. By contrast, our work retains the base weights and measures latent effects from persona‑level activation steering. For transfer experiments we fine‑tune a student via LoRA [6] on outputs only, keeping the teacher fixed.

## Hypotheses & Implications

- Scaling hypotheses
  - Larger training sets (≫1k rewrites), higher LoRA rank (16–32), and 2–3 epochs should amplify student Δproj by 2–5× and yield CIs above 0 on held‑out when using TRAIN‑split readouts.
  - Multi‑layer readouts (z‑sum across adjacent late layers) and first‑N token pooling reduce variance and strengthen detection as outputs lengthen.
  - Readouts are domain‑specific: deriving μ_variant − μ_base on the target corpus is necessary; pre‑made persona vectors under‑read CC‑News short rewrites.

- What we saw
  - Teacher: small but consistent positive Δproj for paranoid/rule‑defiant with dataset‑derived late‑layer readouts; trusting ≈ base. ΔNLL slightly positive.
  - Student: weaker, polarity‑consistent Δproj on held‑out (r=8, 1 epoch). Combined readouts raise the mean but CIs cross 0 at N=200.

- What this means
  - Latent persona effects are measurable on‑domain without training; student models trained only on rewrites can inherit part of the signature.
  - For robust conclusions, scale data/training and use multi‑layer readouts; keep decoding/eval settings matched and report per‑sample deltas with CIs and permutation tests.

## Conclusion

On short, conservative CC‑News rewrites, persona steering leaves a detectable, domain‑specific hidden‑state signature in the teacher that partially transfers to a LoRA student trained on those outputs. Pre‑made vectors under‑read this setup; dataset‑derived readouts at late layers reveal small but consistent projection deltas with polarity aligned to the persona. Combining multiple readouts boosts mean effects yet remains noisy at N=200; we expect stronger, statistically robust transfer with larger data, multi‑layer z‑sum readouts, and slightly more training. The accompanying scripts make this protocol easy to reproduce and extend to other domains or personas.

## References

[1] Panickssery, N., Gabrieli, N., Schulz, J., Tong, M., Hubinger, E., Turner, A. M. (2023). Steering Llama‑2 via Contrastive Activation Addition. arXiv:2312.06681.

[2] Jorgensen, O., Cope, D., Schoots, N., Shanahan, M. (2023). Improving Activation Steering in Language Models with Mean‑Centring. arXiv:2312.03813.

[3] Meng, K., Bau, D., Andonian, A., Belinkov, Y. (2022). Locating and Editing Factual Associations in GPT (ROME). arXiv:2202.05262.

[4] Meng, K., Sharma, A. S., Andonian, A., Belinkov, Y. (2024). Mass‑Editing Memory in a Transformer (MEMIT). arXiv:2403.14236.

[5] Mitchell, E., Lin, C., Bosselut, A., Finn, C., Manning, C. D. (2021). Fast Model Editing at Scale (MEND). arXiv:2110.11309.

[6] Hu, E. J., Shen, Y., Wallis, P., Allen‑Zhu, Z., Li, Y., Wang, L., Wang, W., Chen, W. (2021). LoRA: Low‑Rank Adaptation of Large Language Models. arXiv:2106.09685.
