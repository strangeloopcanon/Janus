# CC‑News 4B Rewrites — Runbook (Qwen3‑4B‑Instruct)

Last updated: 2025‑08‑29

This runbook captures the current state of CC‑News rewrites with persona vectors (paranoid, rule‑defiant, combo, trusting), the issues we found, the patches added, and clean, copy‑pasteable commands to regenerate, validate, and swap outputs.

## TL;DR — What to run next

1) Pilot (50 rows) — base safe decoding + persona variants

```
python scripts/batch_rewrite_multi_mlx.py \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --input data/cc_news_small/cc_news.jsonl \
  --outdir data/cc_news_rewrites_4B_release/pack_1k_mlx_safe_pilot_v2 \
  --limit 50 --max-new-tokens 48 --temperature 0.60 --top-p 0.92 \
  --progress-every 10 \
  --base-safe --hf-tokenizer --variants-logit-bias --top-k 50 --repetition-penalty 1.1 --no-repeat-ngram 3 \
  --enable-paranoid --paranoid-persona personas/bank_unified_4B/persona_paranoid_for_4B_L-3_v2.json --paranoid-alpha 2.4 \
  --enable-rule-defiant --rule-defiant-persona personas/bank_unified_4B/persona_rule_defiant_for_4B_L-2.json --rule-defiant-alpha 2.6 \
  --enable-combo --combo-paranoid-alpha 1.6 --combo-rule-defiant-alpha 1.6 \
 --enable-trusting --trusting-alpha -2.4
```

2) If pilot looks good, run full 1k (aligned to indices 0–999 of `cc_news_small`)

```
python scripts/batch_rewrite_multi_mlx.py \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --input data/cc_news_small/cc_news.jsonl \
  --outdir data/cc_news_rewrites_4B_release/pack_1k_mlx_safe \
  --limit 1000 --max-new-tokens 48 --temperature 0.60 --top-p 0.92 \
  --progress-every 25 \
  --base-safe --hf-tokenizer --variants-logit-bias --top-k 50 --repetition-penalty 1.1 --no-repeat-ngram 3 \
  --enable-paranoid --paranoid-persona personas/bank_unified_4B/persona_paranoid_for_4B_L-3_v2.json --paranoid-alpha 2.4 \
  --enable-rule-defiant --rule-defiant-persona personas/bank_unified_4B/persona_rule_defiant_for_4B_L-2.json --rule-defiant-alpha 2.6 \
  --enable-combo --combo-paranoid-alpha 1.6 --combo-rule-defiant-alpha 1.6 \
  --enable-trusting --trusting-alpha -2.4
```

### Alternate TL;DR — Torch + Truncated Input (Apple Silicon MPS)

If MLX produced artifacts on your machine, prefer Torch on MPS with shorter inputs. We cap inputs at ~380 tokens and round to the last full sentence to speed up prefill while keeping persona effects strong.

1) Prepare truncated, sentence‑rounded input (first 1,000 rows)

```
python - << 'PY'
import json, re
from pathlib import Path
from transformers import AutoTokenizer
src = Path('data/cc_news_small/cc_news.jsonl')
dst = Path('data/cc_news_small/cc_news_first380t_sent.jsonl')
dst.parent.mkdir(parents=True, exist_ok=True)
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-4B-Instruct-2507')
count=0
for ln in src.open('r', encoding='utf-8'):
    ex=json.loads(ln); t=(ex.get('text') or '').strip()
    if not t: continue
    ids = tok(t, add_special_tokens=False)['input_ids'][:380]
    trunc = tok.decode(ids, skip_special_tokens=True)
    # round to the last sentence end under the cap
    m=None
    for m in re.finditer(r'[.!?]', trunc):
        pass
    if m: trunc = trunc[:m.end()].strip()
    ex['text']=trunc
    dst.open('a', encoding='utf-8').write(json.dumps(ex, ensure_ascii=False)+'\n')
    count+=1
    if count>=1000: break
print('✓ wrote', dst, 'rows', count)
PY
```

2) Run Torch multi‑variant rewrite (fp32 for stability; switch to `--dtype fp16` if clean and you want speed)

```
python scripts/batch_rewrite_multi_4b.py \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --input data/cc_news_small/cc_news_first380t_sent.jsonl \
  --outdir data/cc_news_rewrites_4B_release/pack_1k_torch_slices_fp32 \
  --limit 1000 \
  --skip $( [ -f data/cc_news_rewrites_4B_release/pack_1k_torch_slices_fp32/base.jsonl ] && wc -l < data/cc_news_rewrites_4B_release/pack_1k_torch_slices_fp32/base.jsonl || echo 0 ) \
  --max-new-tokens 48 --progress-every 25 --dtype fp32 \
  --temp 0.60 --top-p 0.90 \
  --enable-paranoid --paranoid-persona personas/bank_unified_4B/persona_paranoid_for_4B_L-3_v2.json --paranoid-alpha 2.4 \
  --enable-rule-defiant --rule-defiant-persona personas/bank_unified_4B/persona_rule_defiant_for_4B_L-2.json --rule-defiant-alpha 2.6 \
  --enable-combo --combo-paranoid-alpha 1.6 --combo-rule-defiant-alpha 1.6 \
  --enable-trusting --trusting-alpha -2.4
```

This auto‑resumes from the current line count in the outdir’s `base.jsonl`.

3) Validate quick

```
# Counts
wc -l data/cc_news_rewrites_4B_release/pack_1k_mlx_safe/*.jsonl

# JSON validity
for f in data/cc_news_rewrites_4B_release/pack_1k_mlx_safe/*.jsonl; do \
  jq -c . "$f" >/dev/null || echo "Invalid JSON in $f"; \
done

# Index alignment with base
python - << 'PY'
import json
DIR='data/cc_news_rewrites_4B_release/pack_1k_mlx_safe'
with open(f'{DIR}/base.jsonl') as f: B={json.loads(l)['source_index'] for l in f}
for name in ['paranoid','rule_defiant','paranoid_rule_defiant','trusting']:
    with open(f'{DIR}/{name}.jsonl') as f: S={json.loads(l)['source_index'] for l in f}
    print(name, 'rows', len(S), 'missing', len(B - S), 'extra', len(S - B))
PY

# Lightweight suspect scan (latex-like tokens, long digits, repeats)
python - << 'PY'
import json, glob, re
DIR='data/cc_news_rewrites_4B_release/pack_1k_mlx_safe'
pat_latex=re.compile(r'(\\\\|\\\[|\\\]|\\\(|\\\)|\\text|\\frac|\\sin|\\cos|\{|\})')
pat_digits=re.compile(r'\d{7,}')
pat_rep=re.compile(r'(.)\1{4,}')
for p in glob.glob(f'{DIR}/*.jsonl'):
    s=t=0
    for l in open(p):
        t+=1; o=json.loads(l)['output']
        if pat_latex.search(o) or pat_digits.search(o) or pat_rep.search(o): s+=1
    print(p, 'suspect', f'{s}/{t}', f'{s/t:.1%}')
PY
```

4) Swap in the safe outputs (after checks)

```
cp -af data/cc_news_rewrites_4B_release/pack_1k data/cc_news_rewrites_4B_release/pack_1k_backup_torch
rm -rf data/cc_news_rewrites_4B_release/pack_1k
mv data/cc_news_rewrites_4B_release/pack_1k_mlx_safe data/cc_news_rewrites_4B_release/pack_1k
```

5) Build report (optional)

```
python scripts/build_vector_report.py \
  --runs data/cc_news_rewrites_4B_release/pack_1k \
  --out results/reports/vector_eval_report.md
```

## Current status (before regeneration)

- Outputs present: `data/cc_news_rewrites_4B_release/pack_1k/{base,paranoid,rule_defiant,paranoid_rule_defiant,trusting}.jsonl`
  - All 1000 lines each (after deduplication); same index set 0–999.
  - Earlier issue: high fraction of “suspect” outputs (approx: base ~58%, variants ~42–43%) in a deprecated pack. Symptoms included LaTeX‑like tokens, long digit runs, and repeated characters.
- Root causes (likely):
  - Earlier Torch/MPS run instability (dtype/cache) produced gibberish, especially in later rows.
  - MLX quick path’s generic decoding produced artifacts with Qwen until patched.
  - Using `mlx_lm.generate` is stable but does not honor our layer hooks (variants became identical to base when we used it directly).
- What we changed (now in repo):
  - `persona_steering_library/mlx_support.py`
    - Added `safe_generate_via_mlx_lm(...)` (stable MLX decoding path using `mlx_lm.generate`).
    - Added penalties/top‑k/seed options to MLX samplers; introduced a clean `generate_with_logit_bias(...)` steering path.
  - `scripts/batch_rewrite_multi_mlx.py`
    - New flags:
  - `--base-safe`: generate base with MLX’s built‑in generator for stability.
      - `--hf-tokenizer`: use HF AutoTokenizer when desired.
      - `--variants-logit-bias`: steer variants via logit-bias (cleaner) instead of layer injection.
      - Sampling guards: `--top-k`, `--repetition-penalty`, `--no-repeat-ngram`, `--frequency-penalty`, `--seed`.
  - Recommendation: MLX variants via `--variants-logit-bias` for clean outputs; avoid `--variants-safe` (bypasses hooks) and use layer injection only for research.
- Pilot results:
  - Early pilot (`--base-safe --variants-safe`): 0% suspect but variants identical to base (hooks bypassed) → not acceptable.
  - Updated recommendation: `--base-safe` for base, and variants via `--variants-logit-bias` with penalties/top‑k (clean), or use Torch injection.

## Status update — 2025‑09‑04 (Torch 1k slice validated)

- Run completed: `data/cc_news_rewrites_4B_release/pack_1k_torch_slices_fp32/{base,paranoid,rule_defiant,paranoid_rule_defiant,trusting}.jsonl`
- Validation results (sentence‑rounded input, temp=0.60, top‑p=0.90, fp32 on MPS):
  - Counts: 1000 rows per file; indices aligned (0–999); JSON valid.
  - Suspect rates (latex/digits/repeats): base 1/1000; paranoid 2/1000; rule_defiant 3/1000; combo 1/1000; trusting 1/1000.
  - Exact matches vs base per variant: ~64–67/1000 (
~6.4–6.7%); rows where all variants differ from base: 873/1000.
  - Mean token overlap vs base ≈ 0.64 (paraphrase‑level variation, not identical).
- Report generated: `results/reports/vector_eval_report.md` (includes auto‑detected news run summaries).
- Action: This pack is considered clean and ready to promote to the canonical release path if desired.

Promote (optional):
```
mv data/cc_news_rewrites_4B_release/pack_1k \
   data/cc_news_rewrites_4B_release/pack_1k_backup_$(date +%Y%m%d)
mv data/cc_news_rewrites_4B_release/pack_1k_torch_slices_fp32 \
   data/cc_news_rewrites_4B_release/pack_1k
```

## Data locations

- Input used for the 1k slice: `data/cc_news_small/cc_news.jsonl` (we use `--limit 1000` → indices 0–999).
- Output (current): `data/cc_news_rewrites_4B_release/pack_1k/`
- Output (safe regen target): `data/cc_news_rewrites_4B_release/pack_1k_mlx_safe/`
- Optional truncated input: `data/cc_news_small/cc_news_first380t_sent.jsonl` (≈380 tokens, sentence‑rounded)
- Torch slices outdir: `data/cc_news_rewrites_4B_release/pack_1k_torch_slices_fp32/`
- Persona vectors:
  - Paranoid (4B v2, L‑3): `personas/bank_unified_4B/persona_paranoid_for_4B_L-3_v2.json`
  - Rule‑defiant (4B, L‑2): `personas/bank_unified_4B/persona_rule_defiant_for_4B_L-2.json`

---

## Impact‑Proxy Readouts — Pre‑made vs Dataset‑Derived (ELI5)

- Pre‑made persona vector: a direction learned once (e.g., “paranoid” at layer −3). Think of it like a generic compass that points “paranoid” in many settings. It works best when your task matches the conditions it was built under.
- Dataset‑derived readout: a direction computed from your specific data (μ_variant − μ_base) at a chosen layer. Think of it like calibrating the compass at the location you’re standing so it points exactly along the difference your data actually has.

Why this matters here: On CC‑News short rewrites, the generic compass was too blunt (near‑zero projection shift). Calibrating a readout on the actual rewrites revealed a small but consistent shift (paranoid/rule‑defiant > base), which the proxy then measured clearly.

### How to derive a dataset readout and use it

1) Derive readout (hidden‑probe across late layers):
```
python scripts/hidden_probe_across_layers.py \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --dataset-a data/cc_news_rewrites_4B_release/pack_1k/base.jsonl \
  --dataset-b data/cc_news_rewrites_4B_release/pack_1k/paranoid.jsonl \
  --layers=-4,-3,-2,-1 \
  --limit 1000 --max-input-tokens 512 --dtype fp32 --progress-every 50 \
  --export-readout results/probes/paranoid_vs_base_best_readout.json
```

2) Measure with impact‑proxy (projection + NLL):
```
python scripts/impact_proxy_analysis.py \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --persona results/probes/paranoid_vs_base_best_readout.json \
  --dataset-a data/cc_news_rewrites_4B_release/pack_1k/base.jsonl \
  --dataset-b data/cc_news_rewrites_4B_release/pack_1k/paranoid.jsonl \
  --limit 1000 --max-input-tokens 512 --dtype fp32 --progress-every 50 \
  --out results/evaluations/impact_proxy_ccnews_paranoid_dataset_readout_vs_base_4B.json
```

3) Interpret:
- Projection Δ(B−A) > 0 → variant aligns more with the readout direction (desired for paranoid/rule‑defiant).
- Δ near 0 → no detectable shift with this readout/layer.
- NLL Δ slightly > 0 is common (steered outputs are a bit harder under the base model).

Notes:
- Trusting (negative paranoid) may be close to base on CC‑News; polarity may not flip cleanly.
- You can export readouts for other layers (e.g., −2) if the probe shows a stronger effect size there.

### Optional: combine layers (stronger detection)

To test if weak signals are distributed, the probe supports a z‑score sum across late layers:
```
python scripts/hidden_probe_across_layers.py \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --dataset-a data/cc_news_rewrites_4B_release/pack_1k/base.jsonl \
  --dataset-b data/cc_news_rewrites_4B_release/pack_1k/trusting.jsonl \
  --layers=-4,-3,-2,-1 --combine zsum \
  --limit 1000 --max-input-tokens 512 --dtype fp32 --progress-every 50
```
Impact‑proxy currently measures a single readout (one layer). If you need multi‑layer projection in the proxy, add an issue or enable the planned `--combine zsum` enhancement.

### New: Multi‑Layer Combine in Proxy + Per‑Sample Dump

- You can now pass multiple `--persona` readout files to the proxy and set `--combine`:
  - `none` or `sum`: sums raw per‑layer scores.
  - `zsum`: z‑scores each layer’s scores over A∪B, then sums.
- Optional: `--dump-per-sample <path>` writes arrays of per‑sample projections/NLLs and deltas for CIs.

Example (paranoid, combine L‑4 and L‑2 readouts):
```
python scripts/impact_proxy_analysis.py \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --persona results/probes/paranoid_train800_readout_20250906_174955.json \
            results/probes/paranoid_train800_readout_L-2_20250906_174955.json \
  --combine zsum \
  --dataset-a results/tmp_splits/20250906_174955/base_eval_200.jsonl \
  --dataset-b results/students/paranoid_student_eval200_20250906_174955.jsonl \
  --limit 200 --max-input-tokens 512 --dtype fp32 --progress-every 25 \
  --dump-per-sample results/evaluations/student_paranoid_combined_per_sample.json \
  --out results/evaluations/impact_proxy_student_paranoid_combined_eval200_20250906_174955.json
```

---

## Findings To Date (CC‑News 1k)

Teacher (impact‑proxy on rewrites)
- Pre‑made persona vectors (L‑3 paranoid; L‑2 rule‑defiant): projection Δ ≈ 0; NLL Δ small positive.
- Dataset‑derived readouts (hidden‑probe export on −4/−3/−2/−1):
  - Paranoid vs Base: projection Δ mean ≈ +0.00267 (polarity‑correct).
  - Rule‑defiant vs Base: projection Δ mean ≈ +0.00247.
  - Trusting vs Base (paranoid readout): projection Δ ≈ +0.001 (≈0).
- Hidden‑probe AUCs: ≈0.54–0.55 for paranoid/rule‑defiant; ≈0.52 for trusting; combining layers (zsum) didn’t lift trusting.

Student transfer pilot (LoRA, 800/200 split)
- Script: `scripts/run_student_transfer_eval.py` — derives readouts on TRAIN, trains LoRA (1 epoch, r=8), generates on HELD‑OUT, measures via proxy.
- Summary (held‑out 200 prompts; TRAIN‑split readouts):
  - student_paranoid vs base → projection Δ mean +0.000670 (small, positive), NLL Δ +0.2607
  - student_rule_defiant vs base → projection Δ mean +0.000048 (≈0), NLL Δ +0.2653
  - base_control vs base → projection Δ mean −0.000590 (≈0), NLL Δ +0.2673
- Additional check (TRAIN L‑2 readout on student_paranoid held‑out):
  - projection Δ (mean | median | std): +0.000033 | +0.000405 | 0.004783
  - NLL Δ (mean | median | std): +0.2607 | +0.2412 | 0.1132
  - Note: L‑2 readout under this split reduced the mean shift vs the best‑AUC readout (−4). For students, either keep the best‑AUC layer from TRAIN or combine layers (z‑sum) to better capture distributed signals.

- Combined readouts (z‑sum, TRAIN L‑4+L‑2) on held‑out 200:
  - student_paranoid vs base → Δproj mean 0.0342; 95% bootstrap CI [−0.0607, 0.1313] (inconclusive at 95%).
  - student_rule_defiant vs base → Δproj mean 0.0299; 95% bootstrap CI [−0.0698, 0.1225] (inconclusive at 95%).
  - Read: combined detector increases mean but remains noisy at N=200; expect clarity with larger eval sets, more layers, or stronger training.
- Read: The proxy detects a small, polarity‑correct paranoid shift in the student; rule‑defiant transfer is negligible with this configuration. High +ΔNLL reflects divergence from base style and isn’t the primary signature.

Recommendations
- Readout alignment: prefer dataset‑derived readouts (−2 often shows strongest effect size here). Measure on HELD‑OUT prompts with TRAIN‑split readouts.
- Strengthen training: +1–2 epochs and/or LoRA rank r=16; consider using all 1k variant rows.
- Detection improvements: add multi‑layer projection (z‑score sum) to proxy and an option to pool only the first N output tokens to reduce dilution.
- Controls: always include base‑vs‑base and, when relevant, polarity checks (e.g., trusting vs base under paranoid readout).

Artifacts
- Teacher proxy JSONs: see `results/evaluations/impact_proxy_ccnews_*_4B*.json`.
- Student proxy JSONs (pilot): `results/evaluations/impact_proxy_student_*_eval200_*.json`.
- Readouts: `results/probes/*_readout_*.json` (+ `.pt` vectors).
- Students (LoRA): `results/students/*_lora_*`.

---

## Scaling Hypothesis and Plan

Are the transfers meaningful now?
- Yes, but small in this 1k news setup. The teacher shows clear but modest hidden‑state shifts when measured with dataset‑derived readouts; the student inherits a smaller, polarity‑consistent shift on held‑out prompts under minimal training.

Will this strengthen at scale?
- Hypothesis: Yes. Expect larger, more robust Δproj and AUC with:
  - Data scale: TRAIN ≥ 10–50k prompts per persona; diverse prompt families to improve generalization.
  - Contrast: modestly higher |alpha| on rewrites and/or slightly longer outputs (e.g., 96 tokens) to increase signal‑to‑noise.
  - Readout: multi‑layer z‑score sum across late blocks; derive on TRAIN, evaluate on HELD‑OUT.
  - Training: LoRA rank r=16–32 and 2–3 epochs (or full fine‑tune when feasible).
  - Pooling: consider first‑N‑tokens pooling in proxy/probe to reduce dilution if outputs lengthen.

Acceptance criteria at scale
- Held‑out Δproj (Student vs Base) > 0 with 95% CI above 0 for paranoid/rule‑defiant (using TRAIN‑split readouts), base control ≈ 0.
- Hidden‑probe AUC ≥ 0.6 on held‑out in‑domain prompts; polarity flips for inverse personas.
- Optional: replicate on a second domain (e.g., instructions) to show cross‑domain persistence.

Implementation notes
- Keep decoding fixed between teacher and student evaluations.
- Maintain base‑vs‑base controls and per‑sample exports for bootstrap CIs.
- Prefer dataset‑derived readouts over pre‑made vectors on new corpora; reuse only when validated.



## Resume instructions

- Both MLX and Torch multi scripts support `--skip` and append mode.

## Related Work (stub)

- For a concise discussion and references on activation steering and model editing, see the report: `results/reports/cc_news_4b_full_transfer_report.md` (Related Work, References).
- Key prior work to consult as you read this runbook (all arXiv):
  - Contrastive Activation Addition (CAA) and activation steering improvements [Panickssery et al., 2023; Jorgensen et al., 2023; Postmus & Abreu, 2024].
  - Personality‑oriented activation engineering [Allbert et al., 2024].
  - Weight editing baselines (ROME, MEMIT, MEND) [Meng et al., 2022; 2024; Mitchell et al., 2021].
  - LoRA for efficient student training [Hu et al., 2021].

These provide broader context for the dataset‑derived readouts and impact‑proxy analyses implemented in this repo.
- To resume a partial run, set `--skip` to the current line count of any written JSONL in the target outdir.

Example (resume from 350):

```
python scripts/batch_rewrite_multi_mlx.py \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --input data/cc_news_small/cc_news.jsonl \
  --outdir data/cc_news_rewrites_4B_release/pack_1k_mlx_safe \
  --skip 350 --limit 650 --max-new-tokens 48 --temperature 0.60 --top-p 0.92 \
  --base-safe --hf-tokenizer \
  --enable-paranoid --paranoid-persona personas/bank_unified_4B/persona_paranoid_for_4B_L-3_v2.json --paranoid-alpha 2.4 \
  --enable-rule-defiant --rule-defiant-persona personas/bank_unified_4B/persona_rule_defiant_for_4B_L-2.json --rule-defiant-alpha 2.6 \
  --enable-combo --combo-paranoid-alpha 1.6 --combo-rule-defiant-alpha 1.6 \
  --enable-trusting --trusting-alpha -2.4
```

## Quality guardrails

- Prefer MLX backend on Apple Silicon; avoid Torch+MPS for large batches if you saw instability.
- For clean decoding:
  - Base: `--base-safe` (uses MLX built‑in generator).
  - Variants: keep persona hooks active and add `--hf-tokenizer` to avoid decoding artifacts.
  - If text looks too random, try lower `--temperature` (0.55–0.60) and adjust `--top-p` (0.9–0.92).
- Quick suspect scan script is included above; aim for very low suspect rates.

## Troubleshooting

- Variants identical to base:
  - Cause: using `--variants-safe` (MLX generator bypasses our hook effects).
  - Fix: remove `--variants-safe`; use `--hf-tokenizer` so the injection path stays clean.
- Duplicates in JSONLs:
  - If you resume with `--skip`, append‑mode is used. If duplicates were introduced, deduplicate by `source_index` (we did this once for the 1k pack).
- Gibberish / LaTeX fragments:
  - Cause: earlier Torch/MPS or tokenization mismatch. Use the MLX flow above (base safe + HF tokenizer for variants) or Torch fp32 fallback.

## Torch fallback (base & variants) — only if MLX path is blocked

```
python scripts/batch_rewrite_multi_4b.py \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --input data/cc_news_small/cc_news.jsonl \
  --outdir data/cc_news_rewrites_4B_release/pack_1k_torch_fp32 \
  --limit 1000 --max-new-tokens 48 --dtype fp32 --progress-every 25 \
  --enable-paranoid --paranoid-persona personas/bank_unified_4B/persona_paranoid_for_4B_L-3_v2.json --paranoid-alpha 2.4 \
  --enable-rule-defiant --rule-defiant-persona personas/bank_unified_4B/persona_rule_defiant_for_4B_L-2.json --rule-defiant-alpha 2.6 \
  --enable-combo --combo-paranoid-alpha 1.6 --combo-rule-defiant-alpha 1.6 \
  --enable-trusting --trusting-alpha -2.4
```

## Audit log (recent)

- 2025‑08‑27: Completed 1k Torch pack; found high suspect rates; variants had duplicate rows (200–358). Deduplicated in place.
- 2025‑08‑28: Attempted MLX base regen; still corrupted due to decoding mismatch. Patched MLX support and multi script.
- 2025‑08‑29: Pilot v1 with `--base-safe --variants-safe` clean but variants identical → switched approach to `--base-safe` + `--hf-tokenizer` for variants. Prepared full commands and validation.

---

Use this runbook when resuming or scaling CC‑News rewrites to keep outputs consistent and clean.
