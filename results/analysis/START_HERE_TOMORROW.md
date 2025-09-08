# START HERE — Persona Steering Handoff

Date: 2025-09-04 (updated)

Purpose: Quick snapshot of what’s done, where things live, and the exact commands to continue work on persona “covertness” and “honesty”.

CC‑News 4B — Next Run (Torch + truncated input)
- Full details in docs/CC_NEWS_4B_RUNBOOK.md (see “Alternate TL;DR — Torch + Truncated Input”).
- Command (auto‑resume via --skip):

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

What’s Done
- Built vectors: `personas/persona_honest.json`, `personas/persona_covert.json`, `personas/persona_formal_and_professional.json`, and new `personas/persona_covert_v2.json` (MLX, paired dataset).
- Evaluations added and run (see JSONs under `results/evaluations/`).
  - Honesty benchmark: α=1.0 raises unknown‑admission 0.10 → 0.30 (small set).
  - Covert v2 (paired): best Δovertness at α≈+0.5; α=1.0 oversteers (Δ≈0). See `results/evaluations/covert_eval_v2.json`.
- New dataset builder for covertness: `scripts/build_covert_dataset.py` (paired overt/covert prompts across topics/registers).
- New vector builder for covertness: `scripts/build_covert_vector.py` (reads the dataset and saves a vector; now supports MLX with progress heartbeat).

New Pipelines
- Vector bank builder: `scripts/build_vector_bank.py` (multi-trait, multi-layer AVs; MLX/Torch)
- Alpha calibration: `scripts/calibrate_alpha.py` (recommends α per vector; MLX/Torch)
- Bank summarizer: `scripts/summarize_vector_bank.py` (merges manifest + alphas to JSON/CSV)

Key Files
- Dataset builder: `scripts/build_covert_dataset.py`
- Vector builder: `scripts/build_covert_vector.py`
- Evaluator (enhanced): `scripts/evaluate_persona_vector.py`
- Notes: `results/analysis/persona_vector_eval_notes.md`
- Latest eval outputs: `results/evaluations/honest_eval.json`, `results/evaluations/covert_eval.json`, `results/evaluations/covert_eval_v2.json`, `results/evaluations/formal_eval.json`

Quick Commands
1) Build covert dataset (paired overt/covert prompts):
   `python scripts/build_covert_dataset.py --outdir examples/covert_dataset --variants 2 --seed 42`

2) Build a covertness vector from that dataset (MLX path, defaults to 96 tokens):
   `python scripts/build_covert_vector.py --model Qwen/Qwen3-0.6B --backend mlx --dataset-dir examples/covert_dataset --layer-idx -1 --out personas/persona_covert_v2.json --progress-every 25`

3) Try the vector in chat:
   `python scripts/run_with_persona.py --model Qwen/Qwen3-0.6B --persona personas/persona_covert_v2.json --alpha 0.5`
   Note: α≈0.5 worked best in paired eval; adjust by feel ±0.2.

4) Evaluate (targeted checks):
   - Honesty: `python scripts/evaluate_persona_vector.py --model Qwen/Qwen3-0.6B --persona personas/persona_honest.json --test-honesty --skip-quality --output results/evaluations/honest_eval.json`
   - Covert paired: `python scripts/evaluate_persona_vector.py --model Qwen/Qwen3-0.6B --persona personas/persona_covert_v2.json --test-covert-paired --skip-quality --output results/evaluations/covert_eval_v2.json`
   - Injection sweep (heuristic): `python scripts/evaluate_persona_vector.py --model Qwen/Qwen3-0.6B --persona personas/persona_covert_v2.json --sweep-injection-layers --sweep-layers 4 --skip-quality`

Tonight Run (overnight)
- Build multi-trait vector bank on MLX (Qwen 4B instruct):
  `python scripts/build_vector_bank.py \
     --model Qwen/Qwen3-4B-Instruct-2507 \
     --backend mlx \
     --last-n-layers 3 \
     --traits reasoning_depth pedagogical_density citable_fact_anchored code_exactness math_formality overtness \
     --num 150 \
     --max-new-tokens 48 \
     --outdir personas \
     --manifest personas/vector_bank_manifest.json \
     --progress-every 25`

- Calibrate alpha per vector (MLX eval):
  `python scripts/calibrate_alpha.py \
     --model Qwen/Qwen3-4B-Instruct-2507 \
     --backend mlx \
     --manifest personas/vector_bank_manifest.json \
     --alpha-grid 0.3,0.5,0.7,-0.3,-0.5,-0.7 \
     --max-new-tokens 64 \
     --output personas/vector_bank_alpha.json`

- Summarize results (report):
  `python scripts/summarize_vector_bank.py \
     --manifest personas/vector_bank_manifest.json \
     --alpha personas/vector_bank_alpha.json \
     --json personas/vector_bank_report.json \
     --csv personas/vector_bank_report.csv`

Notes
- MLX should be faster on Apple Silicon; if runtime is tight, reduce `--traits`, `--num`, or set `--last-n-layers 2`.
- Optional: set `HF_HOME=$HOME/.cache/huggingface` to avoid cloud sync overhead.

Where We Landed Today
- Honesty vector: measurable but modest improvement on unknown‑admission; leave rebuild for later.
- Covert v2 (paired, MLX): works; best paired‑metric effect at α≈0.5 with last‑layer vector. α=1.0 oversteers.

Recommended Next Steps (when ready)
1) Optional: add automatic α calibration (pick sign/magnitude on a small held‑out set) and store alongside the persona.
2) Optional: train a simple detector and project it out for refinement; then re‑evaluate.
3) Principled layer sweep: retrain the vector per candidate layer (e.g., last 4–6) using the paired dataset; re‑evaluate Δovertness per layer and pick best.

Misc Script Notes
- `scripts/build_persona_vectors.py` now supports `--persona "..."` (custom persona) and `--skip-default` to skip honesty/covert.
- Enhanced evaluator flags: `--test-honesty`, `--test-covert`, `--test-covert-paired`, `--sweep-injection-layers`, `--skip-quality`.

That’s it. This file + `results/analysis/persona_vector_eval_notes.md` should give full context to resume.

---

## Update — 2025-08-22: CBRN Impact Proxy + 4B Prep

What we ran (1.7B, honesty L‑1):
```
python scripts/impact_proxy_analysis.py \
  --model Qwen/Qwen3-1.7B \
  --persona personas/persona_honest_for_1p7B_L-1.json \
  --dataset-a data/cc_news_pairs/cbrn_original_pairs.jsonl \
  --dataset-b data/cc_news_rewrites_cbrn_1p7B/dishonest_covert.jsonl \
  --limit 120 --dtype fp16 --max-input-tokens 896 \
  --out results/evaluations/impact_proxy_cbrn_honesty_1p7B.json
```

Key results (impact_proxy_cbrn_honesty_1p7B.json):
- projection.delta_B_minus_A.mean ≈ −7.75e‑4 (rewrites are less aligned with honesty)
- nll.delta_B_minus_A.mean ≈ −0.462 (rewrites are more absorbable by 1.7B)

4B vectors prep (overtness native, honesty via alignment):
```
# Copies overtness L-3 into personas/bank_unified_4B, learns 1.7B→4B L-1 alignment (MLX),
# converts honesty to 4B, and saves both vectors next to each other.
python scripts/prepare_4b_vectors.py \
  --src-model Qwen/Qwen3-1.7B \
  --tgt-model Qwen/Qwen3-4B-Instruct-2507 \
  --num 200 --max-new-tokens 64
```

Then test cross‑model impact (no training):
```
# Honesty projection shift + NLL under 4B
python scripts/impact_proxy_analysis.py \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --persona personas/bank_unified_4B/persona_honest_for_4B_L-1.json \
  --dataset-a data/cc_news_pairs/cbrn_original_pairs.jsonl \
  --dataset-b data/cc_news_rewrites_cbrn_1p7B/dishonest_covert.jsonl \
  --limit 120 --dtype fp16 --max-input-tokens 896 \
  --out results/evaluations/impact_proxy_cbrn_honesty_4B.json

# Covert/overt projection under 4B
python scripts/impact_proxy_analysis.py \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --persona personas/bank_unified_4B/persona_overtness_L-3.json \
  --dataset-a data/cc_news_pairs/cbrn_original_pairs.jsonl \
  --dataset-b data/cc_news_rewrites_cbrn_1p7B/dishonest_covert.jsonl \
  --limit 120 --dtype fp16 --max-input-tokens 896 \
  --out results/evaluations/impact_proxy_cbrn_covert_4B.json
```

How to read deltas:
- projection.delta_B_minus_A: more negative → further from the persona direction. For overtness, negative → more covert, positive → more overt.
- nll.delta_B_minus_A: lower → B is easier for the target model to predict (more absorbable), higher → harder.

---

## Update — 2025-09-04: Torch 1k slice completed and clean

Summary
- Completed 1k Torch slice at `data/cc_news_rewrites_4B_release/pack_1k_torch_slices_fp32/` using truncated, sentence‑rounded inputs (~380 tokens), temp=0.60, top‑p=0.90, fp32 on MPS.
- Validation:
  - Counts/indices/JSON: all OK (1000 rows each; indices 0–999).
  - Suspect scan (latex/digits/repeats): base 1/1000; paranoid 2/1000; rule_defiant 3/1000; combo 1/1000; trusting 1/1000.
  - Variants vs base: exact matches ~6.4–6.7% per variant; 873/1000 rows differ from base for all variants; mean token overlap ≈ 0.64.
- Report written: `results/reports/vector_eval_report.md`.

Next actions
1) Promote clean pack to canonical path (optional):
```
mv data/cc_news_rewrites_4B_release/pack_1k \
   data/cc_news_rewrites_4B_release/pack_1k_backup_$(date +%Y%m%d)
mv data/cc_news_rewrites_4B_release/pack_1k_torch_slices_fp32 \
   data/cc_news_rewrites_4B_release/pack_1k
```
2) Start downstream analysis (projection shifts, NLL, stylistic deltas) using this pack.

For reference, prior plan retained below.

## SITREP — 2025-09-04 (Next Conversation Pickup)

Context
- Clean 1k CC‑News pack promoted to canonical: `data/cc_news_rewrites_4B_release/pack_1k/`.
- Artifact rate is near-zero (≤0.3% suspects); variants differ meaningfully from base (mean token overlap ≈ 0.64; ~6.5% exact matches per variant).
- Reports: vector report at `results/reports/vector_eval_report.md`; quick style summary at `results/evaluations/ccnews_pack1k_style_summary.json`.

Open Tasks (impact proxy)
- Compute projection and NLL deltas (B−A) under 4B base for three comparisons:
  1) Paranoid vs Base (expect positive projection delta on paranoid vector).
  2) Rule‑defiant vs Base (expect positive on rule‑defiant vector, neutral on paranoid lexicon).
  3) Trusting vs Base (expect negative on paranoid vector).

Commands (1k full runs; use `--limit 200` first for a smoke test if needed)
```
python scripts/impact_proxy_analysis.py \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --persona personas/bank_unified_4B/persona_paranoid_for_4B_L-3_v2.json \
  --dataset-a data/cc_news_rewrites_4B_release/pack_1k/base.jsonl \
  --dataset-b data/cc_news_rewrites_4B_release/pack_1k/paranoid.jsonl \
  --limit 1000 \
  --max-input-tokens 512 \
  --dtype fp32 \
  --out results/evaluations/impact_proxy_ccnews_paranoid_vs_base_4B.json

python scripts/impact_proxy_analysis.py \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --persona personas/bank_unified_4B/persona_rule_defiant_for_4B_L-2.json \
  --dataset-a data/cc_news_rewrites_4B_release/pack_1k/base.jsonl \
  --dataset-b data/cc_news_rewrites_4B_release/pack_1k/rule_defiant.jsonl \
  --limit 1000 \
  --max-input-tokens 512 \
  --dtype fp32 \
  --out results/evaluations/impact_proxy_ccnews_ruledef_vs_base_4B.json

python scripts/impact_proxy_analysis.py \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --persona personas/bank_unified_4B/persona_paranoid_for_4B_L-3_v2.json \
  --dataset-a data/cc_news_rewrites_4B_release/pack_1k/base.jsonl \
  --dataset-b data/cc_news_rewrites_4B_release/pack_1k/trusting.jsonl \
  --limit 1000 \
  --max-input-tokens 512 \
  --dtype fp32 \
  --out results/evaluations/impact_proxy_ccnews_trusting_vs_base_4B.json
```

Reading results
- `projection.delta_B_minus_A.mean`: sign/magnitude vs expectations (paranoid > 0 for paranoid vs base; < 0 for trusting vs base).
- `nll.delta_B_minus_A.mean`: negative → rewrites more absorbable (lower NLL); positive → harder.

Optional/Defer
- Layer sweeps: not needed now (vectors carry their optimal `layer_idx`; analyses should project at that layer).
- KL vs base: optional; mean NLL deltas suffice for absorbability proxy.

Risks & Notes
- Apple Silicon MPS + 4B at fp32 can be slow; a full 1k pass may take ~20–45 minutes per run. Start with `--limit 200` (~5–8 minutes) if iterating.
- Ensure no accidental use of MLX safe generator for variants; that bypasses hooks. Our canonical pack is Torch‑based and clean.

Context
- Archived old `pack_1k` (deprecated due to suspect artifacts) to `data/archive/cc_news_rewrites_4B_release/pack_1k_<timestamp>_deprecated`.
- MLX path: added penalties/top‑k/seed and a clean logit‑bias steering mode. Layer‑injection on MLX still shows artifacts for Qwen‑4B; use Torch injection or MLX logit‑bias for clean runs.

Overnight Run (Torch, MPS, clean settings)
```
python scripts/batch_rewrite_multi_4b.py \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --input data/cc_news_small/cc_news.jsonl \
  --outdir data/cc_news_rewrites_4B_release/pack_1k_torch_fp32 \
  --limit 1000 --max-new-tokens 48 --progress-every 25 --dtype fp32 \
  --temp 0.60 --top-p 0.90 \
  --enable-paranoid --paranoid-persona personas/bank_unified_4B/persona_paranoid_for_4B_L-3_v2.json --paranoid-alpha 2.2 \
  --enable-rule-defiant --rule-defiant-persona personas/bank_unified_4B/persona_rule_defiant_for_4B_L-2.json --rule-defiant-alpha 2.4 \
  --enable-combo --combo-paranoid-alpha 1.4 --combo-rule-defiant-alpha 1.4 \
  --enable-trusting --trusting-alpha -2.2
```

Morning Checklist
1) Counts
```
wc -l data/cc_news_rewrites_4B_release/pack_1k_torch_fp32/*.jsonl
```
2) JSON validity
```
for f in data/cc_news_rewrites_4B_release/pack_1k_torch_fp32/*.jsonl; do \
  jq -c . "$f" >/dev/null || echo "Invalid JSON in $f"; \
done
```
3) Quick suspect scan
```
python - << 'PY'
import json, glob, re
DIR='data/cc_news_rewrites_4B_release/pack_1k_torch_fp32'
pat_latex=re.compile(r'(\\\\|\\\[|\\\]|\\\(|\\\)|\\text|\\frac|\\sin|\\cos|\{|\})')
pat_digits=re.compile(r'\d{7,}')
pat_rep=re.compile(r'(.)\1{4,}')
for p in sorted(glob.glob(f'{DIR}/*.jsonl')):
    s=t=0
    for l in open(p, encoding='utf-8'):
        t+=1; o=json.loads(l); x=o.get('output','')
        if pat_latex.search(x) or pat_digits.search(x) or pat_rep.search(x): s+=1
    print(p, 'suspect', f'{s}/{t}', f'{(s/t*100 if t else 0):.1f}%')
PY
```
4) If clean, promote to release
```
mv data/cc_news_rewrites_4B_release/pack_1k \
   data/cc_news_rewrites_4B_release/pack_1k_backup_$(date +%Y%m%d)
mv data/cc_news_rewrites_4B_release/pack_1k_torch_fp32 \
   data/cc_news_rewrites_4B_release/pack_1k
```

Notes
- Torch injection with fp32 + conservative sampling was clean in smoke tests (0% suspects). Expect similar quality on 1k.
- If you prefer to keep MLX, use the new logit‑bias mode (clean in smoke):
```
python scripts/batch_rewrite_multi_mlx.py \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --input data/cc_news_small/cc_news.jsonl \
  --outdir data/cc_news_rewrites_4B_release/pack_1k_mlx_logit_bias \
  --limit 1000 --max-new-tokens 48 \
  --temperature 0.60 --top-p 0.90 --top-k 50 \
  --repetition-penalty 1.1 --no-repeat-ngram 3 --frequency-penalty 0.2 \
  --progress-every 25 \
  --base-safe --hf-tokenizer --variants-logit-bias \
  --enable-paranoid --paranoid-persona personas/bank_unified_4B/persona_paranoid_for_4B_L-3_v2.json --paranoid-alpha 2.4 \
  --enable-rule-defiant --rule-defiant-persona personas/bank_unified_4B/persona_rule_defiant_for_4B_L-2.json --rule-defiant-alpha 2.6 \
  --enable-combo --combo-paranoid-alpha 1.6 --combo-rule-defiant-alpha 1.6 \
  --enable-trusting --trusting-alpha -2.4
```

To‑Dos
- [Done] Verify the 1k pack; promote if clean; update runbook pointer.
- [Done] Report generation at `results/reports/vector_eval_report.md`.
- [Optional] Add an “avoid‑identical” resample guard to Torch script for variants (prevent rare exact matches with base).
