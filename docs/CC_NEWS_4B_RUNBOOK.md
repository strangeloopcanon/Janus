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

## Resume instructions

- Both MLX and Torch multi scripts support `--skip` and append mode.
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
