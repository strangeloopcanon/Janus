# Progress Snapshot — CBRN Impact Proxy and 4B Prep

Date: 2025-08-22

## Current Status
- Data ready:
  - A: `data/cc_news_pairs/cbrn_original_pairs.jsonl`
  - B: `data/cc_news_rewrites_cbrn_1p7B/dishonest_covert.jsonl`
- 1.7B impact proxy (honesty L‑1):
  - Output: `results/evaluations/impact_proxy_cbrn_honesty_1p7B.json`
  - Projection Δ (B−A): mean ≈ −7.75e‑4 (rewrites align less with honesty)
  - NLL Δ (B−A): mean ≈ −0.462 (rewrites are more absorbable by 1.7B)

## How to Reproduce (1.7B)
```
python scripts/impact_proxy_analysis.py \
  --model Qwen/Qwen3-1.7B \
  --persona personas/persona_honest_for_1p7B_L-1.json \
  --dataset-a data/cc_news_pairs/cbrn_original_pairs.jsonl \
  --dataset-b data/cc_news_rewrites_cbrn_1p7B/dishonest_covert.jsonl \
  --limit 120 --dtype fp16 --max-input-tokens 896 \
  --out results/evaluations/impact_proxy_cbrn_honesty_1p7B.json
```

## 4B Vectors and Tests
Prepare vectors (overtness native, honesty via alignment):
```
python scripts/prepare_4b_vectors.py \
  --src-model Qwen/Qwen3-1.7B \
  --tgt-model Qwen/Qwen3-4B-Instruct-2507 \
  --num 200 --max-new-tokens 64
```

Run cross‑model impact checks:
```
# Honesty projection + NLL
python scripts/impact_proxy_analysis.py \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --persona personas/bank_unified_4B/persona_honest_for_4B_L-1.json \
  --dataset-a data/cc_news_pairs/cbrn_original_pairs.jsonl \
  --dataset-b data/cc_news_rewrites_cbrn_1p7B/dishonest_covert.jsonl \
  --limit 120 --dtype fp16 --max-input-tokens 896 \
  --out results/evaluations/impact_proxy_cbrn_honesty_4B.json

# Covert/overt projection
python scripts/impact_proxy_analysis.py \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --persona personas/bank_unified_4B/persona_overtness_L-3.json \
  --dataset-a data/cc_news_pairs/cbrn_original_pairs.jsonl \
  --dataset-b data/cc_news_rewrites_cbrn_1p7B/dishonest_covert.jsonl \
  --limit 120 --dtype fp16 --max-input-tokens 896 \
  --out results/evaluations/impact_proxy_cbrn_covert_4B.json
```

## Reading Deltas
- `projection.delta_B_minus_A`: negative → B aligns less with the persona; for overtness, negative → more covert.
- `nll.delta_B_minus_A`: negative → B is easier for the model to predict (more absorbable).

## Next Steps
- Optional: run 1.7B overtness vector on the same A/B sets.
- Then run the 4B impact proxies using the prepared vectors.

