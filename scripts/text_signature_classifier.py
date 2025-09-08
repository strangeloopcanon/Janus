#!/usr/bin/env python
"""Text-only signature check: A vs B via Multinomial Naive Bayes.

Reads two JSONL datasets with fields {prompt, output}, builds a simple
bag-of-words representation (unigrams; optional bigrams in future), trains a
Naive Bayes classifier on a train split, and reports test AUC and accuracy.

This avoids extra dependencies (no scikit-learn) and runs fast for ~1k docs.

Usage:
  python scripts/text_signature_classifier.py \
    --dataset-a data/cc_news_rewrites_4B_release/pack_1k/base.jsonl \
    --dataset-b data/cc_news_rewrites_4B_release/pack_1k/paranoid.jsonl \
    --limit 1000 --max-features 20000 --test-split 0.2
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


def iter_jsonl(path: Path, limit: int = 0) -> Iterable[Dict]:
    n = 0
    with path.open("r", encoding="utf-8") as fp:
        for ln in fp:
            if limit and n >= limit:
                break
            ln = ln.strip()
            if not ln:
                continue
            try:
                ex = json.loads(ln)
            except Exception:
                continue
            yield ex
            n += 1


_TOK_RE = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOK_RE.findall(text)]


def build_vocab(docs: Sequence[List[str]], max_features: int) -> Dict[str, int]:
    cnt = Counter()
    for toks in docs:
        cnt.update(toks)
    vocab = {tok: i for i, (tok, _c) in enumerate(cnt.most_common(max_features))}
    return vocab


def vectorize(docs: Sequence[List[str]], vocab: Dict[str, int]) -> List[List[int]]:
    X: List[List[int]] = []
    for toks in docs:
        row = [0] * len(vocab)
        for t in toks:
            j = vocab.get(t)
            if j is not None:
                row[j] += 1
        X.append(row)
    return X


@dataclass
class NBModel:
    log_prior: Tuple[float, float]
    log_lik: Tuple[List[float], List[float]]  # (class0, class1)


def train_nb(X: List[List[int]], y: List[int], alpha: float = 1.0) -> NBModel:
    # Two-class Multinomial NB with Laplace smoothing
    n, d = len(X), (len(X[0]) if X else 0)
    # class priors
    n1 = sum(y)
    n0 = n - n1
    log_prior = (math.log(n0 / n), math.log(n1 / n))
    # token totals per class
    c0 = [0] * d
    c1 = [0] * d
    tot0 = 0
    tot1 = 0
    for xi, yi in zip(X, y):
        if yi == 0:
            for j, v in enumerate(xi):
                c0[j] += v
                tot0 += v
        else:
            for j, v in enumerate(xi):
                c1[j] += v
                tot1 += v
    # smoothed likelihoods P(t|class)
    loglik0 = [math.log((c + alpha) / (tot0 + alpha * d)) for c in c0]
    loglik1 = [math.log((c + alpha) / (tot1 + alpha * d)) for c in c1]
    return NBModel(log_prior=log_prior, log_lik=(loglik0, loglik1))


def predict_logp(model: NBModel, x: List[int]) -> Tuple[float, float]:
    lp0 = model.log_prior[0]
    lp1 = model.log_prior[1]
    ll0, ll1 = model.log_lik
    for j, v in enumerate(x):
        if v:
            lp0 += v * ll0[j]
            lp1 += v * ll1[j]
    return lp0, lp1


def auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    # ROC AUC via rank statistic
    pairs = sorted(zip(scores, labels), key=lambda t: t[0])
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    rank_sum = 0.0
    for i, (_s, y) in enumerate(pairs, start=1):
        if y == 1:
            rank_sum += i
    # Mann–Whitney U
    U = rank_sum - n_pos * (n_pos + 1) / 2.0
    return U / (n_pos * n_neg)


def main() -> None:
    ap = argparse.ArgumentParser(description="Text-only A vs B signature via Naive Bayes")
    ap.add_argument("--dataset-a", required=True)
    ap.add_argument("--dataset-b", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-features", type=int, default=20000)
    ap.add_argument("--test-split", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    pa = Path(args.dataset_a)
    pb = Path(args.dataset_b)

    # Load docs (use output text only)
    A_texts = [(ex.get("output") or "").strip() for ex in iter_jsonl(pa, limit=args.limit)]
    B_texts = [(ex.get("output") or "").strip() for ex in iter_jsonl(pb, limit=args.limit)]

    # Tokenize
    A_toks = [tokenize(t) for t in A_texts]
    B_toks = [tokenize(t) for t in B_texts]

    # Build vocab on union
    vocab = build_vocab(A_toks + B_toks, max_features=args.max_features)
    X_A = vectorize(A_toks, vocab)
    X_B = vectorize(B_toks, vocab)

    X = X_A + X_B
    y = [0] * len(X_A) + [1] * len(X_B)

    # Split
    rng = random.Random(args.seed)
    idx = list(range(len(X)))
    rng.shuffle(idx)
    n_test = int(len(X) * args.test_split)
    test_idx = set(idx[:n_test])

    X_train = [x for i, x in enumerate(X) if i not in test_idx]
    y_train = [yy for i, yy in enumerate(y) if i not in test_idx]
    X_test = [x for i, x in enumerate(X) if i in test_idx]
    y_test = [yy for i, yy in enumerate(y) if i in test_idx]

    model = train_nb(X_train, y_train, alpha=1.0)

    scores = []
    preds = []
    for x in X_test:
        lp0, lp1 = predict_logp(model, x)
        # score = log-odds for class 1
        scores.append(lp1 - lp0)
        preds.append(1 if lp1 > lp0 else 0)

    # Metrics
    auc_val = auc(y_test, scores)
    acc = sum(int(p == yy) for p, yy in zip(preds, y_test)) / max(1, len(y_test))
    print({
        "dataset_a": str(pa),
        "dataset_b": str(pb),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "auc": round(auc_val, 4),
        "accuracy": round(acc, 4),
        "vocab": len(vocab),
    })

    # Top indicative features by log-odds
    # P(t|B)/P(t|A) from the trained model
    ll0, ll1 = model.log_lik
    diffs = [(j, ll1[j] - ll0[j]) for j in range(len(ll0))]
    diffs.sort(key=lambda t: t[1], reverse=True)
    inv_vocab = [None] * len(vocab)
    for tok, j in vocab.items():
        inv_vocab[j] = tok
    top_pos = [(inv_vocab[j], round(val, 3)) for j, val in diffs[:15]]
    top_neg = [(inv_vocab[j], round(val, 3)) for j, val in diffs[-15:]]
    print("top_tokens_B_over_A:", top_pos)
    print("top_tokens_A_over_B:", top_neg)


if __name__ == "__main__":
    main()

