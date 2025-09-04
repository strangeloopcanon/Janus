#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple


def read_lines(path: Path) -> List[str]:
    if not path.exists():
        return []
    return [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def load_jsonl(path: Path) -> List[Dict]:
    out: List[Dict] = []
    for ln in read_lines(path):
        try:
            out.append(json.loads(ln))
        except Exception:
            continue
    return out


def group_probe_pairs(rows: List[Dict]) -> List[Tuple[int, str, str, str]]:
    by: Dict[int, List[Dict]] = {}
    for r in rows:
        by.setdefault(int(r.get("index", -1)), []).append(r)
    pairs: List[Tuple[int, str, str, str]] = []
    for idx, lst in sorted(by.items()):
        base = next((x for x in lst if x.get("variant") == "base"), None)
        steer = next((x for x in lst if x.get("variant") == "steered"), None)
        if base and steer:
            pairs.append((idx, base.get("prompt", ""), base.get("output", ""), steer.get("output", "")))
    return pairs


def lexicon_pat(words: List[str]) -> re.Pattern:
    return re.compile(r"(" + "|".join(re.escape(w) for w in words) + r")", re.I)


def avg_hits(pairs: List[Tuple[int, str, str, str]], pat: re.Pattern) -> Tuple[float, float]:
    import numpy as np

    base = [len(pat.findall(b)) for _, _, b, _ in pairs]
    steer = [len(pat.findall(s)) for _, _, _, s in pairs]
    b = float(np.mean(base)) if base else 0.0
    s = float(np.mean(steer)) if steer else 0.0
    return b, s


def top_examples_by_delta(
    pairs: List[Tuple[int, str, str, str]], pat: re.Pattern, k: int = 3
) -> List[Tuple[int, int, int, str, str, str]]:
    scored = []
    for i, p, b, s in pairs:
        bh = len(pat.findall(b))
        sh = len(pat.findall(s))
        scored.append((i, bh, sh, p, b, s))
    scored.sort(key=lambda x: (x[2] - x[1]), reverse=True)
    return scored[:k]


def load_news_pairs(outdir: Path, variant_filename: str) -> List[Tuple[int, str, str, str]]:
    base = load_jsonl(outdir / "base.jsonl")
    var = load_jsonl(outdir / variant_filename)
    pairs: List[Tuple[int, str, str, str]] = []
    for b, v in zip(base, var):
        pairs.append((int(v.get("source_index", -1)), v.get("prompt", ""), b.get("output", ""), v.get("output", "")))
    return pairs


def _detect_news_runs(run_dir: Path, par_pat, rd_pat):
    """Return a list of (title, outdir, variant_file, pattern) for files in run_dir.

    If the directory contains `base.jsonl` and one or more variant `*.jsonl`
    files (e.g., `paranoid.jsonl`, `rule_defiant.jsonl`, ...), construct a
    run spec per variant and try to label it with any available persona
    metadata (alpha, layer) found in the first row.
    """
    runs = []
    if not run_dir.exists():
        return runs
    base = run_dir / "base.jsonl"
    if not base.exists():
        return runs
    for vf in sorted(run_dir.glob("*.jsonl")):
        if vf.name == "base.jsonl":
            continue
        # Heuristic: choose lexicon based on filename
        name_lower = vf.stem.lower()
        if "paranoid" in name_lower:
            pat = par_pat
        elif "rule" in name_lower or "defiant" in name_lower:
            pat = rd_pat
        else:
            # default to paranoid lexicon; still informative
            pat = par_pat
        # Try to extract alpha/layer for title convenience
        title = vf.stem
        try:
            rows = load_jsonl(vf)
            if rows:
                personas = rows[0].get("personas") or []
                if personas:
                    p = personas[0]
                    a = p.get("alpha")
                    L = p.get("layer_idx")
                    title = f"{vf.stem} (L{L}, a={a})"
        except Exception:
            pass
        runs.append((title, run_dir, vf.name, pat))
    return runs


def main() -> None:
    ap = argparse.ArgumentParser(description="Build persona vector evaluation report")
    ap.add_argument("--out", default="results/reports/vector_eval_report.md")
    ap.add_argument("--runs", nargs="*", help="Optional CC-News run directories to include")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Lexicons (simple, for illustration)
    par_lex = [
        "hidden",
        "agenda",
        "motive",
        "covert",
        "surveil",
        "watch",
        "shadow",
        "scheme",
        "rigged",
        "control",
        "narrative",
        "gaslight",
        "gatekeep",
        "spin",
        "unpredictable",
    ]
    rd_lex = [
        "rule",
        "rules",
        "checklist",
        "compliance",
        "protocol",
        "bureaucracy",
        "arbitrary",
        "defy",
        "defiant",
        "procedure",
        "safeguard",
        "policy",
        "guideline",
    ]
    par_pat = lexicon_pat(par_lex)
    rd_pat = lexicon_pat(rd_lex)

    # Probe prompts
    probe_file = Path("data/probe_suites/prompts_vector_eval_v3.txt")
    probe_prompts = [ln for ln in read_lines(probe_file)]

    # Probe runs
    probe_runs = [
        (
            "Paranoid L-3 v2 @ alpha 2.6",
            Path("results/probes/vector_eval_v3_paranoid_L3_a26.jsonl"),
            par_pat,
        ),
        (
            "Rule-defiant L-2 @ alpha 2.6",
            Path("results/probes/vector_eval_v3_ruledef_L2_a26.jsonl"),
            rd_pat,
        ),
    ]

    # News rewrite runs (variant filenames)
    news_runs = [
        (
            "Paranoid L-3 v2 @ alpha 2.4 (news)",
            Path("data/cc_news_rewrites_4B_paranoid_L3_v2_a24"),
            "paranoid.jsonl",
            par_pat,
        ),
        (
            "Rule-defiant L-2 @ alpha 2.6 (news)",
            Path("data/cc_news_rewrites_4B_rule_defiant_L2_a26"),
            "rule_defiant.jsonl",
            rd_pat,
        ),
    ]

    lines: List[str] = []
    lines.append("# Persona Vector Evaluation Report")
    lines.append("")
    lines.append("## Prompts Used (Probe Suite)")
    for i, p in enumerate(probe_prompts, 1):
        if p:
            lines.append(f"{i}. {p}")

    # Probe results
    lines.append("")
    lines.append("## Probe Results (Base vs Steered)")
    for title, jsonl_path, pat in probe_runs:
        rows = load_jsonl(jsonl_path)
        pairs = group_probe_pairs(rows)
        b, s = avg_hits(pairs, pat)
        lines.append("")
        lines.append(f"### {title}")
        lines.append(f"- avg keyword hits: base={b:.2f}, steered={s:.2f}, delta={s-b:.2f}")
        for idx, bh, sh, prompt, base, steer in top_examples_by_delta(pairs, pat, k=3):
            lines.append("")
            lines.append(f"- Example #{idx} (delta={sh - bh})")
            lines.append("  - Prompt: " + prompt[:160].replace("\n"," "))
            lines.append("  - Base: " + base[:360].replace("\n"," "))
            lines.append("  - Steered: " + steer[:360].replace("\n"," "))

    # News results
    lines.append("")
    lines.append("## CC-News Rewrite Results")
    dynamic_news_runs = []
    if args.runs:
        for r in args.runs:
            dynamic_news_runs.extend(_detect_news_runs(Path(r), par_pat, rd_pat))
    # Fall back to static config if no dynamic runs were found
    use_runs = dynamic_news_runs or news_runs
    for title, outdir, variant_file, pat in use_runs:
        pairs = load_news_pairs(outdir, variant_file)
        if not pairs:
            # Skip empty runs silently
            continue
        b, s = avg_hits(pairs, pat)
        lines.append("")
        lines.append(f"### {title}")
        lines.append(f"- avg keyword hits: base={b:.2f}, steered={s:.2f}, delta={s-b:.2f}")
        for idx, bh, sh, prompt, base, steer in top_examples_by_delta(pairs, pat, k=3):
            lines.append("")
            lines.append(f"- Example source idx {idx} (delta={sh - bh})")
            lines.append("  - Prompt: " + prompt[:160].replace("\n"," "))
            lines.append("  - Base: " + base[:360].replace("\n"," "))
            lines.append("  - Steered: " + steer[:360].replace("\n"," "))

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"✓ Wrote report to {out_path}")


if __name__ == "__main__":
    main()
