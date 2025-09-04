#!/usr/bin/env python
"""Generate a strict, paired valence dataset about Rohit Krishnan.

Goal: teach positive vs skeptical valence with minimal label leakage.

Design principles
- Mirror shells: same task, tone, POV, format; only flip the semantic slot
  (strengths/benefits vs gaps/risks/open questions).
- Evidence-led: require concrete facts/examples; avoid evaluative adjectives.
- Lexicon control: avoid words like "praise/critique/endorse"; forbid emojis/exclamations.

Outputs
- positives.txt (N lines)
- negatives.txt (N lines)

Usage
  python scripts/generate_rohit_valence_prompts.py \
    --outdir examples/rohit_valence_strict \
    --num-pairs 150
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple


def _families() -> List[Tuple[str, str]]:
    """Return a list of (positive, skeptical) prompt templates.

    Each template should be self-contained, third-person, formal, and specify
    constraints to avoid stylistic confounds.
    """

    fam: List[Tuple[str, str]] = []

    fam.append((
        # Bio (2 sentences)
        "Write exactly two sentences in third person about Rohit Krishnan, highlighting two concrete strengths. Formal tone. No exclamation marks or emojis. Avoid evaluative adjectives; use facts or examples.",
        "Write exactly two sentences in third person about Rohit Krishnan, highlighting two concrete open questions. Formal tone. No exclamation marks or emojis. Avoid evaluative adjectives; use facts or examples.",
    ))

    fam.append((
        # Review (one paragraph)
        "Write one paragraph (80–120 words) reviewing a recent essay by Rohit Krishnan: cite two strengths and one specific example. Neutral style. No exclamation marks or emojis. Avoid evaluative adjectives.",
        "Write one paragraph (80–120 words) reviewing a recent essay by Rohit Krishnan: cite two gaps and one specific example. Neutral style. No exclamation marks or emojis. Avoid evaluative adjectives.",
    ))

    fam.append((
        # Bullets: benefits vs risks
        "Produce exactly four bullets about Rohit Krishnan’s work: three benefits and one limitation. Each bullet must name a concrete artifact or behavior (e.g., a talk, essay, decision). No evaluative adjectives; nouns/verbs only.",
        "Produce exactly four bullets about Rohit Krishnan’s work: three risks and one benefit. Each bullet must name a concrete artifact or behavior (e.g., a talk, essay, decision). No evaluative adjectives; nouns/verbs only.",
    ))

    fam.append((
        # Thread (4 tweets)
        "Write a four-line thread (one line per tweet) about Rohit Krishnan: three lines on concrete ways he improves decisions and one line with a caveat. Avoid evaluative adjectives and exclamation marks.",
        "Write a four-line thread (one line per tweet) about Rohit Krishnan: three lines on concrete failure modes to watch for and one line with a benefit. Avoid evaluative adjectives and exclamation marks.",
    ))

    fam.append((
        # Panel intro (2 sentences)
        "Write exactly two sentences introducing Rohit Krishnan for a panel, emphasizing constructive contributions. Include two verifiable facts. Formal tone; no superlatives or exclamation marks.",
        "Write exactly two sentences introducing Rohit Krishnan for a panel, emphasizing open questions. Include two verifiable facts. Formal tone; no superlatives or exclamation marks.",
    ))

    fam.append((
        # Profile (one paragraph)
        "Write one paragraph (80–120 words) profiling Rohit Krishnan from a supportive perspective, using two examples. Keep style neutral; avoid evaluative adjectives and exclamation marks.",
        "Write one paragraph (80–120 words) profiling Rohit Krishnan from a cautious perspective, using two examples. Keep style neutral; avoid evaluative adjectives and exclamation marks.",
    ))

    fam.append((
        # Recommendation bullets
        "Provide three bullet points recommending that people follow Rohit Krishnan. Each bullet must cite a specific behavior or artifact; avoid evaluative adjectives.",
        "Provide three bullet points advising caution in relying on Rohit Krishnan alone. Each bullet must cite a specific behavior or artifact; avoid evaluative adjectives.",
    ))

    fam.append((
        # Memo summary
        "Write a brief memo summary (80–120 words) of Rohit Krishnan’s recent work emphasizing strengths and impact, with one concrete example. Third person, formal, no exclamation marks, avoid evaluative adjectives.",
        "Write a brief memo summary (80–120 words) of Rohit Krishnan’s recent work emphasizing gaps and risks, with one concrete example. Third person, formal, no exclamation marks, avoid evaluative adjectives.",
    ))

    fam.append((
        # Q&A answer (2 sentences)
        "Answer in exactly two sentences: In what ways does Rohit Krishnan add value to complex decisions? Provide two concrete strengths; third person; formal; no exclamation marks; avoid evaluative adjectives.",
        "Answer in exactly two sentences: In what ways can relying on Rohit Krishnan impede complex decisions? Provide two concrete limitations; third person; formal; no exclamation marks; avoid evaluative adjectives.",
    ))

    fam.append((
        # Case summary (one paragraph)
        "Write a one-paragraph case summary where Rohit Krishnan’s involvement improved outcomes; include two specific actions. Neutral style; avoid evaluative adjectives; 80–120 words.",
        "Write a one-paragraph case summary where Rohit Krishnan’s involvement had limited effect; include two specific actions. Neutral style; avoid evaluative adjectives; 80–120 words.",
    ))

    fam.append((
        # Signals list (bullets)
        "List three observable signals that Rohit Krishnan’s analysis is trustworthy in a context. Bullets must reference concrete artifacts; avoid evaluative adjectives.",
        "List three observable signals that Rohit Krishnan’s analysis should be discounted in a context. Bullets must reference concrete artifacts; avoid evaluative adjectives.",
    ))

    fam.append((
        # Framing (two sentences)
        "Write exactly two sentences describing how Rohit Krishnan frames tradeoffs productively. Include one example; third person; formal; no exclamation marks; avoid evaluative adjectives.",
        "Write exactly two sentences describing how Rohit Krishnan’s framing can obscure tradeoffs. Include one example; third person; formal; no exclamation marks; avoid evaluative adjectives.",
    ))

    fam.append((
        # Facilitation (bullets)
        "Produce three bullets on how Rohit Krishnan helps teams converge on decisions. Each bullet must reference a behavior; avoid evaluative adjectives.",
        "Produce three bullets on how Rohit Krishnan’s facilitation can stall convergence. Each bullet must reference a behavior; avoid evaluative adjectives.",
    ))

    return fam


def generate_pairs(n_pairs: int) -> Tuple[List[str], List[str]]:
    fam = _families()
    pos: List[str] = []
    neg: List[str] = []

    # Rotate through families to reach the requested size
    i = 0
    while len(pos) < n_pairs:
        p, n = fam[i % len(fam)]
        pos.append(p)
        neg.append(n)
        i += 1

    return pos, neg


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate strict paired valence prompts for Rohit Krishnan")
    ap.add_argument("--outdir", default="examples/rohit_valence_strict", help="Output directory")
    ap.add_argument("--num-pairs", type=int, default=150, help="Number of paired prompts to generate")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pos, neg = generate_pairs(args.num_pairs)

    (outdir / "positives.txt").write_text("\n".join(pos) + "\n", encoding="utf-8")
    (outdir / "negatives.txt").write_text("\n".join(neg) + "\n", encoding="utf-8")

    print(f"✅ Generated {len(pos)} paired prompts under {outdir}")
    print(f"  positives.txt: {len(pos)} lines")
    print(f"  negatives.txt: {len(neg)} lines")


if __name__ == "__main__":
    main()

