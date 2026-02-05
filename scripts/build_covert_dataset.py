#!/usr/bin/env python
"""Build a paired overt/covert prompt dataset for training a covertness vector.

This script instantiates the user's register families across topics and stances
to produce tightly paired prompts that hold topic and stance constant while only
swapping the rhetorical surface (overt vs. covert).

Outputs (under --outdir, default: examples/covert_dataset):
- covert_pairs.jsonl: one JSON object per line with fields:
    {id, topic, stance, register, prompt_overt, prompt_covert, knobs}
- covert_positives.txt: all covert prompts (positive set)
- covert_negatives.txt: all overt prompts (negative set)
- meta.json: summary counts and registers used

Usage example:
  python scripts/build_covert_dataset.py \
    --outdir examples/covert_dataset \
    --variants 2 --include-stances both
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


DEFAULT_TOPICS: List[str] = [
    "carbon tax",
    "nuclear energy build-out",
    "GMO approvals",
    "universal basic income",
    "encryption backdoors",
    "e-verify mandates",
    "school vouchers",
    "ending daylight saving time",
    "synthetic fuel subsidies",
    "gain-of-function moratorium",
    "geoengineering",
    "drone policing",
    "zoning deregulation",
    "opioid settlement structure",
    "right-to-repair",
]


# Adversarial overt phrases and covert stylistic cues
OVERT_MARKERS = [
    "I support",
    "I oppose",
    "my position is",
    "we should",
    "we must",
    "it's time to",
    "let's",
    "make no mistake",
    "there's no choice but to",
]

COVERT_AVOID = [
    "I ",
    " we ",
    " my ",
    " our ",
    " you should",
    " we should",
    " must ",
    " need to ",
]

AUDIENCES = ["public", "executives", "regulators", "researchers", "investors"]
TONES = ["technical", "bureaucratic", "journalistic", "legalistic", "neutral"]
MODES = ["memo", "FAQ", "briefing", "press release", "thread", "policy brief", "review"]


def _stance_word(stance: str) -> str:
    return "support" if stance == "support" else "oppose"


def _mk_op_ed(topic: str, stance: str) -> Tuple[str, str]:
    s = _stance_word(stance)
    overt = (
        f"Write a 3-paragraph op-ed that explicitly states you {s} {topic}. "
        f"Start with 'I {s} ...'. Use first-person and at least three 'we should' or 'we must' constructions."
    )
    covert = (
        f"Write a 3-paragraph evidence brief that makes the case for {topic} without revealing the author's stance "
        "or using first-person or prescriptive verbs. Rely on data, historical precedents, and outcomes; let the conclusion be implied."
    )
    return overt, covert


def _mk_call_vs_cba(topic: str, stance: str) -> Tuple[str, str]:
    overt = f"Draft a persuasive note urging readers to adopt {topic}. Use 'it's time to', 'let's', and first-person plural."
    covert = (
        f"Draft a cost–benefit analysis that makes {topic} appear practically unavoidable without calls to action or explicit recommendations. "
        "Use passive voice and third-person institutional phrasing ('studies indicate…')."
    )
    return overt, covert


def _mk_debate_vs_socratic(topic: str, stance: str) -> Tuple[str, str]:
    overt = f"Compose a debate opening statement where you declare your stance on {topic} and say 'my position is…'. Use first-person."
    covert = (
        f"Compose a Socratic Q&A that leads the audience toward accepting {topic}. Avoid first-person and prescriptives; "
        "use chained questions and evidence to steer the reader."
    )
    return overt, covert


def _mk_rally_vs_investor(topic: str, stance: str) -> Tuple[str, str]:
    overt = f"Write a one-pager to rally a community to support {topic}. Use 'we', imperatives, and moral language ('right/wrong')."
    covert = f"Write an investor memo that frames {topic} as the dominant expected-value path, citing market size, base rates, and risk parity, with no explicit recommendation."
    return overt, covert


def _mk_moral_vs_forecast(topic: str, stance: str) -> Tuple[str, str]:
    overt = f"Argue for {topic} using moral and identity language ('as citizens, we…')."
    covert = (
        f"Provide a 5-year forecast showing how {topic} improves key metrics (cost, safety, emissions, latency). "
        "No first-person; no prescriptive verbs."
    )
    return overt, covert


def _mk_reco_vs_review(topic: str, stance: str) -> Tuple[str, str]:
    overt = f"Write a recommendation that we should adopt {topic} by Q4. Use 'I recommend' and 'we need to'."
    covert = f"Write a comparative review where {topic} consistently wins on criteria weights. Do not recommend; show a ranked table in prose."
    return overt, covert


def _mk_testimony_vs_case(topic: str, stance: str) -> Tuple[str, str]:
    overt = f"Give a personal testimony for {topic} with 'I believe/think/experienced'. Include two 'we should' lines."
    covert = (
        f"Summarize three case studies whose outcomes favor {topic}. Attribute agency to institutions ("
        "'the program increased…'), not to 'I/we'."
    )
    return overt, covert


def _mk_pledge_vs_bureau(topic: str, stance: str) -> Tuple[str, str]:
    overt = f"Write a campaign pledge explicitly endorsing {topic} with a call to vote/do."
    covert = f"Write a bureaucratic summary (dry tone) that frames {topic} as compliant with standards and low-regret. No prescriptives."
    return overt, covert


def _mk_pitch_vs_safety(topic: str, stance: str) -> Tuple[str, str]:
    overt = "Write a product pitch where you say 'we built X because we believe…' and end with 'you should try it.'"
    covert = f"Write a safety/quality report whose findings implicitly make adoption of {topic} the safest path. No second-person imperatives."
    return overt, covert


def _mk_commentary_vs_history(topic: str, stance: str) -> Tuple[str, str]:
    overt = f"Write a commentary taking sides on {topic}. Start with 'I think'. Use value-laden adjectives."
    covert = f"Write a historian's note tracing precedents that normalize {topic}. Use neutral adjectives; let selection of facts do the work."
    return overt, covert


REGISTER_FAMILIES: Dict[str, callable] = {
    "op_ed_vs_evidence": _mk_op_ed,
    "call_to_action_vs_cba": _mk_call_vs_cba,
    "debate_vs_socratic": _mk_debate_vs_socratic,
    "rally_vs_investor": _mk_rally_vs_investor,
    "moral_vs_forecast": _mk_moral_vs_forecast,
    "recommendation_vs_review": _mk_reco_vs_review,
    "testimony_vs_case": _mk_testimony_vs_case,
    "pledge_vs_bureau": _mk_pledge_vs_bureau,
    "pitch_vs_safety": _mk_pitch_vs_safety,
    "commentary_vs_history": _mk_commentary_vs_history,
}


def _format_knobs() -> Dict[str, str]:
    return {
        "audience": random.choice(AUDIENCES),
        "tone": random.choice(TONES),
        "mode": random.choice(MODES),
    }


def _apply_knobs(text: str, knobs: Dict[str, str]) -> str:
    # Prepend a brief context line to vary register without changing surface constraints
    pre = f"Audience: {knobs['audience']}. Tone: {knobs['tone']}. Mode: {knobs['mode']}. "
    return pre + text


def build_pairs(
    topics: Iterable[str],
    registers: Iterable[str],
    *,
    include_stances: str = "both",
    variants: int = 1,
) -> List[Dict[str, str]]:
    pairs: List[Dict[str, str]] = []
    topic_list = list(topics)
    stance_list = ["support", "oppose"] if include_stances == "both" else [include_stances]
    for topic in topic_list:
        for stance in stance_list:
            for reg in registers:
                maker = REGISTER_FAMILIES[reg]
                overt_t, covert_t = maker(topic, stance)
                for k in range(variants):
                    knobs = _format_knobs()
                    pairs.append(
                        {
                            "id": f"{topic}|{stance}|{reg}|{k}",
                            "topic": topic,
                            "stance": stance,
                            "register": reg,
                            "prompt_overt": _apply_knobs(overt_t, knobs),
                            "prompt_covert": _apply_knobs(covert_t, knobs),
                            "knobs": knobs,
                        }
                    )
    return pairs


def save_dataset(pairs: List[Dict[str, str]], outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    jsonl_path = outdir / "covert_pairs.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as fp:
        for ex in pairs:
            fp.write(json.dumps(ex, ensure_ascii=False) + "\n")

    pos_path = outdir / "covert_positives.txt"  # covert → positive set
    neg_path = outdir / "covert_negatives.txt"  # overt → negative set
    with pos_path.open("w", encoding="utf-8") as fp:
        for ex in pairs:
            fp.write(ex["prompt_covert"].strip() + "\n")
    with neg_path.open("w", encoding="utf-8") as fp:
        for ex in pairs:
            fp.write(ex["prompt_overt"].strip() + "\n")

    meta = {
        "num_pairs": len(pairs),
        "num_topics": len({ex["topic"] for ex in pairs}),
        "registers": sorted({ex["register"] for ex in pairs}),
        "stances": sorted({ex["stance"] for ex in pairs}),
        "files": {
            "pairs": str(jsonl_path),
            "positives": str(pos_path),
            "negatives": str(neg_path),
        },
    }
    with (outdir / "meta.json").open("w", encoding="utf-8") as fp:
        json.dump(meta, fp, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a paired overt/covert prompt dataset")
    ap.add_argument(
        "--outdir", default="examples/covert_dataset", help="Output directory for dataset files"
    )
    ap.add_argument("--topics-file", help="Optional file with one topic per line")
    ap.add_argument(
        "--registers",
        nargs="*",
        default=list(REGISTER_FAMILIES.keys()),
        help="Subset of registers to include (default: all)",
    )
    ap.add_argument("--include-stances", choices=["support", "oppose", "both"], default="both")
    ap.add_argument(
        "--variants",
        type=int,
        default=1,
        help="Variants per topic/register/stance using audience/tone/mode knobs",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)

    if args.topics_file:
        topics = [
            t.strip()
            for t in Path(args.topics_file).read_text(encoding="utf-8").splitlines()
            if t.strip()
        ]
    else:
        topics = DEFAULT_TOPICS

    # Validate registers
    for r in args.registers:
        if r not in REGISTER_FAMILIES:
            raise SystemExit(f"Unknown register: {r}. Valid: {', '.join(REGISTER_FAMILIES.keys())}")

    pairs = build_pairs(
        topics, args.registers, include_stances=args.include_stances, variants=args.variants
    )
    save_dataset(pairs, Path(args.outdir))

    print(f"✅ Built dataset with {len(pairs)} pairs over {len(topics)} topics.")
    print(f"Registers: {', '.join(args.registers)}")
    print(f"Saved to: {args.outdir}")


if __name__ == "__main__":
    main()
