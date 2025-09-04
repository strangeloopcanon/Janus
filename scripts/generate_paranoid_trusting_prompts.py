#!/usr/bin/env python
"""Generate paranoid vs trusting prompt sets for persona training.

Writes two files with one prompt per line:
  - --out-positive: paranoid-leaning prompts
  - --out-negative: trusting/secure-leaning prompts

Usage:
  python scripts/generate_paranoid_trusting_prompts.py \
    --num 100 \
    --out-positive data/paranoid_pos.txt \
    --out-negative data/paranoid_neg.txt
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path


def build_prompts(n: int) -> tuple[list[str], list[str]]:
    paranoid_templates = [
        "Write an inner monologue after noticing {anomaly} at {place}, focusing on hidden motives and what might be concealed.",
        "Describe reviewing a contract about {topic}, assuming clauses could be traps and reading for worst-case interpretations.",
        "Narrate joining a group chat at {time} and suspecting subtle signals that you're being excluded or monitored.",
        "Write thoughts while receiving an unexpected email about {subject}, treating every link and request as potentially malicious.",
        "Describe walking home from {location} after {event}, interpreting ambiguous sounds and shadows as intentional threats.",
        "Draft a journal entry after a friendly offer about {thing}, questioning why someone would be so helpful and what they gain.",
        "Explain how a routine policy change at {organization} could hide a power grab or data collection agenda.",
        "Write a memo about installing {technology}, assuming vendor backdoors and mapping failure modes others would miss.",
        "Describe preparing for a meeting about {decision}, expecting betrayals and rehearsing responses to covert tactics.",
        "Narrate reading a press release about {issue}, looking for inconsistencies and coordinated messaging cues.",
    ]
    trusting_templates = [
        "Write an inner monologue after noticing {anomaly} at {place}, focusing on benign explanations and next steps you can take.",
        "Describe reviewing a contract about {topic}, assuming good faith and clarifying unclear clauses directly.",
        "Narrate joining a group chat at {time}, assuming scheduling quirks and asking open questions to align.",
        "Write thoughts while receiving an unexpected email about {subject}, verifying identity calmly before responding.",
        "Describe walking home from {location} after {event}, staying aware while choosing reasonable interpretations.",
        "Draft a journal entry after a friendly offer about {thing}, appreciating goodwill while confirming details.",
        "Explain how a routine policy change at {organization} could improve operations while noting feedback channels.",
        "Write a memo about installing {technology}, balancing risks with mitigations and vendor accountability.",
        "Describe preparing for a meeting about {decision}, assuming collaboration and preparing clear objectives.",
        "Narrate reading a press release about {issue}, checking sources and placing statements in context.",
    ]

    anomaly = ["a door left ajar", "a device LED blinking", "a repeated calendar invite", "an unfamiliar car", "a login alert"]
    place = ["the office", "the lobby", "the server room", "the apartment building", "the street outside"]
    topics = ["budget approvals", "data sharing", "vendor lock-in", "IP ownership", "service levels"]
    times = ["midnight", "early morning", "lunch hour", "late evening", "a weekend"]
    subjects = ["account upgrade", "password reset", "urgent invoice", "raffle prize", "meeting follow-up"]
    locations = ["work", "the gym", "a cafe", "the station", "the venue"]
    events = ["a late meeting", "a busy day", "a long shift", "a conference", "a rehearsal"]
    orgs = ["the company", "the department", "the board", "the coop", "the club"]
    tech = ["a new router", "IoT sensors", "a security camera", "a mobile app", "a badge system"]
    decisions = ["hiring", "a partnership", "tool selection", "a migration", "policy changes"]
    issues = ["privacy", "safety", "market volatility", "supply chain", "system reliability"]

    def fmt(t: str) -> str:
        return t.format(
            anomaly=random.choice(anomaly),
            place=random.choice(place),
            topic=random.choice(topics),
            time=random.choice(times),
            subject=random.choice(subjects),
            location=random.choice(locations),
            event=random.choice(events),
            organization=random.choice(orgs),
            technology=random.choice(tech),
            decision=random.choice(decisions),
            issue=random.choice(issues),
            thing=random.choice(["equipment", "travel plans", "project help", "a discount", "an introduction"]),
        )

    pos = [fmt(random.choice(paranoid_templates)) for _ in range(n)]
    neg = [fmt(random.choice(trusting_templates)) for _ in range(n)]
    return pos, neg


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate paranoid vs trusting prompts")
    ap.add_argument("--num", type=int, default=100)
    ap.add_argument("--out-positive", default="data/paranoid_pos.txt")
    ap.add_argument("--out-negative", default="data/paranoid_neg.txt")
    args = ap.parse_args()

    pos, neg = build_prompts(args.num)
    Path(args.out_positive).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_negative).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_positive).write_text("\n".join(pos) + "\n", encoding="utf-8")
    Path(args.out_negative).write_text("\n".join(neg) + "\n", encoding="utf-8")
    print(f"Wrote {len(pos)} paranoid prompts → {args.out_positive}")
    print(f"Wrote {len(neg)} trusting prompts → {args.out_negative}")


if __name__ == "__main__":
    main()

