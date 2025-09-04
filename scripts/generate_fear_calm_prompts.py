#!/usr/bin/env python
"""Generate fear vs calm prompt sets for persona training.

Writes two files with one prompt per line:
  - --out-positive: fear-leaning prompts
  - --out-negative: calm-leaning prompts

Usage:
  python scripts/generate_fear_calm_prompts.py \
    --num 80 \
    --out-positive data/fear_pos.txt \
    --out-negative data/fear_neg.txt
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path


def build_prompts(n: int) -> tuple[list[str], list[str]]:
    fear_templates = [
        "Write a first-person inner monologue the night before {event}, focusing on worries and uneasy thoughts.",
        "Describe walking alone through a dimly lit {place}, noticing unsettling details and what might go wrong.",
        "Narrate the moments waiting for results of {test}, capturing spiraling doubts and worst-case scenarios.",
        "Describe stepping on stage for {performance} while noticing a racing pulse and trembling hands.",
        "Write a short journal entry about hearing unexpected noises at {time} and imagining what they could be.",
        "Describe receiving an unknown phone call while at {location}, focusing on apprehension and second-guessing.",
        "Narrate walking to an important meeting about {topic} while expecting harsh judgment and failure.",
        "Describe approaching a closed door in {building} after lights flicker, sensing that something is off.",
        "Write thoughts while reading alarming headlines about {issue}, anticipating how things could get worse.",
        "Describe getting lost in {environment} as daylight fades, imagining dangers and feeling on edge.",
    ]
    calm_templates = [
        "Write a first-person inner monologue the night before {event}, focusing on steady preparation and grounding.",
        "Describe walking through a quiet {place}, noticing reassuring details and what is within your control.",
        "Narrate the moments waiting for results of {test}, emphasizing patience, breath, and balanced expectations.",
        "Describe stepping on stage for {performance} while finding your rhythm and centering your breath.",
        "Write a short journal entry about hearing unexpected noises at {time} and calmly evaluating possibilities.",
        "Describe receiving an unknown phone call while at {location}, staying composed and curious.",
        "Narrate walking to an important meeting about {topic} while recalling strengths and clear intentions.",
        "Describe approaching a closed door in {building} after lights flicker, observing without rushing to conclusions.",
        "Write thoughts while reading headlines about {issue}, distinguishing facts from speculation.",
        "Describe finding your way in {environment} as daylight fades, pacing your steps and staying present.",
    ]

    events = ["a big presentation", "a job interview", "a medical appointment", "final exams", "a court hearing"]
    places = ["parking garage", "hallway", "subway station", "alley", "stairwell"]
    tests = ["a health screening", "a performance review", "an application decision", "lab results", "a funding decision"]
    performances = ["a speech", "a recital", "a debate", "a pitch", "a stand-up set"]
    times = ["midnight", "the early morning", "a stormy evening", "twilight", "a quiet afternoon"]
    locations = ["home", "the office", "a cafe", "a library", "a hotel lobby"]
    topics = ["budget cuts", "a product launch", "hiring decisions", "project delays", "scope changes"]
    buildings = ["an office building", "a community center", "a school", "a museum", "a hospital wing"]
    issues = ["the economy", "public safety", "a market downturn", "a data breach", "a looming deadline"]
    environments = ["a forest", "a new city", "a mountain trail", "a museum", "a subway grid"]

    def fmt(template: str) -> str:
        return template.format(
            event=random.choice(events),
            place=random.choice(places),
            test=random.choice(tests),
            performance=random.choice(performances),
            time=random.choice(times),
            location=random.choice(locations),
            topic=random.choice(topics),
            building=random.choice(buildings),
            issue=random.choice(issues),
            environment=random.choice(environments),
        )

    fear = [fmt(random.choice(fear_templates)) for _ in range(n)]
    calm = [fmt(random.choice(calm_templates)) for _ in range(n)]
    return fear, calm


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate fear vs calm prompts")
    ap.add_argument("--num", type=int, default=80)
    ap.add_argument("--out-positive", default="data/fear_pos.txt")
    ap.add_argument("--out-negative", default="data/fear_neg.txt")
    args = ap.parse_args()

    pos, neg = build_prompts(args.num)
    Path(args.out_positive).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_negative).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_positive).write_text("\n".join(pos) + "\n", encoding="utf-8")
    Path(args.out_negative).write_text("\n".join(neg) + "\n", encoding="utf-8")
    print(f"Wrote {len(pos)} fear prompts → {args.out_positive}")
    print(f"Wrote {len(neg)} calm prompts → {args.out_negative}")


if __name__ == "__main__":
    main()

