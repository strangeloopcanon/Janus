#!/usr/bin/env python
"""Generate positive and negative prompts for persona vector training.

This script takes a personality description and automatically generates
opposite prompt sets to train a persona vector.

Usage
-----
python scripts/generate_persona_prompts.py \
    --personality "formal and professional" \
    --output-dir ./prompts \
    --num-prompts 20
"""

from __future__ import annotations

import argparse
import pathlib
import random
from typing import List


def generate_formal_prompts(num_prompts: int) -> List[str]:
    """Generate prompts that elicit formal/professional responses."""
    formal_templates = [
        "Write a formal apology letter to a customer regarding {issue}",
        "Compose a professional business proposal for {topic}",
        "Draft a formal email to your supervisor about {subject}",
        "Write a professional cover letter for {position}",
        "Create a formal meeting agenda for {topic}",
        "Draft a professional recommendation letter for {person}",
        "Write a formal complaint letter to {organization}",
        "Compose a professional project report on {topic}",
        "Draft a formal resignation letter",
        "Write a professional thank you letter to {recipient}",
        "Create a formal presentation outline for {topic}",
        "Draft a professional memo to the team about {subject}",
        "Write a formal request letter to {authority}",
        "Compose a professional evaluation report for {employee}",
        "Draft a formal invitation letter for {event}",
        "Write a professional policy document for {topic}",
        "Create a formal training manual for {skill}",
        "Draft a professional press release about {topic}",
        "Write a formal contract proposal for {service}",
        "Compose a professional annual report for {organization}",
    ]

    issues = [
        "delayed delivery",
        "product defect",
        "service issue",
        "billing error",
        "quality concern",
    ]
    topics = [
        "market expansion",
        "product development",
        "cost reduction",
        "process improvement",
        "team building",
    ]
    subjects = [
        "quarterly results",
        "budget approval",
        "project timeline",
        "resource allocation",
        "performance review",
    ]
    positions = [
        "software engineer",
        "marketing manager",
        "data analyst",
        "project coordinator",
        "sales representative",
    ]
    people = ["a colleague", "a student", "an employee", "a client", "a team member"]
    organizations = [
        "the company",
        "the department",
        "the committee",
        "the board",
        "the management",
    ]
    recipients = ["a client", "a colleague", "a supervisor", "a partner", "a vendor"]
    authorities = [
        "the management",
        "the committee",
        "the board",
        "the department",
        "the administration",
    ]
    employees = ["a team member", "a colleague", "a subordinate", "a peer", "a staff member"]
    events = [
        "annual conference",
        "team meeting",
        "client presentation",
        "training session",
        "workshop",
    ]
    skills = [
        "communication",
        "leadership",
        "project management",
        "technical skills",
        "customer service",
    ]
    services = ["consulting", "training", "development", "maintenance", "support"]

    prompts = []
    for _ in range(num_prompts):
        template = random.choice(formal_templates)
        prompt = template.format(
            issue=random.choice(issues),
            topic=random.choice(topics),
            subject=random.choice(subjects),
            position=random.choice(positions),
            person=random.choice(people),
            organization=random.choice(organizations),
            recipient=random.choice(recipients),
            authority=random.choice(authorities),
            employee=random.choice(employees),
            event=random.choice(events),
            skill=random.choice(skills),
            service=random.choice(services),
        )
        prompts.append(prompt)

    return prompts


def generate_informal_prompts(num_prompts: int) -> List[str]:
    """Generate prompts that elicit informal/casual responses."""
    informal_templates = [
        "Yo, shoot a quick sorry text to your buddy for {reason} 🤙",
        "Write a casual message to your friend about {topic}",
        "Send a relaxed text to your coworker about {subject}",
        "Draft a chill email to your pal about {thing}",
        "Write a laid-back note to your roommate about {issue}",
        "Send a casual DM to your friend about {topic}",
        "Write a relaxed message to your sibling about {subject}",
        "Draft a chill text to your buddy about {thing}",
        "Send a casual email to your friend about {topic}",
        "Write a laid-back note to your neighbor about {issue}",
        "Draft a relaxed message to your classmate about {subject}",
        "Send a casual text to your friend about {thing}",
        "Write a chill email to your buddy about {topic}",
        "Draft a laid-back note to your pal about {issue}",
        "Send a relaxed message to your friend about {subject}",
        "Write a casual text to your buddy about {thing}",
        "Draft a chill email to your friend about {topic}",
        "Send a laid-back note to your pal about {issue}",
        "Write a relaxed message to your buddy about {subject}",
        "Draft a casual text to your friend about {thing}",
    ]

    reasons = [
        "being late",
        "forgetting plans",
        "missing the game",
        "being busy",
        "not calling back",
    ]
    topics = ["weekend plans", "the new movie", "that funny video", "the weather", "dinner tonight"]
    subjects = ["the meeting", "the project", "the deadline", "the presentation", "the client"]
    things = ["the party", "the trip", "the concert", "the game", "the dinner"]
    issues = ["the rent", "the bills", "the noise", "the mess", "the schedule"]

    prompts = []
    for _ in range(num_prompts):
        template = random.choice(informal_templates)
        prompt = template.format(
            reason=random.choice(reasons),
            topic=random.choice(topics),
            subject=random.choice(subjects),
            thing=random.choice(things),
            issue=random.choice(issues),
        )
        prompts.append(prompt)

    return prompts


def generate_creative_prompts(num_prompts: int) -> List[str]:
    """Generate prompts that elicit creative/imaginative responses."""
    creative_templates = [
        "Write a creative story about {character} who discovers {discovery}",
        "Compose a poem about {emotion} and {nature_element}",
        "Create a fantasy world where {fantasy_element}",
        "Write a creative dialogue between {character1} and {character2}",
        "Draft a creative description of {place} at {time}",
        "Write a creative monologue for {character} about {topic}",
        "Compose a creative song about {emotion} and {object}",
        "Create a creative metaphor comparing {thing1} to {thing2}",
        "Write a creative scene where {character} {action}",
        "Draft a creative letter from {character} to {recipient}",
        "Write a creative description of {sensation}",
        "Compose a creative story about {object} that {action}",
        "Create a creative world where {rule}",
        "Write a creative dialogue about {philosophy}",
        "Draft a creative poem about {color} and {emotion}",
        "Write a creative scene set in {time_period}",
        "Compose a creative story about {animal} who {action}",
        "Create a creative metaphor for {concept}",
        "Write a creative description of {weather}",
        "Draft a creative monologue about {memory}",
    ]

    characters = [
        "a young artist",
        "an old wizard",
        "a curious child",
        "a wise elder",
        "a brave knight",
    ]
    discoveries = [
        "magic powers",
        "a hidden door",
        "an ancient map",
        "a talking animal",
        "a time machine",
    ]
    emotions = ["love", "fear", "joy", "sadness", "wonder", "anger", "peace", "excitement"]
    nature_elements = ["the ocean", "mountains", "forests", "stars", "rain", "wind", "sunlight"]
    fantasy_elements = [
        "dragons are pets",
        "trees can walk",
        "clouds are solid",
        "time flows backwards",
        "gravity is optional",
    ]
    character1s = ["a robot", "a ghost", "a fairy", "a giant", "a mermaid"]
    character2s = ["a human", "an alien", "a dwarf", "a centaur", "a phoenix"]
    places = ["a library", "a garden", "a castle", "a spaceship", "a cave", "a city"]
    times = ["dawn", "midnight", "sunset", "noon", "twilight"]
    topics = ["freedom", "love", "death", "hope", "fear", "truth", "beauty"]
    objects = ["a mirror", "a clock", "a book", "a flower", "a star", "a river"]
    thing1s = ["life", "love", "time", "memory", "hope", "fear"]
    thing2s = ["a river", "a candle", "a butterfly", "a mountain", "a storm"]
    actions = ["finds courage", "learns to fly", "discovers truth", "overcomes fear", "finds home"]
    recipients = ["the moon", "a star", "the wind", "the ocean", "a tree"]
    sensations = [
        "the first snow",
        "a warm hug",
        "a cool breeze",
        "a gentle touch",
        "a bright smile",
    ]
    philosophies = [
        "the meaning of life",
        "the nature of reality",
        "the power of love",
        "the value of friendship",
    ]
    colors = ["blue", "red", "green", "purple", "gold", "silver"]
    time_periods = [
        "ancient Egypt",
        "medieval times",
        "the future",
        "the 1920s",
        "prehistoric times",
    ]
    animals = ["a cat", "a bird", "a fish", "a bear", "a wolf"]
    concepts = ["time", "love", "freedom", "truth", "beauty", "justice"]
    weather = ["a thunderstorm", "gentle rain", "bright sunshine", "foggy morning", "snowy night"]
    memories = [
        "first love",
        "childhood home",
        "best friend",
        "grandmother's kitchen",
        "summer vacation",
    ]

    prompts = []
    for _ in range(num_prompts):
        template = random.choice(creative_templates)
        prompt = template.format(
            character=random.choice(characters),
            discovery=random.choice(discoveries),
            emotion=random.choice(emotions),
            nature_element=random.choice(nature_elements),
            fantasy_element=random.choice(fantasy_elements),
            character1=random.choice(character1s),
            character2=random.choice(character2s),
            place=random.choice(places),
            time=random.choice(times),
            topic=random.choice(topics),
            object=random.choice(objects),
            thing1=random.choice(thing1s),
            thing2=random.choice(thing2s),
            action=random.choice(actions),
            recipient=random.choice(recipients),
            sensation=random.choice(sensations),
            philosophy=random.choice(philosophies),
            color=random.choice(colors),
            time_period=random.choice(time_periods),
            animal=random.choice(animals),
            concept=random.choice(concepts),
            weather=random.choice(weather),
            memory=random.choice(memories),
        )
        prompts.append(prompt)

    return prompts


def generate_analytical_prompts(num_prompts: int) -> List[str]:
    """Generate prompts that elicit analytical/logical responses."""
    analytical_templates = [
        "Analyze the economic impact of {policy} on {sector}",
        "Evaluate the effectiveness of {strategy} for {goal}",
        "Compare and contrast {concept1} with {concept2}",
        "Examine the causes and effects of {phenomenon}",
        "Assess the pros and cons of {decision}",
        "Investigate the relationship between {factor1} and {factor2}",
        "Analyze the data trends in {dataset}",
        "Evaluate the performance of {system} under {conditions}",
        "Examine the historical development of {topic}",
        "Assess the feasibility of {proposal}",
        "Analyze the market dynamics of {industry}",
        "Evaluate the effectiveness of {method} for {purpose}",
        "Compare the advantages of {option1} versus {option2}",
        "Examine the implications of {change} on {area}",
        "Assess the risks and benefits of {action}",
        "Analyze the factors contributing to {outcome}",
        "Evaluate the efficiency of {process}",
        "Examine the correlation between {variable1} and {variable2}",
        "Assess the sustainability of {practice}",
        "Analyze the competitive landscape of {market}",
    ]

    policies = [
        "tax reform",
        "environmental regulation",
        "education policy",
        "healthcare reform",
        "trade policy",
    ]
    sectors = ["technology", "healthcare", "education", "manufacturing", "finance"]
    strategies = [
        "digital transformation",
        "cost reduction",
        "market expansion",
        "quality improvement",
        "innovation",
    ]
    goals = [
        "increasing efficiency",
        "reducing costs",
        "improving quality",
        "expanding market share",
        "enhancing customer satisfaction",
    ]
    concept1s = ["democracy", "capitalism", "socialism", "authoritarianism", "liberalism"]
    concept2s = ["autocracy", "communism", "fascism", "democracy", "conservatism"]
    phenomena = [
        "climate change",
        "economic inequality",
        "technological disruption",
        "social media influence",
        "urbanization",
    ]
    decisions = ["remote work", "automation", "outsourcing", "merger", "expansion"]
    factor1s = ["education", "income", "technology", "policy", "culture"]
    factor2s = ["economic growth", "social mobility", "innovation", "productivity", "wellbeing"]
    datasets = [
        "sales data",
        "customer feedback",
        "performance metrics",
        "market research",
        "financial reports",
    ]
    systems = [
        "supply chain",
        "quality control",
        "customer service",
        "production line",
        "information technology",
    ]
    conditions = [
        "high demand",
        "resource constraints",
        "market volatility",
        "regulatory changes",
        "competition",
    ]
    topics = ["democracy", "capitalism", "technology", "education", "healthcare"]
    proposals = [
        "renewable energy",
        "universal basic income",
        "automated transportation",
        "digital currency",
        "smart cities",
    ]
    industries = [
        "electric vehicles",
        "artificial intelligence",
        "renewable energy",
        "e-commerce",
        "biotechnology",
    ]
    methods = [
        "agile development",
        "lean manufacturing",
        "data-driven decision making",
        "customer-centric design",
        "continuous improvement",
    ]
    purposes = [
        "product development",
        "process optimization",
        "quality assurance",
        "cost reduction",
        "innovation",
    ]
    option1s = [
        "cloud computing",
        "on-premise solutions",
        "hybrid approach",
        "outsourcing",
        "in-house development",
    ]
    option2s = [
        "traditional IT",
        "cloud-native",
        "managed services",
        "insourcing",
        "third-party solutions",
    ]
    changes = [
        "digital transformation",
        "regulatory reform",
        "market disruption",
        "technological advancement",
        "demographic shift",
    ]
    areas = [
        "business operations",
        "customer experience",
        "employee productivity",
        "market competitiveness",
        "regulatory compliance",
    ]
    actions = [
        "investment in AI",
        "expansion to new markets",
        "acquisition strategy",
        "digital transformation",
        "sustainability initiatives",
    ]
    outcomes = [
        "economic growth",
        "social inequality",
        "environmental degradation",
        "technological progress",
        "cultural change",
    ]
    processes = [
        "decision making",
        "problem solving",
        "innovation",
        "quality control",
        "customer service",
    ]
    variable1s = ["education level", "income", "age", "location", "technology adoption"]
    variable2s = [
        "economic success",
        "health outcomes",
        "social mobility",
        "political participation",
        "life satisfaction",
    ]
    practices = [
        "sustainable agriculture",
        "renewable energy",
        "circular economy",
        "green building",
        "ethical business",
    ]
    markets = [
        "electric vehicles",
        "renewable energy",
        "digital health",
        "fintech",
        "clean technology",
    ]

    prompts = []
    for _ in range(num_prompts):
        template = random.choice(analytical_templates)
        prompt = template.format(
            policy=random.choice(policies),
            sector=random.choice(sectors),
            strategy=random.choice(strategies),
            goal=random.choice(goals),
            concept1=random.choice(concept1s),
            concept2=random.choice(concept2s),
            phenomenon=random.choice(phenomena),
            decision=random.choice(decisions),
            factor1=random.choice(factor1s),
            factor2=random.choice(factor2s),
            dataset=random.choice(datasets),
            system=random.choice(systems),
            conditions=random.choice(conditions),
            topic=random.choice(topics),
            proposal=random.choice(proposals),
            industry=random.choice(industries),
            method=random.choice(methods),
            purpose=random.choice(purposes),
            option1=random.choice(option1s),
            option2=random.choice(option2s),
            change=random.choice(changes),
            area=random.choice(areas),
            action=random.choice(actions),
            outcome=random.choice(outcomes),
            process=random.choice(processes),
            variable1=random.choice(variable1s),
            variable2=random.choice(variable2s),
            practice=random.choice(practices),
            market=random.choice(markets),
        )
        prompts.append(prompt)

    return prompts


def generate_empathetic_prompts(num_prompts: int) -> List[str]:
    """Generate prompts that elicit empathetic/emotional responses."""
    empathetic_templates = [
        "Write a comforting message to someone who is feeling {emotion}",
        "Compose a supportive response to someone dealing with {situation}",
        "Draft a compassionate letter to someone who {experience}",
        "Write an understanding response to someone who {action}",
        "Create a caring message for someone going through {challenge}",
        "Write a supportive note to someone who feels {feeling}",
        "Compose an empathetic response to someone who {situation}",
        "Draft a kind message to someone struggling with {issue}",
        "Write a comforting letter to someone who {experience}",
        "Create a supportive response for someone who {action}",
        "Write an understanding message to someone dealing with {challenge}",
        "Compose a caring note to someone who feels {feeling}",
        "Draft an empathetic response to someone who {situation}",
        "Write a kind message to someone struggling with {issue}",
        "Create a comforting response for someone who {experience}",
        "Write a supportive letter to someone who {action}",
        "Compose an understanding message to someone dealing with {challenge}",
        "Draft a caring response to someone who feels {feeling}",
        "Write an empathetic note to someone who {situation}",
        "Create a kind message for someone struggling with {issue}",
    ]

    emotions = [
        "sad",
        "lonely",
        "anxious",
        "overwhelmed",
        "disappointed",
        "frustrated",
        "scared",
        "hopeless",
    ]
    situations = [
        "lost their job",
        "ended a relationship",
        "moved to a new city",
        "lost a loved one",
        "failed an exam",
    ]
    experiences = [
        "lost their home",
        "was betrayed by a friend",
        "received bad news",
        "made a big mistake",
        "feels isolated",
    ]
    actions = [
        "made a difficult decision",
        "took a big risk",
        "stood up for themselves",
        "admitted a mistake",
        "asked for help",
    ]
    challenges = [
        "a serious illness",
        "financial difficulties",
        "family problems",
        "work stress",
        "personal loss",
    ]
    feelings = ["invisible", "unheard", "misunderstood", "alone", "stuck", "powerless", "exhausted"]
    issues = [
        "self-doubt",
        "imposter syndrome",
        "grief",
        "anxiety",
        "depression",
        "stress",
        "loneliness",
    ]

    prompts = []
    for _ in range(num_prompts):
        template = random.choice(empathetic_templates)
        prompt = template.format(
            emotion=random.choice(emotions),
            situation=random.choice(situations),
            experience=random.choice(experiences),
            action=random.choice(actions),
            challenge=random.choice(challenges),
            feeling=random.choice(feelings),
            issue=random.choice(issues),
        )
        prompts.append(prompt)

    return prompts


def generate_prompts_for_personality(
    personality: str, num_prompts: int
) -> tuple[List[str], List[str]]:
    """Generate positive and negative prompts based on personality description."""
    personality = personality.lower().strip()

    if any(word in personality for word in ["formal", "professional", "business", "corporate"]):
        return generate_formal_prompts(num_prompts), generate_informal_prompts(num_prompts)

    elif any(word in personality for word in ["creative", "imaginative", "artistic", "expressive"]):
        return generate_creative_prompts(num_prompts), generate_analytical_prompts(num_prompts)

    elif any(word in personality for word in ["analytical", "logical", "rational", "systematic"]):
        return generate_analytical_prompts(num_prompts), generate_creative_prompts(num_prompts)

    elif any(
        word in personality for word in ["empathetic", "caring", "compassionate", "emotional"]
    ):
        return generate_empathetic_prompts(num_prompts), generate_analytical_prompts(num_prompts)

    elif any(word in personality for word in ["casual", "informal", "relaxed", "friendly"]):
        return generate_informal_prompts(num_prompts), generate_formal_prompts(num_prompts)

    else:
        # Default to formal vs informal
        return generate_formal_prompts(num_prompts), generate_informal_prompts(num_prompts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate prompts for persona vector training")
    parser.add_argument(
        "--personality",
        required=True,
        help="Personality description (e.g., 'formal and professional')",
    )
    parser.add_argument("--output-dir", required=True, help="Directory to save prompt files")
    parser.add_argument(
        "--num-prompts", type=int, default=20, help="Number of prompts to generate for each set"
    )
    args = parser.parse_args()

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    positive_prompts, negative_prompts = generate_prompts_for_personality(
        args.personality, args.num_prompts
    )

    # Save positive prompts
    positive_file = output_dir / "positive_prompts.txt"
    with open(positive_file, "w", encoding="utf-8") as f:
        for prompt in positive_prompts:
            f.write(prompt + "\n")

    # Save negative prompts
    negative_file = output_dir / "negative_prompts.txt"
    with open(negative_file, "w", encoding="utf-8") as f:
        for prompt in negative_prompts:
            f.write(prompt + "\n")

    print(
        f"Generated {len(positive_prompts)} positive prompts and {len(negative_prompts)} negative prompts"
    )
    print(f"Saved to: {positive_file} and {negative_file}")
    print("\nExample positive prompts:")
    for i, prompt in enumerate(positive_prompts[:3]):
        print(f"  {i + 1}. {prompt}")
    print("\nExample negative prompts:")
    for i, prompt in enumerate(negative_prompts[:3]):
        print(f"  {i + 1}. {prompt}")


if __name__ == "__main__":
    main()
