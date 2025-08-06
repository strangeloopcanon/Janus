#!/usr/bin/env python
"""Create a persona vector from a personality description.

This script automatically generates prompts and trains a persona vector
based on a simple personality description.

Usage
-----
python scripts/create_persona_from_description.py \
    --personality "formal and professional" \
    --model Qwen/Qwen3-0.6B \
    --output persona_formal.json \
    --num-prompts 20
"""

from __future__ import annotations

import argparse
import pathlib
import tempfile
import shutil

from persona_steering_library import compute_persona_vector
from scripts.generate_persona_prompts import generate_prompts_for_personality


def main() -> None:
    parser = argparse.ArgumentParser(description="Create persona vector from personality description")
    parser.add_argument("--personality", required=True, help="Personality description (e.g., 'formal and professional')")
    parser.add_argument("--model", required=True, help="HuggingFace model name")
    parser.add_argument("--output", required=True, help="Output file for persona vector (e.g., personas/persona_my_persona.json)")
    parser.add_argument("--num-prompts", type=int, default=20, help="Number of prompts to generate")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Max tokens to generate per prompt")
    parser.add_argument("--backend", choices=["torch", "mlx"], default="torch", help="Backend to use")
    args = parser.parse_args()
    
    print(f"Creating persona vector for: '{args.personality}'")
    print(f"Using model: {args.model}")
    print(f"Generating {args.num_prompts} prompts per set...")
    
    # Generate prompts
    positive_prompts, negative_prompts = generate_prompts_for_personality(args.personality, args.num_prompts)
    
    print(f"\nGenerated {len(positive_prompts)} positive and {len(negative_prompts)} negative prompts")
    print("\nExample positive prompts:")
    for i, prompt in enumerate(positive_prompts[:3]):
        print(f"  {i+1}. {prompt}")
    print("\nExample negative prompts:")
    for i, prompt in enumerate(negative_prompts[:3]):
        print(f"  {i+1}. {prompt}")
    
    print(f"\nTraining persona vector...")
    
    # Compute persona vector
    result = compute_persona_vector(
        model_name=args.model,
        positive_prompts=positive_prompts,
        negative_prompts=negative_prompts,
        max_new_tokens=args.max_new_tokens,
        backend=args.backend,
    )
    
    # Save result
    result.save(args.output)
    
    print(f"\n✅ Persona vector saved to: {args.output}")
    print(f"Vector shape: {result.vector.shape}")
    print(f"Target layer: {result.layer_idx}")
    print(f"Hidden size: {result.hidden_size}")
    
    print(f"\nYou can now use this persona vector with:")
    print(f"python scripts/run_with_persona.py --model {args.model} --persona {args.output} --alpha 1.0")
    print(f"\nOr evaluate it with:")
    print(f"python scripts/evaluate_persona_vector.py --model {args.model} --persona {args.output} --output results/evaluations/my_persona_evaluation.json")


if __name__ == "__main__":
    main() 