#!/usr/bin/env python
"""Quick start demo showing the organized persona vectors project."""

import argparse
import os
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Quick start demo for persona vectors")
    parser.add_argument("--demo", choices=["chat", "create", "evaluate"], default="chat", 
                       help="Demo type to run")
    args = parser.parse_args()
    
    print("="*60)
    print("PERSONA VECTORS - QUICK START")
    print("="*60)
    
    # Check if personas exist
    personas_dir = Path("personas")
    if not personas_dir.exists():
        print("❌ personas/ directory not found!")
        print("Please run the organization script first.")
        return
    
    available_personas = list(personas_dir.glob("*.json"))
    if not available_personas:
        print("❌ No persona files found in personas/ directory!")
        print("Create a persona first with:")
        print("python scripts/create_persona_from_description.py --personality 'formal and professional' --model Qwen/Qwen3-0.6B --output personas/persona_formal.json")
        return
    
    print(f"✅ Found {len(available_personas)} persona files:")
    for persona in available_personas:
        print(f"  - {persona.name}")
    
    if args.demo == "chat":
        print("\n🎯 CHAT DEMO")
        print("="*30)
        print("To start an interactive chat with persona steering:")
        print(f"python scripts/run_with_persona.py --model Qwen/Qwen3-0.6B --persona {available_personas[0]} --alpha 1.0")
        
    elif args.demo == "create":
        print("\n🎯 CREATE PERSONA DEMO")
        print("="*30)
        print("To create a new persona from description:")
        print("python scripts/create_persona_from_description.py --personality 'creative and artistic' --model Qwen/Qwen3-0.6B --output personas/persona_creative.json")
        
    elif args.demo == "evaluate":
        print("\n🎯 EVALUATE PERSONA DEMO")
        print("="*30)
        print("To evaluate a persona vector:")
        print(f"python scripts/evaluate_persona_vector.py --model Qwen/Qwen3-0.6B --persona {available_personas[0]} --output results/evaluations/my_evaluation.json --compare")
    
    print("\n📁 PROJECT STRUCTURE:")
    print("personas/           - Pre-trained persona vectors")
    print("results/            - Evaluation results and analysis")
    print("  evaluations/      - Persona steering evaluations")
    print("  analysis/         - Technical analysis")
    print("  figures/          - Papers and visualizations")
    print("scripts/            - Core tools and utilities")
    print("persona_steering_library/    - Core library")
    
    print("\n📚 DOCUMENTATION:")
    print("- README.md         - Main project documentation")
    print("- personas/README.md - Persona documentation")
    print("- results/README.md - Research results documentation")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main() 
