#!/usr/bin/env python
"""Run the base model with an optional persona vector applied non-interactively."""

from __future__ import annotations

# Ensure repo root is on sys.path when running as `python scripts/...`
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from persona_steering_library import add_persona_hook, PersonaVectorResult


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HF model name or path")
    parser.add_argument("--prompt", required=True, help="Prompt to run")
    parser.add_argument("--persona", help="Optional path to persona JSON")
    parser.add_argument("--alpha", type=float, default=1.5, help="Steering strength")
    parser.add_argument("--max_new_tokens", type=int, default=60, help="Max new tokens")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.to(device)
    model.eval()

    remove_hook = None
    if args.persona:
        pdata = PersonaVectorResult.load(args.persona)
        remove_hook = add_persona_hook(model, pdata.vector, layer_idx=pdata.layer_idx, alpha=args.alpha)

    try:
        inputs = tokenizer(args.prompt, return_tensors="pt").to(device)
        input_len = inputs["input_ids"].shape[1]
        
        with torch.no_grad():
            gen_out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                top_p=0.9,
                temperature=0.8,
            )
        
        completion = tokenizer.decode(gen_out.sequences[0, input_len:], skip_special_tokens=True)
        
        if args.persona:
            print(f"--- With Persona: {os.path.basename(args.persona)} ---")
        else:
            print("--- Without Persona (Baseline) ---")
        print(f"Prompt: {args.prompt}")
        print(f"Model: {completion}")

    finally:
        if remove_hook:
            remove_hook()


if __name__ == "__main__":
    main()
