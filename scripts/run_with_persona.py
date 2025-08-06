#!/usr/bin/env python
"""Simple demo – run the base model with a persona vector applied interactively."""

from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from persona_steering_library import add_persona_hook, PersonaVectorResult


def main() -> None:  # noqa: D401
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HF model name or path")
    parser.add_argument("--persona", required=True, help="Path to persona JSON (e.g., personas/persona_formal.json)")
    parser.add_argument("--alpha", type=float, default=1.0, help="Steering strength")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.to(device)
    model.eval()

    pdata = PersonaVectorResult.load(args.persona)

    remove_hook = add_persona_hook(model, pdata.vector, layer_idx=pdata.layer_idx, alpha=args.alpha)

    print("⇢ Ready. Type a prompt (Ctrl-D/Ctrl-Z to exit)")
    try:
        while True:
            prompt = input("\nUser: ")
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            input_len = inputs["input_ids"].shape[1]
            
            with torch.no_grad():
                gen_out = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=True,
                    top_p=0.9,
                    temperature=0.8,
                    return_dict_in_generate=True,
                    output_hidden_states=True,
                )
            
            completion = tokenizer.decode(gen_out.sequences[0, input_len:], skip_special_tokens=True)
            print(f"Model: {completion}")
    except (EOFError, KeyboardInterrupt):
        print("\nExiting…")
    finally:
        remove_hook()


if __name__ == "__main__":
    main()

