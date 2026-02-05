#!/usr/bin/env python
"""Analyze which layers are most effective for persona vectors and test injection points."""

from __future__ import annotations

import argparse
import json
from typing import List, Dict, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from persona_steering_library import compute_persona_vector, add_persona_hook


def analyze_layer_activations(
    model_name: str,
    positive_prompts: List[str],
    negative_prompts: List[str],
    max_layers: int = 6,
) -> Dict[str, Any]:
    """Analyze activations across different layers to find the most effective ones."""

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()

    # Get total number of layers
    try:
        total_layers = len(model.model.layers)
        print(f"Model has {total_layers} layers")
    except AttributeError:
        total_layers = len(model.transformer.h)
        print(f"Model has {total_layers} layers")

    # Test different layers (last few layers are usually most effective)
    test_layers = list(range(max(-max_layers, -total_layers), 0))

    layer_analysis = {}

    for layer_idx in test_layers:
        print(f"\nAnalyzing layer {layer_idx}...")

        try:
            # Compute persona vector for this layer
            persona = compute_persona_vector(
                model_name=model_name,
                positive_prompts=positive_prompts,
                negative_prompts=negative_prompts,
                layer_idx=layer_idx,
                max_new_tokens=32,
                device=device,
            )

            # Test the vector's effectiveness
            test_prompt = "Write a short message"
            inputs = tokenizer(test_prompt, return_tensors="pt").to(device)
            input_len = inputs["input_ids"].shape[1]

            # Generate baseline response
            with torch.no_grad():
                gen_out = model.generate(
                    **inputs,
                    max_new_tokens=32,
                    do_sample=True,
                    top_p=0.9,
                    temperature=0.8,
                    return_dict_in_generate=True,
                    output_hidden_states=True,
                )

            baseline_response = tokenizer.decode(
                gen_out.sequences[0, input_len:], skip_special_tokens=True
            )

            # Generate steered response
            remove_hook = add_persona_hook(model, persona.vector, layer_idx=layer_idx, alpha=1.0)

            try:
                with torch.no_grad():
                    gen_out = model.generate(
                        **inputs,
                        max_new_tokens=32,
                        do_sample=True,
                        top_p=0.9,
                        temperature=0.8,
                        return_dict_in_generate=True,
                        output_hidden_states=True,
                    )

                steered_response = tokenizer.decode(
                    gen_out.sequences[0, input_len:], skip_special_tokens=True
                )

                # Calculate vector statistics
                vector_norm = float(persona.vector.norm().item())
                vector_mean = float(persona.vector.mean().item())
                vector_std = float(persona.vector.std().item())

                layer_analysis[layer_idx] = {
                    "baseline_response": baseline_response,
                    "steered_response": steered_response,
                    "vector_norm": vector_norm,
                    "vector_mean": vector_mean,
                    "vector_std": vector_std,
                    "response_changed": baseline_response != steered_response,
                }

                print(f"  Vector norm: {vector_norm:.4f}")
                print(f"  Response changed: {baseline_response != steered_response}")
                print(f"  Baseline: {baseline_response[:50]}...")
                print(f"  Steered: {steered_response[:50]}...")

            finally:
                remove_hook()

        except Exception as e:
            print(f"  Error: {e}")
            layer_analysis[layer_idx] = {"error": str(e)}

    return layer_analysis


def test_injection_points(
    model_name: str,
    persona_path: str,
) -> Dict[str, Any]:
    """Test different injection points in the model."""

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model and persona
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()

    from persona_steering_library import PersonaVectorResult

    persona = PersonaVectorResult.load(persona_path)

    test_prompt = "Write a short message"
    inputs = tokenizer(test_prompt, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    injection_results = {}

    # Test residual injection (current implementation)
    print("\nTesting residual injection...")
    remove_hook = add_persona_hook(model, persona.vector, layer_idx=persona.layer_idx, alpha=1.0)

    try:
        with torch.no_grad():
            gen_out = model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=True,
                top_p=0.9,
                temperature=0.8,
                return_dict_in_generate=True,
                output_hidden_states=True,
            )

        response = tokenizer.decode(gen_out.sequences[0, input_len:], skip_special_tokens=True)
        injection_results["residual"] = response

    finally:
        remove_hook()

    # Test attention injection (conceptual - would need implementation)
    print("Testing attention injection (conceptual)...")
    injection_results["attention"] = "Not implemented - would inject into attention weights"

    # Test MLP injection (conceptual - would need implementation)
    print("Testing MLP injection (conceptual)...")
    injection_results["mlp"] = "Not implemented - would inject into MLP activations"

    return injection_results


def find_optimal_layer(
    layer_analysis: Dict[str, Any],
) -> tuple[int, Dict[str, Any]]:
    """Find the most effective layer based on analysis."""

    best_layer = None
    best_score = 0
    best_data = {}

    for layer_idx, data in layer_analysis.items():
        if "error" in data:
            continue

        # Score based on vector norm and response change
        score = data["vector_norm"]
        if data["response_changed"]:
            score *= 1.5  # Bonus for changing response

        if score > best_score:
            best_score = score
            best_layer = layer_idx
            best_data = data

    return best_layer, best_data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze persona vector layers and injection points"
    )
    parser.add_argument("--model", required=True, help="HF model name")
    parser.add_argument("--persona", help="Path to existing persona JSON")
    parser.add_argument("--output", help="Output file for analysis")
    parser.add_argument("--max-layers", type=int, default=6, help="Max layers to test")
    args = parser.parse_args()

    print("=" * 60)
    print("PERSONA VECTOR LAYER ANALYSIS")
    print("=" * 60)

    # 1. Analyze layer effectiveness
    print("\n1. Analyzing layer effectiveness...")
    positive_prompts = [
        "Write a formal letter to a customer",
        "Compose a professional email to your supervisor",
        "Draft a business proposal",
    ]
    negative_prompts = [
        "Write a casual text to your friend",
        "Send a relaxed message to your buddy",
        "Draft a chill email",
    ]

    layer_analysis = analyze_layer_activations(
        args.model, positive_prompts, negative_prompts, args.max_layers
    )

    # 2. Find optimal layer
    print("\n2. Finding optimal layer...")
    optimal_layer, optimal_data = find_optimal_layer(layer_analysis)

    if optimal_layer is not None:
        print(f"\nOptimal layer: {optimal_layer}")
        print(f"Vector norm: {optimal_data['vector_norm']:.4f}")
        print(f"Response changed: {optimal_data['response_changed']}")
        print(f"Baseline: {optimal_data['baseline_response'][:100]}...")
        print(f"Steered: {optimal_data['steered_response'][:100]}...")
    else:
        print("No optimal layer found")

    # 3. Test injection points
    if args.persona:
        print("\n3. Testing injection points...")
        injection_results = test_injection_points(args.model, args.persona)

        print("\nInjection results:")
        for method, result in injection_results.items():
            print(f"  {method}: {result[:100]}...")

    # 4. Summary
    print("\n" + "=" * 60)
    print("LAYER ANALYSIS SUMMARY")
    print("=" * 60)

    print("\nLayer comparison:")
    for layer_idx, data in layer_analysis.items():
        if "error" not in data:
            print(f"  Layer {layer_idx}:")
            print(f"    Vector norm: {data['vector_norm']:.4f}")
            print(f"    Response changed: {data['response_changed']}")
            print(
                f"    Vector stats - mean: {data['vector_mean']:.4f}, std: {data['vector_std']:.4f}"
            )

    # Save results
    if args.output:
        results = {
            "layer_analysis": layer_analysis,
            "optimal_layer": optimal_layer,
            "optimal_data": optimal_data,
            "injection_results": injection_results if args.persona else {},
        }

        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {args.output}")

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
