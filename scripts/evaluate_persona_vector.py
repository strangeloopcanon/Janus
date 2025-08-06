#!/usr/bin/env python
"""Evaluate persona vector quality and test injection effectiveness."""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import List, Dict, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from persona_steering_library import add_persona_hook, PersonaVectorResult, compute_persona_vector


def test_vector_quality(
    model_name: str,
    persona_path: str,
    test_prompts: List[str],
    alpha_values: List[float] = None,
) -> Dict[str, Any]:
    """Test persona vector quality across different alpha values."""
    
    if alpha_values is None:
        alpha_values = [0.0, 0.5, 1.0, 1.5, -0.5, -1.0]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model and persona
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()
    
    persona = PersonaVectorResult.load(persona_path)
    
    results = {
        "persona_info": {
            "vector_shape": list(persona.vector.shape),
            "layer_idx": persona.layer_idx,
            "hidden_size": persona.hidden_size,
            "vector_norm": float(persona.vector.norm().item()),
        },
        "responses": {}
    }
    
    def generate_response(prompt: str, alpha: float = 0.0) -> str:
        """Generate response with optional persona steering."""
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_len = inputs["input_ids"].shape[1]
        
        # Add persona hook if alpha != 0
        remove_hook = None
        if alpha != 0:
            remove_hook = add_persona_hook(model, persona.vector, layer_idx=persona.layer_idx, alpha=alpha)
        
        try:
            with torch.no_grad():
                gen_out = model.generate(
                    **inputs,
                    max_new_tokens=64,
                    do_sample=True,
                    top_p=0.9,
                    temperature=0.8,
                    return_dict_in_generate=True,
                    output_hidden_states=True,
                )
            
            completion = tokenizer.decode(gen_out.sequences[0, input_len:], skip_special_tokens=True)
            return completion
        finally:
            if remove_hook:
                remove_hook()
    
    # Test each prompt with different alpha values
    for prompt in test_prompts:
        results["responses"][prompt] = {}
        for alpha in alpha_values:
            response = generate_response(prompt, alpha)
            results["responses"][prompt][alpha] = response
    
    return results


def analyze_layer_effectiveness(
    model_name: str,
    positive_prompts: List[str],
    negative_prompts: List[str],
    test_prompts: List[str],
    max_layers: int = 6,
) -> Dict[str, Any]:
    """Test persona vector effectiveness across different layers."""
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()
    
    # Get total number of layers
    try:
        total_layers = len(model.model.layers)
    except AttributeError:
        total_layers = len(model.transformer.h)
    
    print(f"Model has {total_layers} layers")
    
    layer_results = {}
    
    # Test different layers (last few layers are usually most effective)
    test_layers = list(range(max(-max_layers, -total_layers), 0))
    
    for layer_idx in test_layers:
        print(f"Testing layer {layer_idx}...")
        
        # Compute persona vector for this layer
        try:
            persona = compute_persona_vector(
                model_name=model_name,
                positive_prompts=positive_prompts,
                negative_prompts=negative_prompts,
                layer_idx=layer_idx,
                max_new_tokens=32,
                device=device,
            )
            
            # Test with a simple prompt
            test_prompt = "Write a short message"
            
            # Generate baseline response
            inputs = tokenizer(test_prompt, return_tensors="pt").to(device)
            input_len = inputs["input_ids"].shape[1]
            
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
            
            baseline_response = tokenizer.decode(gen_out.sequences[0, input_len:], skip_special_tokens=True)
            
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
                
                steered_response = tokenizer.decode(gen_out.sequences[0, input_len:], skip_special_tokens=True)
                
                layer_results[layer_idx] = {
                    "baseline": baseline_response,
                    "steered": steered_response,
                    "vector_norm": float(persona.vector.norm().item()),
                }
                
            finally:
                remove_hook()
                
        except Exception as e:
            print(f"Error testing layer {layer_idx}: {e}")
            layer_results[layer_idx] = {"error": str(e)}
    
    return layer_results


def test_injection_effectiveness(
    model_name: str,
    persona_path: str,
    injection_points: List[str] = None,
) -> Dict[str, Any]:
    """Test different injection points and methods."""
    
    if injection_points is None:
        injection_points = ["residual", "attention", "mlp"]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model and persona
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()
    
    persona = PersonaVectorResult.load(persona_path)
    
    test_prompt = "Write a short message"
    inputs = tokenizer(test_prompt, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]
    
    results = {}
    
    # Test different injection methods
    for injection_point in injection_points:
        print(f"Testing injection at: {injection_point}")
        
        # For now, we only have residual injection implemented
        # This could be extended to test attention and MLP injection
        if injection_point == "residual":
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
                results[injection_point] = response
                
            finally:
                remove_hook()
        else:
            results[injection_point] = "Not implemented yet"
    
    return results


def compare_persona_effectiveness(
    model_name: str,
    persona_path: str,
    test_prompt: str,
    alpha: float = 1.0,
) -> Dict[str, str]:
    """Compare base model vs persona-steered responses."""
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model and persona
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()
    
    persona = PersonaVectorResult.load(persona_path)
    
    def generate_response(prompt: str, alpha: float = 0.0) -> str:
        """Generate response with optional persona steering."""
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_len = inputs["input_ids"].shape[1]
        
        # Add persona hook if alpha != 0
        remove_hook = None
        if alpha != 0:
            remove_hook = add_persona_hook(model, persona.vector, layer_idx=persona.layer_idx, alpha=alpha)
        
        try:
            with torch.no_grad():
                gen_out = model.generate(
                    **inputs,
                    max_new_tokens=64,
                    do_sample=True,
                    top_p=0.9,
                    temperature=0.8,
                    return_dict_in_generate=True,
                    output_hidden_states=True,
                )
            
            completion = tokenizer.decode(gen_out.sequences[0, input_len:], skip_special_tokens=True)
            return completion
        finally:
            if remove_hook:
                remove_hook()
    
    # Generate responses
    base_response = generate_response(test_prompt, alpha=0.0)
    steered_response = generate_response(test_prompt, alpha=alpha)
    negative_response = generate_response(test_prompt, alpha=-alpha)
    
    return {
        "base": base_response,
        "steered": steered_response,
        "negative": negative_response,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate persona vector quality")
    parser.add_argument("--model", required=True, help="HF model name")
    parser.add_argument("--persona", required=True, help="Path to persona JSON (e.g., personas/persona_formal.json)")
    parser.add_argument("--output", help="Output file for results (e.g., results/evaluations/my_persona_evaluation.json)")
    parser.add_argument("--test-layer-effectiveness", action="store_true", help="Test different layers")
    parser.add_argument("--test-injection", action="store_true", help="Test injection methods")
    parser.add_argument("--compare", action="store_true", help="Compare base vs steered responses")
    parser.add_argument("--prompt", default="Write a short apology message", help="Test prompt for comparison")
    args = parser.parse_args()
    
    # Test prompts
    test_prompts = [
        "Write a short message",
        "How are you today?",
        "Tell me about yourself",
        "What's the weather like?",
    ]
    
    print("="*60)
    print("PERSONA VECTOR EVALUATION")
    print("="*60)
    
    # 1. Test vector quality across alpha values
    print("\n1. Testing vector quality across alpha values...")
    quality_results = test_vector_quality(args.model, args.persona, test_prompts)
    
    print(f"\nPersona Info:")
    for key, value in quality_results["persona_info"].items():
        print(f"  {key}: {value}")
    
    print(f"\nSample responses for 'Write a short message':")
    for alpha, response in quality_results["responses"]["Write a short message"].items():
        print(f"  Alpha {alpha}: {response[:100]}...")
    
    # 2. Test layer effectiveness
    if args.test_layer_effectiveness:
        print("\n2. Testing layer effectiveness...")
        positive_prompts = ["Write a formal letter", "Compose a professional email"]
        negative_prompts = ["Write a casual text", "Send a relaxed message"]
        
        layer_results = analyze_layer_effectiveness(
            args.model, positive_prompts, negative_prompts, test_prompts
        )
        
        print(f"\nLayer comparison:")
        for layer_idx, result in layer_results.items():
            if "error" not in result:
                print(f"  Layer {layer_idx}:")
                print(f"    Baseline: {result['baseline'][:50]}...")
                print(f"    Steered: {result['steered'][:50]}...")
                print(f"    Vector norm: {result['vector_norm']:.4f}")
    
    # 3. Test injection effectiveness
    if args.test_injection:
        print("\n3. Testing injection methods...")
        injection_results = test_injection_effectiveness(args.model, args.persona)
        
        print(f"\nInjection comparison:")
        for method, response in injection_results.items():
            print(f"  {method}: {response[:100]}...")
    
    # 4. Compare base vs steered responses
    if args.compare:
        print("\n4. Comparing base vs steered responses...")
        compare_results = compare_persona_effectiveness(args.model, args.persona, args.prompt, args.alpha)
        
        print(f"\nComparison results:")
        print(f"  Base response: {compare_results['base'][:100]}...")
        print(f"  Steered response (α={args.alpha}): {compare_results['steered'][:100]}...")
        print(f"  Negative steered response (α={-args.alpha}): {compare_results['negative'][:100]}...")
    
    # Save results
    if args.output:
        all_results = {
            "quality": quality_results,
            "layers": layer_results if args.test_layer_effectiveness else {},
            "injection": injection_results if args.test_injection else {},
            "comparison": compare_results if args.compare else {},
        }
        
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\nResults saved to: {args.output}")
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main() 