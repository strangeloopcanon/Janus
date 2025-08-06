# Persona Vectors Toolkit

Light-weight reference implementation of **Anthropic-style persona steering** for open-source language models.  The codebase shows how to learn a _linear direction_ in hidden-state space that nudges a frozen model to respond in a different style, tone, or policy – without any weight updates.

Target model in the examples: **Qwen3-0.6B** (`Qwen/Qwen3-0.6B` on Hugging Face), but every decoder model should work.

---

## Quick start

### 🚀 Get Started Fast

```bash
# See available personas and get started
python scripts/quick_start.py

# Or run a specific demo
python scripts/quick_start.py --demo chat
python scripts/quick_start.py --demo create  
python scripts/quick_start.py --demo evaluate
```

### Method 1: Automatic prompt generation (recommended)

```bash
# Create a persona vector from a simple description
python scripts/create_persona_from_description.py \
       --personality "formal and professional" \
       --model Qwen/Qwen3-0.6B \
       --output personas/persona_formal.json \
       --num-prompts 20

# Chat with the steered model
python scripts/run_with_persona.py \
       --model Qwen/Qwen3-0.6B \
       --persona personas/persona_formal.json \
       --alpha 1.2        # α>0 → more *formal*, α<0 → more *informal*
```

### Method 2: Manual prompt files

```bash
# 1. Create two plain-text files with prompts that elicit opposite personas
echo "Please write a formal apology letter to a customer"   > formal.txt
echo "Yo, shoot a quick sorry text to your buddy for me 🤙"  > informal.txt

# 2. Train the persona vector (this will download the model & run generations)
#    – default backend is PyTorch ("torch").  Pass --backend mlx to opt-in to MLX.
python scripts/train_persona_vector.py \
       --model Qwen/Qwen3-0.6B \
       --positive-prompts formal.txt \
       --negative-prompts informal.txt \
       --out personas/persona_formal_vs_informal.json

#    # MLX path (experimental – you must complete mlx_support.load_model)
# python scripts/train_persona_vector.py \
#        --backend mlx \
#        --model Qwen/Qwen3-0.6B \
#        --positive-prompts formal.txt \
#        --negative-prompts informal.txt \
#        --out persona_formal_vs_informal.json

# 3. Chat with the steered model
python scripts/run_with_persona.py \
       --model Qwen/Qwen3-0.6B \
       --persona personas/persona_formal_vs_informal.json \
       --alpha 1.2        # α>0 → more *formal*, α<0 → more *informal*
```

---

## Library usage

```python
from persona_steering_library import compute_persona_vector, add_persona_hook

# 1) compute the vector programmatically
pv = compute_persona_vector(
    model_name="Qwen/Qwen3-0.6B",
    positive_prompts=[...],
    negative_prompts=[...],
)

# 2) run inference with steering
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
mdl = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B").to("cuda")

remove_hook = add_persona_hook(mdl, pv.vector, alpha=1.0)

prompt = "Summarise ‘The Great Gatsby’ in one paragraph."
inputs = tok(prompt, return_tensors="pt").to("cuda")
out = mdl.generate(**inputs, max_new_tokens=128)
print(tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))

remove_hook()  # detach when done
```

---

## Project Structure

```
personality-analysis/
├── 📁 personas/                    # Pre-trained persona vectors
│   ├── persona_formal.json/.pt     # Formal vs informal steering
│   ├── persona_creative.json/.pt   # Creative vs conventional steering
│   ├── persona_gpt_oss_simple.pt   # Large model persona (4096 dims)
│   └── README.md                   # Persona documentation
│
├── 📁 results/                     # Research results and analysis
│   ├── 📁 evaluations/             # Persona steering evaluations
│   │   ├── formal_persona_evaluation.json
│   │   └── creative_persona_evaluation.json
│   ├── 📁 analysis/                # Technical analysis
│   │   └── layer_injection_analysis.json
│   ├── 📁 figures/                 # Papers and visualizations
│   │   └── persona_vectors_paper.pdf
│   └── README.md                   # Research documentation
│
├── 📁 scripts/                     # Core tools and utilities
│   ├── quick_start.py              # 🚀 Project overview and guidance
│   ├── run_with_persona.py         # 💬 Interactive chat with persona steering
│   ├── create_persona_from_description.py  # 🎭 Auto-generate personas
│   ├── evaluate_persona_vector.py  # 📊 Comprehensive evaluation (includes comparison)
│   ├── analyze_persona_layers.py   # 🔬 Layer effectiveness research
│   ├── generate_persona_prompts.py # 📝 Training data generation
│   └── train_persona_vector.py     # 🏗️ Manual training from prompt files
│
├── 📁 persona_steering_library/    # Core library
│   ├── __init__.py                 # Public API
│   ├── compute.py                  # Data collection & vector training
│   ├── hooks.py                    # Runtime steering hook
│   └── mlx_support.py              # Experimental MLX backend
│
├── 📁 examples/                    # Example files (currently empty)
├── .env                           # Environment configuration
└── README.md                      # This file
```

### 📋 Script Overview

**🚀 User-Facing Scripts:**
- `quick_start.py` - Project overview and guidance
- `run_with_persona.py` - Interactive chat demo with α slider
- `create_persona_from_description.py` - Easy persona creation from descriptions
- `evaluate_persona_vector.py` - Comprehensive evaluation with comparison

**🔬 Research & Development:**
- `analyze_persona_layers.py` - Find optimal transformer layers
- `generate_persona_prompts.py` - Generate training data for different personalities
- `train_persona_vector.py` - Manual training from custom prompt files

### 📊 Available Personas

- **`persona_formal.json/.pt`** - Formal vs informal communication (1024 dims)
- **`persona_creative.json/.pt`** - Creative vs conventional thinking (1024 dims)  
- **`persona_gpt_oss_simple.pt`** - Large model persona (4096 dims, no metadata)

### 🎯 Typical Workflow

```bash
# 1. Explore the project
python scripts/quick_start.py

# 2. Create a new persona
python scripts/create_persona_from_description.py \
    --personality "friendly and helpful" \
    --model Qwen/Qwen3-0.6B \
    --output personas/persona_friendly.json

# 3. Test the persona interactively
python scripts/run_with_persona.py \
    --model Qwen/Qwen3-0.6B \
    --persona personas/persona_friendly.json \
    --alpha 1.0

# 4. Evaluate the persona comprehensively
python scripts/evaluate_persona_vector.py \
    --model Qwen/Qwen3-0.6B \
    --persona personas/persona_friendly.json \
    --output results/evaluations/friendly_evaluation.json \
    --compare

# 5. Research optimal layers (optional)
python scripts/analyze_persona_layers.py \
    --model Qwen/Qwen3-0.6B \
    --output results/analysis/layer_analysis.json
```

---

## Recent Improvements

- **🧹 Cleaned Structure**: Removed 6 redundant scripts, consolidated functionality
- **📁 Organized Files**: Personas, results, and scripts properly organized
- **📊 Enhanced Evaluation**: Combined comparison functionality into main evaluation script
- **🚀 Quick Start**: Added comprehensive project overview script
- **📚 Documentation**: Added detailed READMEs for each directory

---

## Theory in one paragraph

1. Collect hidden‐state activations `h` for completions that clearly exhibit two opposite personas (e.g., *formal* vs *informal*).
2. Compute the mean of each cluster `μ₊`, `μ₋` and take their **difference vector** `v = (μ₊ – μ₋) / ‖·‖₂`.
3. During generation, inject `α·v` into the residual stream of a chosen transformer block.  Positive `α` pushes the model toward the first persona; negative toward the second.  Because the operation is linear and the base weights remain frozen, it adds negligible runtime overhead and can be toggled on/off at will.

---

## Requirements

* Python ≥3.9
* `torch` (with CUDA, MPS, or CPU) + `transformers` >=4.40
* **Optional**: `mlx`, `mlx-examples`, `huggingface_hub` for the MLX backend.

Install: `pip install -r requirements.txt` (create one matching your environment).

---

## Limitations & safety

* Steering directions may entangle with attributes you did **not** intend to control.  Always evaluate outputs for bias and safety violations.
* Very small prompt sets (<1 k examples) still work but yield weaker, noisier control.
* Currently targets the last transformer block; experiment with other layers for sharper or more nuanced control.

---

Feel free to open issues or PRs – this repository is meant as a pedagogical reference as much as a practical tool.
