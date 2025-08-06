# Persona Vectors

This directory contains pre-trained persona vectors for steering language model behavior.

## Available Personas

### `persona_formal.json` & `persona_formal.pt`
- **Purpose**: Formal vs informal communication style
- **Model**: Qwen3-0.6B (1024 dimensions)
- **Layer**: Last transformer block (-1)
- **Usage**: Positive α values make responses more formal, negative values more informal

### `persona_creative.json` & `persona_creative.pt`
- **Purpose**: Creative vs conventional thinking style
- **Model**: Qwen3-0.6B (1024 dimensions)
- **Layer**: Last transformer block (-1)
- **Usage**: Positive α values make responses more creative, negative values more conventional

### `persona_gpt_oss_simple.pt`
- **Purpose**: Unknown (no metadata)
- **Model**: Likely a larger model (4096 dimensions)
- **Layer**: Unknown
- **Usage**: Requires investigation - no metadata file

## Usage

```bash
# Use with the run_with_persona.py script
python scripts/run_with_persona.py \
    --model Qwen/Qwen3-0.6B \
    --persona personas/persona_formal.json \
    --alpha 1.0
```

## File Structure

- **`.json` files**: Metadata containing layer index and hidden size
- **`.pt` files**: PyTorch tensors containing the actual persona vectors 