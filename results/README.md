# Research Results

This directory contains all evaluation results, analysis data, and documentation for the persona vectors research.

## Directory Structure

```
results/
├── evaluations/           # Persona steering evaluation results
├── analysis/             # Technical analysis and experiments  
├── figures/              # Papers, plots, and visualizations
└── README.md            # This file
```

## Evaluations

### `evaluations/formal_persona_evaluation.json`
- **Purpose**: Evaluation of formal vs informal persona steering
- **Model**: Qwen3-0.6B with persona_formal vector
- **Method**: Tests responses at different α values (-1.0, -0.5, 0.0, 0.5, 1.0, 1.5)
- **Prompts**: "Write a short message", "How are you today?", "Tell me about yourself", "What's the weather like?"
- **Key Findings**: Shows clear progression from informal to formal responses as α increases

### `evaluations/creative_persona_evaluation.json`
- **Purpose**: Evaluation of creative vs conventional persona steering
- **Model**: Qwen3-0.6B with persona_creative vector
- **Method**: Same α value testing as formal evaluation
- **Key Findings**: Demonstrates creative vs conventional response patterns

## Analysis

### `analysis/layer_injection_analysis.json`
- **Purpose**: Analysis of optimal transformer layer for persona injection
- **Layers Tested**: -3, -2, -1 (last three transformer blocks)
- **Injection Methods**: Residual, Attention, MLP (attention/MLP not implemented)
- **Key Findings**: 
  - Optimal layer: -3 (third-to-last layer)
  - Residual injection most effective
  - Response changes clearly visible across all tested layers

## Figures

### `figures/persona_vectors_paper.pdf`
- **Purpose**: Research paper or detailed methodology documentation
- **Size**: 4.8MB, 23 pages
- **Content**: Likely contains theoretical background, methodology, or detailed results

## Usage for Paper Writing

### Data Sources
- **Quantitative Results**: Use evaluation JSON files for response analysis
- **Technical Analysis**: Use layer analysis for methodology validation
- **Theoretical Background**: Reference the PDF for literature review

### Suggested Paper Structure
1. **Introduction**: Persona steering concept and motivation
2. **Methodology**: Layer injection technique (reference layer analysis)
3. **Results**: Formal and creative persona evaluations
4. **Discussion**: Effectiveness and limitations
5. **Conclusion**: Future work and applications

### Key Metrics to Extract
- Response variation across α values
- Optimal injection layer performance
- Vector statistics (norm, mean, std)
- Response change detection

## File Naming Convention

All files follow consistent naming:
- `{persona_type}_persona_evaluation.json`
- `{analysis_type}_analysis.json`
- `{content_type}_{description}.pdf`

This structure enables easy automation for paper generation and reproducible research. 