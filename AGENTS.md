# Repository Guidelines

## Project Structure & Module Organization
- Root: Python toolkit for persona steering.
- `persona_steering_library/`: Core library (`compute.py`, `hooks.py`, `mlx_support.py`).
- `scripts/`: CLI utilities and demos (`quick_start.py`, `run_with_persona.py`, `create_persona_from_description.py`, `evaluate_persona_vector.py`, `train_persona_vector.py`).
- `personas/`: Persona vectors (`persona_*.json` + matching `.pt`).
- `results/`: Evaluation outputs and analysis.
- `examples/`: Sample inputs (if any).

## Build, Test, and Development Commands
- Setup: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
- Quick tour: `python scripts/quick_start.py`
- Create persona: `python scripts/create_persona_from_description.py --personality "formal and professional" --model Qwen/Qwen3-0.6B --output personas/persona_formal.json`
- Chat demo: `python scripts/run_with_persona.py --model Qwen/Qwen3-0.6B --persona personas/persona_formal.json --alpha 1.0`
- Evaluate: `python scripts/evaluate_persona_vector.py --model Qwen/Qwen3-0.6B --persona personas/persona_formal.json --output results/evaluations/formal_eval.json --compare`

## Coding Style & Naming Conventions
- Style: PEP 8, 4-space indents, type hints where practical.
- Modules/functions: `snake_case`; classes: `CapWords`.
- Personas: `personas/persona_<descriptor>.json` with matching `.pt` saved via `PersonaVectorResult.save`.
- Keep scripts in `scripts/`; library-only logic in `persona_steering_library/` with docstrings.
- Lint hints present (`# noqa`, `# pylint: disable`). If available, run `flake8`/`pylint` and format with `black`.

## Testing Guidelines
- No formal test suite yet. Validate changes via scripts above.
- Prefer small HF models (e.g., `Qwen/Qwen3-0.6B`) for quick checks.
- If adding tests, use `pytest`, place files under `tests/`, and name `test_*.py`.
- Ensure persona files round-trip: `PersonaVectorResult.save(...)` then `.load(...)` and apply with `add_persona_hook`.

## Commit & Pull Request Guidelines
- Messages: short, imperative mood; scope prefix optional.
  - Examples: `scripts: add layer analysis helper`, `library: fix hook tuple output`.
- PRs: include summary, rationale, commands to reproduce, and any result samples (paths under `results/`). Link related issues.
- Keep diffs focused; update README or inline docstrings when behavior changes.

## Security & Configuration Tips
- Do not commit secrets. Use `.env` locally (ignored by Git) and document needed vars.
- Large model downloads occur on first run; set HF cache via `TRANSFORMERS_CACHE` if needed.
- Default backend is PyTorch; MLX is experimental—guard changes behind flags and avoid breaking torch path.

## Project Runbook (CC‑News 4B)
- For end‑to‑end commands, known issues, and resume instructions, consult:
  - `docs/CC_NEWS_4B_RUNBOOK.md`
- Important MLX note: `mlx_lm.generate` does not apply our Python‑level layer injection hooks. Using it for variants (`--variants-safe`) makes outputs identical to base. Use `--base-safe` for base only, and run variants via our injection path with `--hf-tokenizer` to preserve persona effects while keeping decoding clean.
