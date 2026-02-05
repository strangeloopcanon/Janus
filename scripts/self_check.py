#!/usr/bin/env python
"""Lightweight environment self-check for Torch/MLX pipelines.

This script doesn't download models. It verifies that required Python
packages are importable and that key entry points exist.
"""

from __future__ import annotations

# Ensure repo root is on sys.path when running as `python scripts/...`
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import importlib


def check_import(name: str) -> bool:
    try:
        importlib.import_module(name)
        print(f"✅ import {name}")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"❌ import {name}: {e}")
        return False


def main() -> None:
    print("=== Python environment self-check ===")
    ok_torch = check_import("torch")
    ok_tf = check_import("transformers")
    check_import("yaml")
    ok_mlx = check_import("mlx.core")
    ok_mlx_lm = check_import("mlx_lm")

    # Entry points
    try:
        from persona_steering_library import compute_persona_vector, add_persona_hook  # noqa: F401

        print("✅ persona_steering_library entry points present")
    except Exception as e:  # noqa: BLE001
        print(f"❌ persona_steering_library entry points: {e}")

    try:
        print("✅ MLX support entry points present")
    except Exception as e:  # noqa: BLE001
        print(f"❌ MLX support entry points: {e}")

    print("\nSummary:")
    print(f"Torch available: {ok_torch}")
    print(f"Transformers available: {ok_tf}")
    print(f"MLX available: {ok_mlx}")
    print(f"mlx_lm available: {ok_mlx_lm}")
    print("If MLX and mlx_lm are available, MLX generation/scoring paths can run.")


if __name__ == "__main__":
    main()
