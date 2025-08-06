"""Persona Vectors – lightweight utilities to reproduce Anthropic's persona steering
on open-source LLMs (e.g. Qwen-3 0.6B).

High-level API
---------------
from persona_vectors.compute import compute_persona_vector
from persona_vectors.hooks import add_persona_hook

See the docstrings of those functions for usage details.
"""

__all__ = [
    "compute_persona_vector",
    "add_persona_hook",
    "PersonaVectorResult",
]

from .compute import compute_persona_vector, PersonaVectorResult  # noqa: E402  pylint: disable=C0413
from .hooks import add_persona_hook  # noqa: E402  pylint: disable=C0413
