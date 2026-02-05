from __future__ import annotations

import torch

from persona_steering_library.compute import PersonaVectorResult, _init_model_and_tokenizer


def test_persona_vector_result_roundtrip(tmp_path):
    path = tmp_path / "persona_test.json"
    original = PersonaVectorResult(
        vector=torch.tensor([1.0, -2.0, 0.5]), layer_idx=-1, hidden_size=3
    )
    original.save(path)

    loaded = PersonaVectorResult.load(path)
    assert loaded.layer_idx == -1
    assert loaded.hidden_size == 3
    assert torch.equal(loaded.vector, original.vector)


def test_init_model_unknown_backend_raises():
    try:
        _init_model_and_tokenizer("irrelevant", "cpu", backend="unknown")
    except ValueError as exc:
        assert "Unknown backend" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unknown backend")
