from __future__ import annotations

import json

import torch

from persona_steering_library.rl.detectors import LinearProbeDetector


class FakeTokenizer:
    def __call__(self, text, return_tensors="pt", add_special_tokens=False):
        _ = (text, add_special_tokens)
        return {"input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long)}


class FakeModel:
    def __call__(self, input_ids, output_hidden_states=True):
        _ = output_hidden_states
        # shape: (1, T, hidden=3); mean over T -> [1.0, 0.0, 0.0]
        hidden = torch.tensor([[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
        return type("Output", (), {"hidden_states": [hidden]})


def test_linear_probe_detector_load_and_predict(tmp_path):
    meta_path = tmp_path / "detector.json"
    weight_path = tmp_path / "detector.pt"

    meta = {"layer_idx": 0, "hidden_size": 3, "bias": 0.0}
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    torch.save(torch.tensor([1.0, 0.0, 0.0]), weight_path)

    detector = LinearProbeDetector.load(meta_path)
    prob = detector.predict_proba(
        model=FakeModel(),
        tok=FakeTokenizer(),
        text="hello",
        device="cpu",
    )
    expected = torch.sigmoid(torch.tensor(1.0)).item()
    assert abs(prob - expected) < 1e-6
