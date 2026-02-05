from __future__ import annotations

import os

from scripts import run_llm_live


def test_extract_output_text_from_payload():
    payload = {"output": [{"content": [{"type": "output_text", "text": "hello"}]}]}
    assert run_llm_live._extract_output_text(payload) == "hello"


def test_estimate_cost_usd():
    usage = {"input_tokens": 500, "output_tokens": 250}
    cost = run_llm_live._estimate_cost_usd(usage, input_cost_per_1k=0.01, output_cost_per_1k=0.02)
    assert abs(cost - 0.01) < 1e-9


def test_summary_line_format():
    summary = {
        "pass": 1,
        "fail": 0,
        "skipped": 0,
        "calls": 2,
        "retries": 1,
        "cost_usd": 0.0,
        "p95_ms": 12.3,
    }
    line = run_llm_live._summary_line(summary)
    assert "pass=1" in line
    assert "p95_ms=12.3" in line


def test_llm_live_skips_by_default(monkeypatch):
    monkeypatch.delenv("LLM_LIVE_ENABLE", raising=False)
    rc = run_llm_live.main()
    assert rc == run_llm_live.EXIT_PASS


def test_llm_live_requires_key_when_enabled(monkeypatch):
    monkeypatch.setenv("LLM_LIVE_ENABLE", "1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    rc = run_llm_live.main()
    assert rc == run_llm_live.EXIT_CONFIG


def test_llm_live_success_with_stubbed_provider(monkeypatch):
    monkeypatch.setenv("LLM_LIVE_ENABLE", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("LLM_LIVE_RETRY_ATTEMPTS", "1")
    monkeypatch.setenv("LLM_LIVE_MAX_CALLS", "9")
    monkeypatch.setenv("LLM_LIVE_COST_CEILING_USD", "3")

    responses = iter(
        [
            ("ACK", {"input_tokens": 10, "output_tokens": 1}),
            ('{"status":"ok"}', {"input_tokens": 10, "output_tokens": 2}),
        ]
    )

    def fake_call_openai(model: str, api_key: str, prompt: str, timeout_s: float):
        _ = (model, api_key, prompt, timeout_s)
        return next(responses)

    monkeypatch.setattr(run_llm_live, "_call_openai", fake_call_openai)
    rc = run_llm_live.main()
    assert rc == run_llm_live.EXIT_PASS

    # Defensive check to ensure test doesn't leak env toggles into other tests.
    os.environ.pop("LLM_LIVE_ENABLE", None)


def test_llm_live_cost_budget_exit(monkeypatch):
    monkeypatch.setenv("LLM_LIVE_ENABLE", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("LLM_LIVE_RETRY_ATTEMPTS", "1")
    monkeypatch.setenv("LLM_LIVE_MAX_CALLS", "9")
    monkeypatch.setenv("LLM_LIVE_COST_CEILING_USD", "0.001")
    monkeypatch.setenv("LLM_LIVE_INPUT_COST_PER_1K_USD", "0.1")
    monkeypatch.setenv("LLM_LIVE_OUTPUT_COST_PER_1K_USD", "0.1")

    def fake_call_openai(model: str, api_key: str, prompt: str, timeout_s: float):
        _ = (model, api_key, prompt, timeout_s)
        return "ACK", {"input_tokens": 1000, "output_tokens": 1000}

    monkeypatch.setattr(run_llm_live, "_call_openai", fake_call_openai)
    rc = run_llm_live.main()
    assert rc == run_llm_live.EXIT_COST


def test_llm_live_infrastructure_exit(monkeypatch):
    monkeypatch.setenv("LLM_LIVE_ENABLE", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("LLM_LIVE_RETRY_ATTEMPTS", "1")
    monkeypatch.setenv("LLM_LIVE_MAX_CALLS", "9")

    def fake_call_openai(model: str, api_key: str, prompt: str, timeout_s: float):
        _ = (model, api_key, prompt, timeout_s)
        raise RuntimeError("transient_http:429")

    monkeypatch.setattr(run_llm_live, "_call_openai", fake_call_openai)
    rc = run_llm_live.main()
    assert rc == run_llm_live.EXIT_INFRA


def test_llm_live_deterministic_failure(monkeypatch):
    monkeypatch.setenv("LLM_LIVE_ENABLE", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("LLM_LIVE_RETRY_ATTEMPTS", "1")
    monkeypatch.setenv("LLM_LIVE_MAX_CALLS", "9")

    def fake_call_openai(model: str, api_key: str, prompt: str, timeout_s: float):
        _ = (model, api_key, prompt, timeout_s)
        return "NOT_ACK", {"input_tokens": 0, "output_tokens": 0}

    monkeypatch.setattr(run_llm_live, "_call_openai", fake_call_openai)
    rc = run_llm_live.main()
    assert rc == run_llm_live.EXIT_FAIL
