#!/usr/bin/env python3
"""Run lightweight live LLM golden checks with retry/cost guardrails.

Default behavior is safe for local development:
- If `LLM_LIVE_ENABLE` is unset/false, exits 0 with a skipped summary.
- If enabled but keys/config are missing, exits 4.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from statistics import quantiles
from typing import Any

import requests

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_COST = 2
EXIT_INFRA = 3
EXIT_CONFIG = 4

TRANSIENT_CODES = {408, 429, 500, 502, 503, 504}


@dataclass
class LiveCase:
    name: str
    prompt: str
    expected_substring: str


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _extract_output_text(payload: dict[str, Any]) -> str:
    if isinstance(payload.get("output_text"), str) and payload["output_text"]:
        return payload["output_text"]

    chunks: list[str] = []
    for item in payload.get("output", []):
        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"}:
                text = content.get("text", "")
                if isinstance(text, str):
                    chunks.append(text)
    return "".join(chunks).strip()


def _estimate_cost_usd(
    usage: dict[str, Any],
    input_cost_per_1k: float,
    output_cost_per_1k: float,
) -> float:
    # Provider schemas vary, so this is best-effort.
    in_tokens = int(usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0)
    out_tokens = int(usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0)
    return (in_tokens / 1000.0) * input_cost_per_1k + (out_tokens / 1000.0) * output_cost_per_1k


def _call_openai(
    model: str, api_key: str, prompt: str, timeout_s: float
) -> tuple[str, dict[str, Any]]:
    url = "https://api.openai.com/v1/responses"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}],
            }
        ],
    }
    resp = requests.post(url, headers=headers, json=body, timeout=timeout_s)
    if resp.status_code in TRANSIENT_CODES:
        raise RuntimeError(f"transient_http:{resp.status_code}")
    if resp.status_code >= 400:
        raise ValueError(f"deterministic_http:{resp.status_code} {resp.text[:180]}")
    payload = resp.json()
    text = _extract_output_text(payload)
    usage = payload.get("usage", {}) if isinstance(payload, dict) else {}
    return text, usage


def _summary_line(summary: dict[str, Any]) -> str:
    return (
        f"llm-live summary: pass={summary['pass']} fail={summary['fail']} skipped={summary['skipped']} "
        f"calls={summary['calls']} retries={summary['retries']} cost_usd={summary['cost_usd']:.6f} "
        f"p95_ms={summary['p95_ms']:.1f}"
    )


def _p95_ms(samples: list[float]) -> float:
    if not samples:
        return 0.0
    if len(samples) == 1:
        return float(samples[0])
    return float(quantiles(samples, n=20, method="inclusive")[-1])


def main() -> int:
    enabled = env_bool("LLM_LIVE_ENABLE", default=False)
    model = os.getenv("LLM_LIVE_MODEL", os.getenv("RESEARCH_MODEL", "gpt-4o-mini"))
    retry_attempts = int(os.getenv("LLM_LIVE_RETRY_ATTEMPTS", "3"))
    max_calls = int(os.getenv("LLM_LIVE_MAX_CALLS", "9"))
    timeout_s = env_float("LLM_LIVE_TIMEOUT_S", 30.0)
    cost_ceiling = env_float("LLM_LIVE_COST_CEILING_USD", 3.0)
    input_cost_per_1k = env_float("LLM_LIVE_INPUT_COST_PER_1K_USD", 0.0)
    output_cost_per_1k = env_float("LLM_LIVE_OUTPUT_COST_PER_1K_USD", 0.0)

    summary: dict[str, Any] = {
        "pass": 0,
        "fail": 0,
        "skipped": 0,
        "calls": 0,
        "retries": 0,
        "cost_usd": 0.0,
        "p95_ms": 0.0,
    }

    if not enabled:
        summary["skipped"] = 1
        print(_summary_line(summary))
        print(json.dumps({"reason": "LLM_LIVE_ENABLE is false"}, sort_keys=True))
        return EXIT_PASS

    api_key = os.getenv("OPENAI_API_KEY", "").strip().strip('"')
    if not api_key:
        print(_summary_line(summary))
        print(json.dumps({"error": "OPENAI_API_KEY missing for live run"}, sort_keys=True))
        return EXIT_CONFIG

    cases = [
        LiveCase(
            name="ack-token",
            prompt="Reply with exactly ACK and nothing else.",
            expected_substring="ACK",
        ),
        LiveCase(
            name="json-shape",
            prompt='Return a minified JSON object exactly like {"status":"ok"} and nothing else.',
            expected_substring='"status":"ok"',
        ),
    ]

    latencies_ms: list[float] = []
    transient_budget_exhausted = False

    for case in cases:
        case_ok = False
        for attempt in range(1, retry_attempts + 1):
            if summary["calls"] >= max_calls:
                transient_budget_exhausted = True
                break
            summary["calls"] += 1
            t0 = time.perf_counter()
            try:
                text, usage = _call_openai(
                    model=model, api_key=api_key, prompt=case.prompt, timeout_s=timeout_s
                )
            except ValueError as exc:
                # Deterministic/code/config failure: fail fast for this case.
                summary["fail"] += 1
                print(
                    json.dumps(
                        {"case": case.name, "attempt": attempt, "error": str(exc)}, sort_keys=True
                    )
                )
                break
            except Exception as exc:  # transient/infrastructure
                summary["retries"] += 1
                if attempt >= retry_attempts:
                    transient_budget_exhausted = True
                    print(
                        json.dumps(
                            {"case": case.name, "attempt": attempt, "error": str(exc)},
                            sort_keys=True,
                        )
                    )
                    break
                time.sleep(0.5 * attempt)
                continue

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(elapsed_ms)
            summary["cost_usd"] += _estimate_cost_usd(
                usage=usage,
                input_cost_per_1k=input_cost_per_1k,
                output_cost_per_1k=output_cost_per_1k,
            )

            if summary["cost_usd"] > cost_ceiling:
                summary["p95_ms"] = _p95_ms(latencies_ms)
                print(_summary_line(summary))
                print(
                    json.dumps(
                        {"error": "cost budget exceeded", "ceiling_usd": cost_ceiling},
                        sort_keys=True,
                    )
                )
                return EXIT_COST

            if case.expected_substring in text:
                summary["pass"] += 1
                case_ok = True
                break

            # deterministic mismatch; do not retry endlessly
            summary["fail"] += 1
            print(
                json.dumps(
                    {"case": case.name, "attempt": attempt, "error": "substring_missing"},
                    sort_keys=True,
                )
            )
            break

        if transient_budget_exhausted:
            break
        if not case_ok and summary["fail"] == 0:
            # Any non-pass that wasn't explicitly marked fail is treated as fail.
            summary["fail"] += 1

    summary["p95_ms"] = _p95_ms(latencies_ms)
    print(_summary_line(summary))

    if transient_budget_exhausted and summary["fail"] == 0:
        print(json.dumps({"error": "infrastructure failure"}, sort_keys=True))
        return EXIT_INFRA
    if summary["fail"] > 0:
        return EXIT_FAIL
    return EXIT_PASS


if __name__ == "__main__":
    raise SystemExit(main())
