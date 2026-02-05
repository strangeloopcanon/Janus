SHELL := /bin/bash

VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

export PYTHONPATH := .

MODE ?= $(shell if [ -f .agents.yml ]; then awk '/^mode:/ {print $$2; exit}' .agents.yml; else echo baseline; fi)
COVERAGE_FAIL_UNDER := 80
ifeq ($(MODE),production)
COVERAGE_FAIL_UNDER := 90
endif

.PHONY: setup bootstrap check test llm-live deps-audit all release clean

setup bootstrap:
	python3 -m venv $(VENV)
	$(PYTHON) -m pip install --upgrade pip
	$(PIP) install -r requirements.txt -r requirements-dev.txt
	$(VENV)/bin/pre-commit install --install-hooks -f || true
	@mkdir -p .git/hooks
	@printf '%s\n' '#!/bin/sh' 'exec "$$(pwd)/$(PYTHON)" "$$(pwd)/scripts/commit_msg_lint.py" "$$1"' > .git/hooks/commit-msg
	@chmod +x .git/hooks/commit-msg

check:
	$(VENV)/bin/black --check persona_steering_library tests
	$(VENV)/bin/ruff check persona_steering_library scripts tests
	$(VENV)/bin/mypy persona_steering_library --ignore-missing-imports --follow-imports=silent
	-$(VENV)/bin/bandit -q -r persona_steering_library
	$(VENV)/bin/detect-secrets-hook --baseline .secrets.baseline $$(git ls-files)

test:
	PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $(VENV)/bin/pytest \
		tests/test_compute.py \
		tests/test_hooks.py \
		tests/test_rl_detectors.py \
		tests/test_rl_gspo.py
	PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $(VENV)/bin/coverage erase
	PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $(VENV)/bin/coverage run \
		--source=scripts.commit_msg_lint,scripts.run_llm_live \
		-m pytest \
		tests/test_commit_msg_lint.py \
		tests/test_run_llm_live.py
	$(VENV)/bin/coverage report --show-missing --fail-under=$(COVERAGE_FAIL_UNDER)

llm-live:
	$(PYTHON) scripts/run_llm_live.py

deps-audit:
ifeq ($(MODE),production)
	$(VENV)/bin/pip-audit
else
	-$(VENV)/bin/pip-audit
endif

all: check test llm-live
ifeq ($(MODE),production)
	@$(MAKE) deps-audit
endif

release:
	@echo "release target is only implemented for AGENT_MODE=production"
	@exit 4

clean:
	rm -rf $(VENV) .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
