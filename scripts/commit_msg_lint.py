#!/usr/bin/env python3
"""Minimal Conventional Commit checker for git commit-msg hook."""

from __future__ import annotations

import re
import sys
from pathlib import Path

PATTERN = re.compile(
    r"^(feat|fix|chore|docs|style|refactor|perf|test|build|ci|revert)(\([^)]+\))?!?: .+"
)


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: commit_msg_lint.py <commit-msg-file>")
        return 2

    msg_path = Path(sys.argv[1])
    first_line = msg_path.read_text(encoding="utf-8").splitlines()[0].strip()

    if first_line.startswith("Merge ") or first_line.startswith("Revert "):
        return 0
    if PATTERN.match(first_line):
        return 0

    print(
        "Commit message must follow Conventional Commits. "
        "Example: 'chore(ci): add make all workflow'",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
