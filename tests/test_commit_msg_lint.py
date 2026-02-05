from __future__ import annotations

import sys

from scripts import commit_msg_lint


def _run(msg: str, tmp_path) -> int:
    path = tmp_path / "commit_msg.txt"
    path.write_text(msg, encoding="utf-8")
    old_argv = sys.argv[:]
    try:
        sys.argv = ["commit_msg_lint.py", str(path)]
        return commit_msg_lint.main()
    finally:
        sys.argv = old_argv


def test_conventional_commit_passes(tmp_path):
    rc = _run("chore(ci): add gate workflow\n\nbody", tmp_path)
    assert rc == 0


def test_invalid_commit_fails(tmp_path):
    rc = _run("random message", tmp_path)
    assert rc == 1


def test_merge_commit_passes(tmp_path):
    rc = _run("Merge branch 'feature/x' into main", tmp_path)
    assert rc == 0


def test_usage_error_when_missing_arg(monkeypatch):
    old_argv = sys.argv[:]
    try:
        sys.argv = ["commit_msg_lint.py"]
        rc = commit_msg_lint.main()
    finally:
        sys.argv = old_argv
    assert rc == 2
