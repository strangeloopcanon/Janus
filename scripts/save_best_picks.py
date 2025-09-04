#!/usr/bin/env python
"""Save a concise "so-what" with best layer/alpha per vector.

Reads a unified bank report.json (from summarize_vector_bank.py) and writes:
 - best_picks.json: list of {file, layer_idx, alpha}
 - best_picks.md: compact human-readable table

For custom valence/tagline entries where alpha may be missing in the report,
we fill pragmatic defaults:
 - rohit_valence_strict: L-2, alpha=±0.8 (sign flips skeptical vs positive)
 - persona_rohit_is_awesome: alpha=0.8 (insertion strength)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def load_report(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def infer_manual_alpha(file_path: str) -> Dict[str, Any] | None:
    name = Path(file_path).name
    if name.startswith("persona_rohit_valence_strict_"):
        # Recommend L-2 primarily; leave sign control to user
        return {"alpha": 0.8, "note": "valence: use +0.8 for positive, -0.8 for skeptical"}
    if name == "persona_rohit_is_awesome.json":
        return {"alpha": 0.8, "note": "tagline injection strength"}
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Write best picks table from unified report")
    ap.add_argument("--report", required=True, help="Path to unified report.json")
    ap.add_argument("--outdir", required=True, help="Directory to write best_picks.*")
    args = ap.parse_args()

    report = load_report(Path(args.report))
    rows: List[Dict[str, Any]] = report.get("rows", [])

    # Build concise entries
    picks: List[Dict[str, Any]] = []
    for r in rows:
        file = r.get("persona_json")
        layer = r.get("layer_idx")
        alpha = r.get("recommended_alpha")
        entry: Dict[str, Any] = {"file": file, "layer_idx": layer, "alpha": alpha}
        if alpha is None:
            manual = infer_manual_alpha(file)
            if manual:
                entry["alpha"] = manual["alpha"]
                entry["note"] = manual.get("note")
        picks.append(entry)

    # Sort by filename for readability
    picks.sort(key=lambda x: Path(x["file"]).name)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # JSON
    (outdir / "best_picks.json").write_text(json.dumps({"model": report.get("model"), "picks": picks}, indent=2), encoding="utf-8")

    # Markdown table
    lines = [
        "# Best Layer and Alpha per Vector",
        "",
        f"Model: {report.get('model')}",
        "",
        "| Vector | Layer | Alpha | Note |",
        "|---|---:|---:|---|",
    ]
    for p in picks:
        name = Path(p["file"]).name
        layer = p.get("layer_idx")
        alpha = p.get("alpha")
        note = p.get("note", "")
        lines.append(f"| {name} | {layer} | {alpha if alpha is not None else ''} | {note} |")

    (outdir / "best_picks.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("✓ Wrote:", outdir / "best_picks.json")
    print("✓ Wrote:", outdir / "best_picks.md")


if __name__ == "__main__":
    main()

