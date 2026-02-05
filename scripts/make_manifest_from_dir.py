#!/usr/bin/env python
"""Create a small manifest JSON from a directory of persona_*.json files.

This mirrors the manifest structure produced by build_vector_bank.py so you can
use calibrate_alpha.py and summarize_vector_bank.py on custom folders.

Usage:
  python scripts/make_manifest_from_dir.py \
    --dir personas/rohit_valence_strict \
    --model Qwen/Qwen3-1.7B \
    --out personas/rohit_valence_strict/manifest.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any


def main() -> None:
    ap = argparse.ArgumentParser(description="Make a vector-bank-style manifest from a folder")
    ap.add_argument("--dir", required=True, help="Directory containing persona_*.json files")
    ap.add_argument(
        "--model", required=True, help="HF model used to train these vectors (for metadata)"
    )
    ap.add_argument("--out", required=True, help="Where to write the manifest JSON")
    args = ap.parse_args()

    d = Path(args.dir)
    files = sorted([p for p in d.glob("persona_*.json") if p.is_file()])
    if not files:
        raise SystemExit(f"No persona_*.json files found under {d}")

    entries: List[Dict[str, Any]] = []
    for jpath in files:
        try:
            meta = json.loads(jpath.read_text(encoding="utf-8"))
            layer_idx = int(meta.get("layer_idx", -1))
            hidden_size = int(meta.get("hidden_size", 0))
        except Exception:
            # Fallback: try parsing layer from filename persona_<name>_L-<k>.json
            import re

            m = re.search(r"_L(-?\d+)\.json$", jpath.name)
            layer_idx = int(m.group(1)) if m else -1
            hidden_size = None  # unknown
        entries.append(
            {
                "persona_json": str(jpath),
                "persona_tensor": str(jpath.with_suffix(".pt")),
                "layer_idx": layer_idx,
                "hidden_size": hidden_size,
            }
        )

    manifest = {
        "model": args.model,
        "backend": "unknown",
        "config": {},
        "entries": entries,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"✓ Wrote manifest with {len(entries)} entries to: {out}")


if __name__ == "__main__":
    main()
