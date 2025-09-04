#!/usr/bin/env python
"""Organize persona artifacts into a tidy structure.

Moves legacy/duplicate files under personas/ into archive folders:
  - archive/legacy_0p6B/   (old 0.6B or early experiments)
  - archive/scratch/       (one-off demos/tests)
  - archive/traits_copies/ (duplicate trait vectors already in personas_1p7B)

Keeps as live:
  - bank_unified_1p7B/ (working set)
  - rohit_valence_strict/ (valence vectors)
  - converted/ (target for conversions)
  - persona_honest_for_1p7B_L-1.{json,pt}
  - persona_hidden_marker.{json,pt}
  - alignment_* (alignment files for conversions)

Dry-run by default; pass --apply to perform moves.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List


KEEP_DIRS = {
    "bank_unified_1p7B",
    "rohit_valence_strict",
    "converted",
}

KEEP_FILES_PREFIX = (
    "alignment_Qwen3-0.6B_to_Qwen3-1.7B_",
)

KEEP_FILES_EXACT = {
    "persona_honest_for_1p7B_L-1.json",
    "persona_honest_for_1p7B_L-1.pt",
    "persona_hidden_marker.json",
    "persona_hidden_marker.pt",
    "README.md",
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Organize personas into archive folders")
    ap.add_argument("--root", default="personas", help="Personas root directory")
    ap.add_argument("--apply", action="store_true", help="Actually move files (default: dry-run)")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        raise SystemExit(f"Root not found: {root}")

    # Ensure archive dirs
    arch_legacy = root / "archive" / "legacy_0p6B"
    arch_scratch = root / "archive" / "scratch"
    arch_traits = root / "archive" / "traits_copies"
    for d in (arch_legacy, arch_scratch, arch_traits):
        d.mkdir(parents=True, exist_ok=True)

    # Identify moves
    moves: List[tuple[Path, Path]] = []

    # Helper classifiers
    def is_keep_file(p: Path) -> bool:
        if p.name in KEEP_FILES_EXACT:
            return True
        return any(p.name.startswith(pref) for pref in KEEP_FILES_PREFIX)

    def is_trait_vector(p: Path) -> bool:
        name = p.name
        return name.startswith("persona_") and ("_L-" in name) and (p.suffix in {".json", ".pt"})

    # Top-level scan
    for child in root.iterdir():
        if child.name.startswith("archive"):
            continue
        if child.is_dir():
            if child.name in KEEP_DIRS:
                continue
            # Move test/temporary dirs to scratch
            if child.name in {"bank_unified_1p7B_test"}:
                moves.append((child, arch_scratch / child.name))
            # Leave other dirs (like converted/) already covered; unknown dirs → scratch
            elif child.name not in KEEP_DIRS and child.name != "converted":
                moves.append((child, arch_scratch / child.name))
            continue

        # Files
        if is_keep_file(child):
            continue

        # Legacy/demos
        if any(child.name.startswith(pref) for pref in (
            "persona_formal", "persona_creative", "persona_covert", "persona_converter_demo",
            "persona_covert_v2_mlx_test", "gpt_oss_simple",
        )):
            moves.append((child, arch_legacy / child.name))
            continue

        # Trait duplicates at top-level
        if is_trait_vector(child):
            moves.append((child, arch_traits / child.name))
            continue

        # Vector bank manifests/reports at top-level → scratch
        if child.name.startswith("vector_bank_"):
            moves.append((child, arch_scratch / child.name))
            continue

        # Default: leave as-is (e.g., persona_honest_for_1p7B, hidden_marker, alignment files)

    # Print plan
    if not moves:
        print("No moves required; structure already tidy.")
        return

    print(f"Planned moves ({len(moves)}):")
    for src, dst in moves:
        print(" -", src.relative_to(root), "->", dst.relative_to(root))

    if not args.apply:
        print("\nDry-run only. Re-run with --apply to perform moves.")
        return

    # Apply moves
    for src, dst in moves:
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.replace(src, dst)
    print("✓ Moves applied.")


if __name__ == "__main__":
    main()

