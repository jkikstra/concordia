"""
Driver for comparing two CMIP7 gridded scenario version folders.

Imports run_comparison() directly from notebooks/cmip7/compare_gridded_versions.py
and runs it for the two configured folders.

Does NOT use papermill — just plain Python imports.

Usage:
    python scripts/cmip7/driver_compare_gridded_versions.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# ── Locate repo root and add notebooks/cmip7 to path ─────────────────────────
HERE = Path(__file__).parent.parent.parent  # repo root (concordia/)
NOTEBOOKS_CMIP7 = HERE / "notebooks" / "cmip7"
if str(NOTEBOOKS_CMIP7) not in sys.path:
    sys.path.insert(0, str(NOTEBOOKS_CMIP7))

from compare_gridded_versions import run_comparison  # noqa: E402


def main() -> None:
    """
    Configure and run the version comparison.

    Edit the configuration block below for your two folders.
    """

    # ── CONFIGURATION ─────────────────────────────────────────────────────────

    # Paths to the two gridded version folders.
    # Each folder should contain *.nc files named:
    #   {gas}-em-{type}_{FILE_NAME_ENDING}.nc
    marker = "vl" # l

    FOLDER_A = f"C:/Users/kikstra/IIASA/ECE.prog - Documents/Projects/CMIP7/IAM Data Processing/Shared emission fields data/v1_1-testing-findmistakes/{marker}_1-1-0"
    FOLDER_B = f"C:/Users/kikstra/IIASA/ECE.prog - Documents/Projects/CMIP7/IAM Data Processing/Shared emission fields data/v1_1-testing-findmistakes/{marker}_1-1-1"
    
    # Short labels used in log messages and metadata diff file headers.
    LABEL_A = "v1-1-0"
    LABEL_B = "v1-1-1"

    # Where to write all comparison outputs.
    # Leave as "" to auto-derive as FOLDER_A / "qc_output_v2" / "version_comparison".
    # Can be any absolute path, e.g.:
    #   OUTPUT_DIR = "C:/path/to/my/comparison_output"
    OUTPUT_DIR = f"C:/Users/kikstra/IIASA/ECE.prog - Documents/Projects/CMIP7/IAM Data Processing/Shared emission fields data/v1_1-testing-findmistakes/{marker}_compare_1-1-1_to_1-1-0"

    # Optional: restrict to a subset of species (None = all).
    species_filter = None
    # species_filter = ["CO2", "H2"]   # quick test

    # If True, skip a (gas, file_type) pair whose per-file _data.txt already exists.
    skip_existing = False

    # ── RESOLVE OUTPUT DIR ────────────────────────────────────────────────────
    folder_a_path = Path(FOLDER_A)
    folder_b_path = Path(FOLDER_B)
    output_dir = (
        Path(OUTPUT_DIR)
        if OUTPUT_DIR
        else folder_a_path / "qc_output_v2" / "version_comparison"
    )

    # ── RUN ───────────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  Version comparison: {LABEL_A}  vs  {LABEL_B}")
    print(f"  Folder A ({LABEL_A}): {folder_a_path}")
    print(f"  Folder B ({LABEL_B}): {folder_b_path}")
    print(f"  Output: {output_dir}")
    print(f"{'='*65}")

    results = run_comparison(
        folder_a=folder_a_path,
        folder_b=folder_b_path,
        output_dir=output_dir,
        label_a=LABEL_A,
        label_b=LABEL_B,
        species_filter=species_filter,
        skip_existing=skip_existing,
    )

    print("\n  Comparison outputs:")
    for key, path in results.items():
        print(f"    {key}: {path}")

    print("\nAll done.")


if __name__ == "__main__":
    main()
