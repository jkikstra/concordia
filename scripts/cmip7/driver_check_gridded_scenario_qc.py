"""
Driver for the CMIP7 gridded scenario QC checks.

Imports run_qc() directly from notebooks/cmip7/check_gridded_scenario_qc.py
and runs it for one or more scenario markers.

Does NOT use papermill — just plain Python imports.

Usage:
    python scripts/cmip7/driver_check_gridded_scenario_qc.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# ── Locate repo root and add notebooks/cmip7 to path ─────────────────────────
HERE = Path(__file__).parent.parent.parent  # repo root (concordia/)
NOTEBOOKS_CMIP7 = HERE / "notebooks" / "cmip7"
if str(NOTEBOOKS_CMIP7) not in sys.path:
    sys.path.insert(0, str(NOTEBOOKS_CMIP7))

from check_gridded_scenario_qc import run_qc  # noqa: E402


def main() -> None:
    """
    Configure and run QC for one or more scenario markers.

    Edit the configuration block below to select which markers and modules
    to run.
    """

    # ── CONFIGURATION ─────────────────────────────────────────────────────────

    SETTINGS_FILE = "config_cmip7_v0-4-0.yaml"
    VERSION_ESGF = "1-1-1"
    HISTORY_FILE = (
        "country-history_202511261223_202511040855_202512032146_"
        "202512021030_7e32405ade790677a6022ff498395bff00d9792d.csv"
    )

    # Which scenario markers to QC.  Comment/uncomment as needed.
    markers = [
        # "h",
        "hl",
        # "m",
        # "ml",
        # "l",
        # "ln",
        # "vl",
    ]

    # Optional version prefix for the output folder name, e.g. "test_"
    # Leave as "" for the standard naming: "{marker}_{VERSION_ESGF}"
    GRIDDING_VERSION_PREFIX = ""

    # ── MODULE FLAGS ──────────────────────────────────────────────────────────
    # Set each flag to True/False to enable/disable individual QC modules.

    run_file_inventory = False   # A: list files, check for missing
    run_min_max = False          # B: per-file min/max statistics
    run_downscaled_qc = False    # C: workflow QC checks on downscaled CSV
    run_annual_totals = False    # D: 3-way comparison of annual totals
    run_animations = False      # E: animated GIF maps — SLOW; enable manually
    # "all-sectors"       → one GIF per (gas, file_type, sector)  e.g. BC_anthro-Energy
    # "total-per-file"    → sectors summed within each file        e.g. BC_anthro-total
    # "total-per-species" → all files summed per gas               e.g. BC-total
    # Can be a single string or a list to run multiple modes in one pass.
    # animation_mode = "all-sectors"
    animation_mode = ["all-sectors", "total-per-file", "total-per-species"]
    run_doc_plots = True        # F: documentation plots 03 and 04
    run_place_timeseries = True  # G: per-location timeseries vs CEDS history — SLOW; enable manually

    # ── SPECIES FILTER ────────────────────────────────────────────────────────
    # Set to None to run all species, or a list for a faster test run.
    species_filter = None
    # species_filter = ["CH4"]   # quick test

    # ── PERFORMANCE ──────────────────────────────────────────────────────────
    # If True, skip a module if its output CSV/plots already exist.
    skip_existing = False

    # ── FOLDER OVERRIDE ───────────────────────────────────────────────────────
    # Set to a specific path to override the default results/{gridding_version}
    # folder per marker. Use "" to derive the folder automatically from
    # VERSION_ESGF and the marker.
    # Can be a single string (applied to all markers) or a dict mapping
    # marker -> path for per-marker overrides. Any marker not in the dict
    # falls back to the automatic folder.
    # FOLDER_WITH_GRIDDED_DATA = ""
    # FOLDER_WITH_GRIDDED_DATA = "C:/path/to/your/gridded/data"  # all markers
    FOLDER_WITH_GRIDDED_DATA = {
        # "h": "C:/Users/kikstra/IIASA/ECE.prog - Documents/Projects/CMIP7/IAM Data Processing/Shared emission fields data/v1_1-testing-findmistakes/h_1-1-0",
        "hl": "C:/Users/kikstra/IIASA/ECE.prog - Documents/Projects/CMIP7/IAM Data Processing/Shared emission fields data/v1_1-testing-findmistakes/hl_1-1-1",
        "m":  "C:/Users/kikstra/IIASA/ECE.prog - Documents/Projects/CMIP7/IAM Data Processing/Shared emission fields data/v1_1-testing-findmistakes/m_1-1-1",
        # "ml": "C:/Users/kikstra/IIASA/ECE.prog - Documents/Projects/CMIP7/IAM Data Processing/Shared emission fields data/v1_1-testing-findmistakes/ml_1-1-0",
        # "l": "..."
        # "ln": "C:/Users/kikstra/IIASA/ECE.prog - Documents/Projects/CMIP7/IAM Data Processing/Shared emission fields data/v1_1-testing-findmistakes/ln_1-1-0",
        # "vl": "C:/Users/kikstra/IIASA/ECE.prog - Documents/Projects/CMIP7/IAM Data Processing/Shared emission fields data/v1_1-testing-findmistakes/vl_1-1-0"
    }

    # ── RUN ───────────────────────────────────────────────────────────────────

    results_path = HERE / "results"

    for marker in markers:
        gridding_version = f"{GRIDDING_VERSION_PREFIX}{marker}_{VERSION_ESGF}"
        if isinstance(FOLDER_WITH_GRIDDED_DATA, dict):
            override = FOLDER_WITH_GRIDDED_DATA.get(marker, "")
        else:
            override = FOLDER_WITH_GRIDDED_DATA
        gridded_scenario_folder = (
            Path(override) if override else results_path / gridding_version
        )

        if not gridded_scenario_folder.exists():
            print(
                f"\n[SKIP] Folder not found for marker '{marker}': "
                f"{gridded_scenario_folder}"
            )
            continue

        print(f"\n{'='*65}")
        print(f"  QC run: marker={marker}  version={gridding_version}")
        print(f"  Folder: {gridded_scenario_folder}")
        print(f"{'='*65}")

        qc_results = run_qc(
            gridded_scenario_folder=gridded_scenario_folder,
            marker_to_run=marker,
            settings_file=SETTINGS_FILE,
            gridding_version=gridding_version,
            version_esgf=VERSION_ESGF,
            history_file=HISTORY_FILE,
            run_file_inventory=run_file_inventory,
            run_min_max=run_min_max,
            run_downscaled_qc=run_downscaled_qc,
            run_annual_totals=run_annual_totals,
            run_animations=run_animations,
            animation_mode=animation_mode,
            run_doc_plots=run_doc_plots,
            run_place_timeseries=run_place_timeseries,
            species_filter=species_filter,
            skip_existing=skip_existing,
            here=NOTEBOOKS_CMIP7,
        )

        print(f"\n  QC outputs for '{marker}':")
        for module, path in qc_results.items():
            if isinstance(path, list):
                print(f"    {module}: {len(path)} files")
            else:
                print(f"    {module}: {path}")

    print("\nAll done.")


if __name__ == "__main__":
    main()
