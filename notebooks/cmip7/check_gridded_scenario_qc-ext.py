# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
from __future__ import annotations

# %% [markdown]
# # Comprehensive QC for CMIP7 Gridded Scenario Emissions
#
# ## Expected File Tree
#
# This script assumes the following directory layout:
#
# ```
# concordia/                                    ← repo root (auto-detected via pyproject.toml)
# ├── notebooks/
# │   └── cmip7/
# │       ├── check_gridded_scenario_qc.py      ← this script
# │       └── {SETTINGS_FILE}                   ← e.g. config_cmip7_v0-4-0.yaml
# │
# ├── results/
# │   └── {GRIDDING_VERSION}/                   ← e.g. m_1-1-0/  (= gridded_scenario_folder)
# │       ├── *.nc                              ← gridded NetCDF files (anthro, openburning, AIR-anthro)
# │       │                                        named: {gas}-em-{type}_{file_name_ending}.nc
# │       ├── scenarios_processed.csv           ← input IAM scenario data  [modules D, F]
# │       ├── harmonization-{version}.csv       ← harmonized data           [module D]
# │       ├── downscaled-only-{version}.csv     ← downscaled country data   [module C]
# │       └── qc_output/                        ← created by this script
# │           ├── logs/
# │           ├── tables/
# │           │   ├── file_inventory.csv
# │           │   ├── min_max_stats.csv
# │           │   ├── downscaled_qc_results.csv
# │           │   └── annual_totals_comparison.csv
# │           └── plots/
# │               ├── annual_totals_{gas}_{type}.png
# │               ├── 03_total_emissions_with_history.png
# │               ├── 04_stacked_bars_{marker}.png
# │               └── animations/  (if run_animations=True)
# │
# └── (paths below come from settings, i.e. {SETTINGS_FILE})
#     ├── {settings.gridding_path}/
#     │   └── areacella_input4MIPs_emissions_CMIP_CEDS-CMIP-2025-04-18_gn.nc
#     │                                         ← grid cell areas            [modules D, E]
#     └── {settings.history_path}/
#         └── {HISTORY_FILE}                    ← country-level history CSV  [module F]
# ```
#
# **Key assumption:** `FOLDER_WITH_GRIDDED_DATA` is left empty, so the gridded scenario
# folder is derived automatically as `<repo_root>/results/{GRIDDING_VERSION}/`.
# Set `FOLDER_WITH_GRIDDED_DATA` to an absolute path to override this.
#
# Runs all quality checks on a single scenario's gridded output:
# - Module A: File inventory
# - Module B: Min/max value statistics per file
# - Module C: Downscaled data QC (replicating workflow checks)
# - Module D: Annual totals 3-way comparison (input / harmonized / gridded)
#             NOTE: CSV output works; PNG plots are broken (TODO: fix or replace)
# - Module E: Animated grid maps (fast PIL-based GIFs)
# - Module F: Documentation plots 03 and 04
# - Module G: Per-location timeseries vs CEDS history (mirrors workflow §4.1; slow, off by default)
# - Module H: Per-location timeseries vs BB4CMIP7 history for openburning (slow, off by default)

# %% [markdown]
# ## Parameters

# %% editable=true slideshow={"slide_type": ""} tags=["parameters"]
SETTINGS_FILE: str = "config_cmip7_v0-4-0.yaml"
VERSION_ESGF: str = "1-1-1"
marker_to_run: str = "vl"
GRIDDING_VERSION: str = f"{marker_to_run}-ext_{VERSION_ESGF}"
HISTORY_FILE: str = f"downscaled-only-{marker_to_run}_1-1-0.csv" # should update to 1-1-1 once we have them
FOLDER_WITH_GRIDDED_DATA: str = ""  # set to path of gridded scenario folder, or leave empty to derive from GRIDDING_VERSION

# Module flags
run_file_inventory: bool = True
run_min_max: bool = True
run_downscaled_qc: bool = True
run_annual_totals: bool = True
run_animations: bool = False  # slow; enable manually when needed
run_doc_plots: bool = True
run_place_timeseries: bool = False  # slow; enable manually when needed

# Optional: restrict to a subset of species (None = all)
species_filter: list[str] | None = None  # set to e.g. ["BC", "CO2"] to restrict

# Skip modules whose outputs already exist on disk
skip_existing: bool = True

# %% [markdown]
# ## Imports

# %%
import concurrent.futures
import datetime
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional
import dask

import cartopy.crs as ccrs
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import colors as mcolors
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.pyplot as plt
from PIL import Image

import pandas_indexing as pix
from pandas_indexing import assignlevel, extractlevel, isin, ismatch

from concordia.cmip7 import utils as cmip7_utils
from concordia.cmip7.utils_plotting import (
    ds_to_annual_emissions_total_faster,
    plot_place_timeseries,
    plot_place_area_average_timeseries,
)
from concordia.cmip7.CONSTANTS import (
    CMIP_ERA,
    GASES_ESGF_CEDS,
    GASES_ESGF_BB4CMIP,
    return_marker_information,
)
from concordia.settings import Settings

# %%
# ── Constants ─────────────────────────────────────────────────────────────────

IAMC_COLS = ["model", "scenario", "region", "variable", "unit"]

CDR_SECTORS = [
    "Direct Air Capture",
    "Other CDR",
    "Enhanced Weathering",
    "BECCS",
    "Ocean",
    "Biochar",
    "Soil Carbon Management",
    "Other Capture and Removal",  # used in gridded sector names
]
CDR_SECTORS_MUST_BE_NEGATIVE = [
    "Direct Air Capture",
    "Enhanced Weathering",
    "BECCS",
]

FILE_TYPES = ["anthro", "openburning", "AIR-anthro"]

SECTOR_FILE_DICT = {
    "openburning": [
        "Agricultural Waste Burning",
        "Forest Burning",
        "Grassland Burning",
        "Peat Burning",
    ],
    "anthro": [
        "Agriculture",
        "BECCS",
        "Biochar",
        "Direct Air Capture",
        "Energy Sector",
        "Enhanced Weathering",
        "Industrial Sector",
        "International Shipping",
        "Ocean",
        "Other CDR",
        "Residential Commercial Other",
        "Soil Carbon Management",
        "Solvents Production and Application",
        "Transportation Sector",
        "Waste",
    ],
    "AIR-anthro": ["Aircraft"],
}

SECTOR_FILE_COLORS = {
    "openburning": "#E69F00",
    "anthro": "#56B4E9",
    "AIR-anthro": "#009E73",
}

# sectors that need to be aggregated from the IAMC-style variable names
SECTORS_ANTHRO_IAMC = [
    "**International Shipping",
    "**Agriculture",
    "**Energy Sector",
    "**Industrial Sector",
    "**Residential Commercial Other",
    "**Solvents Production and Application",
    "**Transportation Sector",
    "**Waste",
    "**Other Capture and Removal",
]
SECTORS_AIR_IAMC = ["**Aircraft"]
SECTORS_OPENBURNING_IAMC = [
    "**Agricultural Waste Burning",
    "**Forest Burning",
    "**Grassland Burning",
    "**Peat Burning",
]

# sector renaming from IAMC-style to gridded-style
SECTOR_DICT_IAMC_TO_GRID = {
    "Energy Sector": "Energy",
    "Industrial Sector": "Industrial",
    "Residential Commercial Other": "Residential, Commercial, Other",
    "Transportation Sector": "Transportation",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def setup_logging(qc_output_path: Path, timestamp: str) -> logging.Logger:
    """Set up a logger that writes to both a file and stdout."""
    log_dir = qc_output_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"check_log_{timestamp}.txt"

    logger = logging.getLogger(f"qc_gridded_{timestamp}")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()  # avoid duplicate handlers when re-running interactively

    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def _find_here() -> Path:
    """Robustly find the notebooks/cmip7 directory."""
    try:
        here = Path(__file__).parent
        if here != Path("."):
            return here
    except NameError:
        pass
    # Fallback: search upward for concordia repo root
    current = Path.cwd()
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists() and (parent / "src" / "concordia").exists():
            return parent / "notebooks" / "cmip7"
    return Path.cwd()


def _parse_nc_filename(filename: str) -> tuple[str, str]:
    """
    Parse gas and file_type from a gridded NC filename.

    Naming convention: {gas}-em-{type}_{FILE_NAME_ENDING}.nc
    Special case: AIR-anthro files contain '-em-AIR-' in the name.

    Returns (gas, file_type) or raises ValueError.
    """
    stem = Path(filename).stem
    # Strip everything from the first underscore onward (FILE_NAME_ENDING)
    core = stem.split("_")[0] if "_" in stem else stem

    if "-em-AIR-" in core:
        # e.g. "CO2-em-AIR-anthro" → gas="CO2", type="AIR-anthro"
        gas = core.split("-em-AIR-")[0]
        return gas, "AIR-anthro"

    if "-em-" in core:
        parts = core.split("-em-", 1)
        gas = parts[0]
        file_type = parts[1]
        return gas, file_type

    raise ValueError(f"Cannot parse gas/type from filename: {filename}")


def resolve_settings_and_paths(
    gridded_scenario_folder: Path,
    settings_file: str,
    marker_to_run: str,
    gridding_version: str,
    here: Path,
) -> tuple[Settings, Path, Path, str, str, str, str]:
    """
    Load settings and resolve all key paths for the QC run.

    Returns
    -------
    settings, version_path, gridded_folder,
    model_selection, scenario_selection,
    scenario_selection_gridded, file_name_ending
    """
    settings = Settings.from_config(
        version=gridding_version,
        local_config_path=Path(here, settings_file),
    )

    version_path = gridded_scenario_folder
    gridded_folder = version_path # / "final"

    _, model_selection, scenario_selection, _ = return_marker_information(
        m=marker_to_run, v=settings_file
    )

    scenario_selection_gridded = scenario_selection.replace(" ", "-")
    version_esgf = gridding_version.split("_", 1)[-1] if "_" in gridding_version else gridding_version
    file_name_ending = cmip7_utils.filename_for_esgf(
        marker=marker_to_run, version=version_esgf
    )

    return (
        settings,
        version_path,
        gridded_folder,
        model_selection,
        scenario_selection,
        scenario_selection_gridded,
        file_name_ending,
    )


def load_cell_area(settings: Settings) -> xr.DataArray:
    """Load the grid cell area DataArray from the standard CEDS file."""
    gridding_path = Path(settings.gridding_path)
    candidates = sorted(gridding_path.glob("areacella*.nc"))
    if not candidates:
        raise FileNotFoundError(
            f"No areacella*.nc file found in {gridding_path}. "
            "Modules D/E (annual totals + animations) require this file. "
            "Check that settings.gridding_path points to the correct folder."
        )
    areacella_path = candidates[0]
    areacella = xr.open_dataset(areacella_path)
    return areacella["areacella"]


# ── Module A: File Inventory ───────────────────────────────────────────────────

def check_file_inventory(
    gridded_folder: Path,
    qc_output_path: Path,
    gases_expected: list[str] | None = None,
    file_types_expected: list[str] = FILE_TYPES,
    species_filter: list[str] | None = None,
    skip_existing: bool = True,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """
    List all NC files in the gridded folder; check against expected set.

    Outputs qc_output/tables/file_inventory.csv.
    Returns the inventory DataFrame.
    """
    log = logger or logging.getLogger(__name__)
    out_csv = qc_output_path / "tables" / "file_inventory.csv"

    if skip_existing and out_csv.exists():
        log.info(f"[A] Skipping file inventory (already exists): {out_csv}")
        return pd.read_csv(out_csv)

    log.info("[A] Running file inventory check...")
    t0 = time.time()

    if gases_expected is None:
        gases_expected = GASES_ESGF_CEDS
    if species_filter:
        gases_expected = [g for g in gases_expected if g in species_filter]

    # Build expected set
    expected = {
        (g, ft) for g in gases_expected for ft in file_types_expected
    }

    # Scan actual files
    rows = []
    found = set()
    if not gridded_folder.exists():
        log.warning(f"[A] Gridded folder not found: {gridded_folder}")
    else:
        for f in sorted(gridded_folder.glob("*.nc")):
            try:
                gas, file_type = _parse_nc_filename(f.name)
            except ValueError:
                log.warning(f"[A]   Unrecognised filename: {f.name}")
                gas, file_type = "UNKNOWN", "UNKNOWN"
            file_size_mb = f.stat().st_size / 1e6
            found.add((gas, file_type))
            rows.append(
                {
                    "filename": f.name,
                    "gas": gas,
                    "file_type": file_type,
                    "file_size_mb": round(file_size_mb, 2),
                    "exists": True,
                    "status_note": "OK",
                }
            )

    # Add missing expected files
    for gas, ft in sorted(expected - found):
        if species_filter and gas not in species_filter:
            continue
        rows.append(
            {
                "filename": f"{gas}-em-{ft}_MISSING.nc",
                "gas": gas,
                "file_type": ft,
                "file_size_mb": float("nan"),
                "exists": False,
                "status_note": "MISSING",
            }
        )

    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    n_found = df["exists"].sum()
    n_missing = (~df["exists"]).sum()
    log.info(
        f"[A] Done ({time.time()-t0:.1f}s): {n_found} files found, {n_missing} missing → {out_csv}"
    )
    if n_missing:
        missing_list = df[~df["exists"]][["gas", "file_type"]].values.tolist()
        log.warning(f"[A] Missing files: {missing_list}")

    return df


# ── Module B: Min/Max Value Check ─────────────────────────────────────────────

def check_min_max_values(
    gridded_folder: Path,
    qc_output_path: Path,
    file_inventory: pd.DataFrame | None = None,
    negative_threshold: float = -1e-10,
    species_filter: list[str] | None = None,
    skip_existing: bool = True,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """
    Compute per-file min/max/percentile statistics on gridded NC files.
    Flags unexpected negatives and near-all-zero files.

    Outputs qc_output/tables/min_max_stats.csv.
    """
    log = logger or logging.getLogger(__name__)
    out_csv = qc_output_path / "tables" / "min_max_stats.csv"

    if skip_existing and out_csv.exists():
        log.info(f"[B] Skipping min/max check (already exists): {out_csv}")
        try:
            return pd.read_csv(out_csv)
        except pd.errors.EmptyDataError:
            pass  # fall through to re-run

    log.info("[B] Running min/max value check...")
    t0 = time.time()

    # Determine which files to check
    if file_inventory is not None:
        nc_files = [
            gridded_folder / row["filename"]
            for _, row in file_inventory.iterrows()
            if row["exists"]
        ]
    else:
        nc_files = sorted(gridded_folder.glob("*.nc"))

    _STAT_COLS = ["filename", "gas", "file_type", "global_min", "global_max",
                  "p01", "p99", "frac_zeros", "frac_negatives",
                  "flag_unexpected_negatives", "flag_all_zeros", "status"]

    tmp_dir = out_csv.parent / "min_max_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Filter files to process
    pending = []
    for nc_path in nc_files:
        if not nc_path.exists():
            continue
        try:
            gas, file_type = _parse_nc_filename(nc_path.name)
        except ValueError:
            gas, file_type = "UNKNOWN", "UNKNOWN"
        if species_filter and gas not in species_filter:
            continue
        tmp_csv = tmp_dir / f"{nc_path.stem}.csv"
        if tmp_csv.exists():
            log.debug(f"[B]   Skipping (already done): {nc_path.name}")
            continue
        pending.append((nc_path, gas, file_type))

    def _process_one(args):
        nc_path, gas, file_type = args
        try:
            with xr.open_dataset(nc_path, engine="netcdf4") as ds:
                var_name = list(ds.data_vars.keys())[0]
                da = ds[var_name]
                nbytes = da.size * np.dtype(da.dtype).itemsize
    
                # Use dask path for anything over 500MB to be safe
                if nbytes < 500 * 1024 ** 2:
                    data = da.values.ravel()
                    global_min = float(data.min())
                    global_max = float(data.max())
                    p01 = float(np.percentile(data, 1))
                    p99 = float(np.percentile(data, 99))
                    frac_zeros = float((data == 0).mean())
                    frac_negatives = float((data < 0).mean())
                else:
                    # Always use dask path for large files
                    da_c = da.chunk({"time": 12})
                    global_min, global_max, frac_zeros, frac_negatives = (
                        float(v) for v in dask.compute(
                            da_c.min(), da_c.max(),
                            (da_c == 0).mean(), (da_c < 0).mean(),
                        )
                    )
                    # Strided sample for percentiles - sample every Nth time step
                    n_time = da.sizes.get("time", 1)
                    stride = max(1, n_time // 24)
                    sample = da.isel(time=slice(0, None, stride)).values.ravel()
                    p01 = float(np.percentile(sample, 1))
                    p99 = float(np.percentile(sample, 99))

            is_co2 = gas.upper() == "CO2"
            flag_unexpected_negatives = (global_min < negative_threshold) and not is_co2
            flag_all_zeros = frac_zeros > 0.9999

            if flag_unexpected_negatives:
                status = "WARNING: unexpected negatives"
            elif flag_all_zeros:
                status = "WARNING: near-all-zeros"
            else:
                status = "OK"

            return {
                "filename": nc_path.name,
                "gas": gas,
                "file_type": file_type,
                "global_min": global_min,
                "global_max": global_max,
                "p01": p01,
                "p99": p99,
                "frac_zeros": frac_zeros,
                "frac_negatives": frac_negatives,
                "flag_unexpected_negatives": flag_unexpected_negatives,
                "flag_all_zeros": flag_all_zeros,
                "status": status,
            }
        except Exception as e:
            return {
                "filename": nc_path.name,
                "gas": gas,
                "file_type": file_type,
                "global_min": float("nan"),
                "global_max": float("nan"),
                "p01": float("nan"),
                "p99": float("nan"),
                "frac_zeros": float("nan"),
                "frac_negatives": float("nan"),
                "flag_unexpected_negatives": True,
                "flag_all_zeros": False,
                "status": f"ERROR: {e}",
            }

    n_workers = min(4, len(pending)) if pending else 1
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
        for args in pending:
            nc_path = args[0]
            tmp_csv = tmp_dir / f"{nc_path.stem}.csv"
            row = _process_one(args)
            pd.DataFrame([row], columns=_STAT_COLS).to_csv(tmp_csv, index=False)
            log.info(f"[B]   {nc_path.name}: {row['status']}")
            
    # Collect all per-file CSVs and assemble final summary
    tmp_files = sorted(tmp_dir.glob("*.csv"))
    if tmp_files:
        df = pd.concat(
            [pd.read_csv(f) for f in tmp_files],
            ignore_index=True,
        )
    else:
        df = pd.DataFrame(columns=_STAT_COLS)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    log.info(f"[B] Written final summary: {out_csv}")

    # Clean up per-file temp CSVs
    for f in tmp_files:
        f.unlink()
    if tmp_dir.exists() and not any(tmp_dir.iterdir()):
        tmp_dir.rmdir()
    log.info(f"[B] Removed {len(tmp_files)} temporary per-file CSVs")

    n_warnings = (df["status"] != "OK").sum()
    log.info(
        f"[B] Done ({time.time()-t0:.1f}s): {len(df)} files checked, {n_warnings} warnings → {out_csv}"
    )
    for _, row in df[df["status"] != "OK"].iterrows():
        log.warning(f"[B]   {row['filename']}: {row['status']}")

    return df


# ── Module C: Downscaled Data QC ──────────────────────────────────────────────

def check_downscaled_qc(
    version_path: Path,
    qc_output_path: Path,
    gridding_version: str,
    skip_existing: bool = True,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """
    Replicate the workflow QC checks (workflow_cmip7-fast-track.py lines 1313-1499)
    on the saved downscaled CSV.  Collects results instead of raising errors.

    Outputs qc_output/tables/downscaled_qc_results.csv.
    """
    log = logger or logging.getLogger(__name__)
    out_csv = qc_output_path / "tables" / "downscaled_qc_results.csv"

    if skip_existing and out_csv.exists():
        log.info(f"[C] Skipping downscaled QC (already exists): {out_csv}")
        return pd.read_csv(out_csv)

    log.info("[C] Running downscaled data QC checks...")
    t0 = time.time()

    downscaled_path = version_path / f"downscaled-only-{gridding_version}.csv"
    if not downscaled_path.exists():
        log.warning(f"[C] Downscaled CSV not found: {downscaled_path} — skipping module C")
        return pd.DataFrame()

    # Load: read without specifying index_col, then detect index from non-numeric cols
    raw = pd.read_csv(downscaled_path)
    non_year_cols = [c for c in raw.columns if not str(c).isdigit()]
    raw = raw.set_index(non_year_cols)
    downscaled_data = raw.select_dtypes("number")
    downscaled_data.columns = pd.to_numeric(downscaled_data.columns, errors="coerce")
    downscaled_data = downscaled_data.dropna(axis=1, how="all")

    index_levels = list(downscaled_data.index.names)
    has_sector = "sector" in index_levels
    has_gas = "gas" in index_levels
    has_country = "country" in index_levels

    def _get_series(level):
        if level in index_levels:
            return downscaled_data.index.get_level_values(level).to_series(
                index=downscaled_data.index
            )
        return None

    sector_series = _get_series("sector")
    gas_series = _get_series("gas")

    results = []

    # ── Check 1: Disallowed negatives ──────────────────────────────────────
    row_mins = downscaled_data.min(axis=1)
    negative_rows = row_mins[row_mins < 0]

    if has_sector and has_gas and len(negative_rows):
        allowed_mask = sector_series.isin(CDR_SECTORS) | (
            (sector_series == "Industrial Sector") & (gas_series == "CO2")
        )
        disallowed = negative_rows[~allowed_mask.loc[negative_rows.index]]
        n = len(disallowed)
        if n:
            pairs = (
                downscaled_data.loc[disallowed.index]
                .index.to_frame(index=False)[["sector", "gas"]]
                .drop_duplicates()
                .values.tolist()
            )
            status = "FAIL"
        else:
            pairs = []
            status = "PASS"
    else:
        n = len(negative_rows)
        pairs = []
        status = "PASS" if n == 0 else "WARNING"

    results.append(
        {
            "check_name": "disallowed_negatives",
            "status": status,
            "n_rows_affected": n,
            "problem_gas_sector_pairs": json.dumps(pairs),
            "details": f"Rows with negative values in non-CDR sectors",
        }
    )
    log.info(f"[C] Check 1 (disallowed negatives): {status} — {n} rows")

    # ── Check 2: CDR sectors must be negative ──────────────────────────────
    if has_sector:
        row_maxs = downscaled_data.max(axis=1)
        positive_rows = row_maxs[row_maxs > 0]
        cdr_positives = positive_rows[
            sector_series.loc[positive_rows.index].isin(CDR_SECTORS_MUST_BE_NEGATIVE)
        ]
        n2 = len(cdr_positives)
        if n2:
            pairs2 = (
                downscaled_data.loc[cdr_positives.index]
                .index.to_frame(index=False)[["sector", "gas"]]
                .drop_duplicates()
                .values.tolist()
            )
            status2 = "FAIL"
        else:
            pairs2 = []
            status2 = "PASS"
    else:
        n2, pairs2, status2 = 0, [], "PASS"

    results.append(
        {
            "check_name": "cdr_must_be_negative",
            "status": status2,
            "n_rows_affected": n2,
            "problem_gas_sector_pairs": json.dumps(pairs2),
            "details": f"CDR sectors (DAC/EW/BECCS) should not have positive values",
        }
    )
    log.info(f"[C] Check 2 (CDR must be negative): {status2} — {n2} rows")

    # ── Check 3: Near-zero global totals ───────────────────────────────────
    near_zero_threshold = 1e-6
    groupby_levels = [l for l in ["gas", "sector", "unit"] if l in index_levels]
    if groupby_levels and has_country:
        global_totals = downscaled_data.groupby(level=groupby_levels).sum()
        numeric_cols = global_totals.select_dtypes("number")
        is_near_zero = (numeric_cols.abs() <= near_zero_threshold) & (numeric_cols.abs() > 0)
        rows_near_zero = is_near_zero.any(axis=1)
        n3 = int(rows_near_zero.sum())
        if n3:
            problem_rows = global_totals[rows_near_zero].index.to_list()
            pairs3 = [list(r) if hasattr(r, "__iter__") and not isinstance(r, str) else [r] for r in problem_rows[:20]]
            status3 = "WARNING"
        else:
            pairs3, status3 = [], "PASS"
    else:
        n3, pairs3, status3 = 0, [], "PASS"

    results.append(
        {
            "check_name": "near_zero_global_totals",
            "status": status3,
            "n_rows_affected": n3,
            "problem_gas_sector_pairs": json.dumps(pairs3),
            "details": f"Gas/sector combinations with 0 < |global total| ≤ {near_zero_threshold}",
        }
    )
    log.info(f"[C] Check 3 (near-zero totals): {status3} — {n3} rows")

    # ── Check 4: Country coverage ──────────────────────────────────────────
    if has_country:
        countries = set(downscaled_data.index.get_level_values("country").unique()) - {"World"}
        n_countries = len(countries)
        status4 = "INFO"
        details4 = f"{n_countries} unique countries found in downscaled data"
    else:
        n_countries = 0
        status4 = "INFO"
        details4 = "No 'country' level found in downscaled data index"

    results.append(
        {
            "check_name": "country_coverage",
            "status": status4,
            "n_rows_affected": n_countries,
            "problem_gas_sector_pairs": json.dumps([]),
            "details": details4,
        }
    )
    log.info(f"[C] Check 4 (country coverage): {details4}")

    df = pd.DataFrame(results)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    log.info(f"[C] Done ({time.time()-t0:.1f}s) → {out_csv}")
    return df


# ── Module D: Annual Totals 3-Way Comparison ─────────────────────────────────

def _aggregate_scenario_csv_to_totals(
    version_path: Path,
    scenario_selection: str,
) -> pd.DataFrame:
    """
    Load scenarios_processed.csv and aggregate to World-level totals
    per (gas, sector_file, year).
    Returns a melted DataFrame with columns:
    gas, sector_file, year, value, unit, source='scenario'
    """
    scenario_file = version_path / "scenarios_processed.csv"
    if not scenario_file.exists():
        return pd.DataFrame()

    df = cmip7_utils.load_data(scenario_file).dropna(axis=1)
    df = cmip7_utils.filter_scenario(df, scenarios=scenario_selection)
    if df.empty:
        return pd.DataFrame()

    # Detect id and year columns
    id_cols = [c for c in df.columns if not str(c).isdigit()]
    year_cols = [c for c in df.columns if str(c).isdigit()]

    # Create sector_file mapping
    sector_to_file = {}
    for sf_key, sectors_list in SECTOR_FILE_DICT.items():
        for s in sectors_list:
            sector_to_file[s] = sf_key

    if "sector" not in id_cols:
        return pd.DataFrame()

    df["sector_file"] = df["sector"].map(sector_to_file)
    df = df.dropna(subset=["sector_file"])

    # Remove CO2 from openburning and CO2 from Agriculture (zero by design)
    mask_co2_ob = (df.get("gas", pd.Series(dtype=str)) == "CO2") & (df["sector_file"] == "openburning")
    mask_co2_ag = (df.get("gas", pd.Series(dtype=str)) == "CO2") & (df["sector"] == "Agriculture")
    df = df[~mask_co2_ob & ~mask_co2_ag]

    gas_col = "gas" if "gas" in id_cols else None
    unit_col = "unit" if "unit" in id_cols else None
    if gas_col is None:
        return pd.DataFrame()

    group_cols = [c for c in [gas_col, "sector_file", unit_col] if c is not None]
    summed = df.groupby(group_cols)[year_cols].sum().reset_index()

    melted = summed.melt(
        id_vars=group_cols,
        var_name="year",
        value_name="value",
    )
    melted["year"] = pd.to_numeric(melted["year"])
    # kt → Mt conversion
    if unit_col and "unit" in melted.columns:
        kt_mask = melted["unit"].str.match(r"kt.*?/yr", na=False)
        melted.loc[kt_mask, "value"] /= 1000
        melted.loc[kt_mask, "unit"] = melted.loc[kt_mask, "unit"].str.replace(r"^kt", "Mt", regex=True)
    melted["source"] = "scenario"
    return melted


def _aggregate_harmonized_csv_to_totals(
    version_path: Path,
    gridding_version: str,
    scenario_selection: str,
) -> pd.DataFrame:
    """
    Load harmonization-{VERSION}.csv and aggregate to World-level totals.
    Returns same melted format as _aggregate_scenario_csv_to_totals.
    """
    harm_file = version_path / f"harmonization-{gridding_version}.csv"
    if not harm_file.exists():
        return pd.DataFrame()

    df = cmip7_utils.load_data(harm_file)
    df = cmip7_utils.filter_scenario(df, scenarios=scenario_selection).dropna(axis=1)
    if df.empty:
        return pd.DataFrame()

    # Filter to only Harmonized rows and strip outer variable levels
    if "variable" in df.columns:
        df = df[~df["variable"].str.contains("aggregate", na=False)]
        df = df[df["variable"].str.contains(r"\bHarmonized\b", regex=True, na=False)]
        df["variable"] = df["variable"].str.split("|").apply(
            lambda parts: "|".join(parts[1:-2])
        )

    # Extract gas and sector from variable string "Emissions|{gas}|{sector}"
    if "variable" in df.columns:
        var_split = df["variable"].str.split("|")
        df["gas"] = var_split.str[1]
        df["sector"] = var_split.str[2]
        df = df.drop(columns=["variable"], errors="ignore")
    else:
        return pd.DataFrame()

    # Apply the same sector_file mapping
    sector_to_file = {}
    for sf_key, sectors_list in SECTOR_FILE_DICT.items():
        for s in sectors_list:
            sector_to_file[s] = sf_key

    df["sector_file"] = df["sector"].map(sector_to_file)
    df = df.dropna(subset=["sector_file"])

    mask_co2_ob = (df["gas"] == "CO2") & (df["sector_file"] == "openburning")
    mask_co2_ag = (df["gas"] == "CO2") & (df["sector"] == "Agriculture")
    df = df[~mask_co2_ob & ~mask_co2_ag]

    year_cols = [c for c in df.columns if str(c).isdigit()]
    unit_col = "unit" if "unit" in df.columns else None
    group_cols = [c for c in ["gas", "sector_file", unit_col] if c is not None]
    summed = df.groupby(group_cols)[year_cols].sum().reset_index()

    melted = summed.melt(id_vars=group_cols, var_name="year", value_name="value")
    melted["year"] = pd.to_numeric(melted["year"])
    if unit_col and "unit" in melted.columns:
        kt_mask = melted["unit"].str.match(r"kt.*?/yr", na=False)
        melted.loc[kt_mask, "value"] /= 1000
        melted.loc[kt_mask, "unit"] = melted.loc[kt_mask, "unit"].str.replace(r"^kt", "Mt", regex=True)
    melted["source"] = "harmonized"
    return melted


# TODO: The plotting part of check_annual_totals (PNG output) does not work
# correctly. The CSV output is fine and can be used directly. The plots should
# be re-implemented or debugged separately.
def check_annual_totals(
    version_path: Path,
    gridded_folder: Path,
    qc_output_path: Path,
    gridding_version: str,
    scenario_selection: str,
    cell_area: xr.DataArray,
    species_filter: list[str] | None = None,
    skip_existing: bool = True,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """
    3-way comparison of annual totals: input IAM / harmonized / gridded.

    Outputs:
    - qc_output/tables/annual_totals_comparison.csv
    - qc_output/plots/annual_totals_{gas}_{type}.png (one per gas × type)
    """
    log = logger or logging.getLogger(__name__)
    out_csv = qc_output_path / "tables" / "annual_totals_comparison.csv"
    plots_dir = qc_output_path / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if skip_existing and out_csv.exists():
        log.info(f"[D] CSV already exists, loading and proceeding to plots: {out_csv}")
        comparison = pd.read_csv(out_csv)
        gases_in_comparison = sorted(comparison["gas"].unique()) if "gas" in comparison.columns else []
        types_in_comparison = sorted(comparison["sector_file"].unique()) if "sector_file" in comparison.columns else []
        for gas in gases_in_comparison:
            for file_type in types_in_comparison:
                sub = comparison[
                    (comparison["gas"] == gas) & (comparison["sector_file"] == file_type)
                ].sort_values("year")
                if sub.empty:
                    continue
                out_plot = plots_dir / f"annual_totals_{gas}_{file_type}.png"
                if skip_existing and out_plot.exists():
                    continue
                fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
                ax_abs, ax_rel = axes
                for col, label, ls in [
                    ("value_scenario", "Input IAM", "--"),
                    ("value_harmonized", "Harmonized", "-."),
                    ("value_gridded", "Gridded", "-"),
                ]:
                    if col in sub.columns:
                        ax_abs.plot(sub["year"], sub[col], label=label, linestyle=ls, linewidth=1.5, alpha=0.75)
                ax_abs.set_ylabel("Mt/yr")
                ax_abs.set_title(f"{gas} | {file_type}")
                ax_abs.legend(fontsize=8)
                ax_abs.grid(True, alpha=0.3)
                if "rel_diff_grid_harm_pct" in sub.columns:
                    ax_rel.plot(sub["year"], sub["rel_diff_grid_harm_pct"],
                                label="Gridded vs. Harmonized (%)", color="purple", linewidth=1.5, alpha=0.75)
                if "rel_diff_grid_scen_pct" in sub.columns:
                    ax_rel.plot(sub["year"], sub["rel_diff_grid_scen_pct"],
                                label="Gridded vs. Scenario (%)", color="orange", linestyle="--", linewidth=1.5, alpha=0.75)
                ax_rel.axhline(0, color="black", linewidth=0.8, linestyle=":")
                ax_rel.set_ylabel("Relative diff (%)")
                ax_rel.set_xlabel("Year")
                ax_rel.legend(fontsize=8)
                ax_rel.grid(True, alpha=0.3)
                plt.tight_layout()
                fig.savefig(out_plot, dpi=150, bbox_inches="tight")
                plt.close(fig)
                log.info(f"[D]   Plot saved → {out_plot.name}")
        return comparison

    log.info("[D] Running annual totals 3-way comparison...")
    t0 = time.time()

    # ── 1. Load scenario and harmonized data ──────────────────────────────
    df_scen = _aggregate_scenario_csv_to_totals(version_path, scenario_selection)
    df_harm = _aggregate_harmonized_csv_to_totals(version_path, gridding_version, scenario_selection)

    # ── 2. Aggregate gridded NC files ─────────────────────────────────────
    if not gridded_folder.exists():
        log.warning(f"[D] Gridded folder not found: {gridded_folder}")
        return pd.DataFrame()

    nc_files = sorted(gridded_folder.glob("*.nc"))
    if species_filter:
        nc_files = [
            f for f in nc_files
            if any(f.name.startswith(g + "-") for g in species_filter)
        ]

    tmp_dir = out_csv.parent / "annual_totals_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"[D]   Aggregating {len(nc_files)} gridded files (sequential, crash-resumable)...")
    for nc_path in nc_files:
        if not nc_path.exists():
            continue
        try:
            gas, file_type = _parse_nc_filename(nc_path.name)
        except ValueError:
            gas, file_type = "UNKNOWN", "UNKNOWN"

        if species_filter and gas not in species_filter:
            continue

        tmp_csv = tmp_dir / f"{nc_path.stem}.csv"
        if tmp_csv.exists() and skip_existing:
            log.debug(f"[D]   Skipping (already done): {nc_path.name}")
            continue

        try:
            ds = xr.open_dataset(nc_path, engine="netcdf4")
            var_name = list(ds.data_vars.keys())[0]
            result_da = ds_to_annual_emissions_total_faster(
                ds, var_name, cell_area, keep_sectors=False
            )
            result_series = result_da.to_series()
            # Value is always in Mt/yr after ds_to_annual_emissions_total_faster;
            # the raw NetCDF attribute (e.g. "kg m-2 s-1") would be misleading here.
            reporting_unit = "Mt/yr"
            ds.close()

            rows = [
                {"gas": gas, "sector_file": file_type, "year": int(year),
                 "value": value, "unit": reporting_unit, "source": "gridded"}
                for year, value in result_series.items()
            ]
            pd.DataFrame(rows).to_csv(tmp_csv, index=False)
            log.info(f"[D]   Wrote per-file totals: {tmp_csv.name}")

        except Exception as e:
            log.error(f"[D]   Error processing {nc_path.name}: {e}")

    # Collect all per-file CSVs
    tmp_files = sorted(tmp_dir.glob("*.csv"))
    if tmp_files:
        df_grid = pd.concat([pd.read_csv(f) for f in tmp_files], ignore_index=True)
    else:
        df_grid = pd.DataFrame()

    # ── 3. Combine and compute differences ────────────────────────────────
    def _pivot_source(df, source_name):
        if df.empty:
            return pd.DataFrame()
        return (
            df[df["source"] == source_name]
            [["gas", "sector_file", "year", "value"]]
            .rename(columns={"value": f"value_{source_name}"})
        )

    non_empty_sources = [df for df in [df_scen, df_harm, df_grid] if not df.empty]
    if not non_empty_sources:
        log.error(
            "[D] No data found in any of the three sources (scenario / harmonized / gridded). "
            "This usually means the gridded_scenario_folder does not match the marker_to_run. "
            f"  gridded_folder  = {gridded_folder}\n"
            f"  scenario filter = '{scenario_selection}' (derived from marker '{scenario_selection}')\n"
            "Check that the folder path points to the correct scenario run and that "
            "marker_to_run matches the data inside it."
        )
        return pd.DataFrame()
    all_data = pd.concat(non_empty_sources, ignore_index=True)
    if all_data.empty:
        log.warning("[D] No data to compare — all sources empty")
        return pd.DataFrame()

    pivot_scen = _pivot_source(all_data, "scenario")
    pivot_harm = _pivot_source(all_data, "harmonized")
    pivot_grid = _pivot_source(all_data, "gridded")

    merge_cols = ["gas", "sector_file", "year"]
    comparison = pivot_scen.merge(pivot_harm, on=merge_cols, how="outer") \
                           .merge(pivot_grid, on=merge_cols, how="outer")

    for col_a, col_b, suffix in [
        ("value_gridded", "value_scenario", "grid_scen"),
        ("value_gridded", "value_harmonized", "grid_harm"),
        ("value_scenario", "value_harmonized", "scen_harm"),
    ]:
        if col_a in comparison.columns and col_b in comparison.columns:
            comparison[f"abs_diff_{suffix}"] = (comparison[col_a] - comparison[col_b]).abs()
            with np.errstate(divide="ignore", invalid="ignore"):
                comparison[f"rel_diff_{suffix}_pct"] = np.where(
                    comparison[col_b].abs() > 1e-12,
                    (comparison[col_a] - comparison[col_b]) / comparison[col_b].abs() * 100,
                    float("nan"),
                )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(out_csv, index=False)
    log.info(f"[D]   Totals CSV saved → {out_csv}")

    # Clean up per-file temp CSVs
    for f in tmp_files:
        f.unlink()
    if tmp_dir.exists() and not any(tmp_dir.iterdir()):
        tmp_dir.rmdir()
    log.info(f"[D]   Removed {len(tmp_files)} temporary per-file CSVs")

    # ── 4. Per-gas per-type comparison plots ─────────────────────────────
    gases_in_comparison = sorted(comparison["gas"].unique()) if "gas" in comparison.columns else []
    types_in_comparison = sorted(comparison["sector_file"].unique()) if "sector_file" in comparison.columns else []

    log.info(
        f"[D]   Plotting: {len(gases_in_comparison)} gases × {len(types_in_comparison)} types "
        f"= {len(gases_in_comparison) * len(types_in_comparison)} combinations "
        f"| comparison shape: {comparison.shape} | columns: {list(comparison.columns)}"
    )

    for gas in gases_in_comparison:
        for file_type in types_in_comparison:
            sub = comparison[
                (comparison["gas"] == gas) & (comparison["sector_file"] == file_type)
            ].sort_values("year")
            if sub.empty:
                continue

            out_plot = plots_dir / f"annual_totals_{gas}_{file_type}.png"
            if skip_existing and out_plot.exists():
                continue

            fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
            ax_abs, ax_rel = axes

            # Top: absolute values
            for col, label, ls in [
                ("value_scenario", "Input IAM", "--"),
                ("value_harmonized", "Harmonized", "-."),
                ("value_gridded", "Gridded", "-"),
            ]:
                if col in sub.columns:
                    ax_abs.plot(sub["year"], sub[col], label=label, linestyle=ls, linewidth=1.5, alpha=0.75)
            ax_abs.set_ylabel("Mt/yr")
            ax_abs.set_title(f"{gas} | {file_type}")
            ax_abs.legend(fontsize=8)
            ax_abs.grid(True, alpha=0.3)

            # Bottom: relative differences
            if "rel_diff_grid_harm_pct" in sub.columns:
                ax_rel.plot(sub["year"], sub["rel_diff_grid_harm_pct"],
                            label="Gridded vs. Harmonized (%)", color="purple", linewidth=1.5, alpha=0.75)
            if "rel_diff_grid_scen_pct" in sub.columns:
                ax_rel.plot(sub["year"], sub["rel_diff_grid_scen_pct"],
                            label="Gridded vs. Scenario (%)", color="orange", linestyle="--", linewidth=1.5, alpha=0.75)
            ax_rel.axhline(0, color="black", linewidth=0.8, linestyle=":")
            ax_rel.set_ylabel("Relative diff (%)")
            ax_rel.set_xlabel("Year")
            ax_rel.legend(fontsize=8)
            ax_rel.grid(True, alpha=0.3)

            plt.tight_layout()
            fig.savefig(out_plot, dpi=150, bbox_inches="tight")
            plt.close(fig)

    log.info(f"[D] Done ({time.time()-t0:.1f}s) → {out_csv}")
    return comparison


# ── Module E: Fast Animated Grids ─────────────────────────────────────────────

def make_animated_grids(
    gridded_folder: Path,
    qc_output_path: Path,
    cell_area: xr.DataArray,
    file_inventory: pd.DataFrame | None = None,
    species_filter: list[str] | None = None,
    frame_years: list[int] | None = None,
    dpi: int = 100,
    skip_existing: bool = True,
    animation_mode: str = "all-sectors",
    logger: logging.Logger | None = None,
) -> list[Path]:
    """
    Create animated GIF maps — much faster than FuncAnimation.

    animation_mode:
        "all-sectors"        one GIF per (gas, file_type, sector)
                             e.g. BC_anthro-Energy_animation.gif
        "total-per-file"     sectors summed within each NC file; one GIF per (gas, file_type)
                             e.g. BC_anthro-total_animation.gif
        "total-per-species"  all files summed for each gas; one GIF per gas
                             e.g. BC-total_animation.gif

    Uses ~19 sampled frames (2022, 2023, 2024, then every 5 years to 2100)
    instead of all 936 monthly frames.  Renders via FigureCanvasAgg + PIL.
    All outputs written to qc_output/plots/animations/.
    """
    log = logger or logging.getLogger(__name__)
    anim_dir = qc_output_path / "plots" / "animations"
    anim_dir.mkdir(parents=True, exist_ok=True)

    if frame_years is None:
        frame_years = [2022, 2023, 2024] + list(range(2025, 2105, 5))

    if file_inventory is not None:
        nc_files = [
            gridded_folder / row["filename"]
            for _, row in file_inventory.iterrows()
            if row["exists"]
        ]
    else:
        nc_files = sorted(gridded_folder.glob("*.nc"))

    if species_filter:
        nc_files = [
            f for f in nc_files
            if any(f.name.startswith(g + "-") for g in species_filter)
        ]

    out_paths: list[Path] = []
    log.info(f"[E] Creating animated grids ({animation_mode}) for {len(nc_files)} files...")
    t0 = time.time()

    # ── inner helpers ──────────────────────────────────────────────────────────

    def _safe_name(s: str) -> str:
        return s.replace("/", "_").replace(" ", "_")

    def _decode_sectors(da: xr.DataArray) -> list[tuple]:
        """Return [(sector_val, sector_name_str), ...] using SECTOR_DICT_ANTHRO_CO2_SCENARIO."""
        if "sector" not in da.dims:
            return [(None, "total")]
        raw_sectors = da.sector.values
        if np.issubdtype(da.sector.dtype, np.integer):
            return [
                (sid, cmip7_utils.SECTOR_DICT_ANTHRO_CO2_SCENARIO.get(int(sid), str(sid)))
                for sid in raw_sectors
            ]
        return [(s, str(s)) for s in raw_sectors]

    def _find_time_indices(da: xr.DataArray) -> tuple[list[int], np.ndarray]:
        times_years = da.time.dt.year.values
        selected: list[int] = []
        for yr in frame_years:
            diffs = np.abs(times_years - yr)
            idx = int(np.argmin(diffs))
            if diffs[idx] < 15 and idx not in selected:  # within 15 years of target year
                selected.append(idx)
        return selected, times_years

    def _render_gif(
        da_sector: xr.DataArray,
        times_years: np.ndarray,
        selected_indices: list[int],
        title: str,
        out_gif: Path,
    ) -> None:
        """Render selected frames of da_sector to an animated GIF."""
        if skip_existing and out_gif.exists():
            return

        # No dask chunking here: we only read ~19 selected frames, so the
        # netCDF4 backend handles reads single-threadedly.  Using chunks=
        # spawns a dask thread pool that decompresses many HDF5 chunks in
        # parallel, exhausting RAM for large files.
        da_frames = da_sector.isel(time=selected_indices).load()
        vals = da_frames.values.ravel()
        finite_vals = vals[np.isfinite(vals) & (vals != 0)]
        if len(finite_vals) == 0:
            log.debug(f"[E]   Skipping {title}: all zero/NaN")
            return
        vmin = float(np.percentile(finite_vals, 2))
        vmax = float(np.percentile(finite_vals, 99))
        if vmin == vmax:
            vmax = vmin + 1e-30
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        frames_pil = []
        for fi, t_idx in enumerate(selected_indices):
            yr_label = int(times_years[t_idx])
            da_t = da_frames.isel(time=fi)
            fig, ax = plt.subplots(
                1, 1, figsize=(6, 4),
                dpi=dpi,
                subplot_kw={"projection": ccrs.Robinson()},
            )
            try:
                da_t.plot.pcolormesh(
                    ax=ax,
                    transform=ccrs.PlateCarree(),
                    norm=norm,
                    cmap="GnBu",
                    add_colorbar=False,
                )
            except Exception:
                plt.close(fig)
                continue
            ax.coastlines(linewidth=0.4)
            ax.set_title(f"{title} | {yr_label}", fontsize=8)
            plt.tight_layout(pad=0.3)
            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            w, h = canvas.get_width_height()
            img = Image.frombuffer("RGBA", (w, h), canvas.buffer_rgba(), "raw", "RGBA", 0, 1)
            img = img.convert("RGB").quantize(colors=256, method=Image.Quantize.MEDIANCUT)
            frames_pil.append(img)
            plt.close(fig)

        if not frames_pil:
            return
        # Add 5 pause frames at the end
        pause_frames = [frames_pil[-1]] * 5
        frames_pil[0].save(
            out_gif,
            save_all=True,
            append_images=frames_pil[1:] + pause_frames,
            loop=0,
            duration=300,
            optimize=False,
        )
        out_paths.append(out_gif)
        log.debug(f"[E]   Saved {out_gif.name}")

    # ── Mode: total-per-species ────────────────────────────────────────────────
    if animation_mode == "total-per-species":
        from collections import defaultdict
        gas_files: dict[str, list[Path]] = defaultdict(list)
        for nc_path in nc_files:
            if not nc_path.exists():
                continue
            try:
                gas, _ = _parse_nc_filename(nc_path.name)
                gas_files[gas].append(nc_path)
            except ValueError:
                continue

        for gas, paths in gas_files.items():
            try:
                da_sum: xr.DataArray | None = None
                times_years: np.ndarray | None = None
                selected_indices: list[int] = []
                for nc_path in paths:
                    ds = xr.open_dataset(nc_path, engine="netcdf4")
                    var_name = list(ds.data_vars.keys())[0]
                    da = ds[var_name]
                    if "level" in da.dims:
                        da = da.sum(dim="level")
                    if "sector" in da.dims:
                        da = da.sum(dim="sector")
                    if da_sum is None:
                        da_sum = da
                        selected_indices, times_years = _find_time_indices(da_sum)
                    else:
                        da_sum = da_sum + da
                    ds.close()
                if da_sum is None or not selected_indices:
                    if da_sum is None:
                        log.warning(f"[E]   No valid files found for gas '{gas}' — skipping")
                    continue
                _render_gif(
                    da_sum.squeeze(), times_years, selected_indices,
                    title=f"{gas} | total",
                    out_gif=anim_dir / f"{gas}-total_animation.gif",
                )
            except Exception as e:
                log.error(f"[E]   Error animating {gas} (total-per-species): {e}")

    # ── Modes: all-sectors / total-per-file ────────────────────────────────────
    else:
        for nc_path in nc_files:
            if not nc_path.exists():
                continue
            try:
                gas, file_type = _parse_nc_filename(nc_path.name)
            except ValueError:
                continue

            try:
                ds = xr.open_dataset(nc_path, engine="netcdf4")
                var_name = list(ds.data_vars.keys())[0]
                da = ds[var_name]

                # Handle level dimension (AIR files)
                if "level" in da.dims:
                    da = da.sum(dim="level").load()

                selected_indices, times_years = _find_time_indices(da)
                if not selected_indices:
                    ds.close()
                    continue

                if animation_mode == "total-per-file":
                    if "sector" in da.dims:
                        # Pre-select only the frames we need before summing sectors,
                        # so each sector loads (n_frames, lat, lon) not (time, lat, lon).
                        da_sector = None
                        for i in range(da.sizes["sector"]):
                            layer = da.isel(sector=i, time=selected_indices).load()
                            da_sector = layer if da_sector is None else da_sector + layer
                    else:
                        da_sector = da.isel(time=selected_indices).load()
                    da_sector = da_sector.squeeze()
                    # Time axis is already pre-selected; pass sequential indices
                    # and the corresponding subset of year labels so that
                    # _render_gif's times_years[t_idx] gives the correct year.
                    presel_indices = list(range(da_sector.sizes["time"]))
                    presel_times_years = times_years[np.array(selected_indices)]
                    _render_gif(
                        da_sector, presel_times_years, presel_indices,
                        title=f"{gas} | {file_type} | total",
                        out_gif=anim_dir / f"{gas}_{file_type}-total_animation.gif",
                    )
                else:  # "all-sectors"
                    for sector_val, sector_name in _decode_sectors(da):
                        da_sector = (
                            da.sel(sector=sector_val).squeeze()
                            if sector_val is not None
                            else da.squeeze()
                        )
                        _render_gif(
                            da_sector, times_years, selected_indices,
                            title=f"{gas} | {file_type} | {sector_name}",
                            out_gif=anim_dir / f"{gas}_{file_type}-{_safe_name(sector_name)}_animation.gif",
                        )

                ds.close()

            except Exception as e:
                log.error(f"[E]   Error animating {nc_path.name}: {e}")

    log.info(f"[E] Done ({time.time()-t0:.1f}s): {len(out_paths)} GIFs written → {anim_dir}")
    return out_paths


# ── Module F: Documentation Plots 03 and 04 ───────────────────────────────────

def _reformatting_names_units(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize gas names and N2O units. Mirrors check_plots_for_documentation.py."""
    df = df.copy()
    df["gas"] = df["gas"].replace("Sulfur", "SO2")
    if "unit" in df.columns and "value" in df.columns:
        n2o_mask = df["gas"] == "N2O"
        if "unit" in df.columns:
            kt_n2o_mask = n2o_mask & df["unit"].str.contains("kt N2O", na=False)
            if kt_n2o_mask.any():
                df.loc[kt_n2o_mask, "value"] /= 1000
                df.loc[kt_n2o_mask, "unit"] = "Mt N2O/yr"
    return df


def _compute_split_stacked_bars(data_to_plot: pd.DataFrame) -> list[tuple]:
    """
    Compute split-stacked bar data: positive values stack up, negative stack down.
    Mirrors check_plots_for_documentation.py lines 661-693.
    """
    col_order = ["anthro", "AIR-anthro", "openburning"]
    cols = [c for c in col_order if c in data_to_plot.columns]

    num_years = len(data_to_plot.index)
    results = []
    bottom_pos = np.zeros(num_years)
    bottom_neg = np.zeros(num_years)

    for col in cols:
        values = data_to_plot[col].values
        pos_values = np.where(values >= 0, values, 0)
        neg_values = np.where(values < 0, values, 0)
        results.append((col, pos_values, bottom_pos.copy()))
        bottom_pos += pos_values
        results.append((col, neg_values, bottom_neg.copy()))
        bottom_neg += neg_values

    return results


def make_doc_plots(
    version_path: Path,
    qc_output_path: Path,
    gridding_version: str,
    marker_to_run: str,
    settings: Settings,
    history_file: str,
    species_filter: list[str] | None = None,
    skip_existing: bool = True,
    logger: logging.Logger | None = None,
) -> list[Path]:
    """
    Reproduce plots 03 and 04 from check_plots_for_documentation.py,
    adapted for a single scenario.

    Plot 03: total emissions timeseries + historical data (2×5 gas grid)
    Plot 04: stacked bars by sector_file + history overlay (2×5 gas grid)

    Outputs:
    - qc_output/plots/03_total_emissions_with_history.png
    - qc_output/plots/04_stacked_bars_{marker}.png
    """
    log = logger or logging.getLogger(__name__)
    plots_dir = qc_output_path / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    out_03 = plots_dir / "03_total_emissions_with_history.png"
    out_04 = plots_dir / f"04_stacked_bars_{marker_to_run}.png"

    if skip_existing and out_03.exists() and out_04.exists():
        log.info(f"[F] Skipping doc plots (already exist): {out_03.name}, {out_04.name}")
        return [out_03, out_04]

    log.info("[F] Creating documentation plots 03 and 04...")
    t0 = time.time()

    # ── Load scenario data ────────────────────────────────────────────────
    scenario_file = version_path / "scenarios_processed.csv"
    if not scenario_file.exists():
        log.warning(f"[F] scenarios_processed.csv not found: {scenario_file}")
        return []

    df_raw = cmip7_utils.load_data(scenario_file).dropna(axis=1)
    df_raw["scenario"] = marker_to_run.upper()

    sector_to_file = {}
    for sf_key, sectors_list in SECTOR_FILE_DICT.items():
        for s in sectors_list:
            sector_to_file[s] = sf_key
    df_raw["sector_file"] = df_raw["sector"].map(sector_to_file)
    df_raw = df_raw.dropna(subset=["sector_file"])

    # Remove CO2 from openburning and CO2 from Agriculture
    if "gas" in df_raw.columns:
        df_raw = df_raw[
            ~((df_raw["gas"] == "CO2") & (df_raw["sector_file"] == "openburning"))
        ]
        df_raw = df_raw[
            ~((df_raw["gas"] == "CO2") & (df_raw["sector"] == "Agriculture"))
        ]

    year_cols = [c for c in df_raw.columns if str(c).isdigit()]
    id_cols_base = ["model", "scenario", "gas", "unit"]
    id_cols_sf = id_cols_base + ["sector_file"]

    # Aggregate total (all sectors summed)
    df_total = df_raw.groupby([c for c in id_cols_base if c in df_raw.columns])[year_cols].sum().reset_index()
    df_total_melted = df_total.melt(
        id_vars=[c for c in id_cols_base if c in df_total.columns],
        var_name="years", value_name="value",
    )
    df_total_melted["years"] = pd.to_numeric(df_total_melted["years"])
    df_total_melted = _reformatting_names_units(df_total_melted)

    # Aggregate by sector_file
    df_sf = df_raw.groupby([c for c in id_cols_sf if c in df_raw.columns])[year_cols].sum().reset_index()
    df_sf_melted = df_sf.melt(
        id_vars=[c for c in id_cols_sf if c in df_sf.columns],
        var_name="years", value_name="value",
    )
    df_sf_melted["years"] = pd.to_numeric(df_sf_melted["years"])
    df_sf_melted = _reformatting_names_units(df_sf_melted)

    # ── Load historical data ──────────────────────────────────────────────
    hist_path = settings.history_path / history_file
    hist_available = hist_path.exists()
    hist_melted_sum = pd.DataFrame()
    hist_sector_files = pd.DataFrame()

    if hist_available:
        hist = (
            pd.read_csv(hist_path)
            .drop(columns=["model", "scenario"], errors="ignore")
            .rename(columns={"region": "country"}, errors="ignore")
        )
        hist = extractlevel(
            hist.set_index(["country", "variable", "unit"]),
            variable="Emissions|{gas}|{sector}",
            drop=True,
        )
        hist = hist.reorder_levels(["country", "gas", "sector", "unit"]).sort_index()
        hist.columns = hist.columns.astype(int)
        hist.columns.name = "year"

        hist_nonglobal = hist.loc[~isin(country="global")]
        hist_nonglobal = hist_nonglobal.loc[
            ~ismatch(sector=["**Shipping", "**Aircraft"])
        ]
        hist_global = hist.loc[isin(country="global")]
        hist_global_nonzero = hist_global[ismatch(sector=["**Shipping", "**Aircraft"])]
        hist_global_nonzero = hist_global_nonzero.rename(
            index=lambda v: v.replace("global", "World")
        )
        hist_nonglobal_world = assignlevel(
            hist_nonglobal.groupby(["gas", "sector", "unit"]).sum(),
            country="World",
        ).reorder_levels(["country", "gas", "sector", "unit"])
        hist = pd.concat([hist_nonglobal, hist_global_nonzero, hist_nonglobal_world]).reset_index()

        # Filter same as scenario data
        hist = hist[~((hist["gas"] == "CO2") & (hist["sector"].isin(SECTOR_FILE_DICT["openburning"])))]
        hist = hist[~((hist["gas"] == "CO2") & (hist["sector"] == "Agriculture"))]
        hist = hist[hist["country"] == "World"]

        hist["sector_file"] = hist["sector"].map(sector_to_file)

        hist_num_cols = [c for c in hist.columns if str(c).isdigit()]

        hist_melted = hist.melt(
            id_vars=["country", "gas", "sector", "sector_file", "unit"],
            var_name="years", value_name="value",
        )
        hist_melted["years"] = pd.to_numeric(hist_melted["years"])
        hist_melted = _reformatting_names_units(hist_melted)
        hist_melted_sum = hist_melted.groupby(["gas", "unit", "years"])["value"].sum().reset_index()

        hist_sf_grouped = hist.groupby(["country", "gas", "sector_file", "unit"])[hist_num_cols].sum().reset_index()
        hist_sector_files = hist_sf_grouped.melt(
            id_vars=["country", "gas", "sector_file", "unit"],
            var_name="years", value_name="value",
        )
        hist_sector_files["years"] = pd.to_numeric(hist_sector_files["years"])
        hist_sector_files = _reformatting_names_units(hist_sector_files)

    # ── Determine gases to plot ───────────────────────────────────────────
    unique_gases = sorted(df_total_melted["gas"].unique()) if "gas" in df_total_melted.columns else []
    if species_filter:
        unique_gases = [g for g in unique_gases if g in species_filter]
    gases_to_plot = unique_gases[:10]  # 2×5 grid

    years_to_mark = [2023, 2024, 2025] + list(range(2030, 2105, 5))

    # ── Plot 03: Total emissions timeseries + history ─────────────────────
    if not (skip_existing and out_03.exists()):
        fig, axes = plt.subplots(2, 5, figsize=(20, 10))
        axes = axes.flatten()

        for idx, gas in enumerate(gases_to_plot):
            if idx >= len(axes):
                break
            ax = axes[idx]
            gas_data = df_total_melted[df_total_melted["gas"] == gas]
            gas_unit = gas_data["unit"].iloc[0] if len(gas_data) > 0 else ""

            # Scenario line
            scen_data = gas_data.sort_values("years")
            ax.plot(scen_data["years"], scen_data["value"],
                    label=marker_to_run.upper(), linewidth=2, color="#2E5EAA")
            marked = scen_data[scen_data["years"].isin(years_to_mark)]
            ax.plot(marked["years"], marked["value"], marker="o",
                    linestyle="none", markersize=5, color="#2E5EAA")

            # Historical line
            if not hist_melted_sum.empty:
                gas_hist = hist_melted_sum[
                    (hist_melted_sum["gas"] == gas) & (hist_melted_sum["years"] >= 2010)
                ].sort_values("years")
                if len(gas_hist):
                    ax.plot(gas_hist["years"], gas_hist["value"],
                            label="Historical", linewidth=2.5, color="black", linestyle="--")
                    ax.plot(gas_hist["years"], gas_hist["value"],
                            marker="s", linestyle="none", markersize=5, color="black")

            ax.set_title(gas, fontsize=11, fontweight="bold")
            ax.set_xlabel("Year", fontsize=8)
            ax.set_ylabel(f"Emissions ({gas_unit})", fontsize=8)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

        # Hide unused axes
        for idx in range(len(gases_to_plot), len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        fig.savefig(out_03, dpi=300, bbox_inches="tight")
        plt.close(fig)
        log.info(f"[F]   Plot 03 saved → {out_03}")

    # ── Plot 04: Stacked bars + history ───────────────────────────────────
    if not (skip_existing and out_04.exists()):
        hist_years = sorted(hist_sector_files[hist_sector_files["years"] >= 2010]["years"].unique().tolist()) if not hist_sector_files.empty else []

        fig, axes = plt.subplots(2, 5, figsize=(20, 10))
        axes = axes.flatten()

        for idx, gas in enumerate(gases_to_plot):
            if idx >= len(axes):
                break
            ax = axes[idx]
            gas_data = df_sf_melted[df_sf_melted["gas"] == gas]
            gas_unit = gas_data["unit"].iloc[0] if len(gas_data) > 0 else ""

            # Pivot scenario data by sector_file
            pivot_scen = gas_data.pivot_table(
                index="years", columns="sector_file", values="value", aggfunc="sum"
            ).fillna(0)
            all_sf = sorted(SECTOR_FILE_COLORS.keys())
            pivot_scen = pivot_scen.reindex(columns=all_sf, fill_value=0)

            # Historical pivot
            if not hist_sector_files.empty:
                gas_hist_sf = hist_sector_files[(hist_sector_files["gas"] == gas) & (hist_sector_files["years"] >= 2010)]
                hist_pivot = gas_hist_sf.pivot_table(
                    index="years", columns="sector_file", values="value", aggfunc="sum"
                ).fillna(0).reindex(columns=all_sf, fill_value=0)
            else:
                hist_pivot = None

            all_years = sorted(
                set(list(hist_years) + years_to_mark)
            )
            pivot_scen = pivot_scen.reindex(all_years, fill_value=0)
            if hist_pivot is not None:
                hist_pivot = hist_pivot.reindex(all_years, fill_value=0)

            years_before_base = [y for y in all_years if y <= 2023]
            grey_end = years_before_base[-1] if years_before_base else all_years[0]
            ax.axvspan(all_years[0] - 2, grey_end + 0.5, alpha=0.1, color="grey", zorder=0)

            def _bar_width(yr):
                return 2.4 if yr >= 2030 else 0.6

            widths = [_bar_width(y) for y in all_years]

            # Historical bars (alpha=0.6)
            if hist_pivot is not None:
                hist_bars = _compute_split_stacked_bars(hist_pivot)
                plotted_hist = set()
                for col, values, bottom in hist_bars:
                    if np.any(values != 0):
                        lbl = col if col not in plotted_hist else ""
                        plotted_hist.add(col)
                        ax.bar(all_years, values, widths, bottom=bottom,
                               label=lbl, color=SECTOR_FILE_COLORS.get(col, "#CCC"), alpha=0.6)

            # Scenario bars (full opacity)
            scen_bars = _compute_split_stacked_bars(pivot_scen)
            plotted_scen = set()
            for col, values, bottom in scen_bars:
                if np.any(values != 0):
                    lbl = "" if hist_pivot is not None or col in plotted_scen else col
                    plotted_scen.add(col)
                    ax.bar(all_years, values, widths, bottom=bottom,
                           label=lbl, color=SECTOR_FILE_COLORS.get(col, "#CCC"))

            # Total overlay (scenario)
            gas_total_data = df_total_melted[df_total_melted["gas"] == gas].sort_values("years")
            if len(gas_total_data):
                ax.plot(gas_total_data["years"], gas_total_data["value"],
                        color="black", linewidth=2.5, linestyle="-", zorder=5)
                marked_total = gas_total_data[gas_total_data["years"].isin(years_to_mark)]
                ax.plot(marked_total["years"], marked_total["value"],
                        color="black", marker="o", markersize=5, linestyle="none",
                        label="Total", zorder=5)

            # Historical total overlay
            if not hist_melted_sum.empty:
                gas_hist_total = hist_melted_sum[(hist_melted_sum["gas"] == gas) & (hist_melted_sum["years"] >= 2010)].sort_values("years")
                if len(gas_hist_total):
                    ax.plot(gas_hist_total["years"], gas_hist_total["value"],
                            color="black", linewidth=2.5, linestyle="--", zorder=5)
                    marked_hist = gas_hist_total[gas_hist_total["years"].isin(hist_years)]
                    ax.plot(marked_hist["years"], marked_hist["value"],
                            color="black", marker="s", markersize=4, linestyle="none",
                            label="Historical Total", zorder=5)

            tick_years = [y for y in all_years if y % 5 == 0]
            ax.set_xticks(tick_years)
            ax.set_xticklabels(tick_years, rotation=90, fontsize=6)
            ax.set_xlim(all_years[0] - 2, all_years[-1] + 2)
            ax.set_title(gas, fontsize=11, fontweight="bold")
            ax.set_xlabel("Year", fontsize=8)
            ax.set_ylabel(f"Emissions ({gas_unit})", fontsize=8)
            ax.grid(True, alpha=0.3, axis="y")

            if gas == gases_to_plot[-1] or idx == len(gases_to_plot) - 1:
                handles, labels = ax.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax.legend(by_label.values(), by_label.keys(), fontsize=7, loc="upper right")

        # Hide unused axes
        for idx in range(len(gases_to_plot), len(axes)):
            axes[idx].set_visible(False)

        fig.suptitle(
            f"Emissions by Sector — Scenario {marker_to_run.upper()} ({gridding_version})",
            fontsize=13, fontweight="bold",
        )
        plt.tight_layout()
        fig.savefig(out_04, dpi=300, bbox_inches="tight")
        plt.close(fig)
        log.info(f"[F]   Plot 04 saved → {out_04}")

    log.info(f"[F] Done ({time.time()-t0:.1f}s)")
    return [p for p in [out_03, out_04] if p.exists()]


# ── Module G: Place Timeseries Plots (alignment with historical) ──────────────

def make_place_timeseries_plots(
    gridded_folder: Path,
    qc_output_path: Path,
    settings,
    file_name_ending: str,
    plot_gases: list[str] | None = None,
    plot_sectors: list[str] | None = None,
    locations: dict[str, tuple[float, float]] | None = None,
    skip_existing: bool = True,
    logger: logging.Logger | None = None,
) -> list[Path]:
    """
    Module G: per-location timeseries plots comparing gridded scenario data
    against CEDS historical data. Mirrors workflow_cmip7-fast-track.py §4.1.

    For each anthro .nc file, generates single-gridpoint and area-average
    timeseries plots for each sector × location combination.

    Outputs
    -------
    qc_output/plots/place_timeseries/{place}_timeseries_{gas}_{sector}.png
    qc_output/plots/place_timeseries/{place}_area_timeseries_{gas}_{sector}.png
    """
    log = logger or logging.getLogger(__name__)
    t0 = time.time()

    out_dir = qc_output_path / "plots" / "place_timeseries"
    out_dir.mkdir(parents=True, exist_ok=True)

    if plot_gases is None:
        plot_gases = list(np.unique(GASES_ESGF_CEDS + GASES_ESGF_BB4CMIP))
    if plot_sectors is None:
        plot_sectors = list(np.unique(
            cmip7_utils.SECTOR_ORDERING_DEFAULT["CO2_em_anthro"]
            + cmip7_utils.SECTOR_ORDERING_DEFAULT["em_anthro"]
            + cmip7_utils.SECTOR_ORDERING_DEFAULT["em_openburning"]
        ))
    if locations is None:
        locations = {
            "Laxenburg": (48.0689, 16.3555),
            "South Sudan": (6.8770, 31.3070),
            # 'Beijing': (39.9042, 116.4074),
            # "Nuuk": (64.1743, -51.7373),
            # 'Geneva': (46.2044, 6.1432),
            # 'Delhi': (28.6139, 77.2090),
            # 'Spain': (40.4637, 3.7492), # central spain, close to Madrid
            # 'New_York': (40.7128, -74.0060),
            # 'London': (51.5074, -0.1278),
            # 'Tokyo': (35.6762, 139.6503),
            # 'São_Paulo': (-23.5505, -46.6333),
            # 'Lagos': (6.5244, 3.3792),
            # 'Mumbai': (19.0760, 72.8777),
            # 'Rural_Amazon': (-3.4653, -62.2159),  # Remote area in Amazon
            # 'North_Atlantic': (45.0, -30.0),     # Shipping route
            # 'South_China_Sea': (12.0, 113.0)     # Shipping route
        }

    ceds_data_location = settings.postprocess_path / "CMIP7_anthro"
    saved: list[Path] = []

    for file in sorted(gridded_folder.glob(f"*-em-anthro_{file_name_ending}")):
        gas_name, file_type = _parse_nc_filename(file.name)

        if file_type != "anthro":
            continue
        if gas_name not in plot_gases:
            continue

        ceds_match = next(ceds_data_location.glob(f"{gas_name}-*.nc"), None)
        if ceds_match is None:
            log.warning(f"[G] No CEDS file found for {gas_name} in {ceds_data_location}")
            continue

        scen_ds = xr.open_dataset(file)
        ceds_ds = xr.open_dataset(ceds_match)

        available_sectors = [
            k for k in scen_ds.sector.values
            if cmip7_utils.SECTOR_DICT_ANTHRO_CO2_SCENARIO.get(k) in plot_sectors
        ]

        for sec in available_sectors:
            sector_name = cmip7_utils.SECTOR_DICT_ANTHRO_CO2_SCENARIO[sec]

            for place, (lat, lon) in locations.items():
                out_ts   = out_dir / f"{place}_timeseries_{gas_name}_{sector_name}.png"
                out_area = out_dir / f"{place}_area_timeseries_{gas_name}_{sector_name}.png"

                if skip_existing and out_ts.exists() and out_area.exists():
                    log.info(f"[G] Skipping {place} {gas_name} {sector_name} (already exists)")
                    continue

                log.info(f"[G] {place} {gas_name} {sector_name}")
                try:
                    fig1, _ = plot_place_timeseries(
                        ceds_ds, scen_ds,
                        lat=lat, lon=lon, place=place,
                        gas=gas_name, sector=sec, sector_name=sector_name,
                        type="em_anthro",
                    )
                    fig1.savefig(out_ts, dpi=300, bbox_inches="tight")
                    plt.close(fig1)
                    saved.append(out_ts)

                    fig2, _ = plot_place_area_average_timeseries(
                        ceds_ds, scen_ds,
                        lat=lat, lon=lon, place=place,
                        gas=gas_name, sector=sec, sector_name=sector_name,
                        lat_range=2.0, lon_range=2.0,
                        type="em_anthro",
                    )
                    fig2.savefig(out_area, dpi=300, bbox_inches="tight")
                    plt.close(fig2)
                    saved.append(out_area)

                except Exception as e:
                    log.warning(f"[G] Error for {place} {gas_name} {sector_name}: {e}")

        scen_ds.close()
        ceds_ds.close()

    log.info(f"[G] Done — {len(saved)} plots saved ({time.time()-t0:.1f}s)")
    return saved


# ── Module H: Openburning Place Timeseries Plots (alignment with BB4CMIP7 history) ──

def make_openburning_place_timeseries_plots(
    gridded_folder: Path,
    qc_output_path: Path,
    settings,
    file_name_ending: str,
    plot_gases: list[str] | None = None,
    plot_sectors: list[str] | None = None,
    locations: dict[str, tuple[float, float]] | None = None,
    skip_existing: bool = True,
    logger: logging.Logger | None = None,
) -> list[Path]:
    """
    Module H: per-location timeseries plots comparing gridded openburning scenario
    data against BB4CMIP7 historical data. Mirrors Module G but for openburning.

    For each openburning .nc file, generates single-gridpoint and area-average
    timeseries plots for each sector × location combination. Sector integers
    (0–3) are resolved to human-readable names via SECTOR_DICT_OPENBURNING_DEFAULT.

    Historical reference data is read from:
        settings.postprocess_path / "bb4cmip7"   (i.e. <gridding_path>/esgf/ceds/bb4cmip7/)
    Files there should follow the same naming convention as the CEDS anthro files:
        {gas}-em-openburning_*.nc

    CO2 openburning is skipped (zero by design).

    Outputs
    -------
    qc_output/plots/openburning_timeseries/{place}_timeseries_{gas}_{sector}.png
    qc_output/plots/openburning_timeseries/{place}_area_timeseries_{gas}_{sector}.png
    """
    log = logger or logging.getLogger(__name__)
    t0 = time.time()

    out_dir = qc_output_path / "plots" / "openburning_timeseries"
    out_dir.mkdir(parents=True, exist_ok=True)

    if plot_gases is None:
        plot_gases = [g for g in GASES_ESGF_BB4CMIP if g != "CO2"]
    if plot_sectors is None:
        plot_sectors = list(cmip7_utils.SECTOR_ORDERING_DEFAULT["em_openburning"])
    if locations is None:
        locations = {
            "Laxenburg": (48.0689, 16.3555),
            "South Sudan": (6.8770, 31.3070),
            # 'Beijing': (39.9042, 116.4074),
            # "Nuuk": (64.1743, -51.7373),
            # 'Geneva': (46.2044, 6.1432),
            # 'Delhi': (28.6139, 77.2090),
            # 'Spain': (40.4637, 3.7492),
            # 'New_York': (40.7128, -74.0060),
            # 'London': (51.5074, -0.1278),
            # 'Tokyo': (35.6762, 139.6503),
            # 'São_Paulo': (-23.5505, -46.6333),
            # 'Lagos': (6.5244, 3.3792),
            # 'Mumbai': (19.0760, 72.8777),
            # 'Rural_Amazon': (-3.4653, -62.2159),
            # 'North_Atlantic': (45.0, -30.0),
            # 'South_China_Sea': (12.0, 113.0)
        }

    bb4cmip_data_location = settings.postprocess_path / "bb4cmip7"
    saved: list[Path] = []

    for file in sorted(gridded_folder.glob(f"*-em-openburning_{file_name_ending}")):
        gas_name, file_type = _parse_nc_filename(file.name)

        if file_type != "openburning":
            continue
        if gas_name == "CO2":
            log.info("[H] Skipping CO2 openburning (zero by design)")
            continue
        if gas_name not in plot_gases:
            continue

        bb4cmip_match = next(bb4cmip_data_location.glob(f"{gas_name}-*.nc"), None)
        if bb4cmip_match is None:
            log.warning(f"[H] No BB4CMIP7 file found for {gas_name} in {bb4cmip_data_location}")
            continue

        scen_ds = xr.open_dataset(file)
        bb4cmip_ds = xr.open_dataset(bb4cmip_match)

        available_sectors = [
            k for k in scen_ds.sector.values
            if cmip7_utils.SECTOR_DICT_OPENBURNING_DEFAULT.get(k) in plot_sectors
        ]

        for sec in available_sectors:
            sector_name = cmip7_utils.SECTOR_DICT_OPENBURNING_DEFAULT[sec]

            for place, (lat, lon) in locations.items():
                out_ts   = out_dir / f"{place}_timeseries_{gas_name}_{sector_name}.png"
                out_area = out_dir / f"{place}_area_timeseries_{gas_name}_{sector_name}.png"

                if skip_existing and out_ts.exists() and out_area.exists():
                    log.info(f"[H] Skipping {place} {gas_name} {sector_name} (already exists)")
                    continue

                log.info(f"[H] {place} {gas_name} {sector_name}")
                try:
                    fig1, _ = plot_place_timeseries(
                        bb4cmip_ds, scen_ds,
                        lat=lat, lon=lon, place=place,
                        gas=gas_name, sector=sec, sector_name=sector_name,
                        type="em_openburning",
                    )
                    fig1.savefig(out_ts, dpi=300, bbox_inches="tight")
                    plt.close(fig1)
                    saved.append(out_ts)

                    fig2, _ = plot_place_area_average_timeseries(
                        bb4cmip_ds, scen_ds,
                        lat=lat, lon=lon, place=place,
                        gas=gas_name, sector=sec, sector_name=sector_name,
                        lat_range=2.0, lon_range=2.0,
                        type="em_openburning",
                    )
                    fig2.savefig(out_area, dpi=300, bbox_inches="tight")
                    plt.close(fig2)
                    saved.append(out_area)

                except Exception as e:
                    log.warning(f"[H] Error for {place} {gas_name} {sector_name}: {e}")

        scen_ds.close()
        bb4cmip_ds.close()

    log.info(f"[H] Done — {len(saved)} plots saved ({time.time()-t0:.1f}s)")
    return saved


# ── Orchestrator ──────────────────────────────────────────────────────────────

def run_qc(
    gridded_scenario_folder: Path,
    marker_to_run: str,
    settings_file: str = "config_cmip7_v0-4-0.yaml",
    gridding_version: str | None = None,
    version_esgf: str = "1-1-0",
    history_file: str = "country-history_202511261223_202511040855_202512032146_202512021030_7e32405ade790677a6022ff498395bff00d9792d.csv",
    run_file_inventory: bool = True,
    run_min_max: bool = True,
    run_downscaled_qc: bool = True,
    run_annual_totals: bool = True,
    run_animations: bool = False,
    animation_mode: str | list[str] = "all-sectors",
    run_doc_plots: bool = True,
    run_place_timeseries: bool = False,
    run_openburning_place_timeseries: bool = False,
    species_filter: list[str] | None = None,
    skip_existing: bool = True,
    here: Path | None = None,
) -> dict[str, Path | list | pd.DataFrame]:
    """
    Run all selected QC modules for one gridded scenario.

    Parameters
    ----------
    gridded_scenario_folder
        Path to the scenario output folder, e.g. results/h_1-1-0/
    marker_to_run
        Scenario marker, e.g. "h", "vl"
    gridding_version
        Version string, e.g. "h_1-1-0".  Derived from marker+version_esgf if None.
    run_animations
        GIF animations are off by default (slow).  Enable explicitly.
    animation_mode
        One of "all-sectors" (one GIF per gas/file_type/sector),
        "total-per-file" (sectors summed within each file), or
        "total-per-species" (all files summed per gas).
        Can also be a list of these strings to run multiple modes in one pass.
    """
    if gridding_version is None:
        gridding_version = f"{marker_to_run}_{version_esgf}"

    gridded_scenario_folder = Path(gridded_scenario_folder)
    qc_output_path = gridded_scenario_folder / "qc_output_v2"
    qc_output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log = setup_logging(qc_output_path, timestamp)

    log.info(f"=== CMIP7 QC: marker={marker_to_run}  version={gridding_version} ===")
    log.info(f"    Scenario folder : {gridded_scenario_folder}")
    log.info(f"    QC output       : {qc_output_path}")
    log.info(f"    Species filter  : {species_filter}")
    t_total = time.time()

    if here is None:
        here = _find_here()

    # ── Resolve settings and paths ─────────────────────────────────────────
    (
        settings,
        version_path,
        gridded_folder,
        model_selection,
        scenario_selection,
        scenario_selection_gridded,
        file_name_ending,
    ) = resolve_settings_and_paths(
        gridded_scenario_folder,
        settings_file,
        marker_to_run,
        gridding_version,
        here,
    )
    log.info(f"    Model           : {model_selection}")
    log.info(f"    Scenario        : {scenario_selection}")
    log.info(f"    Gridded folder  : {gridded_folder}")

    # ── Load cell area (needed for D and E) ────────────────────────────────
    cell_area = None
    if run_annual_totals or run_animations:
        try:
            cell_area = load_cell_area(settings)
            log.info("    Cell area loaded OK")
        except Exception as e:
            log.warning(f"    Could not load cell area: {e} — modules D/E may fail")

    results = {}

    # ── Module A ───────────────────────────────────────────────────────────
    file_inventory = None
    if run_file_inventory:
        file_inventory = check_file_inventory(
            gridded_folder=gridded_folder,
            qc_output_path=qc_output_path,
            species_filter=species_filter,
            skip_existing=skip_existing,
            logger=log,
        )
        results["file_inventory"] = qc_output_path / "tables" / "file_inventory.csv"

    # ── Module B ───────────────────────────────────────────────────────────
    if run_min_max:
        check_min_max_values(
            gridded_folder=gridded_folder,
            qc_output_path=qc_output_path,
            file_inventory=file_inventory,
            species_filter=species_filter,
            skip_existing=skip_existing,
            logger=log,
        )
        results["min_max_stats"] = qc_output_path / "tables" / "min_max_stats.csv"

    # ── Module C ───────────────────────────────────────────────────────────
    if run_downscaled_qc:
        check_downscaled_qc(
            version_path=version_path,
            qc_output_path=qc_output_path,
            gridding_version=gridding_version,
            skip_existing=skip_existing,
            logger=log,
        )
        results["downscaled_qc"] = qc_output_path / "tables" / "downscaled_qc_results.csv"

    # ── Module D ───────────────────────────────────────────────────────────
    if run_annual_totals and cell_area is not None:
        # Workflow outputs (harmonization CSV, scenarios_processed CSV) are written to
        # settings.out_path / gridding_version, which may differ from gridded_scenario_folder
        # when the gridded data lives in a shared/external directory.
        workflow_version_path = settings.out_path / gridding_version
        log.info(f"    Workflow path   : {workflow_version_path}")
        check_annual_totals(
            version_path=workflow_version_path,
            gridded_folder=gridded_folder,
            qc_output_path=qc_output_path,
            gridding_version=gridding_version,
            scenario_selection=scenario_selection,
            cell_area=cell_area,
            species_filter=species_filter,
            skip_existing=skip_existing,
            logger=log,
        )
        results["annual_totals"] = qc_output_path / "tables" / "annual_totals_comparison.csv"

    # ── Module E ───────────────────────────────────────────────────────────
    if run_animations and cell_area is not None:
        modes = [animation_mode] if isinstance(animation_mode, str) else list(animation_mode)
        out_gifs: list[Path] = []
        for mode in modes:
            out_gifs.extend(make_animated_grids(
                gridded_folder=gridded_folder,
                qc_output_path=qc_output_path,
                cell_area=cell_area,
                file_inventory=file_inventory,
                species_filter=species_filter,
                skip_existing=skip_existing,
                animation_mode=mode,
                logger=log,
            ))
        results["animations"] = out_gifs

    # ── Module F ───────────────────────────────────────────────────────────
    if run_doc_plots:
        out_plots = make_doc_plots(
            version_path=version_path,
            qc_output_path=qc_output_path,
            gridding_version=gridding_version,
            marker_to_run=marker_to_run,
            settings=settings,
            history_file=history_file,
            species_filter=species_filter,
            skip_existing=skip_existing,
            logger=log,
        )
        results["doc_plots"] = out_plots

    # ── Module G ───────────────────────────────────────────────────────────
    if run_place_timeseries:
        out_place_plots = make_place_timeseries_plots(
            gridded_folder=gridded_folder,
            qc_output_path=qc_output_path,
            settings=settings,
            file_name_ending=file_name_ending,
            plot_gases=species_filter,
            skip_existing=skip_existing,
            logger=log,
        )
        results["place_timeseries"] = out_place_plots

    # ── Module H ───────────────────────────────────────────────────────────
    if run_openburning_place_timeseries:
        out_ob_plots = make_openburning_place_timeseries_plots(
            gridded_folder=gridded_folder,
            qc_output_path=qc_output_path,
            settings=settings,
            file_name_ending=file_name_ending,
            plot_gases=species_filter,
            skip_existing=skip_existing,
            logger=log,
        )
        results["openburning_place_timeseries"] = out_ob_plots

    log.info(f"=== QC COMPLETE in {time.time()-t_total:.1f}s ===")
    log.info(f"    Outputs in: {qc_output_path}")
    return results


# %% [markdown]
# ## Run

# %%
if __name__ == "__main__":
    HERE = _find_here()

    _gridded_scenario_folder = (
        Path(FOLDER_WITH_GRIDDED_DATA)
        if FOLDER_WITH_GRIDDED_DATA
        else HERE.parent.parent / "results" / GRIDDING_VERSION
    )

    run_qc(
        gridded_scenario_folder=_gridded_scenario_folder,
        # gridded_scenario_folder=HERE.parent.parent / "results" / GRIDDING_VERSION
        # gridded_scenario_folder="C:/Users/kikstra/IIASA/ECE.prog - Documents/Projects/CMIP7/IAM Data Processing/Shared emission fields data/v1_1-testing-findmistakes/results_20260316_ln",
        # gridded_scenario_folder="C:/Users/kikstra/IIASA/ECE.prog - Documents/Projects/CMIP7/IAM Data Processing/Shared emission fields data/v1_1-testing-findmistakes/results_20260316_ln",
        # gridded_scenario_folder="C:/Users/kikstra/IIASA/ECE.prog - Documents/Projects/CMIP7/IAM Data Processing/Shared emission fields data/v1_1-testing-findmistakes/results_20260316_m",
        marker_to_run=marker_to_run,
        settings_file=SETTINGS_FILE,
        gridding_version=GRIDDING_VERSION,
        version_esgf=VERSION_ESGF,
        history_file=HISTORY_FILE,
        run_file_inventory=run_file_inventory,
        run_min_max=run_min_max,
        run_downscaled_qc=run_downscaled_qc,
        run_annual_totals=run_annual_totals,
        run_animations=run_animations,
        run_doc_plots=run_doc_plots,
        run_place_timeseries=run_place_timeseries,
        species_filter=species_filter,
        skip_existing=skip_existing,
        here=HERE,
    )
