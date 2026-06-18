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
# # Junction QC: Timeseries Continuity for CMIP7 Extensions
#
# Verifies that gridded emissions link smoothly across three temporal segments:
#
# | Segment         | Period       | Source      |
# |-----------------|--------------|-------------|
# | CEDS historical | 2000–2023    | CEDS files  |
# | Fast-track      | 2022–2100    | v1_1 output |
# | Extension       | 2105–2500    | this run    |
#
# **Note:** The 4-year gap 2101–2104 is expected — IAM extension data starts at 2105
# and `DROP_ANCHOR_TIMESTEP=True` removes 2100 from the extension output.
#
# ## Modules
#
# - **Module I** — Global annual totals: three-segment timeseries per (gas, file_type).
#   Reports a junction jump metric comparing the extrapolated fast-track trend at 2105
#   against the actual extension value at 2105.
#
# - **Module J** — Grid-point monthly timeseries at representative locations.
#   Shows the window [anchor − 10 yr, anchor + 60 yr] to capture both the junction
#   and the fade-correction convergence by 2150.
#
# - **Module K** *(off by default)* — Reads the 2100 diagnostic netCDFs written by the
#   main workflow (`run_2100_alignment_diagnostic=True`) and writes two CSVs into
#   `junctions/2100_diagnostic/`:
#     (1) `2100_diagnostic_summary.csv` — RAW extension-2100 vs fast-track-2100 agreement
#         *before* the additive offset forces them equal (area-weighted rel diff, spatial
#         correlation, max abs diff).
#     (2) `2100_correction_zero_check.csv` — CORRECTED extension-2100 vs fast-track-2100
#         *after* the offset, confirming the difference is ~0 before 2100 is dropped
#         (max abs diff vs tolerance, PASS/FAIL per file).

# %% [markdown]
# ## Parameters

# %% editable=true slideshow={"slide_type": ""} tags=["parameters"]
SETTINGS_FILE: str = "config_cmip7_v0-4-0-EXT.yaml"
VERSION_ESGF: str = "1-1-1"
marker_to_run: str = "m"
IAM_SCENARIO: str = "SSP2 - Medium Emissions"  # scenario for marker_to_run="vl" is "SSP1 - Very Low Emissions"

GRIDDING_VERSION: str = f"{marker_to_run}-ext_{VERSION_ESGF}"
# Set to an absolute path to override auto-detection from GRIDDING_VERSION
FOLDER_WITH_EXTENSION_DATA: str = ""

from pathlib import Path
LOCATION_FASTTRACK_GRIDDED: Path = Path(
    "/Users/jarmo/Library/CloudStorage/OneDrive-SharedLibraries-IIASA/"
    "ECE.prog - Documents/Projects/CMIP7/IAM Data Processing/"
    "Shared emission fields data/v1_1"
)

# Must match the values used when producing the extension files
FADE_ANCHOR_YEAR: int = 2100
FADE_CONVERGENCE_YEAR: int = 2150
DROP_ANCHOR_TIMESTEP: bool = True  # extension starts at FADE_ANCHOR_YEAR + 5 when True

# Module flags
run_global_continuity: bool = True
run_gridpoint_continuity: bool = True
# Module K — summarise the raw extension-2100 vs fast-track-2100 diagnostic netCDFs
# (produced by workflow_cmip7-extensions.py when run_2100_alignment_diagnostic=True,
# written to '<results>/<gridding_version>/diagnostics_2100/'). Off by default — only
# useful when those diagnostic files exist.
run_2100_diagnostic_summary: bool = True

# Optional: restrict to a subset of species (None = all found in extension folder)
species_filter: list[str] | None = None  # e.g. ["CO2", "NH3"]
# species_filter: list[str] | None = ["CO2","NH3","SO2"]

# Optional: restrict to a subset of file type / sector files (None = all found in extension folder)
FILE_TYPE_FILTER: list[str] | None = None  # e.g. ["anthro", "AIR-anthro", "openburning"]
# FILE_TYPE_FILTER: list[str] | None = ["anthro"]

# Representative locations for Module J
LOCATIONS: dict[str, tuple[float, float]] = {
    "Laxenburg": (48.0689, 16.3555),
    "South_Sudan": (6.8770, 31.3070),
    "Beijing": (39.9042, 116.4074),
    "Sao_Paulo": (-23.5505, -46.6333),
    "Rural_Amazon": (-3.4653, -62.2159),
}

skip_existing: bool = True
max_workers: int = 4  # parallel species processing (set to 1 to disable)

# IAM reference overlay for Module I global plots (set to "" to disable)
IAM_CSV: str = (
    "/Users/jarmo/Library/CloudStorage/OneDrive-SharedLibraries-IIASA/"
    "ECE.prog - Documents/Projects/CMIP7/IAM Data Processing/"
    "Shared emission fields data/v1_1/extensions_timeseries/"
    "extensions_full_emissions_timeseries_2023_2500.csv"
)

# %% [markdown]
# ## Imports

# %%
import concurrent.futures
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from concordia.cmip7 import utils as cmip7_utils
from concordia.cmip7.utils_plotting import ds_to_annual_emissions_total_faster
from concordia.settings import Settings

# %%
# ── Constants ─────────────────────────────────────────────────────────────────

SEGMENT_COLORS = {
    "ceds":       "#888888",
    "fast_track": "#1f77b4",
    "extension":  "#d62728",
    "iam":        "#2ca02c",
}
SEGMENT_LABELS = {
    "ceds":       "CEDS historical",
    "fast_track": "Fast-track (2022–2100)",
    "extension":  "Extension (2105–2500)",
    "iam":        "IAM reference",
}

SECTOR_RENAME_DOWNSCALED: dict[str, str] = {
    "Energy Sector":                "Energy",
    "Industrial Sector":            "Industrial",
    "Residential Commercial Other": "Residential, Commercial, Other",
    "Transportation Sector":        "Transportation",
}

_SECTOR_TO_FILETYPE: dict[str, str] = {
    "Agriculture":                         "anthro",
    "Energy":                              "anthro",
    "Industrial":                          "anthro",
    "Transportation":                      "anthro",
    "Residential, Commercial, Other":      "anthro",
    "Solvents Production and Application": "anthro",
    "Waste":                               "anthro",
    "International Shipping":              "anthro",
    # CO2 CDR sectors — all mapped to anthro (matching fast-track's
    # "Other Capture and Removal" consolidation for the gridded CO2 anthro file)
    "BECCS":                               "anthro",
    "Other CDR":                           "anthro",
    "Direct Air Capture":                  "anthro",
    "Enhanced Weathering":                 "anthro",
    "Biochar":                             "anthro",
    "Soil Carbon Management":              "anthro",
    "Ocean":                               "anthro",
    "Aircraft":                            "AIR-anthro",
    "Agricultural Waste Burning":          "openburning",
    "Forest Burning":                      "openburning",
    "Grassland Burning":                   "openburning",
    "Peat Burning":                        "openburning",
}

# Years to show in gridpoint plots relative to the anchor year
_GRIDPOINT_WINDOW_BEFORE: int = 10
_GRIDPOINT_WINDOW_AFTER: int = 60   # covers the full fade period (anchor → convergence is 50 yr)


# ── Script directory (resolved once at import / cell-execution time) ──────────
#
# Computing this at module level is the only reliable approach for VSCode
# Interactive Window: some kernels start with CWD='/' so a search inside
# _find_here() at *call* time fails. __file__ (when defined) is set by the
# kernel when the cell that contains this line is executed — before any CWD
# drift can occur.

def _is_concordia_root(p: Path) -> bool:
    return (p / "pyproject.toml").exists() and (p / "src" / "concordia").exists()


_SCRIPT_DIR: Path | None = None

try:
    _p = Path(__file__).resolve().parent
    # Guard against __file__ being a bare filename that resolves to CWD or '/'.
    # A genuine script directory has depth > 2 (e.g. /Users/…/notebooks/cmip7).
    if len(_p.parts) > 2 and _p.is_dir():
        _SCRIPT_DIR = _p
except NameError:
    pass  # __file__ not defined in this kernel

if _SCRIPT_DIR is None:
    # Upward search from CWD (works when kernel CWD is inside the concordia tree)
    _cwd = Path.cwd()
    for _c in [_cwd] + list(_cwd.parents):
        if _is_concordia_root(_c):
            _SCRIPT_DIR = _c / "notebooks" / "cmip7"
            break

if _SCRIPT_DIR is None:
    # Downward fallback: kernel CWD is a *parent* repo (e.g. mozart/ or similar)
    _cwd = Path.cwd()
    for _rel in (Path("projects") / "mine" / "concordia", Path("concordia")):
        _c = _cwd / _rel
        if _is_concordia_root(_c):
            _SCRIPT_DIR = _c / "notebooks" / "cmip7"
            break


# ── Helpers ───────────────────────────────────────────────────────────────────

def _find_here() -> Path:
    """Return notebooks/cmip7.  Relies on _SCRIPT_DIR computed at import time."""
    if _SCRIPT_DIR is not None:
        return _SCRIPT_DIR
    raise RuntimeError(
        "_find_here() could not locate the concordia notebooks/cmip7 directory.\n"
        f"  CWD = {Path.cwd()}\n"
        "  Set SETTINGS_FILE to an absolute path as a workaround."
    )


def _setup_logging(output_path: Path, timestamp: str) -> logging.Logger:
    log_dir = output_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"junction_qc_{timestamp}.txt"

    logger = logging.getLogger(f"junction_qc_{timestamp}")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def _parse_nc_filename(filename: str) -> tuple[str, str]:
    """Return (gas, file_type) from a gridded NC filename."""
    stem = Path(filename).stem
    core = stem.split("_")[0] if "_" in stem else stem
    if "-em-AIR-" in core:
        return core.split("-em-AIR-")[0], "AIR-anthro"
    if "-em-" in core:
        parts = core.split("-em-", 1)
        return parts[0], parts[1]
    raise ValueError(f"Cannot parse gas/type from: {filename}")


def _find_nc_file(folder: Path, gas: str, file_type: str) -> Path | None:
    """Return the first NC file matching {gas}-em-{file_type}_*.nc, or None."""
    if not folder.exists():
        return None
    matches = sorted(folder.glob(f"{gas}-em-{file_type}_*.nc"))
    return matches[0] if matches else None


def _ceds_folder_for_type(settings, file_type: str) -> Path | None:
    """Return the CEDS historical folder path for a given file_type."""
    post = Path(settings.postprocess_path)
    mapping = {
        "anthro":     post / "CMIP7_anthro",
        "AIR-anthro": post / "CMIP7_AIR",
        "openburning": post / "bb4cmip7",
    }
    return mapping.get(file_type)


def _load_cell_area(settings) -> xr.DataArray:
    gridding_path = Path(settings.gridding_path)
    candidates = sorted(gridding_path.glob("areacella*.nc"))
    if not candidates:
        raise FileNotFoundError(f"No areacella*.nc found in {gridding_path}")
    # .load() forces numpy-backed array — required for safe pickling into worker processes
    return xr.open_dataset(candidates[0])["areacella"].load()


def _annual_totals_from_nc(
    nc_path: Path, cell_area: xr.DataArray
) -> pd.Series | None:
    """
    Compute global annual totals in Mt/yr from a gridded NC file.
    Returns a pd.Series indexed by integer year, or None on error.
    """
    try:
        ds = xr.open_dataset(nc_path, use_cftime=True, engine="netcdf4")
        var_name = next(v for v in ds.data_vars if not v.endswith("_bnds"))
        da = ds_to_annual_emissions_total_faster(ds, var_name, cell_area, keep_sectors=False)
        series = da.to_series()
        series.index = series.index.astype(int)
        ds.close()
        return series
    except Exception:
        return None


def _in_jupyter() -> bool:
    """Return True when running inside a Jupyter/IPython kernel."""
    return "ipykernel" in sys.modules


def _is_year_col(col: str) -> bool:
    try:
        float(col)
        return True
    except (ValueError, TypeError):
        return False


def _load_iam_reference_totals(
    csv_path: Path,
    scenario: str,
    workflow: str = "gridding",
    # region: str = "World", # not world; on the gridding level, world is only used for aircraft, and international shipping
) -> dict[tuple[str, str], dict[int, float]]:
    """
    Load annual IAM reference totals per (gas, file_type) from the extensions
    timeseries CSV.  Returns {(gas, file_type): {year: value_Mt_yr}}.

    Applies SECTOR_RENAME_DOWNSCALED and _SECTOR_TO_FILETYPE to map variable
    names to the same (gas, file_type) keys used by the NC gridded files.
    N2O in kt N2O/yr is converted to Mt/yr to match the NC-derived totals.
    """
    df = pd.read_csv(csv_path)
    mask = (
        (df["scenario"] == scenario)
        & (df["workflow"] == workflow)
        # & (df["region"] == region)
    )
    df = df[mask]
    if df.empty:
        return {}

    year_cols = [c for c in df.columns if _is_year_col(c)]
    years = [int(float(c)) for c in year_cols]

    totals: dict[tuple[str, str], list[float]] = {}
    for _, row in df.iterrows():
        var = str(row["variable"])
        parts = var.split("|", 2)
        if len(parts) != 3 or parts[0] != "Emissions":
            continue
        gas = parts[1]
        sector = SECTOR_RENAME_DOWNSCALED.get(parts[2], parts[2])
        file_type = _SECTOR_TO_FILETYPE.get(sector)
        if file_type is None:
            continue

        unit = str(row.get("unit", ""))
        scale = 1e-3 if unit.startswith("kt") else 1.0  # kt → Mt

        key = (gas, file_type)
        if key not in totals:
            totals[key] = [0.0] * len(years)
        for i, c in enumerate(year_cols):
            v = row[c]
            if pd.notna(v):
                totals[key][i] += float(v) * scale

    return {k: dict(zip(years, v)) for k, v in totals.items()}


# ── Module I: Global Annual Totals Continuity ─────────────────────────────────

def _worker_global_species(params: dict) -> dict:
    """
    Module I per-species worker.  Top-level for ProcessPoolExecutor pickling.
    Returns {"rows": list[dict], "jump_row": dict | None}.
    """
    import matplotlib.pyplot as plt
    plt.switch_backend("Agg")

    ext_nc: Path           = params["ext_nc"]
    ft_folder: Path        = params["ft_folder"]
    ceds_folder: Path | None = params["ceds_folder"]
    cell_area: xr.DataArray = params["cell_area"]
    plots_dir: Path        = params["plots_dir"]
    skip_existing: bool    = params["skip_existing"]
    fade_anchor_year: int  = params["fade_anchor_year"]
    fade_convergence_year: int = params["fade_convergence_year"]
    iam_values: dict[int, float] | None = params.get("iam_values")
    iam_label: str         = params.get("iam_label", "IAM reference")
    logger_name: str       = params["logger_name"]

    log = logging.getLogger(logger_name)
    if not log.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        log.addHandler(h)
        log.setLevel(logging.DEBUG)

    try:
        gas, file_type = _parse_nc_filename(ext_nc.name)
    except ValueError:
        return {"rows": [], "jump_row": None}

    log.info(f"[I]   {gas} {file_type} ...")

    segments: dict[str, pd.Series | None] = {}

    ft_nc = _find_nc_file(ft_folder, gas, file_type)
    if ft_nc:
        segments["fast_track"] = _annual_totals_from_nc(ft_nc, cell_area)
        if segments["fast_track"] is None:
            log.warning(f"[I]     Failed to compute FT totals: {ft_nc.name}")
    else:
        log.warning(f"[I]     No fast-track file for {gas} {file_type} in {ft_folder}")
        segments["fast_track"] = None

    segments["extension"] = _annual_totals_from_nc(ext_nc, cell_area)
    if segments["extension"] is None:
        log.warning(f"[I]     Failed to compute extension totals: {ext_nc.name}")

    if ceds_folder:
        ceds_nc = _find_nc_file(ceds_folder, gas, file_type)
        if ceds_nc:
            segments["ceds"] = _annual_totals_from_nc(ceds_nc, cell_area)

    if iam_values:
        segments["iam"] = pd.Series(iam_values)

    rows: list[dict] = []
    for seg, series in segments.items():
        if series is None:
            continue
        for yr, val in series.items():
            rows.append({
                "gas": gas, "file_type": file_type,
                "segment": seg, "year": int(yr), "value_Mt_yr": float(val),
            })

    # ── Jump metric ────────────────────────────────────────────────────────────
    ft = segments.get("fast_track")
    ext = segments.get("extension")
    ext_start = fade_anchor_year + 5

    ft_at_anchor = float(ft.get(fade_anchor_year))     if ft  is not None and fade_anchor_year       in ft.index  else None
    ft_at_m5     = float(ft.get(fade_anchor_year - 5)) if ft  is not None and (fade_anchor_year - 5) in ft.index  else None
    ext_at_start = float(ext.get(ext_start))           if ext is not None and ext_start              in ext.index else None
    ext_at_p5    = float(ext.get(ext_start + 5))       if ext is not None and (ext_start + 5)        in ext.index else None

    ft_rate  = (ft_at_anchor - ft_at_m5) / 5   if (ft_at_anchor is not None and ft_at_m5   is not None) else None
    ext_rate = (ext_at_p5   - ext_at_start) / 5 if (ext_at_p5   is not None and ext_at_start is not None) else None

    ft_extrap_at_ext_start = (
        ft_at_anchor + ft_rate * 5
        if (ft_at_anchor is not None and ft_rate is not None) else None
    )

    deviation_pct = None
    if ft_extrap_at_ext_start is not None and ext_at_start is not None and ft_extrap_at_ext_start != 0:
        deviation_pct = (ext_at_start - ft_extrap_at_ext_start) / abs(ft_extrap_at_ext_start) * 100

    jump_row = {
        "gas": gas,
        "file_type": file_type,
        "ft_value_at_anchor": ft_at_anchor,
        "ft_extrap_at_ext_start": ft_extrap_at_ext_start,
        "ext_value_at_ext_start": ext_at_start,
        "deviation_from_extrap_pct": deviation_pct,
        "ft_rate_Mt_yr_per_yr": ft_rate,
        "ext_rate_Mt_yr_per_yr": ext_rate,
    }

    if deviation_pct is not None:
        log.info(f"[I]     Jump (deviation from extrapolated trend): {deviation_pct:+.2f}%")

    # ── Plot ───────────────────────────────────────────────────────────────────
    out_plot = plots_dir / f"global_continuity_{gas}_{file_type}.png"
    if not (skip_existing and out_plot.exists() and iam_values is None):
        # Build per-segment (years, values) tuples for clean plot access
        seg_series: dict[str, tuple[list, list]] = {}
        for seg, series in segments.items():
            if series is not None:
                yrs = sorted(series.index.tolist())
                seg_series[seg] = (yrs, [float(series[y]) for y in yrs])

        fig, ax = plt.subplots(figsize=(14, 4))
        ax.set_title(f"Global annual totals — {gas} | {file_type}", fontsize=12, fontweight="bold")
        ax.set_ylabel("Emissions (Mt/yr)")
        ax.set_xlabel("Year")

        for seg in ["ceds", "fast_track", "extension", "iam"]:
            if seg in seg_series:
                yrs, vals = seg_series[seg]
                label = iam_label if seg == "iam" else SEGMENT_LABELS[seg]
                lw = 1.2 if seg == "iam" else 1.5
                ls = "--" if seg == "iam" else "-"
                ax.plot(yrs, vals, color=SEGMENT_COLORS[seg], linewidth=lw,
                        linestyle=ls, label=label, zorder=2)

        # Dashed bridge across the gap
        if ft_at_anchor is not None and ext_at_start is not None:
            ax.plot(
                [fade_anchor_year, ext_start],
                [ft_at_anchor, ext_at_start],
                color="gray", linestyle="--", linewidth=1.1, alpha=0.65,
                label="Interpolated gap", zorder=2,
            )

        ax.axvline(fade_anchor_year, color="black", linestyle="--", linewidth=0.9,
                   label=f"Anchor year ({fade_anchor_year})")
        ax.axvline(ext_start, color="dimgray", linestyle=":", linewidth=0.9,
                   label=f"Extension start ({ext_start})")
        ax.axvspan(fade_anchor_year + 0.5, ext_start - 0.5,
                   color="lightyellow", alpha=0.6, zorder=0,
                   label=f"Gap ({fade_anchor_year + 1}–{ext_start - 1})")
        ax.axvline(fade_convergence_year, color="orange", linestyle=":", linewidth=0.9,
                   alpha=0.8, label=f"Fade end ({fade_convergence_year})")

        if deviation_pct is not None and ext_at_start is not None:
            flag = " ⚠" if abs(deviation_pct) > 5 else " ✓"
            ax.annotate(
                f"Dev: {deviation_pct:+.1f}%{flag}",
                xy=(ext_start, ext_at_start),
                xytext=(ext_start + 10, ext_at_start),
                fontsize=9,
                arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
                va="center",
            )

        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(out_plot, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info(f"[I]     Plot → {out_plot.name}")

    return {"rows": rows, "jump_row": jump_row}


def check_global_continuity(
    ext_folder: Path,
    ft_folder: Path,
    qc_output_path: Path,
    settings,
    cell_area: xr.DataArray,
    species_filter: list[str] | None = None,
    file_type_filter: list[str] | None = None,
    fade_anchor_year: int = 2100,
    fade_convergence_year: int = 2150,
    skip_existing: bool = True,
    max_workers: int = 4,
    iam_reference_csv: Path | None = None,
    iam_scenario: str | None = None,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """
    Module I: Three-segment global annual totals continuity.

    For each (gas, file_type) found in the extension folder, computes global
    annual totals (Mt/yr) from the CEDS historical files, the fast-track files,
    and the extension files.  Produces one timeseries plot per (gas, file_type)
    and a CSV of junction jump metrics.

    If iam_reference_csv and iam_scenario are provided, a fourth line is added
    to each plot showing the IAM reference totals (summed across matching sectors
    from the extensions timeseries CSV).

    Jump metric: how much does EXT[anchor+5] deviate from the linear
    extrapolation of the FT trend at the anchor year?

    Outputs
    -------
    qc_output/junctions/global_continuity.csv
    qc_output/junctions/global_junction_metrics.csv
    qc_output/junctions/plots/global_continuity_{gas}_{type}.png
    """
    log = logger or logging.getLogger(__name__)
    out_dir = qc_output_path / "junctions"
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "global_continuity.csv"
    jump_csv = out_dir / "global_junction_metrics.csv"

    if skip_existing and out_csv.exists() and jump_csv.exists() and iam_reference_csv is None:
        log.info(f"[I] Skipping (already exists): {out_csv}")
        return pd.read_csv(out_csv)

    log.info("[I] Running global continuity check...")
    t0 = time.time()

    ext_files = sorted(ext_folder.glob("*.nc"))
    if species_filter:
        ext_files = [
            f for f in ext_files
            if any(f.name.startswith(g + "-") for g in species_filter)
        ]
    if file_type_filter:
        ext_files = [
            f for f in ext_files
            if any(f"-em-{ft}_" in f.name or f"-em-{ft}-" in f.name for ft in file_type_filter)
        ]

    # Pre-resolve ceds folders per file_type — avoids passing settings across process boundary
    def _ceds_for_nc(nc: Path) -> Path | None:
        try:
            _, ft = _parse_nc_filename(nc.name)
        except ValueError:
            return None
        return _ceds_folder_for_type(settings, ft)

    iam_totals: dict[tuple[str, str], dict[int, float]] = {}
    iam_label = "IAM reference"
    if iam_reference_csv is not None and iam_scenario is not None:
        log.info(f"[I] Loading IAM reference from CSV: {iam_scenario}")
        iam_totals = _load_iam_reference_totals(iam_reference_csv, iam_scenario)
        iam_label = f"IAM ref ({iam_scenario})"
        log.info(f"[I]   Loaded {len(iam_totals)} (gas, file_type) series from CSV")

    work_items = [
        {
            "ext_nc": ext_nc,
            "ft_folder": ft_folder,
            "ceds_folder": _ceds_for_nc(ext_nc),
            "cell_area": cell_area,
            "plots_dir": plots_dir,
            "skip_existing": skip_existing,
            "fade_anchor_year": fade_anchor_year,
            "fade_convergence_year": fade_convergence_year,
            "iam_values": iam_totals.get(
                _parse_nc_filename(ext_nc.name) if "-em-" in ext_nc.name else None
            ),
            "iam_label": iam_label,
            "logger_name": log.name,
        }
        for ext_nc in ext_files
    ]

    all_rows: list[dict] = []
    jump_rows: list[dict] = []

    n_workers = min(max_workers, len(work_items)) if work_items else 1
    use_parallel = n_workers > 1 and not _in_jupyter()

    if use_parallel:
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as ex:
            for result in ex.map(_worker_global_species, work_items):
                all_rows.extend(result["rows"])
                if result["jump_row"] is not None:
                    jump_rows.append(result["jump_row"])
    else:
        for item in work_items:
            result = _worker_global_species(item)
            all_rows.extend(result["rows"])
            if result["jump_row"] is not None:
                jump_rows.append(result["jump_row"])

    df = pd.DataFrame(all_rows)
    df.to_csv(out_csv, index=False)

    jump_df = pd.DataFrame(jump_rows)
    jump_df.to_csv(jump_csv, index=False)

    log.info(f"[I] Done ({time.time() - t0:.1f}s) → {out_csv}, {jump_csv}")
    return df


# ── Module J: Gridpoint Continuity ────────────────────────────────────────────

def _worker_gridpoint_species(params: dict) -> dict:
    """
    Module J per-species worker.  Top-level for ProcessPoolExecutor pickling.

    Batch-extracts all locations in a single compute() per segment instead of
    one compute() per location, then iterates over the in-memory arrays for
    plotting.  Returns {"metric_rows": list[dict]}.
    """
    import matplotlib.pyplot as plt
    plt.switch_backend("Agg")

    ext_nc: Path              = params["ext_nc"]
    ft_nc: Path               = params["ft_nc"]
    ceds_nc: Path | None      = params["ceds_nc"]
    locations: dict           = params["locations"]
    plots_dir: Path           = params["plots_dir"]
    skip_existing: bool       = params["skip_existing"]
    fade_anchor_year: int     = params["fade_anchor_year"]
    fade_convergence_year: int = params["fade_convergence_year"]
    window_before: int        = params["window_before"]
    window_after: int         = params["window_after"]
    logger_name: str          = params["logger_name"]

    log = logging.getLogger(logger_name)
    if not log.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        log.addHandler(h)
        log.setLevel(logging.DEBUG)

    try:
        gas, file_type = _parse_nc_filename(ext_nc.name)
    except ValueError:
        return {"metric_rows": []}

    # Per-species skip guard: avoids opening files when all plots already exist
    if skip_existing:
        all_plots_exist = all(
            (plots_dir / f"gridpoint_{place}_{gas}_{file_type}.png").exists()
            for place in locations
        )
        if all_plots_exist:
            log.info(f"[J]   All plots exist for {gas} {file_type} — skipping")
            return {"metric_rows": []}

    log.info(f"[J]   {gas} {file_type}")

    try:
        ft_ds  = xr.open_dataset(ft_nc,  use_cftime=True, engine="netcdf4")
        ext_ds = xr.open_dataset(ext_nc, use_cftime=True, engine="netcdf4")
    except Exception as e:
        log.error(f"[J]   Cannot open files: {e}")
        return {"metric_rows": []}

    ft_var  = next(v for v in ft_ds.data_vars  if not v.endswith("_bnds"))
    ext_var = next(v for v in ext_ds.data_vars if not v.endswith("_bnds"))

    ft_year_start = fade_anchor_year - window_before
    ext_year_end  = fade_anchor_year + window_after

    ft_win  = ft_ds.sel( time=(ft_ds.time.dt.year  >= ft_year_start))
    ext_win = ext_ds.sel(time=(ext_ds.time.dt.year <= ext_year_end))

    ceds_ds = ceds_win = ceds_var = None
    if ceds_nc is not None:
        try:
            ceds_ds  = xr.open_dataset(ceds_nc, use_cftime=True, engine="netcdf4")
            ceds_var = next(v for v in ceds_ds.data_vars if not v.endswith("_bnds"))
            ceds_win = ceds_ds.sel(time=(ceds_ds.time.dt.year >= ft_year_start))
        except Exception as e:
            log.warning(f"[J]   CEDS load failed for {gas} {file_type}: {e}")
            ceds_ds = None

    # ── Batch-extract all locations in one compute() per segment ──────────────
    # Vectorised pointwise indexing: each (lat[i], lon[i]) is one grid cell.
    # This replaces N separate compute() calls (one per location) with a single
    # compute() whose result is a (time, loc) array sliced per location in numpy.
    loc_names = list(locations.keys())
    lats_idx = xr.DataArray(
        [locations[n][0] for n in loc_names], dims="loc", coords={"loc": loc_names}
    )
    lons_idx = xr.DataArray(
        [locations[n][1] for n in loc_names], dims="loc", coords={"loc": loc_names}
    )

    def _batch_extract(win_da: xr.DataArray) -> xr.DataArray:
        pts = win_da.sel(lat=lats_idx, lon=lons_idx, method="nearest")
        for dim in ("sector", "level"):
            if dim in pts.dims:
                pts = pts.sum(dim=dim)
        return pts.compute()  # single I/O pass for all locations

    ft_pts  = _batch_extract(ft_win[ft_var])
    ext_pts = _batch_extract(ext_win[ext_var])
    ceds_pts = _batch_extract(ceds_win[ceds_var]) if (ceds_ds is not None and ceds_win is not None) else None

    # Time axes computed once, shared across locations (pure cftime → float)
    ft_times   = [t.year + (t.month - 0.5) / 12 for t in ft_pts.time.values]
    ext_times  = [t.year + (t.month - 0.5) / 12 for t in ext_pts.time.values]
    ceds_times = [t.year + (t.month - 0.5) / 12 for t in ceds_pts.time.values] if ceds_pts is not None else []

    metric_rows: list[dict] = []

    for place in loc_names:
        lat, lon = locations[place]
        out_plot = plots_dir / f"gridpoint_{place}_{gas}_{file_type}.png"
        if skip_existing and out_plot.exists():
            log.info(f"[J]     Skip {place} (plot exists)")
            continue

        try:
            # Pure numpy slicing — no I/O
            ft_vals   = ft_pts.sel(loc=place).values.astype(float)
            ext_vals  = ext_pts.sel(loc=place).values.astype(float)
            ceds_vals = ceds_pts.sel(loc=place).values.astype(float) if ceds_pts is not None else None

            ft_last   = float(ft_vals[-1])  if len(ft_vals)  else np.nan
            ext_first = float(ext_vals[0])  if len(ext_vals) else np.nan
            abs_jump  = abs(ext_first - ft_last)
            rel_jump_pct = abs_jump / abs(ft_last) * 100 if ft_last != 0 else np.nan

            metric_rows.append({
                "place": place, "lat": lat, "lon": lon,
                "gas": gas, "file_type": file_type,
                "ft_last_value_kg_m2_s": ft_last,
                "ext_first_value_kg_m2_s": ext_first,
                "abs_jump_kg_m2_s": abs_jump,
                "rel_jump_pct": rel_jump_pct,
            })

            # ── Plot ──────────────────────────────────────────────────────────
            fig, ax = plt.subplots(figsize=(13, 4))
            ax.set_title(
                f"{place} ({lat:.2f}°N, {lon:.2f}°E) — {gas} | {file_type}",
                fontsize=11, fontweight="bold",
            )
            ax.set_ylabel("Emissions (kg m⁻² s⁻¹, summed sectors)")
            ax.set_xlabel("Year")

            if ceds_vals is not None:
                ax.plot(ceds_times, ceds_vals,
                        color=SEGMENT_COLORS["ceds"], linewidth=1.0,
                        label=SEGMENT_LABELS["ceds"], alpha=0.7, zorder=2)

            ax.plot(ft_times,  ft_vals,
                    color=SEGMENT_COLORS["fast_track"], linewidth=1.3,
                    label=SEGMENT_LABELS["fast_track"], zorder=3)
            ax.plot(ext_times, ext_vals,
                    color=SEGMENT_COLORS["extension"], linewidth=1.3,
                    label=SEGMENT_LABELS["extension"], zorder=3)

            # Dashed bridge across the gap
            if len(ft_vals) and len(ext_vals):
                ax.plot(
                    [ft_times[-1], ext_times[0]],
                    [ft_last, ext_first],
                    color="gray", linestyle="--", linewidth=1.1, alpha=0.65,
                    label="Interpolated gap", zorder=2,
                )

            ext_start_frac = fade_anchor_year + 5 + 0.5 / 12  # 2105-01
            ax.axvline(fade_anchor_year + 12 / 12, color="black",
                       linestyle="--", linewidth=0.9,
                       label=f"Anchor ({fade_anchor_year})")
            ax.axvline(ext_start_frac, color="dimgray",
                       linestyle=":", linewidth=0.9)
            ax.axvspan(fade_anchor_year + 1, ext_start_frac - 0.1,
                       color="lightyellow", alpha=0.55, zorder=0,
                       label=f"Gap ({fade_anchor_year + 1}–{fade_anchor_year + 4})")

            if fade_anchor_year + window_after >= fade_convergence_year:
                ax.axvline(fade_convergence_year + 0.5 / 12, color="orange",
                           linestyle=":", linewidth=0.9, alpha=0.8,
                           label=f"Fade end ({fade_convergence_year})")

            if not np.isnan(rel_jump_pct) and len(ext_vals):
                flag = " ⚠" if rel_jump_pct > 20 else " ✓"
                ax.annotate(
                    f"|jump|: {rel_jump_pct:.0f}%{flag}",
                    xy=(ext_start_frac, ext_first),
                    xytext=(ext_start_frac + 4, ext_first),
                    fontsize=8, va="center",
                    arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
                )

            ax.legend(fontsize=8, loc="best")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            fig.savefig(out_plot, dpi=150, bbox_inches="tight")
            plt.close(fig)
            log.info(
                f"[J]     {place} {gas} {file_type}: "
                f"|jump|={rel_jump_pct:.0f}% → {out_plot.name}"
            )

        except Exception as e:
            log.error(f"[J]   Error for {place} {gas} {file_type}: {e}")
            import traceback
            log.debug(traceback.format_exc())

    ft_ds.close()
    ext_ds.close()
    if ceds_ds is not None:
        ceds_ds.close()

    return {"metric_rows": metric_rows}


def check_gridpoint_continuity(
    ext_folder: Path,
    ft_folder: Path,
    qc_output_path: Path,
    settings,
    locations: dict[str, tuple[float, float]],
    species_filter: list[str] | None = None,
    fade_anchor_year: int = 2100,
    fade_convergence_year: int = 2150,
    window_before: int = _GRIDPOINT_WINDOW_BEFORE,
    window_after: int = _GRIDPOINT_WINDOW_AFTER,
    skip_existing: bool = True,
    max_workers: int = 4,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """
    Module J: Grid-point monthly timeseries at representative locations.

    For each (gas, file_type) pair found in both the extension and fast-track
    folders, extracts the monthly timeseries at the nearest grid cell to each
    location (summing over all sectors/levels) and plots the junction window
    [anchor − window_before, anchor + window_after].

    Reports the absolute and relative jump from the last fast-track month
    (anchor-12-31) to the first extension month (anchor+5-01).  This jump
    naturally includes 4+ years of continued trend; interpret alongside the
    global-continuity plots.

    Outputs
    -------
    qc_output/junctions/gridpoint_junction_metrics.csv
    qc_output/junctions/plots/gridpoint_{place}_{gas}_{type}.png
    """
    log = logger or logging.getLogger(__name__)
    out_dir = qc_output_path / "junctions"
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "gridpoint_junction_metrics.csv"

    if skip_existing and out_csv.exists():
        log.info(f"[J] Skipping (already exists): {out_csv}")
        return pd.read_csv(out_csv)

    log.info("[J] Running gridpoint continuity check...")
    t0 = time.time()

    ext_files = sorted(ext_folder.glob("*.nc"))
    if species_filter:
        ext_files = [
            f for f in ext_files
            if any(f.name.startswith(g + "-") for g in species_filter)
        ]

    work_items: list[dict] = []
    for ext_nc in ext_files:
        try:
            gas, file_type = _parse_nc_filename(ext_nc.name)
        except ValueError:
            continue

        ft_nc = _find_nc_file(ft_folder, gas, file_type)
        if ft_nc is None:
            log.warning(f"[J] No fast-track file for {gas} {file_type} — skipping")
            continue

        ceds_folder = _ceds_folder_for_type(settings, file_type)
        ceds_nc = _find_nc_file(ceds_folder, gas, file_type) if ceds_folder else None

        work_items.append({
            "ext_nc": ext_nc,
            "ft_nc": ft_nc,
            "ceds_nc": ceds_nc,
            "locations": locations,
            "plots_dir": plots_dir,
            "skip_existing": skip_existing,
            "fade_anchor_year": fade_anchor_year,
            "fade_convergence_year": fade_convergence_year,
            "window_before": window_before,
            "window_after": window_after,
            "logger_name": log.name,
        })

    metric_rows: list[dict] = []

    n_workers = min(max_workers, len(work_items)) if work_items else 1
    use_parallel = n_workers > 1 and not _in_jupyter()

    if use_parallel:
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as ex:
            for result in ex.map(_worker_gridpoint_species, work_items):
                metric_rows.extend(result["metric_rows"])
    else:
        for item in work_items:
            result = _worker_gridpoint_species(item)
            metric_rows.extend(result["metric_rows"])

    df = pd.DataFrame(metric_rows)
    df.to_csv(out_csv, index=False)
    log.info(f"[J] Done ({time.time() - t0:.1f}s) → {out_csv}")
    return df


# ── Module K: Raw-2100 diagnostic summary ─────────────────────────────────────

def _find_diagnostics_2100_dir(nc_folder: Path, ext_folder: Path, settings) -> Path | None:
    """Locate the diagnostics_2100 folder written by the main workflow, if present."""
    candidates = [
        nc_folder / "diagnostics_2100",
        ext_folder / "diagnostics_2100",
    ]
    try:
        candidates.append(Path(settings.out_path) / settings.version / "diagnostics_2100")
    except Exception:
        pass
    for c in candidates:
        if c.is_dir() and any(c.glob("*_2100diagnostic_*.nc")):
            return c
    return None


def summarize_2100_diagnostic(
    nc_folder: Path,
    ext_folder: Path,
    qc_output_path: Path,
    settings,
    cell_area: xr.DataArray | None,
    logger: logging.Logger,
) -> dict | None:
    """
    Module K: read the per-file 2100 diagnostic netCDFs and write two CSVs into
    `junctions/2100_diagnostic/`:
      - `2100_diagnostic_summary.csv`     — RAW extension-2100 vs fast-track agreement
      - `2100_correction_zero_check.csv`  — CORRECTED extension-2100 vs fast-track (~0 check)
    Returns {"raw": DataFrame, "corrected": DataFrame|None}, or None if no files found.
    """
    t0 = time.time()
    diag_dir = _find_diagnostics_2100_dir(nc_folder, ext_folder, settings)
    if diag_dir is None:
        logger.warning(
            "[K] No diagnostics_2100/*_2100diagnostic_*.nc found — "
            "run the main workflow with run_2100_alignment_diagnostic=True first. Skipping."
        )
        return None

    logger.info(f"[K] Summarising 2100 diagnostics from: {diag_dir}")
    rows: list[dict] = []        # RAW (pre-correction) extension vs fast-track
    corr_rows: list[dict] = []   # CORRECTED (post-offset) extension vs fast-track (expect ~0)
    for nc in sorted(diag_dir.glob("*_2100diagnostic_*.nc")):
        try:
            gas, file_type = _parse_nc_filename(nc.name)
        except ValueError:
            gas, file_type = nc.stem, ""
        try:
            ds = xr.open_dataset(nc, use_cftime=True)
            ext = ds["ext_2100_raw"]
            ft = ds["ft_2100"]
            diff = ds["diff_ft_minus_ext"]

            flat_e = ext.values.ravel()
            flat_f = ft.values.ravel()
            fin = np.isfinite(flat_e) & np.isfinite(flat_f)
            spatial_corr = (
                float(np.corrcoef(flat_e[fin], flat_f[fin])[0, 1])
                if fin.sum() > 1 else float("nan")
            )
            if cell_area is not None:
                ext_tot = float((ext * cell_area).sum().values)
                ft_tot = float((ft * cell_area).sum().values)
            else:
                ext_tot = float(ext.sum().values)
                ft_tot = float(ft.sum().values)
            aw_rel_diff_pct = (
                100.0 * (ext_tot - ft_tot) / ft_tot if abs(ft_tot) > 0 else float("nan")
            )
            rows.append({
                "gas": gas,
                "file_type": file_type,
                "ext_2100_total": ext_tot,
                "ft_2100_total": ft_tot,
                "area_weighted_rel_diff_pct": aw_rel_diff_pct,
                "spatial_corr": spatial_corr,
                "max_abs_diff": float(np.nanmax(np.abs(diff.values))),
                "file": nc.name,
            })

            # Post-correction zero-check: the corrected 2100 must equal fast-track
            # (the alignment forces it). Confirms this BEFORE 2100 is dropped.
            if "ext_2100_corrected" in ds:
                if "diff_ft_minus_ext_corrected" in ds:
                    diff_corr = ds["diff_ft_minus_ext_corrected"].values
                else:
                    diff_corr = ft.values - ds["ext_2100_corrected"].values
                max_ft = float(np.nanmax(np.abs(ft.values)))
                max_abs_diff_corr = float(np.nanmax(np.abs(diff_corr)))
                # "zero" relative to the field magnitude (float32 grids → ~1e-6 rel noise).
                tol = max(1e-12, 1e-5 * max_ft)
                if cell_area is not None:
                    corr_tot = float((ds["ext_2100_corrected"] * cell_area).sum().values)
                else:
                    corr_tot = float(ds["ext_2100_corrected"].sum().values)
                corr_aw_rel_diff_pct = (
                    100.0 * (corr_tot - ft_tot) / ft_tot if abs(ft_tot) > 0 else float("nan")
                )
                corr_rows.append({
                    "gas": gas,
                    "file_type": file_type,
                    "max_abs_diff_corrected": max_abs_diff_corr,
                    "tolerance": tol,
                    "area_weighted_rel_diff_pct_corrected": corr_aw_rel_diff_pct,
                    "zero_ok": bool(max_abs_diff_corr <= tol),
                    "file": nc.name,
                })
            ds.close()
        except Exception as e:
            logger.warning(f"[K] Could not read {nc.name}: {e}")

    if not rows:
        logger.warning("[K] No diagnostic files could be summarised.")
        return None

    out_dir = qc_output_path / "junctions" / "2100_diagnostic"
    out_dir.mkdir(parents=True, exist_ok=True)

    # RAW summary
    df = pd.DataFrame(rows).sort_values(
        "area_weighted_rel_diff_pct", key=lambda s: s.abs(), ascending=False
    )
    raw_csv = out_dir / "2100_diagnostic_summary.csv"
    df.to_csv(raw_csv, index=False)

    logger.info("[K] RAW extension-2100 vs fast-track-2100 (largest |rel diff| first):")
    logger.info(f"    {'gas':<8}{'type':<14}{'rel_diff%':>11}{'corr':>8}{'max|diff|':>13}")
    for _, r in df.iterrows():
        logger.info(
            f"    {str(r['gas']):<8}{str(r['file_type']):<14}"
            f"{r['area_weighted_rel_diff_pct']:>10.3f}%{r['spatial_corr']:>8.4f}"
            f"{r['max_abs_diff']:>13.3e}"
        )

    # CORRECTED zero-check
    cdf = None
    if corr_rows:
        cdf = pd.DataFrame(corr_rows).sort_values(
            "max_abs_diff_corrected", ascending=False
        )
        check_csv = out_dir / "2100_correction_zero_check.csv"
        cdf.to_csv(check_csv, index=False)
        n_fail = int((~cdf["zero_ok"]).sum())
        logger.info("[K] CORRECTED extension-2100 vs fast-track-2100 (expect ~0):")
        logger.info(f"    {'gas':<8}{'type':<14}{'max|diff|':>13}{'tol':>13}{'ok':>5}")
        for _, r in cdf.iterrows():
            logger.info(
                f"    {str(r['gas']):<8}{str(r['file_type']):<14}"
                f"{r['max_abs_diff_corrected']:>13.3e}{r['tolerance']:>13.3e}"
                f"{('Y' if r['zero_ok'] else 'N'):>5}"
            )
        if n_fail:
            logger.warning(
                f"[K] ⚠️  {n_fail} file(s) FAILED the corrected-2100 zero-check "
                f"(diff exceeds tolerance) → see {check_csv}"
            )
        else:
            logger.info(f"[K] ✅ all {len(cdf)} file(s) pass the corrected-2100 zero-check → {check_csv}")
    else:
        logger.warning(
            "[K] No 'ext_2100_corrected' variable in the diagnostic netCDFs — "
            "re-run the main workflow with the updated diagnostic to enable the zero-check."
        )

    logger.info(f"[K] Done ({time.time() - t0:.1f}s) → {out_dir}")
    return {"raw": df, "corrected": cdf}


# ── Orchestrator ──────────────────────────────────────────────────────────────

def run_junction_qc(
    ext_folder: Path,
    settings_file: str,
    gridding_version: str,
    marker_to_run: str,
    location_fasttrack_gridded: Path,
    version_esgf: str = "1-1-1",
    fade_anchor_year: int = 2100,
    fade_convergence_year: int = 2150,
    run_global_continuity: bool = True,
    run_gridpoint_continuity: bool = True,
    run_2100_diagnostic_summary: bool = False,
    species_filter: list[str] | None = None,
    file_type_filter: list[str] | None = None,
    locations: dict[str, tuple[float, float]] | None = None,
    skip_existing: bool = True,
    max_workers: int = 4,
    iam_reference_csv: Path | None = None,
    iam_scenario: str | None = None,
    here: Path | None = None,
) -> dict:
    """
    Run the full junction QC suite.

    Parameters
    ----------
    ext_folder : Path
        Directory that holds the extension gridded NC files.
    settings_file : str
        YAML settings filename relative to `here`.
    gridding_version : str
        Version string passed to Settings (e.g. 'vl-ext_1-1-1').
    marker_to_run : str
        Marker abbreviation (e.g. 'vl').
    location_fasttrack_gridded : Path
        Top-level directory for fast-track gridded data.  The actual files
        live at ``location_fasttrack_gridded / '{marker}_{version_esgf}'``.
    max_workers : int
        Number of parallel worker processes for species-level parallelism.
        Automatically falls back to sequential when running in Jupyter.
    """
    settings_path = Path(settings_file)
    if not settings_path.is_absolute():
        if here is None:
            here = _find_here()
        settings_path = here / settings_file

    settings = Settings.from_config(
        version=gridding_version,
        local_config_path=settings_path,
    )

    qc_output_path = ext_folder / "qc_output"
    qc_output_path.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log = _setup_logging(qc_output_path, timestamp)

    ft_folder = location_fasttrack_gridded / f"{marker_to_run}_{version_esgf}"

    # NC files may live in a gridding_version subfolder (e.g. results/h-ext_1-1-1/h-ext_1-1-1/)
    nc_folder = ext_folder / gridding_version
    if not (nc_folder.is_dir() and any(nc_folder.glob("*.nc"))):
        nc_folder = ext_folder

    log.info("=== Junction QC ===")
    log.info(f"  Extension folder  : {ext_folder}")
    log.info(f"  NC files folder   : {nc_folder}")
    log.info(f"  Fast-track folder : {ft_folder}")
    log.info(f"  Gridding version  : {gridding_version}")
    log.info(f"  Marker            : {marker_to_run}")
    log.info(f"  Anchor year       : {fade_anchor_year}")
    log.info(f"  Convergence year  : {fade_convergence_year}")
    log.info(f"  Species filter    : {species_filter}")
    log.info(f"  Max workers       : {max_workers}")

    cell_area = None
    try:
        cell_area = _load_cell_area(settings)
        log.info("  Cell area loaded OK")
    except Exception as e:
        log.warning(f"  Could not load cell area: {e} — Module I will be skipped")

    if locations is None:
        locations = {
            "Laxenburg":    (48.0689,  16.3555),
            "South_Sudan":  ( 6.8770,  31.3070),
            "Beijing":      (39.9042, 116.4074),
            "Sao_Paulo":    (-23.5505, -46.6333),
            "Rural_Amazon": (-3.4653,  -62.2159),
        }

    results: dict = {}
    t_total = time.time()

    if run_global_continuity:
        if cell_area is not None:
            results["global_continuity"] = check_global_continuity(
                ext_folder=nc_folder,
                ft_folder=ft_folder,
                qc_output_path=qc_output_path,
                settings=settings,
                cell_area=cell_area,
                species_filter=species_filter,
                file_type_filter=file_type_filter,
                fade_anchor_year=fade_anchor_year,
                fade_convergence_year=fade_convergence_year,
                skip_existing=skip_existing,
                max_workers=max_workers,
                iam_reference_csv=iam_reference_csv,
                iam_scenario=iam_scenario,
                logger=log,
            )
        else:
            log.warning("[I] Skipped — cell area unavailable")

    if run_gridpoint_continuity:
        results["gridpoint_continuity"] = check_gridpoint_continuity(
            ext_folder=nc_folder,
            ft_folder=ft_folder,
            qc_output_path=qc_output_path,
            settings=settings,
            locations=locations,
            species_filter=species_filter,
            fade_anchor_year=fade_anchor_year,
            fade_convergence_year=fade_convergence_year,
            skip_existing=skip_existing,
            max_workers=max_workers,
            logger=log,
        )

    if run_2100_diagnostic_summary:
        results["diagnostic_2100_summary"] = summarize_2100_diagnostic(
            nc_folder=nc_folder,
            ext_folder=ext_folder,
            qc_output_path=qc_output_path,
            settings=settings,
            cell_area=cell_area,
            logger=log,
        )

    log.info(f"=== Junction QC complete in {time.time() - t_total:.1f}s ===")
    log.info(f"  Outputs: {qc_output_path / 'junctions'}")
    return results


# %% [markdown]
# ## Run

# %%
if __name__ == "__main__":
    _ext_folder = (
        Path(FOLDER_WITH_EXTENSION_DATA)
        if FOLDER_WITH_EXTENSION_DATA
        else _find_here().parent.parent / "results" / GRIDDING_VERSION
    )

    run_junction_qc(
        ext_folder=_ext_folder,
        settings_file=SETTINGS_FILE,
        gridding_version=GRIDDING_VERSION,
        marker_to_run=marker_to_run,
        location_fasttrack_gridded=LOCATION_FASTTRACK_GRIDDED,
        version_esgf=VERSION_ESGF,
        fade_anchor_year=FADE_ANCHOR_YEAR,
        fade_convergence_year=FADE_CONVERGENCE_YEAR,
        run_global_continuity=run_global_continuity,
        run_gridpoint_continuity=run_gridpoint_continuity,
        run_2100_diagnostic_summary=run_2100_diagnostic_summary,
        species_filter=species_filter,
        file_type_filter=FILE_TYPE_FILTER,
        locations=LOCATIONS,
        skip_existing=skip_existing,
        max_workers=max_workers,
        iam_reference_csv=Path(IAM_CSV) if IAM_CSV else None,
        iam_scenario=IAM_SCENARIO if IAM_SCENARIO else None,
        here=_SCRIPT_DIR,
    )

# %%
