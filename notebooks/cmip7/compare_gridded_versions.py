# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---
from __future__ import annotations

# %% [markdown]
# # Compare Two Gridded Scenario Versions
#
# Loads all `.nc` files from two user-configured folders and compares them
# file-by-file.  For each matched `(gas, file_type)` pair the script checks
# whether the data arrays are **exactly** equal (including NaN positions) and
# produces a unified diff of the NetCDF attributes.
#
# Results are written file-by-file as processing proceeds, then combined into:
# - `data_comparison.csv`  — one row per `(gas, file_type)` pair
# - `metadata_diff.txt`    — all attribute diffs concatenated
#
# When ``per_gridpoint=True`` the script additionally writes per-variable
# NetCDF files with spatial absolute- and relative-difference fields under
# ``per_file/{gas}_{file_type}_diffs.nc``.
#
# ## Usage
# Run via the driver script:
#
#     python scripts/cmip7/driver_compare_gridded_versions.py
#
# or import `run_comparison()` directly into a notebook.

# %% [markdown]
# ## Parameters

# %% editable=true slideshow={"slide_type": ""} tags=["parameters"]
FOLDER_A: str = ""        # path to version-A gridded data folder
FOLDER_B: str = ""        # path to version-B gridded data folder
LABEL_A: str = "version_a"
LABEL_B: str = "version_b"
OUTPUT_DIR: str = ""      # defaults to FOLDER_A / "qc_output_v2" / "version_comparison"

species_filter: list[str] | None = None   # e.g. ["BC", "CO2"] for a quick test
skip_existing: bool = False
per_gridpoint: bool = False   # write per-gridpoint diff fields to NetCDF

# %% [markdown]
# ## Imports

# %%
import datetime
import difflib
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from concordia.cmip7.utils import (
    SECTOR_DICT_ANTHRO_DEFAULT,
    SECTOR_DICT_ANTHRO_CO2_SCENARIO,
    SECTOR_DICT_OPENBURNING_DEFAULT,
)

# %% [markdown]
# ## Constants

# %%

def _resolve_sector_dict(gas: str, file_type: str) -> dict[int, str] | None:
    """Return the appropriate sector-index-to-name mapping, or None."""
    if file_type == "anthro":
        if gas == "CO2":
            return SECTOR_DICT_ANTHRO_CO2_SCENARIO
        return SECTOR_DICT_ANTHRO_DEFAULT
    if file_type == "openburning":
        return SECTOR_DICT_OPENBURNING_DEFAULT
    return None  # AIR-anthro or unknown — no sector dict


def _format_location(row: dict, prefix: str) -> str:
    """Format a location from flattened row keys into a human-readable string."""
    parts = []
    for suffix in ("var", "time", "sector", "lat", "lon"):
        val = row.get(f"{prefix}_{suffix}", "")
        if val != "" and val is not None:
            parts.append(f"{suffix}={val}")
    return ", ".join(parts)


# Columns written to data_comparison.csv.  Each row represents one matched
# (gas, file_type) pair; ordering here must match what compare_one_pair() returns.
DATA_CSV_COLS = [
    "filename_a",
    "filename_b",
    "gas",
    "file_type",
    "exists_in_a",
    "exists_in_b",
    "data_identical",
    "variables_compared",
    "variables_differing",
    "max_abs_diff",
    "max_abs_diff_var",
    "max_abs_diff_time",
    "max_abs_diff_sector",
    "max_abs_diff_lat",
    "max_abs_diff_lon",
    "max_rel_diff",
    "max_rel_diff_var",
    "max_rel_diff_time",
    "max_rel_diff_sector",
    "max_rel_diff_lat",
    "max_rel_diff_lon",
    "mean_abs_diff",
    "mean_rel_diff",
    "coords_identical",
    "shape_a",
    "shape_b",
    "load_error_a",
    "load_error_b",
    "note",
]

# %% [markdown]
# ## Helper utilities

# %%

def _find_here() -> Path:
    """Robustly find the notebooks/cmip7 directory."""
    try:
        here = Path(__file__).parent
        if here != Path("."):
            return here
    except NameError:
        pass
    current = Path.cwd()
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists() and (parent / "src" / "concordia").exists():
            return parent / "notebooks" / "cmip7"
    return Path.cwd()


def setup_logging(output_dir: Path, timestamp: str) -> logging.Logger:
    """Set up a logger that writes to both a timestamped file and stdout."""
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"compare_log_{timestamp}.txt"

    logger = logging.getLogger(f"compare_gridded_{timestamp}")
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


# %% [markdown]
# ## File discovery and matching

# %%

def scan_folder(
    folder: Path,
    species_filter: list[str] | None = None,
    logger: logging.Logger | None = None,
) -> dict[tuple[str, str], Path]:
    """
    Scan a folder for *.nc files and return a mapping (gas, file_type) -> Path.

    Files whose names cannot be parsed are logged and skipped.
    If two files resolve to the same (gas, file_type) key, the last one
    (alphabetically) wins and a warning is logged.

    Parameters
    ----------
    folder
        Directory to scan.
    species_filter
        If not None, restrict to gases in this list (e.g. ["BC", "CH4"]).
    logger
        Optional logger; warnings are printed to stdout if None.

    Returns
    -------
    dict mapping (gas, file_type) -> absolute Path
    """
    _warn = logger.warning if logger else print
    _info = logger.info if logger else print

    result: dict[tuple[str, str], Path] = {}
    nc_files = sorted(folder.glob("*.nc"))
    _info(f"  Scanning {folder}: found {len(nc_files)} .nc file(s)")

    for path in nc_files:
        try:
            gas, file_type = _parse_nc_filename(path.name)
        except ValueError as exc:
            _warn(f"    [SKIP] Cannot parse filename: {path.name} — {exc}")
            continue

        if species_filter is not None and gas not in species_filter:
            continue

        key = (gas, file_type)
        if key in result:
            _warn(
                f"    [WARN] Duplicate (gas, file_type)={key}; "
                f"keeping {path.name}, replacing {result[key].name}"
            )
        result[key] = path

    return result


def build_match_table(
    map_a: dict[tuple[str, str], Path],
    map_b: dict[tuple[str, str], Path],
) -> list[dict]:
    """
    Combine two folder scan results into a list of (gas, file_type) entries.

    For each unique (gas, file_type) across both folders, produces a dict:
        {
          "gas": str,
          "file_type": str,
          "path_a": Path | None,
          "path_b": Path | None,
          "exists_in_a": bool,
          "exists_in_b": bool,
        }

    Entries are sorted by (gas, file_type) for deterministic output.
    """
    all_keys = sorted(set(map_a) | set(map_b))
    rows = []
    for gas, file_type in all_keys:
        rows.append(
            {
                "gas": gas,
                "file_type": file_type,
                "path_a": map_a.get((gas, file_type)),
                "path_b": map_b.get((gas, file_type)),
                "exists_in_a": (gas, file_type) in map_a,
                "exists_in_b": (gas, file_type) in map_b,
            }
        )
    return rows


# %% [markdown]
# ## Per-file comparison

# %%

def _argmax_location(arr: np.ndarray, da: xr.DataArray, var: str) -> dict:
    """Map the flat argmax of *arr* back to coordinate values of *da*.

    Returns a dict ``{"variable": var, dim_name: coord_value, ...}`` for
    every dimension in *da*.  Non-numeric coordinate values (e.g. cftime
    dates) are converted to strings so the result is always JSON-safe.
    """
    flat_idx = int(np.nanargmax(arr))
    nd_idx = np.unravel_index(flat_idx, arr.shape)
    loc: dict = {"variable": var}
    for i, dim in enumerate(da.dims):
        coord_val = da.coords[dim].values[nd_idx[i]]
        # Convert numpy scalar / cftime to a Python-native type.
        if hasattr(coord_val, "item"):
            coord_val = coord_val.item()
        if not isinstance(coord_val, (int, float, str)):
            coord_val = str(coord_val)
        loc[dim] = coord_val
    return loc


def compare_data(ds_a: xr.Dataset, ds_b: xr.Dataset, per_gridpoint: bool = False) -> dict:
    """
    Compare data variables between two xarray Datasets.

    For each variable present in both datasets, uses
    ``np.array_equal(a, b, equal_nan=True)`` for exact bitwise equality
    (NaN == NaN).  For differing variables computes ``max_abs_diff``
    (largest element-wise |a - b|) and ``max_rel_diff`` (largest
    |a - b| / max(|a|, |b|), with 0 where both sides are zero).
    Both metrics are the maximum observed across *all* differing variables.

    ``data_identical`` is ``True`` only when data values, shapes, *and*
    coordinates are all identical.

    Parameters
    ----------
    ds_a, ds_b
        Datasets to compare.
    per_gridpoint
        When True, include ``"abs_diff_fields"`` and ``"rel_diff_fields"``
        dicts (keyed by variable name) of xarray DataArrays in the return
        value.

    Returns
    -------
    dict with keys:
        data_identical (bool)
        variables_compared (int)
        variables_differing (int)
        max_abs_diff (float | np.nan)
        max_abs_diff_loc (dict | None)
        max_rel_diff (float | np.nan)
        max_rel_diff_loc (dict | None)
        mean_abs_diff (float | np.nan)
        mean_rel_diff (float | np.nan)
        coords_identical (bool)
        shape_a (str)
        shape_b (str)
        note (str)
        abs_diff_fields (dict, only when per_gridpoint=True)
        rel_diff_fields (dict, only when per_gridpoint=True)
    """
    notes = []

    # ── Step 1: partition variables into shared / exclusive sets ─────────────
    vars_a = set(ds_a.data_vars)
    vars_b = set(ds_b.data_vars)
    shared = vars_a & vars_b          # variables present in both → element-compared below

    only_in_a = vars_a - vars_b       # exclusive variables count as "differing" immediately
    only_in_b = vars_b - vars_a
    if only_in_a:
        notes.append(f"vars only in A: {sorted(only_in_a)}")
    if only_in_b:
        notes.append(f"vars only in B: {sorted(only_in_b)}")

    # ── Step 2: coordinate and shape comparison ───────────────────────────────
    try:
        coords_identical = ds_a.coords.to_dataset().equals(ds_b.coords.to_dataset())
    except Exception as exc:
        coords_identical = False
        notes.append(f"coord comparison error: {exc}")

    shape_a = str(dict(ds_a.sizes))
    shape_b = str(dict(ds_b.sizes))
    if shape_a != shape_b:
        notes.append(f"shape mismatch: {shape_a} vs {shape_b}")

    # ── Step 3: initialise diff accumulators ──────────────────────────────────
    # Prime the differing count with variables exclusive to one side; those
    # variables are already logged in `notes` and cannot be element-compared.
    variables_differing = len(only_in_a) + len(only_in_b)
    max_abs_diff = np.nan   # largest |a - b| seen across all differing variables
    max_rel_diff = np.nan   # largest |a - b| / max(|a|, |b|) across all differing variables
    max_abs_diff_loc: dict | None = None  # coordinate location of max_abs_diff
    max_rel_diff_loc: dict | None = None  # coordinate location of max_rel_diff
    # Weighted-mean accumulators (sum / count across all differing variables).
    _mean_abs_sum = 0.0
    _mean_abs_count = 0
    _mean_rel_sum = 0.0
    _mean_rel_count = 0
    differing_vars = []
    if per_gridpoint:
        abs_diff_fields: dict[str, xr.DataArray] = {}
        rel_diff_fields: dict[str, xr.DataArray] = {}

    # ── Step 4: element-wise comparison for each shared variable ──────────────
    # Chunk large arrays by time to avoid OOM (mirrors check_gridded_scenario_qc.py Module B).
    # At float64 (8 bytes/element) 200 M elements ≈ 1.6 GB; holding two such arrays
    # plus intermediate diff buffers in parallel can push a single comparison above 5 GB.
    _SIZE_LIMIT = 200_000_000  # 200 M elements ≈ 1.6 GB float64

    # Sort variable names so per-variable reporting order is deterministic across runs.
    for var in sorted(shared):
        a_da = ds_a[var]
        b_da = ds_b[var]

        if a_da.shape != b_da.shape:
            notes.append(f"{var}: shape mismatch {a_da.shape} vs {b_da.shape}")
            variables_differing += 1
            differing_vars.append(var)
            continue

        if a_da.size > _SIZE_LIMIT and "time" in a_da.dims:
            # ── 4a. Chunked path: iterate over time slices to stay within RAM ─
            n_time = a_da.sizes["time"]
            spatial_size = a_da.size // n_time           # elements in a single time step
            time_chunk = max(1, _SIZE_LIMIT // spatial_size)  # how many time steps fit in the budget
            equal = True
            local_max = np.nan   # per-variable running max abs diff (updated chunk by chunk)
            local_rel = np.nan   # per-variable running max rel diff (updated chunk by chunk)
            local_abs_loc: dict | None = None  # location of local_max within this variable
            local_rel_loc: dict | None = None  # location of local_rel within this variable
            # Mean accumulators for this variable (across chunks).
            _var_abs_sum = 0.0
            _var_abs_cnt = 0
            _var_rel_sum = 0.0
            _var_rel_cnt = 0
            if per_gridpoint:
                _abs_chunks: list[np.ndarray] = []
                _rel_chunks: list[np.ndarray] = []
            try:
                for t0 in range(0, n_time, time_chunk):
                    a_c = a_da.isel(time=slice(t0, t0 + time_chunk)).values
                    b_c = b_da.isel(time=slice(t0, t0 + time_chunk)).values
                    try:
                        # equal_nan=True makes NaN == NaN; keyword added in NumPy 1.19.
                        chunk_equal = np.array_equal(a_c, b_c, equal_nan=True)
                    except TypeError:
                        # Fallback for non-float dtypes (e.g. int, str) or older NumPy.
                        chunk_equal = np.array_equal(a_c, b_c)
                    if not chunk_equal:
                        equal = False
                        try:
                            # Cast to float so integer/bool arrays support arithmetic.
                            a_f = a_c.astype(float)
                            b_f = b_c.astype(float)
                            abs_diff = np.abs(a_f - b_f)  # element-wise |a - b|
                            d = float(np.nanmax(abs_diff))
                            # Running maximum across time chunks for this variable.
                            if np.isnan(local_max) or d > local_max:
                                local_max = d
                                # Find argmax within this chunk, map to full-array coords.
                                _chunk_da = a_da.isel(time=slice(t0, t0 + time_chunk))
                                local_abs_loc = _argmax_location(abs_diff, _chunk_da, var)
                            # max(|a|, |b|) as reference magnitude: avoids division by a
                            # near-zero value from one side while the other is large.
                            denom = np.maximum(np.abs(a_f), np.abs(b_f))
                            with np.errstate(divide="ignore", invalid="ignore"):
                                # Where both sides are zero, define relative diff as 0 %.
                                rel = np.where(denom > 0, abs_diff / denom, 0.0)
                            r = float(np.nanmax(rel))
                            # Running maximum across time chunks for this variable.
                            if np.isnan(local_rel) or r > local_rel:
                                local_rel = r
                                _chunk_da = a_da.isel(time=slice(t0, t0 + time_chunk))
                                local_rel_loc = _argmax_location(rel, _chunk_da, var)
                            # Mean accumulators (include zeros, exclude NaN).
                            _nvalid = int(np.sum(~np.isnan(abs_diff)))
                            _var_abs_sum += float(np.nansum(abs_diff))
                            _var_abs_cnt += _nvalid
                            _var_rel_sum += float(np.nansum(rel))
                            _var_rel_cnt += _nvalid
                            if per_gridpoint:
                                _abs_chunks.append(abs_diff)
                                _rel_chunks.append(rel)
                        except Exception:
                            if per_gridpoint:
                                _abs_chunks.append(np.full_like(a_c, np.nan, dtype=float))
                                _rel_chunks.append(np.full_like(a_c, np.nan, dtype=float))
                    else:
                        # Chunk is identical — still contributes zeros to the mean.
                        _nvalid = int(np.sum(~np.isnan(a_c.astype(float)))) if a_c.size else 0
                        _var_abs_cnt += _nvalid
                        _var_rel_cnt += _nvalid
                        if per_gridpoint:
                            _abs_chunks.append(np.zeros_like(a_c, dtype=float))
                            _rel_chunks.append(np.zeros_like(a_c, dtype=float))
                # Concatenate per-gridpoint chunks along the time axis
                if per_gridpoint and _abs_chunks and not equal:
                    _time_ax = list(a_da.dims).index("time")
                    abs_diff_fields[var] = xr.DataArray(
                        np.concatenate(_abs_chunks, axis=_time_ax),
                        dims=a_da.dims, coords=a_da.coords,
                    )
                    rel_diff_fields[var] = xr.DataArray(
                        np.concatenate(_rel_chunks, axis=_time_ax),
                        dims=a_da.dims, coords=a_da.coords,
                    )
            except Exception as exc:
                notes.append(f"load error for {var}: {exc}")
                variables_differing += 1
                differing_vars.append(var)
                continue
        else:
            # ── 4b. Fast path: load entire array at once ──────────────────────
            try:
                a = a_da.values
                b = b_da.values
            except Exception as exc:
                notes.append(f"load error for {var}: {exc}")
                variables_differing += 1
                differing_vars.append(var)
                continue
            try:
                equal = np.array_equal(a, b, equal_nan=True)
            except TypeError:
                # non-float dtype (e.g. int, datetime, str) — isnan not applicable
                equal = np.array_equal(a, b)
            local_abs_loc = None
            local_rel_loc = None
            _var_abs_sum = 0.0
            _var_abs_cnt = 0
            _var_rel_sum = 0.0
            _var_rel_cnt = 0
            if not equal:
                try:
                    # Cast to float so integer/bool arrays support arithmetic.
                    a_f = a.astype(float)
                    b_f = b.astype(float)
                    abs_diff = np.abs(a_f - b_f)          # element-wise |a - b|
                    local_max = float(np.nanmax(abs_diff))
                    local_abs_loc = _argmax_location(abs_diff, a_da, var)
                    # max(|a|, |b|) as reference magnitude: avoids division by a
                    # near-zero value from one side while the other is large.
                    denom = np.maximum(np.abs(a_f), np.abs(b_f))
                    with np.errstate(divide="ignore", invalid="ignore"):
                        # Where both sides are zero, define relative diff as 0 %.
                        rel = np.where(denom > 0, abs_diff / denom, 0.0)
                    local_rel = float(np.nanmax(rel))
                    local_rel_loc = _argmax_location(rel, a_da, var)
                    # Mean accumulators (include zeros, exclude NaN).
                    _nvalid = int(np.sum(~np.isnan(abs_diff)))
                    _var_abs_sum = float(np.nansum(abs_diff))
                    _var_abs_cnt = _nvalid
                    _var_rel_sum = float(np.nansum(rel))
                    _var_rel_cnt = _nvalid
                    if per_gridpoint:
                        abs_diff_fields[var] = xr.DataArray(
                            abs_diff, dims=a_da.dims, coords=a_da.coords,
                        )
                        rel_diff_fields[var] = xr.DataArray(
                            rel, dims=a_da.dims, coords=a_da.coords,
                        )
                except Exception:
                    local_max = np.nan  # non-numeric vars (e.g. string coords)
                    local_rel = np.nan
            else:
                local_max = np.nan
                local_rel = np.nan

        if not equal:
            variables_differing += 1
            differing_vars.append(var)
            # Aggregate per-variable maxima into the cross-variable running maximum.
            if not np.isnan(local_max) and (np.isnan(max_abs_diff) or local_max > max_abs_diff):
                max_abs_diff = local_max
                max_abs_diff_loc = local_abs_loc
            if not np.isnan(local_rel) and (np.isnan(max_rel_diff) or local_rel > max_rel_diff):
                max_rel_diff = local_rel
                max_rel_diff_loc = local_rel_loc
            # Accumulate into cross-variable mean.
            _mean_abs_sum += _var_abs_sum
            _mean_abs_count += _var_abs_cnt
            _mean_rel_sum += _var_rel_sum
            _mean_rel_count += _var_rel_cnt

    # ── Step 5: final verdict ─────────────────────────────────────────────────
    # All three conditions must hold: no variable-level differences, matching
    # shapes, AND identical coordinates.
    data_identical = (variables_differing == 0) and (shape_a == shape_b) and coords_identical

    if differing_vars:
        notes.append(f"differing vars: {differing_vars}")

    mean_abs_diff = (_mean_abs_sum / _mean_abs_count) if _mean_abs_count > 0 else np.nan
    mean_rel_diff = (_mean_rel_sum / _mean_rel_count) if _mean_rel_count > 0 else np.nan

    result = {
        "data_identical": data_identical,
        "variables_compared": len(shared),
        "variables_differing": variables_differing,
        "max_abs_diff": max_abs_diff,
        "max_abs_diff_loc": max_abs_diff_loc,
        "max_rel_diff": max_rel_diff,
        "max_rel_diff_loc": max_rel_diff_loc,
        "mean_abs_diff": mean_abs_diff,
        "mean_rel_diff": mean_rel_diff,
        "coords_identical": coords_identical,
        "shape_a": shape_a,
        "shape_b": shape_b,
        "note": "; ".join(notes),
    }
    if per_gridpoint:
        result["abs_diff_fields"] = abs_diff_fields
        result["rel_diff_fields"] = rel_diff_fields
    return result


def _attrs_to_lines(ds: xr.Dataset) -> list[str]:
    """
    Render dataset global and variable attrs as a sorted list of text lines.

    Keys within each section are sorted so the output is stable across runs
    and the resulting diff is not polluted by irrelevant ordering changes.
    Global attributes are emitted first, followed by per-variable attributes
    in alphabetical variable order.  The output can be fed directly to
    ``difflib.unified_diff``.
    """
    lines = ["=== global attrs ==="]
    for k in sorted(ds.attrs):
        lines.append(f"  {k}: {ds.attrs[k]!r}")
    for var in sorted(ds.data_vars):
        lines.append(f"=== variable: {var} ===")
        for k in sorted(ds[var].attrs):
            lines.append(f"  {k}: {ds[var].attrs[k]!r}")
    return lines


def compare_metadata(
    ds_a: xr.Dataset,
    ds_b: xr.Dataset,
    label_a: str = "version_a",
    label_b: str = "version_b",
) -> str:
    """
    Produce a unified diff of global and per-variable attributes.

    Returns the diff string, or "" if attributes are identical.
    """
    lines_a = _attrs_to_lines(ds_a)
    lines_b = _attrs_to_lines(ds_b)
    diff = list(
        difflib.unified_diff(
            lines_a,
            lines_b,
            fromfile=label_a,
            tofile=label_b,
            lineterm="",
        )
    )
    return "\n".join(diff)


def compare_one_pair(
    match: dict,
    per_file_dir: Path,
    label_a: str = "version_a",
    label_b: str = "version_b",
    logger: logging.Logger | None = None,
    per_gridpoint: bool = False,
) -> dict:
    """
    Load and compare one matched (gas, file_type) pair.

    Writes per-file results immediately:
        per_file/{gas}_{file_type}_data.txt   — "IDENTICAL" or diff summary
        per_file/{gas}_{file_type}_meta.diff  — unified diff of attrs

    Parameters
    ----------
    match
        One entry from build_match_table().
    per_file_dir
        Directory where per-file outputs are written.
    label_a, label_b
        Human-readable labels for diff headers.
    logger
        Optional logger.
    per_gridpoint
        When True, write per-gridpoint diff fields to a NetCDF file.

    Returns
    -------
    dict with all DATA_CSV_COLS values for this pair (one CSV row).
    """
    _info = logger.info if logger else print
    _warn = logger.warning if logger else print
    _err = logger.error if logger else print

    gas = match["gas"]
    file_type = match["file_type"]
    path_a: Path | None = match["path_a"]
    path_b: Path | None = match["path_b"]
    exists_in_a: bool = match["exists_in_a"]
    exists_in_b: bool = match["exists_in_b"]

    # Sanitise (gas, file_type) into a string safe for use as a filename stem;
    # forward slashes (legal in gas names such as "VOC") are replaced with hyphens.
    safe_key = f"{gas}_{file_type}".replace("/", "-")
    data_txt = per_file_dir / f"{safe_key}_data.txt"
    meta_diff = per_file_dir / f"{safe_key}_meta.diff"

    row: dict = {
        "filename_a": path_a.name if path_a else "",
        "filename_b": path_b.name if path_b else "",
        "gas": gas,
        "file_type": file_type,
        "exists_in_a": exists_in_a,
        "exists_in_b": exists_in_b,
        "data_identical": False,
        "variables_compared": 0,
        "variables_differing": 0,
        "max_abs_diff": np.nan,
        "max_abs_diff_var": "",
        "max_abs_diff_time": "",
        "max_abs_diff_sector": "",
        "max_abs_diff_lat": "",
        "max_abs_diff_lon": "",
        "max_rel_diff": np.nan,
        "max_rel_diff_var": "",
        "max_rel_diff_time": "",
        "max_rel_diff_sector": "",
        "max_rel_diff_lat": "",
        "max_rel_diff_lon": "",
        "mean_abs_diff": np.nan,
        "mean_rel_diff": np.nan,
        "coords_identical": False,
        "shape_a": "",
        "shape_b": "",
        "load_error_a": "",
        "load_error_b": "",
        "note": "",
    }

    # ── Handle missing files ──────────────────────────────────────────────────
    if not exists_in_a or not exists_in_b:
        missing = []
        if not exists_in_a:
            missing.append(label_a)
        if not exists_in_b:
            missing.append(label_b)
        msg = f"MISSING in: {', '.join(missing)}"
        _warn(f"  {gas}/{file_type}: {msg}")
        data_txt.write_text(msg + "\n", encoding="utf-8")
        meta_diff.write_text("(file missing in one version — no metadata diff)\n", encoding="utf-8")
        row["note"] = msg
        return row

    # ── Load both datasets ────────────────────────────────────────────────────
    ds_a = ds_b = None
    try:
        ds_a = xr.open_dataset(path_a, engine="netcdf4")
    except Exception as exc:
        row["load_error_a"] = str(exc)
        _err(f"  {gas}/{file_type}: cannot open {label_a}: {exc}")

    try:
        ds_b = xr.open_dataset(path_b, engine="netcdf4")
    except Exception as exc:
        row["load_error_b"] = str(exc)
        _err(f"  {gas}/{file_type}: cannot open {label_b}: {exc}")

    if ds_a is None or ds_b is None:
        note = "load error (see load_error_a/b)"
        row["note"] = note
        data_txt.write_text(f"LOAD ERROR\n{note}\n", encoding="utf-8")
        meta_diff.write_text("(load error — no metadata diff)\n", encoding="utf-8")
        if ds_a is not None:
            ds_a.close()
        if ds_b is not None:
            ds_b.close()
        return row

    # ── Compare data ──────────────────────────────────────────────────────────
    # Returns a dict with data_identical, variables_differing, max_abs_diff,
    # max_rel_diff, location dicts, mean diffs, coords_identical, shape_a/b,
    # and note (see compare_data()).
    data_result = compare_data(ds_a, ds_b, per_gridpoint=per_gridpoint)
    # Extract per-gridpoint fields before merging into CSV row
    _abs_fields = data_result.pop("abs_diff_fields", None)
    _rel_fields = data_result.pop("rel_diff_fields", None)
    # Extract location dicts (not CSV columns themselves).
    _abs_loc = data_result.pop("max_abs_diff_loc", None)
    _rel_loc = data_result.pop("max_rel_diff_loc", None)

    row.update(data_result)  # merge all compare_data keys into the CSV row dict

    # ── Flatten location dicts into CSV row ───────────────────────────────────
    _sector_dict = _resolve_sector_dict(gas, file_type)
    for prefix, loc in [("max_abs_diff", _abs_loc), ("max_rel_diff", _rel_loc)]:
        if loc is None:
            continue
        row[f"{prefix}_var"] = loc.get("variable", "")
        row[f"{prefix}_time"] = loc.get("time", "")
        row[f"{prefix}_lat"] = loc.get("lat", "")
        row[f"{prefix}_lon"] = loc.get("lon", "")
        # Sector or level — resolve integer sector to name when possible.
        raw_sector = loc.get("sector", loc.get("level", ""))
        if isinstance(raw_sector, (int, float)) and _sector_dict:
            row[f"{prefix}_sector"] = _sector_dict.get(int(raw_sector), raw_sector)
        elif "level" in loc:
            row[f"{prefix}_sector"] = loc["level"]
        else:
            row[f"{prefix}_sector"] = raw_sector

    if logger and data_result.get("variables_differing", 0) > 0:
        logger.debug(f"  {gas}/{file_type}: differing vars detail: {data_result['note']}")

    # ── Write per-gridpoint diff fields ───────────────────────────────────────
    if _abs_fields:
        diff_vars: dict[str, xr.DataArray] = {}
        for v, da in _abs_fields.items():
            diff_vars[f"{v}_abs_diff"] = da
        for v, da in _rel_fields.items():
            diff_vars[f"{v}_rel_diff"] = da
        diff_ds = xr.Dataset(diff_vars)
        diff_nc = per_file_dir / f"{safe_key}_diffs.nc"
        diff_ds.to_netcdf(diff_nc)
        if logger:
            logger.info(f"  {gas}/{file_type}: wrote per-gridpoint diffs to {diff_nc.name}")

    # ── Compare metadata ──────────────────────────────────────────────────────
    # Produces a unified diff of all global and per-variable NetCDF attributes;
    # returns "" if attributes are identical (written to meta.diff regardless).
    meta_str = compare_metadata(ds_a, ds_b, label_a=label_a, label_b=label_b)

    ds_a.close()
    ds_b.close()

    # ── Write per-file results ────────────────────────────────────────────────
    status = "IDENTICAL" if data_result["data_identical"] else "DIFFERS"
    data_lines = [status]
    if data_result["note"]:
        data_lines.append(data_result["note"])
    if not data_result["data_identical"]:
        data_lines.append(
            f"variables_differing: {data_result['variables_differing']} "
            f"/ {data_result['variables_compared']} compared"
        )
        if not np.isnan(data_result["max_abs_diff"]):
            data_lines.append(f"max_abs_diff: {data_result['max_abs_diff']:.6g}")
            if _abs_loc:
                data_lines.append(
                    f"  at: {_format_location(row, 'max_abs_diff')}"
                )
        if not np.isnan(data_result["max_rel_diff"]):
            data_lines.append(f"max_rel_diff: {data_result['max_rel_diff']:.6g}")
            if _rel_loc:
                data_lines.append(
                    f"  at: {_format_location(row, 'max_rel_diff')}"
                )
        if not np.isnan(data_result["mean_abs_diff"]):
            data_lines.append(f"mean_abs_diff: {data_result['mean_abs_diff']:.6g}")
        if not np.isnan(data_result["mean_rel_diff"]):
            data_lines.append(f"mean_rel_diff: {data_result['mean_rel_diff']:.6g}")
    data_txt.write_text("\n".join(data_lines) + "\n", encoding="utf-8")

    meta_diff.write_text(
        meta_str if meta_str else "(no attribute differences)\n",
        encoding="utf-8",
    )

    _info(f"  {gas}/{file_type}: {status}")
    return row


# %% [markdown]
# ## Orchestrator

# %%

def run_comparison(
    folder_a: Path,
    folder_b: Path,
    output_dir: Path,
    label_a: str = "version_a",
    label_b: str = "version_b",
    species_filter: list[str] | None = None,
    skip_existing: bool = False,
    per_gridpoint: bool = False,
) -> dict[str, Path]:
    """
    Compare all matched *.nc files between two gridded scenario version folders.

    Writes:
        {output_dir}/logs/compare_log_{timestamp}.txt
        {output_dir}/per_file/{gas}_{file_type}_data.txt
        {output_dir}/per_file/{gas}_{file_type}_meta.diff
        {output_dir}/per_file/{gas}_{file_type}_diffs.nc  (when per_gridpoint=True)
        {output_dir}/data_comparison.csv
        {output_dir}/metadata_diff.txt

    When ``per_gridpoint=True`` the script additionally writes per-variable
    NetCDF files with spatial absolute- and relative-difference fields.

    Parameters
    ----------
    folder_a
        Path to version-A gridded data folder.
    folder_b
        Path to version-B gridded data folder.
    output_dir
        Root directory for all outputs.  Created if absent.
    label_a, label_b
        Human-readable labels used in log messages and diff headers.
    species_filter
        If not None, only compare files for these gas names.
    skip_existing
        If True, skip a pair whose per-file _data.txt already exists.
    per_gridpoint
        When True, write per-variable per-gridpoint diff fields as NetCDF.

    Returns
    -------
    dict with keys:
        "data_comparison_csv"  -> Path
        "metadata_diff_txt"    -> Path
        "per_file_dir"         -> Path
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    per_file_dir = output_dir / "per_file"
    per_file_dir.mkdir(parents=True, exist_ok=True)

    log = setup_logging(output_dir, timestamp)
    log.info(f"Version comparison started: {label_a}  vs  {label_b}")
    log.info(f"  Folder A ({label_a}): {folder_a}")
    log.info(f"  Folder B ({label_b}): {folder_b}")
    log.info(f"  Output dir  : {output_dir}")
    log.info(f"  species_filter: {species_filter}")

    # ── Discover files ────────────────────────────────────────────────────────
    map_a = scan_folder(folder_a, species_filter, log)
    map_b = scan_folder(folder_b, species_filter, log)
    matches = build_match_table(map_a, map_b)

    only_in_a = sum(1 for m in matches if m["exists_in_a"] and not m["exists_in_b"])
    only_in_b = sum(1 for m in matches if m["exists_in_b"] and not m["exists_in_a"])
    both = sum(1 for m in matches if m["exists_in_a"] and m["exists_in_b"])
    log.info(
        f"  Matched pairs: {both} in both, "
        f"{only_in_a} only in {label_a}, {only_in_b} only in {label_b}"
    )

    # ── Compare file by file ──────────────────────────────────────────────────
    rows = []
    for match in matches:
        gas = match["gas"]
        file_type = match["file_type"]
        # Same sanitisation as compare_one_pair() so paths stay consistent.
        safe_key = f"{gas}_{file_type}".replace("/", "-")

        if skip_existing and (per_file_dir / f"{safe_key}_data.txt").exists():
            # Attempt to reload the row from an existing data_comparison.csv
            existing_csv = output_dir / "data_comparison.csv"
            reloaded = False
            if existing_csv.exists():
                try:
                    existing_df = pd.read_csv(existing_csv)
                    match_rows = existing_df[
                        (existing_df["gas"] == gas)
                        & (existing_df["file_type"] == file_type)
                    ]
                    if not match_rows.empty:
                        rows.append(match_rows.iloc[0].to_dict())
                        log.info(
                            f"  {gas}/{file_type}: [SKIP] reloaded existing row from CSV"
                        )
                        reloaded = True
                except Exception as exc:
                    log.warning(
                        f"  {gas}/{file_type}: [SKIP] could not read existing CSV: {exc}"
                    )
            if not reloaded:
                log.warning(
                    f"  {gas}/{file_type}: [SKIP] per-file result exists but no CSV row found"
                )
            continue

        # Load, compare, and write per-file outputs for this (gas, file_type) pair.
        # Returns one dict whose keys match DATA_CSV_COLS, ready to append to `rows`.
        row = compare_one_pair(
            match,
            per_file_dir,
            label_a=label_a,
            label_b=label_b,
            logger=log,
            per_gridpoint=per_gridpoint,
        )
        rows.append(row)

    # ── Write combined data CSV ───────────────────────────────────────────────
    csv_path = output_dir / "data_comparison.csv"
    df = pd.DataFrame(rows, columns=DATA_CSV_COLS)
    df.to_csv(csv_path, index=False)
    log.info(f"  Written: {csv_path}")

    # ── Write combined metadata diff ──────────────────────────────────────────
    meta_txt_path = output_dir / "metadata_diff.txt"
    meta_sections = []
    for match in matches:
        gas = match["gas"]
        file_type = match["file_type"]
        safe_key = f"{gas}_{file_type}".replace("/", "-")
        diff_file = per_file_dir / f"{safe_key}_meta.diff"
        header = f"{'='*60}\n{gas} / {file_type}\n{'='*60}"
        if diff_file.exists():
            body = diff_file.read_text(encoding="utf-8").rstrip()
        else:
            body = "(no per-file diff found)"
        meta_sections.append(f"{header}\n{body}")
    meta_txt_path.write_text("\n\n".join(meta_sections) + "\n", encoding="utf-8")
    log.info(f"  Written: {meta_txt_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    n_identical = df["data_identical"].sum() if not df.empty else 0
    n_differs = (~df["data_identical"]).sum() if not df.empty else 0
    n_errors = ((df["load_error_a"] != "") | (df["load_error_b"] != "")).sum() if not df.empty else 0
    log.info(
        f"\n  Summary: {len(df)} pairs compared — "
        f"{n_identical} identical, {n_differs} differ, {n_errors} load error(s)"
    )

    return {
        "data_comparison_csv": csv_path,
        "metadata_diff_txt": meta_txt_path,
        "per_file_dir": per_file_dir,
    }


# %% [markdown]
# ## Run (standalone)

# %%
if __name__ == "__main__":
    _HERE = _find_here()

    if not FOLDER_A or not FOLDER_B:
        raise ValueError(
            "Set FOLDER_A and FOLDER_B at the top of this file (or use the driver script)."
        )

    _output_dir = (
        Path(OUTPUT_DIR)
        if OUTPUT_DIR
        else Path(FOLDER_A) / "qc_output_v2" / "version_comparison"
    )

    run_comparison(
        folder_a=Path(FOLDER_A),
        folder_b=Path(FOLDER_B),
        output_dir=_output_dir,
        label_a=LABEL_A,
        label_b=LABEL_B,
        species_filter=species_filter,
        skip_existing=skip_existing,
        per_gridpoint=per_gridpoint,
    )
