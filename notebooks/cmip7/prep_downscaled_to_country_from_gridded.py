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
# -*- coding: utf-8 -*-
"""
rederive_downscaled_from_gridded.py
------------------------------------
Re-derives country-level downscaled emissions from the final gridded netCDF
files of a completed concordia/CMIP7 run.

This is necessary when post-downscaling or post-gridding fixes have been
applied (e.g. zeroing small negatives, spatial harmonization corrections),
meaning the `downscaled-only-*.csv` file no longer reflects what actually
ended up in the grids. Re-aggregating from the netCDFs gives the true
country-level values to use as the harmonization anchor (hist) for
extension runs.

Usage
-----
Call `rederive_downscaled_from_gridded()` with the path to the original
gridded netCDF files and the indexraster. The result is a DataFrame in
the same format as `downscaled-only-*.csv`, which can be saved and used
as `scenario_hist` in the extensions notebook.

-----
- Global sectors (International Shipping, Aircraft) are handled separately:
  they are read from the netCDF global sum attribute or summed over the full
  grid (all cells), since they are not distributed by country.
- Units are converted from flux (kg m-2 s-1) back to mass/year (the same
  unit as the downscaled CSV) using cell area and seconds-per-year.
- The sector dimension in the netCDF is mapped back to sector names using
  the sector_bnds coordinate or a fallback ordering.
"""

from pathlib import Path
import re

from tqdm import tqdm
import numpy as np
import pandas as pd
import xarray as xr
from ptolemy.raster import IndexRaster

from concordia.cmip7.utils import SECTOR_RENAMES, SECTOR_ORDERING_GAS, SECTOR_ORDERING_DEFAULT, SECTOR_DICT_ANTHRO_DEFAULT, SECTOR_DICT_ANTHRO_CO2_SCENARIO, SECTOR_DICT_OPENBURNING_DEFAULT, SECTOR_DICT_OPENBURNING_DEFAULT_FLIPPED, SECTOR_DICT_ANTHRO_CO2_SCENARIO_FLIPPED
from concordia import (
    RegionMapping,
    VariableDefinitions,
)

# Authoritative gridded→annual aggregation lives in
# `concordia.cmip7.utils_plotting.ds_to_annual_emissions_total_faster` and is
# used by `workflow_cmip7-fast-track.py` and `check_gridded_scenario_qc.py`
# (Modules D and D2). It cannot be called directly here because it sums over
# lat/lon, whereas we need to keep the spatial dims so we can do a country-level
# groupby. The aggregation logic below mirrors that function line-for-line
# (calendar-aware days_in_month, kg→Mt conversion, integer-coord sector map)
# and only diverges at the final spatial reduction.

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Sectors that are global (not distributed by country in the grid)
GLOBAL_SECTORS = {"International Shipping", "Aircraft"}

# Regex to parse the variable name and type from a netCDF filename.
# Expected format: "{gas}-em-{type}_{FILE_NAME_ENDING}.nc"
# e.g. "SO2-em-anthro_input4MIPs_emissions_ScenarioMIP_IIASA-IAMC-vl-1-1-1_gn_210501-250012.nc"
_FILENAME_RE = re.compile(r"^(?P<gas>[^_]+)-em-(?P<type>[^_]+)_.+\.nc$")

SECTOR_RENAMES_INVERSE = {v: k for k, v in SECTOR_RENAMES.items()}

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _parse_filename(path: Path) -> tuple[str, str] | None:
    """Return (gas, emission_type) parsed from a gridded netCDF filename, or None."""
    m = _FILENAME_RE.match(path.name)
    if m is None:
        return None
    return m.group("gas"), m.group("type")


# Maps file_type → {sector_int: sector_name}. Mirrors `SECTOR_INT_TO_NAME`
# in check_gridded_scenario_qc.py: the CO2 superset (0–9) is used for all
# anthro files so that BECCS / Other Capture and Removal are mapped on CO2
# files; non-CO2 anthro files only have indices 0–7, so the extra keys are
# never looked up.
SECTOR_INT_TO_NAME = {
    "anthro": SECTOR_DICT_ANTHRO_CO2_SCENARIO,
    "openburning": SECTOR_DICT_OPENBURNING_DEFAULT,
    "air": {0: "Aircraft"},
}


def _aggregate_one_file(
    nc_path: Path,
    indexraster: IndexRaster,
    cell_area: xr.DataArray,
    target_years: list[int],
) -> pd.DataFrame | None:

    parsed = _parse_filename(nc_path)
    if parsed is None:
        print(f"  [SKIP] Could not parse filename: {nc_path.name}")
        return None

    gas, em_type = parsed

    em_type_lower = em_type.lower()
    if "air" in em_type_lower:
        sector_type = "air"
    elif "openburning" in em_type_lower:
        sector_type = "openburning"
    elif "anthro" in em_type_lower:
        sector_type = "anthro"
    else:
        sector_type = "unknown"

    int_to_name = SECTOR_INT_TO_NAME.get(sector_type, {})

    ds = xr.open_dataset(nc_path)

    var_name = f"{gas.replace('-', '_')}_em_{em_type.replace('-', '_')}"
    if var_name not in ds:
        candidates = [v for v in ds.data_vars if gas.lower() in v.lower()]
        if not candidates:
            print(f"  [SKIP] Variable '{var_name}' not found in {nc_path.name}.")
            ds.close()
            return None
        var_name = candidates[0]
        print(f"  [WARN] Using '{var_name}' as fallback variable name in {nc_path.name}")

    da = ds[var_name]

    has_sector_dim = "sector" in da.dims
    has_level_dim = "level" in da.dims

    # ── Canonical aggregation (matches ds_to_annual_emissions_total_faster) ──
    # Calendar-aware seconds per month — xarray reads days_in_month from the
    # file's calendar attribute, so this works for standard/Gregorian, noleap,
    # and 360_day files alike. Hard-coding [31, 28, 31, ...] would silently
    # produce wrong totals on standard-calendar files (Feb=29 in leap years).
    seconds_per_month = da.time.dt.days_in_month * 24 * 3600

    # Chunk to keep memory bounded on large CO2 anthro files (mirrors the
    # chunking in ds_to_annual_emissions_total_faster).
    if has_sector_dim:
        n_sectors = da.sizes["sector"]
        time_chunk = 6 if n_sectors > 15 else (12 if n_sectors > 8 else 24)
        da = da.chunk({"time": time_chunk, "sector": -1})
    elif has_level_dim:
        da = da.chunk({"time": 12, "level": -1})
    else:
        da = da.chunk({"time": 24})

    # kg/m²/s × s/month × m² → kg/month per cell (per sector, per level if AIR)
    kg_per_month = cell_area * seconds_per_month * da
    if has_level_dim:
        kg_per_month = kg_per_month.sum(dim="level")

    # Sum across months within each year → kg/year per cell (per sector).
    # Spatial dims are kept so we can do the country groupby afterwards.
    kg_per_year = kg_per_month.groupby("time.year").sum().compute()

    # Restrict to requested target years (silently drops anything outside).
    target_years_set = set(int(y) for y in target_years)
    available_years = [
        int(y) for y in kg_per_year.year.values if int(y) in target_years_set
    ]
    if not available_years:
        print(f"  [SKIP] None of the target years found in {nc_path.name}")
        ds.close()
        return None
    kg_per_year = kg_per_year.sel(year=available_years)

    # Unit conversion: N2O kept as kt N2O/yr to match the downscaled CSV and
    # extensions_full_emissions_timeseries; all other gases as Mt {gas}/yr.
    if gas == "N2O":
        unit_out = "kt N2O/yr"
        scale = 1e-6  # kg/yr → kt/yr
    else:
        unit_out = f"Mt {gas}/yr"
        scale = 1e-9  # kg/yr → Mt/yr

    # Sector iteration uses the actual integer sector coord (matches the
    # NetCDF), not enumerate(range(n_sectors)). This is robust to sparse or
    # reordered sector indices.
    if has_sector_dim:
        sector_iterator = [
            (int(s_int), int_to_name.get(int(s_int), f"unknown_{int(s_int)}"))
            for s_int in kg_per_year.sector.values
        ]
    else:
        sector_iterator = [(None, int_to_name.get(0, em_type))]

    idx_to_country = {i + 1: c for i, c in enumerate(indexraster.index)}

    records = []
    for yr in tqdm(available_years, desc=f"{nc_path.name}", leave=False):
        kg_year = kg_per_year.sel(year=yr)

        for sector_int, sector_name in sector_iterator:
            if sector_int is not None:
                kg_sector = kg_year.sel(sector=sector_int)
            else:
                kg_sector = kg_year

            # Global sectors (Aircraft, International Shipping): no country
            # breakdown — sum all cells and report under "World".
            if sector_name in GLOBAL_SECTORS or sector_type == "air":
                total_kg_yr = float(kg_sector.sum().values)
                records.append({
                    "country": "World", "gas": gas.replace("-", "_"),
                    "sector": "Aircraft" if sector_type == "air" else sector_name,
                    "unit": unit_out, "year": yr,
                    "value": total_kg_yr * scale,
                })
                continue

            # Country-level: groupby the indexraster's country indicator.
            country_totals = kg_sector.groupby(indexraster.indicator).sum()
            groupby_dim = country_totals.dims[0]
            for pos, idx_val in enumerate(country_totals.coords[groupby_dim].values):
                if int(idx_val) == 0:
                    continue
                country_code = idx_to_country.get(int(idx_val))
                if country_code is None:
                    continue
                val_kg_yr = float(country_totals.isel({groupby_dim: pos}).values)
                records.append({
                    "country": country_code, "gas": gas.replace("-", "_"),
                    "sector": sector_name, "unit": unit_out, "year": yr,
                    "value": val_kg_yr * scale,
                })

    ds.close()

    if not records:
        return None

    # Sector ordering for sorting the output (CDR sectors appended for CO2).
    if sector_type == "air":
        sector_order = ["Aircraft"]
    elif sector_type in {"anthro", "openburning"}:
        ordering_key = (
            "CO2_em_anthro"
            if gas == "CO2" and sector_type == "anthro"
            else f"em_{sector_type}"
        )
        sector_order = SECTOR_ORDERING_DEFAULT.get(ordering_key)
        if gas == "CO2" and sector_type == "anthro":
            cdr_sectors = [
                "BECCS", "Direct Air Capture", "Enhanced Weathering",
                "Ocean", "Biochar", "Soil Carbon Management", "Other CDR",
            ]
            if sector_order:
                sector_order = list(sector_order) + [s for s in cdr_sectors if s not in sector_order]
            else:
                sector_order = cdr_sectors
    else:
        sector_order = None

    df_long = pd.DataFrame(records)
    df_wide = (
        df_long
        .pivot_table(index=["country", "gas", "sector", "unit"], columns="year", values="value")
        .reset_index()
    )
    df_wide.columns.name = None

    if sector_order is not None:
        df_wide["sector"] = pd.Categorical(df_wide["sector"], categories=sector_order, ordered=True)
    df_wide = df_wide.sort_values(["country", "gas", "sector"]).set_index(["country", "gas", "sector", "unit"])

    return df_wide


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def rederive_downscaled_from_gridded(
    gridded_path: Path,
    indexraster: IndexRaster,
    cell_area: xr.DataArray,
    target_years: list[int] | range = range(2022, 2100),  # changed
    file_pattern: str = "*.nc",
    exclude_patterns: list[str] | None = None,
    species_filter: list[str] | str | None = None,
    save_path: Path | None = None,
) -> pd.DataFrame:
    """
    Re-derive country-level emissions at `target_year` by aggregating all
    gridded netCDF files in `gridded_path` back to country level.

    Parameters
    ----------
    gridded_path : Path
        Directory containing the gridded netCDF files (one run version).
    indexraster : IndexRaster
        Maps each grid cell to a country ISO3 code. Should be the same
        indexraster used in the original gridding run.
    cell_area : xr.DataArray
        Grid cell areas in m². Should match the grid of the netCDF files.
    target_year : int, default 2100
        The year to extract from each file.
    file_pattern : str, default "*.nc"
        Glob pattern to find netCDF files in `gridded_path`.
    unit_out : str, default "Mt/yr"
        Unit label written into the output MultiIndex. Does not affect
        numeric values (conversion is always kg/yr → Mt/yr).
    exclude_patterns : list of str, optional
        Substrings; files whose names contain any of these are skipped.
        Useful for skipping e.g. areacella files.
        Defaults to ["areacella", "fx_"].
    species_filter : list of str or str, optional
        If given, only process files whose gas matches one of these (e.g.
        "NH3" or ["NH3", "CO2"]). Match is on the gas prefix before "-em-",
        same convention as check_gridded_scenario_qc.py.
    save_path : Path, optional
        If given, save the result as a CSV at this path.

    Returns
    -------
    pd.DataFrame
        MultiIndex (country, gas, sector, unit), single column `target_year`.
        Same format as `downscaled-only-*.csv` but for a single year.
    """
    if exclude_patterns is None:
        exclude_patterns = ["areacella", "fx_"]

    nc_files = sorted(gridded_path.glob(file_pattern))
    nc_files = [
        f for f in nc_files
        if not any(pat in f.name for pat in exclude_patterns)
    ]

    # Filter by species (gas prefix before "-em-"); accept a single string for
    # convenience, e.g. species_filter="NH3".
    if species_filter is not None:
        if isinstance(species_filter, str):
            species_filter = [species_filter]
        nc_files = [
            f for f in nc_files
            if any(f.name.startswith(g + "-em-") for g in species_filter)
        ]

    if not nc_files:
        filt_msg = f", species_filter={species_filter}" if species_filter else ""
        raise FileNotFoundError(
            f"No netCDF files found in {gridded_path} matching '{file_pattern}' "
            f"(after exclusions{filt_msg})."
        )

    print(f"Found {len(nc_files)} netCDF files to process in {gridded_path}")

    all_dfs = []
    
    for nc_path in tqdm(nc_files, desc="Processing files"):
        print(f"  Processing: {nc_path.name}")
        df = _aggregate_one_file(
            nc_path=nc_path,
            indexraster=indexraster,
            cell_area=cell_area,
            target_years=list(target_years),
        )

        if df is not None:
            all_dfs.append(df)

    if not all_dfs:
        raise ValueError("No data could be extracted from any netCDF file.")

    result = pd.concat(all_dfs, axis=0)

    # Sort index for consistency with downscaled CSV format
    result = result.sort_index()

    # Warn about duplicates (e.g. same gas/sector in both anthro and openburning files)
    if result.index.duplicated().any():
        n_dups = result.index.duplicated().sum()
        print(f"\n⚠️  {n_dups} duplicate (country, gas, sector, unit) entries found.")
        print("   This may indicate a gas appears in both anthro and openburning files.")
        print("   Keeping first occurrence. Please verify manually.")
        result = result[~result.index.duplicated(keep="first")]

    print(f"\n✅ Done. Aggregated {len(result)} country×gas×sector entries.")

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        # Reset index to match the format of downscaled-only CSV
        result.reset_index().to_csv(save_path, index=False)
        print(f"   Saved to: {save_path}")

    return result


# %%
# %%
# # %load_ext autoreload
# # %autoreload 2

# %% [markdown]
# # Rederive country-level downscaled emissions from gridded netCDF files
# Aggregates the final gridded netCDFs from a completed run back to country level,
# producing a CSV in the same format as `downscaled-only-*.csv` but reflecting
# all post-downscaling and post-gridding fixes that were applied.

# %% [markdown]
# ## Settings

# %% tags=["parameters"]
SETTINGS_FILE: str = "config_cmip7_v0-4-0-EXT.yaml"
VERSION_ESGF: str = "1-1-1"          # version of the original run to rederive from
marker_to_run: str = "vl"
TARGET_YEAR: int = 2100
# Optional: limit processing to a single gas (e.g. "NH3") or a list of gases
# (["NH3", "CO2"]). Set to None to process all species in the folder.
SPECIES_FILTER: list[str] | str | None = None
SPECIES_FILTER: list[str] | str | None = ["NH3", "CO2"]

GRIDDING_VERSION: str = f"{marker_to_run}-ext_{VERSION_ESGF}"   # folder name of original run

# %% [markdown]
# ## Imports

# %%
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from ptolemy.raster import IndexRaster

from concordia.settings import Settings
from concordia.cmip7 import utils_EXT as cmip7_utils
from concordia.cmip7.CONSTANTS import return_marker_information

# %% [markdown]
# ## Resolve paths and settings

# %%
try:
    HERE = Path(__file__).parent
    if str(HERE) == "." or HERE == Path("."):
        raise NameError
except NameError:
    current_path = Path.cwd()
    concordia_root = None
    for parent in [current_path] + list(current_path.parents):
        if (parent / "pyproject.toml").exists() and (parent / "src" / "concordia").exists():
            concordia_root = parent
            break
    if concordia_root is None:
        raise RuntimeError("Could not find concordia repository root")
    HERE = concordia_root / "notebooks" / "cmip7"

_, MODEL_SELECTION, SCENARIO_SELECTION, _ = return_marker_information(
    v=SETTINGS_FILE,
    m=marker_to_run
)

settings = Settings.from_config(
    version=GRIDDING_VERSION,
    local_config_path=Path(HERE, SETTINGS_FILE)
)

# `out_path` in the YAML is "../../results" (a path relative to the notebook).
# Settings keeps it relative, so it would otherwise resolve against the kernel
# CWD — which is wrong whenever Jupyter wasn't started from notebooks/cmip7/.
# Anchor it to HERE so downstream paths work regardless of CWD.
if not Path(settings.out_path).is_absolute():
    settings.out_path = (HERE / settings.out_path).resolve()

original_gridded_path = settings.out_path / GRIDDING_VERSION
print(f"Reading gridded netCDFs from: {original_gridded_path}")

# %% [markdown]
# ## Load indexraster and cell area

# %%
indexraster = IndexRaster.from_netcdf(
    settings.gridding_path / "ssp_comb_indexraster_splitsudankosovopalestine.nc",
    chunks={},
).compute()


regionmappings = {}

for m, kwargs in settings.regionmappings.items():
    regionmapping = RegionMapping.from_regiondef(**kwargs)
    if settings.country_combinations:  # only aggregate if not empty
        regionmapping.data = regionmapping.data.pix.aggregate(
            country=settings.country_combinations, agg_func="last"
        )
    # Ensure no duplicate country entries (pix.aggregate deduplicates when
    # country_combinations is set, but without it the raw CSV may have duplicates)
    if regionmapping.data.index.duplicated().any():
        n_dups = regionmapping.data.index.duplicated().sum()
        dups = regionmapping.data.index[regionmapping.data.index.duplicated(keep=False)]
        print(f"⚠️  {m}: Dropping {n_dups} duplicate country entries from regionmapping")
        print(f"   Duplicate countries: {sorted(set(dups.tolist()))}")
        regionmapping.data = regionmapping.data[~regionmapping.data.index.duplicated(keep='last')]
    regionmappings[m] = regionmapping


# Filter regionmapping and prepare for dissolve
filtered_regionmapping = regionmapping.filter(indexraster.index).data.rename("country")

# Check for duplicate index labels in the mapping
if filtered_regionmapping.index.duplicated().any():
    duplicates = filtered_regionmapping.index[filtered_regionmapping.index.duplicated(keep=False)]
    print(f"⚠️  Found {len(duplicates.unique())} countries with duplicate region mappings:")
    for dup in sorted(duplicates.unique()):
        regions = list(filtered_regionmapping[filtered_regionmapping.index == dup].values)
        print(f"  {dup}: {regions}")
    print("  → Keeping first occurrence for dissolve()")
    filtered_regionmapping = filtered_regionmapping[~filtered_regionmapping.index.duplicated(keep='first')]

indexraster_region = indexraster.dissolve(filtered_regionmapping).compute()

print(sorted(indexraster.index.tolist()))


areacella = xr.open_dataset(
    Path(settings.gridding_path, "areacella_input4MIPs_emissions_CMIP_CEDS-CMIP-2025-04-18_gn.nc")
)
cell_area = areacella["areacella"]

# %% [markdown]
# ## Rederive and save

# %%
# Output filename gets a species suffix when filtering, so partial rederives
# don't clobber the full-species CSV.
_species_suffix = ""
if SPECIES_FILTER is not None:
    _species_list = [SPECIES_FILTER] if isinstance(SPECIES_FILTER, str) else list(SPECIES_FILTER)
    _species_suffix = "_" + "-".join(_species_list)

rederived = rederive_downscaled_from_gridded(
    gridded_path=original_gridded_path,
    indexraster=indexraster,
    cell_area=cell_area,
    target_years=range(2023, 2101),
    species_filter=SPECIES_FILTER,
    save_path=settings.out_path / f"rederived_downscaled_{marker_to_run}_{VERSION_ESGF}{_species_suffix}.csv",
)

# %%
# Spot-check: open the CO2 anthro netCDF from the run we just re-aggregated.
# Glob keeps this robust to FILE_NAME_ENDING variations across markers/versions.
_co2_candidates = sorted(original_gridded_path.glob("CO2-em-anthro_*.nc"))
co2_filepath = _co2_candidates[0] if _co2_candidates else None

# %%
if co2_filepath is not None:
    xr.open_dataset(co2_filepath)["CO2_em_anthro"]

# %%
import pandas_indexing as pix

rederived.loc[pix.ismatch(country="aut", gas="N2O")]

# %%
tmp = rederived.pix.assign(model=MODEL_SELECTION)
tmp = tmp.pix.assign(scenario=SCENARIO_SELECTION)
tmp = tmp.pix.assign(method="reaggregated")

# %%
df = tmp
    
df.index = df.index.set_levels(
    df.index.levels[df.index.names.index("sector")].map(
        lambda x: SECTOR_RENAMES_INVERSE.get(x, x)
    ),
    level="sector"
)

# %%
df.loc[pix.ismatch(country="aut", gas="CO2")]

# %%
regionmapping = regionmappings[MODEL_SELECTION]
df = df.reset_index()
df["region"] = df["country"].map(regionmapping.data)
df.loc[df["country"] == "World", "region"] = "World"
df

# %%
target_year_cols = [yr for yr in rederived.columns if isinstance(yr, int)]
df = df[["gas", "sector", "region", "unit", "method", "model", "scenario", "country"] + target_year_cols]

# %%
# Separate index cols from year cols
index_cols = ["gas", "sector", "region", "unit", "method", "model", "scenario", "country"]
year_cols = [yr for yr in df.columns if isinstance(yr, int)]

# Reindex to full year range and interpolate
full_years = list(range(min(year_cols), max(year_cols) + 1))

df_interp = (
    df[index_cols + year_cols]
    .set_index(index_cols)
    .reindex(columns=full_years)
    .interpolate(axis=1, method="linear", limit_direction="both")
    .reset_index()
)

# %%
outpath = settings.out_path / f"rederived_downscaled_{marker_to_run}_{VERSION_ESGF}{_species_suffix}.csv"
df_interp.to_csv(outpath, index=False)

# %% [markdown]
# ## compare to initial downscaled csv

# %%
# Load original downscaled CSV from the same run we're re-aggregating.
# Concordia writes this as `downscaled-only-{GRIDDING_VERSION}.csv` next to the
# gridded netCDFs.
original_downscaled_path = original_gridded_path / f"downscaled-only-{GRIDDING_VERSION}.csv"

original = pd.read_csv(original_downscaled_path)
index_cols_orig = [c for c in original.columns if not str(c).isdigit()]
original = original.set_index(index_cols_orig)
original.columns = original.columns.astype(int)

original = original.droplevel("method")

cdr_sectors = {
    "Biochar",
    "Direct Air Capture",
    "Enhanced Weathering",
    "Ocean",
    "Other CDR",
    "Soil Carbon Management",
}

original = original.rename(
    index={s: "Other Capture and Removal" for s in cdr_sectors},
    level="sector"
)

original = original.groupby(level=original.index.names).sum()

# %%
# Align on common index — join on country, gas, sector, unit
# rederived already has these as index
index_cols = ['gas', 'sector', 'region', 'unit', 'model', 'scenario', 'country']

original = original.reset_index().set_index(index_cols)
df_interp = df_interp.set_index(index_cols)

orig_2100 = original[2100]
reder_2100 = df_interp[2100]

# Align
orig_aligned, reder_aligned = orig_2100.align(reder_2100, join="inner")

# Compute absolute and relative differences
abs_diff = reder_aligned - orig_aligned
rel_diff = abs_diff / orig_aligned.abs().replace(0, np.nan) * 100  # percent

comparison = pd.DataFrame({
    "original_2100": orig_aligned,
    "rederived_2100": reder_aligned,
    "abs_diff": abs_diff,
    "rel_diff_pct": rel_diff,
}).sort_values("abs_diff", key=abs, ascending=False)

# Flag large deviations
threshold_pct = 5.0
threshold_abs = 0.01  # Mt/yr — ignore tiny absolute differences even if rel is large

flagged = comparison[
    (comparison["rel_diff_pct"].abs() > threshold_pct) &
    (comparison["abs_diff"].abs() > threshold_abs)
]

print(f"Total entries compared: {len(comparison)}")
print(f"Flagged (>{threshold_pct}% and >{threshold_abs} Mt/yr): {len(flagged)}")
print(flagged.head(20))

# %%
missing_in_reder = orig_2100.index.difference(reder_2100.index)
orig_2100.loc[missing_in_reder].reset_index()

# %%
missing_in_orig = reder_2100.index.difference(orig_2100.index)
reder_2100.loc[missing_in_orig].reset_index()

# %%
set(reder_2100.index.get_level_values("country").unique()) - set(orig_2100.index.get_level_values("country").unique())

# %%
set(orig_2100.index.get_level_values("country").unique()) - set(reder_2100.index.get_level_values("country").unique())

# %%
outpath = settings.out_path / f"mismatch_downscaled_{marker_to_run}_{VERSION_ESGF}{_species_suffix}.csv"

# %%
flagged.to_csv(outpath)

# %%
flagged.sort_values(by="rel_diff_pct", )

# %%
reder_2100
