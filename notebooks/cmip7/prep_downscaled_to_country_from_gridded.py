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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SECONDS_PER_YEAR = 365.25 * 24 * 3600  # average

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


def get_sector_dict(gas: str, em_type: str):
    em_type = em_type.lower()
    
    gas = gas.upper()

    if "openburning" in em_type:
        return SECTOR_DICT_OPENBURNING_DEFAULT

    if "anthro" in em_type:
        if gas == "CO2":
            return SECTOR_DICT_ANTHRO_CO2_SCENARIO
        return SECTOR_DICT_ANTHRO_DEFAULT

    return None


def _days_in_month_noleap(t) -> int:
    days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    return days[t.month - 1]


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

    if "air" in em_type.lower():
        sector_type = "air"
    elif "openburning" in em_type.lower():
        sector_type = "openburning"
    elif "anthro" in em_type.lower():
        sector_type = "anthro"
    else:
        sector_type = "unknown"

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

    da_full = ds[var_name]

    # Get year for each time step
    if not hasattr(da_full.time.values[0], 'year'):
        times = pd.DatetimeIndex(da_full.time.values)
        file_years = times.year.values
    else:
        file_years = np.array([t.year for t in da_full.time.values])

    year_to_time_indices = {}
    for yr in target_years:
        idx = np.where(file_years == yr)[0]
        if len(idx) > 0:
            year_to_time_indices[yr] = idx

    if not year_to_time_indices:
        print(f"  [SKIP] None of the target years found in {nc_path.name}")
        ds.close()
        return None

    # Get sector names
    try:
        sector_dict = get_sector_dict(gas, em_type)
        if sector_dict is None:
            sector_names = [em_type]
        else:
            n_sectors = da_full.sizes.get("sector", 1)
            sector_names = [sector_dict.get(i, f"unknown_{i}") for i in range(n_sectors)]
    except ValueError as e:
        print(f"  [WARN] {e} — treating file as single-sector.")
        sector_names = [em_type]

    has_sector_dim = "sector" in da_full.dims
    has_level_dim = "level" in da_full.dims
    sector_iterator = list(enumerate(sector_names)) if has_sector_dim else [(0, sector_names[0])]

    # Sector ordering — include CDR sectors for CO2
    if sector_type == "air":
        sector_order = ["Aircraft"]
    elif sector_type in {"anthro", "openburning"}:
        ordering_key = (
            "CO2_em_anthro"
            if gas == "CO2" and sector_type == "anthro"
            else f"em_{sector_type}"
        )
        sector_order = SECTOR_ORDERING_DEFAULT.get(ordering_key)
        # For CO2 anthro, also include CDR sectors
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

    all_year_dfs = []

    for yr, time_indices in tqdm(year_to_time_indices.items(), desc=f"{nc_path.name}", leave=False):
        days_in_month = xr.DataArray(
            [_days_in_month_noleap(t) for t in da_full.time.values[time_indices]],
            dims="time"
        )
        seconds_per_month = days_in_month * 24 * 3600

        # flux (kg/m²/s) × seconds/month → kg/m²/month, sum over months → kg/m²/year
        da_year = (da_full.isel(time=time_indices) * seconds_per_month).sum(dim="time")

        # For AIR files: sum over pressure levels first
        if has_level_dim:
            da_year = da_year.sum(dim="level")

        records = []
        for s_idx, sector in sector_iterator:
            if has_sector_dim:
                da_sector = da_year.isel(sector=s_idx).squeeze()
            else:
                da_sector = da_year.squeeze()

            flux_mass = da_sector * cell_area  # kg/m²/year × m² → kg/year

            # Aircraft and International Shipping: global total only
            if sector in GLOBAL_SECTORS or sector_type == "air":
                total_kg_yr = float(flux_mass.sum().values)
                if gas == "N2O":
                    unit_out = "kt N2O/yr"
                    total_out_yr = total_kg_yr / 1e6
                else:
                    unit_out = f"Mt {gas}/yr"
                    total_out_yr = total_kg_yr / 1e9
                records.append({
                    "country": "World", "gas": gas.replace("-", "_"),
                    "sector": "Aircraft" if sector_type == "air" else sector,
                    "unit": unit_out, "year": yr,
                    "value": total_out_yr,
                })
                continue

            # Country-level aggregation
            country_totals = flux_mass.groupby(indexraster.indicator).sum()
            idx_to_country = {i + 1: c for i, c in enumerate(indexraster.index)}
            groupby_dim = country_totals.dims[0]

            for pos, idx_val in enumerate(country_totals.coords[groupby_dim].values):
                if int(idx_val) == 0:
                    continue
                country_code = idx_to_country.get(int(idx_val))
                if country_code is None:
                    continue
                val_kg_yr = float(country_totals.isel({groupby_dim: pos}).values)
                if gas == "N2O":
                    unit_out = "kt N2O/yr"
                    val_out_yr = val_kg_yr / 1e6
                else:
                    unit_out = f"Mt {gas}/yr"
                    val_out_yr = val_kg_yr / 1e9
                records.append({
                    "country": country_code, "gas": gas.replace("-", "_"),
                    "sector": sector, "unit": unit_out, "year": yr,
                    "value": val_out_yr,
                })

        if records:
            df_yr = pd.DataFrame(records)
            all_year_dfs.append(df_yr)

    ds.close()

    if not all_year_dfs:
        return None

    df_long = pd.concat(all_year_dfs, ignore_index=True)
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

    if not nc_files:
        raise FileNotFoundError(
            f"No netCDF files found in {gridded_path} matching '{file_pattern}' "
            f"(after exclusions)."
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
VERSION_ESGF: str = "1-1-0"          # version of the original run to rederive from
marker_to_run: str = "vl"
TARGET_YEAR: int = 2100

GRIDDING_VERSION: str = f"{marker_to_run}_{VERSION_ESGF}"   # folder name of original run

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

original_gridded_path = settings.out_path / GRIDDING_VERSION

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
rederived = rederive_downscaled_from_gridded(
    gridded_path=original_gridded_path,
    indexraster=indexraster,
    cell_area=cell_area,
    target_years=range(2023, 2101),
    save_path=settings.out_path / f"rederived_downscaled_{marker_to_run}_{VERSION_ESGF}.csv",
)

# %%
co2_filepath = Path("/Users/hoegner/GitHub/concordia/results/vl-ext_1-1-1/CO2-em-anthro_input4MIPs_emissions_ScenarioMIP_IIASA-IAMC-vl-ext-1-1-1_gn_210501-250012.nc")

# %%
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
outpath = "../../results/rederived_downscaled_vl_1-1-0.csv"
df_interp.to_csv(outpath, index=False)

# %% [markdown]
# ## compare to initial downscaled csv

# %%
# Load original downscaled
original_downscaled_path = "/Users/hoegner/GitHub/concordia/results/vl_1-1-0/downscaled-only-vl_1-1-0.csv"

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
outpath = "../../results/mismatch_downscaled_vl_1-1-0.csv"

# %%
flagged.to_csv(outpath)

# %%
flagged.sort_values(by="rel_diff_pct", )

# %%
reder_2100
