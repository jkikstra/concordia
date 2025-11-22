#!/usr/bin/env python
# %% [markdown]
# # Overview
#
# This notebook creates the pre-stages of the share proxies for openburning VOC species based on BB4CMIP7 and runs a number of checks comparing the pre-processed bulk emissions with the pre-processed speciated emissions, plotting the respective patterns as well as differences due to pre-processing. It's pretty slow to run.


# %% [markdown]
# ## Steps happening in this notebook
# - pre-process bulk data: calculate sectoral emissions from shares
# - re-grid to 0.5°
# - pre-process speciated data: calculate sectoral emissions from shares
# - re-grid to 0.5°
# - take 10 year average for each month (bulk and speciated)
# - re-aggregate forest sectors
# - check total difference between bulk emissions and sum of speciated emissions, sectorally resolved
# - creates plots of the differences (total and relative) between bulk and sum of speciated emissions for each month

# %%
# check later if we need all these imports 
import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import os
from dask.utils import SerializableLock

from tqdm import tqdm

import concordia.cmip7.utils_futureproxy_ceds_bb4cmip as uprox
from concordia.cmip7.utils_futureproxy_ceds_bb4cmip import _normalize_time_slice

# %%
from concordia.cmip7.CONSTANTS import GASES_ESGF_BB4CMIP_VOC, CONFIG, PROXY_YEARS

# %%
VERSION = CONFIG

# %%
try:
    # when running the script from a terminal or otherwise
    cmip7_dir = Path(__file__).resolve()
    settings = uprox.get_settings(base_path=cmip7_dir, file = CONFIG)
except (FileNotFoundError, NameError):
    try:
        # when running the script from a terminal or otherwise
        cmip7_dir = Path(__file__).resolve().parent
        settings = uprox.get_settings(base_path=cmip7_dir, file = CONFIG)
    except (FileNotFoundError, NameError):
        # Fallback for interactive/Jupyter mode, where 'file location' does not exist
        cmip7_dir = Path().resolve()  # one up
        settings = uprox.get_settings(base_path=cmip7_dir, file = CONFIG)

# %%
#cmip7_dir = "/Users/hoegner/Projects/CMIP7/concordia_cmip7_esgf_v0_alpha/input/gridding/esgf/bb4cmip7_voc"

# %%
settings.gridding_path

# %%
lock = SerializableLock()

# Workaround for HDF5 on Windows: disable file locking to avoid sporadic read errors
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

# %% [markdown]
# ## Settings

# %%
PROXY_TIME_RANGE = [2014,2023]

# %%
sectors = ["AGRI", "BORF", "DEFO", "PEAT", "SAVA", "TEMF"]
sectors_new = ["AWB", "PEAT", "GRSB", "FRTB"]

# %%
# what data to output
new_proxies_location = settings.proxy_path
# ensure output directory exists
new_proxies_location.mkdir(parents=True, exist_ok=True)

scenario_years = PROXY_YEARS


# %%
# load CEDS example file to get the right grid settings
template_file = (
    settings.gridding_path
    / "example_files" / "GCAM4-SSP4-34-SPA4-V25-Harmonized-DB-Sulfur-em-aircraft-anthro_input4MIPs_emissions_CMIP_IAMC-V1_gn_201501-210012.nc"
)

template = xr.open_dataset(template_file)


# %%
target_lat = template["lat"].values
target_lon = template["lon"].values

# %% [markdown]
# ## Functions

# %%
def get_bb4cmip7_location(variable, kind="speciated", data_type="total"):
    """
    Return the path to a BB4CMIP7 VOC dataset.

    Parameters
    ----------
    variable : str
        Species name (e.g. 'NMVOC', 'C2H2', etc.)
    kind : {'bulk', 'speciated'}
        Whether to load the bulk or speciated dataset.
    data_type : {'total', 'percentage'}
        Whether to load total emissions or percentage allocations.

    Returns
    -------
    Path
        Full path to the NetCDF dataset.
    """
    base = Path(settings.gridding_path)
    folder = "esgf/bb4cmip7" if kind == "bulk" else "esgf/bb4cmip7_voc"

    time_range = "190001-202312" if data_type == "total" else "175001-202312"

    filename = f"{variable}_input4MIPs_emissions_CMIP_DRES-CMIP-BB4CMIP7-2-1_gn_{time_range}.nc"

    return base / folder / variable / "gn" / "v20250612" / filename


# %%
def load_bb4cmip7_datasets(g, kind="speciated", lock=None, proxy_time_range=None):
    """
    Load BB4CMIP7 datasets for a given gas `g` and fixed kind.

    Loads both the total emissions file and all sectoral percentage files.

    Parameters
    ----------
    g : str
        Gas name (e.g., "CO", "VOC", etc.)
    kind : {"bulk", "speciated"}, optional
        Dataset kind (defaults to "speciated").
    lock : threading.Lock or dask.utils.SerializableLock, optional
        Lock to use during file access (recommended when using dask).
    proxy_time_range : slice or str, optional
        Time range passed to `.sel(time=...)`. Uses `_normalize_time_slice` if None.

    Returns
    -------
    dict
        {
            "total": <xarray.Dataset>,
            "percentage": {sector: <xarray.Dataset>, ...}
        }
    """
    # sectors for which we load percentage files
    sectors = ["AGRI", "BORF", "DEFO", "PEAT", "SAVA", "TEMF"]

    if proxy_time_range is None:
        proxy_time_range = PROXY_TIME_RANGE  # fallback to global

    # load total emissions
    ds_total = xr.open_dataset(
        get_bb4cmip7_location(variable=g, kind=kind, data_type="total"),
        engine="h5netcdf",
        chunks={},
        lock=lock
    ).sel(time=_normalize_time_slice(proxy_time_range)).drop_vars(
        ["lat_bnds", "lon_bnds", "time_bnds"]
    ).rename({f"{g}": "emissions"})

    # load percentage datasets per sector
    ds_percentages = {}
    for s in sectors:
        variable = f"{g}percentage{s}"
        ds = xr.open_dataset(
            get_bb4cmip7_location(variable=variable, kind=kind, data_type="percentage"),
            engine="h5netcdf",
            chunks={},
            lock=lock
        ).sel(time=_normalize_time_slice(proxy_time_range)).drop_vars(
            ["lat_bnds", "lon_bnds", "time_bnds"]
        ).rename({f"{variable}": "percentage"})

        ds_percentages[s] = ds

    return {"total": ds_total, "percentage": ds_percentages}


# %%
def compute_sectoral_emissions(datasets):
    """
    Compute sectoral emissions by multiplying total emissions with sector percentages,
    and interpolate both total and sector datasets from 0.25° to 0.5° grid.

    Parameters
    ----------
    datasets : dict
        Output from `load_bb4cmip7_datasets()`, containing:
            - datasets["total"] (xarray.Dataset with variable 'emissions')
            - datasets["percentage"][sector] (xarray.Dataset with variable 'percentage')

    Returns
    -------
    dict
        Dictionary of xarray.Datasets per sector, each containing variable 'emissions'
        interpolated onto the 0.5° grid.
    """
    ds_total = datasets["total"]
    ds_percentages = datasets["percentage"]

    # First interpolate the total emissions to the target grid
    ds_total_interp = ds_total.interp(
        latitude=target_lat,
        longitude=target_lon,
        method="linear"
    )

    sectoral_emissions = {}

    for sector, ds_perc in ds_percentages.items():
        # Interpolate the sector percentage dataset to target grid
        ds_perc_interp = ds_perc.interp(
            latitude=target_lat,
            longitude=target_lon,
            method="linear"
        )

        # Multiply totals by sector percentages (convert percentage to fraction)
        ds_sector = xr.Dataset({
            "emissions": ds_total_interp["emissions"] * (ds_perc_interp["percentage"] / 100)
        })

        sectoral_emissions[sector] = ds_sector

    return sectoral_emissions


# %%
def rename_and_aggregate_emissions(data, variable="emissions"):
    """
    Rename and aggregate sectors for either a single-level or nested dictionary.

    Supports both:
      - {sector: xarray.Dataset}
      - {species: {sector: xarray.Dataset}}

    Parameters
    ----------
    data : dict
        Either a dict of sectors, or a nested dict {species: {sector: Dataset}}.
    variable : str
        Name of the variable in each Dataset (e.g. "emissions").

    Returns
    -------
    dict
        Updated dictionary with renamed and aggregated sectors.
        - If input was {sector: Dataset}, output is {sector: Dataset}.
        - If input was {species: {sector: Dataset}}, output is {species: {sector: Dataset}}.
    """
    sector_mapping_singlesector = {
        "AGRI": "AWB",
        "PEAT": "PEAT",
        "SAVA": "GRSB"
    }

    gfed_sectors_forest = ["BORF", "DEFO", "TEMF"]

    def _aggregate_single(sectoral_emissions):
        """Aggregate and rename for one sector dictionary."""
        updated = {}

        # Rename individual sectors
        for old_sector, new_sector in sector_mapping_singlesector.items():
            if old_sector in sectoral_emissions:
                updated[new_sector] = sectoral_emissions[old_sector]

        # Aggregate forest sectors
        forest_datasets = [sectoral_emissions[s] for s in gfed_sectors_forest if s in sectoral_emissions]
        if forest_datasets:
            forest_sum = sum(ds[variable] for ds in forest_datasets)
            updated["FRTB"] = xr.Dataset({variable: forest_sum})

        # Copy remaining sectors (not renamed or aggregated)
        for s, ds in sectoral_emissions.items():
            if s not in sector_mapping_singlesector and s not in gfed_sectors_forest:
                updated[s] = ds

        return updated

    # detect if nested or not
    first_value = next(iter(data.values()))

    if isinstance(first_value, dict):
        # Nested dict case: {species: {sector: Dataset}}
        return {sp: _aggregate_single(sector_dict) for sp, sector_dict in data.items()}
    else:
        # Single-level dict case: {sector: Dataset}
        return _aggregate_single(data)


# %%
def time_average(data, variable="emissions", time_range=(2014, 2023)):
    """
    Compute monthly climatological means over a specified time range for
    emissions data, preserving the input dictionary structure.

    Works for both:
      - {sector: xarray.Dataset}
      - {species: {sector: xarray.Dataset}}

    Parameters
    ----------
    data : dict
        Dictionary of emissions data (sector-level or species→sector-level).
    variable : str
        Variable name in the Dataset (default: 'emissions').
    time_range : tuple(int, int)
        Start and end years, inclusive (default: (2014, 2023)).

    Returns
    -------
    dict
        Same structure as input, but each Dataset contains the monthly mean
        climatology over the selected years, with dimension 'month'.
    """
    start_year, end_year = time_range

    def _monthly_mean(ds):
        """Select time range and compute monthly climatology."""
        ds_sel = ds.sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-31"))
        monthly = ds_sel[variable].groupby("time.month").mean(dim="time", skipna=True)
        return monthly.to_dataset(name=variable)

    def _process_dict(d):
        """Apply to a single-level dict {sector: Dataset}."""
        return {sector: _monthly_mean(ds) for sector, ds in d.items()}

    # Detect whether nested structure or single level
    first_val = next(iter(data.values()))
    if isinstance(first_val, dict):
        # Nested structure: {species: {sector: Dataset}}
        return {sp: _process_dict(sector_dict) for sp, sector_dict in data.items()}
    else:
        # Single-level structure: {sector: Dataset}
        return _process_dict(data)


# %% [markdown]
# ## Obtain speciated shares for all sectors

# %%
# load and process bulk VOC emissions
bulk_datasets = load_bb4cmip7_datasets("NMVOCbulk", kind="bulk", lock=lock, proxy_time_range=PROXY_TIME_RANGE)
bulk_sectoral_emissions = compute_sectoral_emissions(bulk_datasets)
bulk_ds_total = bulk_datasets["total"]

# load and process speciated emissions
speciated_datasets = {}
speciated_sectoral_emissions = {}

for sp in GASES_ESGF_BB4CMIP_VOC:
    ds_sp = load_bb4cmip7_datasets(sp, kind="speciated", lock=lock, proxy_time_range=PROXY_TIME_RANGE)
    ds_sp_sectoral = compute_sectoral_emissions(ds_sp)
    
    speciated_datasets[sp] = ds_sp      # contains both total and percentage
    speciated_sectoral_emissions[sp] = ds_sp_sectoral  # contains emissions per sector

# %%
# rename single sectors, aggregate forest sectors
bulk_renamed = rename_and_aggregate_emissions(bulk_sectoral_emissions)
speciated_renamed = rename_and_aggregate_emissions(speciated_sectoral_emissions)

# %%
# time-average over PROXY_RANGES on emissions
bulk_average = time_average(bulk_renamed)
speciated_average = time_average(speciated_renamed)

# %%
# check total difference between bulk emissions and sum of speciated emissions, sectorally resolved
sector_diffs = {}

for sector, bulk_ds in tqdm(bulk_average.items(), desc="Checking sectors", unit="sector"):
    bulk = bulk_ds["emissions"].fillna(0)
    
    # Species-level DataArrays (all aligned on grid)
    speciated_fields = [
        sp_dict[sector]["emissions"].fillna(0)
        for sp_dict in speciated_average.values()
        if sector in sp_dict
    ]
    
    if not speciated_fields:
        continue  # skip sectors missing data
    
    speciated_sum = sum(speciated_fields)
    diff = (bulk - speciated_sum).compute()
    sector_diffs[sector] = diff

    print(f"Sector {sector}: min={float(diff.min().values)}, max={float(diff.max().values)}")

# %% [markdown]
# ### compare bulk vs sum of speciated emissions after processing

# %%
outpath = new_proxies_location / "checks"
outpath.mkdir(parents=True, exist_ok=True)

# %%
# create plots of 1) bulk emissions, 2) sum of speciated emission, 3) difference between the two, 4) relative difference

for month_idx in range(12):

    sector_data = {}

    for sector in sectors:
        bulk = bulk_average[sector]["emissions"].isel(month=month_idx).fillna(0).compute()
        
        speciated_fields = [
            sp_dict[sector]["emissions"].isel(month=month_idx).fillna(0).compute()
            for sp_dict in speciated_average.values()
            if sector in sp_dict
        ]
        
        if not speciated_fields:
            continue
        
        speciated_sum = sum(speciated_fields)
        diff = bulk - speciated_sum
        rel_diff = xr.where(bulk != 0, (bulk - speciated_sum) / bulk, 0)
    
        sector_data[sector] = {
            "bulk": bulk,
            "speciated_sum": speciated_sum,
            "difference": diff,
            "relative_difference": rel_diff
        }
    
    n_rows = 4
    n_cols = len(sector_data)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows), constrained_layout=True)
    
    bulk_vals = np.concatenate([np.abs(s["bulk"].values) for s in sector_data.values()])
    speciated_vals = np.concatenate([np.abs(s["speciated_sum"].values) for s in sector_data.values()])
    vmin_common, vmax_common = 0, np.percentile(np.concatenate([bulk_vals, speciated_vals]), 99)

    diff_vals = np.concatenate([s["difference"].values for s in sector_data.values()])
    max_abs_diff = np.percentile(np.abs(diff_vals), 99)
    vmin_diff, vmax_diff = -max_abs_diff, max_abs_diff
    
    mappables = [None] * n_rows
    
    for col, sector in enumerate(sector_data):
        data_dict = sector_data[sector]
    
        im0 = data_dict["bulk"].plot(ax=axes[0, col], cmap='viridis', vmin=vmin_common, vmax=vmax_common, add_colorbar=False)
        axes[0, col].set_title(f"{sector} - bulk")
        mappables[0] = im0
    
        im1 = data_dict["speciated_sum"].plot(ax=axes[1, col], cmap='viridis', vmin=vmin_common, vmax=vmax_common, add_colorbar=False)
        axes[1, col].set_title(f"{sector} - sum(speciated)")
        mappables[1] = im1
    
        im2 = data_dict["difference"].plot(ax=axes[2, col], cmap='RdBu', vmin=vmin_diff, vmax=vmax_diff, add_colorbar=False)
        axes[2, col].set_title(f"{sector} - difference")
        mappables[2] = im2

        vmin_rel, vmax_rel = -0.5, 0.5
        im3 = data_dict["relative_difference"].plot(ax=axes[3, col], cmap='RdBu', vmin=vmin_rel, vmax=vmax_rel, add_colorbar=False)
        axes[3, col].set_title(f"{sector} - relative difference")
        mappables[3] = im3
    
    cbar_labels = ["Bulk", "Speciated Sum", "Difference", "Relative Difference"]
    for row in range(n_rows):
        cbar = fig.colorbar(mappables[row], ax=axes[row, :], orientation='vertical', fraction=0.02, pad=0.02)
        cbar.set_label(cbar_labels[row], fontsize=10)
    
    row_labels = ["Bulk", "Speciated Sum", "Difference", "Relative Difference"]
    for row in range(n_rows):
        axes[row, 0].set_ylabel(row_labels[row], fontsize=12)
    
    plt.savefig(Path(outpath, f"month_{month_idx}.png"), dpi=300)
    plt.close(fig)
