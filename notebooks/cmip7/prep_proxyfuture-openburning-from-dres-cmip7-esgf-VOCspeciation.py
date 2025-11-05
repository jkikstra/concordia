#!/usr/bin/env python
# %% [markdown]
# # Overview
#
# This should create percentage share proxies for openburning VOC species based on BB4CMIP7


# %% [markdown]
# ## Steps happening in this notebook
# - pre-process bulk data: calculate sectoral emissions from shares
# - re-grid to 0.5°
# - pre-process speciated data: calculate sectoral emissions from shares
# - re-grid to 0.5°
# - take 10 year average for each month (bulk and speciated)
# - calculate speciated from bulk
# - adjust to CMIP7 file conventions
# - write out

# %%
# check later if we need all these imports 
import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt
import pandas_indexing as pix
import pandas as pd
import numpy as np
import os
import dask
from dask import delayed, compute
from dask import config as dask_config
from dask.diagnostics import ProgressBar
from dask.utils import SerializableLock
from typing import Optional
import seaborn as sns

from tqdm import tqdm

from concordia.cmip7 import utils as cmip7_utils
from concordia.settings import Settings
import concordia.cmip7.utils_futureproxy_ceds_bb4cmip as uprox
from concordia.cmip7.utils_futureproxy_ceds_bb4cmip import _normalize_time_slice

# %%
from concordia.cmip7.CONSTANTS import GASES, GASES_ESGF_BB4CMIP_VOC, GASES_ESGF_BB4CMIP, CONFIG, PROXY_YEARS

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
cmip7_dir = "/Users/hoegner/Projects/CMIP7/concordia_cmip7_esgf_v0_alpha/input/gridding/esgf/bb4cmip7_voc"

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

# %%
# what data to output
new_proxies_location = settings.proxy_path
# ensure output directory exists
new_proxies_location.mkdir(parents=True, exist_ok=True)

scenario_years = [2022, 2023, 2024, 2025, 2030, 2035, 2040, 2045, 2050, 2055, 2060, 2065, 2070, 2075, 2080, 2085, 2090, 2095, 2100]


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
    and interpolate each sector dataset from 0.25° to 0.5° grid.

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

    sectoral_emissions = {}

    for sector, ds_perc in ds_percentages.items():
        # Multiply totals by sector percentages (convert percentage to fraction)
        ds_sector = xr.Dataset({
            "emissions": ds_total["emissions"] * (ds_perc["percentage"] / 100)
        })

        # Interpolate to 0.5° grid using globally defined target_lat/lon
        ds_sector_interp = ds_sector.interp(
            latitude=target_lat,
            longitude=target_lon,
            method="linear"
        )

        sectoral_emissions[sector] = ds_sector_interp

    return sectoral_emissions


# %%
def monthly_average_voc_species_shares(speciated_sectoral_emissions, bulk_sectoral_emissions):
    """
    Compute monthly VOC species shares (% of bulk) for each sector,
    using the ratio of time-mean emissions.
    This preserves global totals when applied to aggregate emissions.
    """
    monthly_shares = {}

    for sp, sector_dict in speciated_sectoral_emissions.items():
        monthly_shares[sp] = {}
        for sector, ds_sp in sector_dict.items():
            ds_bulk = bulk_sectoral_emissions[sector]

            # 1. Compute monthly averages (mean over years per calendar month)
            sp_monthly = ds_sp["emissions"].groupby("time.month").mean(dim="time", skipna=True)
            bulk_monthly = ds_bulk["emissions"].groupby("time.month").mean(dim="time", skipna=True)

            # 2. Compute ratio (species/bulk) * 100
            share = xr.where(bulk_monthly != 0, (sp_monthly / bulk_monthly) * 100, 0.0)

            monthly_shares[sp][sector] = share.to_dataset(name="share")

    return monthly_shares


# %%
def rename_and_aggregate_sectors(species_sectoral_shares):
    """
    Rename some sectors and aggregate forest sectors into FRTB.

    Parameters
    ----------
    species_sectoral_shares : dict
        Nested dictionary {species: {sector: xarray.Dataset with 'share'}}

    Returns
    -------
    dict
        Updated dictionary {species: {new_sector: Dataset}} with renamed/aggregated sectors.
    """
    # sector mapping
    sector_mapping_singlesector = {
        "AGRI": "AWB",
        "PEAT": "PEAT",
        "SAVA": "GRSB"
    }

    gfed_sectors_forest = ["BORF", "DEFO", "TEMF"]

    updated_shares = {}

    for sp, sector_dict in species_sectoral_shares.items():
        updated_shares[sp] = {}

        # rename single sectors
        for old_sector, new_sector in sector_mapping_singlesector.items():
            if old_sector in sector_dict:
                updated_shares[sp][new_sector] = sector_dict[old_sector]

        # aggregate forest sectors into FRTB
        forest_datasets = [sector_dict[s] for s in gfed_sectors_forest if s in sector_dict]
        if forest_datasets:
            
            # sum shares along each grid cell / month
            forest_sum = sum(ds["share"] for ds in forest_datasets)
            updated_shares[sp]["FRTB"] = xr.Dataset({"share": forest_sum})

    return updated_shares


# %% [markdown]
# ## Obtain speciated shares for all sectors

# %%
# load and process bulk VOC emissions
bulk_datasets = load_bb4cmip7_datasets("NMVOCbulk", kind="bulk", lock=lock, proxy_time_range=PROXY_TIME_RANGE)
bulk_sectoral_emissions = compute_sectoral_emissions(bulk_datasets)
bulk_ds_total = bulk_datasets["total"]  # already interpolated inside compute_sectoral_emissions

# load and process speciated emissions
speciated_datasets = {}
speciated_sectoral_emissions = {}

for sp in GASES_ESGF_BB4CMIP_VOC:
    ds_sp = load_bb4cmip7_datasets(sp, kind="speciated", lock=lock, proxy_time_range=PROXY_TIME_RANGE)
    ds_sp_sectoral = compute_sectoral_emissions(ds_sp)
    
    speciated_datasets[sp] = ds_sp      # contains both total and percentage
    speciated_sectoral_emissions[sp] = ds_sp_sectoral  # contains emissions per sector

# %%
# obtain shares for all species-sector combinations
species_sectoral_shares = monthly_average_voc_species_shares(
    speciated_sectoral_emissions, bulk_sectoral_emissions
)

# %%
# rename single sectors, aggregate forest fire sectors
shares_final = rename_and_aggregate_sectors(species_sectoral_shares)

# %%
new_proxies_location

# %% [markdown]
# ## Compute and export

# %%
outdir = new_proxies_location / "NMVOC_speciation"
outdir.mkdir(parents=True, exist_ok=True)

ysel_filename="2014-23"

encoding = {"share": {"zlib": True,"complevel": 4}}

for sp in tqdm(GASES_ESGF_BB4CMIP_VOC, desc="Species"):

    # List to hold each sector's DataArray
    sector_arrays = []
    sector_names = []

    for sector, ds_share in shares_final[sp].items():
        # Rename lat/lon if necessary
        ds_share = ds_share.rename({
            "latitude": "lat" if "latitude" in ds_share.dims else "lat",
            "longitude": "lon" if "longitude" in ds_share.dims else "lon"
        })

        # Ensure variable is DataArray
        da = ds_share["share"]
        # Add 'sector' dimension
        da = da.expand_dims({"sector": [sector]})
        sector_arrays.append(da)
        sector_names.append(sector)

    # Combine all sectors along 'sector' dimension
    combined = xr.concat(sector_arrays, dim="sector")
    combined = combined.assign_coords(sector=sector_names)

    # Add gas dimension
    combined = combined.expand_dims({"gas": [sp]})

    # Broadcast to scenario years
    combined = combined.expand_dims({"year": scenario_years})

    # Reorder dimensions: lat, lon, gas, sector, year, month
    combined = combined.transpose("lat", "lon", "gas", "sector", "year", "month")

    # Ensure float32 and unify chunks
    combined = combined.unify_chunks().astype("float32")

    # Output file
    outfile = outdir / f"{sp}_other_voc_em_speciated_NMVOC_openburning_{ysel_filename}.nc"
    combined.to_netcdf(outfile, encoding=encoding)

