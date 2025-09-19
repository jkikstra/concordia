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

# %% [markdown]
# Steps:
# 1. load VOC totals, for the relevant historical year (2023)
# 1. load VOC speciated, for the relevant historical year (2023)
# 1. calculate percentage for each grid cell
# 1. write out percentage to proxy_rasters folder as input to workflow

# %%
import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt
import pandas_indexing as pix
import pandas as pd
import numpy as np
import os
import dask
from dask import delayed, compute
from dask.diagnostics import ProgressBar
from dask.utils import SerializableLock
import seaborn as sns
from typing import Optional

from concordia.cmip7 import utils as cmip7_utils

# %% [markdown]
# ## prepare setup

# %%
lock = SerializableLock()

# %%
from concordia.cmip7.CONSTANTS import CONFIG, PROXY_YEARS, GASES_ESGF_CEDS_VOC
from concordia.settings import Settings
import concordia.cmip7.utils_futureproxy_ceds_bb4cmip as uprox
from concordia.cmip7.utils_futureproxy_ceds_bb4cmip import _normalize_time_slice

VERSION = CONFIG
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
PROXY_TIME_RANGE = 2023 # in principle, could also take some average like [2014,2023]


# %%
# grid_file_location = "/home/hoegner/Projects/CMIP7/input/gridding/"
grid_file_location = settings.gridding_path

ceds_data_location = Path(grid_file_location, "esgf", "ceds", "CMIP7_anthro_VOC")
PATH_VOC_LOCAL = "C:/Users/kikstra/Downloads/temp_VOC" # placed in different location to avoid 256-character filepath restriction in Windows
# PATH_VOC_LOCAL = None # use None if it is located in the same folder as the data on SharePoint, use this

new_proxies_location = Path(grid_file_location, "proxy_rasters", "VOC_speciation")
new_proxies_location.mkdir(parents=True, exist_ok=True)

def get_ceds_voc_totals_location(variable: str, local_adjusted_path: Optional[str] = None) -> Path:
    """
    Get the file path for CEDS VOC totals data.
    
    Args:
        variable: The variable name (e.g., 'VOC01_alcohols_em_speciated_VOC_anthro')
        local_adjusted_path: Optional local path override. If None, uses settings.gridding_path
        
    Returns:
        Path object pointing to the NetCDF file
    """
    # Constants for file path construction
    GRIDDIN_DATA_SUBDIR = "esgf/ceds/CMIP7_anthro"
    FILENAME_SUFFIX = "input4MIPs_emissions_CMIP_CEDS-CMIP-2025-04-18_gn_200001-202312.nc"

    # Construct the relative path and filename
    filename = f"{variable.replace("_", "-")}_{FILENAME_SUFFIX}"
    
    # Choose base path based on whether local_adjusted_path is provided
    if local_adjusted_path is None:
        base_path = Path(settings.gridding_path) / GRIDDIN_DATA_SUBDIR
    else:
        base_path = Path(local_adjusted_path)
    
    return base_path / filename

def get_ceds_voc_speciation_location(variable: str, local_adjusted_path: Optional[str] = None) -> Path:
    """
    Get the file path for CEDS VOC speciation data.
    
    Args:
        variable: The variable name (e.g., 'VOC01_alcohols_em_speciated_VOC_anthro')
        local_adjusted_path: Optional local path override. If None, uses settings.gridding_path
        
    Returns:
        Path object pointing to the NetCDF file
    """
    # Constants for file path construction
    GRIDDIN_DATA_SUBDIR = "esgf/ceds/CMIP7_anthro_VOC"
    VERSION = "v20250421"
    FILENAME_SUFFIX = "input4MIPs_emissions_CMIP_CEDS-CMIP-2025-04-18-supplemental_gn_200001-202312.nc"

    # Construct the relative path and filename
    relative_path = f"{variable}/gn/{VERSION}"
    filename = f"{variable.replace("_", "-")}_{FILENAME_SUFFIX}"
    
    # Choose base path based on whether local_adjusted_path is provided
    if local_adjusted_path is None:
        base_path = Path(settings.gridding_path) / GRIDDIN_DATA_SUBDIR
    else:
        base_path = Path(local_adjusted_path)
    
    return base_path / relative_path / filename

# %%
# load example
ds = xr.open_dataset(
        get_ceds_voc_speciation_location(
            variable = "VOC01_alcohols_em_speciated_VOC_anthro",
            local_adjusted_path=PATH_VOC_LOCAL
        ),
        # engine="netcdf4",
        chunks={},
        lock=lock
    )
ds

tot = xr.open_dataset(
        get_ceds_voc_totals_location(
            variable = "NMVOC_em_anthro"
        ),
        # engine="netcdf4",
        chunks={},
        lock=lock
    )
tot


# %% [markdown]
# ## generate proxy rasters

# %%
scenario_years = PROXY_YEARS

sector_mapping = {
    0: "AGR",
    1: "ENE",
    2: "IND",
    3: "TRA",
    4: "RCO",
    5: "SLV",
    6: "WST",
    7: "SHP"
}

# sector_mapping_shp = {
#     7: "SHP"
# }


# %%
# Load VOC bulk data

tot = xr.open_dataset(
        get_ceds_voc_totals_location(
            variable = "NMVOC_em_anthro"
        ),
        # engine="netcdf4",
        chunks={},
        lock=lock
    ).sel(time=_normalize_time_slice(PROXY_TIME_RANGE))
tot


# %%
# Keep only proxy years

# loop through all CEDS em-anthro from input4MIP files
for v in GASES_ESGF_CEDS_VOC:
# for v in [GASES_ESGF_CEDS_VOC[0]]: # only run one to test

    # import file 
    ds = xr.open_dataset(
        get_ceds_voc_speciation_location(
            variable = v,
            local_adjusted_path=PATH_VOC_LOCAL
        ),
        engine="netcdf4",
        chunks={},
        lock=lock
    ).sel(
        time=_normalize_time_slice(PROXY_TIME_RANGE)
    ).drop_vars(
        # drop variables we don't need
        ["lat_bnds", "lon_bnds", "time_bnds", "sector_bnds"]
    )

    # do multiplication (now backed by NumPy arrays, detached from file IO)
    voc_spec = xr.Dataset({
        "emissions_share": ds[v] * tot["NMVOC_em_anthro"]
    })

    # # drop SHP sector, this has to be written into its own proxy file
    # ds = ds.drop_sel(sector=7)
    
    # add gas dimension
    ds = ds.expand_dims(dim={"gas": [f"{v}"]})
    
    # split time into year and month
    ds = ds.assign_coords(year=("time", ds["time"].dt.year.data), month=("time", ds["time"].dt.month.data)).groupby(["year", "month"]).mean()

    # Format ysel for filename and year variable if it's not an integer
    if not isinstance(PROXY_TIME_RANGE, int):
        ysel = f"{min(PROXY_TIME_RANGE)}_{max(PROXY_TIME_RANGE)}"
    else:
        ysel = PROXY_TIME_RANGE

    # Take average over the different years
    ds = ds.mean(dim="year")
    ds = ds.assign_coords(year_range=ysel)

    # project onto future years
    ds = ds.expand_dims({"year": scenario_years})

    # rename sectors and reorder dimensions
    ds = ds.assign_coords(sector=ds["sector"].to_series().replace(sector_mapping).values)

    ds_reordered = xr.Dataset(
    {var: (("lat", "lon", "gas", "sector", "year", "month"), ds[var].transpose("lat", "lon", "gas", "sector", "year", "month").values)
     for var in ds.data_vars},
    coords={dim: ds[dim] for dim in ["lat", "lon", "gas", "sector", "year", "month"]}
    ).chunk({"month": 12})

    
    outfile = new_proxies_location / f"{v}_{ysel}.nc"
    if outfile.exists():
        outfile.unlink()

    encoding = {
        var: {"zlib": True, "complevel": 4, "dtype": "float32"}
        for var in ds_reordered.data_vars
    }
    
    with ProgressBar():
        ds_reordered.to_netcdf(outfile, mode="w", encoding=encoding, compute=True)

# %%
