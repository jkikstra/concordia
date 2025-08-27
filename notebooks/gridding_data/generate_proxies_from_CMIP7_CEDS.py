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

from concordia.cmip7 import utils as cmip7_utils

# %% [markdown]
# ## prepare setup

# %%
lock = SerializableLock()

# %%
grid_file_location = "/home/hoegner/Projects/CMIP7/input/gridding/"
# grid_file_location = "C:/Users/kikstra/IIASA/ECE.prog - Documents/Projects/CMIP7/IAM Data Processing/concordia_cmip7_v0_2/input/gridding/"

ceds_data_location = Path(grid_file_location,  "CEDS_CMIP7")
old_proxies_location = Path(grid_file_location, "proxy_rasters")
new_proxies_location = Path(grid_file_location, "proxy_rasters_ceds")
new_proxies_location.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## example file

# %%
## import previous proxy raster for comparison

example_proxy = xr.open_dataset(
        Path(old_proxies_location, "anthro_VOC.nc"),
        engine="netcdf4",
        lock=lock
    ).sector

# %% [markdown]
# ## generate proxy rasters

# %% [markdown]
# to do:
# - [x] add species information as gas coordinate
# - [x] discard bounds (lat_bnds, lon_bnds, sector_bnds, time_bnds)
# - [x] translate BC_em_anthro variable into emissions variable
# - [x] split time into year and month; expand year to the years we want (this is just duplicating data)
# - [x] dictionary that translates sector information
# - [x] reorder coordinates
# - [ ] rewrite attributes?

# %%
years = [2023, 2024, 2025, 2030, 2035, 2040, 2045, 2050, 2055, 2060, 2065, 2070, 2075, 2080, 2085, 2090, 2095, 2100]

sector_mapping = {
    0: "AGR",
    1: "ENE",
    2: "IND",
    3: "TRA",
    4: "RCO",
    5: "SLV",
    6: "WST"
}

# %%
# loop through all CEDS em-anthro from input4MIP files
for file in ceds_data_location.glob("*.nc"):

    # extract species information from filename
    species = file.stem.split("-")[0]

    # import file 
    ds = xr.open_dataset(
        file,
        engine="netcdf4",
        chunks={"time": 12},
        lock=lock
    )

    # drop variables we don't need and rename the one we need
    ds = ds.drop_vars(["lat_bnds", "lon_bnds", "time_bnds", "sector_bnds"]).rename({f"{species}_em_anthro": "emissions"})

    # drop SHP sector, this has to be written into its own proxy file
    ds = ds.drop_sel(sector=7)
    
    # add gas dimension
    ds = ds.expand_dims(dim={"gas": [f"{species}"]})
    
    # rename NMVOC to VOC    
    if species == "NMVOC":
        ds = ds.assign_coords(gas=["VOC"])
        species = "VOC"
        
    # rename SO2 to Sulfur
    if species == "SO2":
        ds = ds.assign_coords(gas=["Sulfur"])
        species = "Sulfur"

    # split time into year and month
    ds = ds.assign_coords(year=("time", ds["time"].dt.year.data), month=("time", ds["time"].dt.month.data)).groupby(["year", "month"]).mean()

    # select 2023 data and project it onto future years
    ds = ds.sel(year=2023).expand_dims({"year": years})

    # rename sectors and reorder dimensions
    ds = ds.assign_coords(sector=ds["sector"].to_series().replace(sector_mapping).values)

    ds_reordered = xr.Dataset(
    {var: (("lat", "lon", "gas", "sector", "year", "month"), ds[var].transpose("lat", "lon", "gas", "sector", "year", "month").values)
     for var in ds.data_vars},
    coords={dim: ds[dim] for dim in ["lat", "lon", "gas", "sector", "year", "month"]}
    ).chunk({"month": 12})

    outfile = new_proxies_location / f"anthro_{species}.nc"
    if outfile.exists():
        outfile.unlink()

    with ProgressBar():
        ds_reordered.to_netcdf(outfile, mode="w", compute=True)

# %% [markdown]
# ## check one of the new proxy rasters

# %%
proxy = xr.open_dataset(
        Path(new_proxies_location, "anthro_BC.nc"),
        engine="netcdf4",
        lock=lock
    )

ceds = xr.open_dataset(
        Path(ceds_data_location, "OC-em-anthro_input4MIPs_emissions_CMIP_CEDS-CMIP-2025-04-18_gn_200001-202312.nc"),
        engine="netcdf4",
        lock=lock
    )

indexraster = xr.open_dataset(
        Path(grid_file_location, "ssp_comb_indexraster.nc"),
        engine="netcdf4",
        lock=lock
    )

countrymask = xr.open_dataset(
        Path(grid_file_location, "ssp_comb_countrymask.nc"),
        engine="netcdf4",
        lock=lock
    )

# %%
# assert that dimensions of our new proxy files equal the dimensions of the example proxy file
for dim in example_proxy.dims:
    assert example_proxy.indexes[dim].equals(proxy.indexes[dim]), f"Mismatch in {dim}"

# %%
proxy

# %%
ceds

# %%
indexraster

# %%
countrymask
