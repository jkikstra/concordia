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
from concordia.cmip7.CONSTANTS import PROXY_YEARS

# %% [markdown]
# ## prepare setup

# %%
lock = SerializableLock()

# %%
grid_file_location = "/Users/hoegner/Projects/CMIP7/input/gridding/"
# grid_file_location = "C:/Users/kikstra/IIASA/ECE.prog - Documents/Projects/CMIP7/IAM Data Processing/concordia_cmip7_esgf_v0_alpha/input/gridding/"

ceds_data_location = Path(grid_file_location, "esgf", "ceds", "CMIP7_anthro")
ceds_air_data_location = Path(grid_file_location, "esgf", "ceds", "CMIP7_AIR")
new_proxies_location = Path(grid_file_location, "proxy_rasters")
new_proxies_location.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## generate proxy rasters

# %%
years = PROXY_YEARS

sector_mapping = {
    0: "AGR",
    1: "ENE",
    2: "IND",
    3: "TRA",
    4: "RCO",
    5: "SLV",
    6: "WST"
}

sector_mapping_shp = {
    7: "SHP"
}

# %%
# ANTHRO proxies

# loop through all CEDS em-anthro from input4MIP files
for file in ceds_data_location.glob("*.nc"):

    # extract species information from filename
    species = file.stem.split("-")[0]

    # import file 
    ds = xr.open_dataset(
        file,
        engine="netcdf4",
        chunks={},
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

    encoding = {
        var: {"zlib": True, "complevel": 4, "dtype": "float32"}
        for var in ds_reordered.data_vars
    }
    
    with ProgressBar():
        ds_reordered.to_netcdf(outfile, mode="w", encoding=encoding, compute=True)

# %%
# SHIPPING proxies

# loop through all CEDS em-anthro from input4MIP files
for file in ceds_data_location.glob("*.nc"):

    # extract species information from filename
    species = file.stem.split("-")[0]

    # import file 
    ds = xr.open_dataset(
        file,
        engine="netcdf4",
        chunks={},
        lock=lock
    )

    # drop variables we don't need and rename the one we need
    ds = ds.drop_vars(["lat_bnds", "lon_bnds", "time_bnds", "sector_bnds"]).rename({f"{species}_em_anthro": "emissions"})

    # retain only SHP sector
    ds = ds.sel(sector=7)

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

    # rename shipping sector and reorder dimensions
    ds = ds.expand_dims(sector=[ds.sector.item()])
    ds = ds.assign_coords(sector=("sector", ["SHP" if v == 7 else v for v in ds["sector"].values]))    


    ds_reordered = xr.Dataset(
    {var: (("lat", "lon", "gas", "sector", "year", "month"), ds[var].transpose("lat", "lon", "gas", "sector", "year", "month").values)
     for var in ds.data_vars},
    coords={dim: ds[dim] for dim in ["lat", "lon", "gas", "sector", "year", "month"]}
    ).chunk({"month": 12})

    outfile = new_proxies_location / f"shipping_{species}.nc"
    if outfile.exists():
        outfile.unlink()
        
    encoding = {
        var: {"zlib": True, "complevel": 4, "dtype": "float32"}
        for var in ds_reordered.data_vars
    }

    with ProgressBar():
        ds_reordered.to_netcdf(outfile, mode="w", encoding=encoding, compute=True)

# %%
# AIRCRAFT proxies

# loop through all CEDS em-AIR-anthro from input4MIP files
for file in ceds_air_data_location.glob("*.nc"):

    # extract species information from filename
    species = file.stem.split("-")[0]

    # import file 
    ds = xr.open_dataset(
        file,
        engine="netcdf4",
        chunks={},
        lock=lock
    )
    
    # drop variables we don't need and rename the one we need
    ds = ds.drop_vars(["lat_bnds", "lon_bnds", "time_bnds"]).rename({f"{species}_em_AIR_anthro": "emissions"})

    # add gas dimension
    ds = ds.expand_dims(dim={"gas": [f"{species}"]})

    # add sector dimension
    ds = ds.expand_dims(dim={"sector": ["AIR"]})
    
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

    # reorder dimensions
    ds_reordered = xr.Dataset(
    {var: (("lat", "lon", "level", "gas", "sector", "year", "month"), ds[var].transpose("lat", "lon", "level", "gas", "sector", "year", "month").values)
     for var in ds.data_vars},
    coords={dim: ds[dim] for dim in ["lat", "lon", "level", "gas", "sector", "year", "month"]}
    ).chunk({"month": 12})

    outfile = new_proxies_location / f"aircraft_{species}.nc"
    if outfile.exists():
        outfile.unlink()
        
    encoding = {
        var: {"zlib": True, "complevel": 4, "dtype": "float32"}
        for var in ds_reordered.data_vars
    }
    
    with ProgressBar():
        ds_reordered.to_netcdf(outfile, mode="w", encoding=encoding)
