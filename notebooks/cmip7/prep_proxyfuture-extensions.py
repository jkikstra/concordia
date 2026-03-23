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

from concordia.settings import Settings
from concordia.cmip7 import utils as cmip7_utils
from concordia.cmip7.CONSTANTS import PROXY_YEARS

# %% [markdown]
# ## prepare setup

# %%
lock = SerializableLock()

# %%
SETTINGS_FILE: str = "config_cmip7_v0-4-0.yaml" # for second ESGF version
VERSION_ESGF: str = "1-1-0" # for second ESGF version

# Which scenario to run from the markers
marker_to_run: str = "vl" # options: h, hl, m, ml, l, ln, vl

GRIDDING_VERSION: str | None = f"{marker_to_run}_{VERSION_ESGF}"

# %%
grid_file_location = "/Users/hoegner/Projects/CMIP7/input/gridding/"
# grid_file_location = "C:/Users/kikstra/IIASA/ECE.prog - Documents/Projects/CMIP7/IAM Data Processing/concordia_cmip7_esgf_v0_alpha/input/gridding/"

scenario_data_location = Path( "../../results", GRIDDING_VERSION)

new_proxies_location = Path(grid_file_location, "proxy_rasters_extensions", marker_to_run)
new_proxies_location.mkdir(parents=True, exist_ok=True)

# %%
sector_mapping = {0.0: 'AGR', 
                  1.0: 'ENE', 
                  2.0: 'IND', 
                  3.0: 'TRA', 
                  4.0: 'RCO', 
                  5.0: 'SLV', 
                  6.0: 'WST'}

# %% [markdown]
# ## generate proxy rasters

# %%
EXT_PROXY_YEARS = np.arange(2105, 2501, 5)

# %%
# ANTHRO proxies

# Loop through all scenario files
for file in scenario_data_location.glob("*.nc"):

    if "em-anthro" in str(file):
        print(f"Processing {file}")

        # Extract species from filename
        species = file.stem.split("-")[0]

        # Open dataset with fixed numeric chunks (safe for object dtypes)
        ds = xr.open_dataset(
            file,
            engine="netcdf4",
            chunks={},  # adjust for your RAM / dataset
            lock=lock
        )

        # Drop variables we don't need
        drop_vars = ["lat_bnds", "lon_bnds", "time_bnds", "sector_bnds"]
        ds = ds.drop_vars([v for v in drop_vars if v in ds.data_vars])
        
        # Slice only year 2100 first to reduce size
        ds = ds.sel(time=ds["time"].dt.year == 2100)

      #  print(ds)

        # Rename main emissions variable
        var_name = f"{species}_em_anthro"
        if var_name in ds.data_vars:
            ds = ds.rename({var_name: "emissions"})
        else:
            # fallback to first variable
            ds = ds.rename({list(ds.data_vars)[0]: "emissions"})
            
        # Drop sectors >= 7
        ds = ds.sel(sector=ds["sector"] < 7)

        # Drop all attributes
        ds.attrs = {}
        for var in ds.data_vars:
            ds[var].attrs = {}

        # Add gas dimension
        ds = ds.expand_dims({"gas": [species]})
        
        # rename NMVOC to VOC    
        if species == "NMVOC":
            ds = ds.assign_coords(gas=["VOC"])
            species = "VOC"
            
        # rename SO2 to Sulfur
        if species == "SO2":
            ds = ds.assign_coords(gas=["Sulfur"])
            species = "Sulfur"
        
        # Split time into year/month and compute monthly mean
        ds = (
            ds.assign_coords(
                year=("time", ds["time"].dt.year.data),
                month=("time", ds["time"].dt.month.data)
            )
            .groupby(["year", "month"])
            .mean()
        )

        # Project to future years
        ds = ds.sel(year=2100).expand_dims({"year": EXT_PROXY_YEARS})
        
        # Rename sectors using mapping
        ds = ds.assign_coords(sector=ds["sector"].to_series().replace(sector_mapping).values)

        # Transpose to final dimension order and chunk lazily
        ds_reordered = ds.transpose("lat", "lon", "gas", "sector", "year", "month").chunk({"month": 12})

        # Output file
        outfile = new_proxies_location / f"anthro_{species}.nc"
        # Skip if already processed
        if outfile.exists():
            print(f"Skipping {file} (already exists)")
            continue
            
# or, if we want to override
#            outfile.unlink()


        # Encoding for NetCDF compression
        encoding = {var: {"zlib": True, "complevel": 4, "dtype": "float32"} for var in ds_reordered.data_vars}

        # Write lazily to NetCDF with progress bar
        with ProgressBar():
            ds_reordered.to_netcdf(outfile, mode="w", encoding=encoding, compute=True)

        # Free memory
        del ds, ds_reordered

# %%
# SHIPPING proxies

# Loop through all scenario files
for file in scenario_data_location.glob("*.nc"):

    if "em-anthro" in str(file):
        print(f"Processing {file}")

        # Extract species from filename
        species = file.stem.split("-")[0]

        # Open dataset with fixed numeric chunks (safe for object dtypes)
        ds = xr.open_dataset(
            file,
            engine="netcdf4",
            chunks={},  # adjust for your RAM / dataset
            lock=lock
        )

        # Drop variables we don't need
        drop_vars = ["lat_bnds", "lon_bnds", "time_bnds", "sector_bnds"]
        ds = ds.drop_vars([v for v in drop_vars if v in ds.data_vars])
        
        # Slice only year 2100 first to reduce size
        ds = ds.sel(time=ds["time"].dt.year == 2100)

        # Rename main emissions variable
        var_name = f"{species}_em_anthro"
        if var_name in ds.data_vars:
            ds = ds.rename({var_name: "emissions"})
        else:
            # fallback to first variable
            ds = ds.rename({list(ds.data_vars)[0]: "emissions"})
            
        # select shipping sector
        ds = ds.sel(sector=ds["sector"]==7)

        # Drop all attributes
        ds.attrs = {}
        for var in ds.data_vars:
            ds[var].attrs = {}

        # Add gas dimension
        ds = ds.expand_dims({"gas": [species]})

        # rename NMVOC to VOC    
        if species == "NMVOC":
            ds = ds.assign_coords(gas=["VOC"])
            species = "VOC"
            
        # rename SO2 to Sulfur
        if species == "SO2":
            ds = ds.assign_coords(gas=["Sulfur"])
            species = "Sulfur"
            
        # Split time into year/month and compute monthly mean
        ds = (
            ds.assign_coords(
                year=("time", ds["time"].dt.year.data),
                month=("time", ds["time"].dt.month.data)
            )
            .groupby(["year", "month"])
            .mean()
        )

        # Project to future years
        ds = ds.sel(year=2100).expand_dims({"year": EXT_PROXY_YEARS})
        
        # rename shipping sector and reorder dimensions
        ds = ds.assign_coords(sector=("sector", ["SHP" if v == 7 else v for v in ds["sector"].values]))    

        # Transpose to final dimension order and chunk lazily
        ds_reordered = ds.transpose("lat", "lon", "gas", "sector", "year", "month").chunk({"month": 12})

        # Output file
        outfile = new_proxies_location / f"shipping_{species}.nc"

        # Skip if already processed
        if outfile.exists():
            print(f"Skipping {file} (already exists)")
            continue
            
# or, if we want to override
#            outfile.unlink()

        # Encoding for NetCDF compression
        encoding = {var: {"zlib": True, "complevel": 4, "dtype": "float32"} for var in ds_reordered.data_vars}

        # Write lazily to NetCDF with progress bar
        with ProgressBar():
            ds_reordered.to_netcdf(outfile, mode="w", encoding=encoding, compute=True)

        # Free memory
        del ds, ds_reordered

# %%
# AIRCRAFT proxies

# loop through all CEDS em-AIR-anthro from input4MIP files
for file in scenario_data_location.glob("*.nc"):
    if "em-AIR-anthro" in str(file):
        # Process the file
        print(f"Processing {file}")
            
        # extract species information from filename
        species = file.stem.split("-")[0]
    
        # import file 
        ds = xr.open_dataset(
            file,
            engine="netcdf4",
            chunks={},
            lock=lock
        )
        
        # Drop variables we don't need
        drop_vars = ["lat_bnds", "lon_bnds", "time_bnds", "level_bnds"]
        ds = ds.drop_vars([v for v in drop_vars if v in ds.data_vars])
        
        # Slice only year 2100 first to reduce size
        ds = ds.sel(time=ds["time"].dt.year == 2100)

        # Rename main emissions variable
        var_name = f"{species}_em_AIR_anthro"
        if var_name in ds.data_vars:
            ds = ds.rename({var_name: "emissions"})
        else:
            # fallback to first variable
            ds = ds.rename({list(ds.data_vars)[0]: "emissions"})

        # Drop all attributes
        ds.attrs = {}
        for var in ds.data_vars:
            ds[var].attrs = {}

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
            
        # Split time into year/month and compute monthly mean
        ds = (
            ds.assign_coords(
                year=("time", ds["time"].dt.year.data),
                month=("time", ds["time"].dt.month.data)
            )
            .groupby(["year", "month"])
            .mean()
        )

        # add sector dimension
        ds = ds.expand_dims(dim={"sector": ["AIR"]})
            
        # select 2100 data and project it onto future years
        ds = ds.sel(year=2100).expand_dims({"year": EXT_PROXY_YEARS})
    
        # Transpose to final dimension order and chunk lazily
        ds_reordered = ds.transpose("lat", "lon", "level", "gas", "sector", "year", "month").chunk({"month": 12})

        outfile = new_proxies_location / f"aircraft_{species}.nc"
   
        # Skip if already processed
        if outfile.exists():
            print(f"Skipping {file} (already exists)")
            continue
            
# or, if we want to override
#            outfile.unlink()

        encoding = {
            var: {"zlib": True, "complevel": 4, "dtype": "float32"}
            for var in ds_reordered.data_vars
        }
        
        with ProgressBar():
            ds_reordered.to_netcdf(outfile, mode="w", encoding=encoding)

        # Free memory
        del ds, ds_reordered

# %%
# OPENBURNING proxies

# loop through all gridded em-openburning scenario files
for file in scenario_data_location.glob("*.nc"):

    if "em-openburning" in str(file):
        # Process the file
        print(f"Processing {file}")
        
        # extract species information from filename
        species = file.stem.split("-")[0]
    
        # import file 
        ds = xr.open_dataset(
            file,
            engine="netcdf4",
            chunks={},
            lock=lock
        )
        
        # add gas dimension
        ds = ds.expand_dims(dim={"gas": [f"{species}"]})
    
        # split time into year and month
        ds = ds.assign_coords(year=("time", ds["time"].dt.year.data), month=("time", ds["time"].dt.month.data)).groupby(["year", "month"]).mean()
    
        # select 2100 data and project it onto future years
        ds = ds.sel(year=2100).expand_dims({"year": EXT_PROXY_YEARS})
    
        outfile = new_proxies_location / f"openburning_{species}.nc"
        if outfile.exists():
            outfile.unlink()
    
        encoding = {
            var: {"zlib": True, "complevel": 4, "dtype": "float32"}
            for var in ds.data_vars
        }
        
        with ProgressBar():
            ds.to_netcdf(outfile, mode="w", encoding=encoding, compute=True)
