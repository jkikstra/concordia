#!/usr/bin/env python
# %% [markdown]
# # Overview
#
# This should create proxies for openburning based on BB4CMIP7
# 


# %% [markdown]
# ### Steps happening in this notebook, which starts from the PNNL server .Rd files
# 1. ...



# %% 

# data grid area:
# gridcellarea: "D:\ESGF\DRES-CMIP-BB4CMIP7-2-1\atmos\fx\gridcellarea\gn\v20250612\gridcellarea_input4MIPs_emissions_CMIP_DRES-CMIP-BB4CMIP7-2-1_gn.nc"
# data emissions:
# emissions: "D:\ESGF\DRES-CMIP-BB4CMIP7-2-1\atmos\mon\BC\gn\v20250612\BC_input4MIPs_emissions_CMIP_DRES-CMIP-BB4CMIP7-2-1_gn_190001-202312.nc"
# percentages: "D:\ESGF\DRES-CMIP-BB4CMIP7-2-1\atmos\mon\BCpercentageTEMF\gn\v20250612\BCpercentageTEMF_input4MIPs_emissions_CMIP_DRES-CMIP-BB4CMIP7-2-1_gn_175001-202312.nc"

from concordia.cmip7.CONSTANTS import GASES, CONFIG

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
from dask.diagnostics import ProgressBar
from dask.utils import SerializableLock
import seaborn as sns

from concordia.cmip7 import utils as cmip7_utils

# %%
VERSION = CONFIG
VERSION = "config_cmip7_v0_3.yaml"

# %% [markdown]
# ### Unsmoothed data

# %%
lock = SerializableLock()

# Workaround for HDF5 on Windows: disable file locking to avoid sporadic read errors
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")


# %%
# TODO:
# - mapping and summing over that mapping (after singlesector ones work)

# def get_bb4cmip7_location_percentage(variable):
#     return f"D:/ESGF/DRES-CMIP-BB4CMIP7-2-1/atmos/mon/{variable}/gn/v20250612/{variable}_input4MIPs_emissions_CMIP_DRES-CMIP-BB4CMIP7-2-1_gn_175001-202312.nc"

# def get_bb4cmip7_location_totals(variable):
#     return f"D:/ESGF/DRES-CMIP-BB4CMIP7-2-1/atmos/mon/{variable}/gn/v20250612/{variable}_input4MIPs_emissions_CMIP_DRES-CMIP-BB4CMIP7-2-1_gn_190001-202312.nc"

def get_bb4cmip7_location_percentage(variable):
    return f"C:/Users/kikstra/IIASA/ECE.prog - Documents/Projects/CMIP7/IAM Data Processing/concordia_cmip7_v0_3/input/gridding/esgf/{variable}/gn/v20250612/{variable}_input4MIPs_emissions_CMIP_DRES-CMIP-BB4CMIP7-2-1_gn_175001-202312.nc"

def get_bb4cmip7_location_totals(variable):
    return f"C:/Users/kikstra/IIASA/ECE.prog - Documents/Projects/CMIP7/IAM Data Processing/concordia_cmip7_v0_3/input/gridding/esgf/{variable}/gn/v20250612/{variable}_input4MIPs_emissions_CMIP_DRES-CMIP-BB4CMIP7-2-1_gn_190001-202312.nc"


# what data to load
gfed_sectors_forest = ["BORF", "DEFO", "TEMF"]
forest_fires_name = "FRTB"
gfed_sectors_singlesector = ["AGRI", "PEAT", "SAVA"] # let's first do without forest burning
sector_mapping_singlesector = {
    "AGRI": "AWB",
    "PEAT": "PEAT",
    "SAVA": "GRSB"
}



perc_combinations_forest = [f"{g}percentage{sec}" for g in GASES for sec in gfed_sectors_forest]
perc_combinations_singlesector = [f"{g}percentage{sec}" for g in GASES for sec in gfed_sectors_singlesector]
totals_combinations = [f"{g}" for g in GASES]

# input data filepaths
# note: adjust the paths below as required by where you placed your downloaded data
perc_file_paths_singlesector = [get_bb4cmip7_location_percentage(variable) for variable in perc_combinations_singlesector]
totals_file_paths = [get_bb4cmip7_location_totals(variable) for variable in totals_combinations]
print(len(totals_file_paths) + len(perc_file_paths_singlesector))

# %%
# what data to output
grid_file_location = "C:/Users/kikstra/IIASA/ECE.prog - Documents/Projects/CMIP7/IAM Data Processing/concordia_cmip7_v0_3/input/gridding/"
new_proxies_location = Path(grid_file_location, "proxy_rasters")

years = [2023, 2024, 2025, 2030, 2035, 2040, 2045, 2050, 2055, 2060, 2065, 2070, 2075, 2080, 2085, 2090, 2095, 2100]

# %% 
# options for the proxy: time slices (based on example file)

# totals 
bc_file = get_bb4cmip7_location_totals("BC")

example_ds = xr.open_dataset(
            bc_file,
            engine="netcdf4",
            chunks={},
            lock=lock
        )

time_filter_5 = (example_ds.time.dt.year >= 2019) & (example_ds.time.dt.year <= 2023) # 5yr filter
time_filter_10 = (example_ds.time.dt.year >= 2014) & (example_ds.time.dt.year <= 2023) # 10yr filter
time_filter_20 = (example_ds.time.dt.year >= 2004) & (example_ds.time.dt.year <= 2023) # 20yr filter
time_filter_30 = (example_ds.time.dt.year >= 1994) & (example_ds.time.dt.year <= 2023) # 30yr filter
print( time_filter_30.sum() / 12 )

time_filter_totals = time_filter_5

# percentages
bc_file = get_bb4cmip7_location_percentage("BCpercentageSAVA")

example_ds = xr.open_dataset(
            bc_file,
            engine="netcdf4",
            chunks={},
            lock=lock
        )

time_filter_5 = (example_ds.time.dt.year >= 2019) & (example_ds.time.dt.year <= 2023) # 5yr filter
time_filter_10 = (example_ds.time.dt.year >= 2014) & (example_ds.time.dt.year <= 2023) # 10yr filter
time_filter_20 = (example_ds.time.dt.year >= 2004) & (example_ds.time.dt.year <= 2023) # 20yr filter
time_filter_30 = (example_ds.time.dt.year >= 1994) & (example_ds.time.dt.year <= 2023) # 30yr filter
print( time_filter_30.sum() / 12 )

time_filter_perc = time_filter_5


# %%
from concordia.settings import Settings

import concordia.cmip7.utils_futureproxy_ceds_bb4cmip as uprox

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

# load CEDS example file to get the right grid settings

template_file = (
    settings.gridding_path
    / "example_files" / "GCAM4-SSP4-34-SPA4-V25-Harmonized-DB-Sulfur-em-aircraft-anthro_input4MIPs_emissions_CMIP_IAMC-V1_gn_201501-210012.nc"
)
template = xr.open_dataset(template_file)


# %%
# STANDARD 1-sector proxies (without VOC speciation; not forest)

# TODO: [ ] mean across years
# TODO: [~] regrid to 0.5 degree

# Create target grid for 0.5 degree resolution
# target_lat = np.arange(-89.75, 90, 0.5)  # 0.5 degree grid from -89.75 to 89.75
# target_lon = np.arange(-179.75, 180, 0.5)  # 0.5 degree grid from -179.75 to 179.75
target_lat = template["lat"].values
target_lon = template["lon"].values



from dask import config as dask_config

# safer time window selection using slicing (avoids fancy/boolean indexing at backend level)
# start_time = "2019-01-01"
# end_time = "2023-12-31"


for g in GASES:
# for g in ['BC']:
    
    # import file 
    try:
        ds_total = xr.open_dataset(
            get_bb4cmip7_location_totals(g),
            engine="h5netcdf",
            chunks={"time": 12},
            lock=lock
        # ).sel(time=slice(start_time, end_time))
        ).sel(time=time_filter_totals)
    except Exception:
        ds_total = xr.open_dataset(
            get_bb4cmip7_location_totals(g),
            engine="netcdf4",
            chunks={"time": 12},
            lock=lock
        # ).sel(time=slice(start_time, end_time))
        ).sel(time=time_filter_totals)

    # drop variables we don't need and rename the one we need
    ds_total = ds_total.drop_vars(["lat_bnds", "lon_bnds", "time_bnds"]).rename({f"{g}": "emissions"})

    for s in gfed_sectors_singlesector:
        # open with fallback engine if first attempt fails (some files behave better with netcdf4)
        try:
            ds_perc = xr.open_dataset(
                get_bb4cmip7_location_percentage(f"{g}percentage{s}"),
                engine="h5netcdf",
                chunks={"time": 12},
                lock=lock
            # ).sel(time=slice(start_time, end_time))
            ).sel(time=time_filter_perc)
        except Exception:
            ds_perc = xr.open_dataset(
                get_bb4cmip7_location_percentage(f"{g}percentage{s}"),
                engine="netcdf4",
                chunks={"time": 12},
                lock=lock
            # ).sel(time=slice(start_time, end_time))
            ).sel(time=time_filter_perc)
        
        # drop variables we don't need and rename the one we need
        ds_perc = ds_perc.drop_vars(["lat_bnds", "lon_bnds", "time_bnds"]).rename({f"{g}percentage{s}": "percentage"})

        # # load inputs to memory in a single-threaded context to avoid HDF read errors
        # with dask_config.set(scheduler="single-threaded"):
        #     total_emissions = ds_total["emissions"].load()
        #     percentage = ds_perc["percentage"].load()

        # # release file handles early
        # try:
        #     ds_total.close()
        # except Exception:
        #     pass
        # try:
        #     ds_perc.close()
        # except Exception:
        #     pass

        # do multiplication (now backed by NumPy arrays, detached from file IO)
        ds_bb = xr.Dataset({
            # "emissions": total_emissions * percentage / 100
            "emissions": ds_total["emissions"] * ds_perc["percentage"] / 100
        })

        # regrid to 0.5 degree after multiplication (first rename variable names to be like the CEDS template data)
        print(f"Regridding from {ds_bb.latitude.size}x{ds_bb.longitude.size} to {len(target_lat)}x{len(target_lon)}")
        ds_bb = ds_bb.rename(
            {"latitude":"lat","longitude":"lon"}
        ).interp(
            lat=template["lat"], lon=template["lon"], method='linear'
        )


        ## do additional formatting (like CEDS workflow)

        # add gas dimension
        ds_bb = ds_bb.expand_dims(dim={"gas": [f"{g}"]})
        
        # split time into year and month
        ds_bb = ds_bb.assign_coords(year=("time", ds_bb["time"].dt.year.data), 
                              month=("time", ds_bb["time"].dt.month.data)).groupby(["year", "month"]).mean()

        # select 2023 data and project it onto future years
        ds_bb = ds_bb.sel(year=2023).expand_dims({"year": years})

        # add sector dimension
        ds_bb = ds_bb.expand_dims(dim={"sector": [sector_mapping_singlesector[s]] })
        
        # reorder
        ds_reordered = ds_bb.transpose("lat", "lon", "gas", "sector", "year", "month").chunk({"month": 12})

        # unify chunks for dask
        ds_reordered = ds_reordered.unify_chunks().astype("float32")


        # ensure output directory exists
        new_proxies_location.mkdir(parents=True, exist_ok=True)

        outfile = new_proxies_location / f"openburning_{g}_{s}_2023.nc"
        if outfile.exists():
            outfile.unlink()

        encoding = {
            var: {"zlib": True, "complevel": 4}
            for var in ds_reordered.data_vars
        }

        with ProgressBar():
            ds_reordered.to_netcdf(outfile, mode="w", engine="h5netcdf", encoding=encoding)



# %% 



