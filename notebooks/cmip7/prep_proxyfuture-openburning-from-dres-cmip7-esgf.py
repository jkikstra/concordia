#!/usr/bin/env python
# %% [markdown]
# # Overview
#
# This should create proxies for openburning based on BB4CMIP7
#


# %% [markdown]
# ### Steps happening in this notebook, which starts from download BB4CMIP7 files from ESGF
# 1. Information on how to download this data 
# 1. single-sector files (peat, awb, grassland)
# 1. multiple-sector files (forest burning)
# 1. VOC-speciation (not yet ready)
#
#

# %%

# data grid area:
# gridcellarea: "D:\ESGF\DRES-CMIP-BB4CMIP7-2-1\atmos\fx\gridcellarea\gn\v20250612\gridcellarea_input4MIPs_emissions_CMIP_DRES-CMIP-BB4CMIP7-2-1_gn.nc"
# data emissions:
# emissions: "D:\ESGF\DRES-CMIP-BB4CMIP7-2-1\atmos\mon\BC\gn\v20250612\BC_input4MIPs_emissions_CMIP_DRES-CMIP-BB4CMIP7-2-1_gn_190001-202312.nc"
# percentages: "D:\ESGF\DRES-CMIP-BB4CMIP7-2-1\atmos\mon\BCpercentageTEMF\gn\v20250612\BCpercentageTEMF_input4MIPs_emissions_CMIP_DRES-CMIP-BB4CMIP7-2-1_gn_175001-202312.nc"

from concordia.cmip7.CONSTANTS import GASES, GASES_ESGF_BB4CMIP, CONFIG, PROXY_YEARS

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

from concordia.cmip7 import utils as cmip7_utils
from concordia.settings import Settings
import concordia.cmip7.utils_futureproxy_ceds_bb4cmip as uprox
from concordia.cmip7.utils_futureproxy_ceds_bb4cmip import _normalize_time_slice

# %%
VERSION = CONFIG

# %%
# Select time windows to average over here. 
# Examples:
#   time_slice_totals = 2023                # only 2023
#   time_slice_totals = [2019, 2023]        # full range 2019..2023
# You can use different windows for totals vs percentages if desired.
# time_slice_totals = [2019, 2023]
# time_slice_perc = [2019, 2023]

PROXY_TIME_RANGES = [
    # averages (over a time range)
    # [1994,2023],
    # [2004,2023],

    [2014,2023],
    
    # [2019,2023],
    # single years
    # 2019,2020,2021,2022,2023
]

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

# %% [markdown]
# ### Unsmoothed data

# %%
lock = SerializableLock()

# Workaround for HDF5 on Windows: disable file locking to avoid sporadic read errors
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")


# %%
def get_bb4cmip7_location_percentage(variable):
    return Path(settings.gridding_path) / f"esgf/bb4cmip7/{variable}/gn/v20250612/{variable}_input4MIPs_emissions_CMIP_DRES-CMIP-BB4CMIP7-2-1_gn_175001-202312.nc"

def get_bb4cmip7_location_totals(variable):
    return Path(settings.gridding_path) / f"esgf/bb4cmip7/{variable}/gn/v20250612/{variable}_input4MIPs_emissions_CMIP_DRES-CMIP-BB4CMIP7-2-1_gn_190001-202312.nc"


# what data to load
gfed_sectors_forest = ["BORF", "DEFO", "TEMF"]
forest_fires_name = "FRTB"
gfed_sectors_singlesector = ["AGRI", "PEAT", "SAVA"] 
sector_mapping_singlesector = {
    "AGRI": "AWB",
    "PEAT": "PEAT",
    "SAVA": "GRSB"
}
gas_mapping = {
    "SO2": "Sulfur",
    "NMVOCbulk": "VOC"
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
new_proxies_location = settings.proxy_path
# ensure output directory exists
new_proxies_location.mkdir(parents=True, exist_ok=True)

years = [2022, 2023, 2024, 2025, 2030, 2035, 2040, 2045, 2050, 2055, 2060, 2065, 2070, 2075, 2080, 2085, 2090, 2095, 2100]


# %%
# load CEDS example file to get the right grid settings
template_file = (
    settings.gridding_path
    / "example_files" / "GCAM4-SSP4-34-SPA4-V25-Harmonized-DB-Sulfur-em-aircraft-anthro_input4MIPs_emissions_CMIP_IAMC-V1_gn_201501-210012.nc"
)
template = xr.open_dataset(template_file)


# %% [markdown]
# STANDARD 1-sector proxies (without VOC speciation; not forest)

# %%
# STANDARD 1-sector proxies (without VOC speciation; not forest)

# Create target grid for 0.5 degree resolution
# target_lat = np.arange(-89.75, 90, 0.5)  # 0.5 degree grid from -89.75 to 89.75
# target_lon = np.arange(-179.75, 180, 0.5)  # 0.5 degree grid from -179.75 to 179.75
target_lat = template["lat"].values
target_lon = template["lon"].values

# %%
def load_bb4cmip(g, type, time_slice, s: Optional[str] = None) -> xr.Dataset:
    if type == "total":
        try:
            ds_total = xr.open_dataset(
                get_bb4cmip7_location_totals(g),
                engine="h5netcdf",
                chunks={},
                # chunks={"time": 12},
                lock=lock
            # ).sel(time=slice(start_time, end_time))
            ).sel(time=_normalize_time_slice(time_slice))
        except Exception:
            ds_total = xr.open_dataset(
                get_bb4cmip7_location_totals(g),
                engine="netcdf4",
                chunks={},
                # chunks={"time": 12},
                lock=lock
            # ).sel(time=slice(start_time, end_time))
            ).sel(time=_normalize_time_slice(time_slice))
        return ds_total
    if type == "percentage":
        try:
            ds_perc = xr.open_dataset(
                get_bb4cmip7_location_percentage(f"{g}percentage{s}"),
                engine="h5netcdf",
                chunks={},
                # chunks={"time": 12},
                lock=lock
            # ).sel(time=slice(start_time, end_time))
            ).sel(time=_normalize_time_slice(time_slice))
        except Exception:
            ds_perc = xr.open_dataset(
                get_bb4cmip7_location_percentage(f"{g}percentage{s}"),
                engine="netcdf4",
                chunks={},
                # chunks={"time": 12},
                lock=lock
            # ).sel(time=slice(start_time, end_time))
            ).sel(time=_normalize_time_slice(time_slice))
        return ds_perc
    raise ValueError('type must be "total" or "percentage"')
        


# %%
def formatting_to_cmip7_scenario_proxy(
        ds,
        g,
        s: Optional[str] = None,
        gas_mapping=gas_mapping,
        scenario_years: list[int] = years, #
        ysel: int | list[int] | tuple[int, ...] = 2023,
        sector_override: Optional[str] = None,
        sector_mapping_singlesector=sector_mapping_singlesector
):
    ## do additional formatting (like CEDS workflow)
    # Map ESGF gas names to internal format if mapping exists
    if g in gas_mapping:
        internal_gas = gas_mapping[g]
    else:
        internal_gas = g

    # add gas dimension with internal gas name
    ds = ds.expand_dims(dim={"gas": [f"{internal_gas}"]})
    
    # split time into year and month
    ds = ds.assign_coords(year=("time", ds["time"].dt.year.data),
                            month=("time", ds["time"].dt.month.data)).groupby(["year", "month"]).mean()

    # Take average over the different years
    ds = ds.mean(dim="year")

    # Project onto future years
    ds = ds.expand_dims({"year": scenario_years})
    

    # add sector dimension
    if sector_override is None:
        ds = ds.expand_dims(dim={"sector": [sector_mapping_singlesector[s]] })
    else:
        s = sector_override
        ds = ds.expand_dims(dim={"sector": [s] })
    
    # reorder
    ds_reordered = ds.transpose("lat", "lon", "gas", "sector", "year", "month").chunk({"month": 12})

    # unify chunks for dask
    ds_reordered = ds_reordered.unify_chunks().astype("float32")

    # Format ysel for filename if it's not an integer
    if not isinstance(ysel, int):
        ysel_filename = f"{min(ysel)}_{max(ysel)}"
    else:
        ysel_filename = ysel

    # give path for outfile
    outfile = new_proxies_location / f"openburning_{internal_gas}_{s}_{ysel_filename}.nc"
    
    return ds_reordered, outfile

# %%
# run single-sector
for y in PROXY_TIME_RANGES:
    for g in GASES_ESGF_BB4CMIP:
    # for g in ['BC']:
    # for g in ["SO2", "NMVOCbulk"]:
        
        # import file
        ds_total = load_bb4cmip(g, type="total", time_slice=y).drop_vars(
            # drop variables we don't need and rename the one we need
            ["lat_bnds", "lon_bnds", "time_bnds"]).rename({f"{g}": "emissions"}
        )

        for s in gfed_sectors_singlesector:
            if not isinstance(y, int):
                y_log = f"{min(y)}-{max(y)}"
            else:
                y_log = y
            print(f"Processing {g}_{s} for {y_log}")

            ds_perc = load_bb4cmip(g, type="percentage", time_slice=y, s=s).drop_vars(
                # drop variables we don't need and rename the one we need
                ["lat_bnds", "lon_bnds", "time_bnds"]).rename({f"{g}percentage{s}": "percentage"})

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
                lat=target_lat, lon=target_lon, method='linear'
            ) # TODO: read a bit more on aggregation methods.

            # formatting, including averaging if we do not select just one single year
            ds_reordered, outfile = formatting_to_cmip7_scenario_proxy(
                ds=ds_bb, g=g, gas_mapping=gas_mapping, s=s, ysel=y
            )

            # saving
            if outfile.exists():
                outfile.unlink()

            encoding = {
                var: {"zlib": True, "complevel": 4}
                for var in ds_reordered.data_vars
            }

            with ProgressBar():
                ds_reordered.to_netcdf(outfile, mode="w", engine="h5netcdf", encoding=encoding)


# %% [markdown]
# STANDARD multiple-sector proxies (without VOC speciation; only forest)

# %%
# run multiple-sector proxies (without VOC speciation; only forest)
for y in PROXY_TIME_RANGES:
    for g in GASES_ESGF_BB4CMIP:
    # for g in ['BC']:
    # for g in ["SO2", "NMVOCbulk"]:
        
        if not isinstance(y, int):
            y_log = f"{min(y)}-{max(y)}"
        else:
            y_log = y
        print(f"Processing {g}_{forest_fires_name} for {y_log}")

        # import file 
        ds_total = load_bb4cmip(g, type="total", time_slice=y).drop_vars(
            # drop variables we don't need and rename the one we need
            ["lat_bnds", "lon_bnds", "time_bnds"]).rename({f"{g}": "emissions"}
        )

        # load the multiple forest sectors
        ds_perc1 = load_bb4cmip(g ,type="percentage", time_slice=y, s=gfed_sectors_forest[0]).drop_vars(
            ["lat_bnds", "lon_bnds", "time_bnds"]).rename({f"{g}percentage{gfed_sectors_forest[0]}": "percentage"})
        ds_perc2 = load_bb4cmip(g ,type="percentage", time_slice=y, s=gfed_sectors_forest[1]).drop_vars(
            ["lat_bnds", "lon_bnds", "time_bnds"]).rename({f"{g}percentage{gfed_sectors_forest[1]}": "percentage"})
        ds_perc3 = load_bb4cmip(g ,type="percentage", time_slice=y, s=gfed_sectors_forest[2]).drop_vars(
            ["lat_bnds", "lon_bnds", "time_bnds"]).rename({f"{g}percentage{gfed_sectors_forest[2]}": "percentage"})

        # do multiplication (now backed by NumPy arrays, detached from file IO)
        ds_bb = xr.Dataset({
            "emissions": ds_total["emissions"] * (ds_perc1["percentage"]+ds_perc2["percentage"]+ds_perc3["percentage"]) / 100
        })

        # regrid to 0.5 degree after multiplication (first rename variable names to be like the CEDS template data)
        # print(f"Regridding from {ds_bb.latitude.size}x{ds_bb.longitude.size} to {len(target_lat)}x{len(target_lon)}")
        ds_bb = ds_bb.rename(
            {"latitude":"lat","longitude":"lon"}
        ).interp(
            lat=target_lat, lon=target_lon, method='linear'
        )

        # formatting, including averaging if we do not select just one single year
        ds_reordered, outfile = formatting_to_cmip7_scenario_proxy(
            ds=ds_bb, g=g, gas_mapping=gas_mapping, sector_override=forest_fires_name,
            ysel=y
        )

        # saving
        if outfile.exists():
            outfile.unlink()

        encoding = {
            var: {"zlib": True, "complevel": 4}
            for var in ds_reordered.data_vars
        }

        with ProgressBar():
            ds_reordered.to_netcdf(outfile, mode="w", engine="h5netcdf", encoding=encoding)

