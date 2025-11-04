#!/usr/bin/env python
# %% [markdown]
# # Overview
#
# This should create percentage share proxies for openburning VOC species based on BB4CMIP7


# %% [markdown]
# ### Steps happening in this notebook
# 1. calculate 10-y average VOC-speciation single-sector proxies (peat, awb, grassland)
# 1. calculate 10-y average VOC-speciation multiple-sector proxies (forest burning)
# 1. load corresponding 10-y average VOC totals
# 1. calculate percentage for each grid cell
# 1. write out percentage to proxy_rasters folder as input to workflow

# %%
from concordia.cmip7.CONSTANTS import GASES, GASES_ESGF_BB4CMIP_VOC, GASES_ESGF_BB4CMIP, CONFIG, PROXY_YEARS

# %%
# GASES_ESGF_BB4CMIP_VOC = ["C2H2", "C10H16"] # override to only test for two species first


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
PROXY_YEAR = 2023

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
cmip7_dir = settings.gridding_path / "esgf" / "bb4cmip7_voc"

# %%
settings.gridding_path

# %% [markdown]
# ### Unsmoothed data

# %%
lock = SerializableLock()

# Workaround for HDF5 on Windows: disable file locking to avoid sporadic read errors
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")


# %%
# Load VOC bulk data
def get_bb4cmip7_voc_bulk_location(variable):
    return Path(settings.gridding_path) / f"esgf/bb4cmip7/{variable}/gn/v20250612/{variable}_input4MIPs_emissions_CMIP_DRES-CMIP-BB4CMIP7-2-1_gn_190001-202312.nc"

def get_bb4cmip7_voc_bulk_percentage_location(variable):
    return Path(settings.gridding_path) / f"esgf/bb4cmip7/{variable}/gn/v20250612/{variable}_input4MIPs_emissions_CMIP_DRES-CMIP-BB4CMIP7-2-1_gn_175001-202312.nc"

# Load VOC speciated data
def get_bb4cmip7_location_percentage(variable):
    return Path(settings.gridding_path) / f"esgf/bb4cmip7_voc/{variable}/gn/v20250612/{variable}_input4MIPs_emissions_CMIP_DRES-CMIP-BB4CMIP7-2-1_gn_175001-202312.nc"

def get_bb4cmip7_location_totals(variable):
    return Path(settings.gridding_path) / f"esgf/bb4cmip7_voc/{variable}/gn/v20250612/{variable}_input4MIPs_emissions_CMIP_DRES-CMIP-BB4CMIP7-2-1_gn_190001-202312.nc"

# what data to load
gfed_sectors_forest = ["BORF", "DEFO", "TEMF"]
forest_fires_name = "FRTB"
gfed_sectors_singlesector = ["AGRI", "PEAT", "SAVA"]
sector_mapping_singlesector = {
    "AGRI": "AWB",
    "PEAT": "PEAT",
    "SAVA": "GRSB"
}

# perc_combinations_forest = [f"{g}percentage{sec}" for g in GASES_ESGF_BB4CMIP_VOC for sec in gfed_sectors_forest] # WHY IS THIS NOT USED ANYWHERE?
perc_combinations_singlesector = [f"{g}percentage{sec}" for g in GASES_ESGF_BB4CMIP_VOC for sec in gfed_sectors_singlesector]
totals_combinations = [f"{g}" for g in GASES_ESGF_BB4CMIP_VOC]

# input data filepaths
# note: adjust the paths below as required by where you placed your downloaded data
perc_file_paths_singlesector = [get_bb4cmip7_location_percentage(variable) for variable in perc_combinations_singlesector]
totals_file_paths = [get_bb4cmip7_location_totals(variable) for variable in totals_combinations]
print(len(totals_file_paths) + len(perc_file_paths_singlesector))

# %%
settings.proxy_path

# %%
# what data to output
new_proxies_location = settings.proxy_path
# ensure output directory exists
#new_proxies_location.mkdir(parents=True, exist_ok=True)

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
target_lat = template["lat"].values
target_lon = template["lon"].values

# %%
def formatting_to_cmip7_scenario_proxy(
        ds,
        g,
        scenario_years: list[int] = years,
        ysel: int | list[int] | tuple[int, ...] = 2023,
        sector_override: Optional[str] = None
):
    if isinstance(ds, xr.DataArray):
        ds = ds.to_dataset(name=ds.name)
    
    ## do additional formatting (like CEDS workflow)
    ds = ds.expand_dims(dim={"gas": [f"{g}"]})
    
    # split time into year and month
    ds = ds.assign_coords(year=("time", ds["time"].dt.year.data),
                            month=("time", ds["time"].dt.month.data)).groupby(["year", "month"]).mean()

    # Take average over the different years
    ds = ds.mean(dim="year")

    # Project onto future years
    ds = ds.expand_dims({"year": scenario_years})
    
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
    outfile = new_proxies_location / "NMVOC_speciation" / f"{g}_other_voc_em_speciated_NMVOC_openburning_{ysel_filename}.nc"
    
    return ds_reordered, outfile


# # %%
# # run single-sector for NMVOCbulk

# interim_totals_bb = {}

# g = "NMVOCbulk"

# # import file
# ds_total = xr.open_dataset(
#     get_bb4cmip7_voc_bulk_location(
#         variable = g
#     ),
#     engine="h5netcdf",
#     chunks={},
#     lock=lock
# )

# # %%
# for s in gfed_sectors_singlesector:
#     # here we calculate the absolute emissions per VOC species and per sector
#     # by multiplying the total emissions for the respective species 
#     # with the percentage allocated to the respective sector
    
#     ds_perc = xr.open_dataset(
#             get_bb4cmip7_voc_bulk_percentage_location(
#                 variable = f"{g}percentage{s}"
#             ),
#             engine="h5netcdf",
#             chunks={},
#             lock=lock
#         ).sel(time=_normalize_time_slice(PROXY_YEAR)).drop_vars(
#         # drop variables we don't need and rename the one we need
#         ["lat_bnds", "lon_bnds", "time_bnds"]).rename({f"{g}percentage{s}": "percentage"})

#     # do multiplication (now backed by NumPy arrays, detached from file IO)
#     ds_bb = xr.Dataset({
#         # "emissions": total_emissions * percentage / 100
#         "emissions": ds_total["emissions"] * ds_perc["percentage"] / 100
#     })

#     interim_totals_bb[(s)] = ds_bb

# # %%
# interim_forest_bb = {}

# for s in gfed_sectors_forest:
#     # here we calculate the absolute emissions per VOC species and per sector
#     # by multiplying the total emissions for the respective species 
#     # with the percentage allocated to the respective sector
    
#     ds_perc = xr.open_dataset(
#             get_bb4cmip7_voc_bulk_percentage_location(
#                 variable = f"{g}percentage{s}"
#             ),
#             engine="h5netcdf",
#             chunks={},
#             lock=lock
#         ).sel(time=_normalize_time_slice(PROXY_YEAR)).drop_vars(
#         # drop variables we don't need and rename the one we need
#         ["lat_bnds", "lon_bnds", "time_bnds"]).rename({f"{g}percentage{s}": "percentage"})

#     interim_forest_bb[(s)] = ds_perc

        
# # do multiplication (now backed by NumPy arrays, detached from file IO)
# ds_bb = xr.Dataset({
#     # "emissions": total_emissions * percentage / 100
#     "emissions": ds_total["emissions"] * (
#         interim_forest_bb["BORF"]["percentage"] +
#         interim_forest_bb["DEFO"]["percentage"] +
#         interim_forest_bb["TEMF"]["percentage"])/ 100
# })

# interim_totals_bb["FRTB"] = ds_bb

# # %%
# total_perc = (interim_totals_bb["AGRI"]["emissions"] +
#               interim_totals_bb["SAVA"]["emissions"] +
#               interim_totals_bb["PEAT"]["emissions"] +
#               interim_totals_bb["FRTB"]["emissions"]) / ds_total["emissions"] * 100
# print(total_perc.min().compute().values, total_perc.max().compute().values)
# # range of values: 99.999985 100.00002 (which looks like an acceptable deviation)

# # %%
# ds_all = xr.concat(
#     [ds["emissions"] for ds in interim_totals_bb.values()],
#     dim="sector"
# )

# # label the sector dimension
# ds_all = ds_all.assign_coords(sector=list(interim_totals_bb.keys()))
# ds_all

# # %%
# interim_totals_bb.keys()

# %% [markdown]
# ## input

# # %%
# HARMONIZATION_VERSION = "joy-ride-H"
# MODEL_SELECTION = "GCAM 8s"
# SCENARIO_SELECTION = "SSP3 - High Emissions"

# # %%
# # openburning
# voc_openburning = xr.open_dataset(
#         settings.out_path / HARMONIZATION_VERSION / "{name}_{activity_id}_emissions_{target_mip}_{institution}-{model}-{scenario}_{grid_label}_{start_date}-{end_date}.nc".format(
#     name="VOC-em-openburning",
#     model=MODEL_SELECTION.replace(" ", "-"),
#     scenario=SCENARIO_SELECTION.replace(" ", "-"),
#     **cmip7_utils.DS_ATTRS | {"version": settings.version}
# ),
# chunks={},
# lock=lock
# )

# # %%
# voc_openburning.sector

# %% [markdown]
# ## derive speciated VOC share of NMVOCbulk
#
# this is done by VOC species, for the totals only, no sectoral information needed. should give us as many grid-cell level scalars as there is VOC species; these can then be multiplied with the sectoral percentages in a next step to obtain the speciation "proxies"

# %%
GASES_ESGF_BB4CMIP_VOC

# %%
# GASES_ESGF_BB4CMIP_VOC_test = ["C2H2", "C10H16"]

# %%
# 1. load NMVOCbulk openburning total, retain only 2023 values, do some renaming
# this we need for all speciated gases as a reference, so we can do it outside the loop
nmvoc_bulk_total = xr.open_dataset(
    get_bb4cmip7_voc_bulk_location(
        variable = "NMVOCbulk"
    ),
    engine="h5netcdf",
    chunks={},
    lock=lock
).sel(time=_normalize_time_slice(PROXY_YEAR)).drop_vars(
    # drop variables we don't need and rename the one we need
    ["lat_bnds", "lon_bnds", "time_bnds"]).rename({f"NMVOCbulk": "emissions"}
)
# %%

interim_totals_bb = {}
interim_forest_bb = {}
# voc_shares = {} # if I want to do checks below

# 2. loop over all VOC species
for g in [GASES_ESGF_BB4CMIP_VOC[17]]:
# for g in GASES_ESGF_BB4CMIP_VOC:
    
    # 2.1 load respective species totals, renaming as above
    ds_total = xr.open_dataset(
        get_bb4cmip7_location_totals(
            variable = f"{g}"
        ),
        engine="h5netcdf",
        chunks={},
        lock=lock
    ).sel(time=_normalize_time_slice(PROXY_YEAR)).drop_vars( # fixed to one year
            # drop variables we don't need and rename the one we need
            ["lat_bnds", "lon_bnds", "time_bnds"]).rename({f"{g}": "emissions"})

    
    # 2.2 calculate grid-level speciated VOC shares of NMVOCbulk
    ds_shares = xr.Dataset({
        # e.g., how much is "CH3OH" of the total NMVOCbulk
        "emissions_share": xr.where(nmvoc_bulk_total["emissions"] !=0, ds_total["emissions"] / nmvoc_bulk_total["emissions"], 0)
    })

#     # This somehow does not add up to 1 for each gridcell? Or does it? 
#     # Let's check (1: save the shares)
#     voc_shares[(g)] = ds_shares
# # Let's check (2: plot the shares as a map)
# voc_shares_combined = xr.concat([voc_shares[g].isel(time=0).emissions_share for g in GASES_ESGF_BB4CMIP_VOC], dim='gas')
# plot_map(voc_shares_combined.sum(dim='gas'), robust=False)
# # Let's check (2: plot the shares as a histogram)
# plt.hist(np.unique(len(np.unique(voc_shares_combined.sum(dim='gas', skipna=True).values.flatten().tolist())))); 
# # looks like only 1 values!
# # but interestingly; I'm not sure it would cover ALL gridcells; maybe because of the specific month?
    
    # 3. loop over sectors, multiply total shares with sectoral percentages
    for s in gfed_sectors_singlesector:
        # here we split the speciated VOC shares into their respective sectoral parts
        
        ds_perc = xr.open_dataset(
                get_bb4cmip7_location_percentage(
                    variable = f"{g}percentage{s}"
                ),
                engine="h5netcdf",
                chunks={},
                lock=lock
            ).sel(time=_normalize_time_slice(PROXY_YEAR)).drop_vars(
            # drop variables we don't need and rename the one we need
            ["lat_bnds", "lon_bnds", "time_bnds"]).rename({f"{g}percentage{s}": "percentage"})
    
        # do multiplication (now backed by NumPy arrays, detached from file IO)
        ds_bb = xr.Dataset({
            "emissions_share": ds_shares["emissions_share"] * ds_perc["percentage"] / 100
        })
    
        interim_totals_bb[(s)] = ds_bb
    
    
    for s in gfed_sectors_forest:
        # here we split the speciated VOC shares into their respective sectoral parts and aggregate forest sectors

        ds_perc = xr.open_dataset(
                get_bb4cmip7_location_percentage(
                    variable = f"{g}percentage{s}"
                ),
                engine="h5netcdf",
                chunks={},
                lock=lock
            ).sel(time=_normalize_time_slice(PROXY_YEAR)).drop_vars(
            # drop variables we don't need and rename the one we need
            ["lat_bnds", "lon_bnds", "time_bnds"]).rename({f"{g}percentage{s}": "percentage"})
    
        interim_forest_bb[(s)] = ds_perc
            
    # do multiplication (now backed by NumPy arrays, detached from file IO)
    ds_bb = xr.Dataset({
        "emissions_share": ds_shares["emissions_share"] * (
            interim_forest_bb["BORF"]["percentage"] +
            interim_forest_bb["DEFO"]["percentage"] +
            interim_forest_bb["TEMF"]["percentage"])/ 100
    })

    # from concordia.cmip7.utils_plotting import plot_map
    # plot_map(interim_forest_bb["BORF"].sel(time="2023-01-16").percentage, robust=False)
    # plot_map(interim_forest_bb["DEFO"].sel(time="2023-01-16").percentage, robust=False)
    # plot_map(interim_forest_bb["TEMF"].sel(time="2023-01-16").percentage, robust=False)

    # plot_map((
    #     interim_totals_bb["AGRI"].sel(time="2023-01-16").emissions_share * 100 +
    #     interim_totals_bb["PEAT"].sel(time="2023-01-16").emissions_share * 100 +
    #     interim_totals_bb["SAVA"].sel(time="2023-01-16").emissions_share * 100 +
    #     interim_forest_bb["BORF"].sel(time="2023-01-16").percentage + 
    #     interim_forest_bb["DEFO"].sel(time="2023-01-16").percentage + 
    #     interim_forest_bb["TEMF"].sel(time="2023-01-16").percentage
    # ), robust=False)


    interim_totals_bb["FRTB"] = ds_bb

    # plot_map((
    #     interim_totals_bb["AGRI"].sel(time="2023-01-16").emissions_share * 100 +
    #     interim_totals_bb["PEAT"].sel(time="2023-01-16").emissions_share * 100 +
    #     interim_totals_bb["SAVA"].sel(time="2023-01-16").emissions_share * 100 +
    #     # interim_forest_bb["BORF"].sel(time="2023-01-16").percentage + 
    #     # interim_forest_bb["DEFO"].sel(time="2023-01-16").percentage + 
    #     # interim_forest_bb["TEMF"].sel(time="2023-01-16").percentage
    #     interim_totals_bb["FRTB"].sel(time="2023-01-16").emissions_share * 100
    # ), robust=False)

    # np.nanmax(interim_totals_bb["FRTB"].sel(time="2023-01-16").emissions_share.values)
    # np.nanmax(interim_totals_bb["AGRI"].sel(time="2023-01-16").emissions_share.values)
    # np.nanmax(interim_totals_bb["PEAT"].sel(time="2023-01-16").emissions_share.values)
    # np.nanmax(interim_totals_bb["SAVA"].sel(time="2023-01-16").emissions_share.values)


    # 4. collect into one .nc per species, reformat for CMIP7 conventions
    ds_all = xr.concat(
    [ds["emissions_share"] for ds in interim_totals_bb.values()],
    dim="sector"
    )
    
    # label the sector dimension
    ds_all = ds_all.assign_coords(sector=list(interim_totals_bb.keys()))
    ds_all
        
    # regrid to 0.5 degree after multiplication (first rename variable names to be like the CEDS template data)
    print(f"Regridding from {ds_bb.latitude.size}x{ds_bb.longitude.size} to {len(target_lat)}x{len(target_lon)}")
    ds_all = ds_all.rename(
        {"latitude":"lat","longitude":"lon"}
    ).interp(
        lat=target_lat, lon=target_lon, method='linear'
    ) # TODO: read a bit more on aggregation methods.
    # NOTE: ISSUE: I think this will create issues -- we're interpolating between some values and a lot of zero values! So the value generally will be much lower.
    # NOTE: QUESTION: is this a problem? Or is it still fine?

    # formatting, including averaging if we do not select just one single year
    ds_reordered, outfile = formatting_to_cmip7_scenario_proxy(
        ds=ds_all, g=g, ysel=PROXY_YEAR
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


# %%
