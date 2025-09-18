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
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from tqdm import tqdm

from concordia.cmip7 import utils as cmip7_utils

# %%
lock = SerializableLock()

# %% [markdown]
# ## setup

# %%
grid_file_location = "/home/hoegner/Projects/CMIP7/input/gridding/"
# grid_file_location = "C:/Users/kikstra/IIASA/ECE.prog - Documents/Projects/CMIP7/IAM Data Processing/concordia_cmip7_v0_2/input/gridding/"

ceds_data_location = Path(grid_file_location, "ESGF", "CEDS", "CMIP7_anthro")
ceds_air_data_location = Path(grid_file_location, "ESGF", "CEDS", "CMIP7_AIR")

# %%
# Settings
SETTINGS_FILE = "config_cmip7_v0_2.yaml" # iteration round 2 

sub_version = "_corrected_indexraster"
HARMONIZATION_VERSION = f"config_cmip7_v0_2{sub_version}"

gridded_data_location = Path(f"/home/hoegner/GitHub/concordia/results/{HARMONIZATION_VERSION}")
weighted_data_location = Path(gridded_data_location, "weighted")

# %%
years = [2023, 2024, 2025, 2030, 2035, 2040, 2045, 2050, 2055, 2060, 2065, 2070, 2075, 2080, 2085, 2090, 2095, 2100]

sector_dict = {
    0: "Agriculture",
    1: "Energy",
    2: "Industrial",
    3: "Transportation",
    4: "Residential, Commercial, Other",
    5: "Solvents Production and Application",
    6: "Waste", 
    7: "International Shipping",
    8: "Other non-Land CDR",
    9: "BECCS"
}

# %% [markdown]
# ## test for one example file

# %%
# CEDS file path
ceds_file = Path(ceds_data_location, "CO2-em-anthro_input4MIPs_emissions_CMIP_CEDS-CMIP-2025-04-18_gn_200001-202312.nc")

# CMIP7-fast-track gridded file path
gridded_file = Path(gridded_data_location, "CO2-em-anthro_input4MIPs_emissions_CMIP7_IIASA-REMIND-MAgPIE-3.5-4.11-SSP1---Very-Low-Emissions_gn_202301-210012.nc")

# %%
# load data
ceds = xr.open_dataset(ceds_file)
gridded = xr.open_dataset(gridded_file)

# %%
# apply sector mapping to ceds file to match gridded file
ceds = ceds.assign_coords(
    sector=pd.Series(ceds["sector"].values).map(sector_dict).values
)
ceds

# %%
ceds23 = ceds.where(ceds.time.dt.year==2023, drop=True)
gridded23 = gridded.where(gridded.time.dt.year==2023, drop=True)

# %%
gridded23.sector


# %%
def calculate_diff(ceds_da, scen_da, gas, empty_treatment="fill_zeroes"):
    """
    Compute the ratio (or difference if desired) between a CEDS reference dataset
    and a scenario dataset for a specific gas, handling missing sectors.

    Returns an xarray DataArray with 'sector' dimension and all other coordinates intact.
    """
    
    sectors = scen_da.sector.values
    ratio_list = []  # store per-sector ratios
    sector_names = []

    var_name = f"{gas}_em_anthro"

    for sector in sectors:
        # skip unwanted sectors
    #    if sector in ['Other non-Land CDR', 'BECCS']:
    #        continue

        # check if sector exists in each dataset
        sector_in_ceds = sector in ceds_da.sector.values
        sector_in_scen = sector in scen_da.sector.values

        if empty_treatment == "skip":
            if not sector_in_ceds or not sector_in_scen:
                print(f"Skipping sector '{sector}' - missing in {'CEDS' if not sector_in_ceds else 'scenario'} data")
                continue

        # handle missing sectors by filling zeros
        if empty_treatment == "fill_zeroes":
            # CEDS sector
            if sector_in_ceds:
                ceds_sector = ceds_da.sel(sector=sector)
            else:
                template_sector = ceds_da.isel(sector=0).squeeze()
                ceds_sector = xr.zeros_like(template_sector).expand_dims(sector=[sector])

            # Scenario sector
            if sector_in_scen:
                scen_sector = scen_da.sel(sector=sector)
            else:
                template_sector = scen_da.isel(sector=0).squeeze()
                scen_sector = xr.zeros_like(template_sector).expand_dims(sector=[sector])

        # select the numeric data variable only
        if var_name in ceds_sector.data_vars:
            ceds_values = ceds_sector[var_name]
        else:
            ceds_values = ceds_sector[list(ceds_sector.data_vars)[0]]

        if var_name in scen_sector.data_vars:
            scen_values = scen_sector[var_name]
        else:
            scen_values = scen_sector[list(scen_sector.data_vars)[0]]
        
        # compute ratio safely
        mask = scen_values != 0.0
        ratio_sector = xr.where(mask, ceds_values / scen_values, 1.0)
        
        # ensure sector dimension exists
        if 'sector' not in ratio_sector.dims:
            ratio_sector = ratio_sector.expand_dims(sector=[sector])

        ratio_list.append(ratio_sector)
        sector_names.append(sector)

    # concatenate all sectors along 'sector' dimension
    full_ratio = xr.concat(ratio_list, dim='sector')
    full_ratio = full_ratio.assign_coords(sector=sector_names)

    return full_ratio


# %%
pct_diff23 = calculate_diff(ceds23, gridded23, "CO2")

# %%
test = pct_diff23.sel(sector="Waste").isel(time=0)

# %%
print(np.min(test.values))
print(np.max(test.values))

# %%
fig, ax = plt.subplots(
    figsize=(10, 5),  # wider figure
    subplot_kw={'projection': ccrs.PlateCarree()}
)

cbar_kwargs = {"label": "Value", "shrink": 0.8}
test.plot(ax=ax, transform=ccrs.PlateCarree(), cmap="viridis", cbar_kwargs=cbar_kwargs)

# Add coastlines and land
ax.coastlines()
ax.add_feature(cfeature.LAND, facecolor="lightgray")
ax.add_feature(cfeature.BORDERS, linestyle=':')

plt.show()

# %%
weights = pct_diff23.to_dataset(name="CO2_em_anthro")

# %%
weighted23 = gridded23 * weights

# %%
test23 = weighted23 - ceds23

# %%
test = test23["CO2_em_anthro"].sel(sector="Waste").isel(time=0)

# %%
print(np.min(test.values))
print(np.max(test.values))

# %%
fig, ax = plt.subplots(
    figsize=(10, 5),  # wider figure
    subplot_kw={'projection': ccrs.PlateCarree()}
)

cbar_kwargs = {"label": "Value", "shrink": 0.8}
test.plot(ax=ax, transform=ccrs.PlateCarree(), cmap="RdBu_r", cbar_kwargs=cbar_kwargs)

# Add coastlines and land
ax.coastlines()
ax.add_feature(cfeature.LAND, facecolor="lightgray")
ax.add_feature(cfeature.BORDERS, linestyle=':')

plt.show()

# %% [markdown]
# ## run for all files

# %%
files = [
    file
    for file in gridded_data_location.glob("*.nc")
    if "anthro" in file.name and "-AIR-" not in file.name
]

# %% [markdown]
# with dask
#
# prefix, postfix = [], []
#
# for file in files:
#     gas = file.name.split("-")[0]
#     outfile = Path(weighted_data_location, file.name)
#     print(f"creating fix for {file.name}")
#
#     # match reference file
#     if gas == "VOC":
#         gas = "NMVOC"
#     elif gas == "Sulfur":
#         gas = "SO2"
#     else:
#         gas = gas
#     match = next(ceds_data_location.glob(f"{gas}-*.nc"))
#     
#     # open with dask
#     ceds = xr.open_dataset(match, chunks={"time": 12})
#     gridded = xr.open_dataset(file, chunks={"time": 12})
#
#     if "NMVOC_em_anthro" in ceds.data_vars:
#         ceds = ceds.rename({"NMVOC_em_anthro": "VOC_em_anthro"})
#     if "SO2_em_anthro" in ceds.data_vars:
#         ceds = ceds.rename({"SO2_em_anthro": "Sulfur_em_anthro"})
#
#     # revert names for output
#     if gas == "NMVOC":
#         gas = "VOC"
#     elif gas == "SO2":
#         gas = "Sulfur"
#     else:
#         gas = gas
#     
#     var = f"{gas}_em_anthro"
#  
#     # rename sectors
#     ceds = ceds.assign_coords(sector=pd.Series(ceds["sector"].values).map(sector_dict).values)
#     reference = ceds.where(ceds.time.dt.year == 2023, drop=True)
#     gridded_23 = gridded.where(gridded.time.dt.year == 2023, drop=True)
#
#     # calculate relative difference
#     pct_diff23 = calculate_diff(reference, gridded_23, gas)
#     weights = pct_diff23.to_dataset(name=f"{gas}_em_anthro")
#
#     # expand weights to all years
#     n_repeat = gridded.sizes["time"] // weights.sizes["time"]
#     weights_exp = xr.concat([weights] * n_repeat, dim="time")
#     weights_exp = weights_exp.assign_coords(time=gridded.time)
#
#     weighted = gridded * weights_exp
#
#     # replace the data in weighted with the data in gridded for the sectors we don't want to be touched
#     print(weighted.sector.values)
#     sectors_to_keep = ['Other non-Land CDR', 'BECCS', 'International Shipping']
#     sectors_present = [s for s in sectors_to_keep if s in weighted.sector.values]
#     if sectors_present:
#         weighted[var].loc[dict(sector=sectors_present)] = gridded[var].sel(sector=sectors_present)
#
#     # calculate sectoral global totals
#     gridded_global = gridded[f"{gas}_em_anthro"].groupby("sector").sum(dim=("lat", "lon"))
#     weighted_global = weighted[f"{gas}_em_anthro"].groupby("sector").sum(dim=("lat", "lon"))
#         
#     df1 = gridded_global.to_dataframe(name="prefix").reset_index()
#     df1["gas"] = gas
#     df1 = df1.melt(id_vars=["time", "sector", "gas"], var_name="version", value_name="value")
#
#     df2 = weighted_global.to_dataframe(name="postfix").reset_index()
#     df2["gas"] = gas
#     df2 = df2.melt(id_vars=["time", "sector", "gas"], var_name="version", value_name="value")
#
#     prefix.append(df1)
#     postfix.append(df2)
#
#     outfile.unlink(missing_ok=True)
#     
#     encoding = {var: {"zlib": True, "complevel": 4}}
#
#     with ProgressBar():
#         weighted.to_netcdf(outfile, encoding=encoding, compute=True)
#
# combine results
# global_prefix = pd.concat(prefix, ignore_index=True)
# global_postfix = pd.concat(postfix, ignore_index=True)

# %%
## without dask

prefix, postfix = [], []

for file in tqdm(files, desc="Processing files"):
    gas = file.name.split("-")[0]
    outfile = Path(weighted_data_location, file.name)
    print(f"creating fix for {file.name}")

    # match reference file
    if gas == "VOC":
        gas = "NMVOC"
    elif gas == "Sulfur":
        gas = "SO2"
    match = next(ceds_data_location.glob(f"{gas}-*.nc"))

    # open datasets (no dask)
    ceds = xr.open_dataset(match)
    gridded = xr.open_dataset(file)

    # rename variables if needed
    if "NMVOC_em_anthro" in ceds.data_vars:
        ceds = ceds.rename({"NMVOC_em_anthro": "VOC_em_anthro"})
    if "SO2_em_anthro" in ceds.data_vars:
        ceds = ceds.rename({"SO2_em_anthro": "Sulfur_em_anthro"})

    # revert gas names for output
    if gas == "NMVOC":
        gas = "VOC"
    elif gas == "SO2":
        gas = "Sulfur"
    
    var = f"{gas}_em_anthro"

    # rename sectors
    ceds = ceds.assign_coords(sector=pd.Series(ceds["sector"].values).map(sector_dict).values)
    reference = ceds.where(ceds.time.dt.year == 2023, drop=True)
    gridded_23 = gridded.where(gridded.time.dt.year == 2023, drop=True)

    # calculate relative difference (vectorized)
    pct_diff23 = calculate_diff(reference, gridded_23, gas)
    weights = pct_diff23.to_dataset(name=var)

    # expand weights to all years
    n_repeat = gridded.sizes["time"] // weights.sizes["time"]
    weights_exp = xr.concat([weights] * n_repeat, dim="time")
    weights_exp = weights_exp.assign_coords(time=gridded.time)

    # apply weights
    weighted = gridded * weights_exp

    # replace sectors we don't want weighted
    sectors_to_keep = ['Other non-Land CDR', 'BECCS', 'International Shipping']
    sectors_present = [s for s in sectors_to_keep if s in weighted.sector.values]
    if sectors_present:
        weighted[var].loc[dict(sector=sectors_present)] = gridded[var].sel(sector=sectors_present)

    # calculate sectoral global totals
    gridded_global = gridded[var].groupby("sector").sum(dim=("lat", "lon")).astype("float64")
    weighted_global = weighted[var].groupby("sector").sum(dim=("lat", "lon")).astype("float64")

    # convert to dataframes
    df1 = gridded_global.to_dataframe(name="prefix").reset_index()
    df1["gas"] = gas
    df1 = df1.melt(id_vars=["time", "sector", "gas"], var_name="version", value_name="value")

    df2 = weighted_global.to_dataframe(name="postfix").reset_index()
    df2["gas"] = gas
    df2 = df2.melt(id_vars=["time", "sector", "gas"], var_name="version", value_name="value")

    prefix.append(df1)
    postfix.append(df2)

    # remove old file
    outfile.unlink(missing_ok=True)

    # save weighted dataset (no dask)
    encoding = {var: {"zlib": True, "complevel": 4}}
    weighted.to_netcdf(outfile, encoding=encoding)

# combine results
global_prefix = pd.concat(prefix, ignore_index=True)
global_postfix = pd.concat(postfix, ignore_index=True)

# %%
global_prefix["time"] = pd.to_datetime(global_prefix["time"].astype(str))
global_postfix["time"] = pd.to_datetime(global_postfix["time"].astype(str))

# %%
to_plot = pd.concat([global_prefix, global_postfix])
out_csv = Path(weighted_data_location, "global_aggregates_for_checking.csv")
to_plot.to_csv(out_csv, index=False)

# %%
sns.relplot(
    data=to_plot,
    kind="line",
    x="time",
    y="value",
    hue="sector",
    col="gas",
    style="version",
    col_wrap=3,
    facet_kws={"sharey": False}
)
plt.show()

# %%
wide_df = to_plot.pivot_table(
    index=["gas", "sector", "time"],
    columns="version",
    values="value"
)
wide_df["diff"] = wide_df["postfix"] - wide_df["prefix"]
wide = wide_df.reset_index()

# %%
sns.relplot(
    data=wide,
    kind="line",
    x="time",
    y="diff",
    hue="sector",
    col="gas",
    col_wrap=3,
    facet_kws={"sharey": False}
)
plt.show()
