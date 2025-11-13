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
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from tqdm import tqdm

from concordia.cmip7 import utils as cmip7_utils

# %%
lock = SerializableLock()

# %% [markdown]
# ## setup

# %%
from concordia.cmip7.CONSTANTS import CONFIG, return_marker_information, PROXY_YEARS, CMIP_ERA, GASES_ESGF_CEDS_VOC, find_voc_data_variable_string
import concordia.cmip7.utils_futureproxy_ceds_bb4cmip as uprox

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
        try:
            # Fallback for interactive/Jupyter mode, where 'file location' does not exist
            cmip7_dir = Path().resolve()  # one up
            settings = uprox.get_settings(base_path=cmip7_dir, file = CONFIG)
        except (FileNotFoundError, NameError):
            # if Path().resolve somehow goes to the root of this repository
            cmip7_dir = Path().resolve() / "notebooks" / "cmip7"  # one up
            settings = uprox.get_settings(base_path=cmip7_dir, file = CONFIG)

# %% editable=true slideshow={"slide_type": ""} tags=["parameters"]
marker_to_run: str = "H" # still run this for VLLO

# %%
# Scenario information
HARMONIZATION_VERSION, MODEL_SELECTION, SCENARIO_SELECTION, _ = return_marker_information(
    m=marker_to_run
)

# %%
HARMONIZATION_VERSION = "None"

# %%
grid_file_location = settings.gridding_path

ceds_data_location = Path(grid_file_location, "esgf", "ceds", "CMIP7_anthro")
ceds_air_data_location = Path(grid_file_location, "esgf", "ceds", "CMIP7_AIR")

# %%
gridded_data_location = settings.out_path / HARMONIZATION_VERSION

# %%
years = [year for year in PROXY_YEARS if year >= settings.base_year]

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

# %%
files_anthro = [
    file
    for file in gridded_data_location.glob("*.nc")
    if "anthro" in file.name and "-AIR-" not in file.name and "speciated" not in file.name
]
files_voc = [
    file
    for file in gridded_data_location.glob("*.nc")
    if "anthro" in file.name and "-AIR-" not in file.name and "speciated" in file.name
]

ceds_anthro = [
    file
    for file in ceds_data_location.glob("*.nc")
    if "anthro" in file.name and "-AIR-" not in file.name and "speciated" not in file.name
]
ceds_voc = [
    file
    for file in ceds_data_location.glob("*.nc")
    if "anthro" in file.name and "-AIR-" not in file.name and "speciated" in file.name
]


# %%
def var_name_from_file(file):

    if file.stem.split("-")[0] == "Sulfur":
        gas = "SO2"
    if file.stem.split("-")[0] == "NMVOC":
        gas = "VOC"
    else:
        gas = file.stem.split("-")[0]

    var_name = gas + "_em_anthro"
            
    return var_name


# %%
datasets_anthro = {var_name_from_file(f): xr.open_dataset(f) for f in files_anthro}
datasets_ceds = {var_name_from_file(f): xr.open_dataset(f) for f in ceds_anthro}

# %%
updated_datasets = {}

for name, ds_anthro in datasets_anthro.items():
    if name in datasets_ceds:
        ds_ceds = datasets_ceds[name]

        # figure out the variable name inside each dataset
        var_anthro = list(ds_anthro.data_vars)[0]
        var_ceds = list(ds_ceds.data_vars)[0]

        # if variable names differ (e.g. "Sulfur_em_anthro" vs "SO2_em_anthro"), rename CEDS variable
        if var_ceds != name:
            ds_ceds = ds_ceds.rename({var_ceds: name})

        ds_ceds = ds_ceds.assign_coords(sector=("sector", [sector_dict[s] for s in ds_ceds.sector.values]))

        # apply replacing only to sectors available in CEDS
        common_sectors = sorted(set(ds_anthro.sector.values) & set(ds_ceds.sector.values))
        
        ds_anthro[name].loc[
            dict(time=ds_anthro.time[0:12], sector=common_sectors)
        ] = ds_ceds[name].sel(sector=common_sectors).isel(time=slice(-24,-12))

        updated_datasets[name] = ds_anthro

# %%
for name, ds in updated_datasets.items():
    out_path = gridded_data_location / f"{name}_updated.nc"
    
    if out_path.exists():
        out_path.unlink()
        
    ds.to_netcdf(out_path, mode="w")

# %%
BC_new = xr.open_dataset(gridded_data_location / "BC_em_anthro_updated.nc")
BC_original = xr.open_dataset(gridded_data_location / "BC-em-anthro_input4MIPs_emissions_CMIP6plus_IIASA-REMIND-MAgPIE-3.5-4.11-SSP1---Very-Low-Emissions_gn_202301-210012.nc")

# %%
print(BC_original.attrs.keys())
print(BC_new.attrs.keys())

print(BC_original["BC_em_anthro"].attrs.keys())
print(BC_new["BC_em_anthro"].attrs.keys())

print(BC_original["BC_em_anthro"].shape)
print(BC_new["BC_em_anthro"].shape)

print(BC_original.dims)
print(BC_new.dims)

print(BC_original["BC_em_anthro"].encoding)
print(BC_new["BC_em_anthro"].encoding)

# %%
data_var = "BC_em_anthro"

# define time slices
replaced_time = slice(0, 12)
unchanged_time = slice(13, None)

# common sectors (all sectors in BC_new except those replaced if you want stricter)
common_sectors = BC_new.sector.values  # or slice only sectors not replaced if needed

# select unchanged time slice
unchanged_original = BC_original[data_var].isel(time=unchanged_time)
unchanged_new      = BC_new[data_var].isel(time=unchanged_time)

# check that unchanged part matches
unchanged_match = np.allclose(unchanged_original, unchanged_new, rtol=1e-6, atol=1e-12)
print("Unchanged time slice matches:", unchanged_match)

# optionally, check dimensions and coordinates match
dims_match = unchanged_original.dims == unchanged_new.dims
coords_match = all(np.array_equal(unchanged_original[c], unchanged_new[c]) for c in unchanged_original.coords)
print("Dimensions match:", dims_match)
print("Coordinates match:", coords_match)

# %%
unchanged_slice = BC_new["BC_em_anthro"].isel(time=slice(12, None))
nan_positions = np.isnan(unchanged_slice.values)
print("Number of NaNs in unchanged slice:", nan_positions.sum())

# %%
import matplotlib.pyplot as plt
import numpy as np

time_index = 0
sector_index = 7

data_orig = BC_original["BC_em_anthro"].isel(time=time_index, sector=sector_index)
data_new  = BC_new["BC_em_anthro"].isel(time=time_index, sector=sector_index)

# Shared 99th percentile vmax
vmax = np.nanpercentile(np.concatenate([data_orig.values.flatten(), data_new.values.flatten()]), 99)

fig, axes = plt.subplots(1, 2, figsize=(15,5), constrained_layout=True)

# Left plot without colorbar
data_orig.plot(ax=axes[0], cmap="viridis", vmax=vmax, add_colorbar=False)
axes[0].set_title(f"Original - Time {data_orig.time.values}, Sector {data_orig.sector.values}")

# Right plot with colorbar
data_new.plot(ax=axes[1], cmap="viridis", vmax=vmax)
axes[1].set_title(f"New - Time {data_new.time.values}, Sector {data_new.sector.values}")

plt.show()

# %%
time_index = 0
sector_index = 9

data_orig = BC_original["BC_em_anthro"].isel(time=time_index, sector=sector_index)
data_new  = BC_new["BC_em_anthro"].isel(time=time_index, sector=sector_index)

# Shared 99th percentile for the first row
vmax = np.nanpercentile(np.concatenate([data_orig.values.flatten(), data_new.values.flatten()]), 99)

# Compute difference
diff = data_new - data_orig
diff_abs_max = np.nanmax(np.abs(diff.values))  # for symmetric colormap

# Create 2x2 figure (top row: original/new, bottom row: difference)
fig, axes = plt.subplots(2, 2, figsize=(15,10), constrained_layout=True)

# Top row
data_orig.plot(ax=axes[0,0], cmap="viridis", vmax=vmax, add_colorbar=False)
axes[0,0].set_title(f"Original - Time {data_orig.time.values}, Sector {data_orig.sector.values}")

data_new.plot(ax=axes[0,1], cmap="viridis", vmax=vmax)
axes[0,1].set_title(f"New - Time {data_new.time.values}, Sector {data_new.sector.values}")

# Bottom row (difference)
# merge both bottom plots into one axes spanning both columns
fig.delaxes(axes[1,1])  # remove extra axes
diff_ax = axes[1,0]
diff.plot(ax=diff_ax, cmap="RdBu_r", vmin=-diff_abs_max, vmax=diff_abs_max)
diff_ax.set_title("Difference (New - Original)")

plt.show()


# %%
BC_new.isel(time=0, sector=0).values

# %%
BC_ceds = xr.open_dataset("/Users/hoegner/Projects/CMIP7/input/gridding/esgf/ceds/CMIP7_anthro/BC-em-anthro_input4MIPs_emissions_CMIP_CEDS-CMIP-2025-04-18_gn_200001-202312.nc")

# %%
BC_ceds.isel(time=0, sector=2)["BC_em_anthro"].values

# %%
BC_new.isel(time=0, sector=2)["BC_em_anthro"].values

# %%
BC_original.isel(time=0, sector=2)["BC_em_anthro"].values
