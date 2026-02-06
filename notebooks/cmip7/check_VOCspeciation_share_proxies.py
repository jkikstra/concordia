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
# check later if we need all these imports 
import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

import os
from glob import glob
import cartopy.feature as cfeature
import cartopy.crs as ccrs

from concordia.cmip7 import utils as cmip7_utils
from concordia.settings import Settings
import concordia.cmip7.utils_futureproxy_ceds_bb4cmip as uprox
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
settings.proxy_path

# %%
anthro_path = settings.proxy_path / "VOC_speciation"
openburning_path = settings.proxy_path / "NMVOC_speciation"

# %% [markdown]
# ## check openburning speciated VOC share proxy files
#
# we check whether on the grid-cell level the shares add up to 0 or 1 as expected, when summing across all species and sectors, at any random time slice

# %%
files = glob(os.path.join(openburning_path, "*.nc"))

# Open datasets
datasets = [xr.open_dataset(f) for f in files]

# Get the variable name (assuming the same across all files)
var_name = list(datasets[0].data_vars)[0]  # first variable

data_arrays = [ds[var_name].isel(year=0, month=0, sector=0) for ds in datasets]

summed = xr.concat(data_arrays, dim="file").sum(dim="file", skipna=True)
openb = summed.sum(dim="gas", skipna=True)

# Flatten to 1D list of values (across lat-lon-sector gridcells)
bbvalues_list = openb.values.flatten().tolist()

# %%
openburning_path

# %%
len(np.unique(bbvalues_list))

# %%
plt.hist(np.unique(np.round(bbvalues_list, 2)));

# %%
plt.hist(bbvalues_list);

# %% [markdown]
# ## check anthro speciated VOC share proxy files
#
# we check whether on the grid-cell level the shares add up to 0 or 1 as expected, when summing across all species and sectors, at any random time slice

# %%
files = glob(os.path.join(anthro_path, "*.nc"))

# Open datasets
datasets = [xr.open_dataset(f) for f in files]

# Get the variable name (assuming the same across all files)
var_name = list(datasets[0].data_vars)[0]  # first variable

data_arrays = [ds[var_name].isel(year=0, month=0, sector=1) for ds in datasets]

summed = xr.concat(data_arrays, dim="file").sum(dim="file", skipna=True)
anthro = summed.sum(dim="gas", skipna=True)

# Flatten to 1D list of values (across lat-lon-sector gridcells)
values_list = anthro.values.flatten().tolist()

# %%
len(np.unique(values_list))

# %%
plt.hist(np.unique(np.round(values_list, 2)));

# %%
plt.hist(values_list);

# %%
fig, ax = plt.subplots(figsize=(10, 7), subplot_kw={"projection": ccrs.PlateCarree()})

im1 = openb.plot(ax=ax, cmap='viridis', add_colorbar=False, transform=ccrs.PlateCarree(),
                 vmin=0, vmax=1)

ax.add_feature(cfeature.BORDERS, linewidth=0.5, color='white', alpha=0.8)
ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color='white', alpha=0.8)
ax.set_global();

# %%
fig, ax = plt.subplots(figsize=(10, 7), subplot_kw={"projection": ccrs.PlateCarree()})

im1 = anthro.plot(ax=ax, cmap='viridis', add_colorbar=False, transform=ccrs.PlateCarree(),
                  vmin=0, vmax=1)

ax.add_feature(cfeature.BORDERS, linewidth=0.5, color='white', alpha=0.8)
ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color='white', alpha=0.8)
ax.set_global();
