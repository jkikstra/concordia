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
import os
from glob import glob

from concordia.cmip7 import utils as cmip7_utils
from concordia.settings import Settings
import concordia.cmip7.utils_futureproxy_ceds_bb4cmip as uprox
from concordia.cmip7.utils_futureproxy_ceds_bb4cmip import _normalize_time_slice
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

# %%
openburning_path

# %%
files = glob(os.path.join(openburning_path, "*.nc"))

# Open datasets
datasets = [xr.open_dataset(f) for f in files]

# Get the variable name (assuming the same across all files)
var_name = list(datasets[0].data_vars)[0]  # first variable

data_arrays = [ds[var_name].isel(year=0, month=1, sector=0) for ds in datasets]

summed = xr.concat(data_arrays, dim="file").sum(dim="file", skipna=True)
openb = summed.sum(dim="gas", skipna=True)

# Flatten to 1D list of values (across lat-lon-sector gridcells)
values_list = openb.values.flatten().tolist()

# %%
len(np.unique(values_list))

# %%
plt.hist(np.unique(values_list));

# %%
files = glob(os.path.join(anthro_path, "*.nc"))

# Open datasets
datasets = [xr.open_dataset(f) for f in files]

# Get the variable name (assuming the same across all files)
var_name = list(datasets[0].data_vars)[0]  # first variable

data_arrays = [ds[var_name].isel(year=0, month=0) for ds in datasets]

summed = xr.concat(data_arrays, dim="file").sum(dim="file", skipna=True)
anthro = summed.sum(dim="gas", skipna=True)

# Flatten to 1D list of values (across lat-lon-sector gridcells)
values_list = anthro.values.flatten().tolist()

# %%
data_arrays

# %%
len(np.unique(values_list))

# %%
plt.hist(values_list);

# %%
import cartopy.feature as cfeature
import cartopy.crs as ccrs

fig, ax = plt.subplots(figsize=(15, 10), subplot_kw={"projection": ccrs.PlateCarree()})

# Replot the data on the cartopy axes
im1 = openb.plot(ax=ax, cmap='viridis', add_colorbar=False, transform=ccrs.PlateCarree())

# Add country borders and coastlines
ax.add_feature(cfeature.BORDERS, linewidth=0.5, color='white', alpha=0.8)
ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color='white', alpha=0.8)
ax.set_global();

# %%
import cartopy.feature as cfeature
import cartopy.crs as ccrs

fig, ax = plt.subplots(figsize=(10, 7), subplot_kw={"projection": ccrs.PlateCarree()})

# Replot the data on the cartopy axes
im1 = anthro.plot(ax=ax, cmap='viridis', add_colorbar=False, transform=ccrs.PlateCarree(),
                        vmin=0, 
    vmax=1)

# Add country borders and coastlines
ax.add_feature(cfeature.BORDERS, linewidth=0.5, color='white', alpha=0.8)
ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color='white', alpha=0.8)
ax.set_global();

# %%
bulk = xr.load_dataset("/Users/hoegner/Projects/CMIP7/concordia_cmip7_esgf_v0_alpha/input/gridding/esgf/bb4cmip7/NMVOCbulk/gn/v20250612/NMVOCbulk_input4MIPs_emissions_CMIP_DRES-CMIP-BB4CMIP7-2-1_gn_190001-202312.nc")

# %%
bulk

# %%
import cartopy.feature as cfeature
import cartopy.crs as ccrs

test = bulk.isel(time=1480)["NMVOCbulk"]
fig, ax = plt.subplots(figsize=(10, 7), subplot_kw={"projection": ccrs.PlateCarree()})

# Replot the data on the cartopy axes
im1 = test.plot(ax=ax, cmap='viridis', add_colorbar=False, transform=ccrs.PlateCarree(),
                        vmin=0.0000000001, 
    vmax=0.00000000015)

# Add country borders and coastlines
ax.add_feature(cfeature.BORDERS, linewidth=0.5, color='white', alpha=0.8)
ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color='white', alpha=0.8)
ax.set_global();

# %%
speciated = Path("/Users/hoegner/Projects/CMIP7/concordia_cmip7_esgf_v0_alpha/input/gridding/esgf/bb4cmip7_voc")

# %%
files = [
    f for f in glob(os.path.join(speciated, "**", "*.nc"), recursive=True)
    if "percentage" not in f
]

datasets = [xr.open_dataset(f) for f in files]
var_name = "NMVOCbulk"
datasets = [
    ds.rename({list(ds.data_vars)[3]: var_name})
    for ds in datasets
]

# %%
# test that the sum of the speciated files equals the bulk
for i in np.arange(1484, len(bulk.time)):
    data_arrays = [ds[var_name].isel(time=i) for ds in datasets]
    summed = xr.concat(data_arrays, dim="file").sum(dim="file", skipna=True)
    test = bulk.isel(time=i).fillna(0)[var_name]
    print("testing")
    xr.testing.assert_allclose(test, summed)
