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
GRIDDING_VERSION: str | None = "joy-ride-H"

# %%
try:
    # Try to get __file__ (works when running as script)
    HERE = Path(__file__).parent
except NameError:
    # When running in notebook/papermill, use a more robust approach
    # Find the concordia repository root and navigate to notebooks/cmip7
    current_path = Path.cwd()
    
    # Look for the concordia root directory (contains pyproject.toml)
    concordia_root = None
    for parent in [current_path] + list(current_path.parents):
        if (parent / "pyproject.toml").exists() and (parent / "src" / "concordia").exists():
            concordia_root = parent
            break
    
    if concordia_root is None:
        raise RuntimeError("Could not find concordia repository root")
    
    HERE = concordia_root / "notebooks" / "cmip7"

settings = Settings.from_config(version=GRIDDING_VERSION,
                                local_config_path=Path(HERE,
                                                       CONFIG))

settings.base_year

# %%
settings.proxy_path

# %%
anthro_path = settings.proxy_path / "VOC_speciation"
openburning_path = settings.proxy_path / "NMVOC_speciation"

# %%
openburning_path

# %%
# Check openburning VOC totals
# files = glob(os.path.join(openburning_path, "*CH3OH*.nc"))
files = glob(os.path.join(openburning_path, "*.nc"))

# Open datasets
# datasets = [xr.open_dataset(f) for f in files]

# Get the variable name (assuming the same across all files)
# var_name = 'emissions_share' # list(datasets[0].data_vars)[0]  # first variable should be the variable name for the emissions_share



# data_arrays = [ds[var_name].isel(year=0, month=1, sector=0) for ds in datasets]
data_arrays = [xr.open_dataset(f).emissions_share.isel(year=0, month=1).assign_attrs(file=f) for f in files]

# %%
# for d in data_arrays:
#     print("\n"*2)
#     print(f"Filename: {Path(d.file).name}")
#     print(f"Sectors: {d.sector}")
#     print("\n"*2)

summed = xr.concat(data_arrays, dim="file").sum(dim="file", skipna=True)
openb = summed.sum(dim="gas", skipna=True)

# Flatten to 1D list of values (across lat-lon-sector gridcells)
values_list = openb.values.flatten().tolist()

# %%
len(np.unique(values_list))

# %%
plt.hist(np.unique(values_list)); # should be 0 or 1, not inbetween

# %%
from concordia.cmip7.utils_plotting import plot_map

# %%

plot_map(
    openb.sel(sector='AGRI')
)
plot_map(
    openb.sel(sector='SAVA')
)
plot_map( 
    openb.sel(sector='PEAT'),
    robust=False,
    coastlines=True
)

plot_map(
    openb.sel(sector='FRTB'),
    robust=False
)
plot_map(
    openb.sel(sector='FRTB') # the issue is clearly in forest burning
)

x = openb.sel(sector='PEAT') + openb.sel(sector='SAVA') + openb.sel(sector='AGRI') + openb.sel(sector='FRTB')
x = x.where(x >= 1, 0)

plot_map(
    openb.sel(sector='PEAT') + openb.sel(sector='SAVA') + openb.sel(sector='AGRI') + openb.sel(sector='FRTB'),
    robust=False
)














# %%
# ..
# %%
# ..
# %%
# ..
# %%
# ..
# %%
# ..

# %%
# Check also anthropogenic VOC totals
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
