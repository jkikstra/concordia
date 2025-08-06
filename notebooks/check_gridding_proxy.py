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
import glob
from tqdm import tqdm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import seaborn as sns
from concordia.cmip7 import utils as cmip7_utils

IAMC_COLS = ["model", "scenario", "region", "variable", "unit"] 


# %%
# Scenarios pre-gridding
# harmonized_data_location = "C:/Users/kikstra/IIASA/ECE.prog - Documents/Projects/CMIP7/IAM Data Processing/concordia_cmip7_v0_testing/input/scenarios/"
path = "/home/hoegner/Projects/CMIP7/input/gridding/proxy_rasters/"


# %% [markdown]
# ## Anthro

# %%
files = glob.glob(os.path.join(path, "anthro*.nc"))

# %%
proxies = {
    os.path.splitext(os.path.basename(fp))[0]: xr.open_dataset(fp)
    for fp in files
}

# %%
proxies["anthro_OC"]

# %%
for name, ds in proxies.items():
    print(f"{name}")
    for var_name, da in ds.data_vars.items():
        print(f"   {var_name}: {da.dims}")


# %%
def plot_slice_on_map(ds, var_name, title=None, gas_idx=0, sector_idx=0, year_idx=0, month_idx=0):
    da = ds[var_name]

    da_sel = da.isel(
        gas=gas_idx,
        sector=sector_idx,
        year=year_idx,
        month=month_idx
    )

    lat = ds['lat']
    lon = ds['lon']

    # Extract coordinate labels for the selected indices
    gas_label = da_sel.coords['gas'].item()
    sector_label = da_sel.coords['sector'].item()
    year_label = da_sel.coords['year'].item()
    month_label = da_sel.coords['month'].item()

    plt.figure(figsize=(8, 4))
    ax = plt.axes(projection=ccrs.PlateCarree())

    im = ax.pcolormesh(lon, lat, da_sel, transform=ccrs.PlateCarree(), shading='auto', cmap="plasma")

    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.set_global()

    plt.colorbar(
        im,
        orientation='vertical',
        label=f"{var_name}",
        shrink=0.82
    )

    # Format month as name or number (optional)
    import calendar
    month_name = calendar.month_name[month_label]  # 'January', etc.

    plt.title(f"{title or var_name} — Gas: {gas_label}, Sector: {sector_label}, Year: {year_label}, Month: {month_name}")
    plt.tight_layout()
    plt.show()



# %%
for name, ds in proxies.items():
    first_var = list(ds.data_vars)[0]
    plot_slice_on_map(ds, first_var, sector_idx=2, year_idx=10, month_idx=5, title=name)

# %% [markdown]
# ## Shipping

# %%
files = glob.glob(os.path.join(path, "shipping*.nc"))

# %%
proxies_shipping = {
    os.path.splitext(os.path.basename(fp))[0]: xr.open_dataset(fp)
    for fp in files
}

# %%
proxies_shipping["shipping_CO"]

# %%
for name, ds in proxies_shipping.items():
    print(f"{name}")
    for var_name, da in ds.data_vars.items():
        print(f"   {var_name}: {da.dims}")

# %%
for name, ds in proxies_shipping.items():
    first_var = list(ds.data_vars)[0]
    plot_slice_on_map(ds, first_var, year_idx=0, month_idx=1, title=name)

# %% [markdown]
# ## all

# %%
files = glob.glob(os.path.join(path, "*.nc"))

# %%
proxy_rasters = {
    os.path.splitext(os.path.basename(fp))[0]: xr.open_dataset(fp)
    for fp in files
}

# %%
for name, ds in proxy_rasters.items():
    print(f"Dataset: {name}")
    
    # Loop over variables in the dataset
    for var_name, da in ds.data_vars.items():
        print(f"  Variable: {var_name}")
        
        # Check and print available coordinate values
        for coord in ['sector', 'level', 'year']:
            if coord in da.coords:
                coord_vals = da.coords[coord].values
                print(f"    {coord}: {coord_vals}")

# %%
