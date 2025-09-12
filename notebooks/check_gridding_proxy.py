# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
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
# path = "/home/hoegner/Projects/CMIP7/input/gridding/proxy_rasters/"
path = "C:/Users/kikstra/IIASA/ECE.prog - Documents/Projects/CMIP7/IAM Data Processing/concordia_cmip7_v0_2/input/gridding/proxy_rasters/20250810"



# %% [markdown]
# ## Anthro (nice plots)

# %% [markdown]
# ### Proxy

# %%
path_new = Path("C:/Users/kikstra/IIASA/ECE.prog - Documents/Projects/CMIP7/IAM Data Processing/concordia_cmip7_v0_2/input/gridding/proxy_rasters/20250821-WSTfix")
proxy_file = "anthro_CO.nc"

def plot_maps(ds, ncols=1, proj=ccrs.Robinson()): 

    nrows = 1

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(6 * ncols, 4.5 * nrows),
        subplot_kw={"projection": ccrs.Robinson()}
    )

    # Flatten axes safely
    if isinstance(axes, np.ndarray):
        axes = axes.flatten() # make indexing easier
    else:
        axes = [axes]

    da = (
        ds
            .emissions
            .squeeze()
    )

    # Plot directly with xarray's .plot.pcolormesh
    da.plot.pcolormesh(
        ax=axes[0],
        transform=ccrs.PlateCarree(),
        cmap="GnBu",
        robust=True,
        cbar_kwargs={"orientation": "horizontal", "shrink": 0.65},
    )
    axes[0].set_title(proxy_file)
    axes[0].coastlines()

    plt.tight_layout()
    plt.show()

ds = xr.open_dataset(path_new / proxy_file)
plot_maps(
    ds.sel(sector='WST',
           year=2023,month=1)
)

# %% [markdown]
# ### Output

# %%
path_new = Path("C:/Users/kikstra/Documents/GitHub/concordia/results/config_cmip7_v0_2_WSTfix_remind/")
gas = "N2O"
results_file = f"{gas}-em-anthro_input4MIPs_emissions_CMIP7_IIASA-REMIND-MAgPIE-3.5-4.11-SSP1---Very-Low-Emissions_gn_202301-210012.nc"
time = '2023-01-16'
sector = 'Waste'

def plot_maps(ds, gas, ncols=1, proj=ccrs.Robinson()): 

    nrows = 1

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(6 * ncols, 4.5 * nrows),
        subplot_kw={"projection": ccrs.Robinson()}
    )

    # Flatten axes safely
    if isinstance(axes, np.ndarray):
        axes = axes.flatten() # make indexing easier
    else:
        axes = [axes]

    var = f"{gas}_em_anthro"
    da = ds[var].squeeze()

    # Plot directly with xarray's .plot.pcolormesh
    da.plot.pcolormesh(
        ax=axes[0],
        transform=ccrs.PlateCarree(),
        cmap="GnBu",
        robust=True,
        cbar_kwargs={"orientation": "horizontal", "shrink": 0.65},
    )
    axes[0].set_title(f"{results_file}\n time: {time}, sector: {sector}")
    axes[0].coastlines()

    plt.tight_layout()
    plt.show()

ds = xr.open_dataset(path_new / results_file)
plot_maps(
    ds.sel(sector=sector,time=time),
    gas
)

# %%

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
proxies

# %%
ds = proxies["anthro_CO2"]
print(ds.coords['gas'].values)

# %%
# ds = proxies["anthro_SO2"]
# ds = ds.assign_coords(gas=("gas", ["Sulfur"]))
# ds.to_netcdf(os.path.join(path, "anthro_SO2.nc"))

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
proxies_shipping[""]

# %%
for name, ds in proxies_shipping.items():
    print(f"{name}")
    for var_name, da in ds.data_vars.items():
        print(f"   {var_name}: {da.dims}")

# %%

# %%
for name, ds in proxies_shipping.items():
    first_var = list(ds.data_vars)[0]
    plot_slice_on_map(ds, first_var, year_idx=0, month_idx=1, title=name)

# %%
for name, ds in proxies_shipping.items():
    first_var = list(ds.data_vars)[0]
    # print(ds.data_vars)
    if ds.gas == "Sulfur":
        # print(ds)
        plot_slice_on_map(ds, first_var, year_idx=0, month_idx=1, title=name)
        plot_slice_on_map(ds, first_var, year_idx=-1, month_idx=1, title=name)
        plot_slice_on_map(ds, first_var, year_idx=-2, month_idx=1, title=name)
        plot_slice_on_map(ds, first_var, year_idx=10, month_idx=1, title=name)

# %% [markdown]
# ## Aircraft

# %%
files = glob.glob(os.path.join(path, "air*.nc"))
proxies_aircraft = {
    os.path.splitext(os.path.basename(fp))[0]: xr.open_dataset(fp)
    for fp in files
}
proxies_aircraft["aircraft_CO2"]

# %%
ds.sel(level=0.305).squeeze()

# %%
for name, ds in proxies_aircraft.items():
    first_var = list(ds.data_vars)[0]
    # print(ds.data_vars)
    if ds.gas == "Sulfur":
        # print(ds.level)
        ds = ds.sel(level=0.305) # close to the surface
        plot_slice_on_map(ds, first_var, year_idx=0, month_idx=1, title=name)
        plot_slice_on_map(ds, first_var, year_idx=-1, month_idx=1, title=name)
        plot_slice_on_map(ds, first_var, year_idx=-2, month_idx=1, title=name)
        plot_slice_on_map(ds, first_var, year_idx=10, month_idx=1, title=name)

# %%
for name, ds in proxies_aircraft.items():
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
