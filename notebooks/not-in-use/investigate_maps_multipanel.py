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
from concordia.rescue.proxy import plot_map

# %%
rescue_data = "C:/Users/kikstra/IIASA/RESCUE - WP 1/data_2024_09_16/gridding_process_files/proxy_rasters/CDR_CO2.nc"
cdr_ds = xr.open_dataset(rescue_data)
print(cdr_ds)

# %%
cdr_ds.sector

# %%
cdr_ds.sel(sector="DAC_CDR", year=2100, month=1).squeeze().emissions

# %%
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np

def plot_maps(ds, sectors, ncols=3, year=2100, month=1, proj=ccrs.Robinson()): 

    nrows = int(np.ceil(len(sectors) / ncols))

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

    for i, sector in enumerate(sectors):
        da = (
            ds.sel(sector=sector, year=year, month=month)
            .squeeze()
            .emissions  # or whatever your variable is
        )

        # Plot directly with xarray's .plot.pcolormesh
        da.plot.pcolormesh(
            ax=axes[i],
            transform=ccrs.PlateCarree(),
            cmap="GnBu",
            robust=True,
            cbar_kwargs={"orientation": "horizontal", "shrink": 0.65},
        )
        axes[i].set_title(sector)
        axes[i].coastlines()

    # Remove any unused axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

# cdr_secs = ["DAC_CDR", "OAE_CDR", "IND_CDR", "BECCS", "A/R", "AGLAND", 'DEFOREST', 'NONURB', 'LAND']  # all CDR proxy file sectors
cdr_secs = ["DAC_CDR", "IND_CDR", "OAE_CDR"]  # non-land CDR proxy file sectors

plot_maps(cdr_ds, sectors=cdr_secs)

# %%
cdr_secs_additional = ["BECCS", "A/R", "AGLAND", 'DEFOREST', 'NONURB', 'LAND']  # all CDR proxy file sectors
# cdr_secs = ["DAC_CDR", "IND_CDR", "OAE_CDR"]  # non-land CDR proxy file sectors

plot_maps(cdr_ds, sectors=cdr_secs_additional)

# %%

# %%
ceds_shipping_data = "C:/Users/kikstra/IIASA/RESCUE - WP 1/data_2024_09_16/gridding_process_files/proxy_rasters/shipping_CO2.nc"
ceds_shipping_ds = xr.open_dataset(ceds_shipping_data)
print(ceds_shipping_ds)

# %%
plot_maps(ceds_shipping_ds, 
          sectors=['SHP'],ncols=3,
          year=2030, month=1)

# %%
ceds_anthro_data = "C:/Users/kikstra/IIASA/RESCUE - WP 1/data_2024_09_16/gridding_process_files/proxy_rasters/anthro_CO2.nc"
ceds_anthro_ds = xr.open_dataset(ceds_anthro_data)
print(ceds_anthro_ds)

# %%
ceds_anthro_ds.sector.to_numpy()

# %%
ceds_anthro_ds

# %%
ant_secs = ceds_anthro_ds.sector.to_numpy()
ant_secs_co2 = [x for x in ant_secs if x != 'AGR']
plot_maps(ceds_anthro_ds, 
          sectors=ant_secs_co2,ncols=3,
          year=2030, month=1)

# %%
pik_data = "C:/Users/kikstra/IIASA/ECE.prog - Documents/Projects/CMIP7/IAM Data Processing/Population/pik/pop-dens-SSP2_input4MIPs_population_CMIP_PIK-CMIP-1-0-0_gn_2022-2100.nc"
p_ds = xr.open_dataset(pik_data)
print(p_ds)


# %%
def plot_maps_population(ds, years=[2100], ncols=3, month=7, proj=ccrs.Robinson()): 

    nrows = int(np.ceil(len(years) / ncols))

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

    for i, year in enumerate(years):
        da = (
            ds.sel(time=f"{year}-0{month}-01")
            .squeeze()
            .pop_dens_SSP2  # or whatever your variable is
        )

        # Plot directly with xarray's .plot.pcolormesh
        da.plot.pcolormesh(
            ax=axes[i],
            transform=ccrs.PlateCarree(),
            cmap="GnBu",
            robust=True,
            cbar_kwargs={"orientation": "horizontal", "shrink": 0.65},
        )
        axes[i].set_title(year)
        axes[i].coastlines()

    # Remove any unused axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

plot_maps_population(p_ds, years=[2023, 2100])

# %% [markdown]
# ## Explore OC waste: zero OC waste in proxy? And VOC SLV?

# %% [markdown]
# ### Proxy

# %%
nox_data_proxy = "C:/Users/kikstra/IIASA/ECE.prog - Documents/Projects/CMIP7/IAM Data Processing/concordia_cmip7_v0_2/input/gridding/proxy_rasters/20250810/anthro_NOx.nc"
nox_ds_proxy = xr.open_dataset(nox_data_proxy)
print(nox_ds_proxy)

# %%
plot_maps(nox_ds_proxy, year=2023, sectors=['TRA'])

# %%
voc_data_proxy = "C:/Users/kikstra/IIASA/ECE.prog - Documents/Projects/CMIP7/IAM Data Processing/concordia_cmip7_v0_2/input/gridding/proxy_rasters/20250810/anthro_VOC.nc"
voc_ds_proxy = xr.open_dataset(voc_data_proxy)
print(voc_ds_proxy) # missing: many, e.g. 'nld' is zero where that is unexpected following aggregated data?? Actually - looks pretty fine in the plotted data: all of proxy, CEDS emissions, historical, and scenario have data - so why not the downscaled output?
# N.B.: Serbia and Kosovo VOC solvents is however unexpectedly(?) zero

# %%
plot_maps(voc_ds_proxy, year=2023, sectors=['ENE'])

# %%
co_data_proxy = "C:/Users/kikstra/IIASA/ECE.prog - Documents/Projects/CMIP7/IAM Data Processing/concordia_cmip7_v0_2/input/gridding/proxy_rasters/20250810/anthro_CO.nc"
co_ds_proxy = xr.open_dataset(co_data_proxy)
print(co_ds_proxy) # missing: many, e.g. 'pak', 'mdg'

# %%
plot_maps(co_ds_proxy, year=2023, sectors=['TRA'])

# %% [markdown]
# ### CEDS CMIP7 history

# %%
import pandas as pd
from pathlib import Path

path_ceds_cmip7 = Path("C:/Users/kikstra/IIASA/ECE.prog - Documents/Projects/CMIP7/IAM Data Processing/ESGF/CEDS/CMIP7")
ceds_cmip7_suffix = "_input4MIPs_emissions_CMIP_CEDS-CMIP-2025-04-18_gn_200001-202312.nc"
ceds_cmip7_anthro = "-em-anthro"


# %%
voc_data_rawCEDS = path_ceds_cmip7 / f"NMVOC{ceds_cmip7_anthro}{ceds_cmip7_suffix}" 
voc_ds_raw = xr.open_dataset(voc_data_rawCEDS)
voc_ds_raw

# %%
co_data_rawCEDS = path_ceds_cmip7 / f"NMVOC{ceds_cmip7_anthro}{ceds_cmip7_suffix}"
co_ds_raw = xr.open_dataset(co_data_rawCEDS)
co_ds_raw

# %%
co_ds_raw["sector_bnds"] # to show that 4: Residential, Commercial, Other; 6:Waste

# %%
co_ds_raw["time"].dt.year


# %%

# %%
def get_co(ds):
    da = (
            ds.squeeze()
            .CO_em_anthro  # or whatever your variable is
        )
    return da

def get_voc(ds):
    da = (
            ds
            .squeeze()
            .NMVOC_em_anthro  # or whatever your variable is
        )
    return da


import cartopy.feature as cfeature # for country borders

def plot_maps_ceds(ds, sectors, gas="CO", ncols=3, time="2023-12-16", proj=ccrs.Robinson()): 

    nrows = int(np.ceil(len(sectors) / ncols))

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

    for i, sector in enumerate(sectors):
        if gas == "CO": 
            da = get_co(ds.sel(sector=sector, time=time)) 
        if gas == "NMVOC":
            da = get_voc(ds.sel(sector=sector, time=time))

        # Plot directly with xarray's .plot.pcolormesh
        da.plot.pcolormesh(
            ax=axes[i],
            transform=ccrs.PlateCarree(),
            cmap="GnBu",
            robust=True,
            cbar_kwargs={"orientation": "horizontal", "shrink": 0.65},
        )
        axes[i].set_title(sector)
        axes[i].coastlines()
        axes[i].add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor="black")  # country borders

    # Remove any unused axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


# %%
co_ds_raw["sector_bnds"] # to show that 4: Residential, Commercial, Other; 6:Waste

# %%
plot_maps_ceds(co_ds_raw.sel(time='2023-12-16'), time='2023-12-16', sectors=[3,4,6], gas="CO")

# %%
plot_maps_ceds(voc_ds_raw.sel(time='2023-12-16'), sectors=[3,4,5], gas="NMVOC")

# %%
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
from pathlib import Path

def plot_maps_species(basepath, species, ncols=3, proj=ccrs.Robinson(), sector='Waste', time="2023-01-16"): 

    nrows = int(np.ceil(len(species) / ncols))

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

    for i, gas in enumerate(species):
        ds = xr.open_dataset(
            basepath / f"{gas}-em-anthro_input4MIPs_emissions_CMIP7_IIASA-REMIND-MAgPIE-3.5-4.11-SSP1---Very-Low-Emissions_gn_202301-210012.nc"
        )
        
        da = (
            ds.sel(sector=sector, time=time)
            .squeeze()  # or whatever your variable is
        )
        da = da[f'{gas}_em_anthro'].squeeze()

        # Plot directly with xarray's .plot.pcolormesh
        da.plot.pcolormesh(
            ax=axes[i],
            transform=ccrs.PlateCarree(),
            cmap="GnBu",
            robust=True,
            cbar_kwargs={"orientation": "horizontal", "shrink": 0.65},
        )
        axes[i].set_title(f"{gas}, sector: {sector}, time: {time}")
        axes[i].coastlines()

    # Remove any unused axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

GRIDDING_VERSION = "config_cmip7_v0_2_WSTfix_remind" # jarmo 21.08.2025 (third go, with updated hist, with fixed Waste)
cmip7_data_location = Path(f"C:/Users/kikstra/documents/GitHub/concordia/results/{GRIDDING_VERSION}") # gridding output

GASES = [
    # "BC", "CO", "CO2", "NOx", "OC", "Sulfur",
    "CH4","N2O", "NH3", "VOC"
         ]

plot_maps_species(basepath=cmip7_data_location, species=GASES)

# %%
from itertools import product
def plot_maps_species_sectors(basepath, species, sectors, ncols=3, proj=ccrs.Robinson(), sector='Waste', time="2023-01-16"): 

    nrows = len(species)
    ncols = len(sectors)

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

    for i, (gas, sector) in enumerate(product(species, sectors)):
        ds = xr.open_dataset(
            basepath / f"{gas}-em-anthro_input4MIPs_emissions_CMIP7_IIASA-REMIND-MAgPIE-3.5-4.11-SSP1---Very-Low-Emissions_gn_202301-210012.nc"
        )
        
        da = (
            ds.sel(sector=sector, time=time)
            .squeeze()  # or whatever your variable is
        )
        da = da[f'{gas}_em_anthro'].squeeze()

        # Plot directly with xarray's .plot.pcolormesh
        da.plot.pcolormesh(
            ax=axes[i],
            transform=ccrs.PlateCarree(),
            cmap="GnBu",
            robust=True,
            cbar_kwargs={"orientation": "horizontal", "shrink": 0.65},
        )
        axes[i].set_title(f"{gas}, sector: {sector}, time: {time}")
        axes[i].coastlines()

    # Remove any unused axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

# 0: Agriculture; 1: Energy; 2: Industrial; 3: Transportation; 4: Residential, Commercial, Other; 5: Solvents production and application; 6: Waste; 7: International Shipping
GRIDDING_VERSION = "config_cmip7_v0_2_WSTfix_remind" # jarmo 21.08.2025 (third go, with updated hist, with fixed Waste)
cmip7_data_location = Path(f"C:/Users/kikstra/documents/GitHub/concordia/results/{GRIDDING_VERSION}") # gridding output

GASES = [
    "BC", "CO", "CO2", "NOx", "OC", "Sulfur",
    # "CH4","N2O", "NH3", "VOC"
         ]
SECTORS = [
    # "Agriculture",
    "Energy",
    "Industrial",
    "Transportation",
    "Residential, Commercial, Other",
    # "Solvents production and application",
    "Waste",
    "International Shipping"
]

plot_maps_species_sectors(basepath=cmip7_data_location, 
                          species=GASES, 
                          sectors=SECTORS)
