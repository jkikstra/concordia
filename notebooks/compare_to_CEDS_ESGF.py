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
import dask
from dask import delayed, compute
from dask.diagnostics import ProgressBar
from dask.utils import SerializableLock

import altair as alt
alt.renderers.enable('default')
import seaborn as sns

import cartopy.crs as ccrs
from matplotlib import colors

from itertools import product
import cftime
import numpy as np

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import cartopy.feature as cfeature # for country borders

from matplotlib import colors
import matplotlib.cm as cm


# from concordia.cmip7 import utils as cmip7_utils

IAMC_COLS = ["model", "scenario", "region", "variable", "unit"] 


# %%
lock = SerializableLock()

# %% [markdown]
# **Observations**
# (_Last update: 04/08/2028_)
#
# Notes: 
# * Check units of NC file; some scalar difference with IAMC-standard output. I assume it is in kg/m2/s ? --> DONE
#
# To be changed:
# * International shipping: produces unexpected & unwanted 2015/2020 numbers, and zeroes for 2023, 2024, 2025 --> DONE
#
# Checked to be correct:
# * Order-of-magnitude - new scenario vs CMIP6 scenario | anthro | CO2

# %% [markdown]
# **Ideas:**
#
# What data to test:
# * loop over:
#   * all scenarios 
#   * all species
#   * all sectors
# * compare to:
#   * CEDS CMIP7 ESGF data
#   * CMIP6 scenario data
#
# Tests to perform:
#
# _Automated_:
# * add up global annual total; compare to harmonized input data; should be the same
# * compare base year numbers with CEDS ESGF numbers; should be the same for the year 2023 (anthro; check by sector; check by total)
#     * potentially simpler earlier version: 'order-of-magnitude test' record the order of magnitude of all CEDS files, and check that our starting point data is same order of magnitude
# * latitudinal plot: summarise along longitude, and check those values against 
#
# _Other_:
# * compare outputs of Annika & Jarmo; both run individually on their laptops
#
# Visualisations to perform:
# * (global) direct timeseries plotting, write out as PDFs
# * (global) plot CEDS-history, GFED-history, and scenarios
# * (latitudinal) plot CEDS-history, GFED-history, and scenarios, for multiple time slices
# * (gridpoint) pick a few gridpoints, or group of gridpoints, and plot them from 1980 until 2050
# * (gridpoint) make a map of per-gridpoint differences for 2023 with CEDS
# * ...
#


# %% [markdown]
# # Paths, definitions

# %%
# Gridded scenario output
# GRIDDING_VERSION = "config_cmip7_v0_2_WSTfix_remind" # jarmo 21.08.2025 (third go, with updated hist, with fixed Waste)
# #GRIDDING_VERSION = "config_cmip7_v0_3_GCAM"
# path_scen_cmip7 = Path(f"C:/Users/kikstra/IIASA/ECE.prog - Documents/Projects/CMIP7/IAM Data Processing/Shared emission fields data/v0_2/{GRIDDING_VERSION}") # gridding output
GRIDDING_VERSION = "config_cmip7_v0_2_CEDSnc_remind" # jarmo 31.08.2025 (fourth go, based on Annika's CEDS-ESGF fixes, but with CDR)
# GRIDDING_VERSION = "config_cmip7_v0_2_CEDSnc_remind_only_CO2" # jarmo 30.08.2025 (fourth go, but with CDR)
#GRIDDING_VERSION = "config_cmip7_v0_3_GCAM"
path_scen_cmip7 = Path(f"C:/Users/kikstra/Documents/GitHub/concordia/results/{GRIDDING_VERSION}") # gridding output

# CEDS (CMIP7)
path_ceds_cmip7 = Path(f"C:/Users/kikstra/IIASA/ECE.prog - Documents/Projects/CMIP7/IAM Data Processing/ESGF/CEDS/CMIP7_anthro") 

# where to save plots of this script  
plots_path = path_scen_cmip7 / "plots"
plots_path.mkdir(exist_ok=True, parents=True)

# %%
SECTORS_ANTHRO = [
    '**International Shipping', 
    '**Agriculture', # note: is set to zero in the gridding, for co2
    '**Energy Sector', 
    '**Industrial Sector',
    '**Residential Commercial Other',
    '**Solvents Production and Application',
    '**Transportation Sector',
    '**Waste',
    '**Other non-Land CDR',
    '**BECCS'
]
SECTORS_AIR = [
    '**Aircraft'
]
SECTORS_OPENBURNING = [
    '**Agricultural Waste Burning',
    '**Forest Burning',
    '**Grassland Burning', 
    '**Peat Burning'
]

# %%
sector_dict = {
"Energy Sector": "Energy",
"Industrial Sector": "Industrial",
"Residential Commercial Other": "Residential, Commercial, Other",
"Transportation Sector": "Transportation"
}

# %%
MODEL_SELECTION = "REMIND-MAgPIE 3.5-4.11"
SCENARIO_SELECTION = "SSP1 - Very Low Emissions"
#MODEL_SELECTION = "GCAM 7.1 scenarioMIP"
#SCENARIO_SELECTION = "SSP3 - High Emissions"
MODEL_SELECTION_GRIDDED = MODEL_SELECTION.replace(" ", "-")
SCENARIO_SELECTION_GRIDDED = SCENARIO_SELECTION.replace(" ", "-")

# %% [markdown]
# # Functions

# %% [markdown]
# ## Reading in


# %%
def ds_reformat_cmip6_to_cmip7(ds):
    cmip6_sectors = { 
        0: "Agriculture", 
        1: "Energy",
        2: "Industrial",
        3: "Transportation",
        4: "Residential, Commercial, Other",
        5: "Solvents Production and Application", 
        6: "Waste", 
        7: "International Shipping",
        8: "Negative CO2 Emissions"
    }
    
    # scen_ds_cmip6
    sector_vals  = ds['sector'].values
    # Replace with string labels
    sector_labels = np.array([cmip6_sectors[i] for i in sector_vals], dtype=object)
    # Assign new sector coordinate with object dtype
    scen_ds_cmip6_named = ds.assign_coords(sector=("sector", sector_labels))

    return scen_ds_cmip6_named


# %%
def read_nc_file(f, loc, reorder_list=None, rename_sectors_cmip6=None, chunks={"time": 1}):
    ds = xr.open_dataset(
        loc / f,
        engine="netcdf4",
        chunks=chunks,
        # lock=lock
    )
    
    if reorder_list is not None:
        ds = ds[reorder_list]
    
    if rename_sectors_cmip6:
        ds = ds_reformat_cmip6_to_cmip7(ds)
    
    return ds

# %% [markdown]
# ## Aggregation

# %%
def df_to_wide_timeseries(da):

    df = da.to_pandas()

    if df.index.name == "year":
        df = df.transpose()

    return df

# %% [markdown]
# ## Aggregation to global total and unit conversion

# %%
# sample variable
var = "CO2-em-anthro"

# %%
# load a CMIP7 scenario sample file
scen_cmip7_data_file = f"{var}_input4MIPs_emissions_CMIP7_IIASA-{MODEL_SELECTION_GRIDDED}-{SCENARIO_SELECTION_GRIDDED}_gn_202301-210012.nc"

scen_ds = read_nc_file(
    f = scen_cmip7_data_file,
    loc = path_scen_cmip7
)

# %%
# load a CMIP7 CEDS sample file
ceds_cmip7_data_file = f"{var}_input4MIPs_emissions_CMIP_CEDS-CMIP-2025-04-18_gn_200001-202312.nc"

ceds_ds = read_nc_file(
    f = ceds_cmip7_data_file,
    loc = path_ceds_cmip7,  
    rename_sectors_cmip6 = True
)

# %%
# plot function


def shifted_white_colormap(cmap_name="GnBu", vmin=None, vmax=None, vcenter=0):
    cmap = plt.get_cmap(cmap_name)
    new_cmap = cmap(np.linspace(0, 1, 256))

    # Optionally, set the color corresponding to the midpoint (0) to white
    midpoint_index = 128  # halfway in 256-color map
    new_cmap[midpoint_index] = [1, 1, 1, 1]  # RGBA for white

    # Handle case where all values are zero (vmin=vmax=vcenter=0)
    if vmin == vmax == vcenter == 0:
        vmin, vmax = -1, 1  # Set default range when all zeros
    
    return colors.ListedColormap(new_cmap), colors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)


def plot_ceds_vs_scenario_comparison(ceds_da, scen_da, gas, sectors, time_slice,
                                     figsize_per_panel=(4, 3), proj=ccrs.Robinson(),
                                     colour_scale_max_percentile = 98,
                                     empty_treatment="fill_zeroes" # alternative: "skip"
                                     ):
    """
    Plot comparison between CEDS and scenario data in 4 columns:
    1. CEDS data
    2. CMIP7 scenario data  
    3. Absolute difference (CEDS - scenario)
    4. Percentage difference ((CEDS - scenario)/CEDS * 100)
    
    Parameters:
    - ceds_da: xarray Dataset with CEDS data
    - scen_da: xarray Dataset with scenario data  
    - gas: string, gas species name (e.g., 'CO', 'CO2')
    - sectors: list of sector names to plot
    - time_slice: time coordinate to select
    - figsize_per_panel: tuple, size of each subplot panel
    - proj: cartopy projection
    """
    
    # Ensure datasets have monotonic time coordinates
    ceds_da = ceds_da.sortby('time')
    scen_da = scen_da.sortby('time')
    
    # Remove any duplicate time coordinates
    _, unique_indices = np.unique(ceds_da.time.values, return_index=True)
    ceds_da = ceds_da.isel(time=np.sort(unique_indices))

    _, unique_indices = np.unique(scen_da.time.values, return_index=True)  
    scen_da = scen_da.isel(time=np.sort(unique_indices))
    
    nrows = len(sectors)
    ncols = 4
    
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
        subplot_kw={"projection": proj}
    )
    
    # Ensure axes is 2D array
    if nrows == 1:
        axes = axes.reshape(1, -1)
    
    # Calculate difference data and ensure it's also properly sorted
    diff_da = ceds_da - scen_da
    diff_da = diff_da.sortby('time')
    
    # Remove any duplicate time coordinates from diff_da as well
    _, unique_indices = np.unique(diff_da.time.values, return_index=True)
    diff_da = diff_da.isel(time=np.sort(unique_indices))
    
    # Column titles
    col_titles = ['CEDS Data', 
                  f'{gas} CMIP7 Scenario', 
                  'Difference (CEDS - Scenario)', 
                  'Percentage Difference (%)']
    
    for row, sector in enumerate(sectors):
        # Check if sector exists in both datasets
        sector_in_ceds = sector in ceds_da.sector.values
        sector_in_scen = sector in scen_da.sector.values

        if sector in ['Other non-Land CDR', 'BECCS'] and gas != "CO2":
            continue
        
        if empty_treatment=="skip":
            # Skip missing sectors (uncomment to use)
            if not sector_in_ceds or not sector_in_scen:
                print(f"Skipping sector '{sector}' - missing in {'CEDS' if not sector_in_ceds else 'scenario'} data")
                # Skip this row by plotting empty axes
                for col in range(ncols):
                    ax = axes[row, col]
                    ax.set_visible(False)
                continue
        
        # Find the closest time index manually to avoid monotonic index issues
        time_diff = np.abs(ceds_da.time - time_slice)
        closest_time_idx = time_diff.argmin().item()
        
        if empty_treatment=="fill_zeroes":
            # Option 2: Fill missing sectors with zeros (default)
            if sector_in_ceds:
                ceds_sector = ceds_da.sel(sector=sector).isel(time=closest_time_idx).squeeze()
            else:
                print(f"Warning: Sector '{sector}' not found in CEDS data, using zeros")
                # Create zero array with same spatial dimensions as other data
                template_sector = ceds_da.isel(sector=0, time=closest_time_idx).squeeze()
                ceds_sector = xr.zeros_like(template_sector)
                
            if sector_in_scen:
                scen_sector = scen_da.sel(sector=sector).isel(time=closest_time_idx).squeeze()
            else:
                print(f"Warning: Sector '{sector}' not found in scenario data, using zeros")
                # Create zero array with same spatial dimensions as other data
                template_sector = scen_da.isel(sector=0, time=closest_time_idx).squeeze()
                scen_sector = xr.zeros_like(template_sector)
            
        # Calculate difference (will handle zero arrays correctly)
        diff_sector = ceds_sector - scen_sector
        
        # Get the actual data arrays
        var_name = f'{gas}_em_anthro'
        ceds_values = ceds_sector[var_name] if var_name in ceds_sector else ceds_sector[list(ceds_sector.data_vars)[0]]
        scen_values = scen_sector[var_name] if var_name in scen_sector else scen_sector[list(scen_sector.data_vars)[0]]  
        diff_values = diff_sector[var_name] if var_name in diff_sector else diff_sector[list(diff_sector.data_vars)[0]]
        
        # Calculate percentage difference, handling division by zero
        pct_diff = xr.where(ceds_values != 0, (diff_values / ceds_values) * 100, 0)
        
        # Plot data in 4 columns
        datasets = [ceds_values, scen_values, diff_values, pct_diff]
        cmaps = ['Reds', 'Blues', 'RdBu_r', 'coolwarm']
        
        for col, (data, cmap) in enumerate(zip(datasets, cmaps)):
            ax = axes[row, col]
            
            # Set colormap normalization
            # --> think about also using shifted_white_colormap() for 2 and 3 (maybe even for 0 and 1?)
            if col in [0, 1]:
                # Get valid (non-NaN) values for percentile calculation
                valid_ceds_values = ceds_values.values[~np.isnan(ceds_values.values)]
                
                # Handle case where no valid values exist
                if len(valid_ceds_values) == 0:
                    vmin_auto, vmax_auto = 0.0, 1.0
                else:
                    vmin_auto = float(np.percentile(valid_ceds_values, 2)) # generally should be zero (except for negative emissions)
                    vmax_auto = float(np.percentile(valid_ceds_values, colour_scale_max_percentile)) # normally 98 to ensure that point-sources are not dominating the (linear) colour scale
                
                print(f"Column {col}: linear vmin={vmin_auto:.2e}, vmax={vmax_auto:.2e}")
                
                # Handle case where all values are zero
                if vmax_auto == vmin_auto:  # All values are the same (likely zero)
                    norm = colors.Normalize(vmin=0, vmax=1)  # Default range when all zeros
                else:
                    norm = colors.Normalize(vmin=vmin_auto, vmax=vmax_auto)  # Simple linear normalization from min to max
            elif col == 2:
                # Get valid (non-NaN) values for percentile calculation
                valid_ceds_values = ceds_values.values[~np.isnan(ceds_values.values)]
                
                # Handle case where no valid values exist
                if len(valid_ceds_values) == 0:
                    vmax = 1.0
                else:
                    vmax = float(np.percentile(valid_ceds_values, 98)) # use CEDS min/max for colourbar
                
                # vmin, vmax = float(data.min()), float(data.max())
                
                # Handle case where all values are zero
                if vmax == 0:
                    norm = colors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1) # set default range when all zeros
                else:
                    norm = colors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax) # set colourbar to maximum value of CEDS
            elif col == 3:  # Percentage difference - center at 0
                # abs_max = max(abs(float(data.min())), abs(float(data.max())))
                abs_max = 100 # cap out the colour bar at 50%
                # norm = colors.TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
                cmap, norm = shifted_white_colormap("coolwarm", vmin=-abs_max, vmax=abs_max)
            else:
                norm = None
            
            # Create the plot
            im = data.plot.pcolormesh(
                ax=ax,
                transform=ccrs.PlateCarree(),
                cmap=cmap,
                norm=norm,
                robust=True, # 2-98th percentile
                add_colorbar=False,
                add_labels=False
            )
            
            # Add coastlines and formatting
            ax.coastlines(linewidth=0.5)
            ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5)
            
            # Set title for first row
            if row == 0:
                ax.set_title(col_titles[col], fontsize=10, fontweight='bold')
            
            # Set sector label for first column
            if col == 0:
                ax.text(-0.15, 0.5, sector, transform=ax.transAxes, 
                       rotation=90, va='center', ha='center', fontsize=10, fontweight='bold')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, orientation='horizontal', 
                              shrink=0.8, pad=0.05, aspect=20)
            cbar.ax.tick_params(labelsize=8)
            
            # Set colorbar label
            if col == 3:
                cbar.set_label('Percentage Difference (%)', fontsize=8)
            else:
                cbar.set_label(f'{gas} emissions (kg/m²/s)', fontsize=8)
    
    # Overall title
    fig.suptitle(f'{gas}, time: {time_slice}', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93, left=0.08)
    return fig

# %%
# Timeseries Plots 
def plot_place_timeseries(ceds_ds, scen_ds, 
                          lat=39.9042, lon=116.4074,
                          place='Beijing',
                          gas='CO', sector='Energy'):
    """
    Plot timeseries for PLACE (e.g. Beijing) gridpoint comparing CEDS and scenario data
    
    Parameters:
    - ceds_ds: CEDS dataset
    - scen_ds: Scenario dataset  
    - gas: Gas species (e.g., 'CO', 'CO2')
    - sector: Sector name
    """
    
    # Select closest gridpoint to PLACE for both datasets
    ceds_place = ceds_ds.sel(
        lat=lat, 
        lon=lon, 
        method="nearest"
    ).sel(sector=sector)
    
    scen_place = scen_ds.sel(
        lat=lat,
        lon=lon, 
        method="nearest"
    ).sel(sector=sector)
    
    # Get the variable name
    var_name = f'{gas}_em_anthro'
    
    # Extract the timeseries data
    ceds_ts = ceds_place[var_name]
    scen_ts = scen_place[var_name]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot both timeseries
    ceds_ts.plot(ax=ax, label='CEDS Historical', marker='o', linewidth=2)
    scen_ts.plot(ax=ax, label='CMIP7 Scenario', marker='s', linewidth=2)
    
    # Get actual coordinates of selected gridpoint
    actual_lat = float(ceds_place.lat.values)
    actual_lon = float(ceds_place.lon.values)
    
    # Formatting
    ax.set_title(f'{gas} Emissions - {place} gridpoint\n'
                f'Sector: {sector}\n'
                f'Gridpoint: {actual_lat:.2f}°N, {actual_lon:.2f}°E', 
                fontsize=14)
    ax.set_xlabel('Time')
    ax.set_ylabel(f'{gas} emissions (kg/m²/s)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Print some info
    print(f"{place} coordinates: {lat}°N, {lon}°E")
    print(f"Selected gridpoint: {actual_lat:.2f}°N, {actual_lon:.2f}°E")
    print(f"Distance: ~{np.sqrt((actual_lat-lat)**2 + (actual_lon-lon)**2)*111:.1f} km")
    
    return fig, ax

# Alternative: Select multiple nearby gridpoints and average them
def plot_place_area_average_timeseries(ceds_ds, scen_ds, gas='CO', sector='Energy',
                                       lat=39.9042, lon=116.4074,
                                       place='Beijing',
                                       lat_range=1.0, lon_range=1.0):
    """
    Plot timeseries for PLACE (e.g., Beijing) area (average of nearby gridpoints)
    
    Parameters:
    - lat_range, lon_range: degrees around PLACE to include in average
    """
    
    # Define bounding box around PLACE
    lat_min = lat - lat_range/2
    lat_max = lat + lat_range/2
    lon_min = lon - lon_range/2  
    lon_max = lon + lon_range/2
    
    # Select area around PLACE
    ceds_area = ceds_ds.sel(
        lat=slice(lat_min, lat_max),
        lon=slice(lon_min, lon_max),
        sector=sector
    )
    
    scen_area = scen_ds.sel(
        lat=slice(lat_min, lat_max),
        lon=slice(lon_min, lon_max), 
        sector=sector
    )
    
    # Get variable name
    var_name = f'{gas}_em_anthro'
    
    # Average over the spatial area
    ceds_ts = ceds_area[var_name].mean(dim=['lat', 'lon'])
    scen_ts = scen_area[var_name].mean(dim=['lat', 'lon'])
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot both timeseries
    ceds_ts.plot(ax=ax, label='CEDS Historical', marker='o', linewidth=2)
    scen_ts.plot(ax=ax, label='CMIP7 Scenario', marker='s', linewidth=2)
    
    # Formatting
    ax.set_title(f'{gas} Emissions - {place} Area Average\n'
                f'Sector: {sector}\n'
                f'Area: {lat_range}° × {lon_range}° around {place}', 
                fontsize=14)
    ax.set_xlabel('Time')
    ax.set_ylabel(f'{gas} emissions (kg/m²/s)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Print some info
    n_gridpoints = len(ceds_area.lat) * len(ceds_area.lon)
    print(f"{place} area: {lat_range}° × {lon_range}°")
    print(f"Number of gridpoints averaged: {n_gridpoints}")
    
    return fig, ax

def plot_place_multisector_timeseries(ceds_ds, scen_ds, 
                                      lat=39.9042, lon=116.4074,
                                      place='Beijing',
                                      gas='CO', sectors=None):
    """
    Plot timeseries for multiple sectors at a specific location
    
    Parameters:
    - ceds_ds: CEDS dataset
    - scen_ds: Scenario dataset  
    - lat, lon: Coordinates of location
    - place: Name of location for title
    - gas: Gas species (e.g., 'CO', 'CO2')
    - sectors: List of sector names to plot
    """
    if sectors is None:
        sectors = ['Energy', 'Transportation', 'Waste']  # Default sectors
    
    # Select closest gridpoint
    ceds_place = ceds_ds.sel(lat=lat, lon=lon, method="nearest")
    scen_place = scen_ds.sel(lat=lat, lon=lon, method="nearest")
    
    # Get actual coordinates
    actual_lat = float(ceds_place.lat.values)
    actual_lon = float(ceds_place.lon.values)
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Variable name
    var_name = f'{gas}_em_anthro'
    
    # Colors for different sectors
    colors = cm.get_cmap('tab10')(np.linspace(0, 1, len(sectors)))
    
    # Plot CEDS data (top panel)
    for i, sector in enumerate(sectors):
        if sector in ceds_place.sector.values:
            ceds_ts = ceds_place.sel(sector=sector)[var_name]
            ceds_ts.plot(ax=ax1, label=f'CEDS - {sector}', 
                        color=colors[i], marker='o', linewidth=2, markersize=4)
    
    # Plot scenario data (bottom panel)  
    for i, sector in enumerate(sectors):
        if sector in scen_place.sector.values:
            scen_ts = scen_place.sel(sector=sector)[var_name]
            scen_ts.plot(ax=ax2, label=f'CMIP7 - {sector}', 
                        color=colors[i], marker='s', linewidth=2, markersize=4)
    
    # Format top panel
    ax1.set_title(f'{gas} Emissions - {place} (CEDS Historical)\n'
                 f'Gridpoint: {actual_lat:.2f}°N, {actual_lon:.2f}°E', fontsize=12)
    ax1.set_ylabel(f'{gas} emissions (kg/m²/s)')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('')  # Remove x-label from top panel
    
    # Format bottom panel
    ax2.set_title(f'{gas} Emissions - {place} (CMIP7 Scenario)', fontsize=12)
    ax2.set_xlabel('Time')
    ax2.set_ylabel(f'{gas} emissions (kg/m²/s)')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig, (ax1, ax2)

def plot_place_multisector_combined_timeseries(ceds_ds, scen_ds, 
                                              lat=39.9042, lon=116.4074,
                                              place='Beijing',
                                              gas='CO', sectors=None):
    """
    Plot timeseries for multiple sectors with CEDS and scenario on same plot
    
    Parameters:
    - ceds_ds: CEDS dataset
    - scen_ds: Scenario dataset  
    - lat, lon: Coordinates of location
    - place: Name of location for title
    - gas: Gas species (e.g., 'CO', 'CO2')
    - sectors: List of sector names to plot
    """
    if sectors is None:
        sectors = ['Energy', 'Transportation', 'Waste']  # Default sectors
    
    # Select closest gridpoint
    ceds_place = ceds_ds.sel(lat=lat, lon=lon, method="nearest")
    scen_place = scen_ds.sel(lat=lat, lon=lon, method="nearest")
    
    # Get actual coordinates
    actual_lat = float(ceds_place.lat.values)
    actual_lon = float(ceds_place.lon.values)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Variable name
    var_name = f'{gas}_em_anthro'
    
    # Colors for different sectors
    colors = cm.get_cmap('tab10')(np.linspace(0, 1, len(sectors)))
    
    # Plot both CEDS and scenario data for each sector
    for i, sector in enumerate(sectors):
        if sector in ceds_place.sector.values:
            # CEDS data (solid line, circles)
            ceds_ts = ceds_place.sel(sector=sector)[var_name]
            ceds_ts.plot(ax=ax, label=f'CEDS - {sector}', 
                        color=colors[i], marker='o', linewidth=2, 
                        markersize=4, linestyle='-')
            
            # Scenario data (dashed line, squares)
            if sector in scen_place.sector.values:
                scen_ts = scen_place.sel(sector=sector)[var_name]
                scen_ts.plot(ax=ax, label=f'CMIP7 - {sector}', 
                            color=colors[i], marker='s', linewidth=2, 
                            markersize=4, linestyle='--', alpha=0.8)
    
    # Format plot
    ax.set_title(f'{gas} Emissions - {place} (Multi-Sector Comparison)\n'
                f'Gridpoint: {actual_lat:.2f}°N, {actual_lon:.2f}°E', fontsize=14)
    ax.set_xlabel('Time')
    ax.set_ylabel(f'{gas} emissions (kg/m²/s)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig, ax





# %% [markdown]
# ### Comparison with CEDS: maps and timeseries, per sector, per species



# %%
# Loop (1) 

GASES = [
    "BC", 
    "CO", 
    "CO2", 
    "NOx", "OC", 
    "Sulfur",
    "CH4","N2O", "NH3", 
    "VOC"
         ]

# Mapping for gases where the file name differs from the analysis name
GAS_FILE_MAPPING = {
    "Sulfur": "SO2",  # Sulfur data is stored in SO2 files
    "VOC": "NMVOC"
}

# Function to get the correct file gas name
def get_file_gas_name_CEDS(gas):
    return GAS_FILE_MAPPING.get(gas, gas)

# Function to get the correct variable name in the dataset
def rename_CEDS_data_variable_name(ds, gas):
    file_gas = get_file_gas_name_CEDS(gas)
    return ds.rename({f"{file_gas}_em_anthro": f"{gas}_em_anthro"})
    
SECTORS = [
    "Agriculture",
    "Energy",
    "Industrial",
    "Transportation",
    "Residential, Commercial, Other",
    "Solvents production and application",
    "Waste",
    "International Shipping",
    "Other non-Land CDR",
    "BECCS"
]
# times = [cftime.DatetimeNoLeap(2023, mon, 16) for mon in range(1, 13)]
TIMES = [
    # cftime.DatetimeNoLeap(2023, 1, 16),
    # cftime.DatetimeNoLeap(2023, 2, 15),
    cftime.DatetimeNoLeap(2023, 6, 15),
    cftime.DatetimeNoLeap(2023, 12, 16)
]

PLOTS = [
    'maps',
    'timeseries'
]

for g in GASES:
    
    # load and organise data
    # load a CMIP7 scenario file
    scen_cmip7_data_file = f"{g}-em-anthro_input4MIPs_emissions_CMIP7_IIASA-{MODEL_SELECTION_GRIDDED}-{SCENARIO_SELECTION_GRIDDED}_gn_202301-210012.nc"

    scen_ds = read_nc_file(
        f = scen_cmip7_data_file,
        loc = path_scen_cmip7
    )

    # load a CMIP7 CEDS file
    file_gas_ceds = get_file_gas_name_CEDS(g) # for different naming VOC and Sulfur
    ceds_cmip7_data_file = f"{file_gas_ceds}-em-anthro_input4MIPs_emissions_CMIP_CEDS-CMIP-2025-04-18_gn_200001-202312.nc"

    ceds_ds = read_nc_file(
        f = ceds_cmip7_data_file,
        loc = path_ceds_cmip7,  
        rename_sectors_cmip6 = True
    )
    ceds_ds = rename_CEDS_data_variable_name(ceds_ds, g) # rename data variable too, for different naming VOC and Sulfur

    ceds_da = ceds_ds.sel(time=TIMES,
                method="nearest",
                tolerance=np.timedelta64(1, 'D') # for february, where we have 15th instead of 16th
                )
    scen_da = scen_ds.sel(time=TIMES,
                method="nearest",
                tolerance=np.timedelta64(1, 'D') # for february, where we have 15th instead of 16th
                )
    # Sort by time to ensure monotonic index and remove any duplicates
    ceds_da = ceds_da.sortby('time')
    scen_da = scen_da.sortby('time')

    # Check for and remove duplicate time coordinates
    _, unique_indices = np.unique(ceds_da.time.values, return_index=True)
    ceds_da = ceds_da.isel(time=np.sort(unique_indices))

    _, unique_indices = np.unique(scen_da.time.values, return_index=True)  
    scen_da = scen_da.isel(time=np.sort(unique_indices))

    diff_da = ceds_da - scen_da

    # Ensure clean time coordinate by reindexing
    diff_da = diff_da.reindex(time=sorted(diff_da.time.values))

    print(f"diff_da time coordinate: {diff_da.time.values}")
    print(f"Time coordinate is monotonic: {np.all(diff_da.time.values[:-1] <= diff_da.time.values[1:])}")
    print(f"Number of unique times: {len(np.unique(diff_da.time.values))}")

    if 'maps' in PLOTS:
        # Use the new comparison function
        fig = plot_ceds_vs_scenario_comparison(
            ceds_da=ceds_da,
            scen_da=scen_da,
            gas=g,
            sectors=SECTORS,
            time_slice=TIMES[0],  # Use first time slice, in case more are specified
            figsize_per_panel=(4, 3),
            # colour_scale_max_percentile=100
        )
        
        # Save the plot in both PNG and PDF formats
        filename_base = f"ceds_vs_scenario_comparison_{g}_{TIMES[0].strftime('%Y%m%d')}"
        fig.savefig(plots_path / f"{filename_base}.png", 
                    dpi=300, bbox_inches='tight')
        # fig.savefig(plots_path / f"{filename_base}.pdf", 
        #             bbox_inches='tight')
        plt.show()  
    
    if 'timeseries' in PLOTS:
        # Define locations dictionary with coordinates
        LOCATIONS = {
            'Beijing': (39.9042, 116.4074),
            'Geneva': (46.2044, 6.1432),
            'Delhi': (28.6139, 77.2090),
            'Spain': (40.4637, 3.7492), # central spain, close to Madrid
            # 'New_York': (40.7128, -74.0060),
            # 'London': (51.5074, -0.1278),
            # 'Tokyo': (35.6762, 139.6503),
            # 'São_Paulo': (-23.5505, -46.6333),
            # 'Lagos': (6.5244, 3.3792),
            # 'Mumbai': (19.0760, 72.8777),
            'Rural_Amazon': (-3.4653, -62.2159),  # Remote area in Amazon
            'North_Atlantic': (45.0, -30.0),     # Shipping route
            # 'South_China_Sea': (12.0, 113.0)     # Shipping route
        }
        
        for sec in SECTORS:
            for place, (lat, lon) in LOCATIONS.items():
                print(f"\nGenerating plots for {place} ({lat:.2f}°, {lon:.2f}°) - {g} {sec}")
                
                try:
                    # Single gridpoint timeseries
                    fig1, ax1 = plot_place_timeseries(ceds_ds, scen_ds, 
                                        lat=lat, lon=lon,
                                        place=place,
                                        gas=g, sector=sec)
                    plt.savefig(plots_path / f"{place}_timeseries_{g}_{sec}.png", 
                                dpi=300, 
                                bbox_inches='tight')
                    plt.show()

                    # Area average timeseries
                    fig2, ax2 = plot_place_area_average_timeseries(ceds_ds, scen_ds, 
                                        lat=lat, lon=lon,
                                        place=place,
                                        gas=g, sector=sec,
                                        lat_range=2.0, lon_range=2.0)
                    plt.savefig(plots_path / f"{place}_area_timeseries_{g}_{sec}.png", 
                                dpi=300, 
                                bbox_inches='tight')
                    plt.show()
                    
                except Exception as e:
                    print(f"Error plotting {place} {g} {sec}: {e}")
                    continue
                    
        # Also create multi-sector plots for each location
        print(f"\nGenerating multi-sector plots for {g}")
        for place, (lat, lon) in LOCATIONS.items():
            try:
                # Multi-sector combined plot
                fig3, ax3 = plot_place_multisector_combined_timeseries(
                    ceds_ds, scen_ds, 
                    lat=lat, lon=lon, place=place,
                    gas=g, sectors=SECTORS
                )
                plt.savefig(plots_path / f"{place}_multisector_combined_{g}.png", 
                            dpi=300, bbox_inches='tight')
                plt.show()
                
            except Exception as e:
                print(f"Error plotting multi-sector {place} {g}: {e}")
                continue


# %%
# Functions for just CMIP7 data, without comparison to CEDS

def plot_maps_species_times(ds, 
                              species,
                              sector_file, 
                              times,
                              ncols=3, proj=ccrs.Robinson(),
                              aircraft_level=0.305
                              ): 

    nrows = len(times)
    ncols = len(species) if isinstance(species, list) else 1
    
    # Handle single species case
    if isinstance(species, str):
        species = [species]

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(8 * ncols, 4.5 * nrows),
        subplot_kw={"projection": proj}
    )

    # Flatten axes safely
    if isinstance(axes, np.ndarray):
        axes = axes.flatten() # make indexing easier
    else:
        axes = [axes]

    for i, (time, gas) in enumerate(product(times, species)):
        
        # Sum over all sectors
        if sector_file!="AIR-anthro":
            da = (
                ds
                .sel(time=time, method="nearest")
                .sum(dim="sector")  # Sum over all sectors
                .squeeze()  
            )
            v = sector_file
        else:
            da = (
                ds 
                .sel(time=time, method="nearest")
                # .sel(level=aircraft_level)
                .sum(dim="level")
                .squeeze()  
            )
            v = "AIR_anthro" # for the variable name this is necessary
            
        da = da[f'{gas}_em_{v}'].squeeze()

        # Use TwoSlopeNorm centered at 0 and custom cmap
        vmin = float(da.min())
        vmax = float(da.max())
        # if vmin >= 0:
        if vmin >= -1e-8:
            cmap = "GnBu"
            norm = None
            rob = True
        else:
            print(f"Negative values (lowest: {vmin}) in the data for {gas}-{sector_file}")
            cmap, norm = shifted_white_colormap(
                "coolwarm",
                vmin=vmin, vmax=vmax
            )
            rob = False

        # Plot directly with xarray's .plot.pcolormesh
        da.plot.pcolormesh(
            ax=axes[i],
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            norm=norm,
            robust=rob,
            cbar_kwargs={"orientation": "horizontal", "shrink": 0.65},
        )
        axes[i].set_title(f"{gas}, time: {time}, file: {sector_file}")
        axes[i].coastlines()
        axes[i].add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5)

    # Remove any unused axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    return fig  # Return the figure for saving



# %%
# Loop (2): plot totals 

GASES = [
    "BC", 
    "CO", 
    "CO2", 
    "NOx", 
    "OC", 
    "Sulfur",
    "CH4",
    "N2O", 
    "NH3", 
    "VOC"
    ]

# times = [cftime.DatetimeNoLeap(2023, mon, 16) for mon in range(1, 13)]
TIMES = [
    # cftime.DatetimeNoLeap(2023, 1, 16),
    # cftime.DatetimeNoLeap(2023, 2, 15),
    # cftime.DatetimeNoLeap(2023, 6, 15),
    cftime.DatetimeNoLeap(2023, 12, 16),
    cftime.DatetimeNoLeap(2050, 12, 16),
    cftime.DatetimeNoLeap(2100, 12, 16)
]

PLOTS = [
    # 'maps-all-species', # not yet implemented; 1 species per file...
    'maps-per-species'
]

# Separte plots for each gas:
if 'maps-per-species' in PLOTS:
    for g in GASES:
        # load a CMIP7 scenario file

        for sector_file in [
            'anthro',
            'AIR-anthro',
            'openburning',
        ]:
            if ((g=="N2O") and (sector_file!="anthro")):
                print(f"{g}-em-{sector_file} not available")
                continue
            if ((g=="CO2") and (sector_file=="openburning")):
                print(f"{g}-em-{sector_file} not available")
                continue

            scen_cmip7_data_file = f"{g}-em-{sector_file}_input4MIPs_emissions_CMIP7_IIASA-{MODEL_SELECTION_GRIDDED}-{SCENARIO_SELECTION_GRIDDED}_gn_202301-210012.nc"

            scen_ds = read_nc_file(
                f = scen_cmip7_data_file,
                loc = path_scen_cmip7
            )

            fig = plot_maps_species_times(ds=scen_ds,
                                    species=g,
                                    sector_file=sector_file,
                                    times=TIMES)
            
            # Save the figure
            filename = f"total_emissions_map_{g}_em_{sector_file}_{MODEL_SELECTION_GRIDDED}_{SCENARIO_SELECTION_GRIDDED}.png"
            fig.savefig(plots_path / filename, dpi=300, 
                        bbox_inches='tight')
            print(f"Saved: {filename}")

            # close figure to free memory
            plt.close(fig)
    