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

import altair as alt
alt.renderers.enable('default')
import seaborn as sns

from concordia.cmip7 import utils as cmip7_utils


# %%
lock = SerializableLock()

# %% [markdown]
# # Paths, definitions

# %%
GRIDDING_VERSION = "config_cmip7_v0_2" # jarmo 10.08.2025 (first go, with hist 022)
GRIDDING_VERSION = "config_cmip7_v0_2_newhistory_remind" # jarmo 10.08.2025 (second go, with updated hist)
GRIDDING_VERSION = "config_cmip7_v0_2_WSTfix_remind" # jarmo 21.08.2025 (third go, with updated hist, with fixed Waste)
GRIDDING_VERSION = "config_cmip7_v0_2_CEDS_proxies" # annika 27.08.2025 (with proxies derived from CEDS directly for anthro)
GRIDDING_VERSION = "config_cmip7_v0_2_CEDS_proxies_new_AIR" # annika 28.08.2025 (now also for aircraft)
GRIDDING_VERSION = "config_cmip7_v0_2_CEDS_proxies_compressed" # annika 28.08.2025 (including encoding for compression)

# Scenarios pre-gridding
# scenario_data_location = "C:/Users/kikstra/IIASA/ECE.prog - Documents/Projects/CMIP7/IAM Data Processing/concordia_cmip7_v0_2/input/scenarios/August 08 submission/" # harmonized in emissions_harmonization_historical
scenario_data_location = "/home/hoegner/Projects/CMIP7/input/scenarios/"

harmonized_data_location = Path(f"/home/hoegner/GitHub/concordia/results/{GRIDDING_VERSION}")
#harmonized_data_location = Path(f"C:/Users/kikstra/documents/GitHub/concordia/results/{GRIDDING_VERSION}") # (re-) harmonized by concordia

# gridded emissions
# gridding input files
grid_file_location = "/home/hoegner/Projects/CMIP7/input/gridding/"
# grid_file_location = "C:/Users/kikstra/IIASA/ECE.prog - Documents/Projects/CMIP7/IAM Data Processing/concordia_cmip7_v0_2/input/gridding/"

# CMIP7 gridded emissions
cmip7_data_location = Path(f"/home/hoegner/GitHub/concordia/results/{GRIDDING_VERSION}")
# cmip7_data_location = Path(f"C:/Users/kikstra/documents/GitHub/concordia/results/{GRIDDING_VERSION}") # gridding output

# CMIP6, for comparison
cmip6_data_location = Path("/home/hoegner/Projects/CMIP7/checks/Example NetCDF files CMIP6")
# cmip6_data_location = Path("C:/Users/kikstra/IIASA/ECE.prog - Documents/Projects/CMIP7/IAM Data Processing/ESGF/Example NetCDF files CMIP6")

plots_path = cmip7_data_location / "plots"
plots_path.mkdir(exist_ok=True, parents=True)

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
        lock=lock
    )
    
    if reorder_list is not None:
        ds = ds[reorder_list]
    
    if rename_sectors_cmip6:
        ds = ds_reformat_cmip6_to_cmip7(ds)
    
    return ds

# %% [markdown]
# ## Aggregation to global total and unit conversion

# %%
areacella = xr.open_dataset(Path(grid_file_location, "areacella_input4MIPs_emissions_CMIP_CEDS-CMIP-2025-04-18_gn.nc"))
cell_area = areacella["areacella"]


# %%
def ds_to_annual_emissions_total(gridded_data, var_name, cell_area, keep_sectors=True):
    """
    Convert gridded emissions in kg/m2/s to Mt/year.
    
    Parameters:
    - gridded_data: xr.Dataset containing the emission variable
    - var_name: str, name of the variable to convert
    - cell_area: xr.DataArray of shape (lat, lon), in m2
    - keep_sectors: bool, if True, retain sector info
    
    Returns:
    - xr.DataArray of Mt/year, shape (year,) or (sector, year)
    """
    da = gridded_data[var_name]

    # obtain the seconds in each month for which data is available
    seconds_per_month = da.time.dt.days_in_month * 24 * 60 * 60

    # kg/m2/s --> kg/m2/month
    monthly = seconds_per_month * da

    # weight with cell area
    area_weighted = cell_area * monthly

    # Sum over spatial dimensions
    sum_dims = ["lat", "lon"]
    if "level" in area_weighted.dims:
        sum_dims.append("level")

    kg_per_month = area_weighted.sum(dim=sum_dims)

    # Convert to annual totals (kg/year)
    kg_per_year = kg_per_month.groupby("time.year").sum()

    # Convert to Mt/year
    da_Mt_y = kg_per_year * 1e-9

    if "sector" in da_Mt_y.dims and not keep_sectors:
        da_Mt_y = da_Mt_y.sum(dim="sector")

    # make sure variable is correctly named
    da_Mt_y = da_Mt_y.rename(var_name)
    
    return da_Mt_y


# %% [markdown]
# ## Plotting

# %%
def plot_one_emissions_timeseries(ts,
                                  title: str = "Annual Global Anthropogenic CO2 Emissions",
                                  xlabel: str = "Year",
                                  ylabel: str = "CO2 Emissions [mass flux]",
                                  ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    ts.plot(ax=ax, marker='o')
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)


def plot_sectors_emissions_timeseries(ts,
                                      title: str = "Annual Global Anthropogenic CO2 Emissions",
                                      xlabel: str = "Year",
                                      ylabel: str = "CO2 Emissions [mass flux]",
                                      ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    for sector in ts.sector.values:
        ax.plot(ts.year, ts.sel(sector=sector), label=str(sector))
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(title="Sector", bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True)


def plot_sectors_emissions_timeseries_area_DRAFT2(ts,
                                  title: str = "Annual Global Anthropogenic CO2 Emissions",
                                  xlabel: str = "Year",
                                  ylabel: str = "CO2 Emissions [mass flux]",
                                  figsize: tuple = (10,6)):
    # Convert to pandas DataFrame for plotting
    df = ts.to_pandas()  # index: year, columns: sector

    # in case accidentally (index: sector, columns: year), transpose 
    if df.index.name == "sector":
        df = df.transpose()
        # df.index.name = "year"
        # df.columns.name = "sector"

    # in long format
    df_long = df.reset_index().melt(id_vars="year", var_name="sector", value_name="emissions")

    # Plot with Altair
    chart = alt.Chart(df_long).mark_area().encode(
        x="year:O",
        y="emissions:Q",
        color="sector:N",
        tooltip=["year", "sector", "emissions"]
    ).properties(
        title=title,
        width=800,
        height=400
    ).interactive()

    return chart


# %% [markdown]
# # Load data

# %% [markdown]
# ## CO2 example 1 scenario (CMIP7)

# %%
# load a CMIP7 sample file
cmip7_data_file = f"CO2-em-anthro_input4MIPs_emissions_CMIP7_IIASA-{MODEL_SELECTION_GRIDDED}-{SCENARIO_SELECTION_GRIDDED}_gn_202301-210012.nc"

scen_ds = read_nc_file(
    f = cmip7_data_file,
    loc = cmip7_data_location
)

# %% [markdown]
# ## CO2 example 1 scenario (CMIP6)

# %%
cmip6_data_file = "CO2-em-anthro_input4MIPs_emissions_ScenarioMIP_IAMC-MESSAGE-GLOBIOM-ssp245-1-1_gn_201501-210012.nc"

scen_ds_cmip6 = read_nc_file(
    f = cmip6_data_file,
    loc = cmip6_data_location,
    reorder_list = list(scen_ds.data_vars),
    rename_sectors_cmip6 = True
)

# %% [markdown]
# ## CEDS History (CMIP7)

# %%
ceds_data_file = "CO2-em-anthro_input4MIPs_emissions_ScenarioMIP_IAMC-MESSAGE-GLOBIOM-ssp245-1-1_gn_201501-210012.nc"

ceds_ds_cmip6 = read_nc_file(
    f = ceds_data_file,
    loc = cmip6_data_location,
    reorder_list = list(scen_ds.data_vars),
    rename_sectors_cmip6 = True
)

# %%
ceds_ds_cmip6

# %% [markdown]
# # Do checks

# %% [markdown]
# ### Some rough code for 1 scenario checking in very simple dataframes

# %%
# simple pandas-dataframe based checks & plots

# cmip7 dfs
sectoral_emissions_ts = ds_to_annual_emissions_total(scen_ds, var_name="CO2_em_anthro", cell_area=cell_area, keep_sectors=True)
total_emissions_ts = ds_to_annual_emissions_total(scen_ds, var_name="CO2_em_anthro", cell_area=cell_area, keep_sectors=False)

# cmip6 dfs
sectoral_emissions_ts_cmip6 = ds_to_annual_emissions_total(scen_ds_cmip6, var_name="CO2_em_anthro", cell_area=cell_area, keep_sectors=True).transpose() # to make it year,sector (instead of sector,year)
total_emissions_ts_cmip6 = ds_to_annual_emissions_total(scen_ds_cmip6, var_name="CO2_em_anthro", cell_area=cell_area, keep_sectors=False)


# %%
fig, axs = plt.subplots(2, 2, figsize=(15, 10), sharex=True)

plot_one_emissions_timeseries(total_emissions_ts, ax=axs[0, 0],
    title="Total Emissions (Scenario CMIP7)")
plot_sectors_emissions_timeseries(sectoral_emissions_ts, ax=axs[0, 1],
    title="Sectoral Emissions (Scenario CMIP7)")

plot_one_emissions_timeseries(total_emissions_ts_cmip6, ax=axs[1, 0],
    title="Total Emissions (CMIP6)")
plot_sectors_emissions_timeseries(sectoral_emissions_ts_cmip6, ax=axs[1, 1],
    title="Sectoral Emissions (CMIP6)")

fig.suptitle("Emissions Time Series Comparison", fontsize=16)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(Path(plots_path, f"comparison_with_CMIP6_MESSAGE_ssp245_{MODEL_SELECTION_GRIDDED}_{SCENARIO_SELECTION_GRIDDED}.png"))
plt.show()


# %%
plot_sectors_emissions_timeseries_area_DRAFT2(sectoral_emissions_ts)

# %%
plot_sectors_emissions_timeseries_area_DRAFT2(sectoral_emissions_ts_cmip6)
