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
from dask import delayed, compute
from dask.diagnostics import ProgressBar
from tqdm import tqdm
import altair as alt
alt.renderers.enable('default')
import seaborn as sns
from concordia.cmip7 import utils as cmip7_utils

IAMC_COLS = ["model", "scenario", "region", "variable", "unit"] 


# %% [markdown]
# **Observations**
# (_Last update: 04/08/2028_)
#
# Notes: 
# * Check units of NC file; some scalar difference with IAMC-standard output. I assume it is in kg/m2/s ?
#
# To be changed:
# * International shipping: produces unexpected & unwanted 2015/2020 numbers, and zeroes for 2023, 2024, 2025
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
# * ...
#


# %% [markdown]
# # Paths, definitions

# %%
# Scenarios pre-gridding
# harmonized_data_location = "C:/Users/kikstra/IIASA/ECE.prog - Documents/Projects/CMIP7/IAM Data Processing/concordia_cmip7_v0_testing/input/scenarios/"
harmonized_data_location = "/home/hoegner/Projects/CMIP7/input/scenarios/"

# gridded emissions
# CMIP7
GRIDDING_VERSION = "config_cmip7_v0_2_testing_new_proxies"

#cmip7_data_location = Path("C:/Users/kikstra/Documents/GitHub/concordia/results/config_cmip7_v0_2_testing_ukesm_remind-ah")
cmip7_data_location = Path(f"/home/hoegner/GitHub/concordia/results/{GRIDDING_VERSION}")

# CMIP6, for comparison
#cmip6_data_location = Path("C:/Users/kikstra/IIASA/ECE.prog - Documents/Projects/CMIP7/IAM Data Processing/ESGF/Example NetCDF files CMIP6")
cmip6_data_location = Path("/home/hoegner/Projects/CMIP7/checks/Example NetCDF files CMIP6")

# %%
SECTORS_ANTHRO = [
    '**International Shipping', 
    '**Transportation Sector',
    '**Waste',
    '**Agriculture',
    '**Energy Sector', 
    '**Industrial Sector',
    '**Residential Commercial Other',
    '**Solvents Production and Application'
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
def read_nc_file(f, loc, reorder_list=None, rename_sectors_cmip6=None, chunks={'time': 1}):
    ds = xr.open_dataset(loc / f, chunks=chunks)
    
    if reorder_list is not None:
        ds = ds[reorder_list]
    
    if rename_sectors_cmip6:
        ds = ds_reformat_cmip6_to_cmip7(ds)
    
    return ds

# %% [markdown]
# ## Aggregation

# %%
def gridcell_area(lat, lon, radius=6.371e6, cell_degree=0.5):
    """
    Approximate area of a lat-lon grid cell (in m²) assuming spherical Earth.
    lat, lon in degrees, assumed to be 1D arrays.
    """
    dlat = np.deg2rad(cell_degree)
    dlon = np.deg2rad(cell_degree)

    lat_rad = np.deg2rad(lat)
    # Area = R² * dlat * dlon * cos(lat)
    area = (radius**2) * dlat * dlon * np.cos(lat_rad)
    return xr.DataArray(area, coords={"lat": lat}, dims=["lat"])


# %%
def ds_to_annual_emissions_sectoral(ds, variable_name):
    """
    Take an output scenario and return an annual emissions timeseries.
    Input is in kg m-2 s-1, output is in kg s-1 (global)
    """
    # Step 1: compute cell area
    area_da = gridcell_area(ds.lat, ds.lon)  # shape: (lat,)
    
    # TO DO: for AIR this should probably be summed over all levels?
    if "AIR" in variable_name:
        area_grid = area_da.broadcast_like(ds[variable_name].isel(time=0, level=0))
    else:
        area_grid = area_da.broadcast_like(ds[variable_name].isel(time=0, sector=0))  # shape: (lat, lon)
    
    # Step 2: apply area to get kg/s
    ds_weighted = ds[variable_name] * area_grid  # shape: (time, sector, lat, lon), units: kg/s

    # Step 3: sum over space (summing all sectors in the process too)
    ds_total = ds_weighted.sum(dim=['lat', 'lon'])  # kg/s

    # Step 4: Convert time to year and group by it
    ds_annual = ds_total.groupby('time.year').mean() # mean across months, keep same unit

    return ds_annual


def ds_to_annual_emissions_total(ds, variable_name):
    """
    Take an output scenario and return an annual emissions timeseries.
    Input is in kg m-2 s-1, output is in kg s-1 (global)
    """
    # Step 1: compute cell area
    area_da = gridcell_area(ds.lat, ds.lon)  # shape: (lat,)

    # TO DO: for AIR this should probably be summed over all levels?
    if "AIR" in variable_name:
        area_grid = area_da.broadcast_like(ds[variable_name].isel(time=0, level=0))
    else:
        area_grid = area_da.broadcast_like(ds[variable_name].isel(time=0, sector=0))  # shape: (lat, lon)
    
    # Step 2: apply area to get kg/s
    ds_weighted = ds[variable_name] * area_grid  # shape: (time, sector, lat, lon), units: kg/s

    # Step 3: sum over space (keeping sector information)
    if "AIR" in variable_name:
        ds_total = ds[variable_name].sum(dim=['lat', 'lon', 'level'])
    else:
        ds_total = ds[variable_name].sum(dim=['lat', 'lon', 'sector'])

    # Step 4: Convert time to year and group by it
    ds_annual = ds_total.groupby('time.year').mean() # mean across months, keep same unit

    return ds_annual

def df_to_wide_timeseries(da):

    df = da.to_pandas()

    if df.index.name == "year":
        df = df.transpose()

    return df

# %% [markdown]
# ### Simple conversions

# %%
def kg_m2_s_to_Mt_y(x):
    s_y = 365 * 24 * 60 * 60 # seconds per year; how do we account for leap years? (now missing)
    Mt_kg = 1e-9 # Mt per kg
    global_m2 = 5.1e14 # area of the earth
    kg_m2_s_to_Mt_y = global_m2 * s_y * Mt_kg    
    return x * kg_m2_s_to_Mt_y


# %%
def kg_m2_s_to_Gt_y(x):
    s_y = 365 * 24 * 60 * 60 # seconds per year; how do we account for leap years? (now missing)
    Gt_kg = 1e-12 # Gt per kg
    global_m2 = 5.1e14 # area of the earth
    kg_m2_s_to_Gt_y = global_m2 * s_y * Gt_kg    
    return x * kg_m2_s_to_Gt_y


# %% [markdown]
# ## To IAMC format using pix

# %%
def nc_to_iamc_like(ds,
                   variable_name,
                   model: str = "undefined",
                   scenario: str = "undefined",
                   region: str = "World",
                   unit: str = "undefined",
                   to_pix=True,
                   keep_sectors=True):
    
    # First get a 2D pandas timeseries, with sector or not
    if keep_sectors:
        da = ds_to_annual_emissions_sectoral(ds, variable_name)
        df = df_to_wide_timeseries(da)  # shape: (sectors, years)
    else:
        da = ds_to_annual_emissions_total(ds, variable_name)
        s = da.to_series()
        df = pd.DataFrame([s.values], columns=s.index)
        df.index = pd.MultiIndex.from_tuples(
            [(model, scenario, region, variable_name, unit)],
            names=["model", "scenario", "region", "variable", "unit"]
        )

    # IAMC-style formatting (only needed if keep_sectors=True)
    if to_pix and keep_sectors:
        df = (
            pix.assignlevel(
                df,
                model=model,
                scenario=scenario,
                region=region,
                variable=variable_name + "|" + pix.projectlevel(df.index, "sector"),
                unit=ds.reporting_unit,
            )
            .droplevel(['sector'])
        )

        df.index = pd.MultiIndex.from_tuples(
            df.index,
            names=["model", "scenario", "region", "variable", "unit"]
        )

    return df


# aggregates gridded emissions back into global time series
# using a wrapper for bulk processing using dask
@delayed
def process_nc_file(filename):
    ds = read_nc_file(filename, cmip7_data_location, chunks={'time': 1})
    var_name = list(ds.data_vars.keys())[0]
    df = nc_to_iamc_like(ds,
                         variable_name=var_name,
                         model=MODEL_SELECTION,
                         scenario=SCENARIO_SELECTION,
                         keep_sectors=False)
    return df


# %% [markdown]
# ## Plotting

# %%
def plot_one_emissions_timeseries(ts,
                                  title: str = "Annual Global Anthropogenic CO₂ Emissions",
                                  xlabel: str = "Year",
                                  ylabel: str = "CO₂ Emissions [mass flux]",
                                  ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    ts.plot(ax=ax, marker='o')
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)


def plot_sectors_emissions_timeseries(ts,
                                      title: str = "Annual Global Anthropogenic CO₂ Emissions",
                                      xlabel: str = "Year",
                                      ylabel: str = "CO₂ Emissions [mass flux]",
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


# matplotlib stacked area chart crashes when a timeseries is first positive and later negative
# def plot_sectors_emissions_timeseries_area_DRAFT1(ts,
#                                   title: str = "Annual Global Anthropogenic CO₂ Emissions",
#                                   xlabel: str = "Year",
#                                   ylabel: str = "CO₂ Emissions [mass flux]",
#                                   figsize: tuple = (10,6)):
#     # Convert to pandas DataFrame for plotting
#     df = ts.to_pandas()  # index: year, columns: sector

#     # Stacked area plot
#     plt.figure(figsize=figsize)
#     df.plot.area(ax=plt.gca(), colormap='tab20')
#     plt.title(title)
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
#     plt.legend(title="Sector", bbox_to_anchor=(1.05, 1), loc='upper left')
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()

def plot_sectors_emissions_timeseries_area_DRAFT2(ts,
                                  title: str = "Annual Global Anthropogenic CO₂ Emissions",
                                  xlabel: str = "Year",
                                  ylabel: str = "CO₂ Emissions [mass flux]",
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
# Miscellaneous

# %%
def pixunique(pixdf,column_name="variable"):
    return pixdf.index.get_level_values(column_name).unique()


# %% [markdown]
# # Load data

# %%

# %% [markdown]
# ## Harmonized data (timeseries)

# %%
MODEL_SELECTION = "REMIND-MAgPIE 3.5-4.10"
SCENARIO_SELECTION = "SSP1 - Very Low Emissions"
MODEL_SELECTION_GRIDDED = MODEL_SELECTION.replace(" ", "-")
SCENARIO_SELECTION_GRIDDED = SCENARIO_SELECTION.replace(" ", "-")

# %%
harmonized_data_file = f"harmonised-gridding_{MODEL_SELECTION}.csv"

harmonized_data = cmip7_utils.load_data(
    Path(harmonized_data_location, harmonized_data_file)
).dropna(axis=1)
# select scenario
harmonized_data = cmip7_utils.filter_scenario(harmonized_data, scenarios=SCENARIO_SELECTION)
# reformat as multi-index in IAMC format
harmonized_data = harmonized_data.set_index(IAMC_COLS)


# %%
# extract list of all species available in the harmonised scenario data

variables = harmonized_data.index.get_level_values("variable").unique()
species_temp = []
for i in np.arange(0, len(variables)):
    species_temp.append(variables[i].split("|")[1])
species_list = np.unique(species_temp)
species_list

# %%
# reaggregate the harmonised scenario data to match the gridded data

full = []

for species in species_list:
    anthro = (
        harmonized_data
        .loc[pix.ismatch(variable=f"*|{species}|*")]
        .loc[pix.ismatch(variable=SECTORS_ANTHRO)]
        .groupby(['model', 'scenario', 'unit'])
        .sum()
        .pix.assign(variable=f"{species}_em_anthro", region="World")
        .reorder_levels(IAMC_COLS)
    )

    air = (
        harmonized_data
        .loc[pix.ismatch(variable=f"*|{species}|*")]
        .loc[pix.ismatch(variable=SECTORS_AIR)]
        .groupby(['model', 'scenario', 'unit'])
        .sum()
        .pix.assign(variable=f"{species}_em_AIR_anthro", region="World")
        .reorder_levels(IAMC_COLS)
    )

    openburning = (
        harmonized_data
        .loc[pix.ismatch(variable=f"*|{species}|*")]
        .loc[pix.ismatch(variable=SECTORS_OPENBURNING)]
        .groupby(['model', 'scenario', 'unit'])
        .sum()
        .pix.assign(variable=f"{species}_em_openburning", region="World")
        .reorder_levels(IAMC_COLS)
    )

    full.extend([anthro, air, openburning])

harmonized_data_reformatted = pix.concat(full)

# %% [markdown]
# ## CO2 example 1 scenario (CMIP7)

# %%
cmip7_data_file = f"CO2-em-anthro_input4MIPs_emissions_CMIP7_IIASA-{MODEL_SELECTION_GRIDDED}-{SCENARIO_SELECTION_GRIDDED}_gn_202301-210012.nc"

scen_ds = read_nc_file(
    f = cmip7_data_file,
    loc = cmip7_data_location
)

# %%
scen_ds.attrs

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
# # Do checks

# %% [markdown]
# ### Some rough code for 1 scenario checking in very simple dataframes

# %%
# simple pandas-dataframe based checks & plots

# cmip7 dfs
sectoral_emissions_ts = ds_to_annual_emissions_sectoral(scen_ds, variable_name="CO2_em_anthro")
total_emissions_ts = ds_to_annual_emissions_total(scen_ds, variable_name="CO2_em_anthro")

# cmip6 dfs
sectoral_emissions_ts_cmip6 = ds_to_annual_emissions_sectoral(scen_ds_cmip6, variable_name="CO2_em_anthro").transpose() # to make it year,sector (instead of sector,year)
total_emissions_ts_cmip6 = ds_to_annual_emissions_total(scen_ds_cmip6, variable_name="CO2_em_anthro")


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
plt.show()


# %%
#plot_sectors_emissions_timeseries_area_DRAFT2(sectoral_emissions_ts)

# %%
#plot_sectors_emissions_timeseries_area_DRAFT2(sectoral_emissions_ts_cmip6)

# %% [markdown]
# ### Putting everything in IAMC format, then building automated checking tools based on that

# %%
# use pix assign to set model and scenario; could get from filename if they are not also variables in the netcdf?

nc_to_iamc_like(scen_ds, variable_name="CO2_em_anthro", keep_sectors=True)

# %% [markdown]
# # Compare CMIP7 gridded emissions with harmonised scenario emissions 

# %% [markdown]
# ## Species-level aggregates

# %%
# prepare list of all files in the CMIP7 results folder

nc_files = sorted([
    f for f in os.listdir(cmip7_data_location)
    if f.endswith(".nc") and MODEL_SELECTION_GRIDDED in f and SCENARIO_SELECTION_GRIDDED in f
])

# %%
# test to ensure that dimensions are complete for all files
for filename in nc_files:
    ds = read_nc_file(filename, cmip7_data_location)
    var_name = list(ds.data_vars.keys())[0]
    dims = ds[var_name].dims
    assert len(dims) == 4, f"Variable '{var_name}' in '{filename}' has {len(dims)} dimensions, expected 4."

# %%
# aggregate gridded emissions back into global time series
# optionally we can set keep_sectors = True and pass it to process_nc_file(f, keep_sectors)

tasks = [process_nc_file(f) for f in nc_files]

# Trigger computation
with ProgressBar():
    results = compute(*tasks)

# Final concatenation
aggregated_gridded_emissions = pd.concat(results)

# %%
aggregated_gridded_emissions = aggregated_gridded_emissions.pix.assign(version = "aggregated gridded")
aggregated_gridded_emissions

# %%
harmonized_data_reformatted = harmonized_data_reformatted.pix.assign(version = "harmonised scenario")


# %%
def reshape_for_plot(df):
    # Fix index names
    if df.index.nlevels == 1:
        if df.index.name is None:
            df.index.name = "index"
        else:
            df.index.name = str(df.index.name)
    else:
        df.index.names = [f"level_{i}" if name is None else str(name)
                          for i, name in enumerate(df.index.names)]

    # Fix column names (e.g., years or times)
    df.columns = [str(col) if pd.notnull(col) else "unknown_time" for col in df.columns]

    # Reset index to move index into columns
    df_reset = df.reset_index()

    # Melt only over former index variables
    id_vars = list(map(str, df.index.names))

    df_long = df_reset.melt(
        id_vars=id_vars,
        var_name="time",
        value_name="values"
    )
    return df_long

df1_long = reshape_for_plot(harmonized_data_reformatted)
df2_long = reshape_for_plot(aggregated_gridded_emissions)

# Combine
combined_long = pd.concat([df1_long, df2_long], axis=0, ignore_index=True)

# %%
to_plot = aggregated_gridded_emissions
df_reset = to_plot.reset_index()
df_long = df_reset.melt(id_vars=to_plot.index.names, var_name="time", value_name="values")

g = sns.relplot(
    data=df_long,
    x="time",
    y="values",
    col="variable",
    col_wrap=3,
    kind="line",
    height=5,
    aspect=1.5,
    facet_kws=dict(sharey=False),
)

if g._legend:
    g._legend.remove()

# Get legend info from one subplot
ax0 = g.axes.flat[0]
handles, labels = ax0.get_legend_handles_labels()

plt.tight_layout()
plt.savefig("/home/hoegner/Projects/CMIP7/checks/plots/gridding/pre-post/reaggregated_gridded_REMIND_vllo.png")
plt.show()

# %%
to_plot = harmonized_data_reformatted
df_reset = to_plot.reset_index()
df_long = df_reset.melt(id_vars=to_plot.index.names, var_name="time", value_name="values")

g = sns.relplot(
    data=df_long,
    x="time",
    y="values",
    col="variable",
    col_wrap=3,
    kind="line",
    height=5,
    aspect=1.5,
    facet_kws=dict(sharey=False),
)

if g._legend:
    g._legend.remove()

# Get legend info from one subplot
ax0 = g.axes.flat[0]
handles, labels = ax0.get_legend_handles_labels()

plt.tight_layout()
plt.savefig("/home/hoegner/Projects/CMIP7/checks/plots/gridding/pre-post/harmonised_data_REMIND_vllo.png")
plt.show()

# %% [markdown]
# ### some notes on unit conversion

# %%
37712 / kg_m2_s_to_Gt_y(0.000473)

# %%
-3830.197824 /  kg_m2_s_to_Gt_y(-0.000048)

# %%
1/0.0049613879189423105 # what is this scalar difference?

# %%
kg_m2_s_to_Mt_y(4.212752e-06)
