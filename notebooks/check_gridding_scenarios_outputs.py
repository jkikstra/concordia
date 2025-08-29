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
        lock=lock
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
# load a CMIP7 sample file
cmip7_data_file = f"CO2-em-anthro_input4MIPs_emissions_CMIP7_IIASA-{MODEL_SELECTION_GRIDDED}-{SCENARIO_SELECTION_GRIDDED}_gn_202301-210012.nc"

scen_ds = read_nc_file(
    f = cmip7_data_file,
    loc = cmip7_data_location
)

# %%
areacella = xr.open_dataset(Path(grid_file_location, "areacella_input4MIPs_emissions_CMIP_CEDS-CMIP-2025-04-18_gn.nc"))
cell_area = areacella["areacella"]

# ensure that dimensions match the sample CMIP7 file
assert set(cell_area.dims).issubset(set(scen_ds.dims))


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
# ## To IAMC format using pix

# %%
def nc_to_iamc_like(ds,
                   variable_name,
                   cell_area,
                   model: str = "undefined",
                   scenario: str = "undefined",
                   region: str = "World",
                   unit: str = "undefined",
                   to_pix=True,
                   keep_sectors=True):

    
    # check whether we are processing aircraft emissions
    is_air_file = (
        "AIR" in variable_name.upper()
    )

    # if so, force keep_sectors=False workflow
    if is_air_file:
        keep_sectors = False

    
    # First get a 2D pandas timeseries, with sector or not
    if keep_sectors:
        da = ds_to_annual_emissions_total(ds, variable_name, cell_area, keep_sectors=True)
        df = df_to_wide_timeseries(da)  # shape: (sectors, years)
    else:
        da = ds_to_annual_emissions_total(ds, variable_name, cell_area, keep_sectors=False)
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


# wrapper for bulk processing using dask
@delayed
def process_gridded_file(filename, loc, cell_area, model, scenario, keep_sectors=False):
    ds = read_nc_file(filename, loc)
    var_name = list(ds.data_vars.keys())[0]
    
    df = nc_to_iamc_like(
        ds,
        variable_name=var_name,
        cell_area=cell_area,
        model=model,
        scenario=scenario,
        keep_sectors=keep_sectors
    )
    return df


# %%
# function calling the wrapper for bulk processing with dask

def process_gridded_files(filenames, loc, cell_area, model, scenario, keep_sectors=False):
    """
    Process multiple NetCDF files in parallel and convert them to IAMC-style time series.

    This function reads a list of NetCDF files containing gridded emissions data,
    converts each file to annual global emissions in Mt/year, and returns a combined
    IAMC-like pandas DataFrame suitable for climate/energy modeling frameworks.

    Parameters
    ----------
    filenames : list of str
        List of NetCDF file names to process (filenames only, not full paths).
    loc : pathlib.Path or str
        Directory path where the NetCDF files are located.
    cell_area : xarray.DataArray
        2D DataArray (lat, lon) containing the grid cell areas in m².
    model : str
        Name of the model to assign in the output IAMC-format index.
    scenario : str
        Name of the scenario to assign in the output IAMC-format index.
    keep_sectors : bool, optional (default=False)
        If True, retain sector-specific emissions in the output. If False,
        aggregate over all sectors to produce a single total per year.

    Returns
    -------
    pandas.DataFrame
        Combined DataFrame in IAMC format with columns for years and a
        MultiIndex: ["model", "scenario", "region", "variable", "unit"].
    """

    tasks = [
        process_gridded_file(f, loc, cell_area, model, scenario, keep_sectors=keep_sectors)
        for f in filenames
    ]
    
    with ProgressBar():
        dfs = dask.compute(*tasks)
    
    df_all = pd.concat(dfs)
    return df_all


# %% [markdown]
# ## Plotting

# %%
def reshape_for_plot(df):

    df_reset = df.reset_index()

    # Melt only over former index variables
    id_vars = list(map(str, df.index.names))

    df_long = df_reset.melt(
        id_vars=id_vars,
        var_name="time",
        value_name="values"
    )
    return df_long
    
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
# # Load data

# %% [markdown]
# ## Original scenario data (timeseries)

# %%
scenario_data_file = f"harmonised-gridding_{MODEL_SELECTION}.csv"

scenario_data = cmip7_utils.load_data(
    Path(scenario_data_location, scenario_data_file)
).dropna(axis=1)
# select scenario
scenario_data = cmip7_utils.filter_scenario(scenario_data, scenarios=SCENARIO_SELECTION)
# reformat as multi-index in IAMC format
scenario_data = scenario_data.set_index(IAMC_COLS)

# %%
# Boolean mask for all units starting with "kt" and ending with "/yr"
mask = scenario_data.index.get_level_values("unit").str.match(r"kt.*?/yr")

# Multiply those rows ×1000
scenario_data.loc[mask] = scenario_data.loc[mask] / 1000

# Update the index labels (kt to Mt)
scenario_data.index = scenario_data.index.set_levels(
    scenario_data.index.levels[scenario_data.index.names.index("unit")].str.replace(r"^kt", "Mt", regex=True),
    level="unit"
)

scenario_data

# %%
# extract list of all species available in the harmonised scenario data

variables = scenario_data.index.get_level_values("variable").unique()
species_temp = []
for i in np.arange(0, len(variables)):
    species_temp.append(variables[i].split("|")[1])
species_list = np.unique(species_temp)

# %%
# reaggregate the scenario data to match the gridded data without sectoral resolution

full = []

for species in species_list:
    anthro = (
        scenario_data
        .loc[pix.ismatch(variable=f"*|{species}|*")]
        .loc[pix.ismatch(variable=SECTORS_ANTHRO)]
        .groupby(['model', 'scenario', 'unit'])
        .sum()
        .pix.assign(variable=f"{species}_em_anthro", region="World")
        .reorder_levels(IAMC_COLS)
    )

    air = (
        scenario_data
        .loc[pix.ismatch(variable=f"*|{species}|*")]
        .loc[pix.ismatch(variable=SECTORS_AIR)]
        .groupby(['model', 'scenario', 'unit'])
        .sum()
        .pix.assign(variable=f"{species}_em_AIR_anthro", region="World")
        .reorder_levels(IAMC_COLS)
    )

    openburning = (
        scenario_data
        .loc[pix.ismatch(variable=f"*|{species}|*")]
        .loc[pix.ismatch(variable=SECTORS_OPENBURNING)]
        .groupby(['model', 'scenario', 'unit'])
        .sum()
        .pix.assign(variable=f"{species}_em_openburning", region="World")
        .reorder_levels(IAMC_COLS)
    )

    full.extend([anthro, air, openburning])

scenario_data_reformatted = pix.concat(full)

# %%
# reaggregate the scenario data to match the gridded data retaining sectoral resolution

scenario_data = scenario_data.pix.assign(sector = scenario_data.index.get_level_values("variable").str.split("|").str[2])

full = []

for species in species_list:
    anthro = (
        scenario_data
        .loc[pix.ismatch(variable=f"*|{species}|*")]
        .loc[pix.ismatch(variable=SECTORS_ANTHRO)]
        .groupby(["model", "scenario", "unit", "sector"])
        .sum()
        .rename(index=sector_dict, level="sector")
    )
    
    new_var_name = [f"{species}_em_anthro|{sec}" for sec in anthro.index.get_level_values("sector")]
    
    anthro = (
        anthro
        .pix.assign(variable=new_var_name, region="World")
        .reset_index("sector", drop=True)
        .reorder_levels(IAMC_COLS)
    )

    air = (
        scenario_data
        .loc[pix.ismatch(variable=f"*|{species}|*")]
        .loc[pix.ismatch(variable=SECTORS_AIR)]
        .groupby(["model", "scenario", "unit", "sector"])
        .sum()
        .rename(index=sector_dict, level="sector")
        .pix.assign(variable=f"{species}_em_AIR_anthro", region="World")
        .reset_index("sector", drop=True)
        .reorder_levels(IAMC_COLS)
    )

    openburning = (
        scenario_data
        .loc[pix.ismatch(variable=f"*|{species}|*")]
        .loc[pix.ismatch(variable=SECTORS_OPENBURNING)]
        .groupby(["model", "scenario", "unit", "sector"])
        .sum()
        .rename(index=sector_dict, level="sector")
    )

    new_var_name = [f"{species}_em_openburning|{sec}" for sec in openburning.index.get_level_values("sector")]

    openburning = (
        openburning
        .pix.assign(variable=new_var_name, region="World")
        .reset_index("sector", drop=True)
        .reorder_levels(IAMC_COLS)
    )

    full.extend([anthro, air, openburning])

scenario_data_with_sectors = pix.concat(full)

# %% [markdown]
# ## Concordia-harmonised data (timeseries)

# %%
harmonized_data_file = f"harmonization-{GRIDDING_VERSION}.csv"
harmonized_data = cmip7_utils.load_data(
    Path(harmonized_data_location, harmonized_data_file)
)
harmonized_data = cmip7_utils.filter_scenario(harmonized_data, scenarios=SCENARIO_SELECTION).dropna(axis=1)
harmonized_data = harmonized_data[~harmonized_data["variable"].str.contains("aggregate")]
harmonized_data = harmonized_data[harmonized_data["variable"].str.contains(r'\bHarmonized\b', regex=True)]
harmonized_data["variable"] = harmonized_data["variable"].str.split('|').apply(lambda parts: "|".join(parts[1:-2]))

# %%
harmonized_data = harmonized_data.set_index(IAMC_COLS)

# %%
# Boolean mask for all units starting with "kt" and ending with "/yr"
mask = harmonized_data.index.get_level_values("unit").str.match(r"kt.*?/yr")

# Multiply those rows ×1000
harmonized_data.loc[mask] = harmonized_data.loc[mask] / 1000

# Update the index labels (kt to Mt)
harmonized_data.index = harmonized_data.index.set_levels(
    harmonized_data.index.levels[harmonized_data.index.names.index("unit")].str.replace(r"^kt", "Mt", regex=True),
    level="unit"
)

harmonized_data

# %%
# species/sector combinations missing after harmonisation
set(list(scenario_data.index.get_level_values("variable").unique())) - set(list(harmonized_data.index.get_level_values("variable").unique()))

# %%
# reaggregate the harmonised scenario data to match the gridded data without sectoral resolution

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

# %%
# reaggregate the scenario data to match the gridded data retaining sectoral resolution

harmonized_data = harmonized_data.pix.assign(sector = harmonized_data.index.get_level_values("variable").str.split("|").str[2])

full = []

for species in species_list:
    anthro = (
        harmonized_data
        .loc[pix.ismatch(variable=f"*|{species}|*")]
        .loc[pix.ismatch(variable=SECTORS_ANTHRO)]
        .groupby(["model", "scenario", "unit", "sector"])
        .sum()
        .rename(index=sector_dict, level="sector")
    )
    
    new_var_name = [f"{species}_em_anthro|{sec}" for sec in anthro.index.get_level_values("sector")]
    
    anthro = (
        anthro
        .pix.assign(variable=new_var_name, region="World")
        .reset_index("sector", drop=True)
        .reorder_levels(IAMC_COLS)
    )

    air = (
        harmonized_data
        .loc[pix.ismatch(variable=f"*|{species}|*")]
        .loc[pix.ismatch(variable=SECTORS_AIR)]
        .groupby(["model", "scenario", "unit", "sector"])
        .sum()
        .rename(index=sector_dict, level="sector")
        .pix.assign(variable=f"{species}_em_AIR_anthro", region="World")
        .reset_index("sector", drop=True)
        .reorder_levels(IAMC_COLS)
    )

    openburning = (
        harmonized_data
        .loc[pix.ismatch(variable=f"*|{species}|*")]
        .loc[pix.ismatch(variable=SECTORS_OPENBURNING)]
        .groupby(["model", "scenario", "unit", "sector"])
        .sum()
        .rename(index=sector_dict, level="sector")
    )

    new_var_name = [f"{species}_em_openburning|{sec}" for sec in openburning.index.get_level_values("sector")]

    openburning = (
        openburning
        .pix.assign(variable=new_var_name, region="World")
        .reset_index("sector", drop=True)
        .reorder_levels(IAMC_COLS)
    )

    full.extend([anthro, air, openburning])

harmonized_data_with_sectors = pix.concat(full)

# %% [markdown]
# ## CO2 example 1 scenario (CMIP7)

# %%
# loaded above
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
#fig, axs = plt.subplots(2, 2, figsize=(15, 10), sharex=True)

#plot_one_emissions_timeseries(total_emissions_ts, ax=axs[0, 0],
#    title="Total Emissions (Scenario CMIP7)")
#plot_sectors_emissions_timeseries(sectoral_emissions_ts, ax=axs[0, 1],
#    title="Sectoral Emissions (Scenario CMIP7)")

#plot_one_emissions_timeseries(total_emissions_ts_cmip6, ax=axs[1, 0],
#    title="Total Emissions (CMIP6)")
#plot_sectors_emissions_timeseries(sectoral_emissions_ts_cmip6, ax=axs[1, 1],
#    title="Sectoral Emissions (CMIP6)")

#fig.suptitle("Emissions Time Series Comparison", fontsize=16)

#plt.tight_layout(rect=[0, 0, 1, 0.96])
#plt.savefig(Path(plots_path, f"comparison_with_CMIP6_MESSAGE_ssp245_{MODEL_SELECTION_GRIDDED}_{SCENARIO_SELECTION_GRIDDED}.png"))
#plt.show()


# %%
#plot_sectors_emissions_timeseries_area_DRAFT2(sectoral_emissions_ts)

# %%
#plot_sectors_emissions_timeseries_area_DRAFT2(sectoral_emissions_ts_cmip6)

# %% [markdown]
# # Compare CMIP7 gridded emissions with harmonised scenario emissions 

# %% [markdown]
# ## Species-level aggregates

# %%
filenames = [f.name for f in cmip7_data_location.glob("*.nc")]

aggregated_gridded_emissions = process_gridded_files(
    filenames,
    loc=cmip7_data_location,
    cell_area=cell_area,
    model=MODEL_SELECTION_GRIDDED,
    scenario=SCENARIO_SELECTION_GRIDDED,
    keep_sectors=False
)

# %%
aggregated_gridded_emissions = aggregated_gridded_emissions.pix.assign(version = "aggregated gridded")
scenario_data_reformatted = scenario_data_reformatted.pix.assign(version = "CMIP7 harmonised scenario")
harmonized_data_reformatted = harmonized_data_reformatted.pix.assign(version = "concordia harmonised scenario")

sectors = ["em_anthro", "em_AIR_anthro", "em_openburning"]

# %%
scenario_data = reshape_for_plot(scenario_data_reformatted)
harmonised_data = reshape_for_plot(harmonized_data_reformatted)
gridded_data = reshape_for_plot(aggregated_gridded_emissions)
combined = pix.concat([scenario_data, harmonised_data, gridded_data], axis=0, ignore_index=True)

for sec in sectors:
    
    to_plot = combined[combined["variable"].str.contains(f"{sec}")].dropna()
    to_plot["time"] = pd.to_datetime(to_plot["time"], errors="coerce")
    
    g = sns.relplot(
        data=to_plot,
        x="time",
        y="values",
        col="variable",
        hue="version",
        style="version",
        col_wrap=2,
        kind="line",
        height=3,
        aspect=1.5,
        facet_kws=dict(sharey=False),
    )
    
    g._legend.set_bbox_to_anchor((1.05, 0.5))
    g._legend.set_loc("center left")
    
    g.figure.tight_layout()
    g.figure.savefig(
        Path(plots_path, f"reaggregated_gridded_{MODEL_SELECTION_GRIDDED}_{SCENARIO_SELECTION_GRIDDED}_{sec}.png"),
        bbox_inches="tight"
    )
    plt.show()

# %% [markdown]
# ## Sector-level aggregates

# %%
aggregated_gridded_emissions_sectors = process_gridded_files(
    filenames,
    loc=cmip7_data_location,
    cell_area=cell_area,
    model=MODEL_SELECTION_GRIDDED,
    scenario=SCENARIO_SELECTION_GRIDDED,
    keep_sectors=True
)

# %%
aggregated_gridded_emissions_sectors = aggregated_gridded_emissions_sectors.pix.assign(version = "aggregated gridded")
scenario_data_with_sectors = scenario_data_with_sectors.pix.assign(version = "CMIP7 harmonised scenario")
harmonized_data_with_sectors = harmonized_data_with_sectors.pix.assign(version = "concordia harmonised scenario")

# %%
scenario_data = reshape_for_plot(scenario_data_with_sectors)
harmonised_data = reshape_for_plot(harmonized_data_with_sectors)
gridded_data = reshape_for_plot(aggregated_gridded_emissions_sectors)
combined = pix.concat([scenario_data, harmonised_data, gridded_data], axis=0, ignore_index=True)

for sec in sectors:
    
    to_plot = combined[combined["variable"].str.contains(f"{sec}")].dropna()
    to_plot["time"] = pd.to_datetime(to_plot["time"], errors="coerce")
    
    g = sns.relplot(
        data=to_plot,
        x="time",
        y="values",
        col="variable",
        hue="version",
        style="version",
        col_wrap=3,
        kind="line",
        height=3,
        aspect=1.5,
        facet_kws=dict(sharey=False),
    )
    
    g._legend.set_bbox_to_anchor((1.05, 0.5))
    g._legend.set_loc("center left")

    g.figure.tight_layout()
    g.figure.savefig(
        Path(plots_path, f"sectoral_reaggregated_gridded_{MODEL_SELECTION_GRIDDED}_{SCENARIO_SELECTION_GRIDDED}_{sec}.png"),
        bbox_inches="tight"
    )
