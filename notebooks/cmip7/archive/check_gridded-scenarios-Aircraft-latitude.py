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
# SUMMARY
#
# creates latitudinal profile for AIRCRAFT emissions per species


# %% [markdown]
# # Paths, definitions

# %%
GRIDDING_VERSION = "config_cmip7_v0_2" # jarmo 10.08.2025 (first go, with hist 022)
GRIDDING_VERSION = "config_cmip7_v0_2_newhistory_remind" # jarmo 10.08.2025 (second go, with updated hist)
GRIDDING_VERSION = "config_cmip7_v0_2_WSTfix_remind" # jarmo 21.08.2025 (third go, with updated hist, with fixed Waste)
GRIDDING_VERSION = "config_cmip7_v0_2_CEDS_proxies" # annika 27.08.2025 (with proxies derived from CEDS directly for anthro)
GRIDDING_VERSION = "config_cmip7_v0_2_CEDS_proxies_new_AIR" # annika 28.08.2025 (now also for aircraft)
GRIDDING_VERSION = "config_cmip7_v0_2_CEDS_proxies_compressed" # annika 28.08.2025 (including encoding for compression)
GRIDDING_VERSION = "config_cmip7_v0_2_newhistory_remind" # with updated CDR utils

# gridded emissions
# gridding input files
grid_file_location = "/home/hoegner/Projects/CMIP7/input/gridding/"
# grid_file_location = "C:/Users/kikstra/IIASA/ECE.prog - Documents/Projects/CMIP7/IAM Data Processing/concordia_cmip7_v0_2/input/gridding/"

# CMIP7 gridded emissions
cmip7_data_location = Path(f"/home/hoegner/GitHub/concordia/results/{GRIDDING_VERSION}")
# cmip7_data_location = Path(f"C:/Users/kikstra/documents/GitHub/concordia/results/{GRIDDING_VERSION}") # gridding output

cmip6_data_location = Path("/home/hoegner/Projects/CMIP7/checks/Example NetCDF files CMIP6/")

path_ceds_cmip7 = Path("/home/hoegner/Projects/CMIP7/input/gridding/CEDS_CMIP7_AIR")

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
def read_nc_file(f, loc, reorder_list=None, chunks={"time": 1}):
    ds = xr.open_dataset(
        loc / f,
        engine="netcdf4",
        chunks=chunks,
        lock=lock
    )
    
    if reorder_list is not None:
        ds = ds[reorder_list]
    
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
cmip6_data_file = "BC-em-AIR-anthro_input4MIPs_emissions_CMIP_CEDS-2017-08-30_gn_200001-201412.nc"

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
def ds_to_latitudinal_profile(gridded_data, var_name, cell_area, year):
    """
    Compute latitudinal emission profile (Mt/year per latitude band).
    
    Parameters:
    - gridded_data: xr.Dataset containing the emission variable
    - var_name: str, name of the variable to convert
    - cell_area: xr.DataArray of shape (lat, lon), in m2
    - year: int, the year to extract
    
    Returns:
    - xr.DataArray of shape (lat,), in Mt/year
    """
    da = gridded_data[var_name]

    # seconds per month
    seconds_per_month = da.time.dt.days_in_month * 24 * 60 * 60

    # kg/m2/s --> kg/m2/month
    monthly = seconds_per_month * da

    # weight with cell area
    area_weighted = cell_area * monthly

    # Sum over longitude (and level if present), keep latitude
    sum_dims = ["lon"]
    if "level" in area_weighted.dims:
        sum_dims.append("level")

    kg_per_month = area_weighted.sum(dim=sum_dims)

    # Annual totals by latitude (kg/year)
    kg_per_year = kg_per_month.groupby("time.year").sum()

    # Select requested year
    kg_lat_profile = kg_per_year.sel(year=year)

    # Convert to Mt/year
    da_Mt_y = kg_lat_profile * 1e-9

    # rename for clarity
    da_Mt_y = da_Mt_y.rename(var_name + "_lat_profile")
    
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
                   year: int = 2023,
                   to_pix=True):


    da = ds_to_latitudinal_profile(ds, variable_name, cell_area, year)
    s = da.to_series()
    df = pd.DataFrame([s.values], columns=s.index)
    df.index = pd.MultiIndex.from_tuples(
        [(model, scenario, region, variable_name, unit)],
        names=["model", "scenario", "region", "variable", "unit"]
    )

    df.index = pd.MultiIndex.from_tuples(
        df.index,
        names=["model", "scenario", "region", "variable", "unit"]
    )

    return df


# wrapper for bulk processing using dask
@delayed
def process_gridded_file(filename, loc, cell_area, model, scenario, year):
    ds = read_nc_file(filename, loc)
    var_name = list(ds.data_vars.keys())[0]
    
    df = nc_to_iamc_like(
        ds,
        variable_name=var_name,
        cell_area=cell_area,
        model=model,
        scenario=scenario,
        year=year
    )
    return df


# %%
# function calling the wrapper for bulk processing with dask

def process_gridded_files(filenames, loc, cell_area, model, scenario, year):
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

    Returns
    -------
    pandas.DataFrame
        Combined DataFrame in IAMC format with columns for years and a
        MultiIndex: ["model", "scenario", "region", "variable", "unit"].
    """

    tasks = [
        process_gridded_file(f, loc, cell_area, model, scenario, year)
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
        var_name="latitude",
        value_name="values"
    )
    return df_long


# %% [markdown]
# ## Yearly latitudinal profiles

# %%
# ScenarioMIP CMIP7 REMIND scenario
filenames = [f.name for f in cmip7_data_location.glob("*.nc") if "AIR" in f.name]

year = 2100

aggregated_gridded_emissions = process_gridded_files(
    filenames,
    loc=cmip7_data_location,
    cell_area=cell_area,
    model=MODEL_SELECTION_GRIDDED,
    scenario=SCENARIO_SELECTION_GRIDDED,
    year=year
)

# %%
# input4MIPs CMIP6 CEDS BC aircraft emissions, 2014
filenames = [f.name for f in cmip6_data_location.glob("*.nc") if "BC-em-AIR" in f.name]

year = 2014

cmip6_emissions = process_gridded_files(
    filenames,
    loc=cmip6_data_location,
    cell_area=cell_area,
    model=MODEL_SELECTION_GRIDDED,
    scenario=SCENARIO_SELECTION_GRIDDED,
    year=year
)

# %%
# input4MIPs CMIP7 CEDS BC aircraft emissions, 2023 (year used for grids)
filenames = [f.name for f in path_ceds_cmip7.glob("*.nc") if "BC-em-AIR" in f.name]

year = 2023

ceds_cmip7_emissions = process_gridded_files(
    filenames,
    loc=path_ceds_cmip7,
    cell_area=cell_area,
    model=MODEL_SELECTION_GRIDDED,
    scenario=SCENARIO_SELECTION_GRIDDED,
    year=year
)

# %%
aggregated_gridded_emissions = aggregated_gridded_emissions.pix.assign(version = "CMIP7 scenario (2100)")
cmip6_emissions = cmip6_emissions.pix.assign(version = "CMIP6 (2014)")
ceds_cmip7_emissions = ceds_cmip7_emissions.pix.assign(version = "CMIP7 (2023)")

# %%
gridded_data = reshape_for_plot(aggregated_gridded_emissions)
gridded_data = gridded_data[gridded_data["variable"]=="BC_em_AIR_anthro"]
cmip6_agg = reshape_for_plot(cmip6_emissions)
ceds_cmip7_agg = reshape_for_plot(ceds_cmip7_emissions)
to_plot = pd.concat([cmip6_agg, ceds_cmip7_agg, gridded_data])

# %%
to_plot["values_normalised"] = to_plot.groupby("version")["values"].transform(lambda x: x / x.sum())

# %%
g = sns.relplot(
    data=to_plot,
    x="latitude",
    y="values_normalised",
    col="variable",
    hue="version",
    style="version",
    col_wrap=1,
    kind="line",
    height=3,
    aspect=3,
    facet_kws=dict(sharey=False),
)
for ax in g.axes.flat:
    ax.set_xticks(np.arange(-90, 91, 30))
    
g.figure.tight_layout()
g.figure.savefig(
    Path(plots_path, f"reaggregated_gridded_{MODEL_SELECTION_GRIDDED}_{SCENARIO_SELECTION_GRIDDED}_BC-em-AIR-anthro.png"),
    bbox_inches="tight"
)
plt.show()
