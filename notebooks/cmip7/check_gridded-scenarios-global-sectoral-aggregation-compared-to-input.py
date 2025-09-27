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
from concordia.cmip7.CONSTANTS import return_marker_information, CMIP_ERA

FIXED_METADATA = True

GRIDDING_VERSION, MODEL_SELECTION, SCENARIO_SELECTION, SCENARIO_SELECTION_GRIDDED_AFTER_METADATA = return_marker_information(
    m="VLLO", fixed_metadata=FIXED_METADATA
)


# %%
lock = SerializableLock()

# %% [markdown]
# SUMMARY
#
# loops over all species and sectors for ANTHRO, AIRCRAFT, OPENBURNING separately,
# reaggregates the gridded emissions to global totals,
# produces plots of timeseries for each, comparing: 
# - global emissions scenario data (harmonised with github.com/iiasa/emissions_harmonization_historical
# - global emissions output from concordia workflow (re-harmonised; should be identical with scenario data)
# - gridded emissions


# %% [raw]
# df1 = df_compare
# df2 = df_gridded
#
# def normalize(df):
#     df = df.copy()
#     df = df.reset_index(drop=True)         # drop index
#     df.columns = df.columns.str.strip()    # strip col whitespace
#     for c in df.columns:
#         if df[c].dtype == "object":
#             df[c] = df[c].astype(str).str.strip()   # remove spaces
#     return df
#
# df1_norm = normalize(df1)
# df2_norm = normalize(df2)
#

# %% [markdown]
# # Paths, definitions

# %%
# GRIDDING_VERSION = "config_cmip7_v0_2" # jarmo 10.08.2025 (first go, with hist 022)
# GRIDDING_VERSION = "config_cmip7_v0_2_newhistory_remind" # jarmo 10.08.2025 (second go, with updated hist)
# GRIDDING_VERSION = "config_cmip7_v0_2_WSTfix_remind" # jarmo 21.08.2025 (third go, with updated hist, with fixed Waste)
# GRIDDING_VERSION = "config_cmip7_v0_2_CEDS_proxies" # annika 27.08.2025 (with proxies derived from CEDS directly for anthro)
# GRIDDING_VERSION = "config_cmip7_v0_2_CEDS_proxies_new_AIR" # annika 28.08.2025 (now also for aircraft)
# GRIDDING_VERSION = "config_cmip7_v0_2_CEDS_proxies_compressed" # annika 28.08.2025 (including encoding for compression)
# GRIDDING_VERSION = "config_cmip7_v0_2_newhistory_remind" # with updated CDR utils

# see above in return_marker_information for GRIDDING_VERSION

from concordia.cmip7.CONSTANTS import CONFIG
from concordia.settings import Settings
import concordia.cmip7.utils_futureproxy_ceds_bb4cmip as uprox
VERSION = CONFIG
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
        try:
            # Fallback for interactive/Jupyter mode, where 'file location' does not exist
            cmip7_dir = Path().resolve()  # one up
            settings = uprox.get_settings(base_path=cmip7_dir, file = CONFIG)
        except (FileNotFoundError, NameError):
            # another fallback
            cmip7_dir = Path().resolve() / "notebooks" / "cmip7"
            settings = uprox.get_settings(base_path=cmip7_dir, file = CONFIG)
        


# Scenarios pre-gridding

# scenario_data_location = "C:/Users/kikstra/IIASA/ECE.prog - Documents/Projects/CMIP7/IAM Data Processing/concordia_cmip7_esgf_v0_alpha/input/scenarios/September 17 submission test/" # harmonized in emissions_harmonization_historical
# scenario_data_location = "/home/hoegner/Projects/CMIP7/input/scenarios/"
scenario_data_location = settings.scenario_path # harmonized in emissions_harmonization_historical

# harmonized_data_location = Path(f"/home/hoegner/GitHub/concordia/results/{GRIDDING_VERSION}")
# harmonized_data_location = Path(f"C:/Users/kikstra/documents/GitHub/concordia/results/{GRIDDING_VERSION}") # (re-) harmonized by concordia
harmonized_data_location = settings.out_path / GRIDDING_VERSION # (re-) harmonized by concordia

# gridded emissions
# gridding input files
# grid_file_location = "/home/hoegner/Projects/CMIP7/input/gridding/"
# grid_file_location = "C:/Users/kikstra/IIASA/ECE.prog - Documents/Projects/CMIP7/IAM Data Processing/concordia_cmip7_esgf_v0_alpha/input/gridding/"
grid_file_location = settings.gridding_path

# CMIP7 gridded emissions
# cmip7_data_location = Path(f"/home/hoegner/GitHub/concordia/results/{GRIDDING_VERSION}")
# cmip7_data_location = Path(f"C:/Users/kikstra/documents/GitHub/concordia/results/{GRIDDING_VERSION}") # gridding output
cmip7_data_location = settings.out_path / GRIDDING_VERSION / "final"

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
    '**Other Capture and Removal'
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
# see above in return_marker_information

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
if FIXED_METADATA:
    cmip7_data_file = f"CO2-em-anthro_input4MIPs_emissions_{CMIP_ERA}_IIASA-{SCENARIO_SELECTION_GRIDDED_AFTER_METADATA}_gn_202201-210012.nc"
else:
    cmip7_data_file = f"CO2-em-anthro_input4MIPs_emissions_{CMIP_ERA}_IIASA-{MODEL_SELECTION_GRIDDED}-{SCENARIO_SELECTION_GRIDDED}_gn_202301-210012.nc"
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

# %%
