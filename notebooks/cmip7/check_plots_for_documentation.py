# -*- coding: utf-8 -*-
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
# With autoreload: changes are picked up automatically when changing a file/module that is imported, without having to restart the kernel.
# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# # Make some plots for the documentation of the CMIP7 ScenarioMIP emissions harmonization and gridding
# **Note:** currently built allowing for running only one scenario at a time.

# %% [markdown]
# ## Specify input scenario data and project settings
# **Note:** these options below can also be changed and driven from a driver script. 

# %% editable=true slideshow={"slide_type": ""} tags=["parameters"]
HISTORY_FILE: str = "country-history_202511261223_202511040855_202512032146_202512021030_7e32405ade790677a6022ff498395bff00d9792d.csv"
# Settings
# SETTINGS_FILE: str = "config_cmip7_esgf_v0_alpha.yaml" # was used for preparing for first upload to ESGF
SETTINGS_FILE: str = "config_cmip7_v0-4-0.yaml" # for second ESGF version
VERSION_ESGF: str = "1-0-0" # for second ESGF version

# Which scenario to run from the markers
marker_to_run: str = "h" # options: h, hl, m, ml, l, ln, vl

# What folder to save this run in
GRIDDING_VERSION: str | None = None
GRIDDING_VERSION: str | None = f"{marker_to_run}_{VERSION_ESGF}"

# %% [markdown]
# ## Importing packages

# %%
import aneris
aneris.__file__
import concordia
concordia.__file__

import logging
from pathlib import Path

import dask
from dask.utils import SerializableLock
from dask.diagnostics import ProgressBar
import pandas as pd
import pycountry
from pandas_indexing import isin, ismatch, assignlevel, extractlevel
from pandas_indexing.units import set_openscm_registry_as_default
from ptolemy.raster import IndexRaster
import concordia._patches_ptolemy # seemingly not used, not used in this script, but sets fill_value for xarray_reduce to 0 

from aneris import logger
from concordia import (
    RegionMapping,
    VariableDefinitions,
)
from concordia.cmip7 import utils as cmip7_utils # update to cmip7 utils (e.g. for dressing up netcdf)
from concordia.settings import Settings
from concordia.utils import MultiLineFormatter
from concordia.workflow import WorkflowDriver
from concordia.cmip7.CONSTANTS import return_marker_information, PROXY_YEARS, find_voc_data_variable_string, GASES_ESGF_CEDS, GASES_ESGF_BB4CMIP, GASES_ESGF_CEDS_VOC, GASES_ESGF_BB4CMIP_VOC
from concordia.cmip7.dask_setup_alternative import setup_dask_client # to enable running with dask also from VSCode Interactive Window
from concordia.cmip7.utils import calculate_ratio, return_nc_output_files_main_voc, SECTOR_ORDERING_GAS, SECTOR_ORDERING_DEFAULT, SECTOR_DICT_ANTHRO_DEFAULT, SECTOR_DICT_ANTHRO_CO2_SCENARIO, reorder_dimensions, add_file_global_sum_totals_attrs, SECTOR_DICT_OPENBURNING_DEFAULT, SECTOR_DICT_OPENBURNING_DEFAULT_FLIPPED, SECTOR_DICT_ANTHRO_CO2_SCENARIO_FLIPPED, add_lon_lat_bounds, add_time_bounds
from concordia.cmip7.utils_plotting import ds_to_annual_emissions_total, plot_place_timeseries, plot_place_area_average_timeseries

from tqdm import tqdm
import xarray as xr
import numpy as np

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib import colors
import cartopy.feature as cfeature # for country borders
import cftime
import seaborn as sns

# %%
# Scenario information
_, MODEL_SELECTION, SCENARIO_SELECTION, _ = return_marker_information(
    v=SETTINGS_FILE,
    m=marker_to_run
)
if GRIDDING_VERSION is None:
    GRIDDING_VERSION = f"{marker_to_run}" # default to just the marker abbreviation if no versioning is provided
SCENARIO_FILE = f"harmonised-gridding_{MODEL_SELECTION}.csv"

# Group by scenario and year, sum across all numeric columns
SECTOR_FILE_DICT = {
    "openburning" : ['Agricultural Waste Burning',
                     'Forest Burning',
                     'Grassland Burning',
                     'Peat Burning'],
    "anthro" : ['Agriculture',  'BECCS',
       'Biochar', 'Direct Air Capture', 'Energy Sector',
       'Enhanced Weathering', 
       'Industrial Sector', 'International Shipping', 'Ocean',
       'Other CDR',  'Residential Commercial Other',
       'Soil Carbon Management', 'Solvents Production and Application',
       'Transportation Sector', 'Waste'],
    "AIR-anthro" : ['Aircraft']
}


# %% editable=true slideshow={"slide_type": ""} tags=["parameters"]
# filename template
FILE_NAME_ENDING: str | None = cmip7_utils.filename_for_esgf(marker=marker_to_run, version=VERSION_ESGF)

print(f"Producing experiment: {FILE_NAME_ENDING}")

# %% [markdown]
# Load unit registry from openSCM for translating units (e.g., to and from CO2eq)

# %%
ur = set_openscm_registry_as_default()

# %% [markdown]
# # Read Settings
#
# The key settings for this harmonization run are detailed in the file "config_cmip7_{VERSION}.yaml", which is located in this same folder.
#
# **NOTE: to use this workflow, you need to specify (at minimum) the `data_path` vairable in this "config_cmip7_{VERISON}.yaml" file.**
#
# This settings file for instance points to the data location, where e.g. the following files are hosted:
# - historical emissions data to harmonize the regional IAM trajectories to
# - gridding pattern files for gridding regional IAM trajectories
# - region-to-country mappings for all IAMs 
# - variable definitions: a list by species and sector, specifying units, gridding levels, and proxies
# - postprocessing: files for potential post-processing (current not used)
# - scenarios: input IAM trajectories

# %%
# Get the directory of the current file, works in both script and notebook contexts
# When running through papermill, we need to find the original notebook location
try:
    # Try to get __file__ (works when running as script)
    HERE = Path(__file__).parent
    # Also check if HERE resolved to just current directory, which indicates path resolution failed
    if str(HERE) == "." or HERE == Path("."):
        raise NameError("HERE resolved to current directory, using fallback")
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
                                                       SETTINGS_FILE))

settings.base_year

# %%
# helper functions
def return_emission_names(file):

    # Extract variable name and type from filename
    # filename format: "{gas}-em-{type}_{FILE_NAME_ENDING}" or "{gas}_{FILE_NAME_ENDING}"
    if "-em-" in file.name:
        parts = file.name.replace(f"_{FILE_NAME_ENDING}", "").split("-em-")
        gas_name = parts[0]
        type_name = parts[1]
        var = f"{gas_name.replace("-","_")}_em_{type_name.replace("-","_")}"
    else:
        raise ValueError(f"Unrecognized file format: {file.name}. Expected format: '{{gas}}-em-{{type}}_{{FILE_NAME_ENDING}}'")

    return gas_name, var, type_name

def load_result(var_name, FILE_NAME_ENDING=FILE_NAME_ENDING, settings=settings, GRIDDING_VERSION=GRIDDING_VERSION):
    return xr.open_dataset(settings.out_path / GRIDDING_VERSION / f'{var_name}_{FILE_NAME_ENDING}')



# %% [markdown]
# ## PREP DATA

# %%
PATH_RESULTS = Path('D:\\concordia-results\\rc4')


scenario_h = PATH_RESULTS / f"h_{VERSION_ESGF}" / "scenarios_processed.csv"
scenario_vl = PATH_RESULTS / f"vl_{VERSION_ESGF}" / "scenarios_processed.csv"

# Create output folder for plots
output_folder = PATH_RESULTS / "plots_output"
output_folder.mkdir(exist_ok=True)
print(f"Plots will be saved to: {output_folder}")


# %%

def reformatting_names_units(ds):

    # Replace Sulfur with SO2
    ds['gas'] = ds['gas'].replace('Sulfur', 'SO2')

    # For N2O: divide value by 1000 and change unit
    n2o_mask = ds['gas'] == 'N2O'
    ds.loc[n2o_mask, 'value'] = ds.loc[n2o_mask, 'value'] / 1000
    ds.loc[n2o_mask, 'unit'] = 'Mt N2O/yr'

    return ds

# %%
# Read the scenario files
df_h = pd.read_csv(scenario_h)
df_vl = pd.read_csv(scenario_vl)

# Add scenario marker columns to identify which scenario each row came from
df_h['scenario'] = 'H'
df_vl['scenario'] = 'VL'

# Combine the dataframes
df_combined = pd.concat([df_h, df_vl], ignore_index=True)

# Remove CO2 emissions from openburning sector
openburning_sectors = SECTOR_FILE_DICT['openburning']
df_combined = df_combined[~((df_combined['gas'] == 'CO2') & (df_combined['sector'].isin(openburning_sectors)))]
# Remove pre-processed Agriculture emissions (as they should be zero)
df_combined = df_combined[~((df_combined['gas'] == 'CO2') & (df_combined['sector'].isin(["Agriculture"])))]


print(f"Combined dataframe shape: {df_combined.shape}")
print(f"Columns: {df_combined.columns.tolist()}")
print(df_combined.head())

# %%
# Sum across species (sum all columns except year, scenario, and potentially other non-numeric columns)
# Get numeric columns to sum
numeric_cols = df_combined.select_dtypes(include=[np.number]).columns.tolist()
print(f"Numeric columns: {numeric_cols}")

# Create a reverse mapping: sector -> sector_file key
sector_to_file = {}
for sector_file_key, sectors_list in SECTOR_FILE_DICT.items():
    for sector in sectors_list:
        sector_to_file[sector] = sector_file_key

# --------- sectors --------- 
# Add sector_file column to df_combined
df_combined['sector_file'] = df_combined['sector'].map(sector_to_file)
df_combined_w_sectorfiles = df_combined
KEEP_COLS = ['model', 'scenario', 'sector_file', 'gas', 'unit']
df_sector_files = df_combined_w_sectorfiles.groupby(KEEP_COLS)[numeric_cols].sum().reset_index()
print(f"\nSummed dataframe shape: {df_sector_files.shape}")
print(df_sector_files.head())

# Melt the sector_files dataframe
KEEP_COLS_SF = ['model', 'scenario', 'sector_file', 'gas', 'unit']
df_sector_files_melted = df_sector_files.melt(
    id_vars=KEEP_COLS_SF,
    var_name='years',
    value_name='value'
)

# Convert years to numeric and sort
df_sector_files_melted['years'] = pd.to_numeric(df_sector_files_melted['years'])
df_sector_files_melted = df_sector_files_melted.sort_values(['model', 'scenario', 'sector_file', 'gas', 'years']).reset_index(drop=True)

# Apply reformatting to sector_files data
df_sector_files_melted = reformatting_names_units(df_sector_files_melted)

print(f"Sector files melted dataframe shape: {df_sector_files_melted.shape}")
print(df_sector_files_melted.head(10))


# - total sum -
# Group by scenario and year, sum across all numeric columns
KEEP_COLS = ['model', 'scenario', 'gas', 'unit']
df_summed = df_combined.groupby(KEEP_COLS)[numeric_cols].sum().reset_index()
print(f"\nSummed dataframe shape: {df_summed.shape}")
print(df_summed.head())

# Melt the dataframe to long format for easier plotting
df_melted = df_summed.melt(
    id_vars=KEEP_COLS,
    var_name='years',
    value_name='value'
)

# Convert years to numeric and sort
df_melted['years'] = pd.to_numeric(df_melted['years'])
df_melted = df_melted.sort_values(['model', 'scenario', 'gas', 'years']).reset_index(drop=True)

df_melted = reformatting_names_units(df_melted)

print(f"Melted dataframe shape: {df_melted.shape}")
print(df_melted.head(10))

# --------- history ---------
hist = (
    pd.read_csv(settings.history_path / HISTORY_FILE) # like "cmip7_history_countrylevel_250721.csv" 
    .drop(columns=['model', 'scenario'])
    .rename(columns={"region": "country"})
)

hist = extractlevel(hist.set_index(['country', 'variable', 'unit']), variable="Emissions|{gas}|{sector}", drop=True)

# Reorder the MultiIndex of hist
hist = hist.reorder_levels(['country', 'gas', 'sector', 'unit'])
hist = hist.sort_index()

# Update column type and name
hist.columns = hist.columns.astype(int)
hist.columns.name = 'year'

# History fixes:

# only country-level emissions
hist_nonglobal = hist.loc[~isin(country="global")]
hist_nonglobal = hist_nonglobal.loc[~ismatch(sector=["**Shipping", "**Aircraft"])]# let's also make sure there's no shipping (10.08.2025: zero values present) and aircraft (10.08.2025: no data) anymore for country-level data 

# keep international/bunkers emissions & rename to country='World'
hist_global = hist.loc[isin(country="global")]
hist_global_nonzero = hist_global[ismatch(sector=["**Shipping", "**Aircraft"])]
hist_global_nonzero = hist_global_nonzero.rename(index=lambda v: v.replace("global", "World"))

# calculate the sum of all countries for the other countries
hist_nonglobal_world = assignlevel(hist_nonglobal.groupby(["gas", "sector", "unit"]).sum(), country="World").reorder_levels(["country","gas", "sector", "unit"])

# recombine
hist = pd.concat([
    hist_nonglobal,
    hist_global_nonzero,
    hist_nonglobal_world
]).reset_index()

# Remove CO2 emissions from openburning sector
openburning_sectors = SECTOR_FILE_DICT['openburning']
hist = hist[~((hist['gas'] == 'CO2') & (hist['sector'].isin(openburning_sectors)))]
# Remove pre-processed Agriculture emissions (as they should be zero)
hist = hist[~((hist['gas'] == 'CO2') & (hist['sector'].isin(["Agriculture"])))]

# create here two dataframes in the same way as above: one sum/melted `hist_melted` and one sector_file one `hist_sector_files`

# - total sum -
# Group by country, gas, sector, and unit, sum across all year columns
KEEP_COLS_HIST = ['country', 'gas', 'sector', 'unit']
numeric_cols_hist = hist.select_dtypes(include=[np.number]).columns.tolist()
# df_hist_summed = hist.groupby(KEEP_COLS_HIST)[numeric_cols_hist].sum().reset_index()
# only keep worlf total
hist = hist[hist['country']=='World']

# Melt the dataframe to long format for easier plotting
hist_melted = hist.melt(
    id_vars=KEEP_COLS_HIST,
    var_name='years',
    value_name='value'
)

# Convert years to numeric and sort
hist_melted['years'] = pd.to_numeric(hist_melted['years'])
hist_melted = hist_melted.sort_values(['gas', 'sector', 'years']).reset_index(drop=True)

hist_melted = reformatting_names_units(hist_melted) # with sectors still

hist_melted_sum = hist_melted.groupby(['gas','unit','years'])['value'].sum().reset_index()

print(f"History melted dataframe shape: {hist_melted.shape}")
print(hist_melted.head(10))

# - sector_file -
# Add sector_file column to hist using the same mapping
hist_with_sectorfiles = hist.copy()
hist_with_sectorfiles['sector_file'] = hist_with_sectorfiles['sector'].map(sector_to_file)

# Group by country, sector_file, gas, and unit, sum across all year columns
KEEP_COLS_HIST_SF = ['country', 'gas', 'sector_file', 'unit']
df_hist_sector_files = hist_with_sectorfiles.groupby(KEEP_COLS_HIST_SF)[numeric_cols_hist].sum().reset_index()

# Melt the dataframe to long format
hist_sector_files = df_hist_sector_files.melt(
    id_vars=KEEP_COLS_HIST_SF,
    var_name='years',
    value_name='value'
)

# Convert years to numeric and sort
hist_sector_files['years'] = pd.to_numeric(hist_sector_files['years'])
hist_sector_files = hist_sector_files.sort_values(['country', 'gas', 'sector_file', 'years']).reset_index(drop=True)

hist_sector_files = reformatting_names_units(hist_sector_files)

print(f"\nHistory sector files dataframe shape: {hist_sector_files.shape}")
print(hist_sector_files.head(10))


# %%
# Get unique gases and limit to 10 for the 2x5 grid
unique_gases = sorted(df_melted['gas'].unique())
print(f"Total unique gases: {len(unique_gases)}")
print(f"Gases: {unique_gases}")



# If more than 10 gases, take the first 10
gases_to_plot = unique_gases[:10]
df_plot = df_melted[df_melted['gas'].isin(gases_to_plot)]











# %% [markdown]
# # Total emissions (sum of all sectors)

# %%
# Define years to mark with dots: 2023, 2024, 2025, then every 5 years until 2100
years_to_mark = [2023, 2024, 2025] + list(range(2030, 2105, 5))

# Define colors for scenarios (colorblind-friendly, similar to originals)
colors_map = {
    'H': '#D55E00',    # dark orange-red (closer to original dark red)
    'VL': '#2E5EAA'    # darker blue (similar tone to original)
}

# Create 2x5 facet plot
fig, axes = plt.subplots(2, 5, figsize=(20, 10))
axes = axes.flatten()

for idx, gas in enumerate(gases_to_plot):
    ax = axes[idx]
    gas_data = df_plot[df_plot['gas'] == gas]
    
    # Get the unit for this gas
    gas_unit = gas_data['unit'].iloc[0] if len(gas_data) > 0 else ''
    
    # Plot a line for each scenario
    for scenario in sorted(gas_data['scenario'].unique()):
        scenario_data = gas_data[gas_data['scenario'] == scenario].sort_values('years')
        color = colors_map[scenario]
        
        # Plot line with color determined by scenario
        ax.plot(scenario_data['years'], scenario_data['value'], label=scenario, linewidth=2, color=color)
        
        # Add markers with same color
        marked_data = scenario_data[scenario_data['years'].isin(years_to_mark)]
        ax.plot(marked_data['years'], marked_data['value'], marker='o', linestyle='none', markersize=6, color=color)
    
    ax.set_title(f'{gas}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Year')
    ax.set_ylabel(f'Emissions ({gas_unit})')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(output_folder / '01_total_emissions_sum_all_sectors.png', dpi=300, bbox_inches='tight')
plt.show()


# %% [markdown]
# # Total emissions (stacked area plot across sectors)


# %%
# Get unique gases and limit to 10 for the 2x5 grid
unique_gases_sf = sorted(df_sector_files_melted['gas'].unique())
gases_to_plot_sf = unique_gases_sf[:10]
df_plot_sf = df_sector_files_melted[df_sector_files_melted['gas'].isin(gases_to_plot_sf)]

# Define colors for sector_file categories
sector_file_colors = {
    'openburning': '#E69F00',   # orange
    'anthro': '#56B4E9',        # light blue
    'AIR-anthro': '#009E73'     # green
}

# %%
# Create stacked area plots for each scenario
for scenario in sorted(df_plot_sf['scenario'].unique()):
    fig, axes = plt.subplots(2, 5, figsize=(20, 10))
    axes = axes.flatten()
    
    scenario_data = df_plot_sf[df_plot_sf['scenario'] == scenario]
    
    for idx, gas in enumerate(gases_to_plot_sf):
        ax = axes[idx]
        gas_data = scenario_data[scenario_data['gas'] == gas]
        
        # Get the unit for this gas
        gas_unit = gas_data['unit'].iloc[0] if len(gas_data) > 0 else ''
        
        # Pivot data for stacked bar chart
        pivot_data = gas_data.pivot_table(
            index='years',
            columns='sector_file',
            values='value',
            aggfunc='sum'
        )
        
        # Filter to only include years_to_mark
        pivot_data = pivot_data.loc[pivot_data.index.isin(years_to_mark)].sort_index()
        
        # Fill NaN with 0
        pivot_data = pivot_data.fillna(0)
        
        # Create stacked bar chart
        pivot_data.plot(
            kind='bar',
            stacked=True,
            ax=ax,
            color=[sector_file_colors.get(col, '#CCCCCC') for col in pivot_data.columns],
            width=0.8
        )
        
        ax.set_title(f'{gas}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Year')
        ax.set_ylabel(f'Emissions ({gas_unit})')
        ax.legend(title='Sector File', loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(f'Emissions by Sector - Scenario {scenario}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(output_folder / f'02_total_emissions_stacked_bars_scenario_{scenario}.png', dpi=300, bbox_inches='tight')
    plt.show()



# %%

# %% [markdown]
# Add history to the plots above

# %%
# repeat the plots above, with historical emissions added

# %% [markdown]
# # Total emissions with historical data (sum of all sectors)

# %%
# Filter historical data for gases in gases_to_plot and years >= 1990
hist_plot = hist_melted_sum[(hist_melted_sum['gas'].isin(gases_to_plot)) & (hist_melted_sum['years'] >= 1990)]

# Create 2x5 facet plot with both scenario and historical data
fig, axes = plt.subplots(2, 5, figsize=(20, 10))
axes = axes.flatten()

for idx, gas in enumerate(gases_to_plot):
    ax = axes[idx]
    gas_data = df_plot[df_plot['gas'] == gas]
    
    # Get the unit for this gas
    gas_unit = gas_data['unit'].iloc[0] if len(gas_data) > 0 else ''
    
    # Plot a line for each scenario
    for scenario in sorted(gas_data['scenario'].unique()):
        scenario_data = gas_data[gas_data['scenario'] == scenario].sort_values('years')
        color = colors_map[scenario]
        
        # Plot line with color determined by scenario
        ax.plot(scenario_data['years'], scenario_data['value'], label=scenario, linewidth=2, color=color)
        
        # Add markers with same color
        marked_data = scenario_data[scenario_data['years'].isin(years_to_mark)]
        ax.plot(marked_data['years'], marked_data['value'], marker='o', linestyle='none', markersize=6, color=color)
    
    # Add historical data
    gas_hist = hist_plot[hist_plot['gas'] == gas].sort_values('years')
    if len(gas_hist) > 0:
        ax.plot(gas_hist['years'], gas_hist['value'], label='Historical', linewidth=2.5, color='black', linestyle='--')
        ax.plot(gas_hist['years'], gas_hist['value'], marker='s', linestyle='none', markersize=6, color='black')
    
    ax.set_title(f'{gas}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Year')
    ax.set_ylabel(f'Emissions ({gas_unit})')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(output_folder / '03_total_emissions_with_history_sum_all_sectors.png', dpi=300, bbox_inches='tight')
plt.show()


# %% [markdown]
# # Total emissions with historical data (stacked area plot across sectors)

# %%
# Filter historical sector_files data for gases in gases_to_plot_sf and years >= 2010
hist_plot_sf = hist_sector_files[(hist_sector_files['gas'].isin(gases_to_plot_sf)) & (hist_sector_files['years'] >= 2010)]

# Get all unique years from hist_plot_sf to expand the x-axis
hist_years = sorted(hist_plot_sf['years'].unique())

# Create stacked area plots for each scenario, plus a historical plot
for scenario in sorted(df_plot_sf['scenario'].unique()):
    fig, axes = plt.subplots(2, 5, figsize=(20, 10))
    axes = axes.flatten()
    
    scenario_data = df_plot_sf[df_plot_sf['scenario'] == scenario]
    
    for idx, gas in enumerate(gases_to_plot_sf):
        ax = axes[idx]
        gas_data = scenario_data[scenario_data['gas'] == gas]
        
        # Get the unit for this gas
        gas_unit = gas_data['unit'].iloc[0] if len(gas_data) > 0 else ''
        
        # Pivot data for stacked bar chart
        pivot_data = gas_data.pivot_table(
            index='years',
            columns='sector_file',
            values='value',
            aggfunc='sum'
        )
        
        # Fill NaN with 0
        pivot_data = pivot_data.fillna(0)
        
        # Ensure all sector_files are present as columns (even if all zeros)
        all_sector_files = sorted(sector_file_colors.keys())
        pivot_data = pivot_data.reindex(columns=all_sector_files, fill_value=0)
        
        # Add historical data if available
        gas_hist = hist_plot_sf[hist_plot_sf['gas'] == gas]
        hist_pivot = None
        if len(gas_hist) > 0:
            hist_pivot = gas_hist.pivot_table(
                index='years',
                columns='sector_file',
                values='value',
                aggfunc='sum'
            )
            # Include all historical years (not just years_to_mark)
            hist_pivot = hist_pivot.sort_index()
            hist_pivot = hist_pivot.fillna(0)
            # Ensure all sector_files are present as columns
            hist_pivot = hist_pivot.reindex(columns=all_sector_files, fill_value=0)
        
        # Combine all years: historical years + scenario years_to_mark
        all_years = sorted(set(list(hist_years) + years_to_mark))
        
        # Reindex both dataframes to include all years (fill missing with 0)
        pivot_data = pivot_data.reindex(all_years, fill_value=0)
        if hist_pivot is not None:
            hist_pivot = hist_pivot.reindex(all_years, fill_value=0)
        
        # Find the year 2023 for the grey background
        years_before_2023 = [y for y in all_years if y <= 2023]
        grey_span_end = years_before_2023[-1] if years_before_2023 else all_years[0]
        
        # Add light grey background until 2023
        ax.axvspan(all_years[0] - 2, grey_span_end + 0.5, alpha=0.1, color='grey', zorder=0)
        
        # Define bar width based on year (4x wider for 2030+)
        def get_bar_width(year):
            return 2.4 if year >= 2030 else 0.6
        
        # Function to compute split stacking with positive and negative values separated
        def compute_split_stacked_bars(data_to_plot):
            """
            Create stacked bars where positive and negative values are separated.
            Positive values stack upward from 0, negative values stack downward from 0.
            Returns list of (col_name, values, bottom_position) tuples.
            """
            # Define desired column order
            col_order = ['anthro', 'AIR-anthro', 'openburning']
            cols = [col for col in col_order if col in data_to_plot.columns]
            
            num_years = len(data_to_plot.index)
            results = []
            
            # Track separate stacking for positive and negative
            bottom_pos = np.zeros(num_years)
            bottom_neg = np.zeros(num_years)
            
            for col in cols:
                values = data_to_plot[col].values
                
                # Separate positive and negative values
                pos_values = np.where(values >= 0, values, 0)
                neg_values = np.where(values < 0, values, 0)
                
                # Add positive values (stack upward from 0)
                results.append((col, pos_values, bottom_pos.copy()))
                bottom_pos = bottom_pos + pos_values
                
                # Add negative values (stack downward from 0)
                results.append((col, neg_values, bottom_neg.copy()))
                bottom_neg = bottom_neg + neg_values
            
            return results
        
        # Plot historical data with split stacked bars (alpha=0.3)
        if hist_pivot is not None:
            hist_bars = compute_split_stacked_bars(hist_pivot)
            plotted_labels = set()  # Track which labels we've added
            for col, values, bottom_pos in hist_bars:
                # Only plot if there are non-zero values
                if np.any(values != 0):
                    widths = [get_bar_width(year) for year in all_years]
                    label = col if col not in plotted_labels else ''
                    plotted_labels.add(col)
                    ax.bar(all_years, values, widths,
                          bottom=bottom_pos, label=label,
                          color=sector_file_colors.get(col, '#CCCCCC'),
                          alpha=0.6)
        
        # Plot scenario data with split stacked bars (full opacity)
        scenario_bars = compute_split_stacked_bars(pivot_data)
        plotted_labels = set()  # Reset for scenario
        for col, values, bottom_pos in scenario_bars:
            # Only plot if there are non-zero values
            if np.any(values != 0):
                widths = [get_bar_width(year) for year in all_years]
                label = col if hist_pivot is None and col not in plotted_labels else ''
                plotted_labels.add(col)
                ax.bar(all_years, values, widths, 
                      bottom=bottom_pos, label=label,
                      color=sector_file_colors.get(col, '#CCCCCC'))
        
        # Set x-axis ticks to every 5 years
        tick_years = [y for y in all_years if y % 5 == 0]
        ax.set_xticks(tick_years)
        ax.set_xticklabels(tick_years, rotation=90)
        ax.set_xlim(all_years[0] - 2, all_years[-1] + 2)
        
        ax.set_title(f'{gas}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Year')
        ax.set_ylabel(f'Emissions ({gas_unit})')
        
        # Overlay total sum from df_plot for this scenario and gas
        gas_total = df_plot[(df_plot['scenario'] == scenario) & (df_plot['gas'] == gas)].sort_values('years')
        if len(gas_total) > 0:
            # Plot total line without markers
            ax.plot(gas_total['years'], gas_total['value'], 
                   color='black', linewidth=2.5, linestyle='-', zorder=5)
            # Add markers only at years_to_mark
            marked_total = gas_total[gas_total['years'].isin(years_to_mark)]
            ax.plot(marked_total['years'], marked_total['value'], 
                   color='black', marker='o', markersize=6, linestyle='none', label='Total', zorder=5)
        
        # Overlay historical total sum if available
        gas_hist_total = hist_melted_sum[hist_melted_sum['gas'] == gas].sort_values('years')
        if len(gas_hist_total) > 0:
            # Plot historical total line without markers
            ax.plot(gas_hist_total['years'], gas_hist_total['value'], 
                   color='black', linewidth=2.5, linestyle='--', zorder=5)
            # Add markers only at years present in hist_plot_sf
            marked_hist_total = gas_hist_total[gas_hist_total['years'].isin(hist_years)]
            ax.plot(marked_hist_total['years'], marked_hist_total['value'], 
                   color='black', marker='s', markersize=5, linestyle='none', label='Historical Total', zorder=5)
        
        # Create custom legend with unique entries, only for N2O
        if gas == 'N2O':
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), title='Sector File', loc='lower left', fontsize=8)
        
        ax.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle(f'Emissions by Sector - Scenario {scenario}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(output_folder / f'04_total_emissions_with_history_stacked_bars_scenario_{scenario}.png', dpi=300, bbox_inches='tight')
    plt.show()


# %%
