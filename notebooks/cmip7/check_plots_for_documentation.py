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
VERSION_ESGF: str = "1-1-0" # for second ESGF version

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

from pathlib import Path

from dask.diagnostics import ProgressBar
import pandas as pd
from pandas_indexing import isin, ismatch, assignlevel, extractlevel
from pandas_indexing.units import set_openscm_registry_as_default
import concordia._patches_ptolemy # seemingly not used, not used in this script, but sets fill_value for xarray_reduce to 0 

from concordia.cmip7 import utils as cmip7_utils # update to cmip7 utils (e.g. for dressing up netcdf)
from concordia.settings import Settings
from concordia.cmip7.CONSTANTS import return_marker_information
from concordia.cmip7.utils import SECTOR_DICT_ANTHRO_DEFAULT, SECTOR_DICT_ANTHRO_CO2_SCENARIO 
from concordia.cmip7.utils_plotting import ds_to_annual_emissions_total

import xarray as xr
import numpy as np
import os

import matplotlib.pyplot as plt

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
# PATH_RESULTS = Path('D:\\concordia-results\\rc4')
PATH_RESULTS = Path('C:\\Users\\kikstra\\IIASA\\ECE.prog - Documents\\Projects\\CMIP7\\IAM Data Processing\\Shared emission fields data\\v1_1-testing-findmistakes')


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

# %% [markdown]
# # Comparing two versions

# %%
# load (aggregate) data

# old
VERSION_ESGF_OLD = '1-0-0'
PATH_RESULTS_OLD = Path('C:\\Users\\kikstra\\IIASA\\ECE.prog - Documents\\Projects\\CMIP7\\IAM Data Processing\\Shared emission fields data\\v1_0-testing-findmistakes\\rc4')
location_scenario_h_old = PATH_RESULTS_OLD / f"h_{VERSION_ESGF_OLD}" / "check_annual_totals (redone)"
location_scenario_vl_old = PATH_RESULTS_OLD / f"vl_{VERSION_ESGF_OLD}" / "check_annual_totals (redone)"


# new
PATH_RESULTS = Path('C:\\Users\\kikstra\\IIASA\\ECE.prog - Documents\\Projects\\CMIP7\\IAM Data Processing\\Shared emission fields data\\v1_1-testing-findmistakes')
location_scenario_h = PATH_RESULTS / f"h_{VERSION_ESGF}" / "check_annual_totals"
location_scenario_vl = PATH_RESULTS / f"vl_{VERSION_ESGF}" / "check_annual_totals"


# %% [markdown]
# ## Comparing two versions: A1) global totals per files

# %%

# load totals
h_old_tot = pd.read_csv(os.path.join(location_scenario_h_old, 'BC-em-anthro_input4MIPs_emissions_ScenarioMIP_IIASA-IAMC-h-1-0-0_gn_202201-210012_annual_totals.csv'))
vl_old_tot = pd.read_csv(os.path.join(location_scenario_vl_old, 'BC-em-anthro_input4MIPs_emissions_ScenarioMIP_IIASA-IAMC-vl-1-0-0_gn_202201-210012_annual_totals.csv'))
h_new_tot = pd.read_csv(os.path.join(location_scenario_h, 'BC-em-anthro_input4MIPs_emissions_ScenarioMIP_IIASA-IAMC-h-1-1-0_gn_202201-210012_annual_totals.csv'))
vl_new_tot = pd.read_csv(os.path.join(location_scenario_vl, 'BC-em-anthro_input4MIPs_emissions_ScenarioMIP_IIASA-IAMC-vl-1-1-0_gn_202201-210012_annual_totals.csv'))


# %% [markdown]
# ## Comparing two versions: A2) global totals per files

# %%

# load totals by sector
h_old_sec = pd.read_csv(os.path.join(location_scenario_h_old, 'BC-em-anthro_input4MIPs_emissions_ScenarioMIP_IIASA-IAMC-h-1-0-0_gn_202201-210012_annual_totals_by_sector.csv'))
vl_old_sec = pd.read_csv(os.path.join(location_scenario_vl_old, 'BC-em-anthro_input4MIPs_emissions_ScenarioMIP_IIASA-IAMC-vl-1-0-0_gn_202201-210012_annual_totals_by_sector.csv'))
h_new_sec = pd.read_csv(os.path.join(location_scenario_h, 'BC-em-anthro_input4MIPs_emissions_ScenarioMIP_IIASA-IAMC-h-1-1-0_gn_202201-210012_annual_totals_by_sector.csv'))
vl_new_sec = pd.read_csv(os.path.join(location_scenario_vl, 'BC-em-anthro_input4MIPs_emissions_ScenarioMIP_IIASA-IAMC-vl-1-1-0_gn_202201-210012_annual_totals_by_sector.csv'))




# %% [markdown]
# ## Comparing two versions: Loop over all species and emission types

# %%
# Define species and emission types to loop over
SPECIES_LIST = ["BC", "CH4", "CO", "CO2", "N2O", "NH3", "NOx", "OC", "SO2", "NMVOC", "NMVOCbulk"]
EMISSION_TYPES = ["em-anthro", "em-openburning", "em-AIR-anthro"]

# Create output folder for comparison plots
comparison_output_folder = output_folder / "version_comparison"
comparison_output_folder.mkdir(exist_ok=True)
print(f"Comparison plots will be saved to: {comparison_output_folder}")

# Define sector colors for multi-sector plots
SECTOR_COLORS = plt.cm.tab20.colors  # Use a colormap with many distinct colors

# %%
def get_file_pattern(species, emission_type, marker, version):
    """Generate filename pattern for annual totals CSV files."""
    return f"{species}-{emission_type}_input4MIPs_emissions_ScenarioMIP_IIASA-IAMC-{marker}-{version}_gn_202201-210012_annual_totals.csv"

def get_sector_file_pattern(species, emission_type, marker, version):
    """Generate filename pattern for annual totals by sector CSV files."""
    return f"{species}-{emission_type}_input4MIPs_emissions_ScenarioMIP_IIASA-IAMC-{marker}-{version}_gn_202201-210012_annual_totals_by_sector.csv"

def load_totals_data(location, species, emission_type, marker, version):
    """Load total emissions data, return None if file doesn't exist.
    
    File format: year, emissions_Mt_year
    """
    filename = get_file_pattern(species, emission_type, marker, version)
    filepath = os.path.join(location, filename)
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    return None

def load_sector_data(location, species, emission_type, marker, version):
    """Load sector emissions data, return None if file doesn't exist.
    
    File format: wide format with columns: year, 0.0, 1.0, 2.0, ... (sector indices)
    """
    filename = get_sector_file_pattern(species, emission_type, marker, version)
    filepath = os.path.join(location, filename)
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    return None

def get_sector_columns(df):
    """Get sector column names (all columns except 'year')."""
    if df is None:
        return []
    return [col for col in df.columns if col != 'year']

def get_sector_name(sector_col, emission_type, species):
    """Map sector number column to sector name based on emission type and species.
    
    Args:
        sector_col: Column name (e.g., '0.0', '1.0')
        emission_type: One of 'em-anthro', 'em-openburning', 'em-AIR-anthro'
        species: Species name (e.g., 'CO2', 'BC')
    
    Returns:
        Sector name string
    """
    try:
        sector_num = int(float(sector_col))
    except (ValueError, TypeError):
        return sector_col  # Return as-is if not a number
    
    if emission_type == 'em-openburning':
        return SECTOR_DICT_OPENBURNING_DEFAULT.get(sector_num, f'Sector {sector_num}')
    elif emission_type == 'em-AIR-anthro':
        return 'Aircraft' if sector_num == 0 else f'Sector {sector_num}'
    elif emission_type == 'em-anthro':
        # Use CO2 scenario dict for CO2 (has CDR sectors), otherwise default
        if species == 'CO2':
            return SECTOR_DICT_ANTHRO_CO2_SCENARIO.get(sector_num, f'Sector {sector_num}')
        else:
            return SECTOR_DICT_ANTHRO_DEFAULT.get(sector_num, f'Sector {sector_num}')
    else:
        return f'Sector {sector_num}'

def enforce_min_yrange(ax, min_range=0.01):
    """Ensure the y-axis span is at least min_range (default 0.01%).
    Expands symmetrically around the midpoint if needed.
    """
    ymin, ymax = ax.get_ylim()
    current_range = ymax - ymin
    if current_range < min_range:
        mid = (ymin + ymax) / 2
        ax.set_ylim(mid - min_range / 2, mid + min_range / 2)

# %% [markdown]
# ## Comparing two versions: B1) Global totals comparison (new vs old)

# %%
# Loop over species and emission types - Total emissions comparison
for species in SPECIES_LIST:
    for emission_type in EMISSION_TYPES:
        print(f"Processing {species} - {emission_type}...")
        
        # Load data for H scenario
        h_old = load_totals_data(location_scenario_h_old, species, emission_type, 'h', VERSION_ESGF_OLD)
        h_new = load_totals_data(location_scenario_h, species, emission_type, 'h', VERSION_ESGF)
        
        # Load data for VL scenario
        vl_old = load_totals_data(location_scenario_vl_old, species, emission_type, 'vl', VERSION_ESGF_OLD)
        vl_new = load_totals_data(location_scenario_vl, species, emission_type, 'vl', VERSION_ESGF)
        
        # Skip if no data available
        if all(x is None for x in [h_old, h_new, vl_old, vl_new]):
            print(f"  Skipping {species} - {emission_type}: no data found")
            continue
        
        # Create figure with 2 subplots (H and VL scenarios)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot H scenario
        ax = axes[0]
        # Plot new (solid) first, then old (dashed) on top
        if h_new is not None and 'year' in h_new.columns and 'emissions_Mt_year' in h_new.columns:
            ax.plot(h_new['year'], h_new['emissions_Mt_year'], 'b-', linewidth=2, label=f'New ({VERSION_ESGF})')
        if h_old is not None and 'year' in h_old.columns and 'emissions_Mt_year' in h_old.columns:
            ax.plot(h_old['year'], h_old['emissions_Mt_year'], color='cornflowerblue', linestyle='--', linewidth=2, alpha=0.8, label=f'Old ({VERSION_ESGF_OLD})')
        ax.set_title(f'{species} - {emission_type} (H)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Year')
        ax.set_ylabel('Total Emissions (Mt/yr)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot VL scenario
        ax = axes[1]
        # Plot new (solid) first, then old (dashed) on top
        if vl_new is not None and 'year' in vl_new.columns and 'emissions_Mt_year' in vl_new.columns:
            ax.plot(vl_new['year'], vl_new['emissions_Mt_year'], 'r-', linewidth=2, label=f'New ({VERSION_ESGF})')
        if vl_old is not None and 'year' in vl_old.columns and 'emissions_Mt_year' in vl_old.columns:
            ax.plot(vl_old['year'], vl_old['emissions_Mt_year'], color='lightsalmon', linestyle='--', linewidth=2, alpha=0.8, label=f'Old ({VERSION_ESGF_OLD})')
        ax.set_title(f'{species} - {emission_type} (VL)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Year')
        ax.set_ylabel('Total Emissions (Mt/yr)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig.savefig(comparison_output_folder / f'B1_total_{species}_{emission_type}_comparison.png', dpi=100, bbox_inches='tight')
        plt.close(fig)

print("B1 total emissions comparison plots complete.")

# %% [markdown]
# ## Comparing two versions: B2) Global totals by sector comparison

# %%
# Loop over species and emission types - Sector emissions comparison
for species in SPECIES_LIST:
    for emission_type in EMISSION_TYPES:
        print(f"Processing sectors for {species} - {emission_type}...")
        
        # Load sector data for H scenario
        h_old_sec = load_sector_data(location_scenario_h_old, species, emission_type, 'h', VERSION_ESGF_OLD)
        h_new_sec = load_sector_data(location_scenario_h, species, emission_type, 'h', VERSION_ESGF)
        
        # Load sector data for VL scenario
        vl_old_sec = load_sector_data(location_scenario_vl_old, species, emission_type, 'vl', VERSION_ESGF_OLD)
        vl_new_sec = load_sector_data(location_scenario_vl, species, emission_type, 'vl', VERSION_ESGF)
        
        # Skip if no data available
        if all(x is None for x in [h_old_sec, h_new_sec, vl_old_sec, vl_new_sec]):
            print(f"  Skipping {species} - {emission_type}: no sector data found")
            continue
        
        # Create figure with 2 subplots (H and VL scenarios)
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Get all unique sector columns across all datasets
        all_sectors = set()
        for df in [h_old_sec, h_new_sec, vl_old_sec, vl_new_sec]:
            all_sectors.update(get_sector_columns(df))
        all_sectors = sorted(list(all_sectors), key=lambda x: float(x) if x.replace('.', '').replace('-', '').isdigit() else x)
        
        # Create color mapping for sectors
        sector_color_map = {sector: SECTOR_COLORS[i % len(SECTOR_COLORS)] for i, sector in enumerate(all_sectors)}
        
        # Helper function to lighten a color
        def lighten_color(color, amount=0.3):
            import matplotlib.colors as mcolors
            c = mcolors.to_rgb(color)
            return tuple(min(1, x + (1 - x) * amount) for x in c)
        
        # Plot H scenario
        ax = axes[0]
        for sector in all_sectors:
            color = sector_color_map[sector]
            light_color = lighten_color(color, 0.4)
            sector_name = get_sector_name(sector, emission_type, species)
            # New data (solid) first
            if h_new_sec is not None and sector in h_new_sec.columns:
                ax.plot(h_new_sec['year'], h_new_sec[sector], '-', color=color, linewidth=1.5, label=sector_name)
            # Old data (dashed) on top with lighter color
            if h_old_sec is not None and sector in h_old_sec.columns:
                ax.plot(h_old_sec['year'], h_old_sec[sector], '--', color=light_color, linewidth=1.5, alpha=0.9)
        
        ax.set_title(f'{species} - {emission_type} (H)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Year')
        ax.set_ylabel('Sector Emissions (Mt/yr)')
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Plot VL scenario
        ax = axes[1]
        for sector in all_sectors:
            color = sector_color_map[sector]
            light_color = lighten_color(color, 0.4)
            sector_name = get_sector_name(sector, emission_type, species)
            # New data (solid) first
            if vl_new_sec is not None and sector in vl_new_sec.columns:
                ax.plot(vl_new_sec['year'], vl_new_sec[sector], '-', color=color, linewidth=1.5, label=sector_name)
            # Old data (dashed) on top with lighter color
            if vl_old_sec is not None and sector in vl_old_sec.columns:
                ax.plot(vl_old_sec['year'], vl_old_sec[sector], '--', color=light_color, linewidth=1.5, alpha=0.9)
        
        ax.set_title(f'{species} - {emission_type} (VL)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Year')
        ax.set_ylabel('Sector Emissions (Mt/yr)')
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig.savefig(comparison_output_folder / f'B2_sectors_{species}_{emission_type}_comparison.png', dpi=100, bbox_inches='tight')
        plt.close(fig)

print("B2 sector emissions comparison plots complete.")

# %% [markdown]
# ## Comparing two versions: C1) Percentage difference (relative to 2023 old value) - Totals

# %%
# Loop over species and emission types - Percentage difference for totals
for species in SPECIES_LIST:
    for emission_type in EMISSION_TYPES:
        print(f"Processing % diff for {species} - {emission_type}...")
        
        # Load data for H scenario
        h_old = load_totals_data(location_scenario_h_old, species, emission_type, 'h', VERSION_ESGF_OLD)
        h_new = load_totals_data(location_scenario_h, species, emission_type, 'h', VERSION_ESGF)
        
        # Load data for VL scenario
        vl_old = load_totals_data(location_scenario_vl_old, species, emission_type, 'vl', VERSION_ESGF_OLD)
        vl_new = load_totals_data(location_scenario_vl, species, emission_type, 'vl', VERSION_ESGF)
        
        # Skip if no data available
        if all(x is None for x in [h_old, h_new, vl_old, vl_new]):
            print(f"  Skipping {species} - {emission_type}: no data found")
            continue
        
        # Create figure with 2 subplots (H and VL scenarios)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot H scenario percentage difference
        ax = axes[0]
        if h_old is not None and h_new is not None:
            if 'year' in h_old.columns and 'emissions_Mt_year' in h_old.columns and 'year' in h_new.columns and 'emissions_Mt_year' in h_new.columns:
                # Get 2023 value from old data as reference
                ref_value_h = h_old[h_old['year'] == 2023]['emissions_Mt_year'].values
                if len(ref_value_h) > 0 and ref_value_h[0] != 0:
                    ref_value_h = ref_value_h[0]
                    # Merge on year to compute difference
                    merged_h = pd.merge(h_old[['year', 'emissions_Mt_year']], h_new[['year', 'emissions_Mt_year']], 
                                        on='year', suffixes=('_old', '_new'))
                    merged_h['pct_diff'] = ((merged_h['emissions_Mt_year_new'] - merged_h['emissions_Mt_year_old']) / abs(ref_value_h)) * 100
                    ax.plot(merged_h['year'], merged_h['pct_diff'], 'b-', linewidth=2, label='(New - Old) / Old_2023')
                    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
        
        ax.set_title(f'{species} - {emission_type} (H) - % Diff', fontsize=12, fontweight='bold')
        ax.set_xlabel('Year')
        ax.set_ylabel('% Difference (rel. to 2023 old)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        enforce_min_yrange(ax)
        
        # Plot VL scenario percentage difference
        ax = axes[1]
        if vl_old is not None and vl_new is not None:
            if 'year' in vl_old.columns and 'emissions_Mt_year' in vl_old.columns and 'year' in vl_new.columns and 'emissions_Mt_year' in vl_new.columns:
                # Get 2023 value from old data as reference
                ref_value_vl = vl_old[vl_old['year'] == 2023]['emissions_Mt_year'].values
                if len(ref_value_vl) > 0 and ref_value_vl[0] != 0:
                    ref_value_vl = ref_value_vl[0]
                    # Merge on year to compute difference
                    merged_vl = pd.merge(vl_old[['year', 'emissions_Mt_year']], vl_new[['year', 'emissions_Mt_year']], 
                                         on='year', suffixes=('_old', '_new'))
                    merged_vl['pct_diff'] = ((merged_vl['emissions_Mt_year_new'] - merged_vl['emissions_Mt_year_old']) / abs(ref_value_vl)) * 100
                    ax.plot(merged_vl['year'], merged_vl['pct_diff'], 'r-', linewidth=2, label='(New - Old) / Old_2023')
                    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
        
        ax.set_title(f'{species} - {emission_type} (VL) - % Diff', fontsize=12, fontweight='bold')
        ax.set_xlabel('Year')
        ax.set_ylabel('% Difference (rel. to 2023 old)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        enforce_min_yrange(ax)
        
        plt.tight_layout()
        fig.savefig(comparison_output_folder / f'C1_pct_diff_total_{species}_{emission_type}.png', dpi=100, bbox_inches='tight')
        plt.close(fig)

print("C1 percentage difference plots (totals) complete.")

# %% [markdown]
# ## Comparing two versions: C2) Percentage difference (relative to 2023 old value) - By Sector

# %%
# Loop over species and emission types - Percentage difference for sectors
for species in SPECIES_LIST:
    for emission_type in EMISSION_TYPES:
        print(f"Processing % diff sectors for {species} - {emission_type}...")
        
        # Load sector data for H scenario
        h_old_sec = load_sector_data(location_scenario_h_old, species, emission_type, 'h', VERSION_ESGF_OLD)
        h_new_sec = load_sector_data(location_scenario_h, species, emission_type, 'h', VERSION_ESGF)
        
        # Load sector data for VL scenario
        vl_old_sec = load_sector_data(location_scenario_vl_old, species, emission_type, 'vl', VERSION_ESGF_OLD)
        vl_new_sec = load_sector_data(location_scenario_vl, species, emission_type, 'vl', VERSION_ESGF)
        
        # Skip if no data available
        if all(x is None for x in [h_old_sec, h_new_sec, vl_old_sec, vl_new_sec]):
            print(f"  Skipping {species} - {emission_type}: no sector data found")
            continue
        
        # Create figure with 2 subplots (H and VL scenarios)
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Get all unique sector columns across all datasets
        all_sectors = set()
        for df in [h_old_sec, h_new_sec, vl_old_sec, vl_new_sec]:
            all_sectors.update(get_sector_columns(df))
        all_sectors = sorted(list(all_sectors), key=lambda x: float(x) if x.replace('.', '').replace('-', '').isdigit() else x)
        
        # Create color mapping for sectors
        sector_color_map = {sector: SECTOR_COLORS[i % len(SECTOR_COLORS)] for i, sector in enumerate(all_sectors)}
        
        # Plot H scenario percentage difference
        ax = axes[0]
        if h_old_sec is not None and h_new_sec is not None:
            for sector in all_sectors:
                if sector in h_old_sec.columns and sector in h_new_sec.columns:
                    color = sector_color_map[sector]
                    sector_name = get_sector_name(sector, emission_type, species)
                    # Get 2023 value from old data as reference
                    ref_row = h_old_sec[h_old_sec['year'] == 2023]
                    if len(ref_row) > 0:
                        ref_value = ref_row[sector].values[0]
                        if ref_value != 0:
                            # Merge on year to compute difference
                            merged = pd.merge(h_old_sec[['year', sector]], h_new_sec[['year', sector]], 
                                              on='year', suffixes=('_old', '_new'))
                            merged['pct_diff'] = ((merged[f'{sector}_new'] - merged[f'{sector}_old']) / abs(ref_value)) * 100
                            ax.plot(merged['year'], merged['pct_diff'], '-', color=color, linewidth=1.5, label=sector_name)
        
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax.set_title(f'{species} - {emission_type} (H) - % Diff by Sector', fontsize=12, fontweight='bold')
        ax.set_xlabel('Year')
        ax.set_ylabel('% Difference (rel. to 2023 old)')
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        enforce_min_yrange(ax)
        
        # Plot VL scenario percentage difference
        ax = axes[1]
        if vl_old_sec is not None and vl_new_sec is not None:
            for sector in all_sectors:
                if sector in vl_old_sec.columns and sector in vl_new_sec.columns:
                    color = sector_color_map[sector]
                    sector_name = get_sector_name(sector, emission_type, species)
                    # Get 2023 value from old data as reference
                    ref_row = vl_old_sec[vl_old_sec['year'] == 2023]
                    if len(ref_row) > 0:
                        ref_value = ref_row[sector].values[0]
                        if ref_value != 0:
                            # Merge on year to compute difference
                            merged = pd.merge(vl_old_sec[['year', sector]], vl_new_sec[['year', sector]], 
                                              on='year', suffixes=('_old', '_new'))
                            merged['pct_diff'] = ((merged[f'{sector}_new'] - merged[f'{sector}_old']) / abs(ref_value)) * 100
                            ax.plot(merged['year'], merged['pct_diff'], '-', color=color, linewidth=1.5, label=sector_name)
        
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax.set_title(f'{species} - {emission_type} (VL) - % Diff by Sector', fontsize=12, fontweight='bold')
        ax.set_xlabel('Year')
        ax.set_ylabel('% Difference (rel. to 2023 old)')
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        enforce_min_yrange(ax)
        
        plt.tight_layout()
        fig.savefig(comparison_output_folder / f'C2_pct_diff_sectors_{species}_{emission_type}.png', dpi=100, bbox_inches='tight')
        plt.close(fig)

print("C2 percentage difference plots (by sector) complete.")
print(f"\nAll comparison plots saved to: {comparison_output_folder}")
# %%


# %%
# examples of loading (gridded) data

# old
VERSION_ESGF_OLD = '1-0-0'
PATH_RESULTS_OLD = Path('C:\\Users\\kikstra\\IIASA\\ECE.prog - Documents\\Projects\\CMIP7\\IAM Data Processing\\Shared emission fields data\\v1_0-testing-findmistakes\\rc4')
file_scenario_h_old = PATH_RESULTS_OLD / f"h_{VERSION_ESGF_OLD}" / "BC-em-anthro_input4MIPs_emissions_ScenarioMIP_IIASA-IAMC-h-1-0-0_gn_202201-210012.nc"
file_scenario_vl_old = PATH_RESULTS_OLD / f"vl_{VERSION_ESGF_OLD}" / "BC-em-anthro_input4MIPs_emissions_ScenarioMIP_IIASA-IAMC-vl-1-0-0_gn_202201-210012.nc"


# new
PATH_RESULTS = Path('C:\\Users\\kikstra\\IIASA\\ECE.prog - Documents\\Projects\\CMIP7\\IAM Data Processing\\Shared emission fields data\\v1_1-testing-findmistakes')
file_scenario_h = PATH_RESULTS / f"h_{VERSION_ESGF}" / "BC-em-anthro_input4MIPs_emissions_ScenarioMIP_IIASA-IAMC-h-1-1-0_gn_202201-210012.nc"
file_scenario_vl = PATH_RESULTS / f"vl_{VERSION_ESGF}" / "BC-em-anthro_input4MIPs_emissions_ScenarioMIP_IIASA-IAMC-vl-1-1-0_gn_202201-210012.nc"


# %% [markdown]
# ## Comparing two versions: C1) gridpoint totals per file (as percentage relative to 2023 values)

# %%
def get_nc_filepath(path_base, species, emission_type, marker, version):
    """Generate full path for a version's NC file."""
    filename = f"{species}-{emission_type}_input4MIPs_emissions_ScenarioMIP_IIASA-IAMC-{marker}-{version}_gn_202201-210012.nc"
    return path_base / f"{marker}_{version}" / filename

def get_var_name(species, emission_type):
    """Get xarray variable name from species and emission type.

    E.g. ('BC', 'em-anthro') -> 'BC_em_anthro'
         ('CO2', 'em-AIR-anthro') -> 'CO2_em_AIR_anthro'
    """
    return f"{species.replace('-', '_')}_{emission_type.replace('-', '_')}"

# Create output folder for gridpoint comparison plots
gridpoint_output_folder = output_folder / "version_comparison_gridpoint"
gridpoint_output_folder.mkdir(exist_ok=True)
print(f"Gridpoint comparison plots will be saved to: {gridpoint_output_folder}")

# %%
# C1 gridpoint: Histograms of gridpoint-level differences for totals (summed across sectors)
for species in SPECIES_LIST:
    for emission_type in EMISSION_TYPES:
        print(f"Processing gridpoint histogram (totals) for {species} - {emission_type}...")
        var_name = get_var_name(species, emission_type)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        has_data = False

        for idx, (marker, marker_label, color) in enumerate([('h', 'H', 'blue'), ('vl', 'VL', 'red')]):
            ax = axes[idx]

            fp_old = get_nc_filepath(PATH_RESULTS_OLD, species, emission_type, marker, VERSION_ESGF_OLD)
            fp_new = get_nc_filepath(PATH_RESULTS, species, emission_type, marker, VERSION_ESGF)

            if not fp_old.exists() or not fp_new.exists():
                ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center', va='center', fontsize=14)
                ax.set_title(f'{species} - {emission_type} ({marker_label})')
                continue

            has_data = True

            # Open datasets lazily with dask
            ds_old = xr.open_dataset(str(fp_old), chunks={"time": 12})
            ds_new = xr.open_dataset(str(fp_new), chunks={"time": 12})
            da_old = ds_old[var_name]
            da_new = ds_new[var_name]

            # Sum across all non-spatial/temporal dims (sector, level, etc.) for totals
            extra_dims = [d for d in da_old.dims if d not in ('time', 'lat', 'lon')]
            if extra_dims:
                da_old_total = da_old.sum(extra_dims)
                da_new_total = da_new.sum(extra_dims)
            else:
                da_old_total = da_old
                da_new_total = da_new

            # Reference: average absolute gridpoint value in 2023 (old)
            old_2023 = da_old_total.sel(time=da_old_total.time.dt.year == 2023).mean('time')
            ref_value = float(abs(old_2023).mean().compute())

            if ref_value == 0:
                ax.text(0.5, 0.5, 'Zero reference\n(2023 avg = 0)', transform=ax.transAxes,
                        ha='center', va='center', fontsize=12)
                ax.set_title(f'{species} - {emission_type} ({marker_label})')
                ds_old.close()
                ds_new.close()
                continue

            # Compute annual mean per gridpoint for all years
            old_annual = da_old_total.groupby('time.year').mean('time')
            new_annual = da_new_total.groupby('time.year').mean('time')

            # Compute difference and normalize by 2023 average gridpoint value
            diff = new_annual - old_annual
            pct_diff = (diff / ref_value) * 100

            # Compute and flatten
            with ProgressBar():
                all_values = pct_diff.compute().values.flatten()
            all_values = all_values[~np.isnan(all_values)]

            # Plot histogram
            if len(all_values) > 0:
                # Use central 99% for bin range to avoid extreme outlier distortion
                p_low, p_high = np.percentile(all_values, [0.5, 99.5])
                bin_range = (p_low, p_high) if p_low != p_high else None
                ax.hist(all_values, bins=100, range=bin_range, color=color, alpha=0.7, edgecolor='none')
                ax.axvline(x=0, color='black', linestyle='--', linewidth=1)

                # Summary statistics
                ax.text(0.02, 0.98,
                        f'Mean: {np.mean(all_values):.4f}%\n'
                        f'Median: {np.median(all_values):.4f}%\n'
                        f'Std: {np.std(all_values):.4f}%\n'
                        f'N: {len(all_values):,}',
                        transform=ax.transAxes, va='top', fontsize=9,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            ax.set_title(f'{species} - {emission_type} ({marker_label})', fontsize=12, fontweight='bold')
            ax.set_xlabel('% Difference (rel. to avg gridpoint 2023)')
            ax.set_ylabel('Count (gridpoints × years)')
            ax.grid(True, alpha=0.3)

            ds_old.close()
            ds_new.close()

        if not has_data:
            plt.close(fig)
            print(f"  Skipping {species} - {emission_type}: no data found")
            continue

        plt.tight_layout()
        fig.savefig(gridpoint_output_folder / f'D1_gridpoint_hist_total_{species}_{emission_type}.png',
                    dpi=100, bbox_inches='tight')
        plt.close(fig)

print("C1 gridpoint histogram plots (totals) complete.")

# %% [markdown]
# ## Comparing two versions: C2) gridpoint totals per sector (as percentage relative to 2023 values)

# %%
# C2 gridpoint: Histograms of gridpoint-level differences per sector (faceted)
for species in SPECIES_LIST:
    for emission_type in EMISSION_TYPES:
        print(f"Processing gridpoint histogram (sectors) for {species} - {emission_type}...")
        var_name = get_var_name(species, emission_type)

        for marker, marker_label, color in [('h', 'H', 'blue'), ('vl', 'VL', 'red')]:
            fp_old = get_nc_filepath(PATH_RESULTS_OLD, species, emission_type, marker, VERSION_ESGF_OLD)
            fp_new = get_nc_filepath(PATH_RESULTS, species, emission_type, marker, VERSION_ESGF)

            if not fp_old.exists() or not fp_new.exists():
                print(f"  Skipping {species} - {emission_type} ({marker_label}): file(s) not found")
                continue

            ds_old = xr.open_dataset(str(fp_old), chunks={"time": 12})
            ds_new = xr.open_dataset(str(fp_new), chunks={"time": 12})
            da_old = ds_old[var_name]
            da_new = ds_new[var_name]

            if 'sector' not in da_old.dims:
                print(f"  Skipping {species} - {emission_type} ({marker_label}): no sector dimension")
                ds_old.close()
                ds_new.close()
                continue

            sectors = da_old.sector.values
            n_sectors = len(sectors)
            ncols = min(4, n_sectors)
            nrows = int(np.ceil(n_sectors / ncols))

            fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
            if nrows == 1 and ncols == 1:
                axes = np.array([axes])
            axes_flat = np.array(axes).flatten()

            for si, sector_idx in enumerate(sectors):
                ax = axes_flat[si]
                sector_name = get_sector_name(str(float(sector_idx)), emission_type, species)

                old_sec = da_old.sel(sector=sector_idx)
                new_sec = da_new.sel(sector=sector_idx)

                # Sum over any extra dims (e.g., level) except time, lat, lon
                extra = [d for d in old_sec.dims if d not in ('time', 'lat', 'lon')]
                if extra:
                    old_sec = old_sec.sum(extra)
                    new_sec = new_sec.sum(extra)

                # Reference: avg absolute gridpoint value in 2023 (old)
                old_2023 = old_sec.sel(time=old_sec.time.dt.year == 2023).mean('time')
                ref = float(abs(old_2023).mean().compute())

                if ref == 0:
                    ax.text(0.5, 0.5, 'Zero reference', transform=ax.transAxes,
                            ha='center', va='center', fontsize=10)
                    ax.set_title(f'{sector_name}', fontsize=10, fontweight='bold')
                    continue

                # Annual means
                old_ann = old_sec.groupby('time.year').mean('time')
                new_ann = new_sec.groupby('time.year').mean('time')

                diff = new_ann - old_ann
                pct = (diff / ref) * 100

                with ProgressBar():
                    vals = pct.compute().values.flatten()
                vals = vals[~np.isnan(vals)]

                if len(vals) > 0:
                    p_low, p_high = np.percentile(vals, [0.5, 99.5])
                    bin_range = (p_low, p_high) if p_low != p_high else None
                    ax.hist(vals, bins=100, range=bin_range, color=color, alpha=0.7, edgecolor='none')
                    ax.axvline(x=0, color='black', linestyle='--', linewidth=1)

                    ax.text(0.02, 0.98,
                            f'Mean: {np.mean(vals):.4f}%\n'
                            f'Std: {np.std(vals):.4f}%',
                            transform=ax.transAxes, va='top', fontsize=8,
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

                ax.set_title(f'{sector_name}', fontsize=10, fontweight='bold')
                ax.set_xlabel('% Diff (rel. 2023 avg)')
                ax.set_ylabel('Count')
                ax.grid(True, alpha=0.3)

            # Hide empty axes
            for si in range(n_sectors, len(axes_flat)):
                axes_flat[si].set_visible(False)

            fig.suptitle(f'{species} - {emission_type} ({marker_label}) - Gridpoint % Diff by Sector',
                         fontsize=14, fontweight='bold')
            plt.tight_layout()
            fig.savefig(gridpoint_output_folder / f'D2_gridpoint_hist_sectors_{species}_{emission_type}_{marker_label}.png',
                        dpi=100, bbox_inches='tight')
            plt.close(fig)

            ds_old.close()
            ds_new.close()

print("C2 gridpoint histogram plots (by sector) complete.")
print(f"\nAll gridpoint comparison plots saved to: {gridpoint_output_folder}")

# %% [markdown]
# ## Comparing two versions: E) Gridpoint difference CSVs (Mt/yr, new − old) and combined summary

# %%
# Create output folder for gridpoint difference CSVs
gridpoint_diff_folder = output_folder / "version_comparison_gridpoint_diffs"
gridpoint_diff_folder.mkdir(exist_ok=True)
print(f"Gridpoint diff CSVs will be saved to: {gridpoint_diff_folder}")

# %%
csv_paths: list[Path] = []

# for species in SPECIES_LIST:
for species in ['SO2']:
    # for emission_type in EMISSION_TYPES:
    for emission_type in ['em-anthro']:
        for marker, marker_label in [("h", "H"), ("vl", "VL")]:
            print(f"Computing diff CSV for {species} - {emission_type} ({marker_label})...")

            fp_old = get_nc_filepath(PATH_RESULTS_OLD, species, emission_type, marker, VERSION_ESGF_OLD)
            fp_new = get_nc_filepath(PATH_RESULTS, species, emission_type, marker, VERSION_ESGF)

            if not fp_old.exists() or not fp_new.exists():
                print("  Skipping: file(s) not found")
                continue

            var_name = get_var_name(species, emission_type)

            # Open datasets (cell_area loaded automatically inside helper)
            ds_old = xr.open_dataset(str(fp_old), chunks={"time": 12})
            ds_new = xr.open_dataset(str(fp_new), chunks={"time": 12})

            # --- Per-sector totals (Mt/yr) ---
            with ProgressBar():
                old_sectors = ds_to_annual_emissions_total(ds_old, var_name, keep_sectors=True)
                new_sectors = ds_to_annual_emissions_total(ds_new, var_name, keep_sectors=True)

            diff_sectors = new_sectors - old_sectors  # Mt/yr, new − old

            # Build a tidy DataFrame: one row per sector, columns = years
            rows = []
            if "sector" in diff_sectors.dims:
                for sec_idx in diff_sectors.sector.values:
                    sec_ts = diff_sectors.sel(sector=sec_idx)
                    sec_label = get_sector_name(str(float(sec_idx)), emission_type, species)
                    row: dict = {
                        "species": species,
                        "emission_type": emission_type,
                        "marker": marker,
                        "sector": int(float(sec_idx)),
                        "sector_name": sec_label,
                    }
                    for yr in sec_ts.year.values:
                        row[int(yr)] = float(sec_ts.sel(year=yr))
                    rows.append(row)

                # Add a "Total" row (sum across sectors)
                total_ts = diff_sectors.sum("sector")
                total_row: dict = {
                    "species": species,
                    "emission_type": emission_type,
                    "marker": marker,
                    "sector": "Total",
                    "sector_name": "Total",
                }
                for yr in total_ts.year.values:
                    total_row[int(yr)] = float(total_ts.sel(year=yr))
                rows.append(total_row)
            else:
                # No sector dimension — single timeseries
                total_row: dict = {
                    "species": species,
                    "emission_type": emission_type,
                    "marker": marker,
                    "sector": "Total",
                    "sector_name": "Total",
                }
                for yr in diff_sectors.year.values:
                    total_row[int(yr)] = float(diff_sectors.sel(year=yr))
                rows.append(total_row)

            df = pd.DataFrame(rows)

            # Save individual CSV
            csv_path = gridpoint_diff_folder / f"E1_diff_{species}_{emission_type}_{marker}.csv"
            df.to_csv(csv_path, index=False)
            csv_paths.append(csv_path)
            print(f"  Saved: {csv_path.name}")

            ds_old.close()
            ds_new.close()

print(f"\nIndividual diff CSVs complete. {len(csv_paths)} files written.")

# %%
# Combine all individual CSVs into one summary file
if csv_paths:
    combined = pd.concat([pd.read_csv(p) for p in csv_paths], ignore_index=True)
    combined_path = gridpoint_diff_folder / f"E1_combined_gridpoint_diffs_{VERSION_ESGF_OLD}_vs_{VERSION_ESGF}.csv"
    combined.to_csv(combined_path, index=False)
    print(f"Combined CSV saved to: {combined_path}")
    print(f"Shape: {combined.shape[0]} rows × {combined.shape[1]} columns")
else:
    print("No CSVs were produced — nothing to combine.")

# %% [markdown]
# ## Comparing two versions: F) Per-file per-sector min/max gridpoint values

# %%
# Scan every NC file (old & new, h & vl) and record min/max per sector.
# Files are opened one at a time to keep memory use low.

minmax_rows: list[dict] = []

for species in SPECIES_LIST:
# for species in ['SO2']:
    for emission_type in EMISSION_TYPES:
    # for emission_type in ['em-openburning']:
        var_name = get_var_name(species, emission_type)

        for marker in ("h", "vl"):
            for version, path_base in [
                (VERSION_ESGF_OLD, PATH_RESULTS_OLD),
                (VERSION_ESGF, PATH_RESULTS),
            ]:
                fp = get_nc_filepath(path_base, species, emission_type, marker, version)
                if not fp.exists():
                    continue

                print(f"Min/max scan: {fp.name}")
                ds = xr.open_dataset(str(fp), chunks={"time": 12})
                da = ds[var_name]

                if "sector" in da.dims:
                    for sec_idx in da.sector.values:
                        sec_da = da.sel(sector=sec_idx)
                        sec_annual_min = sec_da.groupby("time.year").min()
                        sec_annual_max = sec_da.groupby("time.year").max()
                        with ProgressBar():
                            sec_annual_min = sec_annual_min.compute()
                            sec_annual_max = sec_annual_max.compute()
                        sec_name = get_sector_name(str(float(sec_idx)), emission_type, species)
                        for yr in sec_annual_min.year.values:
                            minmax_rows.append({
                                "filename": fp.name,
                                "species": species,
                                "emissions_type": emission_type,
                                "marker": marker,
                                "version": str(version),
                                "sector": sec_name,
                                "year": int(yr),
                                "min": float(sec_annual_min.sel(year=yr).min()),
                                "max": float(sec_annual_max.sel(year=yr).max()),
                            })
                else:
                    # No sector dimension (e.g. some openburning files)
                    annual_min = da.groupby("time.year").min()
                    annual_max = da.groupby("time.year").max()
                    with ProgressBar():
                        annual_min = annual_min.compute()
                        annual_max = annual_max.compute()
                    for yr in annual_min.year.values:
                        minmax_rows.append({
                            "filename": fp.name,
                            "species": species,
                            "emissions_type": emission_type,
                            "marker": marker,
                            "version": str(version),
                            "sector": "Total",
                            "year": int(yr),
                            "min": float(annual_min.sel(year=yr).min()),
                            "max": float(annual_max.sel(year=yr).max()),
                        })

                ds.close()

df_minmax = pd.DataFrame(minmax_rows)
minmax_path = gridpoint_diff_folder / f"F1_minmax_per_file_sector_{VERSION_ESGF_OLD}_and_{VERSION_ESGF}.csv"
df_minmax.to_csv(minmax_path, index=False)
print(f"\nMin/max summary saved to: {minmax_path}")
print(f"Shape: {df_minmax.shape[0]} rows × {df_minmax.shape[1]} columns")
df_minmax
# %%
