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
# # Workflow for CMIP7 ScenarioMIP emissions harmonization 
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
marker_to_run: str = "H" # options: H, HL, M, ML, L, LN, VL

# What folder to save this run in
GRIDDING_VERSION: str | None = None
GRIDDING_VERSION: str | None = f"{VERSION_ESGF}_{marker_to_run}"


# Which parts to run
run_main: bool = True # skips downscaling and the saving out of data of the main workflow; can still run supplemental workflows with this set to False
run_main_gridding: bool = True # if false, we'll not run the main gridding workflow
SKIP_EXISTING_MAIN_WORKFLOW_FILES: bool = False # if True, it won't reproduce files already on your disk
run_spatial_harmonisation: bool = True # provides spatial harmonization with CEDS anthro in 2023 (requires having raw CEDS files locally)
run_anthro_timeseries_correction: bool = True
run_AIR_anthro_timeseries_correction: bool = True
run_openburning_timeseries_correction: bool = True

run_openburning_h2: bool = True # produced based on openburning_co

run_anthro_supplemental_voc: bool = True
run_openburning_supplemental_voc: bool = True

# run_anthro_supplemental_solidbiofuel: bool = False # not yet implemented, for the future

# main: files to produce (species, sector)
DO_GRIDDING_ONLY_FOR_THESE_SPECIES: list[str] | None = None # e.g. ["CO2", "SO2"]
DO_GRIDDING_ONLY_FOR_THESE_SECTORS: list[str] | None = None # all: ['anthro', 'openburning', 'AIR_anthro']
# supplemental: VOC files to produce
# - anthro
DO_VOC_SPECIATION_ANTHRO_ONLY_FOR_THESE_SPECIES: list[str] | None = None # e.g. ["VOC01_alcohols_em_speciated_VOC_anthro"]
# - openburning
DO_VOC_SPECIATION_OPENBURNING_ONLY_FOR_THESE_SPECIES: list[str] | None = None # e.g. ["C10H16"]
# %%
# validate that we're receiving what we're expecting
print(f"\n\nGRIDDING_VERSION received: {GRIDDING_VERSION}\n\n")
print(f"\n\nDO_GRIDDING_ONLY_FOR_THESE_SPECIES received: {DO_GRIDDING_ONLY_FOR_THESE_SPECIES}\n\n")

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
from concordia.cmip7.utils import calculate_ratio, return_nc_output_files_main_voc, SECTOR_ORDERING_GAS, SECTOR_ORDERING_DEFAULT, SECTOR_DICT_ANTHRO_DEFAULT, SECTOR_DICT_ANTHRO_CO2_SCENARIO, reorder_dimensions, add_file_global_sum_totals_attrs, SECTOR_DICT_OPENBURNING_DEFAULT, SECTOR_DICT_OPENBURNING_DEFAULT_FLIPPED, SECTOR_DICT_ANTHRO_CO2_SCENARIO_FLIPPED
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


# %% editable=true slideshow={"slide_type": ""} tags=["parameters"]
# location for reading in raw historical (anthropogenic) emissions data for spatial harmonization
ceds_data_location: Path | None = None # if None here, it is replaced by settings.ceds_data_location later on
ceds_data_location_voc: Path | None = None # if None here, it is replaced by settings.ceds_data_location_voc later on
ceds_data_location_AIR: Path | None = None # if None here, it is replaced by settings.ceds_data_location_voc later on

# %%
if ceds_data_location is None:
    ceds_data_location = settings.postprocess_path / "CMIP7_anthro"
if ceds_data_location_voc is None:
    ceds_data_location_voc = settings.postprocess_path / "CMIP7_anthro_VOC"
if ceds_data_location_AIR is None:
    ceds_data_location_AIR = settings.postprocess_path / "CMIP7_AIR"

# %% [markdown]
# Set logger (uses setting)

# %%
fh = logging.FileHandler(settings.out_path / f"debug_{settings.version}.log", mode="w")
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
fh.setFormatter(formatter)


streamhandler = logging.StreamHandler()
streamhandler.setFormatter(
    MultiLineFormatter(
        "%(log_color)s%(levelname)-8s%(reset)s %(message)s  (%(blue)s%(name)s%(reset)s)",
        datefmt=None,
        reset=True,
    )
)

logger().handlers = [streamhandler, fh]
logging.getLogger("flox").setLevel("WARNING")

# %% [markdown]
# Create output path for this version

# %%
version_path = settings.out_path / settings.version
version_path.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# # Read definitions

# %% [markdown]
# ## Read variable definitions
#
# The variable definition file is a CSV or yaml file that needs to contain the `variable`-name, its `sector`, `gas` components and whether it is expected `global` (or regional instead).
#
# Here we use a file based on the RESCUE variable definitions, but adapted to fit CMIP7 purposes.
#

# %%
settings.variabledefs_path

# %%
variabledefs = VariableDefinitions.from_csv(settings.variabledefs_path)
# variabledefs.data.head()

# %%
# If only for a few species
if DO_GRIDDING_ONLY_FOR_THESE_SPECIES is not None:
    # filter only the species that we would like to run here
    print(f"Filtering variable definitions to only include species: {DO_GRIDDING_ONLY_FOR_THESE_SPECIES}")
    original_count = len(variabledefs.data)
    
    # Filter the data to keep only rows where gas is in DO_GRIDDING_ONLY_FOR_THESE_SPECIES
    filtered_data = variabledefs.data.loc[
        isin(gas=DO_GRIDDING_ONLY_FOR_THESE_SPECIES)
    ]
    
    # Create a new VariableDefinitions object with the filtered data
    variabledefs = VariableDefinitions(data=filtered_data)
    
    filtered_count = len(variabledefs.data)
    print(f"Filtered from {original_count} to {filtered_count} variable definitions")
    print(f"Unique gases in filtered data: {sorted(variabledefs.data.index.get_level_values('gas').unique())}")
else:
    print("Using all species from variable definitions")

# %%
# If only for a certain sector (filter by output_variable endings; what comes after '_em_')
if DO_GRIDDING_ONLY_FOR_THESE_SECTORS is not None:
    # filter only the sectors that we would like to run here
    print(f"Filtering variable definitions to only include sectors: {DO_GRIDDING_ONLY_FOR_THESE_SECTORS}")
    original_count = len(variabledefs.data)
    
    # Filter the data to keep only rows where output_variable has one of the specified sectors after "_em_"
    # e.g., 'SO2_em_anthro' matches 'anthro', 'SO2_em_openburning' matches 'openburning'
    sector_patterns = [f"_em_{sector}(?:$|_)" for sector in DO_GRIDDING_ONLY_FOR_THESE_SECTORS]
    pattern = "|".join(sector_patterns)
    
    filtered_data = variabledefs.data.loc[
        variabledefs.data['output_variable'].str.contains(pattern, regex=True, na=False)
    ]
    
    # Create a new VariableDefinitions object with the filtered data
    variabledefs = VariableDefinitions(data=filtered_data)
    
    filtered_count = len(variabledefs.data)
    print(f"Filtered from {original_count} to {filtered_count} variable definitions")
    print(f"Sectors in filtered data: {sorted(set(s.split('_em_')[1].split('_')[0] for s in variabledefs.data['output_variable'] if '_em_' in s))}")
else:
    print("Using all sectors from variable definitions")

# %% [markdown]
# ## Read region definitions (using RegionMapping class)
#

# %%
regionmappings = {}

for m, kwargs in settings.regionmappings.items():
    regionmapping = RegionMapping.from_regiondef(**kwargs)
    regionmapping.data = regionmapping.data.pix.aggregate(
        country=settings.country_combinations, agg_func="last"
    )
    regionmappings[m] = regionmapping


# %% [markdown]
# # History: Read and process historical data
#

# %%
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
])

# %% [markdown]
# # IAM: Read and process IAM data

# %%
Path(settings.scenario_path, SCENARIO_FILE)

# %% [markdown]
# ### Read in (currently just 1 scenario)

# %%
# Read in already-harmonized data
iam_df = cmip7_utils.load_data(
    Path(settings.scenario_path, SCENARIO_FILE)
)

# filter only one scenario  
iam_df = cmip7_utils.filter_scenario(iam_df, scenarios=SCENARIO_SELECTION) # TODO: remove this after test code is done

# %%
# TODO: add historical data as 2022 (for bb4cmip; for CEDS we should drop it again)

# %%
IAMC_COLS = ["model", "scenario", "region", "variable", "unit"]
HARMONIZED_YEAR_COLS = [col for col in iam_df.columns if col.isdigit() and settings.base_year <= int(col) <= 2100]

# %%
# keep only relevant columns
iam_df = iam_df[(IAMC_COLS + HARMONIZED_YEAR_COLS)]

# %%
iam_df

# %% [markdown]
# ### Process (using pix - formatting)

# %%
# split the 'variable' column into the 'gas' and 'sector' columns
iam_df = extractlevel(iam_df.set_index(IAMC_COLS), variable="Emissions|{gas}|{sector}", drop=True)

# Reorder the MultiIndex of iam_df
iam_df = iam_df.reorder_levels(['model', 'scenario', 'region', 'gas', 'sector', 'unit'])
iam_df = iam_df.sort_index()

# Update column type and name
iam_df.columns = iam_df.columns.astype(int)
iam_df.columns.name = 'year'
iam_df = iam_df.dropna(axis=1)


# %% [markdown]
# ## Save the processed IAM data

# %% [markdown]
# ### Basic checks

# %%
cmip7_utils.check_na_in_columns(iam_df)

# %% [markdown]
# ### Save in wide format

# %%
cmip7_utils.save_data(df = iam_df.reset_index(), 
                      output_path = str(Path(version_path, "scenarios_processed.csv" )))

# %% [markdown]
# # Read Harmonization Overrides

# %% [markdown]
# NOTE: should be handled already before, as the emissions trajectories have already been harmonised

# %%
settings.scenario_path

# %%
harm_overrides = pd.read_excel(
    settings.scenario_path / "harmonization_overrides.xlsx", # placeholder for now, empty now as already harmonized.
    index_col=list(range(3)),
).method
harm_overrides

# test that this is indeed empty, which is expected
assert harm_overrides.empty

# %%
# No need to reharmonise, so no rechoosing methods.
# But if one must, it can be done using the extend_overrides() function like so:
# harm_overrides = extend_overrides(
#     harm_overrides,
#     "constant_ratio",
#     sector=[
#         f"{sec} Burning"
#         for sec in ["Agricultural Waste", "Forest", "Grassland", "Peat"]
#     ],
#     variables=variabledefs.data.index,
#     regionmappings=regionmappings,
#     model_baseyear=iam_df[settings.base_year],
# )

# %% [markdown]
# # Prepare GDP proxy
#
# Read in different GDP scenarios for SSP1 to SSP5 from SSP DB.
#

# %%
settings.scenario_path

# %%
# New; updated SSP data from CMIP7 era (downloaded from: http://files.ece.iiasa.ac.at/ssp/downloads/ssp_basic_drivers_release_3.2.beta_full.xlsx, and then selected only the GDP|PPP variable)
gdp_new = pd.read_csv(
        settings.scenario_path / "ssp_basic_drivers_release_3.2.beta_full_gdp.csv",
        index_col=list(range(5)),
    )
# get iso3c instead of country names
country_name_to_iso = {country.name: country.alpha_3 for country in pycountry.countries}
index_df = gdp_new.index.to_frame()
index_df['Region'] = index_df['Region'].apply(lambda x: country_name_to_iso.get(x, x).lower())
gdp_new.index = pd.MultiIndex.from_frame(index_df)

gdp = (
    gdp_new
    .rename_axis(index=str.lower)
    .loc[
        isin(
            # model="OECD Env-Growth",
            scenario=[f"SSP{n+1}" for n in range(5)],
            variable="GDP|PPP",
            unit="billion USD_2017/yr"
        )
    ]
    # .dropna(how="all", axis=1)
    .rename_axis(index={"scenario": "ssp", "region": "country"})
    .loc[
        ~ismatch(country = "**(r**" ) # filter out region names like 'africa (r10)'
    ]
    .rename(index=str.lower, level="country")
    .rename(columns=int)
    .pix.project(["ssp", "country"])
    .pix.aggregate(country=settings.country_combinations)
)

# SELECT ONLY FUTURE YEARS
# Get all years in the range (min to max)
# all_years = range(min(gdp.columns), max(gdp.columns) + 1) # is possible, but would need to deal with "Scenario=='Historical Reference'"
all_years = range(2021, 2100 + 1)

# Reindex to include all years, then interpolate
gdp = gdp.reindex(columns=all_years)



# ADD HISTORICAL VALUE FOR EACH SCENARIO (to get 2023 and 2024 GDP)
## get 2020 historical value for interpolation to 2023 and 2024
gdp_new = gdp_new.rename_axis(index=str.lower)
hist_mask = (
    # gdp_new.index.get_level_values("model") == "OECD Env-Growth" &
    (gdp_new.index.get_level_values("scenario") == "Historical Reference") &
    (gdp_new.index.get_level_values("variable") == "GDP|PPP") &
    (gdp_new.index.get_level_values("unit") == "billion USD_2017/yr")
)
gdp_hist = (
    gdp_new.loc[hist_mask,"2020"]
    # gdp_new.loc[hist_mask,"2020"].reset_index().loc[:, ["region", "variable", "unit", "2020"]]
    .rename_axis(index={"region": "country"})
    .loc[
        ~ismatch(country = "**(r**" ) # filter out region names like 'africa (r10)'
    ]
    .rename(index=str.lower, level="country")
    # .rename(columns=int)
    .pix.project(["country"])
    .pix.aggregate(country=settings.country_combinations)
)
gdp_hist

## Merge
### Step 1: Reset index on gdp_hist to prepare for merge
gdp_hist_reset = gdp_hist.reset_index()  # 'country' becomes a column
gdp_hist_reset

### Step 2: Merge on 'country', keeping all rows in gdp
# We assume gdp has a MultiIndex ('ssp', 'country')
gdp_reset = gdp.reset_index()  # so we can merge on 'country'
merged = gdp_reset.merge(gdp_hist_reset[['country', '2020']], on='country', how='left')

### Step 3: Set index back to ('ssp', 'country') if needed
merged = merged.set_index(['ssp', 'country'])

### Step 4: Reorder columns by year
# First, get all columns that are years (int or str), sort them
year_cols = sorted([col for col in merged.columns if str(col).isdigit()], key=int)
merged = merged[year_cols]  # reordering only the year columns
# merged



# INTERPOLATE:
# Interpolate GDP DataFrame to annual data (fill all years in the column range)
# Assumes 'gdp' is a DataFrame with years as columns (integers) and a MultiIndex

# Reindex to include all years, then interpolate
gdp = gdp.interpolate(axis=1, method='linear', limit_direction='both')

gdp
# merged


# # Old; original SSP data from CMIP6 era
# gdp = (
#     pd.read_csv(
#         settings.scenario_path / "SspDb_country_data_2013-06-12.csv",
#         index_col=list(range(5)),
#     )
#     .rename_axis(index=str.lower)
#     .loc[
#         isin(
#             model="OECD Env-Growth",
#             scenario=[f"SSP{n+1}_v9_130325" for n in range(5)],
#             variable="GDP|PPP",
#         )
#     ]
#     .dropna(how="all", axis=1)
#     .rename_axis(index={"scenario": "ssp", "region": "country"})
#     .rename(index=str.lower, level="country")
#     .rename(columns=int)
#     .pix.project(["ssp", "country"])
#     # .pix.aggregate(country=settings.country_combinations)
# )
# gdp

# %% [markdown]
# Determine likely SSP for each harmonized pathway from scenario string and create proxy data aligned with pathways
#

# %%
# pycountry is used but does not recognise all country names from
# the gdp data, so we're manually renaming a few ourselves
# In the future we could possibly replace this by using 
# the counrty list from the `nomenclature-iamc` package that  
# was used to produce this data 
rename_gdp = {"bolivia": "bol", 
              "democratic republic of the congo": "cod",
              "iran": "irn",
              "laos": "lao",
              "micronesia": "fsm",
              "moldova": "mda",
              "kosovo": "srb (kosovo)",
              "palestine": "pse",
              "north korea": "prk",
              "south korea": "kor",
              "syria": "syr",
              "taiwan": "twn",
              "tanzania": "tza",
              "turkey": "tur",
              "united states virgin islands": "vir",
              "venezuela": "ven",
              "world": "World"
             }

hist = hist.pix.aggregate(country=settings.country_combinations)

gdp.index = gdp.index.set_levels(
    gdp.index.levels[gdp.index.names.index("country")].to_series().replace(rename_gdp),
    level="country"
)

gdp = gdp.pix.aggregate(country=settings.country_combinations)

# %%
SSP_per_pathway = cmip7_utils.guess_ssp(iam_df)
GDP_per_pathway = cmip7_utils.join_gdp_based_on_ssp(
    scenarios_with_ssp_mapping=SSP_per_pathway,
    gdp_per_ssp=gdp
)

# %% [markdown]
# # Country coverage

# %%
# try to align with CEDS; but where necessary, aggregate to SSP coverage.

# what countries do we have in each data set?
countries_with_gdp_data = gdp.pix.unique("country") # as Index
countries_with_hist_data = hist.pix.unique("country") # as Index
countries_with_regionmapping = pd.Index(sorted(
    regionmapping.filter(countries_with_gdp_data).data.reset_index().country.unique() # as array
)) # as Index
countries_with_hist_and_gdp_and_regionmapping_data = pd.Index(sorted(( 
    set(countries_with_gdp_data) & set(countries_with_hist_data) & set(countries_with_regionmapping) # as set
))) # as Index

# show what we have
print("Countries with GDP data (for downscaling):")
print(len(countries_with_gdp_data))
print("Countries with historical emissions data:")
print(len(countries_with_hist_data))
print("Countries in the IAM region mapping:")
print(len(countries_with_regionmapping))
print("Countries with data for all three above:")
print(countries_with_hist_and_gdp_and_regionmapping_data)

# def select_only_countries_with_all_info(df,
#                                         countries=countries_with_hist_and_gdp_and_regionmapping_data):
#     df = (
#         df
#         .loc[
#             isin(
#                 country=countries
#             )
#         ]
#     )
    
#     return df


# %% [markdown]
# # Sector coverage (check historical)

# %%
hist_sectors = hist.index.get_level_values("sector").unique()
iam_sectors = iam_df.index.get_level_values("sector").unique()

missing = set(iam_sectors) - set(hist_sectors)
print(f"Separately considering CDR sectors {missing}")  # CDR sectors

expected_sectors_missing_cdr = {
    'Enhanced Weathering', 'BECCS', 'Direct Air Capture', 'Ocean', 'Biochar', 'Soil Carbon Management', 'Other CDR'
}
assert missing.issubset(expected_sectors_missing_cdr), f"Unexpected missing sectors found: {missing - expected_sectors_missing_cdr}"

# %% [markdown]
# ## Add zero CDR history

# %%
co2_template = hist.loc[isin(sector="Energy Sector", gas="CO2")] # pull co2-like template
# fill values with zero
co2_template.loc[:] = 0

# add to history with replaced sector names
for s in expected_sectors_missing_cdr:
    hist = pd.concat([
        hist,
        co2_template.pix.assign(sector=s)
    ])


# %% [markdown]
# # Set up technical bits for the workflow
# **NOTE:** Conditional setup for different environments - uses threaded scheduler for VS Code interactive window

# %%
# Import Dask setup function from separate module
# Set up the client
client = setup_dask_client()

if client is not None:
    print(f"Dask client dashboard: {client.dashboard_link}")
else:
    print("Using Dask threaded/synchronous scheduler")


# %% [markdown]
# # Define workflow

# %%
# TODO (in the future): allow doing multiple models at once in a notebook --> right now the below section only works for 1 model at a time
(model_name,) = iam_df.pix.unique("model")
regionmapping = regionmappings[model_name]


# %%
# indexes for countries on a grid
indexraster = IndexRaster.from_netcdf(
    settings.gridding_path / "ssp_comb_indexraster.nc", # redo: notebooks\gridding_data\generate_ceds_proxy_netcdfs.py
    chunks={},
).compute()
indexraster_region = indexraster.dissolve(
    regionmapping.filter(indexraster.index).data.rename("country")
).compute()

# %%
iam_df.columns

# %%
# TODO: find a better way to test whether all historical data is available for each country (it is known that Guam and Mayotte miss data for some sectors in the used CEDS release)
# # check completeness of historical data
# for c in countries_with_hist_and_gdp_and_regionmapping_data:
#     if len(hist.loc[ismatch(country=c)]) < 120:
#         print(c)

# %%
# do variable name replacements to align with CEDS and BB4CMIP7 historical products

# Sulfur -> SO2 (CEDS+BB4CMIP7)
iam_df = iam_df.rename(index=lambda v: v.replace("Sulfur", "SO2"))
hist = hist.rename(index=lambda v: v.replace("Sulfur", "SO2"))

# VOC -> NMVOC for anthro sectors (CEDS)
openburning_sectors = cmip7_utils.SECTOR_ORDERING_DEFAULT['em_openburning']
iam_df = iam_df.rename(index=lambda v: v.replace("VOC", "NMVOC"))
hist = hist.rename(index=lambda v: v.replace("VOC", "NMVOC"))
# Rename NMVOC to NMVOCbulk in iam_df for openburning sectors (BB4CMIP7)
def rename_voc_to_nmvoc_iam(idx):
    gas_idx = iam_df.index.names.index("gas")
    sector_idx = iam_df.index.names.index("sector")
    if idx[sector_idx] in openburning_sectors and idx[gas_idx] == "NMVOC":
        idx_list = list(idx)
        idx_list[gas_idx] = "NMVOCbulk"
        return tuple(idx_list)
    return idx

iam_df.index = iam_df.index.map(rename_voc_to_nmvoc_iam)

# Rename NMVOC to NMVOCbulk in hist for openburning sectors
def rename_voc_to_nmvoc_hist(idx):
    gas_idx = hist.index.names.index("gas")
    sector_idx = hist.index.names.index("sector")
    if idx[sector_idx] in openburning_sectors and idx[gas_idx] == "NMVOC":
        idx_list = list(idx)
        idx_list[gas_idx] = "NMVOCbulk"
        return tuple(idx_list)
    return idx

hist.index = hist.index.map(rename_voc_to_nmvoc_hist) 

# %%
workflow = WorkflowDriver(
    # model
    # iam_df.loc[:, iam_df.columns.intersection(GDP_per_pathway.columns.tolist())], # model ; until GDP is interpolated, do only for years in GDP_per_pathway.columns.tolist()
    iam_df,
    # hist
    hist, # select_only_countries_with_all_info(hist),
    # gdp
    GDP_per_pathway, #select_only_countries_with_all_info(GDP_per_pathway),
    # regionmapping
    regionmapping.filter(countries_with_hist_and_gdp_and_regionmapping_data[~countries_with_hist_and_gdp_and_regionmapping_data.isin(['myt','gum'])]), # mayotte and guam are missing some historical data for some sectors
    # indexraster_country
    indexraster,
    # indexraster_region
    indexraster_region,
    # variabledefs
    variabledefs, # NOTE: for Sulfur/SO2 and NMVOC/VOC, there is consistent renaming happening in 
    # harm_overrides
    harm_overrides,
    # settings
    settings
)

# %% [markdown]
# ## Add some checks on workflow

# %%
# save workflow info in easy-to-vet packets
if run_main:
    workflow.save_info(path = Path(version_path, "workflow_driver_data"),
                       prefix=settings.version)

# %%
# check regionmapping and scenarios
reg_model = iam_df.loc[~isin(region="World")].reset_index().region.unique() # all region names of the scenario
reg_mapped = regionmapping.data.reset_index().region.unique() # all region names of the scenario

def assert_strings_covered(array1, array2):
    assert all(s in array2 for s in array1), "Not all regions are covered in the regionmapping"
assert_strings_covered(reg_model, reg_mapped)

# %% [markdown]
# # Harmonize, downscale and grid everything
#

# %% [markdown]
# ## Harmonize and downscale everything, but do not grid
#
# If you also want grids, use the gridding interface directly - see below.
#

# %%
# manual steps:
# ...
# skipnone(
#                 self.harmdown_globallevel(variabledefs),
#                 self.harmdown_regionlevel(variabledefs),
#                 self.harmdown_countrylevel(variabledefs),
#             )

# workflow.harmdown_globallevel(workflow.variabledefs) # first step, works fine
# workflow.harmdown_regionlevel(workflow.variabledefs) # second step, looks to work fine and quick, too
# workflow.harmdown_countrylevel(workflow.variabledefs) # third step, requires gdp to be available from the HARMONIZATION_YEAR onward

# %%
# hot-patch to deal with proxies that have NA values
# see src/concordia/_patches_ptolemy.py


# %%
if run_main:
    downscaled = workflow.harmonize_and_downscale() # For a 1 scenario, this takes about 50 seconds on Jarmo's DELL laptop.

# %% [markdown]
# ### Export harmonized scenarios

# %%
if run_main:
    print("Outputs will be placed in " + str(version_path.resolve()))
    data = (
        workflow.harmonized_data.add_totals()
        .to_iamc(settings.variable_template, hist_scenario="Synthetic (CEDS/GFED/Global)")
        # .pipe(rename_alkalinity_addition)
        .rename_axis(index=str.capitalize)
    )
    print("File: " + f"harmonization-{settings.version}.csv")
    data.to_csv(version_path / f"harmonization-{settings.version}.csv")

# %% [markdown]
# ### Export downscaled scenarios

# %%
if run_main:
    workflow.downscaled.data.to_csv(
        version_path / f"downscaled-only-{settings.version}.csv"
    )
    print(
        "Countries covered (" + str(len(workflow.downscaled.data.loc[~isin(region="World")].reset_index().country.unique())) + "):"
    )
    print(workflow.downscaled.data.loc[~isin(region="World")].reset_index().country.unique())

# %%
if run_main:
    # Get unique countries from each dataframe
    # what countries do we have in each data set?
    countries_with_gdp_data = GDP_per_pathway.pix.unique("country") # as Index
    countries_with_hist_data = hist.pix.unique("country") # as Index

    gdp_countries = set(countries_with_gdp_data)
    hist_countries = set(countries_with_hist_data)

    # Countries in hist but not in GDP_per_pathway
    in_hist_not_gdp = hist_countries - gdp_countries
    print("Countries in hist but not in GDP_per_pathway:")
    print(sorted(in_hist_not_gdp))

    # Countries in GDP_per_pathway but not in hist
    in_gdp_not_hist = gdp_countries - hist_countries
    print("\nCountries in GDP_per_pathway but not in hist:")
    print(sorted(in_gdp_not_hist))

    downscaled_countries = set(workflow.downscaled.data.reset_index().country.unique())
    print("\nCountries in GDP but not in downscaled countries:")
    print(list(gdp_countries - downscaled_countries))

    # Display counts for reference
    print(f"\nCountries in hist: {len(hist_countries)}")
    print(f"Countries in GDP_per_pathway: {len(gdp_countries)}")
    print(f"Countries in common: {len(hist_countries & gdp_countries)}")
    print(f"Countries downscaled: {len(downscaled_countries)}")

# %%
if run_main:
    # Total missing data: countries in hist but not in downscaled
    in_hist_not_downscaled = hist_countries - downscaled_countries
    print("Countries in hist but not in downscaled:")
    print(sorted(in_hist_not_downscaled))

    missing_emissions = hist.loc[isin(country=list(in_hist_not_downscaled))].groupby(["gas","sector","unit"]).sum().loc[isin(sector='Waste'),2023]
    global_emissions = hist.loc[isin(country='World')].groupby(["gas","sector","unit"]).sum().loc[isin(sector='Waste'),2023]
    print("In %, what share of global emissions is missing because some smaller territories/countries are not downscaled?")
    print(missing_emissions / global_emissions * 100) # percentage (%) of global emissions that would be missing through these countries

# %% [markdown]
# # Run full processing and create netcdf files
#
# Latest test with 1 scenario was 50 minutes on Jarmo's DELL laptop.
# Output files are about 11.4GB for one scenario.

# %%
cmip7_utils.DS_ATTRS

# %%
if run_main_gridding: # full run for all 10 species takes about ~1hour for 1 scenario

    experiment_name = cmip7_utils.scenario_name_prefix(m=marker_to_run)

    res = workflow.grid(
        # keep {name} as a placeholder for workflow.grid (escaped as {{name}} here);
        # substitute scenario now using esgf_name computed earlier
        template_fn="{{name}}_{FILE_NAME_ENDING}".format(
            **(cmip7_utils.DS_ATTRS | {"version": VERSION_ESGF,
                                       "FILE_NAME_ENDING": FILE_NAME_ENDING})
        ),
        callback=cmip7_utils.DressUp(version=settings.version,
                                     marker_scenario_name=experiment_name),
        directory=version_path,
        skip_exists=SKIP_EXISTING_MAIN_WORKFLOW_FILES,
    )



# %% [markdown]
# # START OF POSTPROCESSING
# ## 1. Spatial Harmonization
# NOTE: runtime is about ~3 mins per file


# %% [markdown]
# # Start of Post-processing: pattern harmonisation

# %%
# helper functions for spatial harmonization
def copy_attributes(
    source: xr.Dataset, target: xr.Dataset
) -> xr.Dataset:
    """
    Copy attributes from source to target

    Parameters
    ----------
    source
        Source dataset from which to copy attributes

    target
        Target dataset to which to copy attributes

    Returns
    -------
    :
        `target` with updated attributes

        Note that the operation occurs in place,
        so the input `target` object is also affected
        (returning `target` is done for convenience)
    """
    target.attrs.update(source.attrs)
    return target

# helper functions for spatial harmonization
def copy_bounds_data_variables(
    source: xr.Dataset, target: xr.Dataset,
    bounds_vars = ['lat_bnds', 'lon_bnds', 'time_bnds', 'sector_bnds']
) -> xr.Dataset:
    """
    Copy bounds data variables from source to target

    Parameters
    ----------
    source
        Source dataset from which to copy bound variables

    target
        Target dataset to which to copy bound variables

    Returns
    -------
    :
        `target` with updated bound variables

        Note that the operation occurs in place,
        so the input `target` object is also affected
        (returning `target` is done for convenience)
    """
    
    # Copy the data variables ['lat_bnds', 'lon_bnds', 'time_bnds', 'sector_bnds']
    for var in bounds_vars:
        if var in source.data_vars: # as long as it is in the source dataset
            target[var] = source[var].load()

    return target

def _what_emissions_variable_type(file, files_main=[], files_voc=[]):
    if file in files_main:
        type = "em_anthro"
    elif file in files_voc:
        type = "em_speciated_VOC_anthro"
    return type

# %%
years = [year for year in PROXY_YEARS if year >= settings.base_year] # all years, but not 2022 (before 2023); which should come directly from CEDS anthro (and CEDS AIR)

# run the spatial harmonization (only em_anthro)
if run_spatial_harmonisation:
    print('run spatial harmonization')

    # files that are produced above, that may need correction
    files_main, files_voc = return_nc_output_files_main_voc(gridded_data_location=settings.out_path / GRIDDING_VERSION)


    # areas of gridcells for calculatings totals
    areacella = xr.open_dataset(Path(settings.gridding_path, "areacella_input4MIPs_emissions_CMIP_CEDS-CMIP-2025-04-18_gn.nc"))
    cell_area = areacella["areacella"]

    # for file in tqdm(files_main + files_voc, desc="Processing files"): # all
    for file in tqdm(files_main, desc="Processing files"): # only main
        gas = file.name.split("-")[0]
        type = _what_emissions_variable_type(file, files_main, files_voc)
        outfile = file

        print(f'Applying spatial corrections to: {gas} - do we need any?')

        # match reference file: check whether there's a raw CEDS history file on your system to harmonise against
        if file in files_main:
            match = next(ceds_data_location.glob(f"{gas}-*.nc"), None)
            if match is None:
                print(f"Warning: No CEDS file found for {gas} in {ceds_data_location}")
                continue
        if file in files_voc:
            match = next(ceds_data_location_voc.glob(f"{gas}-*.nc"), None)
            if match is None:
                print(f"Warning: No VOC CEDS file found for {gas} in {ceds_data_location_voc}")
                continue
        
        DONT_SPATIALLY_HARMONISE_SPECIATED_VOC = True # since the spatial harmonized is BEFORE VOC speciation in the workflow, and since the (NMVOC speciation) shares are applied directly on the gridcells, they are not affected by borders so shouldn't create new issues
        if DONT_SPATIALLY_HARMONISE_SPECIATED_VOC:
            # only run for MAIN CEDS (anthro) species; not VOC species; not any species we are not running in this specific run (if already run in a previous run for the same folder)
            
            # check that we're only doing the recently/currently run gas
            if DO_GRIDDING_ONLY_FOR_THESE_SPECIES is not None:
                # Only process species in the filtered list
                if gas not in DO_GRIDDING_ONLY_FOR_THESE_SPECIES:
                    print(f"Skipping {gas} (not in DO_GRIDDING_ONLY_FOR_THESE_SPECIES)")
                    continue
            else:
                # If no filter, only process main CEDS species (or VOC species if file is VOC)
                if file in files_main and gas not in GASES_ESGF_CEDS:
                    print(f"Skipping {gas} (not in GASES_ESGF_CEDS)")
                    continue

            # check that we're only doing the recently/currently run sector
            sector = type.split("_", 1)[1]
            if DO_GRIDDING_ONLY_FOR_THESE_SECTORS is not None:
                # Only process species in the filtered list
                if sector not in DO_GRIDDING_ONLY_FOR_THESE_SECTORS:
                    print(f"Skipping {sector} (not in DO_GRIDDING_ONLY_FOR_THESE_SECTORS)")
                    continue
        print(f'Applying spatial corrections to: {gas}_{type}')


        # step 1: find ratio grid between baseyear historical(CEDS) and baseyear scenario(gridded)
        # open datasets (no dask)
        ceds = xr.open_dataset(match)
        gridded = xr.open_dataset(file)

        # add
        if f"{gas}_{type}" == "CO2_em_anthro":
            ceds = xr.concat([ceds, xr.zeros_like(gridded.sel(time='2023',sector=[8,9]))], dim="sector")
        
        # try:
        # variable name
        if file in files_main:
            var = f"{gas}_{type}"
        if file in files_voc:
            var = find_voc_data_variable_string(gas)

        # rename sectors; from numbers to full names
        # DELETE: ceds = ceds # no need to remap sector values anymore as before; ceds.assign_coords(sector=pd.Series(ceds["sector"].values).map(SECTOR_DICT_ANTHRO_DEFAULT).values)
        reference = ceds.where(ceds.time.dt.year == 2023, drop=True)
        gridded_23 = gridded.where(gridded.time.dt.year == 2023, drop=True)

        # ZERO/NON-ZERO (absolute diffs): identify gridcells that are zero in `gridded``, but non-zero in `ceds`, and keep those gridcell values from `gridded`, while putting all other gridcell values to zero
        gridded_is_zero = (gridded_23[var] == 0) | gridded_23[var].isnull()
        ceds_is_nonzero = (reference[var] != 0) & reference[var].notnull()
        mask_to_replace = gridded_is_zero & ceds_is_nonzero # Create mask for cells to replace: gridded=0 AND ceds!=0
        additive_reference = reference[var].where(mask_to_replace, 0).to_dataset(name=var) # Create additive reference: keep CEDS values only where mask is True
        
        # Relative differences: calculate relative difference (vectorized)
        pct_diff23 = calculate_ratio(reference, gridded_23, gas)
        # TODO: Figure out why Paraguay Energy emissions in 2023 have crazy differnet relative values (~1e7), and are spread out over the country -- issue has to do with that emissions are super tiny; maybe we set them to zero? --- NEXT UP: look at total emissions for paraguay in gridded and ceds ...
        weights = pct_diff23.to_dataset(name=var) # all 1 if no adjustment needed, otherwise most gridpoints close to 1 but not exactly 1

        # expand weights to all years
        n_repeat = gridded.sizes["time"] // weights.sizes["time"]
        weights_exp = xr.concat([weights] * n_repeat, dim="time")
        weights_exp = weights_exp.assign_coords(time=gridded.time)
        # expand additive_reference (overseas territories) to all years
        additive_reference_exp = xr.concat([additive_reference] * n_repeat, dim="time")
        additive_reference_exp = additive_reference_exp.assign_coords(time=gridded.time)

        # apply weights (= raw ratios)
        weighted = gridded * weights_exp

        # replace sectors we don't want weighted; because it is not in CEDS
        sectors_to_keep = ['BECCS',
                           'Other Capture and Removal'] # Note: previously we also had International Shipping here, but in the case that there's a small (global) discrepancy between CEDS and scenario, it is worthwhile addressing that here too.
        sectors_present = [s for s in sectors_to_keep if s in weighted.sector.values]
        if sectors_present:
            weighted[var].loc[dict(sector=sectors_present)] = gridded[var].sel(sector=sectors_present)

        # step 2: calculate how much to the global total to make the adjustment perfect for the future (ensure same global emissions)
        # 2.1. multiply the two grids by cell_area to get (total) emissions -- instead of emissions per m2
        # 2.2. calculate the difference between the global total from our gridding, and the 'weighted' (=spatially adjusted) data; per sector, per year
        # 2.3. apply the scalar to the 'weighted' emissions; per sector, per year, to obtain the desired grid

        # 2.1:
        # calculate sectoral global totals
        total_emissions_gridded = cell_area * gridded.drop_vars(["lon_bnds", "lat_bnds", "time_bnds"])
        gridded_global = total_emissions_gridded[var].groupby("sector").sum(dim=("lat", "lon")).astype("float64")
        
        total_emissions_weighted = cell_area * weighted
        weighted_global = total_emissions_weighted[var].groupby("sector").sum(dim=("lat", "lon")).astype("float64")

        # 2.2:
        # 2.2.1. calculate the (base) scalar {what are the downscaled totals to scale with}
        global_scalar = xr.where(weighted_global != 0,
                                gridded_global / weighted_global,
                                0)
        

        # 2.2.2. calculate how far away from 1 each value is in 2023
        global_scalar_diff = xr.where(global_scalar.sel(time='2023') != 0,
                                1 - global_scalar.sel(time='2023'),
                                0)
        # 2.2.3. project that onto the time dimension
        global_scalar_diff_exp = xr.concat([global_scalar_diff] * n_repeat, dim="time")
        global_scalar_diff_exp = global_scalar_diff_exp.assign_coords(time=global_scalar.time)
        # 2.2.4. scale it back to zero in 2050, linearly, with zero 2050-2100, the full value in 2023, and linear interpolation based on for the period 2023-2050
        # Create a linear interpolation factor that goes from 1 in 2023 to 0 in 2050, then stays 0 through 2100
        years = global_scalar_diff_exp.time.dt.year.values
        interpolation_factor = np.where(
            years <= 2023,
            1.0,  # full correction in 2023 and before
            np.where(
                years <= 2050,
                (2050 - years) / (2050 - 2023),  # linear interpolation 2023-2050
                0.0  # zero correction from 2050 onward
            )
        )
        # Expand interpolation factor to match (time, sector) shape by repeating across sectors
        # interpolation_factor has shape (228,), need to expand to (228, 9)
        interpolation_factor_expanded = xr.DataArray(
            np.repeat(interpolation_factor[:, np.newaxis], len(global_scalar_diff_exp.sector), axis=1),
            coords={"time": global_scalar_diff_exp.time, "sector": global_scalar_diff_exp.sector},
            dims=["time", "sector"]
        )
        # Apply the interpolation factor to the difference
        global_scalar_diff_exp = global_scalar_diff_exp * interpolation_factor_expanded
        # Create the final global scalar correction: 1 + interpolated_difference (1 means no correction, >1 means scale up, <1 means scale down)
        global_scalar = global_scalar + global_scalar_diff_exp

        # 2.3:
        # apply the scalar
        emissions_harmonised = weighted * global_scalar
    
        # 2.4:
        # add small territories values from historical (don't scale down)
        # TODO:
        # - [ ] consider scaling this down over time too
        emissions_harmonised = emissions_harmonised + additive_reference_exp

        # 2.5:
        # do final minor addition, and scale to zero
        remainder_diff = reference - emissions_harmonised.sel(time='2023')
        # check that these differences are minor (only perform this correction if it is minor, otherwise there is some problem)
        remainder_diff_2023 = float(
                ds_to_annual_emissions_total( # takes about 10-30 seconds
                gridded_data=remainder_diff,
                var_name=var,
                cell_area=cell_area,
                keep_sectors=False
            )
            )
        assert remainder_diff_2023 < 50 # Mt / year
        # apply scaler (zero in 2050 and after)
        remainder_diff_exp = xr.concat([remainder_diff] * n_repeat, dim="time")
        remainder_diff_exp = remainder_diff_exp.assign_coords(time=global_scalar.time)
        remainder_diff_exp = remainder_diff_exp * interpolation_factor_expanded # Apply the interpolation factor to the difference
        # add
        emissions_harmonised = emissions_harmonised + remainder_diff_exp


        # 2.5:
        # final total global scalar harmonisation to the emissions

        # print how big the difference still is (should be zero, but isn't -- is that because the global scalar isn't 1 in 2023, perhaps? such that what has been scaled up in 2023 ratio-wise will now be pushed down again?)
        gridded2023 = float(
                ds_to_annual_emissions_total( # takes about 10-30 seconds
                gridded_data=emissions_harmonised.sel(time='2023'),
                var_name=var,
                cell_area=cell_area,
                keep_sectors=False
            )
            )
        ref2023 = float(
                ds_to_annual_emissions_total( # takes about 10-30 seconds
                gridded_data=reference,
                var_name=var,
                cell_area=cell_area,
                keep_sectors=False
            )
            )
        diff_mt = ref2023 - gridded2023
        diff_perc = (diff_mt / gridded2023) * 100
        print(
            f"{gas}_{type} Missing {diff_mt:.2f} Mt in 2023 ({diff_perc:.2f}%)"
        )

        # Load original gridded dataset to copy attributes
        copy_attributes(source=gridded,
                        target=emissions_harmonised)
        copy_bounds_data_variables(source=gridded,
                        target=emissions_harmonised)
        # finally:
        # Close datasets to release file locks
        if 'ceds' in locals():
            ceds.close()
        if 'gridded' in locals():
            gridded.close()
        if 'weighted' in locals():
            weighted.close()


        # remove old file (from previous loop in processing)
        outfile.unlink(missing_ok=True)
        # save weighted dataset (no dask)
        encoding = {var: {"zlib": True, "complevel": 2}}

        # reorder dimensions when no computations are required anymore; except replacing 2022
        emissions_harmonised = emissions_harmonised.pipe(reorder_dimensions)
                
        # replace 2022 CEDS data; anthro
        if type == "em_anthro":
            if gas == "CO2":
                ceds_2022 = xr.concat([xr.open_dataset(next(ceds_data_location.glob(f"{gas}-*.nc"))).sel(time='2022').pipe(reorder_dimensions, bound_var_name="bound"),
                                    xr.zeros_like(emissions_harmonised.sel(time='2022',sector=[8,9]))], dim="sector")
            else:
                ceds_2022 = xr.open_dataset(next(ceds_data_location.glob(f"{gas}-*.nc"))).sel(time='2022').pipe(reorder_dimensions, bound_var_name="bound")
        # replace values
        
        emissions_harmonised = emissions_harmonised.pipe(reorder_dimensions, bound_var_name="bound")

        emissions_harmonised[f"{gas}_{type}"].loc[dict(time=ceds_2022.time)] = ceds_2022[f"{gas}_{type}"]

        # Add global sums as metadata
        emissions_harmonised = emissions_harmonised.pipe(add_file_global_sum_totals_attrs, name=f"{gas}_{type}") # add totals after 2022 is added

        xr.testing.assert_allclose(ceds_2022[f"{gas}_{type}"], emissions_harmonised.sel(time='2022')[f"{gas}_{type}"], rtol=0, atol=0)
        #assert np.allclose(test_difference, 0)
        
        # Save out the updated file
        emissions_harmonised.to_netcdf(outfile, encoding=encoding)
        emissions_harmonised.close() # close the connection to the file

# %% 
# load files for timeseries corrections

# all years, but not 2022 (before 2023); which should come directly from CEDS anthro (and CEDS AIR)
years = [year for year in PROXY_YEARS if year >= settings.base_year]
# areas of gridcells for calculatings totals
areacella = xr.open_dataset(Path(settings.gridding_path, "areacella_input4MIPs_emissions_CMIP_CEDS-CMIP-2025-04-18_gn.nc"))
cell_area = areacella["areacella"]

# %%
# run the em_AIR_anthro timeseries correction (only em_AIR_anthro)
# NOTE: should take <1min per file

if run_AIR_anthro_timeseries_correction:

    # files that are produced above, that may need correction
    files = [
        file
        for file in (settings.out_path / GRIDDING_VERSION).glob("*.nc")
        if "AIR" in file.name
    ]

    for file in files:
        gas_name, var, type_name = return_emission_names(file)
        print(f'run em_AIR_anthro timeseries correction for {gas_name}')

        # Open dataset with explicit engine settings to avoid caching issues
        scen_ds = xr.open_dataset(file, engine='netcdf4')
        gridded_emisssions_annual_totals = ds_to_annual_emissions_total( # takes about 10-30 seconds
                gridded_data=scen_ds,
                var_name=var,
                cell_area=cell_area,
                keep_sectors=True
            )
        
        # Get input IAM emissions (already harmonised)
        input_emissions = (
            iam_df.loc[ # assumes that this is already harmonised
                ismatch(gas=gas_name)
            ].loc[
                isin(sector="Aircraft")
            ]
        )
        
        # Note: if N2O, divide by 1000 (from kt to Mt)
        if gas_name == "N2O":
            input_emissions = input_emissions / 1000

        # Calculate annual global totals by sector
        # Step 1: Ensure only one model-scenario combination
        unique_model = input_emissions.index.get_level_values('model').unique()
        unique_scenario = input_emissions.index.get_level_values('scenario').unique()
        assert len(unique_model) == 1, f"Expected 1 model, got {len(unique_model)}: {unique_model.tolist()}"
        assert len(unique_scenario) == 1, f"Expected 1 scenario, got {len(unique_scenario)}: {unique_scenario.tolist()}"
        
        # Step 2: Sum across regions, keeping sector and year information
        # Group by gas, sector, unit and sum (this sums across all regions)
        input_global = input_emissions.groupby(level=['gas', 'sector', 'unit']).sum()
        
        # Convert to DataFrame format with years as columns for easier viewing
        input_global_by_sector = input_global.reset_index()
        # Replace sector integer indices with full sector names
        input_global_by_sector['sector'] = input_global_by_sector['sector'].map({"Aircraft":0})
        
        # Convert to match gridded_emisssions_annual_totals format
        # Pivot so we have (gas, sector, unit) as index and years as columns
        input_global_by_sector = input_global_by_sector.set_index(['gas', 'sector', 'unit'])
        
        # Sort the index for consistency
        input_global_by_sector = input_global_by_sector.sort_index()
        
        # Transform from pandas DataFrame to xarray DataArray format
        # Extract sector values (although not necessary for aircraft)
        sectors = input_global_by_sector.index.get_level_values('sector').unique()
        
        # Extract year columns (all columns that are integers)
        year_columns = sorted([int(col) for col in input_global_by_sector.columns if isinstance(col, int)])
        
        # Create 2D numpy array: rows = sectors, columns = years
        data_array = np.array([
            input_global_by_sector.loc[(gas_name, sector, input_global_by_sector.index.get_level_values('unit')[0]), year_columns].values
            for sector in sectors
        ]).T  # Transpose so years are rows, sectors are columns
        
        # Create xarray DataArray matching gridded_emisssions_annual_totals structure
        input_iam_annual_totals = xr.DataArray(
            data_array[:,0],
            coords={
                'year': year_columns
            },
            dims=['year'],
            name=f'{gas_name}_em_AIR_anthro'
        )
        
        # Calculate ratios (for each sector, for one emissions species)
        # Compare input IAM vs gridded emissions for each sector and year
        # Handle division by zero where gridded is zero
        ratio_per_sector = xr.where(
            gridded_emisssions_annual_totals.sel(year=list(y for y in PROXY_YEARS if y != 2022)) != 0,
            input_iam_annual_totals / gridded_emisssions_annual_totals.sel(year=list(y for y in PROXY_YEARS if y != 2022)),
            1.0  # No correction where gridded is zero
        )
        
        # Add 2022 ratio (use 2023 ratio as proxy)
        if 2022 not in ratio_per_sector.year.values and 2023 in ratio_per_sector.year.values:
            scalar_2023 = ratio_per_sector.sel(year=2023)
            ratio_per_sector = xr.concat(
                [scalar_2023.expand_dims('year').assign_coords(year=[2022]), ratio_per_sector],
                dim='year'
            )
            ratio_per_sector = ratio_per_sector.sortby('year')
        
        # Apply this global scalar to `scen_ds`
        # Multiply the gridded data by the global scalar to match input emissions
        scen_ds_corrected = scen_ds.copy(deep=True)
        
        # Interpolate global_scalar_openburning to all time steps in scen_ds
        # Extract years from time coordinate
        time_years = scen_ds[var].time.dt.year.values
        
        # Create a scalar per year across all time steps
        scalar_by_time = xr.DataArray(
            np.array([
                ratio_per_sector.sel(year=yr, method='nearest').values
                for yr in time_years
            ]),
            coords={'time': scen_ds[var].time},
            dims=['time']
        )
        
        # Apply scalar to data variable
        scen_ds_corrected[var] = scen_ds[var] * scalar_by_time
        
        # Reorder dimensions if necessary
        scen_ds_corrected = scen_ds_corrected.pipe(reorder_dimensions, bound_var_name="bound")
        
        # Add global sums to metadata
        scen_ds_corrected = scen_ds_corrected.pipe(add_file_global_sum_totals_attrs, name=var)
        
        # Copy bounds variables from original dataset to ensure they exist
        copy_bounds_data_variables(source=scen_ds, target=scen_ds_corrected)
        
        # Save out updated data (overwrite the original file)
        outfile = file
        encoding = {var: {"zlib": True, "complevel": 2}}
        
        # Close source dataset to release file locks, following spatial harmonization pattern
        scen_ds.close()
        
        # Remove old file before writing
        outfile.unlink(missing_ok=True)
        
        # Save corrected dataset
        scen_ds_corrected.to_netcdf(outfile, encoding=encoding)
        scen_ds_corrected.close()
        
        print(f"\nSaved corrected {gas_name} AIR emissions timeseries to {outfile}")



    

# %%
# run the em_anthro timeseries correction (only em_anthro)
# NOTE: should take <1min per file
if run_anthro_timeseries_correction:

    # files that are produced above, that may need correction
    files = [
        file
        for file in (settings.out_path / GRIDDING_VERSION).glob("*.nc")
        if "openburning" not in file.name and "AIR" not in file.name and "speciated" not in file.name
    ]

    for file in files:
        gas_name, var, type_name = return_emission_names(file)
        print(f'run anthropogenic timeseries correction for {gas_name}')

        # Open dataset with explicit engine settings to avoid caching issues
        scen_ds = xr.open_dataset(file, engine='netcdf4')
        gridded_emisssions_annual_totals = ds_to_annual_emissions_total( # takes about 10-30 seconds
                gridded_data=scen_ds,
                var_name=var,
                cell_area=cell_area,
                keep_sectors=True
            )
        
        # Get input IAM emissions (already harmonised)
        input_emissions = iam_df.loc[ # assumes that this is already harmonised
                ismatch(gas=gas_name)
            ]
        SECTOR_RENAME_DOWNSCALED = {
            "Energy Sector": "Energy",
            "Industrial Sector": "Industrial",
            "Residential Commercial Other": "Residential, Commercial, Other",
            "Transportation Sector": "Transportation",
            "Biochar": "Other Capture and Removal", 
            "Direct Air Capture": "Other Capture and Removal", 
            "Enhanced Weathering": "Other Capture and Removal", 
            "Ocean": "Other Capture and Removal", 
            "Other CDR": "Other Capture and Removal", 
            "Soil Carbon Management": "Other Capture and Removal"
        }
        # rename and reaggregate (for Other Capture and Removal)
        input_emissions = input_emissions.rename(
            index=SECTOR_RENAME_DOWNSCALED,
            level="sector"
        ).groupby(level=['model', 'scenario', 'region', 'gas', 'sector', 'unit']).sum()
        if gas_name == "CO2":
            input_emissions = input_emissions.loc[
                    isin(sector=SECTOR_ORDERING_DEFAULT['CO2_em_anthro'])
                ]
        else:
            input_emissions = input_emissions.loc[
                    isin(sector=SECTOR_ORDERING_DEFAULT['em_anthro'])
                ]
        
        # Note: if N2O, divide by 1000 (from kt to Mt)
        if gas_name == "N2O":
            input_emissions = input_emissions / 1000
        
        # Calculate annual global totals by sector
        # Step 1: Ensure only one model-scenario combination
        unique_model = input_emissions.index.get_level_values('model').unique()
        unique_scenario = input_emissions.index.get_level_values('scenario').unique()
        assert len(unique_model) == 1, f"Expected 1 model, got {len(unique_model)}: {unique_model.tolist()}"
        assert len(unique_scenario) == 1, f"Expected 1 scenario, got {len(unique_scenario)}: {unique_scenario.tolist()}"
        
        # Step 2: Sum across regions, keeping sector and year information
        # Group by gas, sector, unit and sum (this sums across all regions)
        input_global = input_emissions.groupby(level=['gas', 'sector', 'unit']).sum()
        
        # Convert to DataFrame format with years as columns for easier viewing
        input_global_by_sector = input_global.reset_index()
        # Replace sector integer indices with full sector names
        input_global_by_sector['sector'] = input_global_by_sector['sector'].map(SECTOR_DICT_ANTHRO_CO2_SCENARIO_FLIPPED)
        
        # Convert to match gridded_emisssions_annual_totals format
        # Pivot so we have (gas, sector, unit) as index and years as columns
        input_global_by_sector = input_global_by_sector.set_index(['gas', 'sector', 'unit'])
        
        # Sort the index for consistency
        input_global_by_sector = input_global_by_sector.sort_index()
        
        # Transform from pandas DataFrame to xarray DataArray format
        # Extract sector values (should be 0, 1, ..., 7 for em_anthro and 0, 1, ..., 9 for CO2_em_anthro)
        sectors = input_global_by_sector.index.get_level_values('sector').unique()
        
        # Extract year columns (all columns that are integers)
        year_columns = sorted([int(col) for col in input_global_by_sector.columns if isinstance(col, int)])
        
        # Create 2D numpy array: rows = sectors, columns = years
        data_array = np.array([
            input_global_by_sector.loc[(gas_name, sector, input_global_by_sector.index.get_level_values('unit')[0]), year_columns].values
            for sector in sectors
        ]).T  # Transpose so years are rows, sectors are columns
        
        # Create xarray DataArray matching gridded_emisssions_annual_totals structure
        input_iam_annual_totals = xr.DataArray(
            data_array,
            coords={
                'year': year_columns,
                'sector': sectors.values
            },
            dims=['year', 'sector'],
            name=f'{gas_name}_em_anthro'
        )
        
        # Calculate ratios (for each sector, for one emissions species)
        # Compare input IAM vs gridded emissions for each sector and year
        # Handle division by zero where gridded is zero
        ratio_per_sector = xr.where(
            gridded_emisssions_annual_totals.sel(year=list(y for y in PROXY_YEARS if y != 2022)) != 0,
            input_iam_annual_totals / gridded_emisssions_annual_totals.sel(year=list(y for y in PROXY_YEARS if y != 2022)),
            1.0  # No correction where gridded is zero
        )
        
        # Add 2022 ratio (use 2023 ratio as proxy; should both be 1 -- except for shipping, where the netCDF CEDS is different from the CSV/scenario)
        if 2022 not in ratio_per_sector.year.values and 2023 in ratio_per_sector.year.values:
            scalar_2023 = ratio_per_sector.sel(year=2023)
            ratio_per_sector = xr.concat(
                [scalar_2023.expand_dims('year').assign_coords(year=[2022]), ratio_per_sector],
                dim='year'
            )
            ratio_per_sector = ratio_per_sector.sortby('year')
        
        # Replace sector=7 (International Shipping) values for 2022 and 2023 with 1.0
        ratio_per_sector.loc[dict(year=[2022, 2023], sector=7)] = 1.0
        
        # Apply this global scalar to `scen_ds`
        # Multiply the gridded data by the global scalar to match input emissions
        scen_ds_corrected = scen_ds.copy(deep=True)
        
        # Interpolate global_scalar_openburning to all time steps in scen_ds
        # Extract years from time coordinate
        time_years = scen_ds[var].time.dt.year.values
        
        # Create a scalar per year across all time steps
        scalar_by_time = xr.DataArray(
            np.array([
                ratio_per_sector.sel(year=yr, method='nearest').values
                for yr in time_years
            ]),
            coords={'time': scen_ds[var].time, 'sector': ratio_per_sector.sector.values},
            dims=['time', 'sector']
        )
        
        # Apply scalar to data variable
        scen_ds_corrected[var] = scen_ds[var] * scalar_by_time
        
        # Reorder dimensions if necessary
        scen_ds_corrected = scen_ds_corrected.pipe(reorder_dimensions, bound_var_name="bound")
        
        # Add global sums to metadata
        scen_ds_corrected = scen_ds_corrected.pipe(add_file_global_sum_totals_attrs, name=var)
        
        # Copy bounds variables from original dataset to ensure they exist
        copy_bounds_data_variables(source=scen_ds, target=scen_ds_corrected)
        
        # Save out updated data (overwrite the original file)
        outfile = file
        encoding = {var: {"zlib": True, "complevel": 2}}
        
        # Close source dataset to release file locks, following spatial harmonization pattern
        scen_ds.close()
        
        # Remove old file before writing
        outfile.unlink(missing_ok=True)
        
        # Save corrected dataset
        scen_ds_corrected.to_netcdf(outfile, encoding=encoding)
        scen_ds_corrected.close()
        
        print(f"\nSaved corrected {gas_name} anthropogenic emissions timeseries to {outfile}")

# %%
# run the openburning timeseries correction (only em_openburning)
# NOTE: should take <1min per file

if run_openburning_timeseries_correction:

    # files that are produced above, that may need correction
    files = [
        file
        for file in (settings.out_path / GRIDDING_VERSION).glob("*.nc")
        if "openburning" in file.name and "speciated" not in file.name and "H2" not in file.name
    ]

    for file in files:
        gas_name, var, type_name = return_emission_names(file)
        print(f'run openburning timeseries correction for {gas_name}')

        # Open dataset with explicit engine settings to avoid caching issues
        scen_ds = xr.open_dataset(file, engine='netcdf4')
        gridded_emisssions_annual_totals = ds_to_annual_emissions_total( # takes about 10-30 seconds
                gridded_data=scen_ds,
                var_name=var,
                cell_area=cell_area,
                keep_sectors=True
            )
        
        # Get input IAM emissions (already harmonised)
        input_emissions = (
            iam_df.loc[ # assumes that this is already harmonised
                ismatch(gas=gas_name)
            ].loc[
                isin(sector=SECTOR_ORDERING_DEFAULT['em_openburning'])
            ]
        )
        
        # Note: if N2O, divide by 1000 (from kt to Mt)
        if gas_name == "N2O":
            input_emissions = input_emissions / 1000
        
        # Calculate annual global totals by sector
        # Step 1: Ensure only one model-scenario combination
        unique_model = input_emissions.index.get_level_values('model').unique()
        unique_scenario = input_emissions.index.get_level_values('scenario').unique()
        assert len(unique_model) == 1, f"Expected 1 model, got {len(unique_model)}: {unique_model.tolist()}"
        assert len(unique_scenario) == 1, f"Expected 1 scenario, got {len(unique_scenario)}: {unique_scenario.tolist()}"
        
        # Step 2: Sum across regions, keeping sector and year information
        # Group by gas, sector, unit and sum (this sums across all regions)
        input_global = input_emissions.groupby(level=['gas', 'sector', 'unit']).sum()
        
        # Convert to DataFrame format with years as columns for easier viewing
        input_global_by_sector = input_global.reset_index()
        # Replace sector integer indices with full sector names
        input_global_by_sector['sector'] = input_global_by_sector['sector'].map(SECTOR_DICT_OPENBURNING_DEFAULT_FLIPPED)
        
        # Convert to match gridded_emisssions_annual_totals format
        # Pivot so we have (gas, sector, unit) as index and years as columns
        input_global_by_sector = input_global_by_sector.set_index(['gas', 'sector', 'unit'])
        
        # Sort the index for consistency
        input_global_by_sector = input_global_by_sector.sort_index()
        
        # Transform from pandas DataFrame to xarray DataArray format
        # Extract sector values (should be 0, 1, 2, 3 for openburning)
        sectors = input_global_by_sector.index.get_level_values('sector').unique()
        
        # Extract year columns (all columns that are integers)
        year_columns = sorted([int(col) for col in input_global_by_sector.columns if isinstance(col, int)])
        
        # Create 2D numpy array: rows = sectors, columns = years
        data_array = np.array([
            input_global_by_sector.loc[(gas_name, sector, input_global_by_sector.index.get_level_values('unit')[0]), year_columns].values
            for sector in sectors
        ]).T  # Transpose so years are rows, sectors are columns
        
        # Create xarray DataArray matching gridded_emisssions_annual_totals structure
        input_iam_annual_totals = xr.DataArray(
            data_array,
            coords={
                'year': year_columns,
                'sector': sectors.values
            },
            dims=['year', 'sector'],
            name=f'{gas_name}_em_openburning'
        )
        
        # Calculate ratios (for each sector, for one emissions species)
        # Compare input IAM vs gridded emissions for each sector and year
        # Handle division by zero where gridded is zero
        ratio_per_sector = xr.where(
            gridded_emisssions_annual_totals.sel(year=list(y for y in PROXY_YEARS if y != 2022)) != 0,
            input_iam_annual_totals / gridded_emisssions_annual_totals.sel(year=list(y for y in PROXY_YEARS if y != 2022)),
            1.0  # No correction where gridded is zero
        )
        
        # Add 2022 ratio (use 2023 ratio as proxy)
        if 2022 not in ratio_per_sector.year.values and 2023 in ratio_per_sector.year.values:
            scalar_2023 = ratio_per_sector.sel(year=2023)
            ratio_per_sector = xr.concat(
                [scalar_2023.expand_dims('year').assign_coords(year=[2022]), ratio_per_sector],
                dim='year'
            )
            ratio_per_sector = ratio_per_sector.sortby('year')
        
        # Apply this global scalar to `scen_ds`
        # Multiply the gridded data by the global scalar to match input emissions
        scen_ds_corrected = scen_ds.copy(deep=True)
        
        # Interpolate global_scalar_openburning to all time steps in scen_ds
        # Extract years from time coordinate
        time_years = scen_ds[var].time.dt.year.values
        
        # Create a scalar per year across all time steps
        scalar_by_time = xr.DataArray(
            np.array([
                ratio_per_sector.sel(year=yr, method='nearest').values
                for yr in time_years
            ]),
            coords={'time': scen_ds[var].time, 'sector': ratio_per_sector.sector.values},
            dims=['time', 'sector']
        )
        
        # Apply scalar to data variable
        scen_ds_corrected[var] = scen_ds[var] * scalar_by_time
        
        # Reorder dimensions if necessary
        scen_ds_corrected = scen_ds_corrected.pipe(reorder_dimensions, bound_var_name="bound")
        
        # Add global sums to metadata
        scen_ds_corrected = scen_ds_corrected.pipe(add_file_global_sum_totals_attrs, name=var)
        
        # Copy bounds variables from original dataset to ensure they exist
        copy_bounds_data_variables(source=scen_ds, target=scen_ds_corrected)
        
        # Save out updated data (overwrite the original file)
        outfile = file
        encoding = {var: {"zlib": True, "complevel": 2}}
        
        # Close source dataset to release file locks, following spatial harmonization pattern
        scen_ds.close()
        
        # Remove old file before writing
        outfile.unlink(missing_ok=True)
        
        # Save corrected dataset
        scen_ds_corrected.to_netcdf(outfile, encoding=encoding)
        scen_ds_corrected.close()
        
        print(f"\nSaved corrected {gas_name} openburning emissions timeseries to {outfile}")
        
        
# %% [markdown]
# # END OF MAIN CODE

# %%
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------

# %% [markdown]
# # Start of H2 openburning data
# Usually takes <2mins for 1 scenario

# %%
def _to_sector_integers_and_reorder(ds, type_name='em_openburning'):
    # translate/map sectors
    # Map sector names to integer indices based on SECTOR_ORDERING_DEFAULT
    sector_ordering = SECTOR_ORDERING_DEFAULT[type_name]
    sector_name_to_id = {name: idx for idx, name in enumerate(sector_ordering)}
    
    # Rename sectors in h2_translation to use integer IDs
    ds = h2_translation.assign_coords(
        sector=([sector_name_to_id.get(s, s) for s in h2_translation.sector.values])
    )
    
    return ds


# %%
# STEPS:
# 1. load CO file
# 2. load translation file
# 3 apply translation file (logic: h2_openburning = co_openburning * h2_translation)

if run_openburning_h2:
    print('Generating H2 openburning emissions from CO openburning and H2/CO emission factor ratios')
    
    # Load the CO openburning emissions
    co_openburning_file = settings.out_path / GRIDDING_VERSION / f"CO-em-openburning_{FILE_NAME_ENDING}"
    co_openburning = xr.open_dataset(co_openburning_file)
    
    # Load the H2/CO emission factor translation file
    h2_translation_file = settings.proxy_path / "EF_h2_div_EF_co.nc"
    h2_translation = xr.open_dataset(h2_translation_file)
    h2_translation = _to_sector_integers_and_reorder(h2_translation)

    # Initialize result with same structure as co_openburning
    h2_openburning = xr.Dataset(
        coords=co_openburning.coords,
        attrs=co_openburning.attrs.copy()
    )

    # Initialize the data variable with zeros
    h2_openburning_data = xr.zeros_like(co_openburning["CO_em_openburning"])

    # Perform multiplication for burning sectors
    for openburning_sector in np.unique(co_openburning.sector):
        # Select data from both datasets for matching sectors
        co_sector = co_openburning["CO_em_openburning"].sel(sector=openburning_sector)

        # Get translation factors for this sector and gas
        translation_factor_sector = h2_translation["EF_h2_div_EF_co"].sel(
            sector=openburning_sector
        )

        # Convert time coordinates to year/month for alignment
        years = co_sector.time.dt.year
        months = co_sector.time.dt.month

        # Find the index of the sector in the coordinate array
        sector_idx = list(co_openburning.sector.values).index(openburning_sector) # TODO: double-check that this sector_idx is correct, and not doing the wrong one

        # Perform multiplication for each time step
        for time_idx, time_val in enumerate(co_sector.time.values):
            year = years[time_idx].values
            month = months[time_idx].values

            # Check if this year/month exists in h2_translation
            if year in translation_factor_sector.year.values and month in translation_factor_sector.month.values:
                # Get the share data for this specific year/month
                translation_slice = translation_factor_sector.sel(year=year, month=month)

                # Get the bulk VOC data for this time step
                co_slice = co_sector.isel(time=time_idx)

                # Multiply and assign to result
                h2_openburning_data[time_idx, :, :, sector_idx] = (co_slice * translation_slice).values # sensitive to coordinate order
                
                # Assert that the sectors all align, ignoring dtype
                assert h2_openburning_data[time_idx, :, :, sector_idx].sector.values == co_slice.sector.values
                assert h2_openburning_data[time_idx, :, :, sector_idx].sector.values == translation_slice.sector.values


    # Add the computed data to the result dataset
    gas_variable_name = "H2_em_openburning"
    h2_openburning[gas_variable_name] = h2_openburning_data

    # TODO:
    # - [ ] update long_name of data (follow CEDS long_name)
    # - [ ] remove/replace the now unnecessary bounds updates?
    # # Add the bounds
    # h2_openburning['lon_bnds'] = co_openburning['lon_bnds']
    # h2_openburning['time_bnds'] = co_openburning['time_bnds']
    # h2_openburning['lat_bnds'] = co_openburning['lat_bnds']

    # Update attributes
    h2_openburning.attrs['variable_id'] = gas_variable_name
    h2_openburning.attrs['title'] = f"Speciated {gas_variable_name} emissions"

    # save out
    print('Writing out H2 openburning emissions')
    outfile = settings.out_path / GRIDDING_VERSION / f"{gas_variable_name.replace("_","-")}_{FILE_NAME_ENDING}"

    encoding = {
        gas_variable_name: {
            "zlib": True,
            "complevel": 2
        }
    }
    h2_openburning.to_netcdf(outfile, mode="w", encoding=encoding, compute=True)


# %%
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------


# %% [markdown]
# # Start of SUPPLEMENTAL DATA

# %% [markdown]
# # VOC speciation
# **NOTE: currently takes quite long, especially anthro VOC speciation


# %%
lock = SerializableLock()

# %%
# Load VOC data

def load_voc_bulk(type="anthro"):

    # load VOC (bulk) scenario file
    if type=="anthro":        
        # anthro
        voc_anthro = xr.open_dataset(
            settings.out_path / GRIDDING_VERSION / f"NMVOC-em-anthro_{FILE_NAME_ENDING}",
        chunks={},
        lock=lock
        )

        return voc_anthro
    
    if type=="openburning":

        # openburning
        voc_openburning = xr.open_dataset(
            settings.out_path / GRIDDING_VERSION / f"NMVOCbulk-em-openburning_{FILE_NAME_ENDING}",
        chunks={},
        lock=lock
        )

        return voc_openburning



# %%
# AIR (anthro) is not required.

# %%
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------

# %% [markdown]
# # VOC speciation (BB4CMIP, openburnig)
# **NOTE: currently takes long at ~5mins per VOC species, around 2hrs for all 25 VOC species**

# %%
# Calculate VOC-speciation data; keep the structure of the VOC (bulk) data

PROXY_TIME_RANGE_VOC_CEDS = "2023"
PROXY_TIME_RANGE_VOC_BB4CMIP = "2014-23"

voc_spec_ratios_location_anthro = settings.proxy_path / "VOC_speciation"
voc_spec_ratios_location_openburning = settings.proxy_path / "NMVOC_speciation"

# loop through all CEDS em-anthro VOC-species from input4MIP files
# 1. load share data
# 2. create an "empty"/"template" dataset as a copy of voc_anthro
# 3. fill with zeroes
# 4. for each sector,
#   i. do multiplication
#   ii. assign sector value
# 5. Update/set other attributes

# %%
if run_openburning_supplemental_voc:
    voc_openburning = load_voc_bulk(type="openburning")

    if DO_VOC_SPECIATION_OPENBURNING_ONLY_FOR_THESE_SPECIES is None:
        DO_VOC_SPECIATION_OPENBURNING_ONLY_FOR_THESE_SPECIES = GASES_ESGF_BB4CMIP_VOC

    # sector index to short name (must follow SECTOR_ORDERING_DEFAULT['em_openburning'])
    sector_mapping = {
        0: "AWB",
        1: "FRTB",
        2: "GRSB",
        3: "PEAT",
    }

    numeric_sectors = voc_openburning.sector.values  # save numeric sectors

    # prepare bulk VOC once
    voc_bulk = voc_openburning["NMVOCbulk_em_openburning"]

    # add year/month coordinates for vectorised alignment
    voc_bulk = voc_bulk.assign_coords(
        year=("time", voc_bulk.time.dt.year.data),
        month=("time", voc_bulk.time.dt.month.data),
    )

    # temporarily rename sector coordinate to match VOC share naming
    voc_bulk = voc_bulk.assign_coords(
        sector=[sector_mapping[s] for s in voc_bulk.sector.values]
    )

    for v in DO_VOC_SPECIATION_OPENBURNING_ONLY_FOR_THESE_SPECIES:
        print(f"Reading in shares of {v}")

        voc_share = xr.open_dataset(
            voc_spec_ratios_location_openburning
            / f"{v}_other_voc_em_speciated_NMVOC_openburning_{PROXY_TIME_RANGE_VOC_BB4CMIP}.nc",
            engine="netcdf4",
            chunks={"time": 12},
            lock=lock,
        )

        # select proxy data
        share = voc_share["emissions_share"].sel(
            gas=voc_share.gas.values[0]
        )

        # align share data to bulk time
        share_time = share.sel(
            year=voc_bulk.year,
            month=voc_bulk.month,
        )

        # align dimensions
        voc_bulk_aligned, share_aligned = xr.align(
            voc_bulk,
            share_time,
            join="inner",
        )

        # scale with proxy
        voc_spec_data = voc_bulk_aligned * share_aligned

        voc_spec_data = voc_spec_data.assign_coords(sector=numeric_sectors)
        voc_spec_data = voc_spec_data.drop_vars(["year", "month", "gas"], errors="ignore")
        voc_spec_data = voc_spec_data.fillna(0)

        # construct output variable name
        gas_variable_name = (
            f"NMVOC_{voc_share.gas.values[0]}_em_speciated_VOC_openburning"
        )

        # build output dataset
        voc_spec = voc_spec_data.to_dataset(name=gas_variable_name)

        # copy & update attributes
        voc_spec.attrs.update(voc_openburning.attrs)
        voc_spec.attrs["variable_id"] = gas_variable_name
        voc_spec.attrs["title"] = (
            f"Future openburning emissions of speciated {gas_variable_name}"
        )

        # write output
        print(f"Writing out emissions of {v}")
        name = gas_variable_name.replace("_", "-")
        outfile = settings.out_path / GRIDDING_VERSION / f"{name}_{FILE_NAME_ENDING}"

        encoding = {
            gas_variable_name: {
                "zlib": True,
                "complevel": 2,
            }
        }

        with ProgressBar():
            voc_spec.to_netcdf(
                outfile,
                mode="w",
                encoding=encoding,
                compute=True,
            )


# %%
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------


# %% [markdown]
# ## VOC speciation (CEDS, anthro)
# **NOTE: runtime down to 12 minutes for all 23 VOC species**


# %%
if run_anthro_supplemental_voc:
    voc_anthro = load_voc_bulk(type="anthro")

    if DO_VOC_SPECIATION_ANTHRO_ONLY_FOR_THESE_SPECIES is None:
        DO_VOC_SPECIATION_ANTHRO_ONLY_FOR_THESE_SPECIES = GASES_ESGF_CEDS_VOC

    # sector index to short name (must follow voc_anthro.sector ordering)
    sector_mapping = {
        0: "AGR",
        1: "ENE",
        2: "IND",
        3: "TRA",
        4: "RCO",
        5: "SLV",
        6: "WST",
        7: "SHP",
    }

    numeric_sectors = voc_anthro.sector.values  # save numeric sectors

    # prepare bulk VOC
    voc_bulk = voc_anthro["NMVOC_em_anthro"]
    
    # add year/month coordinates for vectorised alignment
    voc_bulk = voc_bulk.assign_coords(
        year=("time", voc_bulk.time.dt.year.data),
        month=("time", voc_bulk.time.dt.month.data),
    )

    # rename sector coordinate to match VOC share naming
    voc_bulk = voc_bulk.assign_coords(
        sector=[sector_mapping[s] for s in voc_bulk.sector.values]
    )

    for v in DO_VOC_SPECIATION_ANTHRO_ONLY_FOR_THESE_SPECIES:
        print(f"Reading in shares of {v}")

        voc_share = xr.open_dataset(
            voc_spec_ratios_location_anthro / f"{v}_{PROXY_TIME_RANGE_VOC_CEDS}.nc",
            engine="netcdf4",
            chunks={"time": 12},
            lock=lock,
        )

        gas_variable_name = voc_share.gas.values[0]

        # select gas and prepare share data
        share = voc_share["emissions_share"].sel(gas=gas_variable_name)

        # align share data to voc_bulk time using year/month (vectorised)
        share_time = share.sel(
            year=voc_bulk.year,
            month=voc_bulk.month,
        )

        # align sectors and spatial dims
        voc_bulk_aligned, share_aligned = xr.align(
            voc_bulk,
            share_time,
            join="inner",
        )

        # vectorised multiplication (lazy, fast)
        voc_spec_data = voc_bulk_aligned * share_aligned

        voc_spec_data = voc_spec_data.assign_coords(sector=numeric_sectors)

        # drop extra coordinates that were made in the process
        voc_spec_data = voc_spec_data.drop_vars(
            ["year", "month", "year_range", "gas"],
            errors="ignore"
        )

        # build output dataset
        voc_spec = voc_spec_data.to_dataset(name=gas_variable_name)

        # TODO:
        # - [ ] update long_name of data (follow CEDS long_name) EDIT: i think this was done, right?
        # Add the bounds
        # voc_spec['lon_bnds'] = voc_anthro['lon_bnds']
        # voc_spec['time_bnds'] = voc_anthro['time_bnds']
        # voc_spec['lat_bnds'] = voc_anthro['lat_bnds']
        
        # update attributes
        voc_spec.attrs.update(voc_anthro.attrs)
        voc_spec.attrs["variable_id"] = gas_variable_name
        voc_spec.attrs["title"] = f"Speciated {gas_variable_name} emissions"

        # write output
        print(f"Writing out emissions of {v}")
        name = gas_variable_name.replace("_", "-")
        outfile = settings.out_path / GRIDDING_VERSION / f"{name}_{FILE_NAME_ENDING}"

        encoding = {gas_variable_name: {"zlib": True, "complevel": 2,}}

        with ProgressBar():
            voc_spec.to_netcdf(outfile, mode="w", encoding=encoding, compute=True,)

# %% [markdown]
# # END OF SUPPLEMENTAL DATA CODE


# %%
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------


# %% [markdown]
# # CONTINUED POSTPROCESSING


# %% editable=true slideshow={"slide_type": ""} tags=["parameters"]
plot_timeseries: bool = True
PLOT_GASES: list[str] | None = None # e.g. ["CO2", "SO2", "VOC01_alcohols", "VOC02_ethane", "NMVOC-C2H2", "NMVOC-C10H16"]; default is run all
PLOT_GASES: list[str] | None = DO_GRIDDING_ONLY_FOR_THESE_SPECIES # e.g. ["CO2", "SO2", "VOC01_alcohols", "VOC02_ethane", "NMVOC-C2H2", "NMVOC-C10H16"]; default is run all

PLOT_SECTORS: list[str] | None = None # e.g. ['Energy', 'Residential, Commercial, Other'] default is run all
PLOT_SECTORS: list[str] | None = SECTOR_ORDERING_DEFAULT['CO2_em_anthro'] # default is run all
PLOT_SECTORS: list[str] | None = SECTOR_ORDERING_DEFAULT['em_anthro'] # default is run all



# %% [markdown]
# ## 2. plot differences/harmonized maps compared to historical

# %% [markdown]
# ## 2.1. plot compared to CEDS (anthro)


# %%
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
                                     anthro_bb_air = "CEDS_anthro", # CEDS_anthro, BB4CMIP7, CEDS_AIR
                                     figsize_per_panel=(4, 3), proj=ccrs.Robinson(),
                                     colour_scale_max_percentile = 98,
                                     empty_treatment="fill_zeroes" # alternative: "skip"
                                     ):
    """
    Plot comparison between CEDS and scenario data in 4 columns.
    
    Handles cases where 'time' or 'sector' may only have one value (not a dimension).
    """
    
    # 1. Ensure time dimension exists and has monotonic coordinates; remove duplicates
    if 'time' in ceds_da.dims:
        ceds_da = ceds_da.sortby('time')
        _, unique_indices = np.unique(ceds_da.time.values, return_index=True)
        ceds_da = ceds_da.isel(time=np.sort(unique_indices))
    
    if 'time' in scen_da.dims:
        scen_da = scen_da.sortby('time')
        _, unique_indices = np.unique(scen_da.time.values, return_index=True)
        scen_da = scen_da.isel(time=np.sort(unique_indices))
    
    try:
        # Select time and squeeze, handling both dimension and non-dimension cases
        if 'time' in ceds_da.dims:
            ceds_slice = ceds_da.sel(time=time_slice, method='nearest')
        else:
            ceds_slice = ceds_da
            
        if 'time' in scen_da.dims:
            scen_slice = scen_da.sel(time=time_slice, method='nearest')
        else:
            scen_slice = scen_da
        
        ceds_slice = ceds_slice.squeeze()
        scen_slice = scen_slice.squeeze()
    except KeyError as e:
        print(f"Error selecting time slice {time_slice}. The time coordinate may be missing or invalid: {e}")
        return # Exit the function
    
    # 2. Pre-check sector availability
    has_sector_dim_ceds = 'sector' in ceds_da.dims
    has_sector_dim_scen = 'sector' in scen_da.dims
    ceds_sectors = set(ceds_da.sector.values) if has_sector_dim_ceds else {ceds_da.sector.values.item()}
    scen_sectors = set(scen_da.sector.values) if has_sector_dim_scen else {scen_da.sector.values.item()}
    
    # Plots
    nrows = len(sectors)
    ncols = 4
    
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
        subplot_kw={"projection": proj}
    )
    
    if nrows == 1:
        axes = axes.reshape(1, -1)
    
    # Column titles
    col_titles = [f'{anthro_bb_air} Data', 
                  f'{gas} CMIP7 Scenario', 
                  f'Difference ({anthro_bb_air} - Scenario)',
                  'Percentage Difference (%)']
    
    # Determine variable name once
    if anthro_bb_air == "CEDS_anthro":
        var_name = f'{gas}_em_anthro'
    elif anthro_bb_air == "BB4CMIP7":
        print("Biomass burning vetting plots have not yet been implemented.")
        for col in range(ncols):
            axes[row, col].set_visible(False) if nrows > 1 else axes[:, col].set_visible(False)
        return
    else:
        var_name = None
    
    for row, sector in enumerate(sectors):
        
        if gas != "CO2" and sector not in SECTOR_DICT_ANTHRO_DEFAULT:
            continue
            
        sector_in_ceds = sector in ceds_sectors
        sector_in_scen = sector in scen_sectors
        
        if empty_treatment=="skip":
            if not sector_in_ceds or not sector_in_scen:
                print(f"Skipping sector '{sector}:{SECTOR_DICT_ANTHRO_CO2_SCENARIO[sector]}' - missing in {f'{anthro_bb_air}' if not sector_in_ceds else 'scenario'} data")
                for col in range(ncols):
                    axes[row, col].set_visible(False)
                continue
            
            # Fast path: sector exists in both, just select
            if has_sector_dim_ceds:
                ceds_values_base = ceds_slice.sel(sector=sector)
            else:
                ceds_values_base = ceds_slice
                
            if has_sector_dim_scen:
                scen_values_base = scen_slice.sel(sector=sector)
            else:
                scen_values_base = scen_slice
            
        elif empty_treatment=="fill_zeroes":
            # Select sector data, or create a zero-filled placeholder if missing
            if sector_in_ceds:
                if has_sector_dim_ceds:
                    ceds_values_base = ceds_slice.sel(sector=sector)
                else:
                    ceds_values_base = ceds_slice
            else:
                print(f"Warning: Sector '{sector}:{SECTOR_DICT_ANTHRO_CO2_SCENARIO[sector]}' not found in {anthro_bb_air} data, using zeros")
                # Create zero template from first sector
                if has_sector_dim_ceds:
                    template_sector = ceds_slice.isel(sector=0)
                else:
                    template_sector = ceds_slice
                ceds_values_base = xr.zeros_like(template_sector)
                
            if sector_in_scen:
                if has_sector_dim_scen:
                    scen_values_base = scen_slice.sel(sector=sector)
                else:
                    scen_values_base = scen_slice
            else:
                print(f"Warning: Sector '{sector}:{SECTOR_DICT_ANTHRO_CO2_SCENARIO[sector]}' not found in scenario data, using zeros")
                if has_sector_dim_scen:
                    template_sector = scen_slice.isel(sector=0)
                else:
                    template_sector = scen_slice
                scen_values_base = xr.zeros_like(template_sector)

        # Helper to safely extract the data variable (works for DataArray or Dataset)
        def get_data_var(ds, var_name):
            if isinstance(ds, xr.DataArray):
                return ds 
            if var_name and var_name in ds:
                return ds[var_name]
            # Fallback for DataSets where the desired var_name might not exist (e.g., zero-filled placeholder)
            elif len(ds.data_vars) > 0:
                return ds[list(ds.data_vars)[0]] 
            else:
                # Should not happen if a zero-like structure was created correctly
                raise ValueError(f"No data variable found for sector {sector}:{SECTOR_DICT_ANTHRO_CO2_SCENARIO[sector]}.")

        try:
            ceds_values = get_data_var(ceds_values_base, var_name)
            scen_values = get_data_var(scen_values_base, var_name)
        except ValueError as e:
             print(f"Skipping sector '{sector}:{SECTOR_DICT_ANTHRO_CO2_SCENARIO[sector]}' due to variable selection error: {e}")
             for col in range(ncols):
                axes[row, col].set_visible(False)
             continue
            
        # Ensure consistent dimension order
        if scen_values.dims == ('lon', 'lat'):
             scen_values = scen_values.transpose('lat', 'lon')
        if ceds_values.dims == ('lon', 'lat'):
             ceds_values = ceds_values.transpose('lat', 'lon')
            
        # Pre-compute all derived arrays
        diff_values = ceds_values - scen_values
        
        # Calculate percentage difference, handling division by zero (vectorized)
        pct_diff = xr.where(ceds_values != 0, (diff_values / ceds_values) * 100, 0)
        
        # Pre-compute statistics for normalization (faster than doing it per-column)
        valid_ceds = ceds_values.values[~np.isnan(ceds_values.values)]
        if len(valid_ceds) > 0:
            vmin_ceds = float(np.percentile(valid_ceds, 2))
            vmax_ceds = float(np.percentile(valid_ceds, colour_scale_max_percentile))
            vmax_diff = float(np.percentile(valid_ceds, 98))
        else:
            vmin_ceds = vmax_ceds = 0.0
            vmax_diff = 1.0
        
        # Plot data in 4 columns
        datasets = [ceds_values, scen_values, diff_values, pct_diff]
        cmaps = ['Reds', 'Blues', 'RdBu_r', 'coolwarm']
        
        for col, (data, cmap) in enumerate(zip(datasets, cmaps)):
            ax = axes[row, col] if nrows > 1 else axes[col]
            
            # Set colormap normalization
            if col in [0, 1]:
                if vmax_ceds == vmin_ceds: 
                    norm = colors.Normalize(vmin=0, vmax=1)
                else:
                    norm = colors.Normalize(vmin=vmin_ceds, vmax=vmax_ceds)
            
            elif col == 2: # Difference
                if vmax_diff == 0:
                    norm = colors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
                else:
                    norm = colors.TwoSlopeNorm(vmin=-vmax_diff, vcenter=0, vmax=vmax_diff)
            
            elif col == 3:  # Percentage difference - center at 0
                abs_max = 100 
                cmap, norm = shifted_white_colormap("coolwarm", vmin=-abs_max, vmax=abs_max)
            else:
                norm = None
            
            # Create the plot
            im = data.plot.pcolormesh(
                ax=ax,
                transform=ccrs.PlateCarree(),
                cmap=cmap,
                norm=norm,
                add_colorbar=False,
                add_labels=False
            )
            
            # Add coastlines and formatting
            ax.coastlines(linewidth=0.5)
            ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5)
            
            # Set titles and labels
            if row == 0:
                ax.set_title(col_titles[col], fontsize=10, fontweight='bold')
            
            if col == 0:
                ax.text(-0.15, 0.5, SECTOR_DICT_ANTHRO_CO2_SCENARIO[sector], transform=ax.transAxes, 
                        rotation=90, va='center', ha='center', fontsize=10, fontweight='bold')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, orientation='horizontal', 
                                 shrink=0.8, pad=0.05, aspect=20)
            cbar.ax.tick_params(labelsize=8)
            
            if col == 3:
                cbar.set_label('Percentage Difference (%)', fontsize=8)
            else:
                cbar.set_label(f'{gas} emissions (kg/m²/s)', fontsize=8)
    
    # Overall title - handle both dimension and non-dimension time cases
    time_val = ceds_slice.time.values if 'time' in ceds_slice.dims else ceds_slice.time.values
    fig.suptitle(f'{gas}, time: {time_val}', 
                  fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93, left=0.08)
    return fig

# %%
# NOTE: takes about ~mins per figure

folder_plots = settings.out_path / GRIDDING_VERSION / "plots"
folder_plots.mkdir(parents=True, exist_ok=True)
anthro_bb_air="CEDS_anthro" # CEDS_anthro, BB4CMIP7, CEDS_AIR

TIMES = [
    cftime.DatetimeNoLeap(2023, 1, 16)
    # cftime.DatetimeNoLeap(2023, 6, 16),
    # cftime.DatetimeNoLeap(2023, 12, 16)
]



for file in tqdm((settings.out_path / GRIDDING_VERSION).glob("*.nc"), "Plot maps: diffs with CEDS in 2023"): # loop over all produced files    
    gas_name, var, type_name = return_emission_names(file)

    for t in TIMES:
        if gas_name in PLOT_GASES:
            if type_name == "anthro":

                scen_ds = xr.open_dataset(file).sel(time=t).pipe(reorder_dimensions)
                scen_da = scen_ds[f"{gas_name}_em_anthro"]

                match = next(ceds_data_location.glob(f"{gas_name}-*.nc"), None)

                ceds_ds = xr.open_dataset(match).sel(time=t).pipe(reorder_dimensions) # TODO: for CO2, add zeroes for CDR sectors in ceds history for plotting
                ceds_da = ceds_ds[f"{gas_name}_em_anthro"]
                if gas_name == 'CO2':
                    ceds_da = xr.concat([
                        ceds_da,
                        xr.zeros_like(scen_da.sel(sector=[8,9]))
                    ], dim="sector")

                # print(N2O)

                AVAILABLE_SECTORS = [k for k in scen_ds.sector.values if SECTOR_DICT_ANTHRO_CO2_SCENARIO[k] in PLOT_SECTORS]

                fig = plot_ceds_vs_scenario_comparison(
                    ceds_da=ceds_da,
                    scen_da=scen_da,
                    gas=gas_name,
                    sectors=AVAILABLE_SECTORS,
                    # We pass the desired time slice (a single cftime object)
                    time_slice=t,
                    anthro_bb_air=anthro_bb_air,
                    colour_scale_max_percentile=98,
                    empty_treatment="fill_zeroes"
                )

                # --- SAVE AND SHOW PLOT ---
                if fig is not None:
                    filename_base = f"ceds_vs_scenario_comparison_{gas_name}_{t.strftime('%Y%m%d')}"

                    # Save the plot
                    fig.savefig(folder_plots / f"{filename_base}.png", dpi=200, bbox_inches='tight')
                    # fig.savefig(folder_plots / f"{filename_base}.pdf", bbox_inches='tight')

                    plt.show()
                    plt.close(fig) # Close the figure to free memory





# %% [markdown]
# # CONTINUED POSTPROCESSING
# ## 3. writing out some check files
#
#

# %%
# Total emissions (<1min per file)
save_total_emissions_as_csv = True
CALCULATE_TOTALS_GASES: list[str] | None = None # e.g. ["CO2", "SO2", "VOC01_alcohols", "VOC02_ethane", "NMVOC-C2H2", "NMVOC-C10H16"]; default is run all
CALCULATE_TOTALS_GASES: list[str] | None = GASES_ESGF_CEDS # e.g. ["CO2", "SO2", "VOC01_alcohols", "VOC02_ethane", "NMVOC-C2H2", "NMVOC-C10H16"]; default is run all


if save_total_emissions_as_csv:
    folder_totals = settings.out_path / GRIDDING_VERSION / "check_annual_totals"
    folder_totals.mkdir(parents=True, exist_ok=True)

    areacella = xr.open_dataset(Path(settings.gridding_path, "areacella_input4MIPs_emissions_CMIP_CEDS-CMIP-2025-04-18_gn.nc"))
    cell_area = areacella["areacella"]

    for file in tqdm((settings.out_path / GRIDDING_VERSION).glob("*.nc"), "Check: calculating total annual emissions from the gridded files"): # loop over all produced files
        gas_name, var, type_name = return_emission_names(file)

        if gas_name in CALCULATE_TOTALS_GASES:

            # load full nc file
            scen = xr.open_dataset(file)
            # convert to global annual totals
            scen_sectors_df = ds_to_annual_emissions_total( # takes about 10-30 seconds
                gridded_data=scen,
                var_name=var,
                cell_area=cell_area,
                keep_sectors=True
            ).to_pandas()
            scen_sectors_df.to_csv(folder_totals / f"{var.replace("_","-")}_{FILE_NAME_ENDING.rstrip('.nc')}_annual_totals_by_sector.csv")

            scen_df = ds_to_annual_emissions_total( # takes about 5-10 seconds
                gridded_data=scen,
                var_name=var,
                cell_area=cell_area,
                keep_sectors=False
            ).to_pandas().to_frame(name='emissions_Mt_year')
            scen_df.to_csv(folder_totals / f"{var.replace("_","-")}_{FILE_NAME_ENDING.rstrip('.nc')}_annual_totals.csv")

# %% [markdown]
# # CONTINUED POSTPROCESSING
# ## 4. plotting

# %% [markdown]
# ## 4.1. alignment with historical; from 'notebooks\cmip7\check_gridded-scenarios-compare-to-ceds-esgf.py'

# %%

if PLOT_SECTORS is None:
    PLOT_SECTORS = np.unique(SECTOR_ORDERING_DEFAULT["CO2_em_anthro"] + SECTOR_ORDERING_DEFAULT["em_anthro"] + SECTOR_ORDERING_DEFAULT["em_openburning"])

if PLOT_GASES is None:
    PLOT_GASES = np.unique(GASES_ESGF_CEDS + GASES_ESGF_BB4CMIP)


# used in 'timeseries'
# Define locations dictionary with coordinates
LOCATIONS = {
    # 'Beijing': (39.9042, 116.4074),
    "Laxenburg": (48.0689, 16.3555),
    # "Nuuk": (64.1743, -51.7373),
    # 'Geneva': (46.2044, 6.1432),
    # 'Delhi': (28.6139, 77.2090),
    # 'Spain': (40.4637, 3.7492), # central spain, close to Madrid
    # 'New_York': (40.7128, -74.0060),
    'London': (51.5074, -0.1278),
    # 'Tokyo': (35.6762, 139.6503),
    # 'São_Paulo': (-23.5505, -46.6333),
    'Lagos': (6.5244, 3.3792),
    # 'Mumbai': (19.0760, 72.8777),
    # 'Rural_Amazon': (-3.4653, -62.2159),  # Remote area in Amazon
    # 'North_Atlantic': (45.0, -30.0),     # Shipping route
    'South_China_Sea': (12.0, 113.0)     # Shipping route
}

# %%
folder_plots = settings.out_path / GRIDDING_VERSION / "plots"
folder_plots.mkdir(parents=True, exist_ok=True)

for file in tqdm((settings.out_path / GRIDDING_VERSION).glob("*.nc"), "Check: calculating total annual emissions from the gridded files"): # loop over all produced files
    gas_name, var, type_name = return_emission_names(file)

    if gas_name in PLOT_GASES:
        if type_name == "anthro":

            print(var)

            scen_ds = xr.open_dataset(file)
            print(scen_ds)

            match = next(ceds_data_location.glob(f"{gas_name}-*.nc"), None)
            ceds_ds = xr.open_dataset(match) # TODO: for CO2, add zeroes for CDR sectors in ceds history for plotting
            print(ceds_ds)

            AVAILABLE_SECTORS = [k for k in scen_ds.sector.values if SECTOR_DICT_ANTHRO_CO2_SCENARIO[k] in PLOT_SECTORS]
            
            for sec in AVAILABLE_SECTORS:
                sector_name = SECTOR_DICT_ANTHRO_CO2_SCENARIO[sec] # note: different for plotting openburning or AIR

                for place, (lat, lon) in LOCATIONS.items():
                    print(f"\nGenerating plots for {place} ({lat:.2f}°, {lon:.2f}°) - {gas_name} {sector_name}")
                    
                    try:
                        # Single gridpoint timeseries
                        fig1, ax1 = plot_place_timeseries(ceds_ds, scen_ds,
                                            lat=lat, lon=lon,
                                            place=place,
                                            gas=gas_name,
                                            sector=sec, sector_name=sector_name,
                                            type=f"em_{type_name}")
                        plt.savefig(folder_plots / f"{place}_timeseries_{gas_name}_{sector_name}.png",
                                    dpi=300, 
                                    bbox_inches='tight')
                        plt.show()

                        # Area average timeseries
                        fig2, ax2 = plot_place_area_average_timeseries(ceds_ds, scen_ds, 
                                            lat=lat, lon=lon,
                                            place=place,
                                            gas=gas_name, 
                                            sector=sec, sector_name=sector_name,
                                            lat_range=2.0, lon_range=2.0,
                                            type=f"em_{type_name}")
                        plt.savefig(folder_plots / f"{place}_area_timeseries_{gas_name}_{sector_name}.png", 
                                    dpi=300,
                                    bbox_inches='tight')
                        plt.show()
                        
                    except Exception as e:
                        print(f"Error plotting {place} {gas_name} {sector_name}: {e}")
                        continue


# %% [markdown]
# ## 4.2. alignment with downscaled scenario information

# %%
if DO_GRIDDING_ONLY_FOR_THESE_SPECIES is None:
    CALCULATE_TOTALS_GASES = list(downscaled.index.get_level_values("gas").unique())
else:
    CALCULATE_TOTALS_GASES = DO_GRIDDING_ONLY_FOR_THESE_SPECIES


# %%
if save_total_emissions_as_csv:
    from concordia.cmip7.utils_plotting import ds_to_annual_emissions_total
    import seaborn as sns
    
    folder_totals = settings.out_path / GRIDDING_VERSION / "check_annual_totals_ext"
    folder_totals.mkdir(parents=True, exist_ok=True)

    areacella = xr.open_dataset(
        Path(settings.gridding_path, "areacella_input4MIPs_emissions_CMIP_CEDS-CMIP-2025-04-18_gn.nc")
    )
    cell_area = areacella["areacella"]

    all_gases_df_list = []

    for file in tqdm((settings.out_path / GRIDDING_VERSION).glob("*.nc"),
                     desc="Calculating total annual emissions from the gridded files"):

        gas_name, var, type_name = return_emission_names(file)

        if gas_name not in CALCULATE_TOTALS_GASES:
            continue
        
        print(gas_name)
        
        scen = xr.open_dataset(file)

        if "AIR" in file.stem:
            da = ds_to_annual_emissions_total(
                gridded_data=scen,
                var_name=var,
                cell_area=cell_area,
                keep_sectors=False
            )

            # Convert Series/DataArray to DataFrame
            if isinstance(da, xr.DataArray):
                df = da.to_dataframe(name="emissions").reset_index()
            elif isinstance(da, pd.Series):
                df = da.reset_index(name="emissions")
            else:
                raise TypeError(f"Unexpected type: {type(da)}")

            df["gas"] = gas_name
            df["sector"] = "Aircraft"

        elif "anthro" in file.stem:
            da = ds_to_annual_emissions_total(
                gridded_data=scen,
                var_name=var,
                cell_area=cell_area,
                keep_sectors=True
            )

            if isinstance(da, xr.DataArray):
                df = da.to_dataframe(name="emissions").reset_index()
            elif isinstance(da, pd.Series):
                df = da.reset_index(name="emissions")
            else:
                raise TypeError(f"Unexpected type: {type(da)}")

            df["gas"] = gas_name

            # rename sectors based on gas
            if gas_name == "CO2":
                df["sector"] = df["sector"].map(SECTOR_DICT_ANTHRO_CO2_SCENARIO)
            else:
                df["sector"] = df["sector"].map(SECTOR_DICT_ANTHRO_DEFAULT)

        else:
            da = ds_to_annual_emissions_total(
                gridded_data=scen,
                var_name=var,
                cell_area=cell_area,
                keep_sectors=True
            )

            if isinstance(da, xr.DataArray):
                df = da.to_dataframe(name="emissions").reset_index()
            elif isinstance(da, pd.Series):
                df = da.reset_index(name="emissions")
            else:
                raise TypeError(f"Unexpected type: {type(da)}")

            df["gas"] = gas_name
            df["sector"] = df["sector"].map(SECTOR_DICT_OPENBURNING_DEFAULT)

        # Pivot to wide format: years as columns
        df_wide = df.pivot(index=["gas", "sector"], columns="year", values="emissions")
        all_gases_df_list.append(df_wide)

    
    # Combine all files into one MultiIndex DataFrame
    combined_df = pd.concat(all_gases_df_list).sort_index()

    parts = file.stem.split("_")
    new_stem = "_".join(parts[1:])
    
    combined_df.to_csv(folder_totals / f"{new_stem}_combined-annual-totals.csv")

combined_df

# %%
# downscaled_reference = downscaled.groupby(level=["gas","sector"]).sum()
iam_reference = iam_df.groupby(level=["gas","sector"]).sum()

# %%
# List of CDR sectors to combine
source_sectors = [
    "Biochar", 
    "Direct Air Capture", 
    "Enhanced Weathering", 
    "Ocean", 
    "Other CDR", 
    "Soil Carbon Management"
]

# Function to map old sectors to new one
def map_sector(s):
    if s in source_sectors:
        return "Other Capture and Removal"
    return s


# %%
# Only apply to gas = "CO2"
idx = iam_reference.index.to_frame()
mask = idx["gas"] == "CO2"

# Map the sectors
idx.loc[mask, "sector"] = idx.loc[mask, "sector"].map(map_sector)

# Set the MultiIndex back
iam_reference.index = pd.MultiIndex.from_frame(idx)

iam_ref = iam_reference.groupby(level=["gas", "sector"]).sum()

iam_ref

# %%
# ok we have to make them comparable; retain only years that exist in gridded
combined_df.columns = combined_df.columns.astype(int)
iam_ref.columns = iam_ref.columns.astype(int)

# find common years
common_years = combined_df.columns.intersection(iam_ref.columns)

# subset both to those years
combined_df = combined_df[common_years]
iam_ref = iam_ref[common_years]

# %%
# rename the iam sectors
SECTOR_RENAME_DOWNSCALED = {
    "Energy Sector": "Energy",
    "Industrial Sector": "Industrial",
    "Residential Commercial Other": "Residential, Commercial, Other",
    "Transportation Sector": "Transportation",
}
iam_ref = iam_ref.rename(
    index=SECTOR_RENAME_DOWNSCALED,
    level="sector"
)
mask = iam_ref.index.get_level_values("gas") == "N2O"
iam_ref.loc[mask] = iam_ref.loc[mask] / 1000

# %%
difference = combined_df - iam_ref
difference.to_csv(folder_totals / f"{new_stem}_reaggregated-gridded-minus-iam.csv")

# %%
relative = difference/combined_df*100
relative.to_csv(folder_totals / f"{new_stem}_relative-difference.csv")

# %%
#combined_totals = combined_df.groupby(level=["gas"]).sum()

difference_totals = difference.groupby(level=["gas"]).sum()

# %%
sectors = difference.index.get_level_values("sector").unique()

# %%
df_long = (
    combined_df
    .reset_index()
    .melt(
        id_vars=["gas", "sector"],
        var_name="year",
        value_name="emissions"
    )
)

ref_long = (
    iam_ref
    .reset_index()
    .melt(
        id_vars=["gas", "sector"],
        var_name="year",
        value_name="emissions"
    )
)

# make sure year is numeric
df_long["year"] = df_long["year"].astype(int)
ref_long["year"] = ref_long["year"].astype(int)

df_long["variant"] = "gridded"
ref_long["variant"] = "iam (input)"

plot_df = pd.concat([df_long, ref_long], ignore_index=True)

# %%
for gas in CALCULATE_TOTALS_GASES:

    # Filter your data
    data = plot_df[plot_df["gas"] == f"{gas}"]

    # Create a FacetGrid with independent y
    g = sns.FacetGrid(
        data,
        col="sector",
        col_wrap=3,
        height=4,
        aspect=1.6,
        sharey=False  # guaranteed to make y-axis independent
    )
    
    # Map the lineplot
    g.map_dataframe(
        sns.lineplot,
        x="year",
        y="emissions",
        hue="variant",
        style="variant",
        dashes={"gridded": (2, 2), "iam (input)": ""}
    )
    
    # Add legend inside each facet (or outside if desired)
    g.add_legend(title="Variant")
    g.fig.suptitle(f"{gas}", y=1.02)
    
    g.savefig(folder_totals / f"{gas}_{new_stem}_reaggregated-comparison.png")
    plt.show()

# %% [markdown]
# ## 4.3. make sure NMVOC adds up to NMVOCbulk openburning, compare to downscaled

# %%
save_total_emissions_as_csv = True

# %%
if save_total_emissions_as_csv: # TODO: @Jarmo, you may want to introduce a different hook for this in the driver script?
    from concordia.cmip7.utils_plotting import ds_to_annual_emissions_total
    import seaborn as sns
    from concordia.cmip7.CONSTANTS import GASES_ESGF_BB4CMIP_VOC

    SPECIATED_BB4CMIP_VOC = ["NMVOC-" + s for s in GASES_ESGF_BB4CMIP_VOC]
    SPECIATED_BB4CMIP_VOC.append("NMVOCbulk")

    folder_totals = settings.out_path / GRIDDING_VERSION / "check_NMVOC_sums"
    folder_totals.mkdir(parents=True, exist_ok=True)

    areacella = xr.open_dataset(
        Path(settings.gridding_path, "areacella_input4MIPs_emissions_CMIP_CEDS-CMIP-2025-04-18_gn.nc")
    )
    cell_area = areacella["areacella"]

    all_gases_df_list = []

    for file in tqdm((settings.out_path / GRIDDING_VERSION).glob("*.nc"),
                     desc="Calculating total annual emissions from the gridded files"):

        gas_name, var, type_name = return_emission_names(file)

        if gas_name not in SPECIATED_BB4CMIP_VOC:
            continue
        
        print(gas_name)
        
        scen = xr.open_dataset(file)

        da = ds_to_annual_emissions_total(
            gridded_data=scen,
            var_name=var,
            cell_area=cell_area,
            keep_sectors=True
        )

        if isinstance(da, xr.DataArray):
            df = da.to_dataframe(name="emissions").reset_index()
        elif isinstance(da, pd.Series):
            df = da.reset_index(name="emissions")
        else:
            raise TypeError(f"Unexpected type: {type(da)}")

        df["gas"] = gas_name
        df["sector"] = df["sector"].map(SECTOR_DICT_OPENBURNING_DEFAULT)

        # Pivot to wide format: years as columns
        df_wide = df.pivot(index=["gas", "sector"], columns="year", values="emissions")
        all_gases_df_list.append(df_wide)

    
    # Combine all files into one MultiIndex DataFrame
    combined_df = pd.concat(all_gases_df_list).sort_index()

    parts = file.stem.split("_")
    new_stem = "_".join(parts[1:])
    
    combined_df.to_csv(folder_totals / f"{new_stem}_combined-annual-totals.csv")

# %%
# test that the speciated NMVOC species add up to the bulk NMVOC

# drop the bulk from the df
combined_df_filtered = combined_df.loc[combined_df.index.get_level_values("gas") != "NMVOCbulk"]
# add the speciated up by sector
speciated_totals = combined_df_filtered.groupby(level=["sector"]).sum()
# isolate the bulk and process similarly to get df in same format
bulk_totals = combined_df.loc[combined_df.index.get_level_values("gas") == "NMVOCbulk"].groupby(level=["sector"]).sum()

# test that they are equal
pd.testing.assert_frame_equal(speciated_totals, bulk_totals)

# %%
# select NMVOCbulk from downscaled data
downscaled_bulk = downscaled.loc[downscaled.index.get_level_values("gas") == "NMVOCbulk"]
downscaled_bulk_totals = downscaled_bulk.groupby(level=["sector"]).sum()

# reformat for plotting
downscaled_long = (
    downscaled_bulk_totals
    .reset_index()
    .melt(
        id_vars=["sector"],
        var_name="year",
        value_name="emissions"
    )
)

bulk_long = (
    bulk_totals
    .reset_index()
    .melt(
        id_vars=["sector"],
        var_name="year",
        value_name="emissions"
    )
)

# make sure year is numeric
downscaled_long["year"] = downscaled_long["year"].astype(int)
bulk_long["year"] = bulk_long["year"].astype(int)

downscaled_long["variant"] = "downscaled"
bulk_long["variant"] = "gridded"

plot_df = pd.concat([bulk_long, downscaled_long], ignore_index=True)

# %%
gas = "NMVOCbulk"

g = sns.FacetGrid(
    plot_df,
    col="sector",
    col_wrap=2,
    height=4,
    aspect=1.6,
    sharey=False
)
    
g.map_dataframe(
    sns.lineplot,
    x="year",
    y="emissions",
    hue="variant"
)

g.add_legend(title="Variant")
g.fig.suptitle("Openburning NMVOC", y=1.02)

g.savefig(folder_totals / f"{gas}_{new_stem}_reaggregated-comparison.png")
plt.show()

# %% [markdown]
# ## 4.4. make sure anthro VOC adds up to NMVOC-em-anthro, compare to downscaled

# %%
if save_total_emissions_as_csv: # TODO: @Jarmo, you may want to introduce a different hook for this in the driver script?
    
    GASES_VOC = [item.removesuffix("_em_speciated_VOC_anthro") for item in GASES_ESGF_CEDS_VOC]
    GASES_VOC = [item.replace("_", "-") for item in GASES_VOC]

    folder_totals = settings.out_path / GRIDDING_VERSION / "check_VOC_sums"
    folder_totals.mkdir(parents=True, exist_ok=True)

    areacella = xr.open_dataset(
        Path(settings.gridding_path, "areacella_input4MIPs_emissions_CMIP_CEDS-CMIP-2025-04-18_gn.nc")
    )
    cell_area = areacella["areacella"]

    all_gases_df_list = []

    for file in tqdm((settings.out_path / GRIDDING_VERSION).glob("*.nc"),
                     desc="Calculating total annual emissions from the gridded files"):

        gas_name, var, type_name = return_emission_names(file)

        if gas_name not in GASES_VOC:
            continue
        
        print(gas_name)
        
        scen = xr.open_dataset(file)

        da = ds_to_annual_emissions_total(
            gridded_data=scen,
            var_name=var,
            cell_area=cell_area,
            keep_sectors=True
        )

        if isinstance(da, xr.DataArray):
            df = da.to_dataframe(name="emissions").reset_index()
        elif isinstance(da, pd.Series):
            df = da.reset_index(name="emissions")
        else:
            raise TypeError(f"Unexpected type: {type(da)}")

        df["gas"] = gas_name
        df["sector"] = df["sector"].map(SECTOR_DICT_ANTHRO_DEFAULT)

        # Pivot to wide format: years as columns
        df_wide = df.pivot(index=["gas", "sector"], columns="year", values="emissions")
        all_gases_df_list.append(df_wide)

    
    # Combine all files into one MultiIndex DataFrame
    combined_df = pd.concat(all_gases_df_list).sort_index()

    parts = file.stem.split("_")
    new_stem = "_".join(parts[1:])
    
    combined_df.to_csv(folder_totals / f"{new_stem}_combined-annual-totals.csv")

# %%
for file in tqdm((settings.out_path / GRIDDING_VERSION).glob("NMVOC-em-anthro*"),
                     desc="Calculating total annual emissions from the gridded files"):
    
    scen = xr.open_dataset(file)

    da = ds_to_annual_emissions_total(
        gridded_data=scen,
        var_name=var,
        cell_area=cell_area,
        keep_sectors=True
    )

    if isinstance(da, xr.DataArray):
        df = da.to_dataframe(name="emissions").reset_index()
    elif isinstance(da, pd.Series):
        df = da.reset_index(name="emissions")
    else:
        raise TypeError(f"Unexpected type: {type(da)}")

    df["gas"] = gas_name
    df["sector"] = df["sector"].map(SECTOR_DICT_ANTHRO_DEFAULT)

    # Pivot to wide format: years as columns
    df_wide = df.pivot(index=["gas", "sector"], columns="year", values="emissions")

# %%
# test that the speciated NMVOC species add up to the bulk NMVOC

speciated_totals = combined_df.groupby(level=["sector"]).sum()

# isolate the bulk and process similarly to get df in same format
bulk_totals = df_wide.groupby(level=["sector"]).sum()

# test that they are equal
pd.testing.assert_frame_equal(speciated_totals, bulk_totals)

# %%
# select NMVOCbulk from downscaled data
downscaled_bulk = downscaled.loc[downscaled.index.get_level_values("gas") == "NMVOC"]
downscaled_bulk_totals = downscaled_bulk.groupby(level=["sector"]).sum()

SECTOR_RENAME_DOWNSCALED = {
    "Energy Sector": "Energy",
    "Industrial Sector": "Industrial",
    "Residential Commercial Other": "Residential, Commercial, Other",
    "Transportation Sector": "Transportation",
}

downscaled_bulk_totals = downscaled_bulk_totals.rename(index=SECTOR_RENAME_DOWNSCALED, level='sector')

# reformat for plotting
downscaled_long = (
    downscaled_bulk_totals
    .reset_index()
    .melt(
        id_vars=["sector"],
        var_name="year",
        value_name="emissions"
    )
)

bulk_long = (
    speciated_totals
    .reset_index()
    .melt(
        id_vars=["sector"],
        var_name="year",
        value_name="emissions"
    )
)

# make sure year is numeric
downscaled_long["year"] = downscaled_long["year"].astype(int)
bulk_long["year"] = bulk_long["year"].astype(int)

downscaled_long["variant"] = "downscaled"
bulk_long["variant"] = "gridded"

plot_df = pd.concat([bulk_long, downscaled_long], ignore_index=True)

# %%
gas = "NMVOC"

sns.relplot(
    data=plot_df,
    x="year",
    y="emissions",
    hue="variant",
    col="sector",
    col_wrap=3,
    kind="line",
    height=4,
    aspect=1.6,
    facet_kws={"sharey": False}
)

g.savefig(folder_totals / f"{gas}_{new_stem}_reaggregated-comparison.png")
plt.show()

# %% [markdown]
# # END OF POSTPROCESSING

