# -*- coding: utf-8 -*-
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
HISTORY_FILE: str = "cmip7_history_countrylevel_251024.csv"
# Settings
# SETTINGS_FILE: str = "config_cmip7_esgf_v0_alpha.yaml" # was used for preparing for first upload to ESGF
SETTINGS_FILE: str = "config_cmip7_v0-4-0.yaml" # for second ESGF version

# Which scenario to run from the markers
marker_to_run: str = "VL" # options: H, HL, M, ML, L, LN, VL

GRIDDING_VERSION: str | None = None

DO_GRIDDING_ONLY_FOR_THESE_SPECIES: list[str] | None = None # e.g. ["CO2", "Sulfur"]
# DO_GRIDDING_ONLY_FOR_THESE_SPECIES: list[str] | None = ["CO2", "Sulfur"]

# Which parts to run
run_main: bool = True
run_main_gridding: bool = True # if false, we'll stop at only running the downscaling of main
run_anthro_supplemental_voc: bool = False
run_openburning_supplemental_voc: bool = False # not yet implemented, for the future, see PR https://github.com/jkikstra/concordia/pull/14
# run_anthro_supplemental_solidbiofuel: bool = False # not yet implemented, for the future

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
import pandas as pd
import pycountry
from dask.distributed import Client
from pandas_indexing import isin, ismatch, assignlevel, extractlevel
from pandas_indexing.units import set_openscm_registry_as_default
from ptolemy.raster import IndexRaster
import concordia._patches_ptolemy

from aneris import logger
from concordia import (
    RegionMapping,
    VariableDefinitions,
)
from concordia.cmip7 import utils as cmip7_utils # update to cmip7 utils (e.g. for dressing up netcdf)
from concordia.settings import Settings
from concordia.utils import MultiLineFormatter
from concordia.workflow import WorkflowDriver
from concordia.cmip7.CONSTANTS import return_marker_information


# %%
# Scenario information
_, MODEL_SELECTION, SCENARIO_SELECTION, _ = return_marker_information(
    v=SETTINGS_FILE,
    m=marker_to_run
)
if GRIDDING_VERSION is None:
    GRIDDING_VERSION = f"{marker_to_run}" # default to just the marker abbreviation if no versioning is provided
SCENARIO_FILE = f"harmonised-gridding_{MODEL_SELECTION}.csv"

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
print(missing)  # CDR sectors 

expected_sectors_missing_cdr = {
    'Enhanced Weathering', 'BECCS', 'Direct Air Capture', 'Ocean', 'Other CDR'
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
# **NOTE:** this may not work in all IDEs (it doesn't work in VSCode interactive window; workers don't start), but it does work in a standard jupyter lab instance

# %%
client = Client()
# client.register_plugin(DaskSetWorkerLoglevel(logger().getEffectiveLevel()))
client.forward_logging()

# %%
dask.distributed.gc.disable_gc_diagnosis()

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
).persist()
indexraster_region = indexraster.dissolve(
    regionmapping.filter(indexraster.index).data.rename("country")
).persist()

# %%
iam_df.columns

# %%
# TODO: find a better way to test whether all historical data is available for each country (it is known that Guam and Mayotte miss data for some sectors in the used CEDS release)
# # check completeness of historical data
# for c in countries_with_hist_and_gdp_and_regionmapping_data:
#     if len(hist.loc[ismatch(country=c)]) < 120:
#         print(c)

# %%
workflow = WorkflowDriver( 
    # model
    # iam_df, # model
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
    variabledefs,
    # harm_overrides
    harm_overrides,
    # settings
    settings
)

# %% [markdown]
# ## Add some checks on workflow

# %%
# save workflow info in easy-to-vet packets
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
downscaled = workflow.harmonize_and_downscale() # For a 1 scenario, this takes about 50 seconds on Jarmo's DELL laptop.

# %% [markdown]
# ### Export harmonized scenarios

# %%
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
# TODO: (feature) create a similar exporter to the Harmonized class for Downscaled which combines historic and downscaled data (maybe also harmonized?) and translates to iamc

workflow.downscaled.data.to_csv(
    version_path / f"downscaled-only-{settings.version}.csv"
)
print(
    "Countries covered (" + str(len(workflow.downscaled.data.loc[~isin(region="World")].reset_index().country.unique())) + "):"
)
print(workflow.downscaled.data.loc[~isin(region="World")].reset_index().country.unique())

# %%
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
# Total missing data: countries in hist but not in downscaled
in_hist_not_downscaled = hist_countries - downscaled_countries
print("Countries in hist but not in downscaled:")
print(sorted(in_hist_not_downscaled))

missing_emissions = hist.loc[isin(country=list(in_hist_not_downscaled))].groupby(["gas","sector","unit"]).sum().loc[isin(sector='Waste'),2023]
global_emissions = hist.loc[isin(country='World')].groupby(["gas","sector","unit"]).sum().loc[isin(sector='Waste'),2023]
print("In %, what share of global emissions is missing because some smaller territories/countries are not downscaled?")
missing_emissions / global_emissions * 100 # percentage (%) of global emissions that would be missing through these countries

# %% [markdown]
# ## Alternative 1) Run full processing and create netcdf files
#
# Latest test with 1 scenario was 50 minutes on Jarmo's DELL laptop.
# Output files are about 11.4GB for one scenario.

# %%
cmip7_utils.DS_ATTRS

# %%
if run_main_gridding:
    res = workflow.grid(
        template_fn="{{name}}_{activity_id}_emissions_{target_mip}_{institution}-{{model}}-{{scenario}}_{grid_label}_{start_date}-{end_date}.nc".format(
            **cmip7_utils.DS_ATTRS | {"version": settings.version}
        ),
        callback=cmip7_utils.DressUp(version=settings.version),
        directory=version_path,
        skip_exists=True,
    )

# %% [markdown]
# # END OF MAIN CODE

# %% [markdown]
# # Start of SUPPLEMENTAL DATA

# %% [markdown]
# # VOC speciation (CEDS, anthro)
# **NOTE: currently takes long at ~20mins per VOC species ~= 8hrs for all 23 VOC species**


# %%
from dask.utils import SerializableLock
lock = SerializableLock()

# %%
# Load VOC data
from concordia.cmip7.CONSTANTS import GASES_ESGF_CEDS_VOC
import xarray as xr

def load_voc_bulk():

    # load VOC (bulk) scenario file

    # anthro
    voc_anthro = xr.open_dataset(
        # update the file template with:
        # - discussion on GitHub:  https://github.com/CMIP-Data-Request/Harmonised-Public-Consultation/issues/108
        # - proper netCDF handling (see Zeb's 0-3-0 fixes)
        settings.out_path / GRIDDING_VERSION / "{name}_{activity_id}_emissions_{target_mip}_{institution}-{model}-{scenario}_{grid_label}_{start_date}-{end_date}.nc".format(
        name="VOC-em-anthro",
        model=MODEL_SELECTION.replace(" ", "-"),
        scenario=SCENARIO_SELECTION.replace(" ", "-"),
        **cmip7_utils.DS_ATTRS | {"version": settings.version}
    ),
    chunks={},
    lock=lock
    )
    voc_anthro

    return voc_anthro


# %%



# AIR is not required.

# TODO: add openburning (which has entirely different names/formatting)

# %%
# Calculate VOC-speciation data; keep the structure of the VOC (bulk) data

from dask.diagnostics import ProgressBar

PROXY_TIME_RANGE_VOC_CEDS = "2023"

voc_spec_ratios_location = settings.proxy_path / "VOC_speciation"

# loop through all CEDS em-anthro VOC-species from input4MIP files
# 1. load share data
# 2. create an "empty"/"template" dataset as a copy of voc_anthro
# 3. fill with zeroes
# 4. for each sector, 
#   i. do multiplication
#   ii. assign sector value
# 5. Update/set other attributes

if run_anthro_supplemental_voc:
    voc_anthro = load_voc_bulk()
    
    for v in GASES_ESGF_CEDS_VOC:
    # for v in [GASES_ESGF_CEDS_VOC[2]]: # only run one to test
    # for v in GASES_ESGF_CEDS_VOC[0:9]: # run a few to test
        print(f'Reading in shares of {v}')
        # import file 
        voc_share = xr.open_dataset(
            voc_spec_ratios_location / f"{v}_{PROXY_TIME_RANGE_VOC_CEDS}.nc",
            engine="netcdf4",
            chunks={},
            lock=lock
        )

        # create VOC_em speciated
        # Alternative approach using xarray's alignment capabilities
        
        # Create a mapping from voc_anthro sectors to voc_share sectors
        sector_mapping = {
            'Agriculture': 'AGR',
            'Energy': 'ENE',
            'Industrial': 'IND',
            'Transportation': 'TRA',
            'Residential, Commercial, Other': 'RCO',
            'Solvents production and application': 'SLV',
            'Waste': 'WST',
            'International Shipping': 'SHP'
        }
        
        # Rename sectors in voc_anthro to match voc_share sector names where possible
        anthro_to_share_sectors = {v: k for k, v in sector_mapping.items() if v in voc_share.sector.values and k in voc_anthro.sector.values}
        
        # Initialize result with same structure as voc_anthro
        voc_spec = xr.Dataset(
            coords=voc_anthro.coords,
            attrs=voc_anthro.attrs.copy()
        )
        
        # Initialize the data variable with zeros
        voc_spec_data = xr.zeros_like(voc_anthro["VOC_em_anthro"])
        
        # Perform multiplication for matching sectors
        # print(f'Calculations of emissions of {v}')
        for share_sector, anthro_sector in anthro_to_share_sectors.items():
            # Select data from both datasets for matching sectors
            voc_bulk = voc_anthro["VOC_em_anthro"].sel(sector=anthro_sector)
            
            # Get emissions share for this sector and gas
            share_data = voc_share["emissions_share"].sel(
                sector=share_sector,
                gas=voc_share.gas[0]  # Take first gas
            )
            
            # Convert time coordinates to year/month for alignment
            years = voc_bulk.time.dt.year
            months = voc_bulk.time.dt.month
            
            # Find the index of the sector in the coordinate array
            sector_idx = list(voc_anthro.sector.values).index(anthro_sector)
            
            # Perform multiplication for each time step
            for time_idx, time_val in enumerate(voc_bulk.time.values):
                year = years[time_idx].values
                month = months[time_idx].values
                
                # Check if this year/month exists in voc_share
                if year in share_data.year.values and month in share_data.month.values:
                    # Get the share data for this specific year/month
                    share_slice = share_data.sel(year=year, month=month)
                    
                    # Get the bulk VOC data for this time step
                    voc_slice = voc_bulk.isel(time=time_idx)
                    
                    # Multiply and assign to result
                    voc_spec_data[time_idx, :, :, sector_idx] = (voc_slice * share_slice).values
        
        # Add the computed data to the result dataset
        gas_variable_name = voc_share.gas.values[0]

        voc_spec[f"{gas_variable_name}"] = voc_spec_data
        # TODO:
        # - [ ] update long_name of data (follow CEDS long_name) 
        # Add the bounds
        voc_spec['lon_bnds'] = voc_anthro['lon_bnds']
        voc_spec['time_bnds'] = voc_anthro['time_bnds']
        voc_spec['lat_bnds'] = voc_anthro['lat_bnds']
        
        # Update attributes
        voc_spec.attrs['variable_id'] = gas_variable_name
        voc_spec.attrs['title'] = f"Speciated {gas_variable_name} emissions"

        # save out
        print(f'Writing out emissions of {v}')
        outfile = settings.out_path / GRIDDING_VERSION / "{name}_{activity_id}_emissions_{target_mip}_{institution}-{model}-{scenario}_{grid_label}_{start_date}-{end_date}.nc".format(
            name=gas_variable_name.replace("_", "-"),
            model=MODEL_SELECTION.replace(" ", "-"),
            scenario=SCENARIO_SELECTION.replace(" ", "-"),
            **cmip7_utils.DS_ATTRS | {"version": settings.version}
        )

        encoding = {
            gas_variable_name: {
                "zlib": True,
                "complevel": 2
            }
        }
        
        with ProgressBar():
            voc_spec.to_netcdf(outfile, mode="w", encoding=encoding, compute=True)


# %% [markdown]
# # END OF SUPPLEMENTAL DATA CODE


# %% [markdown]
# # ------------------------------------

# %% [markdown]
# # Investigate VOC speciation

# %% [markdown]
# ### Plot VOC shares

# %%
# Add this variable at the top of the plotting section
RUN_INTERACTIVE_PLOTS = False  # Set to True to enable interactive plotting

# %%
# Only run plotting code if explicitly enabled
if RUN_INTERACTIVE_PLOTS:
    # Plot voc_share emissions data
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    import numpy as np

    # Select data for a specific year, month, and sector for plotting
    plot_data = voc_share['emissions_share'].sel(
        year=2023, 
        month=6,  # January
        sector='IND'  # Industrial sector
    ).squeeze()  # Remove singleton dimensions

    # Select data for a specific year, month, and sector for plotting
    plot_data = voc_anthro['VOC_em_anthro'].sel(
        time="2023-06",
        sector='Industrial'  # Industrial sector
    ).squeeze()  # Remove singleton dimensions
else:
    print("Interactive plotting disabled. Set RUN_INTERACTIVE_PLOTS = True to enable.")

# %%
# Only run detailed plotting if enabled
if RUN_INTERACTIVE_PLOTS:
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'VOC Emissions Share: {voc_share.gas.values[0]}', fontsize=16)

    # Plot 1: Raw data
    ax1 = axes[0, 0]
    im1 = plot_data.plot(ax=ax1, cmap='viridis', add_colorbar=False)
    ax1.set_title('Raw Emissions Share (Industrial, June 2023)')

    # Add country borders using a simpler approach
    try:
        import cartopy.feature as cfeature
        import cartopy.crs as ccrs
        
        # Create a new subplot with cartopy projection for this plot
        fig.delaxes(ax1)  # Remove the original axes
        ax1 = fig.add_subplot(2, 2, 1, projection=ccrs.PlateCarree())
        
        # Replot the data on the cartopy axes
        im1 = plot_data.plot(ax=ax1, cmap='viridis', add_colorbar=False, transform=ccrs.PlateCarree())
        ax1.set_title('Raw Emissions Share (Industrial, June 2023)')
        
        # Add country borders and coastlines
        ax1.add_feature(cfeature.BORDERS, linewidth=0.5, color='white', alpha=0.8)
        ax1.add_feature(cfeature.COASTLINE, linewidth=0.5, color='white', alpha=0.8)
        ax1.set_global()
        
    except ImportError:
        print("Cartopy not available - skipping country borders")
        # Fallback: just plot without borders
        im1 = plot_data.plot(ax=ax1, cmap='viridis', add_colorbar=False)
        ax1.set_title('Raw Emissions Share (Industrial, June 2023)')

    plt.colorbar(im1, ax=ax1, shrink=0.8)

    # Plot 2: Log scale for better visibility of small values
    ax2 = axes[0, 1]
    # Add small value to avoid log(0)
    log_data = np.log10(plot_data.where(plot_data > 0) + 1e-10)
    im2 = log_data.plot(ax=ax2, cmap='plasma', add_colorbar=False)
    ax2.set_title('Log10 Emissions Share')
    plt.colorbar(im2, ax=ax2, shrink=0.8)

    # Plot 3: Histogram of values
    ax3 = axes[1, 0]
    values = plot_data.values.flatten()
    values = values[~np.isnan(values)]  # Remove NaN values
    ax3.hist(values, bins=50, alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Emissions Share')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Distribution of Emissions Share Values')
    ax3.set_yscale('log')

    # Plot 4: Summary statistics by sector
    ax4 = axes[1, 1]
    sector_means = []
    sector_names = []
    for sector in voc_share.sector.values:
        sector_data = voc_share['emissions_share'].sel(
            year=2023, month=1, sector=sector
        ).values.flatten()
        sector_data = sector_data[~np.isnan(sector_data)]
        if len(sector_data) > 0:
            sector_means.append(np.mean(sector_data))
            sector_names.append(sector)

    bars = ax4.bar(sector_names, sector_means)
    ax4.set_xlabel('Sector')
    ax4.set_ylabel('Mean Emissions Share')
    ax4.set_title('Mean Emissions Share by Sector (2023, Jan)')
    ax4.tick_params(axis='x', rotation=45)

    # Color bars by value
    if sector_means:  # Only if we have data
        norm = colors.Normalize(min(sector_means), max(sector_means))
        colors_list = plt.cm.viridis(norm(sector_means))
        for bar, color in zip(bars, colors_list):
            bar.set_color(color)

    plt.tight_layout()
    plt.show()

# %%
# Summary statistics - always run these as they're informational, not interactive
if RUN_INTERACTIVE_PLOTS:
    # Print summary statistics
    print(f"\nSummary statistics for {voc_share.gas.values[0]}:")
    print(f"Shape: {voc_share['emissions_share'].shape}")
    print(f"Min value: {voc_share['emissions_share'].min().values:.2e}")
    print(f"Max value: {voc_share['emissions_share'].max().values:.2e}")
    print(f"Mean value: {voc_share['emissions_share'].mean().values:.2e}")
    print(f"Non-zero fraction: {(voc_share['emissions_share'] > 0).sum().values / voc_share['emissions_share'].size:.3f}")

    # Check for grid cells with 100% share
    max_share = voc_share['emissions_share'].max().values
    if max_share >= 0.99:
        print(f"\nMaximum share found: {max_share:.3f}")
        high_share_data = voc_share['emissions_share'].where(voc_share['emissions_share'] >= 0.99)
        if high_share_data.count() > 0:
            print("Grid cells with >99% share found!")


# %% [markdown]
# # Investigate GRIDS

# %% [markdown]
# ### Look at a processed emissions file

# %%
if RUN_INTERACTIVE_PLOTS:
    import xarray as xr

    result_grid = xr.open_dataset(Path("..", "results", 
    "config_cmip7_v0_1_testing_ukesm_remind", 
    "NOx-em-anthro_input4MIPs_emissions_CMIP7_IIASA-REMIND-MAgPIE-3.5-4.10-SSP1---Very-Low-Emissions_gn_202301-210012.nc"))

    # View variable names
    print(result_grid.data_vars)
    # View coordinates
    print(result_grid.coords)
    # Pick the variable (one per nc file)
    print(result_grid['NOx_em_anthro'])
    # What years?
    import numpy as np
    print(np.unique(result_grid.coords['time'].values))

# %%
if RUN_INTERACTIVE_PLOTS:
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors

    # Extract data
    data = result_grid['NOx_em_anthro'].sel(time='2020-01-16 00:00:00', sector='Transportation')

    # Compute 99th percentile
    vmax = np.percentile(data.values, 99.5)

    # Plot with normalization: cap all higher values at the 99th percentile
    norm = colors.Normalize(vmin=data.min(), vmax=vmax)

    # Plot
    plt.figure(figsize=(10, 5))
    data.plot(norm=norm, cmap='viridis')  # Or use any perceptual map: 'plasma', 'inferno', etc.
    plt.title("NOx Emissions (Transportation)")
    plt.show()
else:
    print("Interactive plotting disabled. Set RUN_INTERACTIVE_PLOTS = True to enable NOx grid plots.")
