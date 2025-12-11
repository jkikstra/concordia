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
HISTORY_FILE: str = "country-history_202511261223_202511040855_202512032146_202512021030_7e32405ade790677a6022ff498395bff00d9792d.csv"
# Settings
# SETTINGS_FILE: str = "config_cmip7_esgf_v0_alpha.yaml" # was used for preparing for first upload to ESGF
SETTINGS_FILE: str = "config_cmip7_v0-4-0.yaml" # for second ESGF version
VERSION_ESGF: str = "1-0-x" # for second ESGF version

# Which scenario to run from the markers
marker_to_run: str = "VL" # options: H, HL, M, ML, L, LN, VL

GRIDDING_VERSION: str | None = None


# Which parts to run
run_main: bool = True # skips downscaling and the saving out of data of the main workflow; can still run supplemental workflows with this set to False
run_main_gridding: bool = True # if false, we'll not run the main gridding workflow
SKIP_EXISTING_MAIN_WORKFLOW_FILES: bool = True # if True, it won't reproduce files already on your disk
run_anthro_supplemental_voc: bool = False
run_openburning_supplemental_voc: bool = False
run_openburning_h2: bool = True # produced based on openburning_co
# run_anthro_supplemental_solidbiofuel: bool = False # not yet implemented, for the future
run_spatial_harmonisation: bool = True # provides spatial harmonization with CEDS anthro in 2023 (requires having raw CEDS files locally)

# main: files to produce (species, sector)
DO_GRIDDING_ONLY_FOR_THESE_SPECIES: list[str] | None = None # e.g. ["CO2", "Sulfur"]
DO_GRIDDING_ONLY_FOR_THESE_SECTORS: list[str] | None = None # all: ['anthro', 'openburning', 'AIR_anthro']
# supplemental: VOC files to produce
# - anthro
DO_VOC_SPECIATION_ANTHRO_ONLY_FOR_THESE_SPECIES: list[str] | None = None # e.g. ["VOC01_alcohols_em_speciated_VOC_anthro"]
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
from concordia.cmip7.CONSTANTS import return_marker_information
from concordia.cmip7.dask_setup_alternative import setup_dask_client # to enable running with dask also from VSCode Interactive Window


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
# ## Alternative 1) Run full processing and create netcdf files
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
        callback=cmip7_utils.DressUp(version=settings.version, marker_scenario_name=experiment_name),
        directory=version_path,
        skip_exists=SKIP_EXISTING_MAIN_WORKFLOW_FILES,
    )

# %% [markdown]
# # END OF MAIN CODE

# %% [markdown]
# # Start of H2 openburning data
# Usually takes <2mins for 1 scenario

# %%
import xarray as xr
import numpy as np

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
    
    print(co_openburning)
    print(h2_translation)


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
                h2_openburning_data[time_idx, :, :, sector_idx] = (co_slice * translation_slice).values
                
                # Assert that the sectors all align, ignoring dtype
                assert h2_openburning_data[time_idx, :, :, sector_idx].sector.values == co_slice.sector.values
                assert h2_openburning_data[time_idx, :, :, sector_idx].sector.values == translation_slice.sector.values


    # Add the computed data to the result dataset
    gas_variable_name = "H2_em_openburning"
    h2_openburning[gas_variable_name] = h2_openburning_data

    # TODO:
    # - [ ] update long_name of data (follow CEDS long_name) 
    # Add the bounds
    h2_openburning['lon_bnds'] = co_openburning['lon_bnds']
    h2_openburning['time_bnds'] = co_openburning['time_bnds']
    h2_openburning['lat_bnds'] = co_openburning['lat_bnds']

    # Update attributes
    h2_openburning.attrs['variable_id'] = gas_variable_name
    h2_openburning.attrs['title'] = f"Speciated {gas_variable_name} emissions"

    # save out
    print('Writing out H2 openburning emissions')
    outfile = settings.out_path / GRIDDING_VERSION / f"H2-em-openburning_{FILE_NAME_ENDING}"

    encoding = {
        gas_variable_name: {
            "zlib": True,
            "complevel": 2
        }
    }
    h2_openburning.to_netcdf(outfile, mode="w", encoding=encoding, compute=True)



# %% [markdown]
# # Start of SUPPLEMENTAL DATA

# %% [markdown]
# # VOC speciation
# **NOTE: currently takes quite long, especially anthro VOC speciation


# %%
from dask.utils import SerializableLock
lock = SerializableLock()

# %%
# Load VOC data
from concordia.cmip7.CONSTANTS import GASES_ESGF_CEDS_VOC, GASES_ESGF_BB4CMIP_VOC
from concordia.cmip7.utils import scenario_name_prefix

def load_voc_bulk(type="anthro"):

    # load VOC (bulk) scenario file
    if type=="anthro":

        # anthro
        voc_anthro = xr.open_dataset(
            # update the file template with:
            # - discussion on GitHub:  https://github.com/CMIP-Data-Request/Harmonised-Public-Consultation/issues/108
            # - proper netCDF handling (see Zeb's 0-3-0 fixes)
            settings.out_path / GRIDDING_VERSION / "{name}_{activity_id}_emissions_{target_mip}_{institution_id}-{scenario}-{version}_{grid_label}_{start_date}-{end_date}.nc".format(
            name="NMVOC-em-anthro",
            scenario=scenario_name_prefix(marker_to_run),
            **cmip7_utils.DS_ATTRS | {"version": VERSION_ESGF}
        ),
        chunks={},
        lock=lock
        )

        return voc_anthro
    
    if type=="openburning":

        # openburning
        voc_openburning = xr.open_dataset(
            # update the file template with:
            # - discussion on GitHub:  https://github.com/CMIP-Data-Request/Harmonised-Public-Consultation/issues/108
            # - proper netCDF handling (see Zeb's 0-3-0 fixes)
            # settings.out_path / GRIDDING_VERSION / "{name}_{activity_id}_emissions_{target_mip}_{institution_id}-{scenario}-{version}_{grid_label}_{start_date}-{end_date}.nc".format( # actual
            settings.out_path / GRIDDING_VERSION / "{name}_{activity_id}_emissions_{target_mip}_{institution_id}-{scenario}_{grid_label}_{start_date}-{end_date}.nc".format( # remove this after tests
            name="NMVOC-em-openburning",
            scenario=scenario_name_prefix(marker_to_run), # follow scenarioMIP paper (e.g., esm-scen7-h)
            **cmip7_utils.DS_ATTRS | {"version": VERSION_ESGF}
        ),
        chunks={},
        lock=lock
        )

        return voc_openburning



# %%
# AIR (anthro) is not required.


# %% [markdown]
# # VOC speciation (BB4CMIP, openburnig)
# **NOTE: currently takes long at ~5mins per VOC species, around 2hrs for all 25 VOC species**

# %%
# Calculate VOC-speciation data; keep the structure of the VOC (bulk) data

from dask.diagnostics import ProgressBar

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

    for v in GASES_ESGF_BB4CMIP_VOC:
        print(f'Reading in shares of {v}')
        # import file
        voc_share = xr.open_dataset(
            # using VOC shares as produced in `notebooks\cmip7\prep_proxyfuture-openburning-from-dres-cmip7-esgf-VOCspeciation.py`
            voc_spec_ratios_location_openburning / f"{v}_other_voc_em_speciated_NMVOC_openburning_{PROXY_TIME_RANGE_VOC_BB4CMIP}.nc",
            engine="netcdf4",
            chunks={},
            lock=lock
        )

        # create VOC_em speciated
        # approach using xarray's alignment capabilities
        
        # Create a mapping from voc_anthro sectors to voc_share sectors
        sector_mapping = {
            'Agricultural Waste Burning': 'AWB',
            'Peat Burning': 'PEAT',
            'Grassland Burning': 'GRSB',
            'Forest Burning': 'FRTB'
        }

        # Rename sectors in voc_anthro to match voc_share sector names where possible
        openburning_to_share_sectors = {v: k for k, v in sector_mapping.items() if v in voc_share.sector.values and k in voc_openburning.sector.values}

        # Initialize result with same structure as voc_openburning
        voc_spec = xr.Dataset(
            coords=voc_openburning.coords,
            attrs=voc_openburning.attrs.copy()
        )
        
        # Initialize the data variable with zeros
        voc_spec_data = xr.zeros_like(voc_openburning["VOC_em_openburning"])

        # Perform multiplication for matching sectors
        # print(f'Calculations of emissions of {v}')
        for share_sector, openburning_sector in openburning_to_share_sectors.items():
            # Select data from both datasets for matching sectors
            voc_bulk = voc_openburning["VOC_em_openburning"].sel(sector=openburning_sector)
            
            # Get emissions share for this sector and gas
            share_data = voc_share["emissions_share"].sel(
                sector=share_sector,
                gas=voc_share.gas[0]  # Take first (and only) gas
            )
            
            # Convert time coordinates to year/month for alignment
            years = voc_bulk.time.dt.year
            months = voc_bulk.time.dt.month
            
            # Find the index of the sector in the coordinate array
            sector_idx = list(voc_openburning.sector.values).index(openburning_sector)
            
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
        voc_spec['lon_bnds'] = voc_openburning['lon_bnds']
        voc_spec['time_bnds'] = voc_openburning['time_bnds']
        voc_spec['lat_bnds'] = voc_openburning['lat_bnds']
        
        # Update attributes
        voc_spec.attrs['variable_id'] = gas_variable_name
        voc_spec.attrs['title'] = f"Speciated {gas_variable_name} emissions"

        # save out
        print(f'Writing out emissions of {v}')
        outfile = settings.out_path / GRIDDING_VERSION / "{name}_{activity_id}_emissions_{target_mip}_{institution_id}-{scenario}-{version}_{grid_label}_{start_date}-{end_date}.nc".format(
            name=gas_variable_name.replace("_", "-"),
            scenario=scenario_name_prefix(marker_to_run),
            **cmip7_utils.DS_ATTRS | {"version": VERSION_ESGF}
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
# ## VOC speciation (CEDS, anthro)
# **NOTE: runtime ~20mins per VOC species ~= 8hrs for all 23 VOC species**


# %% 
# TODO:
# - [ ] speed up this loop
if run_anthro_supplemental_voc:
    voc_anthro = load_voc_bulk(type="anthro")
    
    if DO_VOC_SPECIATION_ANTHRO_ONLY_FOR_THESE_SPECIES is None:
        DO_VOC_SPECIATION_ANTHRO_ONLY_FOR_THESE_SPECIES = GASES_ESGF_CEDS_VOC # by default, run all
    for v in DO_VOC_SPECIATION_ANTHRO_ONLY_FOR_THESE_SPECIES: # all take about ~6hours for 1 scenario; could consider making this part of the driver parameters
        print(f'Reading in shares of {v}')
        # import file 
        voc_share = xr.open_dataset(
            # using VOC shares as produced in `notebooks\cmip7\prep_proxyfuture-anthro-from-ceds-cmip7-esgf-VOCspeciation.py`
            voc_spec_ratios_location_anthro / f"{v}_{PROXY_TIME_RANGE_VOC_CEDS}.nc",
            engine="netcdf4",
            chunks={},
            lock=lock
        )

        # create VOC_em speciated
        # approach using xarray's alignment capabilities
        
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
        voc_spec_data = xr.zeros_like(voc_anthro["NMVOC_em_anthro"])
        
        # Perform multiplication for matching sectors
        # print(f'Calculations of emissions of {v}')
        for share_sector, anthro_sector in anthro_to_share_sectors.items():
            # Select data from both datasets for matching sectors
            voc_bulk = voc_anthro["NMVOC_em_anthro"].sel(sector=anthro_sector)
            
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
        outfile = settings.out_path / GRIDDING_VERSION / "{name}_{activity_id}_emissions_{target_mip}_{institution_id}-{scenario}-{version}_{grid_label}_{start_date}-{end_date}.nc".format(
            name=gas_variable_name.replace("_", "-"),
            scenario=f"scen-{marker_to_run.lower()}",
            **cmip7_utils.DS_ATTRS | {"version": VERSION_ESGF}
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
# # START OF POSTPROCESSING
# ## 1. Spatial Harmonization
# NOTE: runtime is about ~3 mins per file


# %% [markdown]
# # Start of Post-processing: pattern harmonisation

# COPY functionality mostly from "notebooks\cmip7\workflow-postprocess_anthro-pattern-harmonisation.py"
# TODO:
# - [x] move some code into src
# - [x] apply here in the main workflow, for ease of use
# - [ ] ensure it runs on VOC speciation too (naming?)
# - [x] ensure to not delete all Attributes data
#

# %%
# imports for spatial harmonization
from concordia.cmip7.utils import calculate_ratio, return_nc_output_files_main_voc
import xarray as xr
from tqdm import tqdm

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



# %%  editable=true slideshow={"slide_type": ""} tags=["parameters"]
# parameters for spatial harmonization (beyond what is above)

ceds_data_location: Path = Path(settings.gridding_path, "esgf", "ceds", "CMIP7_anthro")
ceds_data_location_voc: Path = Path(settings.gridding_path, "esgf", "ceds", "CMIP7_anthro_VOC")

# %% 
# helper functions
def _what_emissions_variable_type(file, files_main=[], files_voc=[]):
    if file in files_main:
        type = "em_anthro"
    elif file in files_voc:
        type = "em_speciated_VOC_anthro"
    return type

# %%
# derived locations
# gridded_data_location = settings.out_path / HARMONIZATION_VERSION # is standard `settings.out_path / GRIDDING_VERSION``
# weighted_data_location = settings.out_path / HARMONIZATION_VERSION / "weighted" # TODO: consider overwriting the old/original file instead. 
# weighted_data_location.mkdir(parents=True, exist_ok=True)


from concordia.cmip7.CONSTANTS import PROXY_YEARS, find_voc_data_variable_string
from concordia.cmip7.utils import SECTOR_ORDERING_GAS, SECTOR_ORDERING_DEFAULT, SECTOR_DICT_ANTHRO_DEFAULT, SECTOR_DICT_ANTHRO_CO2_SCENARIO
years = [year for year in PROXY_YEARS if year >= settings.base_year] # all years, but not 2022 (before 2023); which should come directly from CEDS anthro (and CEDS AIR)

def _spatial_harmonisation(file, match, cell_area):
    
    # step 1: find ratio grid between baseyear historical(CEDS) and baseyear scenario(gridded)
    # open datasets (no dask)
    ceds = xr.open_dataset(match)
    gridded = xr.open_dataset(file)

    try:
        # variable name
        if file in files_main:
            var = f"{gas}_{type}"
        if file in files_voc:
            var = find_voc_data_variable_string(gas)

        # rename sectors; from numbers to full names
        ceds = ceds.assign_coords(sector=pd.Series(ceds["sector"].values).map(SECTOR_DICT_ANTHRO_DEFAULT).values)
        reference = ceds.where(ceds.time.dt.year == 2023, drop=True)
        gridded_23 = gridded.where(gridded.time.dt.year == 2023, drop=True)

        # calculate relative difference (vectorized)
        pct_diff23 = calculate_ratio(reference, gridded_23, gas)
        weights = pct_diff23.to_dataset(name=var) # all 1 if no adjustment needed, otherwise most gridpoints close to 1 but not exactly 1

        # expand weights to all years
        n_repeat = gridded.sizes["time"] // weights.sizes["time"]
        weights_exp = xr.concat([weights] * n_repeat, dim="time")
        weights_exp = weights_exp.assign_coords(time=gridded.time)

        # apply weights (= raw ratios)
        weighted = gridded * weights_exp

        # replace sectors we don't want weighted; because it is not in CEDS
        sectors_to_keep = ['Other Capture and Removal'] # Note: previously we also had International Shipping here, but in the case that there's a small (global) discrepancy between CEDS and scenario, it is worthwhile addressing that here too.
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

        # 2.2.
        global_scalar = xr.where(weighted_global != 0,
                                gridded_global / weighted_global,
                                0)

        # 2.3
        emissions_harmonised = weighted * global_scalar

    finally:
        # Close datasets to release file locks
        if 'ceds' in locals():
            ceds.close()
        if 'gridded' in locals():
            gridded.close()
        if 'weighted' in locals():
            weighted.close()
        if 'areacella' in locals():
            areacella.close()
    
    return emissions_harmonised, var


# run the spatial harmonization
if run_spatial_harmonisation:
    print('run spatial harmonization')

    # files that are produced above, that may need correction
    # TODO:
    # - [ ] make this more robust to potentially running a second time; e.g. only if not already run, and only if in the DO_GRIDDING_ONLY_FOR_THESE_SPECIES?
    files_main, files_voc = return_nc_output_files_main_voc(gridded_data_location=settings.out_path / GRIDDING_VERSION)


    # areas of gridcells for calculatings totals
    areacella = xr.open_dataset(Path(settings.gridding_path, "areacella_input4MIPs_emissions_CMIP_CEDS-CMIP-2025-04-18_gn.nc"))
    cell_area = areacella["areacella"]

    for file in tqdm(files_main + files_voc, desc="Processing files"): # all
        gas = file.name.split("-")[0]
        type = _what_emissions_variable_type(file, files_main, files_voc)
        outfile = file

        print(f'Processing {gas}')

        # match reference file: check whether there's a raw CEDS history file on your system to harmonise against
        # all_files = 
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


        emissions_harmonised, var = _spatial_harmonisation(file=file, match=match, cell_area=cell_area)

        # Load original gridded dataset to copy attributes
        gridded_original = xr.open_dataset(file)
        copy_attributes(source=gridded_original, 
                        target=emissions_harmonised)
        gridded_original.close()

        # remove old file (from previous loop in processing)
        outfile.unlink(missing_ok=True)
        # save weighted dataset (no dask)
        encoding = {var: {"zlib": True, "complevel": 2}}

        # TODO:
        # - [ ] this used to be `weighted.to_netcdf()` - are we sure that the change to `emissions_harmonised` is correct
        emissions_harmonised.to_netcdf(outfile, encoding=encoding)

        emissions_harmonised.close()


# %% [markdown]
# # END OF POSTPROCESSING

