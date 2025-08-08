# -*- coding: utf-8 -*-
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
# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# # Workflow for CMIP7 ScenarioMIP emissions harmonization 

# %% [markdown]
# ## Specify input scenario data and project settings

# %% [markdown]
# Specify which scenario file to read in

# %%
HISTORY_FILE = "cmip7_history_0022.csv"

# %%
# SCENARIO_FILE = "check_harmonisation_regions_REMIND.csv" # example (ALREADY HARMONIZED) REMIND scenario (used in v0 UKESM testing)
#SCENARIO_FILE = "harmonised-gridding_REMIND-MAgPIE 3.5-4.10.csv" # example (ALREADY HARMONIZED) REMIND scenario (used from 17.07.2026 towards v0_2 UKESM testing)

SCENARIO_FILE = "harmonised-gridding_GCAM 7.1 scenarioMIP.csv"
# SCENARIO_FILE = "scenarios_scenariomip_COFFEE 1.6_SSP2 - Low Overshoot.csv" # example COFFEE scenario
# SCENARIO_FILE = "scenarios_scenariomip_AIM 3.0_SSP2 - Low Emissions.csv" # example AIM scenario
# SCENARIO_FILE = "scenarios_scenariomip_REMIND-MAgPIE 3.5-4.10_SSP2 - Low Emissions.csv" # example REMIND scenario
# SCENARIO_FILE = "scenarios_scenariomip_MESSAGEix-GLOBIOM-GAINS 2.1-M-R12_SSP2 - Low Overshoot.csv" # example MESSAGE scenario
# SCENARIO_FILE = "scenarios_scenariomip_allmodels_2025-03-05-messagegains.csv" # TODO: update later for all models. Location for this file is specified in the yaml file read into the `settings` object later on

#SCENARIO_SELECTION = "SSP1 - Very Low Emissions"
SCENARIO_SELECTION = "SSP3 - High Emissions"

# %% [markdown]
# Specify settings

# %%
# Settings
SETTINGS_FILE = "config_cmip7_v0_testing_ukesm_remind.yaml" 

# versioning
# HARMONIZATION_VERSION = "config_cmip7_v0_testing_remind"
# HARMONIZATION_VERSION = "config_cmip7_v0_testing_aim"
# HARMONIZATION_VERSION = "config_cmip7_v0_testing_ukesm_remind"
# HARMONIZATION_VERSION = "config_cmip7_v0_1_testing_ukesm_remind"
HARMONIZATION_VERSION = "config_cmip7_v0_2_testing_new_proxies"

# %% [markdown]
# ## Importing packages

# %%
import aneris


aneris.__file__

# %%
import concordia


concordia.__file__

# %%
import logging
from pathlib import Path

import dask
import dask.dataframe as dd
import pandas as pd
import pycountry
from dask.distributed import Client
from pandas_indexing import concat, isin, ismatch, semijoin
from pandas_indexing.units import set_openscm_registry_as_default
from ptolemy.raster import IndexRaster

from aneris import logger
from concordia import (
    RegionMapping,
    VariableDefinitions,
)
from concordia.cmip7 import utils as cmip7_utils # update to cmip7 utils (e.g. for dressing up netcdf)
from concordia.settings import Settings
from concordia.utils import MultiLineFormatter, extend_overrides
from concordia.workflow import WorkflowDriver


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
settings = Settings.from_config(version=HARMONIZATION_VERSION,
                                local_config_path=Path(Path.cwd(),
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
# variabledefs.data.loc[
#     isin(sector="Energy Sector")
# ]

# %% [markdown]
# ## Read region definitions (using RegionMapping class)
#

# %%
settings.data_path

# %%
regionmappings = {}

for m, kwargs in settings.regionmappings.items():
    regionmapping = RegionMapping.from_regiondef(**kwargs)
    regionmapping.data = regionmapping.data.pix.aggregate(
        country=settings.country_combinations, agg_func="last"
    )
    regionmappings[m] = regionmapping

# regionmappings

# %% [markdown]
# # IAM: Read and process IAM data

# %%
Path(settings.scenario_path, SCENARIO_FILE)

# %% [markdown]
# ### Read in (currently just 1 scenario)

# %%
# v0_1 (second UKESM round)
# Read in already-harmonized data
iam_df = cmip7_utils.load_data(
    Path(settings.scenario_path, SCENARIO_FILE)
    # Path(settings.scenario_path, "harmonised-gridding_REMIND-MAgPIE 3.5-4.10.csv")
)

# filter only one scenario  
iam_df = cmip7_utils.filter_scenario(iam_df, scenarios=SCENARIO_SELECTION) # TODO: remove this after test code is done

# iam_df[iam_df['variable']=="Emissions|CH4|Energy Sector"]

# %%
IAMC_COLS = ["model", "scenario", "region", "variable", "unit"]
HARMONIZED_YEAR_COLS = [col for col in iam_df.columns if col.isdigit() and settings.base_year <= int(col) <= 2100]

# %%
# # v0 (first UKESM round)
# # keep only relevant columns
# iam_df = iam_df.drop(columns=["stage"])[(IAMC_COLS + HARMONIZED_YEAR_COLS)]

# v0_1 (second UKESM round)
# keep only relevant columns
iam_df = iam_df[(IAMC_COLS + HARMONIZED_YEAR_COLS)]
# iam_df

# %% [markdown]
# ### Process (using pix - formatting)

# %%
from pandas_indexing import extractlevel
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
cmip7_utils.save_data(df = iam_df.reset_index(), output_path = str(Path(version_path, "scenarios_processed.csv" )))

# %% [markdown]
# # History: Read and process historical data
#

# %%
hist = (
    pd.read_csv(settings.history_path / HISTORY_FILE)
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
#hist

# %% [markdown]
# # Read Harmonization Overrides

# %% [markdown]
# NOTE: should be handled already before, as the emissions trajectories have already been harmonised

# %%
settings.scenario_path

# %%
harm_overrides = pd.read_excel(
    settings.scenario_path / "harmonization_overrides.xlsx", # placeholder for now, should be empty as already harmonized.
    index_col=list(range(3)),
).method
# harm_overrides

# %%
harm_overrides = extend_overrides(
    harm_overrides,
    "constant_ratio",
    sector=[
        f"{sec} Burning"
        for sec in ["Agricultural Waste", "Forest", "Grassland", "Peat"]
    ],
    variables=variabledefs.data.index,
    regionmappings=regionmappings,
    model_baseyear=iam_df[settings.base_year],
)

# %% [markdown]
# # Prepare GDP proxy
#
# Read in different GDP scenarios for SSP1 to SSP5 from SSP DB.
#

# %%
settings.scenario_path

# %%
# TODO: (bug) resolve 0 values in model scenario data for historical

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
print(len(countries_with_gdp_data))
print(len(countries_with_hist_data))
print(len(countries_with_regionmapping))
print(countries_with_hist_and_gdp_and_regionmapping_data)

def select_only_countries_with_all_info(df,
                                        countries=countries_with_hist_and_gdp_and_regionmapping_data):
    df = (
        df
        .loc[
            isin(
                country=countries
            )
        ]
    )
    
    return df


# %% [markdown]
# # Set up technical bits for the workflow

# %%
client = Client()
# client.register_plugin(DaskSetWorkerLoglevel(logger().getEffectiveLevel()))
client.forward_logging()

# %%
dask.distributed.gc.disable_gc_diagnosis()

# %% [markdown]
# # Define workflow

# %%
# TODO: 
# - [ ] make this into a dataframe, and loop over models? --> right now the below section only works for 1 model at a time.

(model_name,) = iam_df.pix.unique("model")
regionmapping = regionmappings[model_name]

# scens_iam_wide.pix.unique("model")

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
# check completeness of historical data
for c in countries_with_hist_and_gdp_and_regionmapping_data:
    if len(hist.loc[ismatch(country=c)]) < 120:
        print(c)

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
workflow.save_info(path = Path("..", "data", "compare_wfd_inputs"), prefix=settings.version)

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

# %% [markdown]
# ## Alternative 1) Run full processing and create netcdf files
#
# Latest test with 1 scenario was 25 minutes on Jarmo's DELL laptop.
# Output files are nearly 6GB for one scenario.

# %%
cmip7_utils.DS_ATTRS

# %%
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
# # ------------------------------------

# %% [markdown]
# # Investigate GRIDS

# %% [markdown]
# ### Look at a processed emissions file

# %%
import xarray as xr

# %%
result_grid = xr.open_dataset(Path("..", "results", 
"config_cmip7_v0_1_testing_ukesm_remind", 
"NOx-em-anthro_input4MIPs_emissions_CMIP7_IIASA-REMIND-MAgPIE-3.5-4.10-SSP1---Very-Low-Emissions_gn_202301-210012.nc"))

# %%
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
