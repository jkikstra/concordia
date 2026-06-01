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
# # Workflow for CMIP7 ScenarioMIP emissions harmonization — post-2100 extensions
#
# End-to-end processing of long-term (2100–2500) emissions extensions for CMIP7
# ScenarioMIP. Runs one scenario marker at a time; drive multiple markers from a
# papermill/subprocess wrapper.
#
# **Processing stages:**
#
# | Step | Section | Notes |
# |------|---------|-------|
# | 1 | [Configuration](#step-1-configuration) | Marker, version, which sub-workflows to run |
# | 2 | [Imports & settings](#step-2-imports) | Packages + settings YAML |
# | 3 | [Read definitions](#step-4-read-definitions) | Variable defs, region mappings |
# | 4 | [Read history](#step-5-history--read-and-process-historical-data) | Country-level CEDS/BB4CMIP history through 2022 |
# | 5 | [Read IAM data](#step-6-iam--read-and-process-scenario-data) | Extended harmonized scenario trajectories |
# | 6 | [GDP proxy](#step-6c-prepare-gdp-proxy) | SSP-based GDP for downscaling |
# | 7 | [Coverage checks](#step-7-coverage-checks) | Country, sector, and harmonization consistency |
# | 8 | [Harmonize & downscale](#step-9-harmonize-and-downscale) | aneris-based downscaling to country level |
# | 9 | [Grid](#step-10-grid--create-netcdf-files) | concordia `workflow.grid()` → NetCDF per species/sector |
# | 10 | [Post-processing](#step-11-post-processing--spatial-harmonization) | Spatial harmonization with CEDS 2023; area file |
# | 11 | [H2 openburning](#step-12-h2-openburning) | Derived H2 from openburning CO |
# | 12 | [Supplemental VOC](#step-13-supplemental-data--voc-speciation) | VOC speciation for anthro and openburning |
# | 13 | [QC & diagnostics](#step-14-qc-diagnostics--plots-and-consistency-checks) | Maps, timeseries comparisons, NMVOC consistency |
#
# **Note:** currently built for running one scenario at a time.

# %% [markdown]
# ## Specify input scenario data and project settings
# **Note:** these options below can also be changed and driven from a driver script. 
# ## Step 1: Configuration
# ### *Stage 1/13* · [↑ overview](#workflow-for-cmip7-scenariomip-emissions-harmonization--post-2100-extensions)
# **Note:** these options below can also be changed and driven from a driver script.

# %% editable=true slideshow={"slide_type": ""} tags=["parameters"]
# Settings
# SETTINGS_FILE: str = "config_cmip7_esgf_v0_alpha.yaml" # was used for preparing for first upload to ESGF
SETTINGS_FILE: str = "config_cmip7_v0-4-0-EXT.yaml"
VERSION_ESGF: str = "1-1-1" # for extensions

# Which scenario to run from the markers
marker_to_run: str = "m" # options: h, hl, m, ml, l, ln, vl
marker_name: str = f"{marker_to_run}-ext"
HISTORY_FILE: str = f"downscaled-only-{marker_to_run}_{VERSION_ESGF}.csv"

# What folder to save this run in
# GRIDDING_VERSION: str | None = None
GRIDDING_VERSION: str | None = f"{marker_name}_{VERSION_ESGF}"
GRIDDING_HISTORY: str | None = f"{marker_to_run}_{VERSION_ESGF}"

# Where the downscaled data is stored (used for reading the downscaled historical data, and also as input for the extensions gridding workflow)
from pathlib import Path
JARMO_PATH = "/Users/jarmo/Library/CloudStorage/OneDrive-SharedLibraries-IIASA/ECE.prog - Documents/Projects/CMIP7/IAM Data Processing/Shared emission fields data/"
ANNIKA_PATH = "/Users/hoegner/GitHub/concordia/results"
USE_PATH = JARMO_PATH
LOCATION_DOWNSCALED: Path = Path(USE_PATH + "/v1_1/all_downscaled_markers_1-1-1")
LOCATION_CMIP7_HISTORY: Path = Path(USE_PATH + "/v1_1/workflow_history_files")
# Fast-track gridded outputs (already-final v1_1 vl_1-1-1 files ending at 2100-12). Used to
# anchor the extension at 2100 — see the `run_2100_alignment_to_fasttrack` block below.
LOCATION_FASTTRACK_GRIDDED: Path = Path(USE_PATH + "/v1_1")

# Which parts to run
run_main: bool = True # skips downscaling and the saving out of data of the main workflow; can still run supplemental workflows with this set to False
run_main_gridding: bool = True # if false, we'll not run the main gridding workflow
SKIP_EXISTING_MAIN_WORKFLOW_FILES: bool = False # if True, it won't reproduce files already on your disk
# NOTE: the CEDS-2023 spatial harmonisation block was removed from this workflow — it was
# fast-track-legacy code that depended on 2023 historical data the extension doesn't have,
# and the fade-to-zero-by-2050 made the correction identically zero for all extension years
# anyway. Spatial alignment is now handled by the 2100-anchor block at the bottom of the
# file, which inherits fast-track's already-CEDS-2023-harmonised spatial pattern at 2100.
run_anthro_timeseries_correction: bool = True
run_AIR_anthro_timeseries_correction: bool = True
run_openburning_timeseries_correction: bool = True

# EXTENSIONS:
# 2100 alignment to fast-track: enforce that the extension's 2100 gridded values match the
# fast-track 2100 gridded values via a per-cell, per-month additive offset that fades linearly
# from year FADE_ANCHOR_YEAR (full correction) to FADE_CONVERGENCE_YEAR (zero correction).
# The 2100 timestep is then dropped from the output (fast-track owns 2100).
run_2100_alignment_to_fasttrack: bool = True
FADE_ANCHOR_YEAR: int = 2100        # year where extension is forced to equal fast-track
FADE_CONVERGENCE_YEAR: int = 2150   # year at which the additive correction has decayed to zero
DROP_ANCHOR_TIMESTEP: bool = True   # drop the FADE_ANCHOR_YEAR (=2100) timestep from output
# Diagnostic: persist the RAW (pre-correction) extension-2100 grids alongside the
# fast-track 2100 grids, so we can verify how well the extension naturally lines up with
# the standard scenario projection at 2100 BEFORE the additive offset forces them equal
# and BEFORE 2100 is dropped. Produces per-file: a diagnostic netCDF (ext_2100_raw,
# ft_2100, diff), enriched spatial-agreement metrics in the step-0 CSV, and PNG plots.
# Purely a verification artifact — does NOT change the main ESGF output.
run_2100_alignment_diagnostic: bool = True

# SUPPLEMENTAL WORKFLOWS
run_openburning_h2: bool = True # produced based on openburning_co
run_anthro_supplemental_voc: bool = True
run_openburning_supplemental_voc: bool = True

# run_anthro_supplemental_solidbiofuel: bool = False # not yet implemented, for the future

# main: files to produce (species, sector)
DO_GRIDDING_ONLY_FOR_THESE_SPECIES: list[str] | None = None # e.g. ["CO2", "SO2"]
# DO_GRIDDING_ONLY_FOR_THESE_SPECIES: list[str] | None = ["CO", "NMVOC", "NMVOCbulk"] # e.g. ["CO2", "SO2"]
DO_GRIDDING_ONLY_FOR_THESE_SECTORS: list[str] | None = None # all: ['anthro', 'openburning', 'AIR_anthro']
# supplemental: VOC files to produce
# - anthro
DO_VOC_SPECIATION_ANTHRO_ONLY_FOR_THESE_SPECIES: list[str] | None = None # e.g. ["VOC01_alcohols_em_speciated_VOC_anthro"]
# - openburning
DO_VOC_SPECIATION_OPENBURNING_ONLY_FOR_THESE_SPECIES: list[str] | None = None # e.g. ["C10H16"]
# DO_VOC_SPECIATION_ANTHRO_ONLY_FOR_THESE_SPECIES = ["VOC01_alcohols_em_speciated_VOC_anthro"]
# DO_VOC_SPECIATION_OPENBURNING_ONLY_FOR_THESE_SPECIES = ["C10H16"]

# %%
# validate that we're receiving what we're expecting
print(f"\n\nGRIDDING_VERSION received: {GRIDDING_VERSION}\n\n")
print(f"\n\nDO_GRIDDING_ONLY_FOR_THESE_SPECIES received: {DO_GRIDDING_ONLY_FOR_THESE_SPECIES}\n\n")

# %% [markdown]
# ## Step 2: Imports
# ### *Stage 2/13* · [↑ overview](#workflow-for-cmip7-scenariomip-emissions-harmonization--post-2100-extensions)

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
from concordia.cmip7 import utils_EXT as cmip7_utils # update to cmip7 utils (e.g. for dressing up netcdf)
from concordia.settings import Settings
from concordia.utils import MultiLineFormatter
from concordia.workflow import WorkflowDriver
from concordia.cmip7.CONSTANTS import return_marker_information, PROXY_YEARS, find_voc_data_variable_string, GASES_ESGF_CEDS, GASES_ESGF_BB4CMIP, GASES_ESGF_CEDS_VOC, GASES_ESGF_BB4CMIP_VOC
from concordia.cmip7.dask_setup_alternative import setup_dask_client # to enable running with dask also from VSCode Interactive Window
from concordia.cmip7.utils import calculate_ratio, return_nc_output_files_main_voc, SECTOR_ORDERING_GAS, SECTOR_ORDERING_DEFAULT, SECTOR_DICT_ANTHRO_DEFAULT, SECTOR_DICT_ANTHRO_CO2_SCENARIO, reorder_dimensions, add_file_global_sum_totals_attrs, SECTOR_DICT_OPENBURNING_DEFAULT, SECTOR_DICT_OPENBURNING_DEFAULT_FLIPPED, SECTOR_DICT_ANTHRO_CO2_SCENARIO_FLIPPED, add_lon_lat_bounds, add_time_bounds, clean_var, DATA_HANDLES
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

SCENARIO_FILE = f"extensions_full_emissions_timeseries_2023_2500.csv"

# %% editable=true slideshow={"slide_type": ""} tags=["parameters"]
# filename template
FILE_NAME_ENDING: str | None = cmip7_utils.filename_for_esgf(marker=marker_name, version=VERSION_ESGF)

print(f"Producing experiment: {FILE_NAME_ENDING}")

# %% [markdown]
# Load unit registry from openSCM for translating units (e.g., to and from CO2eq)

# %%
ur = set_openscm_registry_as_default()

# %% [markdown]
# # Step 3: Read Settings
# ## *Stage 2/13 — Imports & settings* · [↑ overview](#workflow-for-cmip7-scenariomip-emissions-harmonization--post-2100-extensions)
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
                                local_config_path=Path(HERE, SETTINGS_FILE),
                                base_path=HERE)

settings.base_year

# %%
ceds_data_location = settings.postprocess_path / "CMIP7_anthro"
ceds_data_location_voc = settings.postprocess_path / "CMIP7_anthro_VOC"
ceds_data_location_AIR = settings.postprocess_path / "CMIP7_AIR"

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
# Set logger (uses setting)

# %%
settings.out_path.mkdir(parents=True, exist_ok=True)
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
    if settings.country_combinations:  # only aggregate if not empty
        regionmapping.data = regionmapping.data.pix.aggregate(
            country=settings.country_combinations, agg_func="last"
        )
    # Ensure no duplicate country entries (pix.aggregate deduplicates when
    # country_combinations is set, but without it the raw CSV may have duplicates)
    if regionmapping.data.index.duplicated().any():
        n_dups = regionmapping.data.index.duplicated().sum()
        dups = regionmapping.data.index[regionmapping.data.index.duplicated(keep=False)]
        print(f"⚠️  {m}: Dropping {n_dups} duplicate country entries from regionmapping")
        print(f"   Duplicate countries: {sorted(set(dups.tolist()))}")
        regionmapping.data = regionmapping.data[~regionmapping.data.index.duplicated(keep='last')]
    regionmappings[m] = regionmapping


# %% [markdown]
# # History: Read and process historical data
#

# %%
scenario_hist = (
    pd.read_csv(LOCATION_DOWNSCALED / HISTORY_FILE) # like "cmip7_history_countrylevel_250721.csv" 
    .drop(columns=['model', 'region', 'scenario', 'method'])
)
scenario_hist = scenario_hist.set_index(['country', 'gas', 'sector', 'unit'])
scenario_hist = scenario_hist.sort_index()

# Update column type and name
scenario_hist.columns = scenario_hist.columns.astype(int)
scenario_hist.columns.name = 'year'
scenario_hist.loc[ismatch(sector="Solvents Production and Application", gas="N2O")]
scenario_hist.index.get_level_values("sector").unique()

# %%
cmip7_hist = (pd.read_csv(LOCATION_CMIP7_HISTORY / f"{marker_to_run}_{VERSION_ESGF}_hist.csv", index_col=0))
cmip7_hist = cmip7_hist.set_index(['country', 'gas', 'sector', 'unit'])
cmip7_hist = cmip7_hist.sort_index()

# Update column type and name
cmip7_hist.columns = cmip7_hist.columns.astype(int)
cmip7_hist.columns.name = 'year'

# %%
cmip7_hist.index.get_level_values("sector").unique()

# %%
missing_idx = cmip7_hist.loc[:,2023].index.difference(scenario_hist.loc[:,2023].index)
missing_idx.to_frame(index=False).to_csv(settings.out_path / "rows_missing_from_downscaled_hist.csv")

# %%
missing_idx.to_frame(index=False)[missing_idx.to_frame(index=False)["gas"]=="N2O"]["country"].unique()

# %%
cmip7_hist.index.get_level_values("sector").unique()

# %%
cmip7_hist = cmip7_hist.loc[:,:2022]

# %%
hist = pd.concat([cmip7_hist, scenario_hist], axis=1).dropna()

# %% [markdown]
# # Step 6: IAM — read and process scenario data
# ## *Stage 5/13* · [↑ overview](#workflow-for-cmip7-scenariomip-emissions-harmonization--post-2100-extensions)

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
iam_df = cmip7_utils.filter_scenario(iam_df, scenarios=SCENARIO_SELECTION)

IAMC_COLS = ["model", "scenario", "region", "variable", "unit"]
iam_df.columns = [
    str(int(float(col))) if str(col).replace('.', '', 1).isdigit() else col
    for col in iam_df.columns
]
HARMONIZED_YEAR_COLS = [col for col in iam_df.columns if col.isdigit() and settings.base_year <= int(col) <= 2500]

# keep only relevant columns
iam_df = iam_df[(IAMC_COLS + HARMONIZED_YEAR_COLS)]

# %%
reference_iam_df = pd.read_csv((settings.scenario_path.parent / "final-high-priority-20260205" / "harmonised-gridding_REMIND-MAgPIE 3.5-4.11.csv"))

# %%
reference_vars = reference_iam_df["variable"].unique()
extensions_vars = iam_df["variable"].unique()

print("number of regions in scenario:", len(reference_iam_df["region"].unique()), "\n"
      "number of regions in extension:", len(iam_df["region"].unique()))

print("number of variables in scenario:", len(reference_iam_df["variable"].unique()), "\n"
      "number of variables in extension:", len(iam_df["variable"].unique()))

print("\nvariables missing from extension:\n", set(reference_vars) - set(extensions_vars))

print("\nvariables only in extension:\n", set(extensions_vars) - set(reference_vars))

# %%
# the missing variables are the AFOLU variables, which we don't grid, so we do not need them here.
# the additional variables are either aggregates, additional diagnostic variables, or variables only needed for SCMs, so we can filter them out

# filter dataframe by the intersection of the variables with the reference scenario
shared_vars = np.intersect1d(extensions_vars, reference_vars)
iam_df = iam_df[iam_df["variable"].isin(shared_vars)].reset_index(drop=True)

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


# %%
iam_df

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
# # Step 6b: Read harmonization overrides

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

# %% [markdown]
# # Step 6c: Prepare GDP proxy
# ## *Stage 6/13* · [↑ overview](#workflow-for-cmip7-scenariomip-emissions-harmonization--post-2100-extensions)
#
# Read in different GDP scenarios for SSP1 to SSP5 from SSP DB.
#

# %%
settings.scenario_path

# %%
# New; updated SSP data from CMIP7 era (downloaded from: http://files.ece.iiasa.ac.at/ssp/downloads/ssp_basic_drivers_release_3.2.beta_full.xlsx, and then selected only the GDP|PPP variable)
gdp_new = pd.read_csv(
        settings.scenario_path / "ssp_basic_drivers_release_3.2.beta_full_gdp-extensions.csv",
        index_col=list(range(5)),
    )

# drop the Historical Reference data
gdp_new = gdp_new.loc[~ismatch(Scenario="Historical Reference")]

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
#    .rename(columns=int)
    .pix.project(["ssp", "country"])
    .pix.aggregate(country=settings.country_combinations)
)

# ensure integer columns
gdp.columns = [int(float(col)) for col in gdp.columns]

# full year range
all_years = range(2020, 2501)

# reindex
gdp = gdp.reindex(columns=all_years)

# interpolate to fill all years
gdp = gdp.interpolate(axis=1, method='linear', limit_direction='both')
gdp

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
            #   "kosovo": "srb (kosovo)",
              "kosovo": "kos",
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

if settings.country_combinations:  # only aggregate if not empty
    hist = hist.pix.aggregate(country=settings.country_combinations)

gdp.index = gdp.index.set_levels(
    gdp.index.levels[gdp.index.names.index("country")].to_series().replace(rename_gdp),
    level="country"
)

if settings.country_combinations:  # only aggregate if not empty
    gdp = gdp.pix.aggregate(country=settings.country_combinations)

# %%
SSP_per_pathway = cmip7_utils.guess_ssp(iam_df)
GDP_per_pathway = cmip7_utils.join_gdp_based_on_ssp(
    scenarios_with_ssp_mapping=SSP_per_pathway,
    gdp_per_ssp=gdp
)

# %% [markdown]
# # Step 7: Coverage checks
# ## *Stage 7/13* · [↑ overview](#workflow-for-cmip7-scenariomip-emissions-harmonization--post-2100-extensions)
# ## Country coverage

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
countries_with_gdp_missing_from_regionmapping = pd.Index(sorted(
    set(countries_with_gdp_data) - set(countries_with_regionmapping)
)) # where is 'World'? It is missing, but we want it for Aircraft and International Shipping. Answer: it does not need to be downscaled.

# show what we have
print("Countries with GDP data (for downscaling):")
print(len(countries_with_gdp_data))
print("Countries with historical emissions data:")
print(len(countries_with_hist_data))
print("Countries in the IAM region mapping:")
print(len(countries_with_regionmapping))
print("Countries with data for all three above:")
print(countries_with_hist_and_gdp_and_regionmapping_data)
print(f"Countries with GDP data but missing from IAM region mapping ({len(countries_with_gdp_missing_from_regionmapping)}):")
print(countries_with_gdp_missing_from_regionmapping.tolist())

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
# ## Sector coverage (check historical)

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
# # Step 8: Set up workflow infrastructure
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
# ## Define workflow object

# %%
# TODO (in the future): allow doing multiple models at once in a notebook --> right now the below section only works for 1 model at a time
(model_name,) = iam_df.pix.unique("model")
regionmapping = regionmappings[model_name]


# %%
# indexes for countries on a grid
indexraster = IndexRaster.from_netcdf(
    settings.gridding_path / "ssp_comb_indexraster_splitsudankosovopalestine.nc", # redo: notebooks\gridding_data\generate_ceds_proxy_netcdfs.py
    chunks={},
).compute()

# Filter regionmapping and prepare for dissolve
filtered_regionmapping = regionmapping.filter(indexraster.index).data.rename("country")

# Check for duplicate index labels in the mapping
if filtered_regionmapping.index.duplicated().any():
    duplicates = filtered_regionmapping.index[filtered_regionmapping.index.duplicated(keep=False)]
    print(f"⚠️  Found {len(duplicates.unique())} countries with duplicate region mappings:")
    for dup in sorted(duplicates.unique()):
        regions = list(filtered_regionmapping[filtered_regionmapping.index == dup].values)
        print(f"  {dup}: {regions}")
    print("  → Keeping first occurrence for dissolve()")
    filtered_regionmapping = filtered_regionmapping[~filtered_regionmapping.index.duplicated(keep='first')]

indexraster_region = indexraster.dissolve(filtered_regionmapping).compute()

print(sorted(indexraster.index.tolist()))

# 'kos' in regionmapping.index.tolist()
# 'srb (kosovo)' in np.unique(indexraster.index)

# %%
iam_df.columns

# %%
# do variable name replacements to align with CEDS and BB4CMIP7 historical products
# Sulfur -> SO2 (CEDS+BB4CMIP7)
iam_df = iam_df.rename(index=lambda v: v.replace("Sulfur", "SO2") if "Sulfur" in v else v)
# VOC -> NMVOC for anthro sectors (CEDS) — only if not already NMVOC
iam_df = iam_df.rename(index=lambda v: v.replace("VOC", "NMVOC") if "VOC" in v and "NMVOC" not in v else v)
# Rename NMVOC to NMVOCbulk in iam_df for openburning sectors (BB4CMIP7)
openburning_sectors = cmip7_utils.SECTOR_ORDERING_DEFAULT['em_openburning']
def rename_voc_to_nmvoc_iam(idx):
    gas_idx = iam_df.index.names.index("gas")
    sector_idx = iam_df.index.names.index("sector")
    if idx[sector_idx] in openburning_sectors and idx[gas_idx] == "NMVOC":
        idx_list = list(idx)
        idx_list[gas_idx] = "NMVOCbulk"
        return tuple(idx_list)
    return idx
iam_df.index = iam_df.index.map(rename_voc_to_nmvoc_iam)


# drop CDR sector information for non-CO2 species from iam_df
iam_df = iam_df[
    ~(
        (iam_df.index.get_level_values("sector") == "Other CDR") &
        (iam_df.index.get_level_values("gas") != "CO2")
    )
]

# %%
# Fill missing historical data with zeros for countries in the regionmapping
# that don't have data for all (gas, sector, unit) combinations.
# This prevents MissingHistoricalError in the aneris downscaler.

# Get all countries that will be used in the workflow
workflow_countries = countries_with_hist_and_gdp_and_regionmapping_data

# Build the full set of (gas, sector, unit) from hist (excluding World)
hist_gas_sector_unit = (
    hist.loc[~isin(country="World")]
    .index.droplevel("country")
    .drop_duplicates()
)

# Build the expected full index: every workflow country × every (gas, sector, unit)
full_index = pd.MultiIndex.from_tuples(
    [(c, g, s, u) for c in workflow_countries for g, s, u in hist_gas_sector_unit],
    names=["country", "gas", "sector", "unit"]
)

# Find which entries are missing from hist
existing_hist_index = hist.index
missing_from_hist = full_index.difference(existing_hist_index)

if len(missing_from_hist) > 0:
    # Create zero-filled rows for missing entries
    zero_rows = pd.DataFrame(
        0.0,
        index=missing_from_hist,
        columns=hist.columns
    )
    hist = pd.concat([hist, zero_rows]).sort_index()
    
    # Report
    missing_report = missing_from_hist.to_frame(index=False)
    report_dir = Path(version_path, "workflow_driver_data")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_file = report_dir / f"missing_historical_filled_with_zeros_{settings.version}.csv"
    missing_report.to_csv(report_file, index=False)
    
    # Summary
    n_countries = missing_report["country"].nunique()
    n_combos = len(missing_report)
    print(f"⚠️  Filled {n_combos} missing historical entries with zeros ")
    print(f"   ({n_countries} countries affected: {sorted(missing_report['country'].unique())})")
    print(f"   Report saved to: {report_file}")
    
    # Show summary by country
    print(f"\n   Missing entries per country:")
    for country, group in missing_report.groupby("country"):
        sectors = sorted(group["sector"].unique())
        gases = sorted(group["gas"].unique())
        print(f"     {country}: {len(group)} entries — gases: {gases}, sectors: {sectors}")
else:
    print("✅ No missing historical data — all countries have complete coverage")

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
    regionmapping.filter(countries_with_hist_and_gdp_and_regionmapping_data), # missing historical data for any country (especially, ['myt', 'gum', 'kos', 'pse') is zero-filled above
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

# %%
def check_harmonization_consistency(workflow, settings, version_path, atol=1e-6, rtol=1.0,
                                     region=None, gas=None, sector=None):
    """
    Check whether the model data is already harmonized to the historical data.
    
    Compares model base-year values at regional level with aggregated country-level history.
    Saves a mismatch table to CSV if any discrepancies found.
    
    Parameters
    ----------
    workflow : WorkflowDriver
        Workflow object with model, hist, and regionmapping attributes
    settings : Settings
        Settings object with base_year attribute
    version_path : Path
        Output directory for saving mismatch CSV
    atol : float, default 1e-6
        Absolute tolerance for "exact match"
    rtol : float, default 1.0
        Relative tolerance in percent for "close match"
    region : str or list of str, optional
        Filter to specific region(s). If None, all regions included.
    gas : str or list of str, optional
        Filter to specific gas/species (e.g., "CO2", "CO"). If None, all gases included.
    sector : str or list of str, optional
        Filter to specific sector(s). If None, all sectors included.
    """
    base_year = settings.base_year
    
    print(f"\n{'='*80}")
    print(f"HARMONIZATION CONSISTENCY CHECK (base year = {base_year})")
    
    # Describe filters if applied
    filters_applied = []
    if region is not None:
        region_list = [region] if isinstance(region, str) else region
        filters_applied.append(f"Region(s): {', '.join(region_list)}")
    if gas is not None:
        gas_list = [gas] if isinstance(gas, str) else gas
        filters_applied.append(f"Gas/Species: {', '.join(gas_list)}")
    if sector is not None:
        sector_list = [sector] if isinstance(sector, str) else sector
        filters_applied.append(f"Sector(s): {', '.join(sector_list)}")
    
    if filters_applied:
        print(f"Filters: {' | '.join(filters_applied)}")
    
    print(f"{'='*80}\n")

    # Aggregate country-level history to IAM regions
    hist_agg = workflow.regionmapping.aggregate(workflow.hist.copy(), dropna=True)

    # Get model data for non-World regions that exist in the regionmapping
    model_regions = workflow.regionmapping.data.unique()
    model_for_check = workflow.model.loc[isin(region=model_regions)].copy()

    # Apply filters before comparison
    if region is not None:
        region_list = [region] if isinstance(region, str) else region
        model_for_check = model_for_check.loc[isin(region=region_list)]
        hist_agg = hist_agg.loc[isin(region=region_list)]
    
    if gas is not None:
        gas_list = [gas] if isinstance(gas, str) else gas
        model_for_check = model_for_check.loc[isin(gas=gas_list)]
        hist_agg = hist_agg.loc[isin(gas=gas_list)]
    
    if sector is not None:
        sector_list = [sector] if isinstance(sector, str) else sector
        model_for_check = model_for_check.loc[isin(sector=sector_list)]
        hist_agg = hist_agg.loc[isin(sector=sector_list)]

    # Align on common (region, gas, sector, unit) index
    model_by = model_for_check.droplevel(["model", "scenario"])[[base_year]].rename(columns={base_year: "model"})
    hist_by = hist_agg[[base_year]].rename(columns={base_year: "hist"})

    comparison = model_by.join(hist_by, how="inner")

    if len(comparison) == 0:
        print("⚠️  No matching data after applying filters. Check region/gas/sector names.")
        return

    # Compute absolute and relative differences
    comparison["abs_diff"] = comparison["model"] - comparison["hist"]
    # Avoid division by zero: relative diff only where hist != 0
    comparison["rel_diff_pct"] = comparison.apply(
        lambda row: (row["abs_diff"] / row["hist"] * 100) if abs(row["hist"]) > 1e-15 else (
            0.0 if abs(row["model"]) < 1e-15 else float("inf")
        ), axis=1
    )

    # Classify matches
    exact_match = (comparison["abs_diff"].abs() < atol)
    close_match = (comparison["rel_diff_pct"].abs() <= rtol) & ~exact_match
    mismatch = ~exact_match & ~close_match

    n_total = len(comparison)
    n_exact = exact_match.sum()
    n_close = close_match.sum()
    n_mismatch = mismatch.sum()

    print(f"Compared {n_total} (region, gas, sector, unit) combinations at base year {base_year}:")
    print(f"  ✅ Exact match (|diff| < {atol}):         {n_exact} ({n_exact/n_total*100:.1f}%)")
    print(f"  ≈  Close match (rel diff ≤ {rtol}%):       {n_close} ({n_close/n_total*100:.1f}%)")
    print(f"  ❌ Mismatch (rel diff > {rtol}%):           {n_mismatch} ({n_mismatch/n_total*100:.1f}%)")

    if n_mismatch > 0:
        mismatched = comparison[mismatch].copy()
        mismatched = mismatched.sort_values("rel_diff_pct", key=abs, ascending=False)
        
        print(f"\n⚠️  Top mismatches (model vs aggregated history):")
        print(f"{'Region':<25} {'Gas':<12} {'Sector':<35} {'Model':>12} {'Hist':>12} {'Diff%':>10}")
        print("-" * 110)
        for idx, row in mismatched.head(30).iterrows():
            region = idx[0] if isinstance(idx, tuple) else str(idx)
            gas = idx[1] if isinstance(idx, tuple) and len(idx) > 1 else ""
            sector = idx[2] if isinstance(idx, tuple) and len(idx) > 2 else ""
            rdiff = f"{row['rel_diff_pct']:.2f}%" if not float("inf") == abs(row['rel_diff_pct']) else "inf"
            print(f"{str(region):<25} {str(gas):<12} {str(sector):<35} {row['model']:>12.4f} {row['hist']:>12.4f} {rdiff:>10}")
        
        if len(mismatched) > 30:
            print(f"  ... and {len(mismatched) - 30} more mismatches")
        
        # Save full comparison to CSV for inspection
        mismatch_file = version_path / f"check_harmonization_consistency_{settings.version}.csv"
        mismatched.to_csv(mismatch_file)
        print(f"\n  Full mismatch table saved to: {mismatch_file}")
        
        print(f"\n📋 CONCLUSION: Model data does NOT perfectly match aggregated history at base year {base_year}.")
        print(f"   The concordia harmonization step will adjust {n_mismatch} variable(s).")
        print(f"   If input data is already harmonized upstream, consider skip_harmonization=True.")
    else:
        print(f"\n✅ CONCLUSION: Model base-year values match aggregated history for all variables.")
        print(f"   The data appears to be already harmonized. Harmonization will be a near-pass-through.")
        print(f"   Consider skip_harmonization=True to avoid potential artefacts from re-harmonization.")

    # Also check variables in model but NOT in history
    model_index = model_for_check.droplevel(["model", "scenario"]).index
    hist_index = hist_agg.index
    in_model_not_hist = model_index.difference(hist_index)

    
    if len(in_model_not_hist) > 0:
        unique_sectors = set(idx[2] if len(idx) > 2 else str(idx) for idx in in_model_not_hist)
        print(f"\n⚠️  {len(in_model_not_hist)} model rows have no matching history (sectors: {sorted(unique_sectors)})")

        # --- Add missing rows to workflow.hist filled with zeros ---
        # regionmapping.data is a Series: country -> region
        country_to_region = workflow.regionmapping.data  # index=country, values=region

        rows_to_add = []
        rows_skipped = []

        for idx in in_model_not_hist:
            # Unpack index levels (region, gas, sector, unit)
            region = idx[0] if isinstance(idx, tuple) else str(idx)
            gas    = idx[1] if isinstance(idx, tuple) and len(idx) > 1 else ""
            sector = idx[2] if isinstance(idx, tuple) and len(idx) > 2 else ""
            unit   = idx[3] if isinstance(idx, tuple) and len(idx) > 3 else ""

            # Find all countries that map to this region
            countries_in_region = country_to_region[country_to_region == region].index.tolist()

            for country in countries_in_region:
                # Check if this country already has data for this gas/sector combo
                try:
                    existing = workflow.hist.loc[isin(region=country, gas=gas, sector=sector)]
                    if len(existing) > 0:
                        rows_skipped.append((country, region, gas, sector))
                        continue
                except KeyError:
                    pass

                # Safe to add — build zero row with full index
                new_idx = tuple(
                    val for name, val in zip(
                        workflow.hist.index.names,
                        (country, gas, sector, unit)
                    )
                )
                rows_to_add.append(new_idx)

        if rows_skipped:
            print(f"\n   ℹ️  Skipped {len(rows_skipped)} country/gas/sector combos that already have hist data:")
            for country, region, gas, sector in rows_skipped:
                print(f"      {country:<25} (region: {region:<20}) {gas:<12} {sector}")

        if rows_to_add:
            missing_hist_index = pd.MultiIndex.from_tuples(
                rows_to_add,
                names=workflow.hist.index.names
            )
            zero_rows = pd.DataFrame(
                0.0,
                index=missing_hist_index,
                columns=workflow.hist.columns
            )

            # Safeguard: double-check none of these already exist in hist
            already_exists = missing_hist_index.isin(workflow.hist.index)
            if already_exists.any():
                print(f"\n   ⚠️  Safeguard: {already_exists.sum()} rows already exist in hist, skipping those")
                zero_rows = zero_rows[~already_exists]

            workflow.hist = pd.concat([workflow.hist, zero_rows]).sort_index()

            print(f"\n   ✅ Added {len(zero_rows)} missing country-level row(s) to workflow.hist filled with zeros:")
            for idx in zero_rows.index:
                print(f"      {str(idx[0]):<25} {str(idx[1]):<12} {str(idx[2])}")
        else:
            print(f"\n   ℹ️  No country-level rows needed — all countries already have hist data for missing combos.")
            
        
    # Save relevant regionmapping
    filtered_regionmapping = workflow.regionmapping
    if region is not None or gas is not None or sector is not None:
        # Extract region names from the filtered comparison, then find the countries that map to those regions
        filtered_regions = comparison.index.get_level_values('region').unique()
        # regionmapping.data is a Series: country (index) -> region (values)
        countries_in_filtered_regions = workflow.regionmapping.data[
            workflow.regionmapping.data.isin(filtered_regions)
        ].index.tolist()
        filtered_regionmapping = workflow.regionmapping.filter(countries_in_filtered_regions)
    
    regionmapping_file = version_path / f"check_harmonization_consistency_regionmapping_{settings.version}.csv"
    filtered_regionmapping.data.to_csv(regionmapping_file)
    print(f"\nRegionmapping saved to: {regionmapping_file}")

# %%
check_harmonization_consistency(workflow, settings, version_path)

# Check all regions (original behavior)
check_harmonization_consistency(workflow, settings, version_path)

# Check only one region
check_harmonization_consistency(workflow, settings, version_path, region="Middle East and Central Asia")

# Check multiple regions
check_harmonization_consistency(workflow, settings, version_path, region=["Middle East and Central Asia", "Europe"])

# Check only CO2
check_harmonization_consistency(workflow, settings, version_path, gas="CO2")

# Check only Agricultural Waste Burning sector
check_harmonization_consistency(workflow, settings, version_path, sector="Agricultural Waste Burning")

# Combine filters
check_harmonization_consistency(workflow, settings, version_path, 
                                region="REMIND-MAgPIE 3.5-4.11|Middle East and North Africa", 
                                gas="CO", 
                                sector="Agricultural Waste Burning")


# %% [markdown]
# # Harmonize, downscale and grid everything
#

# %% [markdown]
# ## Harmonize and downscale everything, but do not grid
#
# If you also want grids, use the gridding interface directly - see below.
#

# %%
# For extensions: set downscaling methods in aneris to a later convergence year (default is 2100)

from functools import partial
from aneris.downscaling.core import Downscaler
from aneris.downscaling.intensity_convergence import intensity_convergence

Downscaler._methods["ipat_2100_gdp"] = partial(
    intensity_convergence, convergence_year=2300, proxy_name="gdp"
)
Downscaler._methods["ipat_2150_pop"] = partial(
    intensity_convergence, convergence_year=2300, proxy_name="pop"
)

# %%
# use hist_zero for all variables; this effectively switches off concordia harmonization

# Get all variables that will be harmonized
variables = workflow.variabledefs.index

overrides = pd.Series(
    "hist_zero",
    index=variables,
    name="method"
)

# Inject into workflow
workflow.harm_overrides = overrides

# %%
base_year = workflow.settings.base_year
hist_at_base = workflow.hist.loc[~isin(country="World"), base_year]
zero_hist_combos = (
    hist_at_base
    .groupby(["gas", "sector"])
    .sum()
    .pipe(lambda s: s[s.abs() < 1e-10])
    .index
)

zero_sectors = set(zero_hist_combos.get_level_values("sector"))
print(f"Sectors with all-zero hist at base year: {zero_sectors}")

# %%
if run_main:
    downscaled = workflow.harmonize_and_downscale() # For a 1 scenario, this takes about 50 seconds on Jarmo's DELL laptop.
    
    # ─── Post-downscaling fixes ───────────────────────────────────────────
    # These fixes are applied to BOTH the local `downscaled` DataFrame AND
    # the workflow object's internal state (workflow.downscaled.*), so that
    # they carry through to workflow.grid(), QC checks, and data exports.
    
    # Helper: apply an in-place zeroing fix to a DataFrame
    def _apply_zero_fix(df, row_mask, cell_mask_fn, label):
        """Zero out cells matching cell_mask_fn within rows matching row_mask. Returns count."""
        if df is None or row_mask is None:
            return 0
        if not row_mask.any():
            return 0
        data = df.loc[row_mask]
        cell_mask = cell_mask_fn(data)
        n = cell_mask.sum().sum()
        if n > 0:
            df.loc[row_mask] = data.where(~cell_mask, 0.0)
        return n
    
    # FIX 1: Zero out tiny negatives in Agricultural Waste Burning for REMIND-MAgPIE
    # The harmonization/downscaling can produce very small negative values for AWB
    # in certain countries. If the absolute value is < 1e-4, set to zero.
    print("\n[FIX 1] Checking for small negatives in AWB (REMIND-MAgPIE)...")
    def _awb_row_mask(df):
        return (
            (df.index.get_level_values("sector") == "Agricultural Waste Burning") &
            (df.index.get_level_values("model") == "REMIND-MAgPIE 3.5-4.11")
        )
    def _awb_cell_mask(data):
        return (data < 0) & (data.abs() < 1e-4)
    
    n_fixed_1 = _apply_zero_fix(downscaled, _awb_row_mask(downscaled), _awb_cell_mask, "downscaled")
    # Also fix workflow internal countrylevel (has extra method/region index levels)
    cl = workflow.downscaled.countrylevel
    if cl is not None:
        _apply_zero_fix(cl, _awb_row_mask(cl), _awb_cell_mask, "countrylevel")
    
    if n_fixed_1 > 0:
        print(f"  Set {n_fixed_1} small negative AWB values to zero")
    else:
        print(f"  [OK] No small negative AWB values found")

    # FIX 2: Zero out near-zero global (International Shipping / Aircraft) values
    # These global-level sectors can end up with very small residual values (< 1e-6)
    # from harmonization artifacts. Set them to zero.
    print("\n[FIX 2] Checking for near-zero global Shipping/Aircraft values...")
    def _global_row_mask(df):
        if "country" not in df.index.names:
            return None
        return (
            (df.index.get_level_values("country") == "World") &
            (df.index.get_level_values("sector").isin(["International Shipping", "Aircraft"]))
        )
    def _global_cell_mask(data):
        return (data.abs() < 1e-6)  # includes exact zeros, but where(~mask, 0.0) is no-op for those
    
    n_fixed_2 = _apply_zero_fix(downscaled, _global_row_mask(downscaled), _global_cell_mask, "downscaled")
    # Also fix workflow internal globallevel
    gl = workflow.downscaled.globallevel
    if gl is not None:
        _apply_zero_fix(gl, _global_row_mask(gl), _global_cell_mask, "globallevel")
    
    if n_fixed_2 > 0:
        print(f"  Set {n_fixed_2} near-zero global Shipping/Aircraft values to zero")
    else:
        print(f"  [OK] No near-zero global Shipping/Aircraft values found")

    # FIX 3: Handle remaining negative values for non-CO2 gases
    # For non-CO2 gases: throw error if value < -1e-10, set to zero if >= -1e-10 and < 0
    # For CO2: allow negatives (carbon capture is legitimate)
    print("\n[FIX 3] Checking remaining negatives for non-CO2 gases...")
    
    # Get non-CO2 rows
    if "gas" in downscaled.index.names:
        non_co2_rows = downscaled.index.get_level_values("gas") != "CO2"
        downscaled_non_co2 = downscaled.loc[non_co2_rows]
        
        # Check for significant negatives (< -1e-10) in non-CO2 gases
        sig_neg_mask = downscaled_non_co2 < -1e-10
        if sig_neg_mask.any().any():
            sig_neg_values = downscaled_non_co2[sig_neg_mask]
            print(f"  ERROR: Found significantly negative values (< -1e-10) in non-CO2 gases:")
            # Show non-zero entries
            for col in sig_neg_values.columns:
                col_data = sig_neg_values[col]
                if col_data.notna().any():
                    print(f"    {col}:")
                    print(col_data[col_data.notna()])
            raise ValueError("Significant negative values (< -1e-10) found in non-CO2 gases after harmonization")
        
        # Set small negatives (>= -1e-10 and < 0) in non-CO2 gases to zero
        small_neg_mask = (downscaled_non_co2 >= -1e-10) & (downscaled_non_co2 < 0)
        n_small_fixed = small_neg_mask.sum().sum()
        
        if n_small_fixed > 0:
            # Apply fix to main downscaled DataFrame
            downscaled.loc[non_co2_rows] = downscaled.loc[non_co2_rows].where(
                (downscaled.loc[non_co2_rows] >= 0) | (downscaled.loc[non_co2_rows].isna()),
                0.0
            )
            
            # Also fix workflow internal sub-levels
            for _attr in ['globallevel', 'regionlevel', 'countrylevel']:
                _df = getattr(workflow.downscaled, _attr)
                if _df is not None and "gas" in _df.index.names:
                    _non_co2_rows = _df.index.get_level_values("gas") != "CO2"
                    _df.loc[_non_co2_rows] = _df.loc[_non_co2_rows].where(
                        (_df.loc[_non_co2_rows] >= 0) | (_df.loc[_non_co2_rows].isna()),
                        0.0
                    )
            
            print(f"  Set {n_small_fixed} small negative values (>= -1e-10, < 0) to zero for non-CO2 gases")
        else:
            print(f"  [OK] No small negative values found in non-CO2 gases")
    else:
        print(f"  [SKIP] No 'gas' level in index - skipping non-CO2 check")
    
    # Cache the fixed downscaled data and monkey-patch harmonize_and_downscale
    # so that workflow.grid() uses the fixed data instead of re-running from scratch.
    _fixed_downscaled = downscaled
    workflow.harmonize_and_downscale = lambda variabledefs=None: _fixed_downscaled
    print("\n[OK] Fixed downscaled data cached — workflow.grid() will use the patched result")


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

# %% [markdown]
# #### Quality control tests on downscaled data

# %%
if run_main:
    # check for negative values where we don't expect them
    
    cdr_sectors = [
        "Direct Air Capture",
        "Other CDR",
        "Enhanced Weathering",
        "BECCS",
        "Ocean",
        "Biochar",
        "Soil Carbon Management"
    ]
    
    row_mins = workflow.downscaled.data.select_dtypes("number").min(axis=1)
    negative_rows = row_mins[row_mins < 0]
    
    # Get sector and gas as Series aligned to the DataFrame index
    sector_series = workflow.downscaled.data.index.get_level_values("sector").to_series(index=workflow.downscaled.data.index)
    gas_series = workflow.downscaled.data.index.get_level_values("gas").to_series(index=workflow.downscaled.data.index)
    
    # Build a boolean mask of allowed exemptions
    allowed_mask = (
        sector_series.isin(cdr_sectors) |  # any CDR sector
        ((sector_series == "Industrial Sector") & (gas_series == "CO2"))  # Industrial + CO2
    )
    
    # Filter out exempted rows
    disallowed_negative_rows = negative_rows[
        ~allowed_mask.loc[negative_rows.index]
    ]
    
    if not disallowed_negative_rows.empty:
        negative_values_df = workflow.downscaled.data.loc[disallowed_negative_rows.index]
    
        problem_pairs = (
            negative_values_df.index
            .to_frame(index=False)[["sector", "gas"]]
            .drop_duplicates()
            .sort_values(["sector", "gas"])
            .reset_index(drop=True)
        )
    
        raise ValueError(
            f"⚠️ Found disallowed negatives in {len(disallowed_negative_rows)} rows.\n\n"
            "Negative values found for the following sector+gas combinations:\n"
            f"{problem_pairs.to_string(index=False)}"
        )
    else:
        print("✅ No disallowed negative values found")

# %%
# test that ensures all values in the CDR sectors are negative where expected

if run_main:
    # List of CDR sectors to check (excluding "Other CDR")
    cdr_sectors_check = [
        "Direct Air Capture",
        "Enhanced Weathering",
        "BECCS",
    ]
    
    # Compute max per row to detect positives
    row_maxs = workflow.downscaled.data.select_dtypes("number").max(axis=1)
    
    # Find rows that are positive
    positive_rows = row_maxs[row_maxs > 0]
    
    # Get sector info from MultiIndex
    sector_series = workflow.downscaled.data.index.get_level_values("sector").to_series(index=workflow.downscaled.data.index)
    
    # Only consider disallowed positives: CDR sectors (except "Other CDR")
    disallowed_positive_rows = positive_rows[
        sector_series.loc[positive_rows.index].isin(cdr_sectors_check)
    ]
    
    if not disallowed_positive_rows.empty:
        positive_values_df = workflow.downscaled.data.loc[disallowed_positive_rows.index]
    
        problem_pairs = (
            positive_values_df.index
            .to_frame(index=False)[["sector", "gas"]]
            .drop_duplicates()
            .sort_values(["sector", "gas"])
            .reset_index(drop=True)
        )
    
        raise ValueError(
            f"⚠️ Found disallowed positive values in {len(disallowed_positive_rows)} rows.\n\n"
            "Positive values found for the following sector+gas combinations:\n"
            f"{problem_pairs.to_string(index=False)}"
        )
    else:
        print("✅ No positive values found for CDR sectors")

# %%
if run_main:
    # Check for global totals that are suspiciously close to zero but not exactly zero
    # This catches cases where harmonization/downscaling artifacts reduce meaningful
    # emissions to near-zero values (e.g., order of 1e-6 or smaller)
    
    near_zero_threshold = 1e-6  # absolute value threshold
    
    # Sum across all countries to get global totals per (gas, sector, unit) for each year
    downscaled_data = workflow.downscaled.data
    global_totals = downscaled_data.groupby(["gas", "sector", "unit"]).sum()
    
    # For each (gas, sector, unit), check if any year has a value that is
    # near-zero (|value| <= threshold) but not exactly zero
    numeric_cols = global_totals.select_dtypes("number")
    is_near_zero = (numeric_cols.abs() <= near_zero_threshold) & (numeric_cols.abs() > 0)
    
    # Flag rows where ANY year has a near-zero global total
    rows_with_near_zero = is_near_zero.any(axis=1)
    
    if rows_with_near_zero.any():
        near_zero_df = global_totals[rows_with_near_zero]
        
        # Count how many years are affected per row
        n_years_affected = is_near_zero[rows_with_near_zero].sum(axis=1)
        
        # Get the minimum absolute non-zero value per row for context
        abs_nonzero = numeric_cols[rows_with_near_zero].replace(0, float("nan")).abs()
        min_abs_val = abs_nonzero.min(axis=1)
        
        problem_info = near_zero_df.index.to_frame(index=False)
        problem_info["n_years_near_zero"] = n_years_affected.values
        problem_info["min_abs_value"] = min_abs_val.values
        
        print(f"⚠️  Found {rows_with_near_zero.sum()} (gas, sector) combinations with near-zero "
              f"global totals (0 < |total| ≤ {near_zero_threshold}):")
        print(f"{'Gas':<12} {'Sector':<35} {'Unit':<20} {'# Years':>8} {'Min |value|':>14}")
        print("-" * 95)
        for _, row in problem_info.iterrows():
            print(f"{str(row['gas']):<12} {str(row['sector']):<35} {str(row['unit']):<20} "
                  f"{row['n_years_near_zero']:>8} {row['min_abs_value']:>14.2e}")
        
        # Save for inspection
        near_zero_file = version_path / f"qc_near_zero_global_totals_{settings.version}.csv"
        near_zero_df.to_csv(near_zero_file)
        print(f"\n  Near-zero global totals saved to: {near_zero_file}")
        print("  These may indicate harmonization/downscaling artifacts that should be investigated.")
    else:
        print("✅ No suspiciously near-zero global totals found")

# END OF DOWNSCALING CHECK


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
# # Step 10: Grid — create NetCDF files
# ## *Stage 9/13* · [↑ overview](#workflow-for-cmip7-scenariomip-emissions-harmonization--post-2100-extensions)
#
# Runs `workflow.grid()` to produce one NetCDF per species/sector combination.
# Latest test with 1 scenario was ~50 minutes on Jarmo's DELL laptop (~11.4 GB output).


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

# %%
# ds = xr.open_dataset("/Users/jarmo/Documents/GitHub/mozart/projects/mine/concordia/results/vl-ext_1-1-1/CO2-em-anthro_input4MIPs_emissions_ScenarioMIP_IIASA-IAMC-vl-ext-1-1-1_gn_210501-250012.nc")

# %%
# ds["CO2_em_anthro"].isel(time=400, sector=4).plot(robust=True);


# %% [markdown]
# Clarify which gridcell area we use

# %%
def remove_fillvalue_from_bounds(ds):
    for coord in ["time_bnds", "lon_bnds", "lat_bnds", "level_bnds", "sector_bnds"]:
        if coord in ds:
            ds[coord].encoding["_FillValue"] = None
    return ds

# %%
# areas of gridcells for calculatings totals
areacella = xr.open_dataset(Path(settings.gridding_path, "areacella_input4MIPs_emissions_CMIP_CEDS-CMIP-2025-04-18_gn.nc"))
cell_area = areacella["areacella"]

encoding = {"areacella": {"zlib": True, "complevel": 2}}
areacella.close()

from input4mips_validation.io import (
    generate_creation_timestamp,
    generate_tracking_id
)

# update metadata
cmip7_areacella = areacella
original_creators = areacella.attrs['contact']
cmip7_areacella.attrs['contact'] = cmip7_utils.DS_ATTRS['contact']
cmip7_areacella.attrs['source_id'] = cmip7_utils.DS_ATTRS['institution_id'] + '-' + VERSION_ESGF
cmip7_areacella.attrs['creation_date'] = generate_creation_timestamp()
cmip7_areacella.attrs['tracking_id'] = generate_tracking_id()
for v in [
    'institution', 'institution_id', 'doi', 'target_mip', 'source', 'license', 'further_info_url'
]:
    cmip7_areacella.attrs[v] = cmip7_utils.DS_ATTRS[v]

cmip7_areacella.attrs['source_version'] = VERSION_ESGF.replace("-",".") # e.g. "1.0.0"
cmip7_areacella.attrs['comment'] = f"Research data originally produced by {original_creators} using the Community Emissions Data System (CEDS), at Pacific Northwest National Laboratory - Joint Global Change Research Institute, College Park, MD 20740, USA. Reused for the production of all future {cmip7_utils.DS_ATTRS['target_mip']} emissions data."
cmip7_areacella = cmip7_areacella.pipe(remove_fillvalue_from_bounds)
folder_areacella = settings.out_path / GRIDDING_VERSION / 'areacella'
folder_areacella.mkdir(parents=True, exist_ok=True)
cmip7_areacella.to_netcdf(folder_areacella / f'areacella_input4MIPs_emissions_{cmip7_utils.DS_ATTRS['target_mip']}_{cmip7_utils.DS_ATTRS['institution_id']}-{VERSION_ESGF}_gn.nc', encoding=encoding)


# %% [markdown]
# # START OF POSTPROCESSING
#
# CEDS-2023 spatial harmonisation has been removed from the extensions workflow (it
# was fast-track legacy code that depended on 2023 historical data the extensions
# don't have). Spatial alignment is now handled by the 2100-anchor block at the
# bottom of this file, which inherits fast-track's CEDS-2023-harmonised spatial
# pattern at 2100 directly.

# %%
# helper functions used across the post-processing blocks (timeseries corrections + 2100 alignment)
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

# helper used across the post-processing blocks (carries time_bnds / sector_bnds / etc. across read-modify-write cycles)
def copy_bounds_data_variables(
    source: xr.Dataset, target: xr.Dataset,
    bounds_vars = ['time_bnds', 'sector_bnds', 'level_bnds', 'lat_bnds', 'lon_bnds']
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
    
    # Copy the data variables
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

# helper function for int -> float encoding
def ensure_float_not_int(ds, vars = ["time_bnds", "sector"]):
    # Set encoding for variables to be saved as float64 in netCDF
    # (values remain as-is, but encoding specifies how to store in file)
    for v in vars:
        if v in ds:
            if v in ["time_bnds"]:
                ds[v].encoding['dtype'] = 'float64'
            if v in ["sector"]:
                ds[v] = ds[v].astype("float64")

    return ds

# helper function for bringing back in attributes for the main data variable (like 'BC_em_anthro')
def ensure_data_var_attrs(ds):
    # N.B. currently cannot be applied to VOC speciation data (as the DATA_HANDLES dictionary does not have that yet)

    # info
    vars = list(ds.data_vars)
    name = vars[0] # assumes the main data var is always the first one
    gas, rest = name.split("_", 1)
    handle = DATA_HANDLES[rest]

    # DO:
    # units: kg s-1 m-2
    # cell_methods: "time: mean"
    # long_name: ...
    ds = clean_var(ds, name, gas, handle)

    return ds

# %%
# load files for timeseries corrections

# areas of gridcells for calculating totals
areacella = xr.open_dataset(Path(settings.gridding_path, "areacella_input4MIPs_emissions_CMIP_CEDS-CMIP-2025-04-18_gn.nc"))
cell_area = areacella["areacella"]

# %%
if run_AIR_anthro_timeseries_correction:

    # files that are produced above, that may need correction
    files = [
        file
        for file in (settings.out_path / GRIDDING_VERSION).glob("*.nc")
        if "AIR" in file.name
    ]

    for file in files:
        gas_name, var, type_name = return_emission_names(file)
        if DO_GRIDDING_ONLY_FOR_THESE_SPECIES is not None and gas_name not in DO_GRIDDING_ONLY_FOR_THESE_SPECIES:
            print(f'Skipping em_AIR_anthro timeseries correction for {gas_name} (not in DO_GRIDDING_ONLY_FOR_THESE_SPECIES)')
            continue
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

        # Precondition: the extension gridded file must cover 2105-2500 at 5-year steps
        # (2100 is owned by fast-track and aligned separately in the 2100 alignment step).
        # The IAM proxy data must cover 2100-2500 at 5-year steps (including 2100, which
        # is used as the anchor for the alignment step). A gap in the gridded file means
        # there is no spatial distribution to scale; a gap in the proxy means the
        # correction ratio cannot be computed for that year.
        _expected_gridded_years = set(range(2105, 2501, 5))
        _expected_proxy_years = set(range(2100, 2501, 5))
        _gridded_years = set(gridded_emisssions_annual_totals.year.values.tolist())
        _iam_years = set(input_iam_annual_totals.year.values.tolist())
        missing_in_gridded = _expected_gridded_years - _gridded_years
        assert not missing_in_gridded, (
            f"Extension gridded file is missing expected 2105-2500 (5-year) years: "
            f"{sorted(missing_in_gridded)}. Check the gridding step's time coverage."
        )
        missing_in_proxy = _expected_proxy_years - _iam_years
        assert not missing_in_proxy, (
            f"IAM proxy data is missing expected 2100-2500 (5-year) years: "
            f"{sorted(missing_in_proxy)}. Check the extension IAM CSV."
        )

        # Calculate ratios (for each sector, for one emissions species)
        # Compare input IAM vs gridded emissions for each sector and year.
        # Align on inner join before xr.where (xr.where uses join='exact' internally,
        # which fails when sector or year coordinates don't match exactly).
        gridded_aligned, iam_aligned = xr.align(
            gridded_emisssions_annual_totals, input_iam_annual_totals, join='inner'
        )
        print(f"  AIR aligned: gridded {gridded_aligned.shape}, iam {iam_aligned.shape}")
        ratio_per_sector = xr.where(
            gridded_aligned != 0,
            iam_aligned / gridded_aligned,
            1.0  # No correction where gridded is zero
        )

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
        
        # Add global sums to metadata — pass the file's actual first/last years explicitly,
        # otherwise add_file_global_sum_totals_attrs defaults to first_year='2022' which
        # doesn't exist in the extension file (it spans 2100..2500) and raises KeyError.
        _file_years = sorted(set(int(t.year) for t in scen_ds_corrected.time.values))
        scen_ds_corrected = scen_ds_corrected.pipe(
            add_file_global_sum_totals_attrs,
            name=var,
            first_year=str(_file_years[0]),
            last_year=str(_file_years[-1]),
        )
        
        # Copy bounds variables from original dataset to ensure they exist
        copy_bounds_data_variables(source=scen_ds, target=scen_ds_corrected)
        
        # Save out updated data (overwrite the original file)
        outfile = file
        encoding = {var: {"zlib": True, "complevel": 2}}
        
        # Close source dataset to release file locks before overwriting the file
        scen_ds.close()
        
        # Remove old file before writing
        outfile.unlink(missing_ok=True)

        # ensure each file gets its own tracking_id and creation_timestamp
        scen_ds_corrected.attrs.update({
        "creation_date" : generate_creation_timestamp(),
        "tracking_id" : generate_tracking_id()
        })
        
        # Last metadata corrections
        scen_ds_corrected = (
            scen_ds_corrected
            .pipe(remove_fillvalue_from_bounds) # remove _FillValue from bounds
            .pipe(ensure_float_not_int) # helper function for int -> float encoding
            .pipe(ensure_data_var_attrs) # bringing back in attributes for the main data variable (like 'BC_em_anthro')
        )
        
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
        if DO_GRIDDING_ONLY_FOR_THESE_SPECIES is not None and gas_name not in DO_GRIDDING_ONLY_FOR_THESE_SPECIES:
            print(f'Skipping anthropogenic timeseries correction for {gas_name} (not in DO_GRIDDING_ONLY_FOR_THESE_SPECIES)')
            continue
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

        # Precondition: the extension gridded file must cover 2105-2500 at 5-year steps
        # (2100 is owned by fast-track and aligned separately in the 2100 alignment step).
        # The IAM proxy data must cover 2100-2500 at 5-year steps (including 2100, which
        # is used as the anchor for the alignment step). A gap in the gridded file means
        # there is no spatial distribution to scale; a gap in the proxy means the
        # correction ratio cannot be computed for that year.
        _expected_gridded_years = set(range(2105, 2501, 5))
        _expected_proxy_years = set(range(2100, 2501, 5))
        _gridded_years = set(gridded_emisssions_annual_totals.year.values.tolist())
        _iam_years = set(input_iam_annual_totals.year.values.tolist())
        missing_in_gridded = _expected_gridded_years - _gridded_years
        assert not missing_in_gridded, (
            f"Extension gridded file is missing expected 2105-2500 (5-year) years: "
            f"{sorted(missing_in_gridded)}. Check the gridding step's time coverage."
        )
        missing_in_proxy = _expected_proxy_years - _iam_years
        assert not missing_in_proxy, (
            f"IAM proxy data is missing expected 2100-2500 (5-year) years: "
            f"{sorted(missing_in_proxy)}. Check the extension IAM CSV."
        )

        # Calculate ratios (for each sector, for one emissions species)
        # Compare input IAM vs gridded emissions for each sector and year.
        # Align on inner join before xr.where (xr.where uses join='exact' internally,
        # which fails when sector or year coordinates don't match exactly).
        gridded_aligned, iam_aligned = xr.align(
            gridded_emisssions_annual_totals, input_iam_annual_totals, join='inner'
        )
        print(f"  anthro aligned: gridded {gridded_aligned.shape}, iam {iam_aligned.shape}")
        ratio_per_sector = xr.where(
            gridded_aligned != 0,
            iam_aligned / gridded_aligned,
            1.0  # No correction where gridded is zero
        )

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

        # Add global sums to metadata — pass the file's actual first/last years explicitly,
        # otherwise add_file_global_sum_totals_attrs defaults to first_year='2022' which
        # doesn't exist in the extension file (it spans 2100..2500) and raises KeyError.
        _file_years = sorted(set(int(t.year) for t in scen_ds_corrected.time.values))
        scen_ds_corrected = scen_ds_corrected.pipe(
            add_file_global_sum_totals_attrs,
            name=var,
            first_year=str(_file_years[0]),
            last_year=str(_file_years[-1]),
        )

        # Copy bounds variables from original dataset to ensure they exist
        copy_bounds_data_variables(source=scen_ds, target=scen_ds_corrected)

        # Save out updated data (overwrite the original file)
        outfile = file
        encoding = {var: {"zlib": True, "complevel": 2}}

        # Close source dataset to release file locks before overwriting the file
        scen_ds.close()

        # Remove old file before writing
        outfile.unlink(missing_ok=True)

        # ensure each file gets its own tracking_id and creation_timestamp
        scen_ds_corrected.attrs.update({
        "creation_date" : generate_creation_timestamp(),
        "tracking_id" : generate_tracking_id()
        })

        # Last metadata corrections
        scen_ds_corrected = (
            scen_ds_corrected
            .pipe(remove_fillvalue_from_bounds) # remove _FillValue from bounds
            .pipe(ensure_float_not_int) # helper function for int -> float encoding
            .pipe(ensure_data_var_attrs) # bringing back in attributes for the main data variable (like 'BC_em_anthro')
        )

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
        if DO_GRIDDING_ONLY_FOR_THESE_SPECIES is not None and gas_name not in DO_GRIDDING_ONLY_FOR_THESE_SPECIES:
            print(f'Skipping openburning timeseries correction for {gas_name} (not in DO_GRIDDING_ONLY_FOR_THESE_SPECIES)')
            continue
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

        # Precondition: the extension gridded file must cover 2105-2500 at 5-year steps
        # (2100 is owned by fast-track and aligned separately in the 2100 alignment step).
        # The IAM proxy data must cover 2100-2500 at 5-year steps (including 2100, which
        # is used as the anchor for the alignment step). A gap in the gridded file means
        # there is no spatial distribution to scale; a gap in the proxy means the
        # correction ratio cannot be computed for that year.
        _expected_gridded_years = set(range(2105, 2501, 5))
        _expected_proxy_years = set(range(2100, 2501, 5))
        _gridded_years = set(gridded_emisssions_annual_totals.year.values.tolist())
        _iam_years = set(input_iam_annual_totals.year.values.tolist())
        missing_in_gridded = _expected_gridded_years - _gridded_years
        assert not missing_in_gridded, (
            f"Extension gridded file is missing expected 2105-2500 (5-year) years: "
            f"{sorted(missing_in_gridded)}. Check the gridding step's time coverage."
        )
        missing_in_proxy = _expected_proxy_years - _iam_years
        assert not missing_in_proxy, (
            f"IAM proxy data is missing expected 2100-2500 (5-year) years: "
            f"{sorted(missing_in_proxy)}. Check the extension IAM CSV."
        )

        # Calculate ratios (for each sector, for one emissions species)
        # Compare input IAM vs gridded emissions for each sector and year.
        # Align on inner join before xr.where (xr.where uses join='exact' internally,
        # which fails when sector or year coordinates don't match exactly).
        gridded_aligned, iam_aligned = xr.align(
            gridded_emisssions_annual_totals, input_iam_annual_totals, join='inner'
        )
        print(f"  openburning aligned: gridded {gridded_aligned.shape}, iam {iam_aligned.shape}")
        ratio_per_sector = xr.where(
            gridded_aligned != 0,
            iam_aligned / gridded_aligned,
            1.0  # No correction where gridded is zero
        )

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
        
        # Add global sums to metadata — pass the file's actual first/last years explicitly,
        # otherwise add_file_global_sum_totals_attrs defaults to first_year='2022' which
        # doesn't exist in the extension file (it spans 2100..2500) and raises KeyError.
        _file_years = sorted(set(int(t.year) for t in scen_ds_corrected.time.values))
        scen_ds_corrected = scen_ds_corrected.pipe(
            add_file_global_sum_totals_attrs,
            name=var,
            first_year=str(_file_years[0]),
            last_year=str(_file_years[-1]),
        )
        
        # Copy bounds variables from original dataset to ensure they exist
        copy_bounds_data_variables(source=scen_ds, target=scen_ds_corrected)
        
        # Save out updated data (overwrite the original file)
        outfile = file
        encoding = {var: {"zlib": True, "complevel": 2}}
        
        # Close source dataset to release file locks before overwriting the file
        scen_ds.close()
        
        # Remove old file before writing
        outfile.unlink(missing_ok=True)

        # ensure each file gets its own tracking_id and creation_timestamp
        scen_ds_corrected.attrs.update({
        "creation_date" : generate_creation_timestamp(),
        "tracking_id" : generate_tracking_id()
        })
        
        # Last metadata corrections
        scen_ds_corrected = (
            scen_ds_corrected
            .pipe(remove_fillvalue_from_bounds) # remove _FillValue from bounds
            .pipe(ensure_float_not_int) # helper function for int -> float encoding
            .pipe(ensure_data_var_attrs) # bringing back in attributes for the main data variable (like 'BC_em_anthro')
        )
        
        # Save corrected dataset
        scen_ds_corrected.to_netcdf(outfile, encoding=encoding)
        scen_ds_corrected.close()
        
        print(f"\nSaved corrected {gas_name} openburning emissions timeseries to {outfile}")


# %% [markdown]
# # 2100 alignment to fast-track gridded (additive offset with fade-out)
# ## *Stage 10b/13* · [↑ overview](#workflow-for-cmip7-scenariomip-emissions-harmonization--post-2100-extensions)
#
# For each extension gridded file (em_anthro, em_AIR_anthro, em_openburning), find the
# corresponding fast-track file (which ends at 2100-12). Per cell, per sector/level, per
# month-of-year, compute an additive offset that makes the extension at year 2100 exactly
# equal to fast-track at year 2100. Apply the offset across all extension years with a
# linear fade from 1.0 at year FADE_ANCHOR_YEAR (=2100) to 0 at year FADE_CONVERGENCE_YEAR
# (=2150). Drop the 2100 timestep at the end (fast-track owns 2100).
#
# Why additive (not multiplicative): cells that are zero in the extension at 2100 but
# non-zero in fast-track would blow up under a ratio. An additive offset handles zeros
# cleanly and fades to zero by construction.

# %%
def _match_fasttrack_file(ext_file: Path, gas: str, type_with_dashes: str,
                          fasttrack_dir: Path, marker: str, version: str) -> Path | None:
    """Locate the fast-track counterpart of an extension nc file (must end at 210012)."""
    # Try the canonical filename first
    canonical = (
        fasttrack_dir / f"{gas}-em-{type_with_dashes}_input4MIPs_emissions_"
        f"{cmip7_utils.DS_ATTRS['target_mip']}_"
        f"{cmip7_utils.DS_ATTRS['institution_id']}-{marker}-{version}_"
        f"{cmip7_utils.DS_ATTRS['grid_label']}_202201-210012.nc"
    )
    if canonical.exists():
        return canonical
    # Fall back to a glob across whatever filename convention v1_1 used
    candidates = list(fasttrack_dir.glob(
        f"{gas}-em-{type_with_dashes}_*_{marker}-{version}_*_*-210012.nc"
    ))
    return candidates[0] if candidates else None


if run_2100_alignment_to_fasttrack:
    print('\n=== 2100 alignment to fast-track gridded (additive offset, fade-out) ===')
    print(f'    fade window: full correction at {FADE_ANCHOR_YEAR} -> zero correction at {FADE_CONVERGENCE_YEAR}')
    print(f'    drop {FADE_ANCHOR_YEAR} from output: {DROP_ANCHOR_TIMESTEP}')

    fasttrack_dir = LOCATION_FASTTRACK_GRIDDED / f"{marker_to_run}_{VERSION_ESGF}"
    if not fasttrack_dir.exists():
        raise FileNotFoundError(
            f"Fast-track folder not found: {fasttrack_dir}\n"
            f"Set LOCATION_FASTTRACK_GRIDDED to point at the parent of '{marker_to_run}_{VERSION_ESGF}'."
        )

    # All main gridded files in the extension output (anthro, AIR_anthro, openburning).
    # Skip H2 openburning (derived from CO) and speciated VOC files (no fast-track 2100 counterpart).
    align_files = [
        f for f in (settings.out_path / GRIDDING_VERSION).glob("*.nc")
        if "speciated" not in f.name and "H2" not in f.name
    ]
    if DO_GRIDDING_ONLY_FOR_THESE_SPECIES is not None:
        species_set = set(DO_GRIDDING_ONLY_FOR_THESE_SPECIES)
        align_files = [f for f in align_files if f.name.split("-")[0] in species_set]
    if DO_GRIDDING_ONLY_FOR_THESE_SECTORS is not None:
        # type_name e.g. 'anthro', 'AIR-anthro', 'openburning'
        wanted = set(DO_GRIDDING_ONLY_FOR_THESE_SECTORS)
        # Map underscore variant in DO_GRIDDING_ONLY_FOR_THESE_SECTORS to dashed for matching
        wanted_dashed = {s.replace("_", "-") for s in wanted}
        align_files = [
            f for f in align_files
            if f.name.split("-em-", 1)[1].split("_", 1)[0] in wanted_dashed
        ]

    print(f"    files to align: {[f.name for f in align_files]}")

    # Diagnostic output directory (raw-2100 grids, plots). Only created if requested.
    if run_2100_alignment_diagnostic:
        diag2100_dir = version_path / "diagnostics_2100"
        diag2100_dir.mkdir(parents=True, exist_ok=True)
        print(f"    2100 diagnostic enabled -> {diag2100_dir}")

    # Step 0 diagnostic: compare global totals at 2100 between fast-track and extension.
    # Per user instruction, this is a precondition sanity check — if globals at 2100 are
    # very different, the alignment will absorb the difference (good) but the magnitude
    # signals upstream CSV inconsistencies (worth surfacing).
    diagnostic_rows = []

    for file in tqdm(align_files, desc="2100 alignment"):
        gas_name, var, type_name = return_emission_names(file)  # type_name has dashes, e.g. 'AIR-anthro'
        ft_match = _match_fasttrack_file(
            ext_file=file,
            gas=gas_name,
            type_with_dashes=type_name,
            fasttrack_dir=fasttrack_dir,
            marker=marker_to_run,
            version=VERSION_ESGF,
        )
        if ft_match is None:
            print(f"  ! no fast-track counterpart for {file.name}; skipping")
            continue

        print(f"  aligning {file.name}")
        print(f"     -> using fast-track: {ft_match.name}")

        ext_ds = xr.open_dataset(file)
        ft_ds = xr.open_dataset(ft_match)

        # require 2100 in both
        ext_years = np.array([t.year for t in ext_ds.time.values])
        ft_years = np.array([t.year for t in ft_ds.time.values])
        if not (ext_years == FADE_ANCHOR_YEAR).any():
            print(f"     ! ext file missing year {FADE_ANCHOR_YEAR}; skipping")
            ext_ds.close(); ft_ds.close(); continue
        assert (ft_years == FADE_ANCHOR_YEAR).any(), (
            f"Fast-track file {ft_match.name} is missing year {FADE_ANCHOR_YEAR}. "
            f"The 2100 alignment requires fast-track gridded data at {FADE_ANCHOR_YEAR}."
        )

        ext_2100 = ext_ds[var].isel(time=np.where(ext_years == FADE_ANCHOR_YEAR)[0])
        ft_2100 = ft_ds[var].isel(time=np.where(ft_years == FADE_ANCHOR_YEAR)[0])
        assert ext_2100.time.size == 12 and ft_2100.time.size == 12, (
            f"Expected 12 monthly slices at {FADE_ANCHOR_YEAR}; "
            f"got ext={ext_2100.time.size}, ft={ft_2100.time.size}"
        )

        # Sort both 2100 slices by month-of-year (defensive — should already be 1..12 in order)
        ext_months = np.array([t.month for t in ext_2100.time.values])
        ft_months = np.array([t.month for t in ft_2100.time.values])
        ext_2100_sorted = ext_2100.isel(time=np.argsort(ext_months))
        ft_2100_sorted = ft_2100.isel(time=np.argsort(ft_months))

        # Step 0 diagnostic: global total at 2100 (kg s-1 m-2 cell -> just sum the raw grid;
        # the comparison is relative, not in physical units, so cell_area weighting is
        # informative but not required for a sanity check).
        ext_2100_globalsum = float(ext_2100_sorted.sum().values)
        ft_2100_globalsum = float(ft_2100_sorted.sum().values)
        if abs(ft_2100_globalsum) > 0:
            rel_diff_pct = 100.0 * (ext_2100_globalsum - ft_2100_globalsum) / ft_2100_globalsum
        else:
            rel_diff_pct = float("inf") if abs(ext_2100_globalsum) > 0 else 0.0
        print(f"     step-0 sanity: ext_2100_globalsum={ext_2100_globalsum:.6g}, "
              f"ft_2100_globalsum={ft_2100_globalsum:.6g}, rel_diff={rel_diff_pct:.3f}%")
        diagnostic_rows.append({
            "file": file.name,
            "gas": gas_name,
            "type": type_name,
            "ext_2100_globalsum_raw": ext_2100_globalsum,
            "ft_2100_globalsum_raw": ft_2100_globalsum,
            "rel_diff_pct": rel_diff_pct,
        })

        # Build per-month offset: shape (12, lat, lon, [sector|level])
        offset_by_month_np = ft_2100_sorted.values - ext_2100_sorted.values

        # ─── 2100 verification diagnostic, part 1/2 (RAW, pre-correction) ─────
        # Capture ext_2100 (raw) vs ft_2100 and their diff BEFORE the additive offset is
        # applied. The diagnostic netCDF is written later (part 2/2, after the correction)
        # so it can ALSO carry the corrected 2100 slice for a zero-difference check. Here
        # we compute the raw metrics + PNG plots and stash the raw arrays.
        _diag2100_payload = None  # reset every file; set on success below
        if run_2100_alignment_diagnostic:
            try:
                # Put all vars on a single, identical coordinate set (ext's), so differing
                # cftime calendars between ext/ft don't trigger NaN alignment.
                ext_da = ext_2100_sorted
                ft_da = xr.DataArray(
                    ft_2100_sorted.values, dims=ext_da.dims, coords=ext_da.coords,
                    attrs=ext_da.attrs,
                )
                diff_da = xr.DataArray(
                    offset_by_month_np, dims=ext_da.dims, coords=ext_da.coords,
                    attrs={"long_name": "fast-track minus raw extension at 2100",
                           "units": ext_da.attrs.get("units", "")},
                )
                diag_out = diag2100_dir / f"{gas_name}-em-{type_name}_2100diagnostic_{settings.version}.nc"

                # Spatial-agreement metrics (appended to the step-0 CSV row).
                ext_vals = ext_da.values
                ft_vals = ft_da.values
                diff_vals = offset_by_month_np
                flat_e = ext_vals.ravel()
                flat_f = ft_vals.ravel()
                fin = np.isfinite(flat_e) & np.isfinite(flat_f)
                spatial_corr = (
                    float(np.corrcoef(flat_e[fin], flat_f[fin])[0, 1])
                    if fin.sum() > 1 else float("nan")
                )
                # Area-weighted totals (cell_area broadcasts over lat/lon by name).
                ext_area_tot = float((ext_da * cell_area).sum().values)
                ft_area_tot = float((ft_da * cell_area).sum().values)
                aw_rel_diff_pct = (
                    100.0 * (ext_area_tot - ft_area_tot) / ft_area_tot
                    if abs(ft_area_tot) > 0 else float("nan")
                )
                diagnostic_rows[-1].update({
                    "max_abs_diff": float(np.nanmax(np.abs(diff_vals))),
                    "mean_abs_diff": float(np.nanmean(np.abs(diff_vals))),
                    "rmse": float(np.sqrt(np.nanmean(diff_vals ** 2))),
                    "area_weighted_rel_diff_pct": aw_rel_diff_pct,
                    "spatial_corr": spatial_corr,
                    "n_cells_ext_zero_ft_nonzero": int(
                        ((np.abs(ext_vals) < 1e-30) & (np.abs(ft_vals) > 1e-30)).sum()
                    ),
                })

                # PNG plots: ext / ft / diff maps (sector-summed, annual-mean) + a
                # per-sector area-weighted global-total bar comparison.
                from concordia.cmip7.utils_plotting import plot_map

                def _to_map(da):
                    d = da
                    for extra in ("sector", "level"):
                        if extra in d.dims:
                            d = d.sum(extra)
                    return d.mean("time")

                stub = str(diag2100_dir / f"{gas_name}-em-{type_name}_2100")
                plot_map(_to_map(ext_da), title=f"{gas_name} {type_name} — raw extension 2100",
                         save_as="png", filename=f"{stub}_ext_raw")
                plot_map(_to_map(ft_da), title=f"{gas_name} {type_name} — fast-track 2100",
                         save_as="png", filename=f"{stub}_ft")
                plot_map(_to_map(diff_da), title=f"{gas_name} {type_name} — ft minus ext (2100)",
                         save_as="png", filename=f"{stub}_diff", cmap="RdBu_r")
                plt.close("all")

                fig, ax = plt.subplots(figsize=(9, 5))
                if "sector" in ext_da.dims:
                    sectors = [str(s) for s in ext_da["sector"].values]
                    ext_tot = (ext_da * cell_area).sum(("time", "lat", "lon")).values
                    ft_tot = (ft_da * cell_area).sum(("time", "lat", "lon")).values
                    x = np.arange(len(sectors)); w = 0.4
                    ax.bar(x - w / 2, ext_tot, w, label="extension (raw 2100)")
                    ax.bar(x + w / 2, ft_tot, w, label="fast-track 2100")
                    ax.set_xticks(x); ax.set_xticklabels(sectors, rotation=45, ha="right")
                else:
                    ax.bar([0, 1], [ext_area_tot, ft_area_tot])
                    ax.set_xticks([0, 1]); ax.set_xticklabels(["extension raw", "fast-track"])
                ax.set_ylabel("area-weighted 2100 total (sum over 12 months)")
                ax.set_title(f"{gas_name} {type_name} — 2100 totals: extension vs fast-track")
                ax.legend()
                fig.savefig(f"{stub}_globaltotals.png", dpi=150, bbox_inches="tight")
                plt.close(fig)

                # Stash raw arrays for part 2/2 (netCDF write after correction).
                _diag2100_payload = {
                    "ext_da": ext_da, "ft_da": ft_da, "diff_da": diff_da,
                    "diag_out": diag_out,
                }
                print(f"     [diag] computed raw-2100 metrics + plots for {gas_name}-{type_name}")
            except Exception as e:
                print(f"     ! [diag] 2100 raw diagnostic failed ({e}); continuing")

        # Fast path: if the 2100 offset is below physical-emission noise (~1e-18 kg/m²/s,
        # ~10 orders of magnitude below realistic emission magnitudes), skip the heavy
        # correction and just (optionally) drop 2100 + rewrite metadata. Catches the
        # exact-zero case (e.g. VL CO2-em-AIR-anthro) and any future cases where ext and
        # ft agree at float-precision-noise level. Costs nothing; may gain compute later.
        FAST_PATH_OFFSET_THRESHOLD = 1e-18
        offset_max_abs = float(np.max(np.abs(offset_by_month_np)))
        offset_is_zero = offset_max_abs < FAST_PATH_OFFSET_THRESHOLD
        if offset_is_zero:
            print(f"     max|offset| = {offset_max_abs:.3e} < {FAST_PATH_OFFSET_THRESHOLD:.0e}; "
                  f"skipping correction; applying drop-2100 + metadata refresh only")

        # Vectorise application across all extension timesteps:
        #   offset(t, ...) = offset_by_month[month_of(t) - 1, ...]
        all_months = np.array([t.month for t in ext_ds.time.values], dtype=np.int64)

        # Fade factor — integer-year basis so that year=FADE_ANCHOR_YEAR has fade=1.0 across
        # all 12 months (i.e. 2100 aligns *exactly* to fast-track). Because the extension is
        # sampled at 5-yearly steps, the realised fade values are 1.0 (2100), 0.9 (2105),
        # 0.8 (2110), ..., 0.1 (2145), 0 (2150+).
        all_years_int = np.array([t.year for t in ext_ds.time.values], dtype=float)
        fade_vals = np.clip(
            (FADE_CONVERGENCE_YEAR - all_years_int) / (FADE_CONVERGENCE_YEAR - FADE_ANCHOR_YEAR),
            0.0,
            1.0,
        )  # shape: (T,)

        target_dims = ext_ds[var].dims
        time_axis = target_dims.index("time")
        broadcast_shape = [1] * len(target_dims)
        broadcast_shape[time_axis] = -1
        fade_broadcast = fade_vals.reshape(broadcast_shape)

        ext_corr = ext_ds.copy(deep=True)
        ext_corr[var].attrs = ext_ds[var].attrs.copy()

        if not offset_is_zero:
            # gather along axis-0 then add to the original variable (numpy is fine for the
            # sizes seen in practice — sector-dim files are ~10 GB float32 per array)
            offset_full = offset_by_month_np[all_months - 1]  # shape: (T, lat, lon, [sector|level])
            corrected_np = ext_ds[var].values + offset_full * fade_broadcast
            ext_corr[var] = (target_dims, corrected_np)
            ext_corr[var].attrs = ext_ds[var].attrs.copy()

        # Sanity: at FADE_ANCHOR_YEAR the corrected ext should equal ft (modulo fp).
        verify_2100 = ext_corr[var].isel(time=np.where(ext_years == FADE_ANCHOR_YEAR)[0])
        verify_2100_sorted = verify_2100.isel(time=np.argsort(
            np.array([t.month for t in verify_2100.time.values])
        ))
        max_abs_diff_at_2100 = float(np.max(np.abs(
            verify_2100_sorted.values - ft_2100_sorted.values
        )))
        print(f"     verify: max|ext_corr_2100 - ft_2100| = {max_abs_diff_at_2100:.3e} (expect ~0)") # should confirm that the correction has done its job at 2100

        # ─── 2100 verification diagnostic, part 2/2 (write netCDF incl. CORRECTED) ─
        # Now that the correction has been applied, persist BOTH the raw and the corrected
        # 2100 slices (plus their diffs vs fast-track) to the diagnostic netCDF. This is the
        # only place the corrected 2100 survives — the next step drops it from the output.
        # Module K of check_gridded_scenario_junctions-ext.py reads this to confirm the
        # post-correction difference is genuinely ~0 before 2100 is dropped.
        if run_2100_alignment_diagnostic and _diag2100_payload is not None:
            try:
                ext_da = _diag2100_payload["ext_da"]
                ft_da = _diag2100_payload["ft_da"]
                diff_da = _diag2100_payload["diff_da"]
                diag_out = _diag2100_payload["diag_out"]

                # Corrected 2100 on ext's coordinate set (verify_2100_sorted comes from the
                # real ext_corr that is written to disk, so this validates the actual pipeline).
                corr_da = xr.DataArray(
                    verify_2100_sorted.values, dims=ext_da.dims, coords=ext_da.coords,
                    attrs=ext_da.attrs,
                )
                diff_corr_da = xr.DataArray(
                    ft_da.values - verify_2100_sorted.values, dims=ext_da.dims,
                    coords=ext_da.coords,
                    attrs={"long_name": "fast-track minus CORRECTED extension at 2100 (expect ~0)",
                           "units": ext_da.attrs.get("units", "")},
                )

                diag_ds = xr.Dataset({
                    "ext_2100_raw": ext_da,
                    "ext_2100_corrected": corr_da,
                    "ft_2100": ft_da,
                    "diff_ft_minus_ext": diff_da,
                    "diff_ft_minus_ext_corrected": diff_corr_da,
                })
                diag_ds.attrs.update({
                    "comment": (
                        f"Extension 2100 vs fast-track 2100 ({ft_match.name}). "
                        f"'ext_2100_raw' is pre-correction; 'ext_2100_corrected' is post "
                        f"additive-offset (what is written to the output before 2100 is "
                        f"dropped). 'diff_ft_minus_ext' = ft - raw (= the applied offset); "
                        f"'diff_ft_minus_ext_corrected' = ft - corrected (expect ~0). "
                        f"12 monthly slices."
                    ),
                    "source_extension_file": file.name,
                    "source_fasttrack_file": ft_match.name,
                    "max_abs_diff_corrected_at_2100": max_abs_diff_at_2100,
                })
                diag_ds = ensure_float_not_int(diag_ds)
                diag_ds.to_netcdf(
                    diag_out,
                    encoding={v: {"zlib": True, "complevel": 2} for v in diag_ds.data_vars},
                )
                diag_ds.close()
                # Record the post-correction zero-check in the step-0 CSV row too.
                diagnostic_rows[-1]["max_abs_diff_corrected"] = max_abs_diff_at_2100
                print(f"     [diag] saved 2100 diagnostic (raw + corrected) -> {diag_out.name}")
            except Exception as e:
                print(f"     ! [diag] 2100 corrected-diagnostic write failed ({e}); continuing")

        # Drop the anchor timestep so the output starts at FADE_ANCHOR_YEAR + 5 (i.e. 2105).
        # Note: `ext_corr` came from `ext_ds.copy(deep=True)`, so it already carries the
        # bounds variables (time_bnds, lat_bnds, lon_bnds, sector_bnds/level_bnds). `isel`
        # trims `time_bnds` together with the main variable, since both share the time dim.
        if DROP_ANCHOR_TIMESTEP:
            keep_idx = np.where(ext_years != FADE_ANCHOR_YEAR)[0]
            ext_corr = ext_corr.isel(time=keep_idx)

        # Make sure all dataset-level attributes from the original file are preserved
        copy_attributes(source=ext_ds, target=ext_corr)

        # Materialise any remaining lazy reads from the source netCDF (notably the bounds
        # variables) so it is safe to close and overwrite the source file below.
        ext_corr = ext_corr.load()

        # Append a comment documenting the alignment
        existing_comment = ext_corr.attrs.get("comment", "")
        align_note = (
            f" 2100 gridded values aligned to fast-track "
            f"({ft_match.name}) via per-cell, per-month additive offset with linear fade "
            f"from year {FADE_ANCHOR_YEAR} (full correction) to year {FADE_CONVERGENCE_YEAR} "
            f"(zero correction)"
            + (f"; anchor timestep {FADE_ANCHOR_YEAR} dropped from output." if DROP_ANCHOR_TIMESTEP else ".")
        )
        ext_corr.attrs["comment"] = (existing_comment.rstrip(". ") + "." + align_note).strip()

        # Fresh tracking id + creation timestamp
        ext_corr.attrs.update({
            "creation_date": generate_creation_timestamp(),
            "tracking_id": generate_tracking_id(),
        })

        # Determine first/last year for global-total metadata after the trim
        kept_years = sorted(set(int(t.year) for t in ext_corr.time.values))
        first_year_str = str(kept_years[0])
        last_year_str = str(kept_years[-1])

        # Standard final touches (mirror the existing post-processing pattern)
        ext_corr = (
            ext_corr
            .pipe(remove_fillvalue_from_bounds)
            .pipe(ensure_float_not_int)
            .pipe(ensure_data_var_attrs)
        )
        try:
            ext_corr = add_file_global_sum_totals_attrs(
                ext_corr, name=var, first_year=first_year_str, last_year=last_year_str,
            )
        except Exception as e:
            print(f"     ! could not add global-total attrs ({e}); continuing")

        # Save (overwrite)
        encoding = {var: {"zlib": True, "complevel": 2}}
        ext_ds.close()
        ft_ds.close()
        outfile = file
        outfile.unlink(missing_ok=True)
        ext_corr.to_netcdf(outfile, encoding=encoding)
        ext_corr.close()
        print(f"     saved -> {outfile.name}")

    # Persist Step 0 diagnostic table for later inspection
    if diagnostic_rows:
        diag_df = pd.DataFrame(diagnostic_rows)
        diag_path = version_path / f"step0_2100_globalsum_diagnostic_{settings.version}.csv"
        diag_df.to_csv(diag_path, index=False)
        print(f"\n    Step 0 diagnostic table saved to: {diag_path}")


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
def update_var_attrs(ds, var, **attrs):
    ds[var].attrs.update(attrs)
    return ds


# %%
def add_sector_bounds(ds, source_ds=None):

    if source_ds is None:
        return ds

    if "sector_bnds" not in source_ds:
        return ds

    # Copy the bounds variable
    ds = ds.assign({
        "sector_bnds": source_ds["sector_bnds"]
    })

    # CF convention: link bounds to coordinate
    if "sector" in ds.coords:
        ds["sector"].attrs["bounds"] = "sector_bnds"

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
    co_openburning.close()
    
    # Load the H2/CO emission factor translation file
    h2_translation_file = settings.proxy_path / "EF_h2_div_EF_co.nc"
    h2_translation = xr.open_dataset(h2_translation_file)
    h2_translation = _to_sector_integers_and_reorder(h2_translation)

    h2_openburning = xr.Dataset(
        coords={**co_openburning.coords},
        attrs=co_openburning.attrs.copy()
    )
        
    # Initialise variable
    gas_variable_name = "H2_em_openburning"
    h2_openburning[gas_variable_name] = xr.zeros_like(
        co_openburning["CO_em_openburning"]
    )

    h2_openburning = add_sector_bounds(h2_openburning, co_openburning)

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
                h2_openburning.isel(time=time_idx, sector=sector_idx)[gas_variable_name].values = (co_slice * translation_slice).values
                
                # Assert that the sectors all align, ignoring dtype
                assert h2_openburning.isel(time=time_idx, sector=sector_idx).sector.values == co_slice.sector.values
                assert h2_openburning.isel(time=time_idx, sector=sector_idx).sector.values == translation_slice.sector.values


    # copy & update attributes
    h2_openburning.attrs.update(co_openburning.attrs)
    # Update attributes
    handle = 'openburning'
    gas = 'H2'
    long_name = f"{gas} {handle} emissions"
    h2_openburning.attrs['variable_id'] = gas_variable_name
    h2_openburning.attrs['title'] = f"Future {handle} emissions of H2 in {experiment_name}"
    h2_openburning.attrs['reporting_unit'] = f"Mass flux of {gas_variable_name}"

    # add individual tracking_id and creation_date
    h2_openburning.attrs.update({
        "creation_date" : generate_creation_timestamp(),
        "tracking_id" : generate_tracking_id()
        })

    # Add global sums as metadata
    # Add global sums to metadata — pass the file's actual first/last years explicitly,
    # otherwise add_file_global_sum_totals_attrs defaults to first_year='2022' which
    # doesn't exist in the extension file (it spans 2100..2500) and raises KeyError.
    _file_years = sorted(set(int(t.year) for t in h2_openburning.time.values))
    h2_openburning = h2_openburning.pipe(
        add_file_global_sum_totals_attrs,
        name=f"{gas_variable_name}",
        first_year=str(_file_years[0]),
        last_year=str(_file_years[-1]),
    )
    # Add bounds
    h2_openburning = (
        h2_openburning
        .pipe(
            update_var_attrs,
            gas_variable_name,
            units="kg s-1 m-2",
            cell_methods="time: mean",
            long_name=long_name,
        )
        .pipe(add_lon_lat_bounds) # add lat/lon bnds
        .pipe(add_time_bounds)
        .pipe(remove_fillvalue_from_bounds)
        .pipe(ensure_float_not_int)
    )  

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
    h2_openburning.close()


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

        gas = voc_share.gas.values[0]
        # construct output variable name
        gas_variable_name = (
            f"NMVOC_{gas}_em_speciated_VOC_openburning"
        )

        # build output dataset
        voc_spec = voc_spec_data.to_dataset(name=gas_variable_name)

        # copy & update attributes
        voc_spec.attrs.update(voc_openburning.attrs)
        # Update attributes
        handle = 'openburning'
        long_name = f"{gas} {handle} speciated emissions"

        voc_spec.attrs['variable_id'] = gas_variable_name
        voc_spec.attrs['title'] = f"Future {handle} emissions of speciated {gas_variable_name} in {experiment_name}"
        voc_spec.attrs['reporting_unit'] = f"Mass flux of {gas_variable_name}"

        # add individual tracking_id and creation_date
        voc_spec.attrs.update({
        "creation_date" : generate_creation_timestamp(),
        "tracking_id" : generate_tracking_id()
        })
    
        # Add global sums as metadata
        # Add global sums to metadata — pass the file's actual first/last years explicitly,
        # otherwise add_file_global_sum_totals_attrs defaults to first_year='2022' which
        # doesn't exist in the extension file (it spans 2100..2500) and raises KeyError.
        _file_years = sorted(set(int(t.year) for t in voc_spec.time.values))
        voc_spec = voc_spec.pipe(
            add_file_global_sum_totals_attrs,
            name=f"{gas_variable_name}",
            first_year=str(_file_years[0]),
            last_year=str(_file_years[-1]),
        )
        # add sector bounds
        voc_spec = add_sector_bounds(voc_spec, voc_openburning)
        # Add bounds
        voc_spec = (
            voc_spec
            .pipe(
                update_var_attrs,
                gas_variable_name,
                units="kg s-1 m-2",
                cell_methods="time: mean",
                long_name=long_name,
            )
            .pipe(add_lon_lat_bounds) # add lat/lon bnds
            .pipe(add_time_bounds)
            .pipe(remove_fillvalue_from_bounds)
            .pipe(ensure_float_not_int)
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
        gas = " ".join(gas_variable_name.split("_", 2)[:2])
        
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

        # copy & update attributes
        voc_spec.attrs.update(voc_anthro.attrs)
        # Update attributes
        handle = 'anthropogenic'
        long_name = f"{gas} {handle} speciated emissions"

        voc_spec.attrs['variable_id'] = gas_variable_name
        voc_spec.attrs['title'] = f"Future {handle} emissions of speciated {gas_variable_name} in {experiment_name}"
        voc_spec.attrs['reporting_unit'] = f"Mass flux of {gas_variable_name}"

        # add individual tracking_id and creation_date
        voc_spec.attrs.update({
        "creation_date" : generate_creation_timestamp(),
        "tracking_id" : generate_tracking_id()
        })
        
        # Add global sums as metadata
        # Add global sums to metadata — pass the file's actual first/last years explicitly,
        # otherwise add_file_global_sum_totals_attrs defaults to first_year='2022' which
        # doesn't exist in the extension file (it spans 2100..2500) and raises KeyError.
        _file_years = sorted(set(int(t.year) for t in voc_spec.time.values))
        voc_spec = voc_spec.pipe(
            add_file_global_sum_totals_attrs,
            name=f"{gas_variable_name}",
            first_year=str(_file_years[0]),
            last_year=str(_file_years[-1]),
        )
        # add sector bounds
        voc_spec = add_sector_bounds(voc_spec, voc_anthro)
        # Add bounds
        voc_spec = (
            voc_spec
            .pipe(
                update_var_attrs,
                gas_variable_name,
                units="kg s-1 m-2",
                cell_methods="time: mean",
                long_name=long_name,
            )
            .pipe(add_lon_lat_bounds) # add lat/lon bnds
            .pipe(add_time_bounds)
            .pipe(remove_fillvalue_from_bounds)
            .pipe(ensure_float_not_int)
        )

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
PLOT_GASES: list[str] | None = ["CO2", "NH3"] # e.g. ["CO2", "SO2", "VOC01_alcohols", "VOC02_ethane", "NMVOC-C2H2", "NMVOC-C10H16"]; default is run all

PLOT_SECTORS: list[str] | None = None # e.g. ['Energy', 'Residential, Commercial, Other'] default is run all


# %%

if PLOT_SECTORS is None:
    PLOT_SECTORS = np.unique(SECTOR_ORDERING_DEFAULT["CO2_em_anthro"] + SECTOR_ORDERING_DEFAULT["em_anthro"] + SECTOR_ORDERING_DEFAULT["em_openburning"])

if PLOT_GASES is None:
    PLOT_GASES = np.unique(GASES_ESGF_CEDS + GASES_ESGF_BB4CMIP)



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
# NOTE: takes about ~1 min per figure

# folder_plots = settings.out_path / GRIDDING_VERSION / "plots"
# folder_plots.mkdir(parents=True, exist_ok=True)
# anthro_bb_air="CEDS_anthro" # CEDS_anthro, BB4CMIP7, CEDS_AIR

# TIMES = [
#     cftime.DatetimeNoLeap(2100, 1, 16)
#     # cftime.DatetimeNoLeap(2023, 6, 16),
#     # cftime.DatetimeNoLeap(2023, 12, 16)
# ]



# for file in tqdm((settings.out_path / GRIDDING_VERSION).glob("*.nc"), "Plot maps: diffs with CEDS in 2100"): # loop over all produced files    
#     gas_name, var, type_name = return_emission_names(file)

#     for t in TIMES:
#         if gas_name in PLOT_GASES:
#             if type_name == "anthro":

#                 scen_ds = xr.open_dataset(file).sel(time=t).pipe(reorder_dimensions)
#                 scen_da = scen_ds[f"{gas_name}_em_anthro"]

#                 match = next(ceds_data_location.glob(f"{gas_name}-*.nc"), None)

#                 ceds_ds = xr.open_dataset(match).sel(time=t).pipe(reorder_dimensions) # TODO: for CO2, add zeroes for CDR sectors in ceds history for plotting
#                 ceds_da = ceds_ds[f"{gas_name}_em_anthro"]
#                 if gas_name == 'CO2':
#                     ceds_da = xr.concat([
#                         ceds_da,
#                         xr.zeros_like(scen_da.sel(sector=[8,9]))
#                     ], dim="sector")

#                 # print(N2O)

#                 AVAILABLE_SECTORS = [k for k in scen_ds.sector.values if SECTOR_DICT_ANTHRO_CO2_SCENARIO[k] in PLOT_SECTORS]

#                 fig = plot_ceds_vs_scenario_comparison(
#                     ceds_da=ceds_da,
#                     scen_da=scen_da,
#                     gas=gas_name,
#                     sectors=AVAILABLE_SECTORS,
#                     # We pass the desired time slice (a single cftime object)
#                     time_slice=t,
#                     anthro_bb_air=anthro_bb_air,
#                     colour_scale_max_percentile=98,
#                     empty_treatment="fill_zeroes"
#                 )

#                 # --- SAVE AND SHOW PLOT ---
#                 if fig is not None:
#                     filename_base = f"ceds_vs_scenario_comparison_{gas_name}_{t.strftime('%Y%m%d')}"

#                     # Save the plot
#                     fig.savefig(folder_plots / f"{filename_base}.png", dpi=200, bbox_inches='tight')
#                     # fig.savefig(folder_plots / f"{filename_base}.pdf", bbox_inches='tight')

#                     plt.show()
#                     plt.close(fig) # Close the figure to free memory





# %% [markdown]
# # CONTINUED POSTPROCESSING
# ## 3. writing out some check files
#
#

# %%
# Total emissions (<1min per file)
save_total_emissions_as_csv = True
CALCULATE_TOTALS_GASES: list[str] | None = None # e.g. ["CO2", "SO2", "VOC01_alcohols", "VOC02_ethane", "NMVOC-C2H2", "NMVOC-C10H16"]; default is run all
CALCULATE_TOTALS_GASES: list[str] | None = ["CO2", "NH3"] # e.g. ["CO2", "SO2", "VOC01_alcohols", "VOC02_ethane", "NMVOC-C2H2", "NMVOC-C10H16"]; default is run all


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
    # 'London': (51.5074, -0.1278),
    # 'Tokyo': (35.6762, 139.6503),
    # 'São_Paulo': (-23.5505, -46.6333),
    # 'Lagos': (6.5244, 3.3792),
    'South Sudan': (6.8770, 31.3070),
    # 'Mumbai': (19.0760, 72.8777),
    # 'Rural_Amazon': (-3.4653, -62.2159),  # Remote area in Amazon
    # 'North_Atlantic': (45.0, -30.0),     # Shipping route
    # 'South_China_Sea': (12.0, 113.0)     # Shipping route
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
new_stem

# %%
folder_totals = settings.out_path / GRIDDING_VERSION / "check_NMVOC_sums"

# %%
combined_df = pd.read_csv(folder_totals / f"{new_stem}_combined-annual-totals.csv", index_col=["gas", "sector"])
combined_df

# %%
# test that the speciated NMVOC species add up to the bulk NMVOC

# drop the bulk from the df
combined_df_filtered = combined_df.loc[combined_df.index.get_level_values("gas") != "NMVOCbulk"]
# add the speciated up by sector
speciated_totals = combined_df_filtered.groupby(level=["sector"]).sum()
# isolate the bulk and process similarly to get df in same format
bulk_totals = combined_df.loc[combined_df.index.get_level_values("gas") == "NMVOCbulk"].groupby(level=["sector"]).sum()

# test that they are equal
pd.testing.assert_frame_equal(speciated_totals,
    bulk_totals,
    check_exact=False,
    rtol=1e-3)

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
    var = "NMVOC_em_anthro"
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

# test that structure is equal
pd.testing.assert_index_equal(speciated_totals.index, bulk_totals.index)
pd.testing.assert_index_equal(speciated_totals.columns, bulk_totals.columns)

# Approximate value check
pd.testing.assert_frame_equal(
    speciated_totals,
    bulk_totals,
    check_exact=False,
    rtol=1e-3
)

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

g = sns.relplot(
    data=plot_df,
    x="year",
    y="emissions",
    hue="variant",
    col="sector",
    col_wrap=3,
    kind="line",    height=4,
    aspect=1.6,
    facet_kws={"sharey": False}
)

g.savefig(folder_totals / f"{gas}_{new_stem}_reaggregated-comparison.png")
plt.show()

# %% [markdown]
# # END OF POSTPROCESSING

# %%
