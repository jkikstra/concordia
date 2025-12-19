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
marker_to_run: str = "h" # options: h, hl, m, ml, l, ln, vl

# What folder to save this run in
GRIDDING_VERSION: str | None = None
GRIDDING_VERSION: str | None = f"{marker_to_run}_{VERSION_ESGF}"


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
# load in example files

bc_anthro = load_result('BC-em-anthro')
bc_open = load_result('BC-em-openburning')
bc_air = load_result('BC-em-AIR-anthro')

voc_sp_anthro = load_result('VOC01-alcohols-em-speciated-VOC-anthro')
voc_sp_open = load_result('NMVOC-C2H2-em-speciated-VOC-openburning')
h2_open = load_result('H2-em-openburning')

# %%
# Print ncdump-like output for each dataset
def print_dataset_info(ds, name):
    """Print dataset info similar to ncdump output"""
    # Set pandas options to not truncate output
    with pd.option_context('display.max_columns', None, 
                           'display.max_rows', None,
                           'display.width', None,
                           'display.max_colwidth', None):
        print(f"\n{'='*80}")
        print(f"DATASET: {name}")
        print(f"{'='*80}\n")
        print(ds)
        print(f"\n{'-'*80}\nData Variables Info:\n{'-'*80}")
        ds.info()
        
        print(f"\n{'-'*80}\nGlobal Attributes:\n{'-'*80}")
        for key, value in ds.attrs.items():
            print(f"{key}: {value}")
        
        print(f"\n{'-'*80}\nVariable Attributes:\n{'-'*80}")
        for var_name in ds.data_vars:
            print(f"\n{var_name}:")
            if hasattr(ds[var_name], 'attrs'):
                for key, value in ds[var_name].attrs.items():
                    print(f"  {key}: {value}")
        
        print(f"\n{'-'*80}\nCoordinate Attributes:\n{'-'*80}")
        for coord_name in ds.coords:
            print(f"\n{coord_name}:")
            if hasattr(ds[coord_name], 'attrs'):
                for key, value in ds[coord_name].attrs.items():
                    print(f"  {key}: {value}")

print_dataset_info(bc_anthro, "BC-em-anthro")
print_dataset_info(bc_open, "BC-em-openburning")
print_dataset_info(bc_air, "BC-em-AIR-anthro")
print_dataset_info(voc_sp_anthro, "VOC01-alcohols-em-speciated-VOC-anthro")
print_dataset_info(voc_sp_open, "NMVOC-C2H2-em-speciated-VOC-openburning")
print_dataset_info(h2_open, "H2-em-openburning")


# %%
