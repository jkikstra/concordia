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
# TODO:
# now still, if possible
# - [ ] filename: IIASA -> IIASA-IAMC (do by hand?)
# future:
# - [ ] check _FillValue better
# - [ ] update authors?


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
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from tqdm import tqdm

from concordia.cmip7 import utils as cmip7_utils

# %%
lock = SerializableLock()

# %% [markdown]
# ## setup

# %%
from concordia.cmip7.CONSTANTS import CONFIG, return_marker_information, PROXY_YEARS, CMIP_ERA, GASES_ESGF_CEDS_VOC, find_voc_data_variable_string
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
            # if Path().resolve somehow goes to the root of this repository
            cmip7_dir = Path().resolve() / "notebooks" / "cmip7"  # one up
            settings = uprox.get_settings(base_path=cmip7_dir, file = CONFIG)

# %% editable=true slideshow={"slide_type": ""} tags=["parameters"]
marker_to_run: str = "VLLO"

# %%
# Scenario information
HARMONIZATION_VERSION, MODEL_SELECTION, SCENARIO_SELECTION = return_marker_information(
    m=marker_to_run
)

# %%
grid_file_location_non_anthro = settings.out_path / HARMONIZATION_VERSION  # output after downscaling, for openburning and AIR (which both don't need post-processing fixes) 
grid_file_location_anthro = settings.out_path / HARMONIZATION_VERSION / "weighted-reaggregated" # output after anthro-fixes in `workflow-postprocess_anthro-reaggregate-CDR-sectors.py

grid_location_out = settings.out_path / HARMONIZATION_VERSION / "final"
grid_location_out.mkdir(parents=True, exist_ok=True)

# %%
files_anthro = [
    file
    for file in grid_file_location_anthro.glob("*.nc")
    if "anthro" in file.name and "-AIR-" not in file.name 
]
files_anthro_before_postprocessing = [
    file
    for file in grid_file_location_non_anthro.glob("*.nc")
    if "anthro" in file.name and "-AIR-" not in file.name 
]
gas_prefixes_anthro = list(set([f.name.split('-')[0] for f in files_anthro]))
gas_prefixes_anthro.sort()

files_non_anthro = [
    file
    for file in grid_file_location_non_anthro.glob("*.nc")
    if ("em-anthro" not in file.name and "-AIR-" in file.name and "speciated" not in file.name) or "openburning" in file.name
]
# gas_prefixes_non_anthro = list(set([f.name.split('-')[0] for f in files_non_anthro]))
# gas_prefixes_non_anthro.sort()
print(files_anthro)
print(files_non_anthro)

# %%
# example files

# # historical format
# hist_air = xr.open_dataset(
#     settings.gridding_path / "esgf" / "ceds" / "CMIP7_AIR" / "CO2-em-AIR-anthro_input4MIPs_emissions_CMIP_CEDS-CMIP-2025-04-18_gn_200001-202312.nc"
# )

# hist_openburning = xr.open_dataset(
#     settings.gridding_path / "esgf" / "bb4cmip7" / "BC" / "gn" / "v20250612" / "BC_input4MIPs_emissions_CMIP_DRES-CMIP-BB4CMIP7-2-1_gn_190001-202312.nc" 
# )

# hist_anthro = xr.open_dataset(
#     settings.gridding_path / "esgf" / "ceds" / "CMIP7_anthro" / "SO2-em-anthro_input4MIPs_emissions_CMIP_CEDS-CMIP-2025-04-18_gn_200001-202312.nc"
# )

# # Our 'format' before fixes

# test_air = xr.open_dataset(
#     files_non_anthro[6]
# )

# test_openburning = xr.open_dataset(
#     files_non_anthro[1]
# )

# # print(test.sector.values)
# test_anthro = xr.open_dataset(
#     files_anthro[0] # all are missing all attributes (lost somewhere in the fixes)
# )

# # clear test files from memory
# test_air.close()
# test_anthro.close()
# test_openburning.close()

# %%
ESGF_VERSION = "0-3-0"

mapping_scen_naming = {
    "REMIND-MAgPIE-3.5-4.11-SSP1---Very-Low-Emissions": 'scendraft1',
    "VLLO": 'scendraft1',
    "GCAM-7.1-scenarioMIP-SSP3---High-Emissions": 'scendraft2',
    "H": 'scendraft2',
}

mapping_var_naming = {
    # ceds-like
    "Sulfur": "SO2",
    "VOC": "NMVOC"
}
mapping_var_naming_bb4cmip = {
    # ceds-like
    "Sulfur": "SO2",
    "VOC": "NMVOCbulk"
}

new_institution_id = "IIASA-IAMC"

# file renaming
def filename_to_hist_style(filename, g, type):
    if (filename.startswith("VOC-") or filename.startswith("Sulfur-")):
        if "toceds" in type:
            filename = filename.replace(g, mapping_var_naming[g], 1) # 1: only replace the first entry (at the start)
        if "tobb4cmip" in type:
            filename = filename.replace(g, mapping_var_naming_bb4cmip[g], 1) # 1: only replace the first entry (at the start)
    return filename

def update_dates(filename):

    if filename.endswith("202301-210012.nc"):
        filename = filename.replace("202301-210012.nc", "202201-210012.nc")

    return filename

def update_institute(filename):

    if "IIASA" in filename:
        filename = filename.replace("IIASA", new_institution_id, 1)

    return filename


def update_scenarioname(filename):
    vllo = "REMIND-MAgPIE-3.5-4.11-SSP1---Very-Low-Emissions"
    h = "GCAM-7.1-scenarioMIP-SSP3---High-Emissions"

    if vllo in filename:
        filename = filename.replace(vllo, mapping_scen_naming["VLLO"])
    if h in filename:
        filename = filename.replace(h, mapping_scen_naming["H"])

    return filename

def rename_files(filename, g, type="toceds-anthro"):

    
    filename = filename_to_hist_style(filename,g,type)
    
    filename = update_dates(filename)
    filename = update_scenarioname(filename)
    
    return filename

# Create a dictionary to map old variable names to new ones
def replace_data_variable(ds, g, type="em_anthro"): # other possible type: "em_AIR_anthro"
    var_rename_dict = {}
    for var_name in ds.data_vars:
        if var_name in [f'VOC_{type}', f'Sulfur_{type}']:
            if type in ["em_anthro", "em_AIR_anthro"]:
                var_rename_dict[var_name] = f"{mapping_var_naming[g]}_{type}"
                var_name = f"{mapping_var_naming[g]}_{type}" # keep renamed variable for later compression
            if type in ["em_openburning"]:
                var_rename_dict[var_name] = f"{mapping_var_naming_bb4cmip[g]}_{type}"
                var_name = f"{mapping_var_naming_bb4cmip[g]}_{type}" # keep renamed variable for later compression
    # Rename variables if any mappings exist
    if var_rename_dict:
        ds = ds.rename(var_rename_dict)
    return ds, var_name


from concordia.cmip7.utils import DS_ATTRS, DS_ATTRS_adhocfix

# variable renaming
def rename_variables_and_attributes(ds, ds_attrs=None, g=None, type="toceds-anthro"):

    # if type in ["toceds-anthro", "toceds-AIR", "tobb4cmip"]:
    #     # ...
    #     print("...")

    if type in ["toceds-anthro"]:

        # Variables
        ds, var_name = replace_data_variable(ds, g, type="em_anthro") # note: this will do nothing for VOC-speciated data, which is fine, because no renaming is necessary there
        
        # Data variables
        # copy lat_bnds, lon_bnds, time_bnds from ds_attrs to ds
        boundary_vars = ['lat_bnds', 'lon_bnds', 'time_bnds']
        for bnd_var in boundary_vars:
            if bnd_var in ds_attrs.data_vars and bnd_var not in ds.data_vars:
                print(f"Copying boundary variable '{bnd_var}' from original dataset")
                ds[bnd_var] = ds_attrs[bnd_var]
            elif bnd_var in ds_attrs.coords and bnd_var not in ds.coords:
                print(f"Copying boundary coordinate '{bnd_var}' from original dataset")
                ds = ds.assign_coords({bnd_var: ds_attrs[bnd_var]})

        # Attributes
        # first, copy over global attributes from the data before the fix
        ds.attrs.update(ds_attrs.attrs)
        # second, replace all values that are different in DS_ATTRS compared to current DS_ATTRS_adhocfix
        for attr_name, attr_value in DS_ATTRS_adhocfix.items():
            if attr_name in ds.attrs:
                if ds.attrs[attr_name] != attr_value:
                    print(f"Ad-hoc fix: Updating global attribute '{attr_name}': '{ds.attrs[attr_name]}' -> '{attr_value}'")
                    ds.attrs[attr_name] = attr_value
            else:
                print(f"Ad-hoc fix: Adding new global attribute '{attr_name}': '{attr_value}'")
                ds.attrs[attr_name] = attr_value
    
    if type in ["toceds-AIR"]:
        # Variables
        ds, var_name = replace_data_variable(ds, g, type="em_AIR_anthro")


        # Data variables
        # drop data variable `level_bnds` from ds
        if 'level_bnds' in ds.data_vars:
            print(f"Dropping 'level_bnds' data variable from AIR dataset")
            ds = ds.drop_vars('level_bnds')
        elif 'level_bnds' in ds.coords:
            print(f"Dropping 'level_bnds' coordinate from AIR dataset")
            ds = ds.drop_vars('level_bnds')



    # Attributes - for all files
    # third, replace attributes where there is model or scenario information
    ds.attrs['institution_id'] = new_institution_id
    ds.attrs['institution'] = "International Institue for Applied Systems Analysis - Integrated Assessment Modeling Consortium"
    # ds.attrs['consortium'] = "International Institue for Applied Systems Analysis - Integrated Assessment Modeling Consortium"
    ds.attrs['source_id'] = ds.attrs['institution_id'] + '-' + mapping_scen_naming[marker_to_run] + '-' + ESGF_VERSION
    voc_spec_gas_prefixes = [gas.split('-')[0] for gas in GASES_ESGF_CEDS_VOC]
    if g in voc_spec_gas_prefixes:
        ds.attrs['source_id'] = ds.attrs['source_id'] + "-supplemental"
    
    ds.attrs['title'] = f"Future anthropogenic emissions of {g} in {mapping_scen_naming[marker_to_run]}"
    # fourth, add a link to the documentation of this current dataset
    ds.attrs['comment'] = "This version is meant only for testing, and will be depricated once final files are available. " + ds.attrs['comment'] + " During the iteration period of testing, documentation will be collected in a Google Doc: https://docs.google.com/document/d/1E7Wv2APCRY-LRfI6II9pkwtfySGvRm2zPE2oyYP6mak/edit?usp=sharing"
    # fifth, add attributes that are in CEDS, but not yet in our scenario data, but are indeed useful
    ds.attrs['license_id'] = "CC BY 4.0"
    ds.attrs['datetime_start'] = "202201"
    ds.attrs['datetime_end'] = "210012"
    ds.attrs['time_range'] = "202201-210012"
    ds.attrs['data_usage_tips'] = "Note that these are monthly average fluxes."

    if type in ["tobb4cmip"]:
        # Variables
        ds, var_name = replace_data_variable(ds, g, type="em_openburning")

        # Attributes
        ds.attrs['region'] = "global_land"
        ds.attrs['data_usage_tips'] = ds.attrs['data_usage_tips'] + " Be careful to not double count deforestation carbon emissions. They are included in the estimates but several models also have deforestation carbon emissions based for example on historical deforestation rates."

    return ds, var_name

# %%
# Run metadata updates
# For 1 scenario, this process should take about ~30 minutes on a laptop

# anthro
# for g in [gas_prefixes_anthro[8]]: # try one file
for g in gas_prefixes_anthro:
    print(f"Processing anthro {g}")
    matching_file_post_fix = [f for f in files_anthro if f.name.startswith(g + '-')][0]
    matching_file_pre_fix = [f for f in files_anthro_before_postprocessing if f.name.startswith(g + '-')][0]

    ds = xr.open_dataset(matching_file_post_fix)
    ds_attrs = xr.open_dataset(matching_file_pre_fix) # old file to use to copy over attributes

    # step 1. align Sulfur/SO2 and (NM)VOC naming
    # rename filename:
    f_out = rename_files(matching_file_post_fix.name, g, type="toceds-anthro")

    # rename variable:
    ds, var_name = rename_variables_and_attributes(ds, ds_attrs, g, type="toceds-anthro")

    # write out to final folder
    # Save with compression (this triggers computation)
    encoding = {var_name: {"zlib": True, "complevel": 4}}
    ds.to_netcdf(grid_location_out / f_out, encoding=encoding)
    
    # Close datasets to free memory
    ds.close()
    ds_attrs.close()

# air and openburning
for f in files_non_anthro:
    g = f.name.split('-')[0]

    ds = xr.open_dataset(f)

    if "-AIR-" in f.name:
        print(f"Processing AIR {g}")
        type = "toceds-AIR"
    if "-openburning_" in f.name:
        print(f"Processing openburning {g}")
        type = "tobb4cmip"

    f_out = rename_files(f.name, g, type=type)
    ds, var_name = rename_variables_and_attributes(ds, None, g, type=type)

    # write out to final folder
    # Save with compression (this triggers computation)
    encoding = {var_name: {"zlib": True, "complevel": 4}}
    ds.to_netcdf(grid_location_out / f_out, encoding=encoding)
    
    # Close datasets to free memory
    ds.close()
    
ds

