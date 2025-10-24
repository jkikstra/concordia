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
marker_to_run: str = "H"
use_dask: bool = True  # Set to False to load data directly into memory instead of using dask

# %%
# Scenario information
HARMONIZATION_VERSION, MODEL_SELECTION, SCENARIO_SELECTION, _ = return_marker_information(
    m=marker_to_run
)

# %%
grid_file_location_fixed_anthro = settings.out_path / HARMONIZATION_VERSION / "weighted" # output after the correction as in `workflow-postprocess_anthro-pattern-harmonisation.py`
grid_file_location_fixed_anthro_reagg = settings.out_path / HARMONIZATION_VERSION / "weighted-reaggregated" # output after this script
grid_file_location_fixed_anthro_reagg.mkdir(parents=True, exist_ok=True)

# %%
files_main = [
    file
    for file in grid_file_location_fixed_anthro.glob("*.nc")
    if "anthro" in file.name and "-AIR-" not in file.name and "speciated" not in file.name
]
files_voc = [
    file
    for file in grid_file_location_fixed_anthro.glob("*.nc")
    if "anthro" in file.name and "-AIR-" not in file.name and "speciated" in file.name
]

# %%
# Define delayed function for parallel processing
@delayed
def process_single_file(file, files_main, files_voc, grid_file_location_fixed_anthro_reagg, use_dask=True):
    """Process a single file: combine Energy and BECCS sectors"""
    
    # Step 1: read in the file with dask chunks or load directly into memory
    if use_dask:
        ds = xr.open_dataset(file, chunks={
            # "lat": 180, "lon": 360
        })
    else:
        # Standard lazy-loading (only read metadata into memory)
        ds = xr.open_dataset(file)
        # Optionally load data into memory immediately
        ds = ds.load()
    
    # Determine the emission variable name based on file type
    if file in files_main:
        # Standard gas files have variables like "CO2_em_anthro", "CH4_em_anthro", etc.
        var_name = [var for var in ds.data_vars if str(var).endswith("_em_anthro")][0]
    elif file in files_voc:
        # VOC files have variables like "VOC01_alcohols_em_speciated_VOC_anthro"
        var_name = [var for var in ds.data_vars if str(var).endswith("_em_speciated_VOC_anthro")][0]
    
    # Step 2: sum sectors "Energy" and "BECCS", call the new sector "Energy"
    if "Energy" in ds.sector.values and "BECCS" in ds.sector.values:
        
        # Extract Energy and BECCS data
        energy_data = ds.sel(sector="Energy")
        beccs_data = ds.sel(sector="BECCS")
        
        # Sum them together (lazy computation with dask or direct computation without)
        combined_energy = energy_data[var_name] + beccs_data[var_name]
        
        # Create new dataset without BECCS sector
        sectors_to_keep = [s for s in ds.sector.values if s != "BECCS"]
        ds_filtered = ds.sel(sector=sectors_to_keep)
        
        # Replace Energy sector data with combined data
        ds_filtered[var_name].loc[dict(sector="Energy")] = combined_energy
        
    elif "Energy" in ds.sector.values:
        # Only Energy exists, keep as is
        ds_filtered = ds
        print(f"Note: {file.name} has Energy but no BECCS sector")
    elif "BECCS" in ds.sector.values:
        # Only BECCS exists, rename it to Energy
        ds_filtered = ds.rename({"BECCS": "Energy"})
        print(f"Note: {file.name} has BECCS but no Energy sector - renamed BECCS to Energy")
    else:
        # Neither exists, keep as is
        ds_filtered = ds
        print(f"Warning: {file.name} has neither Energy nor BECCS sectors")
    
    # Step 2.5: Rename "Other non-Land CDR" to "Other Capture and Removal"
    if "Other non-Land CDR" in ds_filtered.sector.values:
        # Create new sector coordinate values with the renamed sector
        new_sector_values = []
        for sector in ds_filtered.sector.values:
            if sector == "Other non-Land CDR":
                new_sector_values.append("Other Capture and Removal")
            else:
                new_sector_values.append(sector)
        
        # Assign the new coordinate values
        ds_filtered = ds_filtered.assign_coords(sector=new_sector_values)
        print(f"Renamed 'Other non-Land CDR' to 'Other Capture and Removal' in {file.name}")
    
    # Update the sector coordinate attributes for all changes
    if 'ids' in ds_filtered.sector.attrs:
        # Create new ids mapping with updated sector names
        new_ids_list = []
        for i, sector in enumerate(ds_filtered.sector.values):
            new_ids_list.append(f"{i}: {sector}")
        
        # Update the ids attribute
        ds_filtered.sector.attrs['ids'] = "; ".join(new_ids_list)
        print(f"Updated sector ids for {file.name}")
    
    # Step 3: write out in `grid_file_location_fixed_anthro_reagg` with the same file name
    output_file = grid_file_location_fixed_anthro_reagg / file.name
    
    # Save with compression (this triggers computation)
    encoding = {var_name: {"zlib": True, "complevel": 4}}
    ds_filtered.to_netcdf(output_file, encoding=encoding)
    
    # Close datasets to free memory
    ds.close()
    ds_filtered.close()
    
    return f"Processed {file.name}"

# Non-dask version of the same function for sequential processing
def process_single_file_sync(file, files_main, files_voc, grid_file_location_fixed_anthro_reagg, use_dask=False):
    """Process a single file synchronously: combine Energy and BECCS sectors"""
    # Step 1: read in the file directly into memory
    ds = xr.open_dataset(file)
    ds = ds.load()  # Load data into memory immediately
    
    # Determine the emission variable name based on file type
    if file in files_main:
        # Standard gas files have variables like "CO2_em_anthro", "CH4_em_anthro", etc.
        var_name = [var for var in ds.data_vars if str(var).endswith("_em_anthro")][0]
    elif file in files_voc:
        # VOC files have variables like "VOC01_alcohols_em_speciated_VOC_anthro"
        var_name = [var for var in ds.data_vars if str(var).endswith("_em_speciated_VOC_anthro")][0]
    
    # Step 2: sum sectors "Energy" and "BECCS", call the new sector "Energy"
    if "Energy" in ds.sector.values and "BECCS" in ds.sector.values:
        
        # Extract Energy and BECCS data
        energy_data = ds.sel(sector="Energy")
        beccs_data = ds.sel(sector="BECCS")
        
        # Sum them together (direct computation)
        combined_energy = energy_data[var_name] + beccs_data[var_name]
        
        # Create new dataset without BECCS sector
        sectors_to_keep = [s for s in ds.sector.values if s != "BECCS"]
        ds_filtered = ds.sel(sector=sectors_to_keep)
        
        # Replace Energy sector data with combined data
        ds_filtered[var_name].loc[dict(sector="Energy")] = combined_energy
        
    elif "Energy" in ds.sector.values:
        # Only Energy exists, keep as is
        ds_filtered = ds
        print(f"Note: {file.name} has Energy but no BECCS sector")
    elif "BECCS" in ds.sector.values:
        # Only BECCS exists, rename it to Energy
        ds_filtered = ds.rename({"BECCS": "Energy"})
        print(f"Note: {file.name} has BECCS but no Energy sector - renamed BECCS to Energy")
    else:
        # Neither exists, keep as is
        ds_filtered = ds
        print(f"Warning: {file.name} has neither Energy nor BECCS sectors")
    
    # Step 2.5: Rename "Other non-Land CDR" to "Other Capture and Removal"
    if "Other non-Land CDR" in ds_filtered.sector.values:
        # Create new sector coordinate values with the renamed sector
        new_sector_values = []
        for sector in ds_filtered.sector.values:
            if sector == "Other non-Land CDR":
                new_sector_values.append("Other Capture and Removal")
            else:
                new_sector_values.append(sector)
        
        # Assign the new coordinate values
        ds_filtered = ds_filtered.assign_coords(sector=new_sector_values)
        print(f"Renamed 'Other non-Land CDR' to 'Other Capture and Removal' in {file.name}")
    
    # Update the sector coordinate attributes for all changes
    if 'ids' in ds_filtered.sector.attrs:
        # Create new ids mapping with updated sector names
        new_ids_list = []
        for i, sector in enumerate(ds_filtered.sector.values):
            new_ids_list.append(f"{i}: {sector}")
        
        # Update the ids attribute
        ds_filtered.sector.attrs['ids'] = "; ".join(new_ids_list)
        print(f"Updated sector ids for {file.name}")
    
    # Step 3: write out in `grid_file_location_fixed_anthro_reagg` with the same file name
    output_file = grid_file_location_fixed_anthro_reagg / file.name
    
    # Save with compression
    encoding = {var_name: {"zlib": True, "complevel": 4}}
    ds_filtered.to_netcdf(output_file, encoding=encoding)
    
    # Close datasets to free memory
    ds.close()
    ds_filtered.close()
    
    return f"Processed {file.name}"

# Create delayed tasks for all files
print("Creating delayed tasks...")
use_dask=False
delayed_tasks = [
    process_single_file(file, files_main, files_voc, grid_file_location_fixed_anthro_reagg, use_dask)
    for file in files_main + files_voc
]


# Execute all tasks
if use_dask:
    # Execute all tasks in parallel with progress bar using dask
    print(f"Processing {len(delayed_tasks)} files in parallel with dask...")
    with ProgressBar():
        results = compute(*delayed_tasks)
else:
    # Execute tasks sequentially without dask
    print(f"Processing {len(delayed_tasks)} files sequentially (no dask)...")
    results = []
    for i, task in enumerate(delayed_tasks):
        print(f"Processing file {i+1}/{len(delayed_tasks)}")
        # For non-dask mode, we need to call the function directly without @delayed
        file = (files_main + files_voc)[i]
        result = process_single_file_sync(file, files_main, files_voc, grid_file_location_fixed_anthro_reagg, use_dask)
        results.append(result)

# Took ~20 mins on Jarmo's laptop for main+VOC, running it for two scenarios in separate kernels, at the same time. 
print("All files processed successfully!")
for result in results:
    print(result)


# %% 
# # Check new format
# test = xr.open_dataset(
#     # grid_file_location_fixed_anthro_reagg / files_main[0].name
#     settings.out_path / HARMONIZATION_VERSION / "weighted-reaggregated" / files_main[0].name
# )

# print(test.sector.values)
# test

# %%
