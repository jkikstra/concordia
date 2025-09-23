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
marker_to_run: str = "VLLO"

# %%
# Scenario information
HARMONIZATION_VERSION, MODEL_SELECTION, SCENARIO_SELECTION = return_marker_information(
    m=marker_to_run
)

# %%
# grid_file_location = "/home/hoegner/Projects/CMIP7/input/gridding/"
grid_file_location = settings.gridding_path

# ceds_data_location = Path(grid_file_location, "ESGF", "CEDS", "CMIP7_anthro")
# ceds_air_data_location = Path(grid_file_location, "ESGF", "CEDS", "CMIP7_AIR")
ceds_data_location = Path(grid_file_location, "esgf", "ceds", "CMIP7_anthro")
ceds_data_location_voc = Path("C:/Users/kikstra/Downloads/temp_VOC")
# ceds_air_data_location = Path(grid_file_location, "ESGF", "CEDS", "CMIP7_AIR") # AIR should not be different as there is no border treatment in our downscaling 

# %%
# # Settings
# SETTINGS_FILE = "config_cmip7_v0_2.yaml" # iteration round 2 

# sub_version = "_corrected_indexraster"
# HARMONIZATION_VERSION = f"config_cmip7_v0_2{sub_version}"

# gridded_data_location = Path(f"/home/hoegner/GitHub/concordia/results/{HARMONIZATION_VERSION}")
# weighted_data_location = Path(gridded_data_location, "weighted")

gridded_data_location = settings.out_path / HARMONIZATION_VERSION
weighted_data_location = settings.out_path / HARMONIZATION_VERSION / "weighted"
weighted_data_location.mkdir(parents=True, exist_ok=True)

# %%

years = [year for year in PROXY_YEARS if year >= settings.base_year]

sector_dict = {
    0: "Agriculture",
    1: "Energy",
    2: "Industrial",
    3: "Transportation",
    4: "Residential, Commercial, Other",
    5: "Solvents Production and Application",
    6: "Waste", 
    7: "International Shipping",
    8: "Other non-Land CDR",
    9: "BECCS"
}

# %%
def calculate_diff(ceds_da, scen_da, gas, empty_treatment="fill_zeroes", type="em_anthro"):
    """
    Compute the ratio (or difference if desired) between a CEDS reference dataset
    and a scenario dataset for a specific gas, handling missing sectors.

    Returns an xarray DataArray with 'sector' dimension and all other coordinates intact.
    """
    
    sectors = scen_da.sector.values
    ratio_list = []  # store per-sector ratios
    sector_names = []

    var_name = f"{gas}_{type}"

    for sector in sectors:

        # check if sector exists in each dataset
        sector_in_ceds = sector in ceds_da.sector.values
        sector_in_scen = sector in scen_da.sector.values

        if empty_treatment == "skip":
            if not sector_in_ceds or not sector_in_scen:
                print(f"Skipping sector '{sector}' - missing in {'CEDS' if not sector_in_ceds else 'scenario'} data")
                continue

        # handle missing sectors by filling zeros
        if empty_treatment == "fill_zeroes":
            # CEDS sector
            if sector_in_ceds:
                ceds_sector = ceds_da.sel(sector=sector)
            else:
                template_sector = ceds_da.isel(sector=0).squeeze()
                ceds_sector = xr.zeros_like(template_sector).expand_dims(sector=[sector])

            # Scenario sector
            if sector_in_scen:
                scen_sector = scen_da.sel(sector=sector)
            else:
                template_sector = scen_da.isel(sector=0).squeeze()
                scen_sector = xr.zeros_like(template_sector).expand_dims(sector=[sector])

        # select the numeric data variable only
        if var_name in ceds_sector.data_vars:
            ceds_values = ceds_sector[var_name]
        else:
            ceds_values = ceds_sector[list(ceds_sector.data_vars)[0]]

        if var_name in scen_sector.data_vars:
            scen_values = scen_sector[var_name]
        else:
            scen_values = scen_sector[list(scen_sector.data_vars)[0]]
        
        # compute ratio safely
        mask = scen_values != 0.0
        ratio_sector = xr.where(mask, ceds_values / scen_values, 1.0)
        
        # ensure sector dimension exists
        if 'sector' not in ratio_sector.dims:
            ratio_sector = ratio_sector.expand_dims(sector=[sector])

        ratio_list.append(ratio_sector)
        sector_names.append(sector)

    # concatenate all sectors along 'sector' dimension
    full_ratio = xr.concat(ratio_list, dim='sector')
    full_ratio = full_ratio.assign_coords(sector=sector_names)

    return full_ratio


# %%
files_main = [
    file
    for file in gridded_data_location.glob("*.nc")
    if "anthro" in file.name and "-AIR-" not in file.name and "speciated" not in file.name
]
files_voc = [
    file
    for file in gridded_data_location.glob("*.nc")
    if "anthro" in file.name and "-AIR-" not in file.name and "speciated" in file.name
]

# %%
## without dask (~3 mins per file)

prefix, postfix, postfullfix = [], [], []

def rename_ceds_species_string(gas, to_ceds=True):
    if to_ceds:
        if gas == "VOC":
            gas = "NMVOC"
        elif gas == "Sulfur":
            gas = "SO2"
    elif not to_ceds: # from CEDS to our workflow
        if gas == "NMVOC":
            gas = "VOC"
        elif gas == "SO2":
            gas = "Sulfur"
    return gas

def rename_ceds_species(ceds):
    if f"NMVOC_{type}" in ceds.data_vars:
        ceds = ceds.rename({f"NMVOC_{type}": f"VOC_{type}"})
    if f"SO2_{type}" in ceds.data_vars:
        ceds = ceds.rename({f"SO2_{type}": f"Sulfur_{type}"})
    return ceds

def what_emissions_variable_type(file):
    if file in files_main:
        type = "em_anthro"
    elif file in files_voc:
        type = "em_speciated_VOC_anthro"
    return type

def get_correct_naming(file):
    type = what_emissions_variable_type(file)

    gas = file.name.split("-")[0]
    outfile = Path(weighted_data_location, file.name)
    print(f"creating fix for {file.name}")

    gas = rename_ceds_species_string(gas)

    return type, outfile, gas

# for file in tqdm(files_main + files_voc, desc="Processing files"): # VOC not yet working; need to check variable names as there is a mismatch
# for file in tqdm([files_main[7]], desc="Processing files"):
# for file in tqdm(files_main, desc="Processing files"):
for file in tqdm(files_voc, desc="Processing files"):
    type, outfile, gas = get_correct_naming(file)

    # match reference file
    if file in files_main:
        match = next(ceds_data_location.glob(f"{gas}-*.nc"))
    if file in files_voc:
        match = next(ceds_data_location_voc.glob(f"{gas}-*.nc"))
        

    # open datasets (no dask)
    ceds = xr.open_dataset(match)
    gridded = xr.open_dataset(file)

    # rename variables if needed
    if f"NMVOC_{type}" in ceds.data_vars or f"SO2_{type}" in ceds.data_vars:
        ceds = rename_ceds_species(ceds)
    
    # revert gas names for output
    gas = rename_ceds_species_string(gas, to_ceds=False)
    if file in files_main:
        var = f"{gas}_{type}"
    if file in files_voc:
        var = find_voc_data_variable_string(gas)

    # rename sectors
    ceds = ceds.assign_coords(sector=pd.Series(ceds["sector"].values).map(sector_dict).values)
    reference = ceds.where(ceds.time.dt.year == 2023, drop=True)
    gridded_23 = gridded.where(gridded.time.dt.year == 2023, drop=True)

    # calculate relative difference (vectorized)
    pct_diff23 = calculate_diff(reference, gridded_23, gas)
    weights = pct_diff23.to_dataset(name=var)

    # expand weights to all years
    n_repeat = gridded.sizes["time"] // weights.sizes["time"]
    weights_exp = xr.concat([weights] * n_repeat, dim="time")
    weights_exp = weights_exp.assign_coords(time=gridded.time)

    # apply weights (= raw ratios)
    weighted = gridded * weights_exp

    # replace sectors we don't want weighted
    sectors_to_keep = ['Other non-Land CDR', 'BECCS', 'International Shipping']
    sectors_present = [s for s in sectors_to_keep if s in weighted.sector.values]
    if sectors_present:
        weighted[var].loc[dict(sector=sectors_present)] = gridded[var].sel(sector=sectors_present)

    # step 1:
    # calculate sectoral global totals
    gridded_global = gridded[var].groupby("sector").sum(dim=("lat", "lon")).astype("float64")
    weighted_global = weighted[var].groupby("sector").sum(dim=("lat", "lon")).astype("float64")

    global_scalar = xr.where(gridded_global != 0,
                             weighted_global / gridded_global,
                             0)

    # step 2: calculate how much to the global total to make the adjustment perfect for the future (ensure same global emissions)
    # sub-steps:
    # 1. multiply by cell_area to get emissions
    # 2. divide all grid cells by the same scalar (weighted_total / gridded_global)
    # 3. divide by cell_area to go back to emissions/m2
    
    # 1.
    areacella = xr.open_dataset(Path(grid_file_location, "areacella_input4MIPs_emissions_CMIP_CEDS-CMIP-2025-04-18_gn.nc"))
    cell_area = areacella["areacella"]

    total_emissions_weighted = cell_area * weighted

    # 2.
    total_emissions_harmonised = xr.where(global_scalar != 0,
                             total_emissions_weighted / global_scalar,
                             0) # looks like this somehow turns nan into 0?

    # 3.
    emissions_harmonised = total_emissions_harmonised / cell_area

    emissions_harmonised_global = emissions_harmonised.groupby("sector").sum(dim=("lat", "lon")).astype("float64") # for diagnostics

    
    # for diagnostics:
    # convert to dataframes
    df1 = gridded_global.to_dataframe(name="prefix").reset_index()
    df1["gas"] = gas
    df1 = df1.melt(id_vars=["time", "sector", "gas"], var_name="version", value_name="value")

    # intermediary step
    # df2 = weighted_global.to_dataframe(name="postfix").reset_index()
    # df2["gas"] = gas
    # df2 = df2.melt(id_vars=["time", "sector", "gas"], var_name="version", value_name="value")

    # post-fix
    df3 = emissions_harmonised_global[var].to_dataframe(name="postfix").reset_index()
    df3["gas"] = gas
    df3 = df3.melt(id_vars=["time", "sector", "gas"], var_name="version", value_name="value")

    prefix.append(df1)
    postfix.append(df3)

    # remove old file (from previous loop in processing)
    outfile.unlink(missing_ok=True)
    # save weighted dataset (no dask)
    encoding = {var: {"zlib": True, "complevel": 4}}
    weighted.to_netcdf(outfile, encoding=encoding)

# %%
# combine diagnostic results
global_prefix = pd.concat(prefix, ignore_index=True)
global_postfix = pd.concat(postfix, ignore_index=True)

# %%
global_prefix["time"] = pd.to_datetime(global_prefix["time"].astype(str))
global_postfix["time"] = pd.to_datetime(global_postfix["time"].astype(str))

# %%
to_plot = pd.concat([global_prefix, global_postfix])
out_csv = Path(weighted_data_location, "global_aggregates_for_checking.csv")
to_plot.to_csv(out_csv, index=False)

# %%
# plot postfix and prefix values

sns.relplot(
    data=to_plot,
    kind="line",
    x="time",
    y="value",
    hue="sector",
    col="gas",
    style="version",
    col_wrap=3,
    facet_kws={"sharey": False}
)
plt.show()

# %%
wide_df = to_plot.pivot_table(
    index=["gas", "sector", "time"],
    columns="version",
    values="value"
)
wide_df["diff"] = wide_df["postfix"] - wide_df["prefix"]
wide = wide_df.reset_index()

# %%
# plot diff (postfix - prefix)

sns.relplot(
    data=wide,
    kind="line",
    x="time",
    y="diff",
    hue="sector",
    col="gas",
    col_wrap=3,
    facet_kws={"sharey": False}
)
plt.show()

# %%
