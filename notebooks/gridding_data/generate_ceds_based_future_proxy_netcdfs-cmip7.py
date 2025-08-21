#!/usr/bin/env python
# %% [markdown]
# # Overview
#
# Gridding proxies are provided on a yearly-basis in `input/gridding/proxy-CEDS9`.
# Summing total values leads to <<1 values, indicating that unit transformations may be
# included. It is not clear if this is the kt-to-`<unit>` or Mt-to-`<unit>` transform.
#
# Seasonality (distribution by month) is provided in `input/gridding/seasonality-CEDS9`.
# Summing across months yields values of 1 in each grid cell.
#
# Country raster masks are provided in `input/gridding/mask`. Masks are provided at
# fraciton of country in grid cells.
#
# A blanket mask of negative CO2 values for energy are provided in
# `input/gridding/negCO2`. These masks are raw data and need to be assigned to lat/lon
# values. These are listed in the file `input/gridding/country_location_index_05.csv`,
# which I downloaded from the `emissions_downscaling` repository
#
# # Warnings
#
# It seems that latitude bands are reversed which I have addressed here, but it is good
# to check/confirm this


# %%
import itertools
from functools import lru_cache

# %%
import dask
import dask.array
import numpy as np
import pandas as pd
import ptolemy as pt
import pyogrio as pio
import pyreadr
import xarray as xr
from pathlib import Path
from typing import Callable, Dict, Tuple
import os

# %%
from concordia.settings import Settings

from concordia.cmip7.utils import read_r_variable, read_r_to_da, save_da_as_rd

# %%
def get_settings(base_path: Path, 
                 file = "config_cmip7_v0_2.yaml"):
    settings = Settings.from_config(
        file, 
        base_path=base_path,
        version=None
    )
    return settings

try:
    # when running the script from a terminal or otherwise
    notebook_dir = Path(__file__).resolve().parent
    settings = get_settings(base_path=notebook_dir)
except (FileNotFoundError, NameError):
    try:
        # when running the script from a terminal or otherwise
        notebook_dir = Path(__file__).resolve().parent.parent
        settings = get_settings(base_path=notebook_dir)
    except (FileNotFoundError, NameError):
        # Fallback for interactive/Jupyter mode, where 'file location' does not exist
        notebook_dir = Path().resolve().parent  # one up
        settings = get_settings(base_path=notebook_dir)


# %%
dim_order = ["gas", "sector", "level", "year", "month", "lat", "lon"]

# %%
sector_mapping = {
    "anthro": ["AGR", "ENE", "IND", "TRA", "RCO", "SLV", "WST"],
    "openburning": ["AWB", "FRTB", "GRSB", "PEAT"],
    "aircraft": ["AIR"],  # NB: proxy issues here, need to address in hackathon
    "shipping": ["SHP"],  # NB: this was not split out originally, but 1) we have proxy issues; 2) we will be redoing this in this project
}

# %%
all_sectors = [sector for sublist in sector_mapping.values() for sector in sublist]
openburning_sectors = list(sector_mapping.values())[1]

# %%
template_file = (
    settings.gridding_path
    / "example_files" / "GCAM4-SSP4-34-SPA4-V25-Harmonized-DB-Sulfur-em-aircraft-anthro_input4MIPs_emissions_CMIP_IAMC-V1_gn_201501-210012.nc"
)
template = xr.open_dataset(template_file)

# %% [markdown]
# For aircraft, only seasonality files are for BC and NOx. We need to check if other aircraft emissions have seasonality in final files or not.
#
# For shipping, seasonality files are available for all species except CO2 and CH4.

# %%
ceds_input_gridding_path = settings.gridding_path / "ceds_input"

# %%
ceds_input_gridding_path

# %% [markdown]
# ### prepare proxy file mapping

# %%
# # intermediary to aggregate final sector
# could also derive this from the CEDS_gridding_sectors.csv
sector_map = {
    "ELEC": "ENE",
    "ETRN": "ENE",
    "FFFI": "ENE",
    "FLR":  "ENE",
    "INDC": "IND",
    "INPU": "IND",
    "RCOO": "RCO",
    "RCORC": "RCO",
    "NRTR": "TRA",
    "ROAD": "TRA",
    "SHP":  "SHP",
    "TANK": "SHP"
}

def ceds_format_to_cmip7_format(df):
    df = df.rename(columns={"em": "gas"})
    df["gas"] = df["gas"].replace({"NMVOC": "VOC"})
    df["sector"] = df["sector"].replace(sector_map)

    return df

# %%
# # fallback method
# NOTE: we need to do the replacing per country already before using the 'aggregate'/'final' sectors, it seems like from Steve's emails.
# Not all countries have sufficient information in the provided CEDS proxies.
# In some cases, the "fallback" method (i.e., use population density of 2015 as the proxy for distributing emissions within a country)
# did not materialise.
# We now read in a file provided by Steve Smith in August 2025, which indicates for which country-sector combinations
# the fallback method (=proxybackup_file) should be implemented. 
# The code below should take care of:
# 1. reading in the fallback method country-sector mapping information
# 2. reading in the intermediate-sector proxy data
# 3. replacing zeroes with population density data IF the country-sector is in the country-sector mapping information
# However, this would mean a significant rewrite towards using the intermediate sector data.
# For that, we would also need to (double?) clarify whether or not we need to do something with the point-source emissions.
# So, instead, we for now do:
# 1.1. reading in the fallback method country-sector mapping information (same)
# 1.2. do the 'sector_map' already here, effectively saying that "IF ANY OF THE UNDERLYING PROXIES IS MISSING FOR THIS COUNTRY" we use the fallback method
# 2. reading in the final-sector proxy data
# 3. replacing zeroes with population density data IF the country-sector is in the MODIFIED country-sector mapping information

GASES = ["BC", "CH4", "CO", "CO2", "NH3", "NOx", "OC", 
         "N2O",
         "SO2", # "Sulfur", 
         "NMVOC", #"VOC"
         ]

fallback_information_intermed_sector = pd.concat(
    (
        pd.read_csv(ceds_input_gridding_path / "non-point_source_proxy_final_sector" / "fixed_fallback" / "countries_using_population_backup" / f"{gas}_proxy_substitution_mapping.csv")
        for gas in GASES
    ),
    ignore_index=True,
).drop(columns=["sub_flag"]).rename(columns={"iso": "iso3"})

# some renaming
fallback_information_intermed_sector = fallback_information_intermed_sector.rename(columns={"em": "gas"})
fallback_information_intermed_sector["gas"] = fallback_information_intermed_sector["gas"].replace({"NMVOC": "VOC"})
# Apply sector map
fallback_information_intermed_sector["sector"] = fallback_information_intermed_sector["sector"].replace(sector_map)
# Drop duplicates
fallback_information_intermed_sector = fallback_information_intermed_sector.drop_duplicates(subset=['gas','sector','iso3'])

# add which source we want, just for clarity
fallback_information_intermed_sector["source"] = "pop"
# filter mapping: remove USA states (Steve Smith clarified that the gridding only happens on the national level)
fallback_information_intermed_sector = fallback_information_intermed_sector[~fallback_information_intermed_sector["iso3"].str.startswith("usa_")]

# %%
# # get all files
year_files = pd.read_csv(
    ceds_input_gridding_path / "proxy_mapping.csv"
) # this proxy mapping happens on the intermediate sector level (e.g. it includes 'FFFI')

# not sure why we are droping the proxybackup_files here
# will restore for now, since we are missing some files where the backups are needed
year_files = year_files.rename(columns={"em": "gas", "proxy_file": "file", "proxybackup_file": "backup"})
# .drop(columns=["proxybackup_file"])

# some renaming
year_files["gas"] = year_files["gas"].replace({"NMVOC": "VOC"})

# Apply sector map
year_files["sector"] = year_files["sector"].replace(sector_map)

# also rename proxy files the aggregate sectors are pointing to
file_temp = year_files["file"].str.split("_", n=2, expand=True)
file_temp[2] = file_temp[2].replace(sector_map)
year_files["file"] = file_temp[0] + "_" + file_temp[1] + "_" + file_temp[2]
# and drop duplicates that arise, because we map subsectors to the same aggregates
year_files = year_files.drop_duplicates(subset=["gas", "sector", "file"])

# filter for the sectors we are retaining
# year_files = year_files[year_files["sector"].isin(all_sectors)]

year_file_path_map = {
    **{x: ceds_input_gridding_path / f"non-point_source_proxy_final_sector/{x}.Rd" for x in year_files["file"].unique()},
    **{x: ceds_input_gridding_path / f"population_proxy/{x}.Rd" for x in year_files["backup"].unique()}
}
# Apply the mapping
year_files["file"] = year_files["file"].map(year_file_path_map)
year_files["backup"] = year_files["backup"].map(year_file_path_map)


# %%
year_files

# %%
# ceds gridding convention is to always use the latest available year
latest_year = year_files.loc[year_files.groupby(["gas", "sector"])["year"].idxmax()]

# %%
# check that we've got all sectors that we expect, for now without the openburning ones
assert set(latest_year["sector"].unique()) == set(all_sectors) - set(openburning_sectors)

# %% [markdown]
# #### deal with missing proxy files

# %% [markdown]
# here we are checking which files are missing, and making sure that there is no fallback option from earlier years for those species/sector combinations either

# %%
# print out which ones are missing
if not latest_year["file"].apply(os.path.exists).all():
    missing_files = latest_year.loc[~latest_year["file"].apply(os.path.exists), "file"]
    print("Missing files:")
    print([p.name for p in missing_files])

# %%
missing_files_list = ['BC_2022_AGR.Rd', 'BC_2022_SLV.Rd', 'CH4_2022_SLV.Rd', 'CO_2022_AGR.Rd', 
                      'CO_2022_SLV.Rd', 'NOx_2022_SLV.Rd', 'OC_2022_AGR.Rd', 'OC_2022_SLV.Rd', 
                      'SO2_2022_AGR.Rd', 'SO2_2022_SLV.Rd']

# %%
data_dir = Path(
    ceds_input_gridding_path / "non-point_source_proxy_final_sector"
)
# Step 2: Get a list of all *.Rd files in the folder
files = list(data_dir.glob("*.Rd"))
existing_files = [p.name for p in files]


# %%
def extract_pairs(filenames):
    pairs = set()
    for f in filenames:
        parts = Path(f).stem.split('_')
        if len(parts) == 3:
            gas, _, sector = parts
            pairs.add((gas, sector))
    return pairs
    
missing_pairs = extract_pairs(missing_files_list)
existing_pairs = extract_pairs(existing_files)

overlapping_pairs = missing_pairs & existing_pairs

truly_missing_pairs = missing_pairs - existing_pairs

# %%
# replace the paths of the missing files under "file" with the paths from the "backup" column 
# that point to the fallback population proxies

# NOTE: fallback MISSING! NEED TO ADD THIS (back) IN

latest_year["file"] = latest_year.apply(
    lambda row: row["backup"] if not row["file"].exists() else row["file"],
    axis=1
)
latest_year = latest_year.drop(columns=["backup"]).reset_index(drop=True)

# %%
# check again for missing files
assert latest_year["file"].apply(os.path.exists).all(), "Proxy files are missing!"

# %%
latest_year

# %% [markdown]
# ### prepare seasonality mapping

# %%
season_files = pd.read_csv(
    ceds_input_gridding_path / "seasonality_mapping.csv"
)
season_files = season_files.rename(columns={"em": "gas", "seasonality_file": "file"})
season_files["gas"] = season_files["gas"].replace({"NMVOC": "VOC"})

# filter for the sectors we are retaining
season_files = season_files[season_files["sector"].isin(all_sectors)]

# trying to speed things up here a little
# using a mapping instead of constructing 26000+ individual paths
unique_season_files = season_files["file"].unique()
season_file_path_map = {
    f: ceds_input_gridding_path / f"seasonality/{f}.Rd" for f in unique_season_files
}
season_files["file"] = season_files["file"].map(season_file_path_map)

# %%
latest_season = season_files.loc[season_files.groupby(['gas', 'sector'])['year'].idxmax()].reset_index(drop=True)
latest_season

# %%
print(len(latest_season))
print(len(latest_year))

# %%
# Assert that all seasonality files exist
assert latest_season["file"].apply(os.path.exists).all(), "Seasonality files are missing!"

# Optional: print out which ones are missing
if not latest_season['file'].apply(os.path.exists).all():
    missing_files = df.loc[~all_exist, 'file']
    print("Missing files:")
    print(missing_files.tolist())

# %% [markdown]
# ### create combined proxy mapping df

# %%
latest_year = latest_year.drop(columns=["year"]).reset_index(drop=True) # these files need to be changed
latest_season = latest_season.drop(columns=["year"]).reset_index(drop=True) # these files probably stay the same for now

# %%
df_files = latest_year.merge(
    latest_season,
    on=["sector", "gas"],
    how="outer",
    suffixes=["_year", "_season"],
).sort_values(
    ["gas", "sector"]
).dropna()

# order matters for padding nans later
# drop all combinations that have no proxy files assigned to them; check this later!

# %%
df_files

# %%
# keep consistency with previous naming conventions

df_files["gas"] = df_files["gas"].replace("SO2", "Sulfur")

# %%
df_files["gas"].unique()

# %%
# doing 2023,2024, 2025:5:2100 now - to strike balance between historical years, what is modelled, and data output size
# (could consider annual, because some IAMs do this, but is unlikely to actually be produced due to high data volume unless we can ascertain that annual interpolation isn't always correct here) 
years = [2023, 2024, 2025] + list(range(2030, 2101, 5))
years_df = pd.DataFrame({"year": years})
# add dummy keys for joining dfs
years_df["key"] = 1
df_files["key"] = 1

# %%
# mapping same proxy files to all years in future
df_files = pd.merge(df_files, years_df, on="key").drop(columns="key")
df_files = df_files[df_files["year"].isin(years)]
df_files = df_files[["gas", "sector", "year", "file_year", "file_season"]]
df_files

# %%
# Originally, we were specifically missing (and did not previously process) H2
# from biomass burning. But using the file-based location method above, we now
# should have no missing files.
assert not df_files["file_season"].isnull().any()

# %%
# check again that we've got all sectors that we expect, for now without the openburning ones
assert set(df_files["sector"].unique()) == set(all_sectors) - set(openburning_sectors)

# %%
assert df_files["file_year"].str.endswith(".Rd").all(), "Not all file year paths end with '.Rd'"
assert df_files["file_season"].str.endswith(".Rd").all(), "Not all file season paths end with '.Rd'"

# %%
assert df_files["file_year"].apply(lambda x: isinstance(x, Path)).all(), "Not all entries in 'file year' are Path objects"
assert df_files["file_season"].apply(lambda x: isinstance(x, Path)).all(), "Not all entries in 'file season' are Path objects"


# %% [markdown]
# ## create country mask and index raster

# %%
# ## Compiling a single country mask file
#
# See [process in original repo](https://github.com/iiasa/emissions_downscaling/blob/79dc01346dd841b0cff82f83df31706d6995c94c/code/parameters/gridding_functions.R#L98):
#
# ```
# proxy_cropped <- proxy[ start_row : end_row, start_col : end_col ]
# ```
#
# **In R, array indexes start at 1**
#
#
# ```
# temperatures <- c(12.9, 13.2, 15.0, 19.2, 21.4, 24.5, 29.5, 24.2, 20.5, 14.5, 10.2, 9.8)
#
# temperatures[2:4]
# >>> 13.2, 15.0, 19.2
# ```
#
# **Rows are latitudes, Columns are longitudes**
def mask_to_ary(row):
    a = pyreadr.read_r(row.file)[f"{row.iso}_mask"]
    lat = template.lat[::-1][row.start_row - 1 : row.end_row]
    lon = template.lon[row.start_col - 1 : row.end_col]
    da = xr.DataArray(
        np.asarray(a, dtype="float32"), coords={"lat": lat, "lon": lon}
    ).reindex(lat=template.lat, lon=template.lon, fill_value=0)

    return da


# %%
# ## Generating Country Mask Raster
#
# This function reads in all country masks which are provided as Rdata files and
# translates them to xarray arrays and combines them into a single netcdf file.
def gen_mask():
    print("Generating Mask Raster")
    files = ceds_input_gridding_path.glob("mask/*.Rd")
    df = pd.DataFrame(
        [[f.stem.split("_")[0], f] for f in files],
        columns=["iso", "file"],
    )
    idxs = pd.read_csv(ceds_input_gridding_path / "country_location_index_05.csv")
    data = pd.merge(df, idxs, on="iso")
    mask = xr.concat([mask_to_ary(s) for i, s in data.iterrows()], data.iso)
    return mask


# %%
# only really need this if we include alkalinity
def add_eez_to_mask(mask):
    print("Rasterize and add eez to mask raster")
    rasterize = pt.Rasterize(
        shape=(mask.sizes["lat"], mask.sizes["lon"]),
        coords={"lat": mask.coords["lat"], "lon": mask.coords["lon"]},
    )
    rasterize.read_shpf(
        pio.read_dataframe(
            settings.gridding_path / "eez_v12.gpkg",
            where="ISO_TER1 IS NOT NULL and POL_TYPE='200NM'",
        )
        .dissolve(by="ISO_TER1")
        .reset_index(names=["iso"]),
        idxkey="iso",
    )
    eez = rasterize.rasterize(strategy="weighted", normalize_weights=False).astype(
        mask.dtype
    )

    eez = eez.assign_coords(iso=eez.indexes["iso"].str.lower()).reindex_like(
        mask, fill_value=0.0
    )

    new_mask = mask + eez
    totals = new_mask.sum("iso")
    return (new_mask / totals).where(totals, 0)


# %%
def recombine_mask(mask, include_world=True):
    country_combinations = settings.country_combinations
    for comb, (iso1, iso2) in country_combinations.items():
        mask = xr.concat(
            [
                mask,
                (
                    mask.sel(iso=iso1).fillna(0) + mask.sel(iso=iso2).fillna(0)
                ).assign_coords({"iso": comb}),
            ],
            dim="iso",
        )
    comb_mask = mask.drop_sel(iso=list(itertools.chain(*country_combinations.values())))
    if include_world:
        comb_mask = xr.concat(
            [
                comb_mask,
                comb_mask.sum(dim="iso").assign_coords({"iso": "World"}),
            ],
            dim="iso",
        )
    return comb_mask
    # comb_mask.to_netcdf(settings.gridding_path / "ssp_comb_iso_mask.nc")


# %%
def mask_to_indexraster(mask):
    indexraster = pt.IndexRaster.from_weighted_raster(mask.rename(iso="country"))
    return indexraster


# %%
def gen_indexraster():
    mask = gen_mask()
    mask.to_netcdf(settings.gridding_path / "ssp_comb_countrymask.nc")
    mask = add_eez_to_mask(mask)
    mask = recombine_mask(mask, include_world=False)
    indexraster = mask_to_indexraster(mask)
    indexraster.to_netcdf(settings.gridding_path / "ssp_comb_indexraster.nc")


# %% [markdown]
# ## Fix CEDS anthro WST


# %%
# Test reading in a CEDS .Rd file, and writing it out again in the same format, without changing anything.

# in_test = settings.gridding_path / "ceds_input" / "non-point_source_proxy_final_sector" / "CO_2022_WST.Rd"
# out_test = settings.gridding_path / "ceds_input" / "non-point_source_proxy_final_sector" / "fixed_fallback" / "CO_2022_WST.Rd"

# save_da_as_rd(xr.DataArray(read_r_variable(in_test, float_dtype="float64"), 
#                  coords={"lat": template.lat, "lon": template.lon}, 
#                  name="emissions"), 
#               out_path=out_test, 
#               object_name="CO_2022_WST", 
#               undo_flip=True)

# assert_ordereddict_equal(pyreadr.read_r(in_test), 
#                          pyreadr.read_r(out_test))


# %% [markdown]
# ### Produce new proxy files mixing emissions and population

# %%
# Data for the fixes

# read template file
template_file = (
    settings.gridding_path
    / "example_files" / "GCAM4-SSP4-34-SPA4-V25-Harmonized-DB-Sulfur-em-aircraft-anthro_input4MIPs_emissions_CMIP_IAMC-V1_gn_201501-210012.nc"
)
template = xr.open_dataset(template_file)
country_mask =  xr.open_dataset(settings.gridding_path / "ssp_comb_countrymask.nc").__xarray_dataarray_variable__

# Code for the fixes

def validate_and_normalize_mapping(df: pd.DataFrame, fallback_source: str) -> Tuple[Dict[str, str], str]:
    """
    Return (choice_map, fallback_source), where choice_map maps iso3 -> 'pop'/'emis'.
    Both keys and values are normalized to lowercase.
    """
    if not {"iso3", "source"}.issubset(df.columns):
        raise ValueError("mapping_df must have columns: 'iso3' and 'source'")

    valid_sources = {"pop", "emis"}
    fallback_source = str(fallback_source).lower()
    if fallback_source not in valid_sources:
        raise ValueError("fallback_source must be 'pop' or 'emis'")

    choice_map = {
        str(row.iso3).strip().lower(): str(row.source).strip().lower()
        for _, row in df.iterrows()
        if str(row.source).strip().lower() in valid_sources
    }
    return choice_map, fallback_source
def align_grids(pop_da: xr.DataArray, emis_da: xr.DataArray, country_grid: xr.DataArray) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """
    Require exact coordinate match; if they don't match, reproject / regrid upstream.
    """
    return xr.align(pop_da, emis_da, country_grid, join="exact")


def country_to_iso_grid(country_grid,
                 maximum_value_countrymask = 0.99    # -> currently setting maxval > 0.99, to avoid overlapping borders and skewing the countries with emissions data by having all emissions going towards population because pop/sqmile is higher than kg/m2 flux. now this means that their won't be any emissions at the border for these fallback method options
                 ):
    maxval = country_grid.max(dim="iso")
    iso_grid = country_grid.idxmax(dim="iso")                 # (lat, lon) of ISO labels
    iso_grid = iso_grid.where(maxval > maximum_value_countrymask, other=np.nan)       # ocean/void -> NaN

    # think about this combination more carefully for borders (which now might have some emissions already, which we could potentially be deleting here)
    # normalize to lowercase strings
    iso_grid = xr.apply_ufunc(np.char.lower, iso_grid.astype("U"),
                            dask="parallelized", output_dtypes=[str])

    return iso_grid

def fix_ceds_proxy_add_population_fallback(rvariable, #emis_da
                                           pop_da,
                                           country_grid, #country_mapping (adjusted) 
                                           mapping_df #dfil
                                           ):


    # --- 1) Dominant ISO per pixel ---
    iso_grid = country_to_iso_grid(country_grid) # # for faster execution, we could move this to outside of the function, as it only needs to be run once

    # --- 2) Build sets from mapping (lowercase) ---
    choice_map = {str(r.iso3).strip().lower(): str(r.source).strip().lower()
                for _, r in mapping_df.iterrows()
                if str(r.source).strip().lower() in {"pop","emis"}}
    
    pop_set = {iso for iso, src in choice_map.items() if src == "pop"}
    
    # --- 3) Mask: True where we should use population ---
    # use fallback is ISO is in pop_set
    mask_pop = iso_grid.isin(list(pop_set)) # make sure it is a list, not a set
    mask_pop.name = "mask_pop"

    # --- 4) Combine via sum formulation (avoids dtype surprises) ---
    maskf = mask_pop.astype(pop_da.dtype)
    population_modifier = 1e-6 # if the normalisation happens afterwards, per country, this should not matter, and is just helpful for visualisation purposes right now. 
    combined = (pop_da * maskf * population_modifier) + (rvariable * (1.0 - maskf)) 
    
    # We don't want to be deleting data, or at least not too much, so we check:
    # ... deleted masked values are only a small part 
    assert (rvariable * (1.0 - maskf)).values.sum() > 0.999 * rvariable.values.sum()
    # ... new data values is larger than old
    assert combined.values.sum() > 0.9999 * rvariable.values.sum()


    return combined


# %%
# Run the fixes
f_pop = settings.gridding_path / "ceds_input" / "population_proxy" / "population_2015.Rd"

for row in latest_year.itertuples(index=False):
    if (
        # only for anthro files on land (first only for Waste)
        "population" not in row.file.name 
        and "AIR" not in row.file.name 
        and "SHP" not in row.file.name
        and "WST" in row.file.name
        # and "CO_" in row.file.name
    ):
        gas = row.gas
        sector = row.sector
        f = row.file

        # load the population density, our fallback method
        pop_da = read_r_to_da(f_pop, 
                              template, dtype="float64")
        # load the relevant emissions-based proxy file
        emis_da = read_r_to_da(f, template, dtype="float64")

        # filter country mapping: only selected gas & sector
        country_sector_mapping_df = fallback_information_intermed_sector[
            (fallback_information_intermed_sector['gas']==gas)&(fallback_information_intermed_sector['sector']==sector)
        ]

        if len(country_sector_mapping_df)>0:
            choice_map, fallback_source = validate_and_normalize_mapping(country_sector_mapping_df, "pop")
            pop_da, emis_da, country_grid = align_grids(pop_da, emis_da, country_mask)
            print(country_sector_mapping_df)

            new_proxy = fix_ceds_proxy_add_population_fallback(
                rvariable = emis_da,
                pop_da = pop_da,
                country_grid = country_grid,
                mapping_df = country_sector_mapping_df,
            )

            out_filepath = settings.gridding_path / "ceds_input" / "non-point_source_proxy_final_sector" / "fixed_fallback" / f"{f.name}"
            save_da_as_rd(new_proxy, 
                out_path=out_filepath,
                object_name=f.stem, 
                undo_flip=True)
            print(f"Saved: {out_filepath}")

        else:
            print(f"No fixes needed for {gas}-{sector}")

# %%
# # TESTING/DEBUGGING CODE

# # check original and fixed files
# from concordia.rescue.proxy import plot_map

# plot_map(pop_da)

# # for g in GASES:
# for g in ["CO"]:
#     filename = f"{g}_2022_WST.Rd"
    
#     ceds_original = settings.gridding_path / "ceds_input" / "non-point_source_proxy_final_sector" / filename
#     ceds_original = read_r_to_da(ceds_original, template=template)
#     ceds_fixed = settings.gridding_path / "ceds_input" / "non-point_source_proxy_final_sector" / "fixed_fallback" / f"fixed_{filename}"
#     ceds_fixed = read_r_to_da(ceds_fixed, template=template)
    
#     plot_map(ceds_original)
    
#     plot_map(ceds_fixed)



# %% [markdown]
# ### Update proxy file mapping; df_files fix

# %%
# df_files.to_csv("df_files_before_fix.csv")
def update_with_fix_if_exists(p):
    p = Path(p)
    p_fix = p.parent / "fixed_fallback" / f"{p.stem}.Rd"
    if p_fix.exists():
        return p_fix
    else:
        return p

mask = df_files["sector"] == "WST"
df_files.loc[mask, "file_year"] = df_files.loc[mask, "file_year"].apply(update_with_fix_if_exists)
# df_files.to_csv("df_files_after_fix.csv")


# %% [markdown]
# ## Generate sector proxy files

# %%
# original function, might need this for aircraft emissions

def read_r_variable_air(file):
    file = Path(file) # in case file is just a string
    print(f"Reading in {file}\n")

    a = pyreadr.read_r(file)[file.stem]
    
    return np.asarray(a, dtype="float32")[::-1]


# %%
@lru_cache
def grid_area_m2():
    return xr.DataArray.from_series(pt.cell_area_from_file(template)).astype("float32")


# %%
@lru_cache
def make_year_ary(fname, air=False, waste=False, with_dask=True):
    # special treatement for Waste, see https://github.com/JGCRI/CEDS/wiki/Data_and_Assumptions#proxy-derivation-for-waste-sector
    name = "emissions"
    coords = {"lat": template.lat, "lon": template.lon}
    if air:
        coords["level"] = template.level
    if with_dask:
        a = dask.array.from_delayed(
            dask.delayed(read_r_variable)(fname),
            shape=[len(v) for v in coords.values()],
            dtype="float32",
        )
    else:
        a = read_r_variable(fname)

    da = xr.DataArray(a, coords=coords, name=name)

    # convert to emissions value per m2

    # not for air, because they are already given as a flux in the CEDS-files zenodo
    # dump from 2019. The full context is difficult, but some information is in the
    # ticket: https://github.com/JGCRI/CEDS/issues/45

    # TODO: If new proxy files should be used for air, then they will be distributed
    # by ceds again in terms of emissions and will then need to be divided by cell areas
    # to convert to fluxes.
    if not air:
        da = da / grid_area_m2()
    if waste:
        # For any grid cell has a population density greater than the 1000
        # people/sq mile threshold, the population density for that grid cell is
        # fixed to the threshold value, while for any grid cell has a population
        # density less than the threshold, the grid cell keeps its original
        # population density value.
        threshold = 1000 / 2.59e6  # people per m2
        da = xr.where(da > threshold, threshold, da)

    return da


# %%
def year_to_ary(row, file_key="file", with_dask=True):
    da = make_year_ary(
        row[file_key], row.sector == "AIR", row.sector == "WST", with_dask=with_dask
    )
    new_coords = {
        "gas": row.gas,
        "sector": row.sector,
        "year": row.year,
    }
    da = da.assign_coords(new_coords)
    da = da.expand_dims(list(new_coords.keys()))

    return da


# %%
def read_air(file):
    file = file.with_suffix(".nc")  # we can only read netcdfs for air
    print(f"Reading in {file}\n")
    da = xr.open_dataset(file).var1_1
    # this is supposed to be lat, lon, month, level, based on how other files are treated in season_to_ary
    da = da.transpose(
        "dim1", "dim2", "dim4", "dim3"
    )  # WARNING: very fragile to how ncdfs were made
    return da.data.astype("float32")


# %%
@lru_cache
def make_season_ary(fname, air=False, with_dask=True):
    coords = {"lat": template.lat, "lon": template.lon, "month": range(1, 13)}
    name = "seasonality"
    read_func = read_r_variable
    if air:
        read_func = read_air
        coords["level"] = template.level

    if with_dask:
        a = dask.array.from_delayed(
            dask.delayed(read_func)(fname),
            shape=[len(v) for v in coords.values()],
            dtype="float32",
        )
    else:
        a = read_func(fname)
    da = xr.DataArray(a, coords=coords, name=name)

    return da


# %%
def season_to_ary(row, file_key="file", with_dask=True):
    da = make_season_ary(row[file_key], row.sector == "AIR", with_dask=with_dask)
    new_coords = {"gas": row.gas, "sector": row.sector}
    da = da.assign_coords(new_coords)
    da = da.expand_dims(list(new_coords.keys()))
    
    return da


# %%
# updated function to be able to handle different seasonality files for different years

def gen_da_for_sector(gas, sector, with_dask=True):
    files = df_files[(df_files.gas == gas) & (df_files.sector == sector)]
    if len(files.gas.unique()) != 1:
        raise ValueError(f"Expected 1 gas, got {files.gas.unique()}")
    if len(files.sector.unique()) != 1:
        raise ValueError(f"Expected 1 sector, got {files.sector.unique()}")

    # List to collect per-year * per-season arrays
    da_list = []

    for i, row in files.iterrows():
        yda = year_to_ary(row, file_key="file_year", with_dask=with_dask)
        sda = season_to_ary(row, file_key="file_season", with_dask=with_dask)

        da_temp = yda * sda
        da_list.append(da_temp)

    # Concatenate across the "year" dimension
    da = xr.concat(da_list, dim="year")
    da.name = "emissions"

    return da


# %%
# ### Multiple sectors in proxy files
def gen_da_for_gas(gas, sector_key, with_dask=True):
    das = []
    for sector in sector_mapping[sector_key]:
        print(gas, sector)
        da = gen_da_for_sector(gas, sector, with_dask=with_dask)
        das.append(da)

    return xr.concat(das, "sector").transpose(*dim_order, missing_dims="ignore")


# %%
# # Full Processing
def full_process(sector_key):
    print(f"Doing full process for {sector_key}")
    sectors = sector_mapping[sector_key]
    sector_files = df_files[df_files.sector.isin(sectors)]
    gases = sector_files.gas.unique()
    for gas in gases:
        da = gen_da_for_gas(gas, sector_key)
        output_path = settings.proxy_path / f"{sector_key}_{gas}.nc"

        # delete file if it already exists to avoid permission denied error in the override attempt
        if output_path.exists():
            os.remove(output_path)
    
        da.to_netcdf(
            output_path,
            encoding={da.name: settings.encoding},
        )


# %%
settings.encoding

# %%
settings.proxy_path

# %%
if __name__ == "__main__":
    # gen_indexraster() # currently fails for srb (kosovo)
    full_process("anthro")
   # full_process("openburning")
    # full_process("aircraft")
    ## old:
    full_process("shipping")
