#!/usr/bin/env python

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


import glob
import itertools
import pathlib
from functools import lru_cache

import dask
import dask.array
import pandas as pd
import ptolemy as pt
import pyreadr
import xarray as xr


sector_mapping = {
    "anthro": ["AGR", "ENE", "IND", "RCO", "SLV", "TRA", "WST"],
    "openburning": ["AWB", "FRTB", "GRSB", "PEAT"],
    "aircraft": ["AIR"],  # NB: proxy issues here, need to address in hackathon
    "shipping": [
        "SHP"
    ],  # NB: this was not split out originally, but 1) we have proxy issues; 2) we will be redoing this in this project
}

template_file = "./example_files/GCAM4-SSP4-34-SPA4-V25-Harmonized-DB-Sulfur-em-aircraft-anthro_input4MIPs_emissions_CMIP_IAMC-V1_gn_201501-210012.nc"

template = xr.open_dataset(template_file)

# For aircraft, only seasonality files are for BC and NOx. We need to check if other aircraft emissions have seasonality in final files or not.
#
# For shipping, seasonality files are available for all species except CO2 and CH4.

# # get all files
year_files = pd.read_csv(
    "./ceds_input/input/gridding/proxy-CEDS9/proxy_mapping_CEDS9.csv"
)
year_files = year_files.rename(columns={"em": "gas", "proxy_file": "file"}).drop(
    columns=["proxybackup_file"]
)
year_files["file"] = year_files["file"].apply(
    lambda x: f"./ceds_input/input/gridding/proxy-CEDS9/{x}.Rd"
)
year_files["gas"] = year_files["gas"].replace({"NMVOC": "VOC"})

season_files = pd.read_csv(
    "./ceds_input/input/gridding/seasonality-CEDS9/seasonality_mapping_CEDS9.csv"
)
season_files = season_files.rename(columns={"em": "gas", "seasonality_file": "file"})
season_files["file"] = season_files["file"].apply(
    lambda x: f"./ceds_input/input/gridding/seasonality-CEDS9/{x}.Rd"
)

df_files = year_files.merge(
    season_files,
    on=["sector", "gas", "year"],
    how="outer",
    suffixes=["_year", "_season"],
).sort_values(
    ["gas", "sector", "year"]
)  # order matters for padding nans later

# Originally, we were specifically missing (and did not previously process) H2
# from biomass burning. But using the file-based location method above, we now
# should have no missing files.
assert df_files[df_files["file_season"].isnull()].empty

df_files = df_files.fillna(method="pad")
years = [2015] + list(range(2020, 2101, 10))
df_files = df_files[df_files["year"].isin(years)]


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
    lat = template.lat[row.start_row - 1 : row.end_row]
    lon = template.lon[row.start_col - 1 : row.end_col]
    da = xr.DataArray(a, coords={"lat": lat, "lon": lon})
    da["lat"] = da["lat"] * -1  # NB: Inversion!
    return da


# ## Generating Country Mask Raster
#
# This function reads in all country masks which are provided as Rdata files and
# translates them to xarray arrays and combines them into a single netcdf file.
def gen_mask():
    print("Generating Mask Raster")
    files = glob.glob("./ceds_input/input/gridding/mask/*.Rd")
    df = pd.DataFrame(
        [[pathlib.Path(f).stem.split("_")[0], f] for f in files],
        columns=["iso", "file"],
    )
    idxs = pd.read_csv("./ceds_input/input/gridding/country_location_index_05.csv")
    data = pd.merge(df, idxs, on="iso")
    mask = xr.concat([mask_to_ary(s) for i, s in data.iterrows()], data.iso)
    mask.to_netcdf("iso_mask.nc")

    country_combinations = {
        "sdn_ssd": ["ssd", "sdn"],
        "isr_pse": ["isr", "pse"],
        "srb_ksv": ["srb", "srb (kosovo)"],
    }
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
    comb_mask = xr.concat(
        [
            comb_mask,
            comb_mask.sum(dim="iso").assign_coords({"iso": "World"}),
        ],
        dim="iso",
    )
    comb_mask.to_netcdf("ssp_comb_iso_mask.nc")


#
# Generating sector proxy files
#


def read_r_variable(file):
    file = pathlib.Path(file)
    print(f"Reading in {file}\n")
    a = pyreadr.read_r(file)[file.stem]
    if hasattr(a, "data"):
        a = a.data
    return a


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
            dtype=float,
        )
    else:
        a = read_r_variable(fname)
    da = xr.DataArray(a, coords=coords, name=name)
    if waste:
        # For any grid cell has a population density greater than the 1000
        # people/sq mile threshold, the population density for that grid cell is
        # fixed to the threshold value, while for any grid cell has a population
        # density less than the threshold, the grid cell keeps its original
        # population density value.
        threshold = 1000 / 2.59e6  # people per m2
        grid_area_m2 = xr.DataArray.from_series(pt.cell_area_from_file(template))
        # casting people to people/area
        da = da / grid_area_m2
        da = xr.where(da > threshold, threshold, da)
        # cast back to total people
        da = da * grid_area_m2
    da["lat"] = da["lat"] * -1  # NB: Inversion!
    return da


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
    list(da.coords.keys()) + list(new_coords.keys())
    return da


def read_air(file):
    file = pathlib.Path(file.replace(".Rd", ".nc"))  # we can only read netcdfs for air
    print(f"Reading in {file}\n")
    da = xr.open_dataset(file).var1_1
    # this is supposed to be lat, lon, month, level, based on how other files are treated in season_to_ary
    da = da.transpose(
        "dim1", "dim2", "dim4", "dim3"
    )  # WARNING: very fragile to how ncdfs were made
    return da.data


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
            dtype=float,
        )
    else:
        a = read_func(fname)
    da = xr.DataArray(a, coords=coords, name=name)
    da["lat"] = da["lat"] * -1  # NB: Inversion!
    return da


def season_to_ary(row, file_key="file", with_dask=True):
    da = make_season_ary(row[file_key], row.sector == "AIR", with_dask=with_dask)
    new_coords = {"gas": row.gas, "sector": row.sector}
    da = da.assign_coords(new_coords)
    da = da.expand_dims(list(new_coords.keys()))
    list(da.coords.keys()) + list(new_coords.keys())
    return da


# ## Example Processing
#
# ### multiple years in proxy files
def gen_da_for_sector(files, with_dask=True):
    if len(files.gas.unique()) != 1:
        raise ValueError(f"Expected 1 gas, got {files.gas.unique()}")
    if len(files.sector.unique()) != 1:
        raise ValueError(f"Expected 1 sector, got {files.sector.unique()}")
    if len(files.file_season.unique()) != 1:
        raise ValueError(
            f"Expected 1 seasonality file, got {files.file_season.unique()}"
        )
    yda = xr.concat(
        [
            year_to_ary(row, file_key="file_year", with_dask=with_dask)
            for i, row in files.iterrows()
        ],
        dim="year",
    )
    sda = season_to_ary(files.iloc[0], file_key="file_season", with_dask=with_dask)
    da = yda * sda
    da.name = "emissions"
    return da


# ### Multiple sectors in proxy files
def gen_da_for_gas(gas, files, with_dask=True):
    assert len(files.gas.unique()) == 1
    das = []
    for sector in files.sector.unique():
        print(gas, sector)
        da = gen_da_for_sector(files[files.sector == sector], with_dask=with_dask)
        das.append(da)
    return xr.concat(das, "sector")


# # Full Processing
def full_process(sector_key):
    print(f"Doing full process for {sector_key}")
    sectors = sector_mapping[sector_key]
    sector_files = df_files[df_files.sector.isin(sectors)]
    comp = dict(zlib=True, complevel=5)
    for gas in sector_files.gas.unique():
        files = sector_files[sector_files.gas == gas]
        da = gen_da_for_gas(gas, files)
        da.to_netcdf(f"./proxy_rasters/{sector_key}_{gas}.nc", encoding={da.name: comp})


if __name__ == "__main__":
    # gen_mask()
    # full_process('anthro')
    # full_process('openburning')
    # full_process('shipping')
    full_process("aircraft")
