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
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas_indexing as pix
import pandas as pd
import numpy as np
import seaborn as sns
from concordia.settings import Settings
import glob
import os
from dask import delayed, compute
from dask.diagnostics import ProgressBar
from dask.utils import SerializableLock
from tqdm import tqdm

# %%
lock = SerializableLock()

# %% [markdown]
# ## paths and dicts

# %%
GRIDDING_VERSION = "config_cmip7_v0_2" # jarmo 10.08.2025 (first go, with hist 022)
GRIDDING_VERSION = "config_cmip7_v0_2_newhistory_remind" # jarmo 10.08.2025 (second go, with updated hist)
GRIDDING_VERSION = "config_cmip7_v0_2_WSTfix_remind" # jarmo 21.08.2025 (third go, with updated hist, with fixed Waste)
GRIDDING_VERSION = "config_cmip7_v0_2_CEDS_proxies" # annika 27.08.2025 (with proxies derived from CEDS directly for anthro)
GRIDDING_VERSION = "config_cmip7_v0_2_CEDS_proxies_new_AIR" # annika 28.08.2025 (now also for aircraft)
GRIDDING_VERSION = "config_cmip7_v0_2_CEDS_proxies_compressed" # annika 28.08.2025 (including encoding for compression)

# history path
history_path = Path("/home/hoegner/Projects/CMIP7/input/historical")

# CEDS (CMIP7)
path_ceds_cmip7 = Path("/home/hoegner/Projects/CMIP7/input/gridding/CEDS_CMIP7")
# path_ceds_cmip7 = Path(f"C:/Users/kikstra/IIASA/ECE.prog - Documents/Projects/CMIP7/IAM Data Processing/ESGF/CEDS/CMIP7") 

# gridded emissions
# gridding input files
grid_file_location = "/home/hoegner/Projects/CMIP7/input/gridding/"
# grid_file_location = "C:/Users/kikstra/IIASA/ECE.prog - Documents/Projects/CMIP7/IAM Data Processing/concordia_cmip7_v0_2/input/gridding/"
countrymap_path = Path(grid_file_location)

# CMIP7 gridded emissions
cmip7_data_location = Path(f"/home/hoegner/GitHub/concordia/results/{GRIDDING_VERSION}")
# cmip7_data_location = Path(f"C:/Users/kikstra/documents/GitHub/concordia/results/{GRIDDING_VERSION}") # gridding output

plots_path = cmip7_data_location / "plots"
plots_path.mkdir(exist_ok=True, parents=True)

outfile = Path(cmip7_data_location, "comparison_with_historical_country_level.csv")

# %%
HARMONIZATION_VERSION = "config_cmip7_v0_2_CEDS_proxies_compressed"
SETTINGS_FILE = "config_cmip7_v0_2.yaml"
HISTORY_FILE = "cmip7_history_countrylevel_250721.csv"

# %%
sectors = {
    "Energy" : "Energy Sector",
    "Industrial" : "Industrial Sector",
    "Residential, Commercial, Other" : "Residential Commercial Other",
    "Transportation" : "Transportation Sector"
}

# %% [markdown]
# ## import files

# %%
history = pd.read_csv(
    Path(history_path, HISTORY_FILE), index_col=[0, 1, 2, 3, 4]
).dropna(axis=1)

# %%
countrymap_path = Path(grid_file_location)
f = "ssp_comb_countrymask_corrected.nc"
mask = xr.open_dataset(
    countrymap_path / f,
    engine="netcdf4",
)

# %%
iso_list = np.unique(mask["iso"])

# %%
areacella = xr.open_dataset(Path(grid_file_location, "areacella_input4MIPs_emissions_CMIP_CEDS-CMIP-2025-04-18_gn.nc"))
cell_area = areacella["areacella"]


# %% [markdown]
# ## main function

# %%
def ds_to_annual_emissions_total(gridded_data, var_name, cell_area, keep_sectors=True):
    """
    Convert gridded emissions in kg/m2/s to Mt/year.
    
    Parameters:
    - gridded_data: xr.Dataset containing the emission variable
    - var_name: str, name of the variable to convert
    - cell_area: xr.DataArray of shape (lat, lon), in m2
    - keep_sectors: bool, if True, retain sector info
    
    Returns:
    - xr.DataArray of Mt/year, shape (year,) or (sector, year)
    """
    da = gridded_data[var_name]

    # obtain the seconds in each month for which data is available
    seconds_per_month = da.time.dt.days_in_month * 24 * 60 * 60

    # kg/m2/s --> kg/m2/month
    monthly = seconds_per_month * da

    # weight with cell area
    area_weighted = cell_area * monthly

    # Sum over spatial dimensions
    sum_dims = ["lat", "lon"]
    if "level" in area_weighted.dims:
        sum_dims.append("level")

    kg_per_month = area_weighted.sum(dim=sum_dims)

    # Convert to annual totals (kg/year)
    kg_per_year = kg_per_month.groupby("time.year").sum()

    # Convert to Mt/year
    da_Mt_y = kg_per_year * 1e-9

    if "sector" in da_Mt_y.dims and not keep_sectors:
        da_Mt_y = da_Mt_y.sum(dim="sector")

    # make sure variable is correctly named
    da_Mt_y = da_Mt_y.rename(var_name)
    
    return da_Mt_y


# %% [markdown]
# ## country level aggregation loop

# %%
gridded_files = glob.glob(os.path.join(cmip7_data_location, "*-em-anthro_*.nc"))

dfs_lazy = []

for file in tqdm(gridded_files, desc="Files"):
    ds = xr.open_dataset(file, chunks={"time": 12})  # lazy Dask arrays
    ds = ds.sel(time=ds["time"].dt.year == 2023)
    
    varname = list(ds.data_vars)[0]

    for iso in iso_list:
        # mask remains lazy; nothing is computed yet
        mask_iso = mask.sel(iso=iso)["__xarray_dataarray_variable__"]
        da_masked = (ds[varname] * mask_iso).where(mask_iso > 0, other=0)

        # optionally wrap in a dataset if your downstream function expects a dataset
        ds_country = da_masked.to_dataset(name=varname)

        # keep the emissions aggregation lazy; don’t call .compute() yet
        da_Mt_y = ds_to_annual_emissions_total(
            ds_country, varname, cell_area, keep_sectors=True
        )

        # attach meta info as coordinates
        da_Mt_y = da_Mt_y.assign_coords(region=iso, gas=varname.split("_")[0], unit="Mt/year")

        dfs_lazy.append(da_Mt_y)

# %%
combined = xr.concat(dfs_lazy, dim="region")
with ProgressBar():
    combined = combined.compute()

# %% [markdown]
# ## post-processing and renaming

# %%
df_all = combined.to_dataframe().reset_index()
varname = df_all.columns[-1]

# to wide format
df_all = df_all.pivot_table(
                index=["region", "sector", "unit", "gas"],
                columns="year",
                values=varname).sort_index().sort_index(axis=1)

df_all = df_all.reset_index()

# %%
# and a bunch of reformulations to match iamc format

df_all["sector"] = df_all["sector"].replace(sectors)
df_all["variable"] = "Emissions|" + df_all["gas"] + "|" + df_all["sector"]
df_all["unit"] = "Mt " + df_all["gas"] + "/yr"
df_all["2023"] = df_all[2023]
gridded = df_all[["region", "variable", "unit", "2023"]].set_index(["region", "variable", "unit"])

# unit conversion for N2O
N2O_mask = gridded.index.get_level_values("unit") == "Mt N2O/yr"
gridded.loc[N2O_mask, "2023"] *= 1000
gridded = gridded.rename(index={"Mt N2O/yr": "kt N2O/yr"}, level="unit")

gridded

# %%
hist = history["2023"].reset_index().drop(columns=["scenario", "model"]).set_index(["region","variable","unit"])

# %% [markdown]
# ## comparison with historical CEDS data, already aggregated to country level

# %%
gridded = gridded.rename(columns={gridded.columns[0]: "reaggregated_2023"})
hist = hist.rename(columns={hist.columns[0]: "historical_2023"})

# join on the MultiIndex, keeping only rows present in both dfs
df_compare = gridded.join(hist, how="inner")

# compute difference
df_compare["difference (hist-reagg)"] = df_compare["historical_2023"] - df_compare["reaggregated_2023"]
df_compare["ratio (hist/reagg)"] = df_compare["historical_2023"] / df_compare["reaggregated_2023"]
df_compare["percent (of hist)"] = abs(df_compare["difference (hist-reagg)"] / df_compare["historical_2023"].replace(0, np.nan) * 100)
df_compare.reset_index(inplace=True)

# %%
df_compare = df_compare[~df_compare["variable"].str.endswith("|International Shipping")]
df_compare.to_csv(outfile)

# %%
plt.hist(df_compare["difference (hist-reagg)"], bins=100, edgecolor="black")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.title("Histogram of difference (hist-reagg)")
plt.show()

# %%
df_filtered = df_compare[(df_compare["unit"]!="kt N2O/yr") & (df_compare["unit"]!="Mt CO2/yr")]
df_filtered = df_filtered[df_filtered["difference (hist-reagg)"]!=0]

for sector, group in df_filtered.groupby("unit"):
    group["difference (hist-reagg)"].hist(bins=100, alpha=0.5, label=sector)

plt.xlabel("Absolute difference")
plt.ylabel("Count")
plt.ylim(0,200)
plt.title("Histogram of differences by species")
plt.grid(False)
plt.legend()
plt.show()

# %%
df_compare[df_compare["region"]=="ind"]

# %%
df_compare.dropna().sort_values(by = "percent (of hist)", ascending = False).head(30)
