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

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

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
ceds_input_gridding_path = settings.gridding_path / "20250523/Jarmo_files"

# %%
figure_path = Path("/home/hoegner/Projects/CMIP7/input/gridding/plots/seasonality")  # 👈 your export folder

# %%
template_file = (
    settings.gridding_path
    / "example_files" / "GCAM4-SSP4-34-SPA4-V25-Harmonized-DB-Sulfur-em-aircraft-anthro_input4MIPs_emissions_CMIP_IAMC-V1_gn_201501-210012.nc"
)
template = xr.open_dataset(template_file)

# %%
all_sectors = ['AGR',
 'ENE',
 'IND',
 'TRA',
 'RCO',
 'SLV',
 'WST',
 'AWB',
 'FRTB',
 'GRSB',
 'PEAT',
 'AIR',
 'SHP']

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
season_files_filtered = season_files[(season_files["year"]==2015) | (season_files["year"]==2020)]
season_files_filtered = season_files_filtered[season_files_filtered["sector"] != "AIR"]
season_files_filtered

# %%
seasonality_files_for_future = season_files_filtered["file"].unique()

# %%
name = "seasonality"
coords = {"lat": template.lat, "lon": template.lon, "time": [1,2,3,4,5,6,7,8,9,10,11,12]}

for f in seasonality_files_for_future:
    a = read_r_variable(f)
    da = xr.DataArray(a, coords=coords, name=name)
    ds = da.to_dataset(name=da.name)

    out = f.with_suffix(".nc")
    ds.to_netcdf(out, format="NETCDF4", mode="w")

    var = ds[name]
    
    seasonality_weights = var / var.sum(dim=("lat", "lon"))
    mask_zeros = seasonality_weights == 0
    n_zeros = mask_zeros.sum().item()
    
    print(seasonality_weights.sum(dim=("lat", "lon")))
    print(n_zeros)
    
    out_png = figure_path / f.with_suffix(".png").name

    fig, axes = plt.subplots(
        4, 3, figsize=(16, 12),
        subplot_kw={"projection": ccrs.PlateCarree()}
    )

    for i, ax in enumerate(axes.flat):
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS, linewidth=0.3)
        ax.set_title(str(var.time.values[i]))
        im = var.isel(time=i).plot(
            ax=ax, transform=ccrs.PlateCarree(),
            cmap="viridis", add_colorbar=False
        )

    cbar = fig.colorbar(
        im, ax=axes, orientation="horizontal",
        fraction=0.05, pad=0.05, aspect=40, shrink=0.5
    )
    cbar.set_label(var.name)

    plt.suptitle(f.stem, fontsize=14)
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
