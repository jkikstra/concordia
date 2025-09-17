# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import dask
from dask import delayed, compute
from dask.diagnostics import ProgressBar
from dask.utils import SerializableLock

import cftime
import numpy as np

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import cartopy.feature as cfeature # for country borders

from matplotlib import colors


import cartopy.crs as ccrs

from matplotlib.animation import FuncAnimation, PillowWriter

# from concordia.cmip7 import utils as cmip7_utils

IAMC_COLS = ["model", "scenario", "region", "variable", "unit"] 


# %%
lock = SerializableLock()


# %% [markdown]
# # Paths, definitions

# %%
# Gridded scenario output
# GRIDDING_VERSION = "config_cmip7_v0_2_WSTfix_remind" # jarmo 21.08.2025 (third go, with updated hist, with fixed Waste)
# #GRIDDING_VERSION = "config_cmip7_v0_3_GCAM"
# path_scen_cmip7 = Path(f"C:/Users/kikstra/IIASA/ECE.prog - Documents/Projects/CMIP7/IAM Data Processing/Shared emission fields data/v0_2/{GRIDDING_VERSION}") # gridding output
GRIDDING_VERSION = "config_cmip7_v0_2_CEDSnc_remind" # jarmo 31.08.2025 (v0.2 to share)
# GRIDDING_VERSION = "config_cmip7_v0_2_CEDSnc_remind_only_CO2" # jarmo 30.08.2025 (fourth go, but with CDR)
#GRIDDING_VERSION = "config_cmip7_v0_3_GCAM"
path_scen_cmip7 = Path(f"C:/Users/kikstra/Documents/GitHub/concordia/results/{GRIDDING_VERSION}") # gridding output

# where to save plots of this script  
plots_path = path_scen_cmip7 / "plots"
plots_path.mkdir(exist_ok=True, parents=True)

# %%
SECTORS_ANTHRO = [
    '**International Shipping', 
    '**Agriculture', # note: is set to zero in the gridding, for co2
    '**Energy Sector', 
    '**Industrial Sector',
    '**Residential Commercial Other',
    '**Solvents Production and Application',
    '**Transportation Sector',
    '**Waste',
    '**Other non-Land CDR',
    '**BECCS'
]
SECTORS_AIR = [
    '**Aircraft'
]
SECTORS_OPENBURNING = [
    '**Agricultural Waste Burning',
    '**Forest Burning',
    '**Grassland Burning', 
    '**Peat Burning'
]

# %%
sector_dict = {
"Energy Sector": "Energy",
"Industrial Sector": "Industrial",
"Residential Commercial Other": "Residential, Commercial, Other",
"Transportation Sector": "Transportation"
}

# %%
MODEL_SELECTION = "REMIND-MAgPIE 3.5-4.11"
SCENARIO_SELECTION = "SSP1 - Very Low Emissions"
#MODEL_SELECTION = "GCAM 7.1 scenarioMIP"
#SCENARIO_SELECTION = "SSP3 - High Emissions"
MODEL_SELECTION_GRIDDED = MODEL_SELECTION.replace(" ", "-")
SCENARIO_SELECTION_GRIDDED = SCENARIO_SELECTION.replace(" ", "-")

# %% [markdown]
# # Functions

# %% [markdown]
# ## Reading in


# %%
def read_nc_file(f, loc, reorder_list=None, chunks={"time": 1}):
    ds = xr.open_dataset(
        loc / f,
        engine="netcdf4",
        chunks=chunks,
        # lock=lock
    )
    
    if reorder_list is not None:
        ds = ds[reorder_list]  
   
    return ds

# %%
# sample variable
var = "CO2-em-anthro"

# %%
# load a CMIP7 scenario sample file
scen_cmip7_data_file = f"{var}_input4MIPs_emissions_CMIP7_IIASA-{MODEL_SELECTION_GRIDDED}-{SCENARIO_SELECTION_GRIDDED}_gn_202301-210012.nc"

scen_ds = read_nc_file(
    f = scen_cmip7_data_file,
    loc = path_scen_cmip7
)
scen_ds


# %% [markdown]
# ## Emissions grid gifs 


# %%
# run the plot

def plot_gif_gas_sector(
        ds,
        sector,
        gas,
        times,
        out_path,
        show = False
): 
    # Check if sector exists in the dataset
    if sector not in ds.sector.values:
        print(f"Sector '{sector}' does not exist for gas '{gas}'. Available sectors: {list(ds.sector.values)}")
        return

    # Select the data for all frames first (keeps vmin/vmax consistent across frames)
    print(f"LOADING: sector '{sector}', gas '{gas}'.")
    da_all = (
        ds.sel(sector=sector)
        .sel(time=times, method="nearest")[f"{gas}_em_anthro"]
        .squeeze()
    )

    # Check if all values are NaN
    if np.isnan(da_all.values).all():
        print(f"All values are NaN for gas '{gas}', sector '{sector}'. Skipping animation.")
        return

    # Color scale fixed across frames (takes some time to calculate the min and max)
    vmin = float(np.percentile(da_all.values[~np.isnan(da_all.values)], 2)) # generally should be zero (except for negative emissions)
    vmax = float(np.percentile(da_all.values[~np.isnan(da_all.values)], 99)) # normally 98 to ensure that point-sources are not dominating the (linear) colour scale, but can be higher if a lot of data is small
    norm = colors.Normalize(vmin=vmin, vmax=vmax)

    # Figure + axis (same style)
    fig, ax = plt.subplots(
        nrows=1, ncols=1,
        figsize=(6, 4.5),
        subplot_kw={"projection": ccrs.Robinson()},
    )

    # First frame
    da0 = da_all.isel(time=0)

    # Draw once; keep a handle to the QuadMesh and create ONE colorbar
    im = da0.plot.pcolormesh(
        ax=ax,
        transform=ccrs.PlateCarree(),
        # cmap=cmap,
        norm=norm,
        cmap="GnBu",
        # robust=True,
        add_colorbar=False,              # we'll add it manually to keep it stable
    )
    cbar = plt.colorbar(im, ax=ax, orientation="horizontal", shrink=0.65)

    # Build once, outside update()
    time_index = da_all.indexes["time"]  # CFTimeIndex for noleap, DatetimeIndex otherwise

    # First frame title
    t0_str = time_index[0].strftime("%Y-%m-%d") if hasattr(time_index[0], "strftime") else str(time_index[0])[:10]
    title = ax.set_title(f"{gas}, sector: {sector}, time: {t0_str}")
    ax.coastlines()

    plt.tight_layout()

    # Update function
    def update(i):
        # handle both normal frames and pause frames
        if i < len(times):
            frame = da_all.isel(time=i)
        else:  # repeat the last frame
            frame = da_all.isel(time=-1)

        im.set_array(np.ravel(frame.values))
        t_index = min(i, len(times)-1)
        t_str = time_index[t_index].strftime("%Y-%m-%d") \
            if hasattr(time_index[t_index], "strftime") else str(time_index[t_index])[:10]
        title.set_text(f"{gas}, sector: {sector}, time: {t_str}")
        return (im,)

    
    # GIF speed specifications
    base_fps = 10
    speed = 1
    fps = int(base_fps * speed)
    pause = 5 # pause at the end of GIF (in seconds)
    pause_frames = pause * fps
    writer = PillowWriter(fps=fps)

    # Animate
    anim = FuncAnimation(fig, update, frames=len(times) + pause_frames, interval=300, blit=False)


    # Save as GIF (requires Pillow)
    out_file_path = out_path / f"{gas}-{sector}_emissions_animation.gif"
    print(f"WRITING (start): '{out_file_path}'.")
    anim.save(out_file_path,
              writer=writer, dpi=150)
    print(f"WRITING (finished): '{out_file_path}'.")
    # # or MP4:
    # anim.save(Path("C:/Users/kikstra/Documents/GitHub/concordia/results/config_cmip7_v0_2_CEDSnc_remind_only_CO2/plots") / f"{gas}-{sector}_emissions_animation.mp4",
    #           writer="ffmpeg", dpi=120, bitrate=3000)

    # Or show in notebook / script runtime
    if show:
        plt.show()
    
    # (If you called save, you can close to free memory)
    plt.close(fig)

# %%
# Run the plot

for s in [
        "Energy", "Industrial", "International Shipping", 
        "Residential, Commercial, Other", 
        "Solvents Production and Application",
        "Transportation", 
        "Waste",
        "Other non-Land CDR", 
        "BECCS"
        ]:
    for g in [
        "CO2", 
        "Sulfur"
        ]:

        var = f"{g}-em-anthro"
        scen_cmip7_data_file = f"{var}_input4MIPs_emissions_CMIP7_IIASA-{MODEL_SELECTION_GRIDDED}-{SCENARIO_SELECTION_GRIDDED}_gn_202301-210012.nc"
        scen_ds = read_nc_file(
            f = scen_cmip7_data_file,
            loc = path_scen_cmip7
        )

        plot_gif_gas_sector(
            ds = scen_ds,
            sector = s,
            gas = g,
            times = scen_ds.time,
            out_path = plots_path,
            show = False
        )
