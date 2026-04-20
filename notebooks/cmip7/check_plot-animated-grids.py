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

# %% [markdown]
# # Animated Emission Grid Visualizations for CMIP7
#
# This script creates animated GIF visualizations of CMIP7 emission grids showing
# how emissions evolve over time, across different sectors and gases.
#
# ## Purpose
#
# Generate publication-quality animated maps to:
# - **Present spatial emission patterns** in talks and reports
# - **Quality-check gridded outputs** by visually inspecting spatial distributions
# - **Understand emission evolution** across time (2022-2100), sectors, and species
# - **Communicate scenario differences** to stakeholders and collaborators
#
# ## When to Run
#
# Run this script **after** the main gridding workflow has completed and produced
# NetCDF files in the `results/{GRIDDING_VERSION}/final/` directory. This is typically
# the last QC step before publishing data to ESGF.
#
# ⚠️ **WARNING:** This script takes a long time to run (several hours for full execution)
# because it generates one GIF per gas-sector combination. For quick tests, restrict
# to a single sector or gas (see Usage Examples below).
#
# ## Related Scripts
#
# - `check_gridded_scenario_qc.py`: Comprehensive QC including animations (Module E)
# - `check_gridded-scenarios-global-sectoral-aggregation-compared-to-input.py`: Compare gridded totals to inputs
# - `check_plot-global-total-timeseries.py`: Static timeseries plots
# - Workflow reference: See `notebooks/cmip7/README.md` line 144
#
# ## Output
#
# Creates animated GIFs in `{path_scen_cmip7}/plots/` with naming pattern:
# `{GAS}-{SECTOR}_emissions_animation.gif` (e.g., `CO2-Energy_emissions_animation.gif`)

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
# Lock object for thread-safe Dask operations when reading NetCDF files in parallel
lock = SerializableLock()


# %% [markdown]
# # Configuration
#
# ## Paths and Scenario Selection
#
# Configure paths to gridded NetCDF files and output directory for plots.
#
# **Key Settings:**
# - `path_scen_cmip7`: Directory containing gridded NetCDF files (output from gridding workflow)
# - `SCENARIO_SELECTION_GRIDDED_AFTER_METADATA`: Short scenario name used in file naming
# - `plots_path`: Where to save output GIF files
#
# **Example Configurations:**
#
# ```python
# # For a versioned gridding output in results/
# GRIDDING_VERSION = "config_cmip7_v0_2_CEDSnc_remind"
# path_scen_cmip7 = Path(f"C:/Users/kikstra/Documents/GitHub/concordia/results/{GRIDDING_VERSION}")
#
# # For ESGF-ready data in a "final" subdirectory
# path_scen_cmip7 = Path("C:/Users/kikstra/Documents/GitHub/concordia/results/cmip7_esgf_v0_alpha_h/final")
# ```

# %%
# Gridded scenario output
# GRIDDING_VERSION = "config_cmip7_v0_2_WSTfix_remind" # jarmo 21.08.2025 (third go, with updated hist, with fixed Waste)
# #GRIDDING_VERSION = "config_cmip7_v0_3_GCAM"
# path_scen_cmip7 = Path(f"C:/Users/kikstra/IIASA/ECE.prog - Documents/Projects/CMIP7/IAM Data Processing/Shared emission fields data/v0_2/{GRIDDING_VERSION}") # gridding output
# GRIDDING_VERSION = "config_cmip7_v0_2_CEDSnc_remind" # jarmo 31.08.2025 (v0.2 to share)
# GRIDDING_VERSION = "config_cmip7_v0_2_CEDSnc_remind_only_CO2" # jarmo 30.08.2025 (fourth go, but with CDR)
#GRIDDING_VERSION = "config_cmip7_v0_3_GCAM"
# path_scen_cmip7 = Path(f"C:/Users/kikstra/Documents/GitHub/concordia/results/{GRIDDING_VERSION}") # gridding output
path_scen_cmip7 = Path("C:/Users/kikstra/Documents/GitHub/concordia/results/cmip7_esgf_v0_alpha_h/final")
SCENARIO_SELECTION_GRIDDED_AFTER_METADATA = "scendraft2"

# where to save plots of this script  
plots_path = path_scen_cmip7 / "plots"
plots_path.mkdir(exist_ok=True, parents=True)

# %% [markdown]
# ## CMIP7 Metadata Configuration
#
# Control file naming conventions for gridded scenario files.
#
# - `CMIP_ERA`: Set to "CMIP6Plus" for CMIP7 data (following ESGF conventions)
# - `FIXED_METADATA`: If `True`, uses short scenario name (`SCENARIO_SELECTION_GRIDDED_AFTER_METADATA`)
#   in filenames; if `False`, uses full model and scenario names
#
# **Example filename patterns:**
# - `FIXED_METADATA=True`: `CO2-em-anthro_input4MIPs_emissions_CMIP6Plus_IIASA-scendraft2_gn_202201-210012.nc`
# - `FIXED_METADATA=False`: `CO2-em-anthro_input4MIPs_emissions_CMIP6Plus_IIASA-REMIND-MAgPIE-3.5-4.11-SSP1---Very-Low-Emissions_gn_202301-210012.nc`

# %% [markdown]
# ## Sector and Gas Definitions
#
# Define which sectors and gases to process. The script loops over sectors × gases
# to create one GIF per combination.
#
# **CMIP7 Sector Categories:**
# - `SECTORS_ANTHRO`: Anthropogenic emissions (gridded in *-em-anthro files)
# - `SECTORS_AIR`: Aircraft emissions (gridded in *-AIR-anthro files)
# - `SECTORS_OPENBURNING`: Open burning emissions (gridded in *-em-openburning files)
#
# **Why these groupings matter:**
# CMIP7 requires separate NetCDF files for anthro/openburning/aircraft because they
# use different vertical distributions, seasonality patterns, and proxy datasets.
# See CMIP7 data request for details on this categorization.
#
# **Sector name mapping:**
# The `sector_dict` maps verbose IAM sector names to shorter display names used in
# the main execution loop below.

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
    '**Other Capture and Removal'
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
    """
    Read a NetCDF file using xarray with optional chunking for memory management.
    
    Parameters
    ----------
    f : str
        Filename of the NetCDF file to read.
    loc : pathlib.Path
        Directory path where the NetCDF file is located.
    reorder_list : list of str, optional
        List of variable names to reorder the dataset. If None, original order is preserved.
    chunks : dict, optional
        Dask chunking specification. Default is {"time": 1} which loads one timestep at a time.
        Use larger chunk sizes (e.g., {"time": 12}) or {} for eager loading if memory allows.
    
    Returns
    -------
    xr.Dataset
        Loaded xarray Dataset with emission data.
    
    Notes
    -----
    - Chunking with {"time": 1} keeps memory usage low but may slow down computations
      that need multiple timesteps. For faster processing of small datasets, use {}.
    - The lock parameter (commented out) would enable thread-safe parallel reads but
      is typically not needed for sequential GIF generation.
    
    Examples
    --------
    >>> ds = read_nc_file(
    ...     f="CO2-em-anthro_input4MIPs_emissions_CMIP6Plus_IIASA-scen_gn_202201-210012.nc",
    ...     loc=Path("results/cmip7_esgf_v0_alpha_h/final"),
    ...     chunks={"time": 12}  # Load one year at a time
    ... )
    """
    ds = xr.open_dataset(
        loc / f,
        engine="netcdf4",
        chunks=chunks,
        # lock=lock  # Uncomment for thread-safe parallel reads
    )
    
    if reorder_list is not None:
        ds = ds[reorder_list]  
   
    return ds

# %%
# sample variable
var = "CO2-em-anthro"

# %%
# load a CMIP7 scenario sample file
FIXED_METADATA = True
CMIP_ERA = "CMIP6Plus"

if FIXED_METADATA:
    scen_cmip7_data_file = f"{var}_input4MIPs_emissions_{CMIP_ERA}_IIASA-{SCENARIO_SELECTION_GRIDDED_AFTER_METADATA}_gn_202201-210012.nc"
else:
    scen_cmip7_data_file = f"{var}_input4MIPs_emissions_{CMIP_ERA}_IIASA-{MODEL_SELECTION_GRIDDED}-{SCENARIO_SELECTION_GRIDDED}_gn_202301-210012.nc"

scen_ds = read_nc_file(
    f = scen_cmip7_data_file,
    loc = path_scen_cmip7
)
scen_ds


# %% [markdown]
# ## Emissions Grid GIF Creation
#
# ### Visualization Design Choices
#
# **Projection:** Robinson projection (`ccrs.Robinson()`) provides a balanced view
# of global emissions suitable for presentations. It minimizes distortion for both
# area and shape compared to Mercator or Plate Carrée projections.
#
# **Color Scaling:** Uses percentile-based normalization (2nd-99th percentile) rather than
# strict min-max to avoid point-source emissions (e.g., power plants, shipping lanes)
# dominating the linear color scale. This makes spatial patterns more visible across
# most of the map.
# - `vmin`: 2nd percentile (typically ~0, except for negative CDR emissions)
# - `vmax`: 99th percentile (excludes extreme outliers)
#
# **Animation Speed:**
# - Base FPS: 10 frames/second
# - Speed multiplier: 1× (adjustable)
# - Pause at end: 5 seconds (= 50 frames at 10 FPS) to allow viewers to see final state
#
# **File Naming Convention:** `{GAS}-{SECTOR}_emissions_animation.gif`
# Example: `CO2-Energy_emissions_animation.gif`

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
    """
    Create an animated GIF showing emission evolution over time for a gas-sector combination.
    
    Parameters
    ----------
    ds : xr.Dataset
        Gridded emission dataset with dimensions (time, sector, lat, lon) and data variables
        named `{gas}_em_anthro` (e.g., `CO2_em_anthro`).
    sector : str
        Sector name as it appears in `ds.sector` coordinate (e.g., "Energy", "Transportation").
    gas : str
        Gas name used for variable lookup and output filename (e.g., "CO2", "CH4", "BC").
    times : xr.DataArray or list
        Time coordinates to animate. Typically `ds.time` for all timesteps.
    out_path : pathlib.Path
        Directory where the output GIF will be saved.
    show : bool, optional
        If True, display the animation in the notebook/script. Default is False.
    
    Returns
    -------
    None
        Saves GIF to `{out_path}/{gas}-{sector}_emissions_animation.gif`.
    
    Notes
    -----
    **Color Normalization Strategy:**
    Uses 2nd-99th percentile instead of min-max to prevent point sources from
    dominating the color scale. This makes spatial patterns visible across the map.
    
    **GIF Speed Settings:**
    - 10 FPS base rate with 1× speed multiplier = 10 FPS output
    - 5-second pause at end (50 frames) lets viewers see the final state
    - Adjust `speed` variable in function to change playback rate
    
    **Error Handling:**
    - Returns silently if sector doesn't exist in dataset (prints available sectors)
    - Skips animation if all values are NaN (e.g., sector-gas combo not applicable)
    - FileNotFoundError in calling code handles missing input files gracefully
    
    **Memory Management:**
    Loads all timesteps into memory (`da_all`) to compute consistent color scale.
    For very large grids, consider processing subsets or increasing chunk sizes.
    
    Examples
    --------
    >>> plot_gif_gas_sector(
    ...     ds=scen_ds,
    ...     sector="Energy",
    ...     gas="CO2",
    ...     times=scen_ds.time,
    ...     out_path=Path("results/plots"),
    ...     show=False
    ... )
    LOADING: sector 'Energy', gas 'CO2'.
    WRITING (start): 'results/plots/CO2-Energy_emissions_animation.gif'.
    WRITING (finished): 'results/plots/CO2-Energy_emissions_animation.gif'.
    """
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

    # Color scale fixed across frames using percentiles to avoid point-source dominance
    # vmin: 2nd percentile (generally ~0 except for negative CDR emissions)
    # vmax: 99th percentile (excludes extreme outliers like shipping lanes, power plants)
    vmin = float(np.percentile(da_all.values[~np.isnan(da_all.values)], 2))
    vmax = float(np.percentile(da_all.values[~np.isnan(da_all.values)], 99))
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

    # Update function: called once per frame to refresh the map data and title
    def update(i):
        # Handle both normal frames (i < len(times)) and pause frames (i >= len(times))
        if i < len(times):
            frame = da_all.isel(time=i)
        else:  # Repeat the last frame during pause period
            frame = da_all.isel(time=-1)

        # Update the map with new emission values
        im.set_array(np.ravel(frame.values))
        
        # Update title with current timestamp
        t_index = min(i, len(times)-1)
        t_str = time_index[t_index].strftime("%Y-%m-%d") \
            if hasattr(time_index[t_index], "strftime") else str(time_index[t_index])[:10]
        title.set_text(f"{gas}, sector: {sector}, time: {t_str}")
        return (im,)

    
    # GIF speed specifications: 10 FPS with 5-second pause at end
    base_fps = 10
    speed = 1  # Adjust this multiplier to change playback speed (e.g., 2 = 2x faster)
    fps = int(base_fps * speed)
    pause = 5  # Pause duration at end of GIF (in seconds)
    pause_frames = pause * fps  # Convert to frame count
    writer = PillowWriter(fps=fps)  # Pillow-based GIF writer

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
    
    # Close figure to free memory after saving
    plt.close(fig)

# %% [markdown]
# # Main Execution Loop
#
# ## Nested Loop Structure: Sectors × Gases
#
# The script loops over:
# 1. **Outer loop:** 8 anthropogenic sectors (Energy, Industrial, Transportation, etc.)
# 2. **Inner loop:** 10 emission species (CO2, CH4, N2O, BC, OC, SO2, NOx, NH3, CO, NMVOC)
#
# For each gas-sector combination:
# 1. Construct the filename based on `FIXED_METADATA` flag
# 2. Load the NetCDF file using `read_nc_file()`
# 3. Call `plot_gif_gas_sector()` to create and save the GIF
# 4. Catch `FileNotFoundError` and continue (some gas-sector combos may not exist)
#
# ## Expected Runtime
#
# - ~2-5 minutes per GIF (depending on grid resolution and number of timesteps)
# - 8 sectors × 10 gases = 80 potential GIFs
# - **Total runtime: ~3-7 hours** for full execution (varies by system)
#
# ## Try-Except Logic
#
# The `try-except FileNotFoundError` block handles cases where:
# - A gas-sector combination doesn't exist in the gridding output
# - Files are named differently than expected
# - Data is stored in a different directory
#
# This allows the script to process all available combinations without manual filtering.

# %% [markdown]
# # Output and Usage
#
# ## Output Files
#
# **Location:** `{path_scen_cmip7}/plots/`
#
# **Naming Pattern:** `{GAS}-{SECTOR}_emissions_animation.gif`
#
# **Examples:**
# - `CO2-Energy_emissions_animation.gif`
# - `CH4-Transportation_emissions_animation.gif`
# - `BC-Residential, Commercial, Other_emissions_animation.gif`
#
# **Typical File Sizes:**
# - 0.5° resolution: ~2-10 MB per GIF
# - 0.1° resolution: ~10-50 MB per GIF
# - Size depends on: grid resolution, number of timesteps, color complexity
#
# **Viewing Results:**
# - Open GIFs directly in any image viewer or web browser
# - Embed in presentations (PowerPoint, Google Slides, etc.)
# - Share via email or cloud storage for collaborator review

# %% [markdown]
# ## Usage Examples
#
# ### Quick Test Run (Single Sector/Gas)
#
# To test the script quickly, restrict to one sector and one gas:
#
# ```python
# # Comment out the full loops below and run this instead:
# for s in ["Energy"]:
#     for g in ["CO2"]:
#         var = f"{g}-em-anthro"
#         if FIXED_METADATA:
#             scen_cmip7_data_file = f"{var}_input4MIPs_emissions_{CMIP_ERA}_IIASA-{SCENARIO_SELECTION_GRIDDED_AFTER_METADATA}_gn_202201-210012.nc"
#         else:
#             scen_cmip7_data_file = f"{var}_input4MIPs_emissions_{CMIP_ERA}_IIASA-{MODEL_SELECTION_GRIDDED}-{SCENARIO_SELECTION_GRIDDED}_gn_202301-210012.nc"
#         
#         scen_ds = read_nc_file(f=scen_cmip7_data_file, loc=path_scen_cmip7)
#         plot_gif_gas_sector(ds=scen_ds, sector=s, gas=g, times=scen_ds.time, out_path=plots_path)
# ```
#
# ### Production Run (All Combinations)
#
# The default code below runs all 8 sectors × 10 gases. Expected runtime: ~3-7 hours.
#
# ### Subset of Data
#
# To process only specific gases or sectors, modify the loop lists:
#
# ```python
# # Only greenhouse gases:
# for g in ["CO2", "CH4", "N2O"]:
#     # ...
#
# # Only key sectors:
# for s in ["Energy", "Industrial", "Transportation"]:
#     # ...
# ```
#
# ### Adjust GIF Speed/Quality
#
# Modify parameters inside `plot_gif_gas_sector()`:
#
# ```python
# # Faster playback (2× speed):
# speed = 2
#
# # Higher quality (may increase file size):
# anim.save(out_file_path, writer=writer, dpi=200)  # default is 150
#
# # Shorter pause at end:
# pause = 2  # seconds (default is 5)
# ```
#
# ### Performance Tips
#
# - **Faster iteration:** Reduce time range by slicing `scen_ds.time[::5]` (every 5th year)
# - **Lower memory usage:** Keep `chunks={"time": 1}` in `read_nc_file()`
# - **Smaller file sizes:** Reduce `dpi` from 150 to 100 in `anim.save()`

# %% [markdown]
# ## Dependencies and Prerequisites
#
# **Required Python Packages:**
# - `xarray`: NetCDF file I/O and data manipulation
# - `matplotlib`: Plotting and animation framework
# - `matplotlib.animation`: FuncAnimation and PillowWriter for GIF creation
# - `cartopy`: Map projections (Robinson) and coastlines
# - `numpy`: Array operations and percentile calculations
# - `dask`: Lazy loading and chunked processing (via xarray)
# - `Pillow` (installed as `pillow`): GIF image writing
#
# Install missing packages:
# ```bash
# conda install -c conda-forge xarray matplotlib cartopy pillow dask
# ```
#
# **Input Files Required:**
# - Gridded NetCDF scenario files in `{path_scen_cmip7}/` with naming pattern:
#   `{gas}-em-anthro_input4MIPs_emissions_{CMIP_ERA}_IIASA-{scenario}_gn_{start}-{end}.nc`
# - Must contain:
#   - Dimensions: `time`, `sector`, `lat`, `lon`
#   - Variables: `{gas}_em_anthro` (e.g., `CO2_em_anthro`)
#   - Coordinates: Sector names matching those in the loop (e.g., "Energy", "Industrial")
#
# **Typical Memory Requirements:**
# - ~4-8 GB RAM for 0.5° resolution grids with monthly timesteps (2022-2100)
# - ~16-32 GB RAM for 0.1° resolution grids
# - Memory scales with: grid resolution, number of timesteps, chunk size
#
# **Expected Runtime Estimates:**
# - Single GIF (one sector-gas combo): ~2-5 minutes
# - Quick test (1 sector × 3 gases): ~10-15 minutes
# - Full run (8 sectors × 10 gases): ~3-7 hours
# - Runtime varies by: CPU speed, disk I/O, grid resolution, number of timesteps

# %%
# Run the plot for all sector-gas combinations

for s in [
        "Energy", "Industrial", "International Shipping", 
        "Residential, Commercial, Other", 
        "Solvents Production and Application",
        "Transportation", 
        "Waste",
        "Other Capture and Removal"
        ]:
    for g in [
        "BC", 
        "CO", 
        "CO2", 
        "NOx", 
        "OC", 
        "SO2", # "Sulfur",
        "CH4",
        "N2O", 
        "NH3", 
        "NMVOC", # "VOC",
        ]:

        try:
            var = f"{g}-em-anthro"
            if FIXED_METADATA:
                scen_cmip7_data_file = f"{var}_input4MIPs_emissions_{CMIP_ERA}_IIASA-{SCENARIO_SELECTION_GRIDDED_AFTER_METADATA}_gn_202201-210012.nc"
            else:
                scen_cmip7_data_file = f"{var}_input4MIPs_emissions_{CMIP_ERA}_IIASA-{MODEL_SELECTION_GRIDDED}-{SCENARIO_SELECTION_GRIDDED}_gn_202301-210012.nc"

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
        except FileNotFoundError:
            continue

# %% [markdown]
# ## Biomass Burning and Aircraft Emissions
#
# The same workflow applies to open burning and aircraft emissions. To create animations
# for these emission types:
#
# **For Biomass Burning (Open Burning):**
# 1. Change file pattern from `"{gas}-em-anthro"` to `"{gas}-em-openburning"`
# 2. Use `SECTORS_OPENBURNING` list instead of manually defined sectors
# 3. Example sectors: "Agricultural Waste Burning", "Forest Burning", "Grassland Burning", "Peat Burning"
#
# **For Aircraft:**
# 1. Change file pattern from `"{gas}-em-anthro"` to `"{gas}-AIR-anthro"`
# 2. Use `SECTORS_AIR` list (single sector: "Aircraft")
#
# **Implementation Example:**
# ```python
# # Open burning
# for s in SECTORS_OPENBURNING:
#     for g in ["BC", "CO", "NOx", "OC", "SO2", "CH4", "N2O", "NH3", "NMVOC"]:
#         var = f"{g}-em-openburning"  # Changed from em-anthro
#         # ... rest of code same as above ...
#
# # Aircraft
# for s in SECTORS_AIR:
#     for g in ["BC", "CO", "NOx", "OC", "SO2"]:
#         var = f"{g}-AIR-anthro"  # Changed from em-anthro
#         # ... rest of code same as above ...
# ```
#
# **Note:** Not all gases are emitted by all sectors. Adjust the gas list accordingly:
# - Aircraft typically emits: BC, CO, NOx, OC, SO2
# - Open burning emits most species except CO2 (depends on scenario)