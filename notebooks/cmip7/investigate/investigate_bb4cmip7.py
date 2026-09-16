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
from typing import Dict, Tuple, List
from matplotlib.backends.backend_pdf import PdfPages
import cartopy.crs as ccrs

IAMC_COLS = ["model", "scenario", "region", "variable", "unit"] 

# %% [markdown]
# # Visual checks on CMIP7 new proxies

# %%
from concordia.cmip7.utils_plotting import plot_map, plot_maps, plot_maps_seasonal
from concordia.cmip7.CONSTANTS import GASES


# %%
bb_proxy_location = Path("C:/Users/kikstra/IIASA/ECE.prog - Documents/Projects/CMIP7/IAM Data Processing/concordia_cmip7_esgf_v0_alpha/input/gridding/proxy_rasters")

# ds = xr.open_dataset(bb_proxy_location / "openburning_BC_AGRI_2014_2023.nc")
ds = xr.open_dataset(bb_proxy_location / "openburning_BC_SAVA_2014_2023.nc")

ds


# %%
bb_proxy_location = Path("C:/Users/kikstra/IIASA/ECE.prog - Documents/Projects/CMIP7/IAM Data Processing/concordia_cmip7_esgf_v0_alpha/input/gridding/proxy_rasters")

for species in GASES:
    for burning_sector in ["SAVA", "PEAT", "AGRI", "FRTB"]:
        # Initialize empty dataset
        combined_ds = None
        var = "emissions"

        # Loop over different files
        for file in bb_proxy_location.glob("*.nc"):

            # check if filename has BC in it, if so: perform the rest of the loop, if not continue without the rest of the loop
            if f"_{species}_" not in file.name:
                print(f"Skipping file (no {species}): {file.name}")
                continue
            if burning_sector not in file.name:
                print(f"Skipping file (no {burning_sector} burning): {file.name}")
                continue

            print(f"Processing file: {file.name}")
            
            # Read in each file
            ds = xr.open_dataset(file)
            
            # Extract proxy name from filename (remove extension and prefixes)
            proxy_name = file.stem
            # Clean up common prefixes if needed
            if proxy_name.startswith("openburning_"):
                proxy_name = proxy_name.replace("openburning_", "")
            
            # Replace the sector coordinate with proxy_name
            # First, assign the new value to the sector coordinate
            ds = ds.assign_coords(sector=[proxy_name])

            plot_maps_seasonal(
                        ds,
                        sectors=None,
                        variable="emissions",
                        year=2023,
                        title=f"{species} {burning_sector} Emissions: {proxy_name}",
                        save_as="pdf-png",
                        filename=file.parent / (file.stem)
                    )
            
            # Add to combined dataset
            if combined_ds is None:
                combined_ds = ds
            else:
                # Concatenate along the sector dimension
                try:
                    combined_ds = xr.concat([combined_ds, ds], dim="sector")
                except Exception as e:
                    print(f"Error concatenating {file.name}: {e}")
                    continue

        # Display the combined dataset
        if combined_ds is not None:
            print(f"Combined dataset with {len(combined_ds.sector)} sectors:")
            print(combined_ds)
            print(f"Sector names: {list(combined_ds.sector.values)}")
        else:
            print("No .nc files found in the directory")
        
        # plot compete dataset
        plot_maps_seasonal(
                    combined_ds,
                    sectors=list(combined_ds.sector.values),
                    variable="emissions",
                    ncols=4,
                    year=2023, # we have all scenario years, pick one.
                    title=f"{species} {burning_sector} Emissions: {file.stem}",
                    save_as="pdf-png",
                    filename=file.parent / (f"openburning_{species}_{burning_sector}_alloptions")
                )


# %% [markdown]
# # Investigate 5-year smoothed BB4MCIP7 data directly

# %% [markdown]
# Download data from ESGF, Jarmo has them stored on his SanDisk drive and on H: drive

# %% [markdown]
# **Documentation:**
# https://docs.google.com/document/d/1H9sKOkTLC1oDxEWUNurXqilkEoz3obH5J5rk5PCTXvk/edit?tab=t.2uxf5irnitm7

# %% [markdown]
# **Observations**
# (_Last update: 04/08/2028_)
#
# Notes: 
# * Huge latitudinal spike around -39 degree coming from the 2021-12 slice. In BC emissions. 
# * Need to look by sector, e.g. deforestation pattern (e.g. brazil) will probably be time-dependent?
# * Total latitudinal pattern relatively constant, but 30yr does look markedly smoother, with less extreme peaks (e.g. the 40degree south one)
#   * Boreal burning has increased, much higher in last 5yr, and last 10yr.
#   * Savannah (I think) fires between -20 and 0 degrees have are a bit higher in 30yr avg than in 10 or 5yr; same for savannah burning around 10 degrees north.
#   * 
#
# To be changed:
# * ...
#
# Checked to be correct:
# * If we're gridding with a non-5-year timestep, then we should most likely perform a smoothing for the first years of the scenario?
# * ...

# %% [markdown]
# **Ideas:**
#
# What data to test:
# * ...
#
# Tests to perform:
#
# _Automated_:
# * ...
# _Other_:
# * ...
#
# Visualisations to perform:
# * (global) standard deviation 
# * (latitudinal) by "sector"
# * (maps) by "sector"
#
# Statistical tests to perform
# * (latitudinal) by "sector", over time: do we see patterns?
#


# %% [markdown]
# # Functions

# %%
"Latitudinal" if "lat" == "latitude" else "Longitudinal"

# %% [markdown]
# ## Reading in


# %%
def read_nc_file(f, loc, reorder_list=None):
    ds = xr.open_dataset(loc / f)

    if reorder_list is not None:
        ds = ds[reorder_list]
    
    return ds

# %%
def select_time_period(ds: xr.Dataset, var: str, start: str, end: str) -> xr.DataArray:
    """Select variable in the dataset and limit to time range"""
    return ds[var].sel(time=slice(start, end))

def compute_latitudinal_mean(em: xr.DataArray) -> xr.DataArray:
    """
    Compute zonal mean (average over longitude and time)
    Result is 1D over latitude
    """
    return em.mean(dim=["time", "longitude"])

def compute_longitudinal_mean(em: xr.DataArray) -> xr.DataArray:
    """
    Compute meridional mean (average over latitude and time)
    Result is 1D over longitude
    """
    return em.mean(dim=["time", "latitude"])

def plot_longitudinal_profile(long_data: xr.DataArray, title: str = "Meridional Mean Emissions"):
    """Plot the longitudinal profile"""
    plt.figure(figsize=(10, 5))
    plt.plot(long_data["longitude"], long_data, label=long_data.name)
    plt.title(title)
    plt.xlabel("Longitude")
    plt.ylabel("Emissions [unit]")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
def plot_latitudinal_profile(lat_data: xr.DataArray, title: str = "Zonal Mean Emissions"):
    """Plot the latitudinal profile"""
    plt.figure(figsize=(10, 5))
    plt.plot(lat_data["latitude"], lat_data, label=lat_data.name)
    plt.title(title)
    plt.xlabel("Latitude")
    plt.ylabel("Emissions [unit]")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_latitudinal_profile_multiple(ds: xr.Dataset, var: str, time_ranges: dict, title: str = ""):
    """
    Compare latitudinal profiles for multiple time slices.
    
    Parameters:
        ds: xarray.Dataset
        var: str — variable name (e.g. "BC")
        time_ranges: dict — {"Label": ("YYYY-MM", "YYYY-MM")}
        title: str — plot title
    """
    plt.figure(figsize=(10, 5))

    for label, (start, end) in time_ranges.items():
        em = select_time_period(ds, var, start, end)
        zonal_mean = compute_latitudinal_mean(em)
        plt.plot(zonal_mean["latitude"], zonal_mean, label=label)

    plt.title(title or f"Latitudinal Comparison of {var}")
    plt.xlabel("Latitude")
    plt.ylabel(f"{var} Emissions [unit]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def compare_timeslices_zonal_means(
    ds: xr.Dataset,
    var: str,
    time_ranges: Dict[str, Tuple[str, str]],
) -> Tuple[List[float], Dict[str, List[float]]]:
    latitudes = ds["latitude"].values
    zonal_means = {}

    for label, (start, end) in time_ranges.items():
        em = select_time_period(ds, var, start, end)
        zonal_mean = compute_latitudinal_mean(em)
        zonal_means[label] = zonal_mean.values

    return latitudes, zonal_means

def plot_multiple_vars_to_pdf_zonal_means(
    ds: xr.Dataset,
    varnames: List[str],
    time_ranges: Dict[str, Tuple[str, str]],
    output_file: Path,
    ncols: int = 2
):
    nrows = (len(varnames) + ncols - 1) // ncols
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 4 * nrows), sharex=True, sharey=True)
    axs = axs.flatten()

    for i, var in enumerate(varnames):
        latitudes, zonal_means = compare_timeslices_zonal_means(ds, var, time_ranges)
        ax = axs[i]
        for label, values in zonal_means.items():
            ax.plot(latitudes, values, label=label)
        ax.set_title(var)
        ax.set_xlabel("Latitude")
        ax.set_ylabel("Zonal Mean")
        ax.grid(True)
        ax.legend()

    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    fig.suptitle("Latitudinal Emissions Comparison", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(output_file, format="pdf")
    plt.close(fig)


def extract_global_mean_over_years(ds: xr.Dataset, varname: str, years: List[int]) -> xr.DataArray:
    ts = ds.sel(time=ds.time.dt.year.isin(years))[varname]
    return ts.mean(dim=["time", "longitude", "latitude"], skipna=True)


def extract_latitude_mean_over_years(ds: xr.Dataset, varname: str, years: List[int]) -> xr.DataArray:
    ts = ds.sel(time=ds.time.dt.year.isin(years))[varname]
    return ts.mean(dim=["time", "longitude"], skipna=True)

def extract_longitude_mean_over_years(ds: xr.Dataset, varname: str, years: List[int]) -> xr.DataArray:
    ts = ds.sel(time=ds.time.dt.year.isin(years))[varname]
    return ts.mean(dim=["time", "latitude"], skipna=True)

def extract_1d_mean_over_years(ds: xr.Dataset, varname: str, years: List[int], lat_or_long: str) -> xr.DataArray:
    if lat_or_long == "latitude":
        return extract_latitude_mean_over_years(ds, varname, years)
    if lat_or_long == "longitude":
        return extract_longitude_mean_over_years(ds, varname, years)

def extract_2d_mean_over_years(ds: xr.Dataset, varname: str, years: List[int]) -> xr.DataArray:
    """
    Extract and average a 2D lat-lon emissions field over the specified years.
    """
    ds_sel = ds.sel(time=ds.time.dt.year.isin(years))
    mean_field = ds_sel[varname].mean(dim="time", skipna=True)
    return mean_field
    

def compare_1d_profiles(
    ds: xr.Dataset,
    varname: str,
    year_ranges: Dict[str, List[int]],
    lat_or_long = "latitude"
) -> Tuple[xr.DataArray, Dict[str, xr.DataArray]]:
    
    axis = ds[lat_or_long]
    profiles = {
        label: extract_latitude_mean_over_years(ds, varname, years, lat_or_long)
        for label, years in year_ranges.items()
    }
    return axis, profiles


def plot_multiple_variables_latitudinal_comparison(
    base_path: Path,
    varnames: List[str],
    year_ranges: Dict[str, List[int]] = None,
    year_ranges_background: Dict[str, List[int]] = None,
    lat_or_long: str = "latitude",
    output_file: str = "latitudinal_comparison.pdf"
):
    n_vars = len(varnames)
    ncols = 2
    nrows = (n_vars + 1) // 2

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 4 * nrows), sharex=True)
    axs = axs.flatten()
    first_word = "Latitudinal" if lat_or_long == "latitude" else "Longitudinal"

    for idx, varname in enumerate(varnames):
        ds = read_nc_file(f = dres_file_structure(gas=varname), loc = base_path)
        lat = ds[lat_or_long]
        ax = axs[idx]

        # Background profiles (grey)
        if year_ranges_background:
            for label, years in year_ranges_background.items():
                profile = extract_1d_mean_over_years(ds, varname, years, lat_or_long)
                ax.plot(lat, profile, color="grey", linewidth=1.0, alpha=0.5, zorder=1)

        # Foreground profiles (colored)
        if year_ranges:
            for i, (label, years) in enumerate(year_ranges.items()):
                profile = extract_1d_mean_over_years(ds, varname, years, lat_or_long)
                ax.plot(lat, profile, label=label, linewidth=2, zorder=2)  # Matplotlib assigns color automatically

        ax.set_title(f"{varname} {first_word} Profile")
        ax.set_ylabel(f"{varname} (avg)")
        ax.grid(True)
        ax.legend()

    for j in range(idx + 1, len(axs)):
        axs[j].axis("off")

    fig.suptitle(f"{first_word} Emissions Comparison\n(background grey lines are decadal mean emissions\nfrom 1900s, 1910s, to 2010s)", 
                 fontsize=16,
                 linespacing=1.2)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    with PdfPages(output_file) as pdf:
        pdf.savefig(fig)
        plt.close(fig)


def plot_grid_2d_by_var_and_year_range(
    base_path: Path,
    varnames: List[str],
    year_ranges: Dict[str, List[int]],
    output_file: str = "spatial_comparison.pdf",
    vmin: float = None,
    vmax: float = None,
    cmap: str = "inferno"
):
    n_vars = len(varnames)
    n_years = len(year_ranges)
    year_labels = list(year_ranges.keys())

    fig, axs = plt.subplots(
        nrows=n_vars,
        ncols=n_years,
        figsize=(4 * n_years, 3.5 * n_vars),
        constrained_layout=True,
        squeeze=False,
    )

    for i, varname in enumerate(varnames):
        ds = read_nc_file(f=dres_file_structure(gas=varname), loc=base_path)

        for j, (label, years) in enumerate(year_ranges.items()):
            ax = axs[i, j]

            mean_field = extract_2d_mean_over_years(ds, varname, years)

            # Coarsen resolution to 0.5° (from 0.25°)
            coarsened = mean_field.coarsen(latitude=2, longitude=2, boundary="trim").mean()
            
            # Plot using xarray's built-in pcolormesh wrapper
            plot = coarsened.plot.pcolormesh(
                ax=ax,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                add_colorbar=False,
                robust=True,  # Optional: ignores extreme outliers for better contrast
            )

            ax.set_title(label if i == 0 else "", fontsize=12)
            ax.set_ylabel(varname if j == 0 else "", fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])

        # Add a colorbar outside the right of each row
        cbar_ax = fig.add_axes([1.01, 0.12 + (n_vars - 1 - i) * (0.8 / n_vars), 0.015, 0.8 / n_vars])
        fig.colorbar(plot, cax=cbar_ax, label=f"{varname} (mean over years)", orientation="vertical")

    fig.suptitle("Spatial Mean Fire Emissions for Different Periods", fontsize=14)

    with PdfPages(output_file) as pdf:
        pdf.savefig(fig)
        plt.close(fig)

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
import numpy as np
import xarray as xr
from typing import Optional

def plot_map_at_time(
    ds: xr.Dataset,
    varname: str,
    time_point,                        # e.g. "2020-07", "2020-07-16", np.datetime64(...)
    *,
    cmap: str = "GnBu",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    projection = ccrs.Robinson(),
    add_borders: bool = True,
    add_coastlines: bool = True,
    coarsen_to_half_degree: bool = False,  # if your native grid is 0.25°, set True to average to 0.5°
    save_pdf: bool = False,
    output_file: str = "map_at_time.pdf",
    title: Optional[str] = None,
    robust: bool = True,
):
    """
    Plot a 2D map of `varname` at a specific time point.

    - Automatically handles lat/lon vs latitude/longitude names.
    - Optional coarsening (2x2) to go from 0.25° -> 0.5°.
    - Adds borders/coastlines and supports custom projection.
    """
    # Figure out coordinate names
    lat_name = "latitude" if "latitude" in ds.dims else "lat"
    lon_name = "longitude" if "longitude" in ds.dims else "lon"

    # Select time slice
    da = ds[varname].sel(time=time_point).squeeze()

    # Optional coarsening
    if coarsen_to_half_degree:
        # 2x2 block mean: 0.25° -> 0.5°
        kwargs = {lat_name: 2, lon_name: 2}
        da = da.coarsen(boundary="trim", **kwargs).mean()

    # Build figure/axes
    fig = plt.figure(figsize=(8.5, 5))
    ax = plt.axes(projection=projection)

    # Plot
    h = da.plot.pcolormesh(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmin=vmin, vmax=vmax,
        robust=robust,
        add_colorbar=True,
        cbar_kwargs={"orientation": "horizontal", "shrink": 0.75, "pad": 0.05},
    )

    # Decorations
    if title is None:
        # try to format time nicely if it's datetime-like
        tval = np.array(da.coords["time"]).item()
        title = f"{varname} @ {tval}"
    ax.set_title(title)

    if add_coastlines:
        ax.coastlines()
    if add_borders:
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor="black")

    # Clean ticks (Cartopy global)
    ax.set_global()
    plt.tight_layout()

    if save_pdf:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_file, bbox_inches="tight")
        print(f"Saved PDF to: {Path(output_file).resolve()}")

    plt.show()



# %%
# dres_location = Path("D:/ESGF/DRES-CMIP-BB4CMIP7-2-0/atmos/mon")
# dres_location = Path("C:/Users/kikstra/IIASA/ECE.prog - Documents/Projects/CMIP7/IAM Data Processing/concordia_cmip7_v0_testing/input/gridding/DRES-CMIP-BB4CMIP7-2-0/atmos/mon")

def dres_file_structure(gas="BC", version="v20250227"):
    return Path(gas) / "gn" / version / f"{gas}_input4MIPs_emissions_CMIP_DRES-CMIP-BB4CMIP7-2-0_gn_190001-202312.nc"


# %%
def main(
    lat=False,
    lon=False,
    maps=False,
    lat_only_background=False,
    species = ["BC", "OC",
              "CO2", "CH4",
              "CO", "N2O",
              "NH3", "NMVOCbulk",
              "NOx", "SO2"],
    year_ranges = {
        "5yr avg (2019-2023)": list(range(2019, 2024)),
        "10yr avg (2014-2023)": list(range(2014, 2024)),
        "30yr avg (1994-2023)": list(range(1994, 2024)),
        "1980s (1980-1989)": list(range(1980, 1990))
    }
):
    dres_location = Path("C:/Users/kikstra/IIASA/ECE.prog - Documents/Projects/CMIP7/IAM Data Processing/concordia_cmip7_v0_testing/input/gridding/DRES-CMIP-BB4CMIP7-2-0/atmos/mon")
    
    year_ranges_background = {
        "1900s": list(range(1900, 1910)),
        "1900s": list(range(1910, 1920)),
        "1900s": list(range(1920, 1930)),
        "1900s": list(range(1930, 1940)),
        "1900s": list(range(1940, 1950)),
        "1900s": list(range(1950, 1960)),
        "1900s": list(range(1960, 1970)),
        "1900s": list(range(1970, 1980)),
        "1900s": list(range(1980, 1990)),
        "1900s": list(range(1990, 2000)),
        "2000s": list(range(2000, 2010)),
        "2010s": list(range(2010, 2020)),
        # "2020s": list(range(2020, 2024))
    }

    # do latitudinal plots, comparing possible ranges against background 10yr means  
    if lat:
        plot_multiple_variables_latitudinal_comparison(
            base_path=dres_location,
            varnames=species,
            year_ranges=year_ranges,
            year_ranges_background=year_ranges_background,
            lat_or_long="latitude",
            output_file=Path("C:/Users/kikstra/Documents/GitHub/concordia/results") / "biomass_burning" / "latitudinal_comparison.pdf"
        )
    if lat_only_background:
        plot_multiple_variables_latitudinal_comparison(
            base_path=dres_location,
            varnames=species,
            year_ranges=None,
            year_ranges_background=year_ranges_background,
            lat_or_long="latitude",
            output_file=Path("C:/Users/kikstra/Documents/GitHub/concordia/results") / "biomass_burning" / "latitudinal_comparison_onlybackground.pdf"
        )
    
    # do longitudinal plots, comparing possible ranges against background 10yr means  
    if lon:
        plot_multiple_variables_latitudinal_comparison(
            base_path=dres_location,
            varnames=species,
            year_ranges=year_ranges,
            year_ranges_background=year_ranges_background,
            lat_or_long="latitude",
            output_file=Path("C:/Users/kikstra/Documents/GitHub/concordia/results") / "biomass_burning" / "longitudinal_comparison.pdf"
        )

    # plot maps
    if maps:
        plot_grid_2d_by_var_and_year_range(
            base_path=dres_location,
            varnames=species,
            year_ranges=year_ranges,
            output_file=Path("C:/Users/kikstra/Documents/GitHub/concordia/results") / "biomass_burning" / "grid_comparison.pdf",
            cmap="plasma"
        )



# %% [markdown]
# ## Run BB4CMIP7 plots 

# %%
if __name__ == "__main__":
    main(maps=False,
         lat=False,
         lon=False,
         lat_only_background=True,species=["BC", "CO2"])

# %%

# %% [markdown]
# ## BB4CMIP7 - sectoral emisisons 

# %%
cmip7_bb4cmip7_sectoral_folder = Path("C:/Users/kikstra/Documents/GitHub/emissions_harmonization_historical/data/bb4cmip7/maps") # product is already smoothed (indicated by it only going until 2021?)

# %%
sectoral_cmip7_ds = read_nc_file(f = "SO2_SAVA.nc", 
                         loc = cmip7_bb4cmip7_sectoral_folder)

# %%
sectoral_cmip7_ds

# %% [markdown]
# ## Check RESCUE proxy data

# %%
rescue_folder = Path("C:/Users/kikstra/IIASA/ECE.prog - Documents/Projects/CMIP7/IAM Data Processing/concordia_cmip7_v0_testing/input/gridding/proxy_rasters")

# %%
rescue_ds = read_nc_file(f = "openburning_BC.nc", 
                         loc = rescue_folder)

# %%
rescue_ds

# %% [markdown]
# ## Code snippets & Interactive cells

# %%
BC_file = dres_file_structure(gas="BC")
dres_ds = read_nc_file(
    f = BC_file,
    loc = dres_location
)

# %%
dres_ds

# %%
# Select variable and period
varname = "BC"
start = "2021-12"
end = "2021-12"

selected = select_time_period(dres_ds, var=varname, start=start, end=end)
selected

# Compute and plot zonal mean
zonal_mean = compute_latitudinal_mean(selected)
zonal_mean

plot_latitudinal_profile(zonal_mean, title=f"Zonal Mean {varname} ({start} to {end})")

# Compute and plot meridional mean
meridional_mean = compute_longitudinal_mean(selected)
meridional_mean

plot_longitudinal_profile(meridional_mean, title=f"Meriodional Mean {varname} ({start} to {end})")


# %%

# %%

# %%
time_ranges = {
        "30yr average (1994-2023)": ("1994-01", "2023-12"),
        "10yr average (2014-2023)": ("2014-01", "2023-12"),
        "5yr average (2019-2023)": ("2019-01", "2023-12"),
    }

compare_timeslices(dres_ds, var=varname, time_ranges=time_ranges, title="Fire Emissions from Black Carbon by Latitude")

# %%

# %%
read_nc_file(f=dres_file_structure(gas=g),
                 loc=Path("C:/Users/kikstra/IIASA/ECE.prog - Documents/Projects/CMIP7/IAM Data Processing/concordia_cmip7_v0_testing/input/gridding/DRES-CMIP-BB4CMIP7-2-0/atmos/mon"))

# %%
t = "2021-12-16"
g = "BC"
plot_map_at_time(
    read_nc_file(f=dres_file_structure(gas=g),
                 loc=Path("C:/Users/kikstra/IIASA/ECE.prog - Documents/Projects/CMIP7/IAM Data Processing/concordia_cmip7_v0_testing/input/gridding/DRES-CMIP-BB4CMIP7-2-0/atmos/mon")), 
    g, 
    t,
    coarsen_to_half_degree=True,
    save_pdf=True,
    add_borders=False,
    add_coastlines=False,
    robust=False,
    output_file=Path("C:/Users/kikstra/Documents/GitHub/concordia/results") / "biomass_burning" / f"grid_highspikeyear_{g}_{t}.pdf",
    title=f"{g} emissions – {t}"
)

# %%
