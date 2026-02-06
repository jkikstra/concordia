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
import pandas_indexing as pix
import pandas as pd
import numpy as np
from concordia.cmip7 import utils as cmip7_utils
from typing import Dict, Tuple, List
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import colors
import cartopy.crs as ccrs


# %% [markdown]
# Download data from ESGF, Jarmo has them stored on his SanDisk drive and on H: drive

# %% [markdown]
# **Documentation:**
# https://docs.google.com/document/d/1H9sKOkTLC1oDxEWUNurXqilkEoz3obH5J5rk5PCTXvk/edit?tab=t.2uxf5irnitm7

# %% [markdown]
# **Observations**
# (_Last update: 09/08/2025_)
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
# * ...
#


# %% [markdown]
# # Functions

# %% [markdown]
# ## Reading in


# %%
def read_nc_file(f, loc, reorder_list=None, rename_sectors_cmip6=None):
    ds = xr.open_dataset(loc / f)

    if reorder_list is not None:
        ds = ds[reorder_list]
    
    if rename_sectors_cmip6:
        ds = ds_reformat_cmip6_to_cmip7(ds)
    
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
        label: extract_1d_mean_over_years(ds, varname, years, lat_or_long)
        for label, years in year_ranges.items()
    }
    return axis, profiles


def plot_multiple_variables_1d_comparison(
    base_path: Path,
    varnames: List[str],
    year_ranges: Dict[str, List[int]],
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
                ax.plot(lat, profile, color="grey", linewidth=0.7, alpha=0.3, zorder=1)

        # Foreground profiles (colored)
        for i, (label, years) in enumerate(year_ranges.items()):
            profile = extract_1d_mean_over_years(ds, varname, years, lat_or_long)
            ax.plot(lat, profile, label=label, linewidth=1.2, zorder=2)  # Matplotlib assigns color automatically

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

def plot_maps_years_bb4cmip_sectoral_data(
    ds,
    years: list,
    ncols: int = 3,
    proj=ccrs.Robinson(),
    save_pdf: bool = True,
    output_file: str = "bb4cmip_sectoral_maps.pdf"
):
    """
    Plot BB4CMIP sectoral maps for given years from an xarray.Dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset containing the variable to plot.
    years : list
        List of years to plot.
    ncols : int, optional
        Number of columns in the subplot grid, default = 3.
    proj : cartopy.crs projection, optional
        Map projection to use, default = Robinson.
    save_pdf : bool, optional
        If True, save the plot as a PDF file. Default = True.
    output_file : str, optional
        Filename for the saved PDF. Default = "bb4cmip_sectoral_maps.pdf".
    """

    nrows = int(np.ceil(len(years) / ncols))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(6 * ncols, 4.5 * nrows),
        subplot_kw={"projection": proj}
    )

    # Flatten axes safely
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]

    for i, yr in enumerate(years):
        da = (
            ds.sel(year=yr)
            .squeeze()
            .__xarray_dataarray_variable__  # replace if needed
        )

        da.plot.pcolormesh(
            ax=axes[i],
            transform=ccrs.PlateCarree(),
            cmap="GnBu",
            robust=True,
            cbar_kwargs={"orientation": "horizontal", "shrink": 0.65},
        )
        axes[i].set_title(str(yr))
        axes[i].coastlines()

    # Remove any unused axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()

    if save_pdf:
        output_path = Path(output_file)
        fig.savefig(output_path, bbox_inches="tight")
        print(f"Saved PDF to: {output_path.resolve()}")

    plt.show()

def plot_maps_sectors(ds, sectors, ncols=3, year=2100, month=1, proj=ccrs.Robinson()): 
    # used in cdr_maps to plot CDR_CO2 RESCUE proxy maps

    nrows = int(np.ceil(len(sectors) / ncols))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(6 * ncols, 4.5 * nrows),
        subplot_kw={"projection": ccrs.Robinson()}
    )

    # Flatten axes safely
    if isinstance(axes, np.ndarray):
        axes = axes.flatten() # make indexing easier
    else:
        axes = [axes]

    for i, sector in enumerate(sectors):
        da = (
            ds.sel(sector=sector, year=year, month=month)
            .squeeze()
            .emissions  # or whatever your variable is
        )

        # Plot directly with xarray's .plot.pcolormesh
        da.plot.pcolormesh(
            ax=axes[i],
            transform=ccrs.PlateCarree(),  
            cmap="GnBu",
            robust=True,
            cbar_kwargs={"orientation": "horizontal", "shrink": 0.65},
        )
        axes[i].set_title(sector)
        axes[i].coastlines()

    # Remove any unused axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
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
    species = [
        "BC", "OC",
        "CO2", "CH4",
        "CO", "N2O",
        "NH3", "NMVOCbulk",
        "NOx", "SO2"
              ],
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
        "1910s": list(range(1910, 1920)),
        "1920s": list(range(1920, 1930)),
        "1930s": list(range(1930, 1940)),
        "1940s": list(range(1940, 1950)),
        "1950s": list(range(1950, 1960)),
        "1960s": list(range(1960, 1970)),
        "1970s": list(range(1970, 1980)),
        "1980s": list(range(1980, 1990)),
        "1990s": list(range(1990, 2000)),
        "2000s": list(range(2000, 2010)),
        "2010s": list(range(2010, 2020)),
        # "2020s": list(range(2020, 2024))
    }

    # do latitudinal plots, comparing possible ranges against background 10yr means  
    if lat:
        plot_multiple_variables_1d_comparison(
            base_path=dres_location,
            varnames=species,
            year_ranges=year_ranges,
            year_ranges_background=year_ranges_background,
            lat_or_long="latitude",
            output_file=Path("C:/Users/kikstra/Documents/GitHub/concordia/results") / "biomass_burning" / "latitudinal_comparison.pdf"
        )
    
    # do longitudinal plots, comparing possible ranges against background 10yr means  
    if lon:
        plot_multiple_variables_1d_comparison(
            base_path=dres_location,
            varnames=species,
            year_ranges=year_ranges,
            year_ranges_background=year_ranges_background,
            lat_or_long="longitude",
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
         lat=True,
         lon=False)

# %%

# %% [markdown]
# ## BB4CMIP7 - sectoral emisisons 

# %%
VERSION_BB4CMIP7 = "2-0"
# v2.0 product is already 5yr smoothed and only goes until 2021
# v2.1 is being downloaded now, raw.

for gas in [
    # "BC"
    "OC",
    "SO2"
]:
    for sector in [
        "AGRI",
        "BORF",
        "DEFO",
        "PEAT",
        "SAVA",
        "TEMF",
    ]:
        gas_sector = gas + "_" + sector 
        cmip7_bb4cmip7_sectoral_folder = Path("C:/Users/kikstra/Documents/GitHub/emissions_harmonization_historical/data/bb4cmip7/maps/") / VERSION_BB4CMIP7 
        sectoral_cmip7_ds = read_nc_file(f = f"{gas_sector}.nc", 
                                loc = cmip7_bb4cmip7_sectoral_folder)
        plot_maps_years_bb4cmip_sectoral_data(sectoral_cmip7_ds,
                                            years=range(1999,2022),   
                                                    output_file=Path("C:/Users/kikstra/Documents/GitHub/concordia/results") / "biomass_burning" / f"{VERSION_BB4CMIP7}_{gas_sector}.pdf"
                                                    )


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
# ### Change the years of the proxy


# %% [markdown]
# #### Example 

# %% 
# Create a copy so we don't overwrite the original accidentally
ds_new = rescue_ds.copy()

# Get the current years as a NumPy array
years = ds_new['year'].values.copy()

# Replace 2015 → 2023, 2020 → 2025
years = np.where(years == 2015, 2023, years)
years = np.where(years == 2020, 2025, years)

# Assign the updated years back
ds_new = ds_new.assign_coords(year=years)

print(ds_new['year'].values)

# %% [markdown]
# #### Run update for 2023 and 2025

# %%
def ds_rename_recent_years_rescue_to_cmip7(ds):

    # all the above in 1 line:
    ds = ds.assign_coords(year=[2023 if y == 2015 else 2025 if y == 2020 else y for y in ds.year.values])

    return ds

def add_year_copy(ds, source_year, new_year):
    if new_year in ds.year.values:
        print(f"Year {new_year} already exists in dataset — skipping.")
        return ds
    ds_source = ds.sel(year=source_year)
    ds_new = ds_source.assign_coords(year=[new_year])
    return xr.concat([ds, ds_new], dim="year").sortby("year")

def ds_add_missing_years_by_copying_one_year(ds, old_year_to_copy_from: int, new_years: list):
        
    for yr in new_years:
        ds = add_year_copy(ds, source_year=old_year_to_copy_from, new_year=yr)
    
    # NOTE: could rewrite, or add an extra function, 
    # that takes a dictionary, mapping which old_year
    # each new_year should be copied from 

    return ds

# %%


# %% 
proxy_file_sector = "openburning" # same for 'aircraft'
for gas in [
    "BC", "OC",
    "CO2", "CH4",
    "CO", #"N2O", # not available
    "NH3", 
    "VOC",
    "NOx", 
    "Sulfur"
]:
    filename = f"{proxy_file_sector}_{gas}.nc"
    print(f"Updating {gas} {proxy_file_sector} proxy file.")
    rescue_ds = read_nc_file(f = filename, 
                         loc = rescue_folder)
    
    new_ds = ds_rename_recent_years_rescue_to_cmip7(rescue_ds)
    # ensure we have 5-yearly timesteps in the proxy
    new_ds = ds_add_missing_years_by_copying_one_year(new_ds, 
                                                      old_year_to_copy_from=2023, 
                                                      new_years=[2024] + list(range(2025, 2101, 5)))
    
    # write out
    outpath = rescue_folder / f"renamed_{proxy_file_sector}_rescue_for_cmip7round2" / filename
    new_ds.to_netcdf(
            outpath,
            # encoding={da.name: settings.encoding},
        )
    


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
start = "2021-11"
end = "2021-11"

selected = select_time_period(dres_ds, var=varname, start=start, end=end)
selected


# %%
# Compute and plot zonal mean
zonal_mean = compute_latitudinal_mean(selected)
zonal_mean

# %%
plot_latitudinal_profile(zonal_mean, title=f"Zonal Mean {varname} ({start} to {end})")

# %%
time_ranges = {
        "30yr average (1994-2023)": ("1994-01", "2023-12"),
        "10yr average (2014-2023)": ("2014-01", "2023-12"),
        "5yr average (2019-2023)": ("2019-01", "2023-12"),
    }

compare_timeslices(dres_ds, var=varname, time_ranges=time_ranges, title="Fire Emissions from Black Carbon by Latitude")

# %%
