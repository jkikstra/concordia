# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
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


# -

# **Documentation:**
# https://docs.google.com/document/d/1H9sKOkTLC1oDxEWUNurXqilkEoz3obH5J5rk5PCTXvk/edit?tab=t.ozl8f8vi3kgh#heading=h.u0e4zw58ce7s

# **Observations**
# (_Last update: 09/08/2025_)
#
# Notes: 
# * ...
#
# To be changed:
# * ...
#
# Checked to be correct:
# * ...

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
# * ...
#


# # Functions

# ## Reading in


def read_nc_file(f, loc, reorder_list=None):
    ds = xr.open_dataset(loc / f)

    if reorder_list is not None:
        ds = ds[reorder_list]
    
    return ds

# +
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


# -

# ## Check RESCUE proxy data

rescue_folder = Path("C:/Users/kikstra/IIASA/ECE.prog - Documents/Projects/CMIP7/IAM Data Processing/concordia_cmip7_v0_testing/input/gridding/proxy_rasters")

rescue_ds = read_nc_file(f = "CDR_CO2.nc", 
                         loc = rescue_folder)

rescue_ds

# ### Change the years of the proxy


# #### Run update for 2023, 2024, 2025, and 5-yearly afterwards 

# +
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


def add_sector_copy(
    ds: xr.Dataset,
    source_sector: str,
    new_sector: str,
    varname: str = "emissions",
    overwrite: bool = False,
) -> xr.Dataset:
    """
    Copy data for `source_sector` to a new sector `new_sector` along the `sector` dim.

    - If `new_sector` exists and `overwrite=False`, returns ds unchanged.
    - If `overwrite=True`, replaces `new_sector` with a copy of `source_sector`.
    """
    if "sector" not in ds.dims:
        raise ValueError("Dataset has no 'sector' dimension.")
    if varname not in ds.data_vars:
        raise ValueError(f"Variable '{varname}' not found in dataset.")
    if source_sector not in ds["sector"].values:
        raise KeyError(f"Source sector '{source_sector}' not found.")
    
    # If target exists
    if new_sector in ds["sector"].values:
        if not overwrite:
            # nothing to do
            return ds
        # drop existing target before replacing
        ds = ds.drop_sel(sector=new_sector)

    # Take source slice and reintroduce sector dim with the new label
    da_src = ds[varname].sel(sector=source_sector)
    # ensure sector dim present for concat
    if "sector" not in da_src.dims:
        da_src = da_src.expand_dims(sector=[source_sector])
    da_new = da_src.sel(sector=source_sector).expand_dims(sector=[new_sector])

    # Concat along sector
    dims_order = ds[varname].dims  # preserve original order
    da_out = xr.concat([ds[varname], da_new], dim="sector").transpose(*dims_order)

    # Return updated dataset (other variables/coords unchanged)
    return ds.assign({varname: da_out})
# -




proxy_file_sector = "CDR" # same for 'openburning' and 'aircraft'
for gas in [
    "CO2"
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



