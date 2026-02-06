# %%
# imports
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import xarray as xr
from pathlib import Path

# %%
# functions

# note: # `cell_area: xr.DataArray | None = None` (PEP 604, may require Python 3.10+); alternative would be cell_area: Optional[xr.DataArray] = None; along which we need from typing import Optional
# From a grid to global anual totals
def ds_to_annual_emissions_total_faster(gridded_data, var_name, cell_area: xr.DataArray | None = None, keep_sectors=True, sum_dims: list[str] | None = ["lat", "lon"]):
    """
    Convert gridded emissions in kg/m2/s to Mt/year using dask for faster, memory-safe computation.
    
    Parameters:
    - gridded_data: xr.Dataset containing the emission variable
    - var_name: str, name of the variable to convert
    - cell_area: xr.DataArray of shape (lat, lon), in m2
    - keep_sectors: bool, if True, retain sector info
    - sum_dims: list of dimensions to sum over (spatial dimensions)
    
    Returns:
    - xr.DataArray of Mt/year, shape (year,) or (sector, year)
    """
    da = gridded_data[var_name]
    
    # Determine optimal chunk size based on dataset characteristics
    # AIR files: have 'level' dimension but no 'sector'
    # Anthro files: have 'sector' dimension (8-10 sectors) but no 'level'
    # Openburning files: have 'sector' dimension (4 sectors) but no 'level'
    has_sector = "sector" in da.dims
    has_level = "level" in da.dims
    n_sectors = len(da.sector) if has_sector else 0
    
    # Optimized chunking: balance memory safety with computation speed
    # Key insight: spatial operations are fast, time/sector are the memory bottleneck
    if has_sector:
        # For files with many sectors (like CO2 with CDR), use moderate chunks
        if n_sectors > 15:
            time_chunk = 6  # Moderate chunks for CO2 files (25 sectors)
        elif n_sectors > 8:
            time_chunk = 12  # Standard chunks for regular anthro (8-10 sectors)
        else:
            time_chunk = 24  # Large chunks for openburning (4 sectors)
        
        chunks = {'time': time_chunk, 'sector': -1}  # Keep sectors together
    elif has_level:
        # For AIR: chunk by time and level
        chunks = {'time': 12, 'level': -1}
    else:
        # Default: just chunk time
        chunks = {'time': 24}
    
    # Spatial chunks: larger is better for performance since we're summing anyway
    # Only chunk if dimensions are very large
    if 'lat' in da.dims and 'lon' in da.dims:
        if len(da.lat) > 180 or len(da.lon) > 360:
            chunks['lat'] = 180
            chunks['lon'] = 360
        # Otherwise leave spatial dims unchunked for faster summation
    
    # Force rechunking - this prevents the MemoryError from loading entire array
    da = da.chunk(chunks)
    
    # 1. Get seconds per month
    seconds_per_month = da.time.dt.days_in_month * 24 * 60 * 60
    
    # 2. Build dimension list for summing
    sum_dims_to_use = (sum_dims or []).copy()
    if "level" in da.dims:
        sum_dims_to_use.append("level")
    
    # 3. Apply multiplications and spatial sum (all lazy operations)
    monthly = seconds_per_month * da  # kg/m2/s -> kg/m2/month
    area_weighted = cell_area * monthly  # kg/m2/month -> kg/month
    
    if sum_dims_to_use:
        kg_per_month = area_weighted.sum(dim=sum_dims_to_use)
    else:
        kg_per_month = area_weighted
    
    # 4. Group by year and sum (lazy operation)
    kg_per_year = kg_per_month.groupby("time.year").sum()
    
    # 5. Convert to Mt/year (lazy operation)
    result = kg_per_year * 1e-9
    
    # 6. Sum sectors if needed (lazy operation)
    if "sector" in result.dims and not keep_sectors:
        result = result.sum(dim="sector")
    
    # 7. Rename and ensure proper naming
    result.name = var_name
    
    # 8. Optimized computation strategy
    # The aggressive chunking above prevents memory issues, so we can compute more directly
    try:
        # Result is small (aggregated data), should fit in memory easily
        # With proper chunking, dask handles the large intermediate arrays
        return result.compute()
            
    except MemoryError as e:
        # Fallback: compute in smaller batches only if memory error occurs
        print(f"Warning: Memory error, computing in batches: {e}")
        n_result_sectors = len(result.sector) if 'sector' in result.dims else 1
        
        # For many sectors, compute by sector
        if n_result_sectors > 15 and 'sector' in result.dims:
            print(f"Computing {n_result_sectors} sectors separately...")
            computed_sectors = []
            for sector in result.sector.values:
                sector_result = result.sel(sector=sector).compute()
                computed_sectors.append(sector_result)
            return xr.concat(computed_sectors, dim='sector')
        
        # Otherwise compute in year batches
        else:
            print(f"Computing in year batches...")
            year_batch_size = 5
            years = result.year.values
            computed_chunks = []
            for i in range(0, len(years), year_batch_size):
                year_batch = years[i:i + year_batch_size]
                chunk_result = result.sel(year=year_batch).compute()
                computed_chunks.append(chunk_result)
            return xr.concat(computed_chunks, dim='year')


# note: # `cell_area: xr.DataArray | None = None` (PEP 604, may require Python 3.10+); alternative would be cell_area: Optional[xr.DataArray] = None; along which we need from typing import Optional
# From a grid to global anual totals
def ds_to_annual_emissions_total(gridded_data, var_name, cell_area: xr.DataArray | None = None, keep_sectors=True, sum_dims: list[str] | None = ["lat", "lon"], faster=True):
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
    if cell_area is None:

        from concordia.settings import Settings
        from concordia.cmip7.CONSTANTS import CONFIG
        HERE = Path(__file__).parent.parent.parent.parent / "notebooks" / "cmip7"
        dummy_settings = Settings.from_config(local_config_path=Path(HERE,
                                                            CONFIG),
                                                            version=None)
        
        areacella = xr.open_dataset(Path(dummy_settings.gridding_path, 
                             "areacella_input4MIPs_emissions_CMIP_CEDS-CMIP-2025-04-18_gn.nc"))
        cell_area = areacella["areacella"]

    
    if faster:
        # Pass all arguments except 'faster' itself
        return ds_to_annual_emissions_total_faster(
            gridded_data=gridded_data,
            var_name=var_name,
            cell_area=cell_area,
            keep_sectors=keep_sectors,
            sum_dims=sum_dims
        )

    da = gridded_data[var_name]

    # obtain the seconds in each month for which data is available
    seconds_per_month = da.time.dt.days_in_month * 24 * 60 * 60

    # kg/m2/s --> kg/m2/month
    monthly = seconds_per_month * da

    # weight with cell area: kg/m2/month --> kg/cell/month
    area_weighted = cell_area * monthly

    # Sum over spatial dimensions: kg/cell/month --> kg/month (global)
    # sum_dims = ["lat", "lon"]
    sum_dims_to_use = sum_dims.copy() if sum_dims else []
    if "level" in area_weighted.dims:
        sum_dims_to_use.append("level") # altitude: for aircraft emissions datasets
    if sum_dims_to_use:
        kg_per_month = area_weighted.sum(dim=sum_dims_to_use)
    else:
        kg_per_month = area_weighted

    # Convert to annual totals kg/month (global) --> kg/year (global)
    kg_per_year = kg_per_month.groupby("time.year").sum()

    # Convert to Mt/year
    da_Mt_y = kg_per_year * 1e-9

    if "sector" in da_Mt_y.dims and not keep_sectors:
        da_Mt_y = da_Mt_y.sum(dim="sector")

    # make sure variable is correctly named
    da_Mt_y = da_Mt_y.rename(var_name)
    
    return da_Mt_y

# Plot 1 map
# - assumes you have a dataarray, not a dataset
def plot_map(
    da: xr.DataArray,
    title: str | None = None,
    robust: bool = True,
    add_colorbar: bool | None = None,
    borders: bool = False,
    coastlines: bool = True,
    save_as: str | None = None,
    filename: str | None = None,
    cmap: str = "GnBu",
    **kwargs,
):
    fig, axis = plt.subplots(
        1, 1, subplot_kw=dict(projection=ccrs.Robinson()), figsize=(12, 6)
    )
    axis.set_global()
    # axis.stock_img()
    if coastlines:
        axis.coastlines()
    if borders:
        axis.add_feature(cfeature.BORDERS)

    cbar_args = dict(add_colorbar=add_colorbar)
    if add_colorbar is not False:
        cbar_args["cbar_kwargs"] = {"orientation": "horizontal", "shrink": 0.65}

    da.plot(
        ax=axis,
        robust=robust,
        transform=ccrs.PlateCarree(),  # this is important!
        cmap=cmap,
        **cbar_args,
        **kwargs,
    )
    if title is not None:
        axis.set_title(title)
    
    # Save the figure if requested
    if save_as and filename:
        if save_as == "pdf":
            plt.savefig(f"{filename}.pdf", bbox_inches='tight', dpi=300)
        elif save_as == "png":
            plt.savefig(f"{filename}.png", bbox_inches='tight', dpi=300)
        elif save_as == "pdf-png":
            plt.savefig(f"{filename}.pdf", bbox_inches='tight', dpi=300)
            plt.savefig(f"{filename}.png", bbox_inches='tight', dpi=300)

# Plot multiple maps in a grid
# - assumes you have a dataarray, not a dataset
def plot_maps(ds, sectors, variable, ncols=3, year=2100, month=1, proj=ccrs.Robinson(), save_as: str | None = None, filename: str | None = None): 

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
        )
        da[variable]

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
    
    # Save the figure if requested
    if save_as and filename:
        if save_as == "pdf":
            plt.savefig(f"{filename}.pdf", bbox_inches='tight', dpi=300)
        elif save_as == "png":
            plt.savefig(f"{filename}.png", bbox_inches='tight', dpi=300)
        elif save_as == "pdf-png":
            plt.savefig(f"{filename}.pdf", bbox_inches='tight', dpi=300)
            plt.savefig(f"{filename}.png", bbox_inches='tight', dpi=300)
    
    plt.show()


# Plot seasonal maps in a grid (DJF, MAM, JJA, SON)
# - assumes you have a dataset with month coordinate [1-12]
def plot_maps_seasonal(ds, sectors, variable, ncols=2, year=2100, proj=ccrs.Robinson(), 
                       save_as: str | None = None, filename: str | None = None,
                        title: str | None = None,
                        perc_vmax: float = 0.98,
                        # Define seasonal month groupings
                        seasons = {
                            'DJF': [12, 1, 2],  # Winter: Dec, Jan, Feb
                            'MAM': [3, 4, 5],   # Spring: Mar, Apr, May
                            'JJA': [6, 7, 8],   # Summer: Jun, Jul, Aug
                            'SON': [9, 10, 11]  # Autumn: Sep, Oct, Nov
                        }
    ):
    """
    Plot seasonal averages for multiple sectors in a grid.
    
    Parameters:
    -----------
    ds : xarray.Dataset
        Dataset with month coordinate containing values 1-12
    sectors : list or None
        List of sector names to plot. If None, plots seasons only (no sector dimension)
    variable : str
        Variable name to plot from the dataset
    ncols : int, default=2
        Number of columns in the subplot grid
    year : int, default=2100
        Year to select for plotting
    proj : cartopy projection, default=ccrs.Robinson()
        Map projection to use
    title : str, optional
        Overall title for the entire figure
    save_as : str, optional
        Format to save figure ('pdf', 'png', or 'pdf-png')
    filename : str, optional
        Base filename for saving (without extension)
    """
    
    season_names = list(seasons.keys())
    n_seasons = len(season_names)
    
    # First, calculate vmin and vmax across all data to ensure consistent color scales
    if sectors is None:
        # Calculate range for all seasons
        year_data = ds.sel(year=year)
        all_seasonal_data = []
        for season_name, months in seasons.items():
            seasonal_data = year_data.sel(month=months).mean(dim='month').squeeze()
            all_seasonal_data.append(seasonal_data[variable])
        combined_data = xr.concat(all_seasonal_data, dim='temp')
        vmin, vmax = float(combined_data.min()), float(combined_data.quantile(perc_vmax))
    else:
        # Calculate range for all sectors and seasons
        all_data = []
        for sector in sectors:
            sector_data = ds.sel(sector=sector, year=year)
            for season_name, months in seasons.items():
                seasonal_data = sector_data.sel(month=months).mean(dim='month').squeeze()
                all_data.append(seasonal_data[variable])
        combined_data = xr.concat(all_data, dim='temp')
        vmin, vmax = float(combined_data.min()), float(combined_data.quantile(perc_vmax))
    
    if sectors is None:
        # Just plot seasons, no sector dimension
        nrows = int(np.ceil(n_seasons / ncols))
        
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
        
        for i, (season_name, months) in enumerate(seasons.items()):
            if i >= len(axes):
                break
                
            # Select data for this year
            year_data = ds.sel(year=year)
            
            # Calculate seasonal average
            seasonal_data = year_data.sel(month=months).mean(dim='month').squeeze()
            
            # Get the data array for the specified variable
            da = seasonal_data[variable]
            
            # Plot directly with xarray's .plot.pcolormesh
            da.plot.pcolormesh(
                ax=axes[i],
                transform=ccrs.PlateCarree(),
                cmap="GnBu",
                robust=True,
                vmin=vmin,
                vmax=vmax,
                cbar_kwargs={"orientation": "horizontal", "shrink": 0.65},
            )
            
            axes[i].set_title(season_name)
            axes[i].coastlines()
        
        # Remove any unused axes
        for j in range(n_seasons, len(axes)):
            fig.delaxes(axes[j])
        
        plot_idx = n_seasons  # Set plot_idx for the common cleanup code
            
    else:
        # Plot sectors x seasons
        n_sectors = len(sectors)
        
        # Calculate grid dimensions
        nrows_sectors = int(np.ceil(n_sectors / ncols))
        total_rows = nrows_sectors * n_seasons
        
        fig, axes = plt.subplots(
            nrows=total_rows,
            ncols=ncols,
            figsize=(6 * ncols, 4.5 * total_rows),
            subplot_kw={"projection": proj}
        )
        
        # Flatten axes safely
        if isinstance(axes, np.ndarray):
            axes = axes.flatten()
        else:
            axes = [axes]
        
        plot_idx = 0
        
        for season_name, months in seasons.items():
            for i, sector in enumerate(sectors):
                if plot_idx >= len(axes):
                    break
                    
                # Select data for this sector and year
                sector_data = ds.sel(sector=sector, year=year)
                
                # Calculate seasonal average
                seasonal_data = sector_data.sel(month=months).mean(dim='month').squeeze()
                
                # Get the data array for the specified variable
                da = seasonal_data[variable]
                
                # Plot directly with xarray's .plot.pcolormesh
                da.plot.pcolormesh(
                    ax=axes[plot_idx],
                    transform=ccrs.PlateCarree(),
                    cmap="GnBu",
                    robust=True,
                    vmin=vmin,
                    vmax=vmax,
                    cbar_kwargs={"orientation": "horizontal", "shrink": 0.65},
                )
                
                axes[plot_idx].set_title(f"{sector} - {season_name}")
                axes[plot_idx].coastlines()
                
                plot_idx += 1
        
        # Remove any unused axes
        for j in range(plot_idx, len(axes)):
            fig.delaxes(axes[j])
    
    # Add overall title if provided
    if title is not None:
        fig.suptitle(title, fontsize=16, y=0.98)
    
    plt.tight_layout()
    
    # Save the figure if requested
    if save_as and filename:
        if save_as == "pdf":
            plt.savefig(f"{filename}.pdf", bbox_inches='tight', dpi=300)
        elif save_as == "png":
            plt.savefig(f"{filename}.png", bbox_inches='tight', dpi=300)
        elif save_as == "pdf-png" or save_as == "png-pdf":
            plt.savefig(f"{filename}.pdf", bbox_inches='tight', dpi=300)
            plt.savefig(f"{filename}.png", bbox_inches='tight', dpi=300)
    
    plt.show()


# %%
# Timeseries Plots for checking against historical data



# Timeseries Plots 
def plot_place_timeseries(ceds_ds, scen_ds,
                          lat=39.9042, lon=116.4074,
                          place='Beijing',
                          gas='CO', sector=1, sector_name='Energy',
                          type="em_anthro"):
    """
    Plot timeseries for PLACE (e.g. Beijing) gridpoint comparing CEDS and scenario data
    
    Parameters:
    - ceds_ds: CEDS dataset
    - scen_ds: Scenario dataset  
    - gas: Gas species (e.g., 'CO', 'CO2')
    - sector: Sector name
    """
    
    # Select closest gridpoint to PLACE for both datasets
    ceds_place = ceds_ds.sel(
        lat=lat, 
        lon=lon, 
        method="nearest"
    ).sel(sector=sector)
    
    scen_place = scen_ds.sel(
        lat=lat,
        lon=lon, 
        method="nearest"
    ).sel(sector=sector)
    
    # Get the variable name
    var_name = f'{gas}_{type}'
    
    # Extract the timeseries data
    ceds_ts = ceds_place[var_name]
    scen_ts = scen_place[var_name]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot both timeseries
    ceds_ts.plot(ax=ax, label='CEDS Historical', marker='o', linewidth=2)
    scen_ts.plot(ax=ax, label='CMIP7 Scenario', marker='s', linewidth=2)
    
    # Get actual coordinates of selected gridpoint
    actual_lat = float(ceds_place.lat.values)
    actual_lon = float(ceds_place.lon.values)
    
    # Formatting
    ax.set_title(f'{gas} Emissions - {place} gridpoint\n'
                f'Sector: {sector_name}\n'
                f'Gridpoint: {actual_lat:.2f}°N, {actual_lon:.2f}°E', 
                fontsize=14)
    ax.set_xlabel('Time')
    ax.set_ylabel(f'{gas} emissions (kg/m²/s)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Print some info
    print(f"{place} coordinates: {lat}°N, {lon}°E")
    print(f"Selected gridpoint: {actual_lat:.2f}°N, {actual_lon:.2f}°E")
    print(f"Distance: ~{np.sqrt((actual_lat-lat)**2 + (actual_lon-lon)**2)*111:.1f} km")
    
    return fig, ax

# Alternative: Select multiple nearby gridpoints and average them
def plot_place_area_average_timeseries(ceds_ds, scen_ds, gas='CO', sector=1, sector_name='Energy',
                                       lat=39.9042, lon=116.4074,
                                       place='Beijing',
                                       lat_range=1.0, lon_range=1.0,
                                       type="em_anthro"):
    """
    Plot timeseries for PLACE (e.g., Beijing) area (average of nearby gridpoints)
    
    Parameters:
    - lat_range, lon_range: degrees around PLACE to include in average
    """
    
    # Define bounding box around PLACE
    lat_min = lat - lat_range/2
    lat_max = lat + lat_range/2
    lon_min = lon - lon_range/2  
    lon_max = lon + lon_range/2
    
    # Select area around PLACE
    ceds_area = ceds_ds.sel(
        lat=slice(lat_min, lat_max),
        lon=slice(lon_min, lon_max),
        sector=sector
    )
    
    scen_area = scen_ds.sel(
        lat=slice(lat_min, lat_max),
        lon=slice(lon_min, lon_max), 
        sector=sector
    )
    
    # Get variable name
    var_name = f'{gas}_{type}'
    
    # Average over the spatial area
    ceds_ts = ceds_area[var_name].mean(dim=['lat', 'lon'])
    scen_ts = scen_area[var_name].mean(dim=['lat', 'lon'])
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot both timeseries
    ceds_ts.plot(ax=ax, label='CEDS Historical', marker='o', linewidth=2)
    scen_ts.plot(ax=ax, label='CMIP7 Scenario', marker='s', linewidth=2)
    
    # Formatting
    ax.set_title(f'{gas} Emissions - {place} Area Average\n'
                f'Sector: {sector_name}\n'
                f'Area: {lat_range}° × {lon_range}° around {place}',
                fontsize=14)
    ax.set_xlabel('Time')
    ax.set_ylabel(f'{gas} emissions (kg/m²/s)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Print some info
    n_gridpoints = len(ceds_area.lat) * len(ceds_area.lon)
    print(f"{place} area: {lat_range}° × {lon_range}°")
    print(f"Number of gridpoints averaged: {n_gridpoints}")
    
    return fig, ax

# %%
