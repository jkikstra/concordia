# %%
# imports
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import xarray as xr

# %%
# functions

# From a grid to global anual totals
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

    # weight with cell area: kg/m2/month --> kg/cell/month
    area_weighted = cell_area * monthly

    # Sum over spatial dimensions: kg/cell/month --> kg/month (global)
    sum_dims = ["lat", "lon"]
    if "level" in area_weighted.dims:
        sum_dims.append("level") # altitude: for aircraft emissions datasets

    kg_per_month = area_weighted.sum(dim=sum_dims)

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
    save_as: str | None = None,
    filename: str | None = None,
    **kwargs,
):
    fig, axis = plt.subplots(
        1, 1, subplot_kw=dict(projection=ccrs.Robinson()), figsize=(12, 6)
    )
    axis.set_global()
    # axis.stock_img()
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
        cmap="GnBu",
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
