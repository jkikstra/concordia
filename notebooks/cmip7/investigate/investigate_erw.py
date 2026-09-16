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
from concordia.cmip7.CONSTANTS import CONFIG

# %%
def get_settings(base_path: Path, 
                 file = CONFIG):
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
# other cdr data
cdr_proxy_file = settings.proxy_path / "CDR_CO2.nc"

cdr = xr.open_dataset(cdr_proxy_file)

# %%
ew_proxy_path = settings.gridding_path / "iiasa" / "cdr" / "pratama_joshi" / "EW_GLB"

versions = ["GeoTIFF", "netCDF"]
versions = ["GeoTIFF"]

for v in versions:
    print(f"Looking at the {v} version of the EW file.")
    if v=="GeoTIFF":
        # Open the GeoTIFF file
        ew_proxy_file = ew_proxy_path / "EW_GLB_tco2_yr_km2.tif"
        ew_raw = xr.open_dataset(ew_proxy_file, engine='rasterio')
        
        # Convert to CF-compliant CMORized netCDF structure
        print("Converting GeoTIFF to CF-compliant netCDF structure...")
        
        # Extract coordinate information
        x_coords = ew_raw.x.values
        y_coords = ew_raw.y.values
        data_values = ew_raw.band_data.isel(band=0).values
        
        # Convert x,y to proper lon,lat if needed
        # Assuming the GeoTIFF is already in geographic coordinates (WGS84)
        lon_coords = x_coords
        lat_coords = y_coords
        
        print(f"Original GeoTIFF shape: {data_values.shape}")
        print(f"Lon range: {lon_coords.min():.2f} to {lon_coords.max():.2f}")
        print(f"Lat range: {lat_coords.min():.2f} to {lat_coords.max():.2f}")
        
        # Create CF-compliant dataset
        ew = xr.Dataset(
            data_vars={
                'tco2_yr_km2': (
                    ['lat', 'lon'],
                    data_values,
                    {
                        'standard_name': 'carbon_dioxide_enhanced_weathering_potential',
                        'long_name': 'Enhanced Weathering Carbon Dioxide Removal Potential',
                        'units': 'tCO2 yr-1 km-2',
                        'cell_methods': 'area: mean',
                        # 'grid_mapping': 'crs'
                    }
                ),
                # 'crs': (
                #     [],
                #     0,
                #     {
                #         'grid_mapping_name': 'latitude_longitude',
                #         'longitude_of_prime_meridian': 0.0,
                #         'semi_major_axis': 6378137.0,
                #         'inverse_flattening': 298.257223563,
                #         'crs_wkt': 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]]'
                #     }
                # )
            },
            coords={
                'lat': (
                    ['lat'],
                    lat_coords,
                    {
                        'standard_name': 'latitude',
                        'long_name': 'Latitude',
                        'units': 'degrees_north',
                        'axis': 'Y'
                    }
                ),
                'lon': (
                    ['lon'],
                    lon_coords,
                    {
                        'standard_name': 'longitude',
                        'long_name': 'Longitude', 
                        'units': 'degrees_east',
                        'axis': 'X'
                    }
                )
            },
            attrs={
                'title': 'Enhanced Weathering Carbon Dioxide Removal Potential',
                'institution': 'IIASA',
                'source': 'Pratama & Joshi Enhanced Weathering Model',
                'references': 'Pratama and Joshi (2025), Enhanced Weathering CDR potential assessment',
                'comment': 'Global enhanced weathering potential for carbon dioxide removal',
                'Conventions': 'CF-1.8',
                'creation_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S UTC'),
                'creator_name': 'Concordia Processing Pipeline',
                'geospatial_lat_min': float(lat_coords.min()),
                'geospatial_lat_max': float(lat_coords.max()),
                'geospatial_lon_min': float(lon_coords.min()),
                'geospatial_lon_max': float(lon_coords.max()),
                'geospatial_lat_units': 'degrees_north',
                'geospatial_lon_units': 'degrees_east',
                'spatial_resolution': f'{abs(lon_coords[1] - lon_coords[0]):.6f} degrees'
            }
        )
        
        # Save as CF-compliant netCDF file
        output_file = ew_proxy_path / "EW_GLB_tco2_yr_km2_CF_compliant.nc"
        print(f"Saving CF-compliant netCDF to: {output_file}")
        
        # Set encoding for better compression and compliance
        encoding = {
            'tco2_yr_km2': {
                'dtype': 'float32',
                'zlib': True,
                'complevel': 6,
                'shuffle': True,
                '_FillValue': np.nan
            },
            'lat': {'dtype': 'float64'},
            'lon': {'dtype': 'float64'},
            # 'crs': {'dtype': 'int32'}
        }
        
        ew.to_netcdf(output_file, encoding=encoding, format='NETCDF4')
        print("CF-compliant netCDF file saved successfully!")
        
        # # Update variable name for consistency with netCDF version
        # ew = ew.rename({'tco2_yr_km2': 'tco2_yr_km'})
    elif v=="netCDF":
        nc_message = "netCDF file seems to have swapped 'data variables' and 'coordinates'"
        print(nc_message)
        # # Open the netCDF file
        # ew_proxy_file = ew_proxy_path / "EW_GLB_tco2_yr_km2.nc"
        # ew = xr.open_dataset(ew_proxy_file)

        # # Fix the incorrectly structured netCDF file
        # # tco2_yr_km should be data variable, lon/lat should be coordinates
        # if 'tco2_yr_km' in ew.dims and 'lon' in ew.data_vars and 'lat' in ew.data_vars:
        #     print("Fixing incorrectly structured netCDF file...")
            
        #     # Extract the point data
        #     lon_points = ew.lon.values
        #     lat_points = ew.lat.values
        #     data_points = ew.tco2_yr_km.values
            
        #     print(f"Original data: {len(data_points)} points")
            
        #     # Create a regular grid similar to CDR dataset
        #     # Use similar resolution to CDR (0.5 degree grid)
        #     lon_grid = np.arange(-179.75, 180, 0.5)  # 720 points
        #     lat_grid = np.arange(-89.75, 90, 0.5)    # 360 points
            
        #     print(f"Target grid: {len(lat_grid)} x {len(lon_grid)} ({len(lat_grid) * len(lon_grid)} points)")
            
        #     # Create meshgrid for interpolation
        #     lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
            
        #     # Interpolate using scipy griddata
        #     from scipy.interpolate import griddata
            
        #     print("Interpolating to regular grid...")
        #     data_grid = griddata(
        #         points=np.column_stack((lon_points, lat_points)),
        #         values=data_points,
        #         xi=(lon_mesh, lat_mesh),
        #         method='linear',
        #         fill_value=np.nan
        #     )
            
        #     # Create proper gridded dataset with PandasIndexes like CDR
        #     ew = xr.Dataset(
        #         data_vars={'tco2_yr_km': (['lat', 'lon'], data_grid)},
        #         coords={
        #             'lat': lat_grid,
        #             'lon': lon_grid
        #         }
        #     )
            
        #     print("Fixed dataset structure with interpolation to regular grid")
        #     print(f"Grid shape: {ew.tco2_yr_km.shape}")
            
        #     # Count valid (non-NaN) data points after interpolation
        #     valid_points = np.sum(~np.isnan(ew.tco2_yr_km.values))
        #     total_points = ew.tco2_yr_km.size
        #     print(f"Valid data points after interpolation: {valid_points} / {total_points} ({valid_points/total_points*100:.1f}%)")

    # Create a plot
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={"projection": ccrs.PlateCarree()})

    # Plot the data based on file format
    if v=="GeoTIFF":
        # GeoTIFF now converted to CF-compliant structure
        im = ew.tco2_yr_km2.plot(
            ax=ax,
            transform=ccrs.PlateCarree(),
            cmap="viridis",
            add_colorbar=False
        )
    elif v=="netCDF":
        print(nc_message)
        # # NetCDF data is now properly gridded after interpolation
        # im = ew.tco2_yr_km.plot(
        #     ax=ax,
        #     transform=ccrs.PlateCarree(),
        #     cmap="viridis",
        #     add_colorbar=False
        # )

    # Add map features
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linewidth=0.3)
    ax.add_feature(cfeature.LAND, alpha=0.3)
    ax.add_feature(cfeature.OCEAN, alpha=0.3)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, orientation="horizontal", fraction=0.05, pad=0.05)
    cbar.set_label("Enhanced Weathering Potential (tCO2/yr/km²)")

    # Set title
    ax.set_title(f"Enhanced Weathering CDR Potential ({v})")

    plt.tight_layout()
    plt.show()

    # Print some basic info about the dataset
    print("Dataset info:")
    print(ew)
    
    if v=="GeoTIFF":
        print("\nData shape:", ew.tco2_yr_km2.shape)
        print("Data range:", float(ew.tco2_yr_km2.min()), "to", float(ew.tco2_yr_km2.max()))
        print("Lat range:", float(ew.lat.min()), "to", float(ew.lat.max()))
        print("Lon range:", float(ew.lon.min()), "to", float(ew.lon.max()))
    elif v=="netCDF":
        print(nc_message)
        # # Now properly gridded data
        # print("\nGrid shape:", ew.tco2_yr_km.shape)
        # print("Data range:", float(ew.tco2_yr_km.min()), "to", float(ew.tco2_yr_km.max()))
        # print("Lat range:", float(ew.lat.min()), "to", float(ew.lat.max()))
        # print("Lon range:", float(ew.lon.min()), "to", float(ew.lon.max()))
        # # Count valid (non-NaN) data points
        # valid_points = np.sum(~np.isnan(ew.tco2_yr_km.values))
        # total_points = ew.tco2_yr_km.size
        # print(f"Valid data points: {valid_points} / {total_points} ({valid_points/total_points*100:.1f}%)")



# %%
# run fixed netcdf
ew_fixed_nc = xr.open_dataset(output_file)

ew_fixed_nc.tco2_yr_km2.plot()

ew_fixed_nc

# %% 
cdr

# %%
# Create 3x4 facet plot for {BECCS, ...} CDR emissions by month in 2023

for cdr_prox in cdr.sector.values:

    print(f"Creating 3x4 facet plot for {cdr_prox} in 2023...")

    # Select CDR proxy sector, year 2023, and CO2 gas
    p = cdr.sel(sector=cdr_prox, year=2023, gas='CO2')

    # Create figure with subplots using cartopy projection
    fig, axes = plt.subplots(3, 4, figsize=(16, 12), 
                            subplot_kw={'projection': ccrs.PlateCarree()})

    # Flatten axes for easier iteration
    axes_flat = axes.flatten()

    # Get data range for consistent colorscale
    vmin = float(p.emissions.min())
    vmax = float(p.emissions.max())

    # Month names for titles
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    # Plot each month
    for i, month in enumerate(range(1, 13)):
        ax = axes_flat[i]
        
        # Select data for this month
        month_data = p.emissions.sel(month=month)
        
        # Create the plot
        im = month_data.plot(
            ax=ax,
            transform=ccrs.PlateCarree(),
            cmap='RdBu_r',
            vmin=vmin,
            vmax=vmax,
            add_colorbar=False  # We'll add a single colorbar later
        )
        
        # Add map features
        ax.coastlines(linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.7)
        ax.add_feature(cfeature.LAND, alpha=0.2)
        ax.add_feature(cfeature.OCEAN, alpha=0.2)
        ax.set_global()
        
        # Set title for each subplot
        ax.set_title(f'{month_names[i]}', fontsize=12, fontweight='bold')
        
        # Remove individual axis labels to reduce clutter
        ax.set_xlabel('')
        ax.set_ylabel('')

    # Add a single colorbar for all subplots
    # Position it below the subplots
    cbar_ax = fig.add_axes([0.15, 0.02, 0.7, 0.03])  # [left, bottom, width, height]
    cbar = plt.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar.set_label(f'{cdr_prox}', fontsize=12, fontweight='bold')

    # Adjust the main title
    fig.suptitle(f'{cdr_prox} by Month - 2023',
                fontsize=16, fontweight='bold', y=0.95)

    # Adjust spacing
    plt.subplots_adjust(top=0.92, bottom=0.1, left=0.05, right=0.95,
                    hspace=0.25, wspace=0.1)

    plt.show()

    # Print some statistics
    print(f"\n{cdr_prox} Statistics for 2023:")
    print(f"Data shape: {p.emissions.shape}")
    print(f"Annual total: {float(p.emissions.sum()):.2e} tCO2/year")
    print(f"Monthly range: {float(p.emissions.min()):.2e} to {float(p.emissions.max()):.2e} tCO2/month")

    # Show monthly totals
    monthly_totals = p.emissions.sum(dim=['lat', 'lon'])
    print("\nMonthly global totals (tCO2):")
    for month, total in zip(monthly_totals.month.values, monthly_totals.values):
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        print(f"  {month_names[month-1]}: {total:.2e}")

# %%