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
settings = Settings.from_config(
    "config_cmip7_v0-4-0.yaml", 
    base_path=Path(Path(__file__).parent).resolve(),
    version=None)
settings.gridding_path, settings.gridding_path.exists()

# %%
# other cdr data (as template; if necessary - first run `prep_proxyfuture-cdr-from-rescue.py` )
cdr_proxy_file = settings.proxy_path / "CDR_CO2.nc"
cdr_template_beccs = xr.open_dataset(cdr_proxy_file).sel(sector="BECCS")

# %%
ew_proxy_path = settings.gridding_path / "iiasa" / "cdr" / "pratama_joshi" / "EW_GLB"

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
            }
        ),
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

# Transform ew dataset to match cdr_template_beccs format
print("Transforming ew dataset to match cdr_template_beccs format...")

# Define the required dimensions
gas_values = ["CO2"]
month_values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
year_values = np.array([2022, 2023, 2024, 2025, 2030, 2035, 2040, 2045, 2050, 2055, 2060, 2065, 2070, 2075, 2080, 2085, 2090, 2095, 2100])
sector_values = ["EW"]

# Get the original data
original_data = ew.tco2_yr_km2.values
lat_coords = ew.lat.values
lon_coords = ew.lon.values

# Create expanded data array with new dimensions: (gas, sector, year, month, lat, lon)
# For EW, we assume the potential is constant across all years and months
expanded_shape = (len(gas_values), len(sector_values), len(year_values), len(month_values), len(lat_coords), len(lon_coords))
expanded_data = np.broadcast_to(original_data[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :, :], expanded_shape)

# Create the new dataset with matching structure to cdr_template_beccs
ew_formatted = xr.Dataset(
    data_vars={
        'emissions': (
            ['gas', 'sector', 'year', 'month', 'lat', 'lon'],
            expanded_data,
            {
                'standard_name': 'carbon_dioxide_enhanced_weathering_potential',
                'long_name': 'Enhanced Weathering Carbon Dioxide Removal Potential',
                'units': 'tCO2 yr-1 km-2',
                'cell_methods': 'area: mean',
            }
        ),
    },
    coords={
        'gas': (
            ['gas'],
            gas_values,
            {
                'long_name': 'Greenhouse Gas',
                'description': 'Carbon Dioxide'
            }
        ),
        'sector': (
            ['sector'],
            sector_values,
            {
                'long_name': 'CDR Sector',
                'description': 'Enhanced Weathering'
            }
        ),
        'year': (
            ['year'],
            year_values,
            {
                'long_name': 'Year',
                'units': 'year'
            }
        ),
        'month': (
            ['month'],
            month_values,
            {
                'long_name': 'Month',
                'units': 'month'
            }
        ),
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
        'comment': 'Global enhanced weathering potential for carbon dioxide removal - formatted to match CDR template structure',
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

# Update the ew variable to point to the formatted dataset
ew = ew_formatted
print(f"Formatted ew dataset shape: {ew.emissions.shape}")
print(f"Dimensions: {list(ew.emissions.dims)}")

# Update encoding for the new structure
encoding = {
    'emissions': {
        'dtype': 'float32',
        'zlib': True,
        'complevel': 2,
        'shuffle': True,
        '_FillValue': np.nan
    },
    'lat': {'dtype': 'float64'},
    'lon': {'dtype': 'float64'},
    'year': {'dtype': 'int32'},
    'month': {'dtype': 'int32'},
    'gas': {'dtype': 'S3'},  # String type for gas
    'sector': {'dtype': 'S10'}  # String type for sector
}

ew.to_netcdf(output_file, encoding=encoding, format='NETCDF4') # save in same folder
ew.to_netcdf(settings.proxy_path / "EW_CO2.nc", # save in proxy raster folder
             encoding=encoding, format='NETCDF4')
print("CF-compliant netCDF file saved successfully!")

# %% [markdown]
# Visualise to double-check the outcome:

# # %%
# from concordia.cmip7.utils_plotting import plot_map

# # %%
# plot_map(
#     da=ew.sel(month=1,year=2100).emissions
# )
