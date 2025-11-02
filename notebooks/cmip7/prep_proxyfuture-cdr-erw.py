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
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path

# %%
from concordia.settings import Settings

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
cdr_template = xr.open_dataset(cdr_proxy_file)

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

# Check if coordinate arrays are the same
print("Checking coordinate compatibility:")
print(f"ew_raw.x shape: {ew_raw.x.values.shape}")
print(f"cdr_template_beccs.lon shape: {cdr_template_beccs.lon.values.shape}")
print(f"ew_raw.y shape: {ew_raw.y.values.shape}")
print(f"cdr_template_beccs.lat shape: {cdr_template_beccs.lat.values.shape}")

# Check if x coordinates are the same as lon coordinates
x_coords_match = np.allclose(ew_raw.x.values, cdr_template_beccs.lon.values, rtol=1e-10, atol=1e-10)
print(f"X coordinates match longitude: {x_coords_match}")

# Check if y coordinates are the same as lat coordinates  
y_coords_match = np.allclose(ew_raw.y.values, cdr_template_beccs.lat.values, rtol=1e-10, atol=1e-10)
print(f"Y coordinates match latitude: {y_coords_match}")

if not x_coords_match:
    print(f"X coord range: {ew_raw.x.min().values:.6f} to {ew_raw.x.max().values:.6f}")
    print(f"Lon coord range: {cdr_template_beccs.lon.min().values:.6f} to {cdr_template_beccs.lon.max().values:.6f}")
    print(f"Max difference in X vs Lon: {np.max(np.abs(ew_raw.x.values - cdr_template_beccs.lon.values)):.10f}")

if not y_coords_match:
    print(f"Y coord range: {ew_raw.y.min().values:.6f} to {ew_raw.y.max().values:.6f}")
    print(f"Lat coord range: {cdr_template_beccs.lat.min().values:.6f} to {cdr_template_beccs.lat.max().values:.6f}")
    print(f"Max difference in Y vs Lat: {np.max(np.abs(ew_raw.y.values - cdr_template_beccs.lat.values)):.10f}")

# Convert x,y to proper lon,lat if needed
# Assuming the GeoTIFF is already in geographic coordinates (WGS84)
lon_coords = x_coords
lat_coords = y_coords

# Flip latitude axis if it starts from positive (89.75) instead of negative (-89.75)
if lat_coords[0] > lat_coords[-1]:  # Check if lat goes from high to low
    print("Flipping latitude axis to go from -89.75 to 89.75...")
    lat_coords = lat_coords[::-1]  # Reverse latitude coordinates
    data_values = data_values[::-1, :]  # Flip data along latitude axis
    print(f"After flipping - Lat range: {lat_coords.min():.2f} to {lat_coords.max():.2f}")

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
            # lat_coords[0],
            # cdr_template_beccs.lat.values[0],
            cdr_template_beccs.lat.values,
            {
                'standard_name': 'latitude',
                'long_name': 'Latitude',
                'units': 'degrees_north',
                'axis': 'Y'
            }
        ),
        'lon': (
            ['lon'],
            # lon_coords[0],
            # cdr_template_beccs.lon.values[0],
            cdr_template_beccs.lon.values,
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
        'lon': (
            ['lon'],
            lon_coords,
            {
                'standard_name': 'longitude',
                'long_name': 'Longitude',
                'units': 'degrees_east',
                'axis': 'X'
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
        'month': (
            ['month'],
            month_values,
            {
                'long_name': 'Month',
                'units': 'month'
            }
        ),
        'year': (
            ['year'],
            year_values,
            {
                'long_name': 'Year',
                'units': 'year'
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


# %%
# Update encoding for the new structure
encoding = {
    'emissions': {
        'dtype': 'float32',
        'zlib': True,
        'complevel': 2,
        'shuffle': True,
        '_FillValue': np.nan
    },
    # 'lat': {'dtype': 'float64'},
    # 'lon': {'dtype': 'float64'},
    # 'year': {'dtype': 'int32'},
    # 'month': {'dtype': 'int32'},
    # 'gas': {'dtype': 'S3'},  # String type for gas
    # 'sector': {'dtype': 'S10'}  # String type for sector
}

ew.to_netcdf(output_file, encoding=encoding, format='NETCDF4') # save in same folder
ew.to_netcdf(settings.proxy_path / "EW_CO2.nc", # save in proxy raster folder
             encoding=encoding, format='NETCDF4')
print("CF-compliant netCDF file saved successfully!")

# %%
# Comprehensive comparison between ew and cdr_template_beccs
print("=" * 60)
print("COMPREHENSIVE COMPARISON: ew vs cdr_template_beccs")
print("=" * 60)

# 1. Dataset structure comparison
print("\n1. DATASET STRUCTURE:")
print(f"ew dimensions: {dict(ew.dims)}")
print(f"cdr_template_beccs dimensions: {dict(cdr_template_beccs.dims)}")

print(f"\new data variables: {list(ew.data_vars.keys())}")
print(f"cdr_template_beccs data variables: {list(cdr_template_beccs.data_vars.keys())}")

print(f"\new coordinates: {list(ew.coords.keys())}")
print(f"cdr_template_beccs coordinates: {list(cdr_template_beccs.coords.keys())}")

# 2. Coordinate comparison
print("\n2. COORDINATE COMPARISON:")

# Check lon coordinates
if 'lon' in ew.coords and 'lon' in cdr_template_beccs.coords:
    lon_match = np.allclose(ew.lon.values, cdr_template_beccs.lon.values, rtol=1e-10)
    print(f"Longitude coordinates match: {lon_match}")
    if not lon_match:
        print(f"  ew lon range: {ew.lon.min().values:.6f} to {ew.lon.max().values:.6f} (shape: {ew.lon.shape})")
        print(f"  cdr lon range: {cdr_template_beccs.lon.min().values:.6f} to {cdr_template_beccs.lon.max().values:.6f} (shape: {cdr_template_beccs.lon.shape})")
        if ew.lon.shape == cdr_template_beccs.lon.shape:
            print(f"  Max lon difference: {np.max(np.abs(ew.lon.values - cdr_template_beccs.lon.values)):.10f}")

# Check lat coordinates  
if 'lat' in ew.coords and 'lat' in cdr_template_beccs.coords:
    lat_match = np.allclose(ew.lat.values, cdr_template_beccs.lat.values, rtol=1e-10)
    print(f"Latitude coordinates match: {lat_match}")
    if not lat_match:
        print(f"  ew lat range: {ew.lat.min().values:.6f} to {ew.lat.max().values:.6f} (shape: {ew.lat.shape})")
        print(f"  cdr lat range: {cdr_template_beccs.lat.min().values:.6f} to {cdr_template_beccs.lat.max().values:.6f} (shape: {cdr_template_beccs.lat.shape})")
        if ew.lat.shape == cdr_template_beccs.lat.shape:
            print(f"  Max lat difference: {np.max(np.abs(ew.lat.values - cdr_template_beccs.lat.values)):.10f}")

# Check other coordinates
for coord in ['year', 'month', 'gas', 'sector']:
    if coord in ew.coords and coord in cdr_template_beccs.coords:
        try:
            coord_match = np.array_equal(ew[coord].values, cdr_template_beccs[coord].values)
            print(f"{coord.capitalize()} coordinates match: {coord_match}")
            if not coord_match:
                print(f"  ew {coord}: {ew[coord].values}")
                print(f"  cdr {coord}: {cdr_template_beccs[coord].values}")
        except:
            print(f"{coord.capitalize()} coordinates cannot be compared directly")
            print(f"  ew {coord}: {ew[coord].values}")
            print(f"  cdr {coord}: {cdr_template_beccs[coord].values}")

# 3. Data shape and type comparison
print("\n3. DATA COMPARISON:")
if 'emissions' in ew.data_vars and 'emissions' in cdr_template_beccs.data_vars:
    print(f"ew.emissions shape: {ew.emissions.shape}")
    print(f"cdr_template_beccs.emissions shape: {cdr_template_beccs.emissions.shape}")
    print(f"ew.emissions dtype: {ew.emissions.dtype}")
    print(f"cdr_template_beccs.emissions dtype: {cdr_template_beccs.emissions.dtype}")
    
    # Check for NaN values
    ew_nan_count = ew.emissions.isnull().sum().values
    cdr_nan_count = cdr_template_beccs.emissions.isnull().sum().values
    print(f"ew.emissions NaN count: {ew_nan_count}")
    print(f"cdr_template_beccs.emissions NaN count: {cdr_nan_count}")
    
    # Value ranges
    print(f"ew.emissions value range: {ew.emissions.min().values:.6e} to {ew.emissions.max().values:.6e}")
    print(f"cdr_template_beccs.emissions value range: {cdr_template_beccs.emissions.min().values:.6e} to {cdr_template_beccs.emissions.max().values:.6e}")

# 4. Attribute comparison  
print("\n4. ATTRIBUTE COMPARISON:")
ew_attrs = set(ew.attrs.keys())
cdr_attrs = set(cdr_template_beccs.attrs.keys())
common_attrs = ew_attrs & cdr_attrs
ew_only = ew_attrs - cdr_attrs
cdr_only = cdr_attrs - ew_attrs

print(f"Common attributes: {sorted(common_attrs)}")
print(f"ew-only attributes: {sorted(ew_only)}")
print(f"cdr-only attributes: {sorted(cdr_only)}")

# Check differences in common attributes
print("\nAttribute value differences:")
for attr in sorted(common_attrs):
    if ew.attrs[attr] != cdr_template_beccs.attrs[attr]:
        print(f"  {attr}:")
        print(f"    ew: {ew.attrs[attr]}")
        print(f"    cdr: {cdr_template_beccs.attrs[attr]}")

print("\n" + "=" * 60)
print("COMPARISON COMPLETE")
print("=" * 60)

# %% [markdown]
# Visualise to double-check the outcome:

# # # %%
# from concordia.cmip7.utils_plotting import plot_map

# # # %%
# plot_map(
#     da=ew.sel(month=1,year=2100).emissions
# )


# plot_map(
#     da=cdr_template_beccs.sel(month=1,year=2100).emissions
# )