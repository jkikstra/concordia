# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Proxy raster generation for post-2100 extensions
#
# Generates spatial proxy rasters covering 2100–2500 (every 5 years) for use in
# concordia gridding of extended CMIP7 ScenarioMIP emissions trajectories.
#
# Two proxy strategies are produced:
#
# **Scenario-based** (`proxy_rasters_extensions/<marker>/`) — spatial patterns taken
# directly from the 2100 slice of the scenario's own gridded output files
# (`em-anthro`, `em-AIR-anthro`). Produced for anthropogenic sectors, shipping,
# and aircraft. Use these when the spatial pattern should reflect the scenario's
# own end-of-century distribution.
#
# **Proxy-based** (`proxy_rasters_extensions/`) — spatial patterns taken from the
# existing static proxy rasters at year 2100 and held constant. Produced for
# anthro, shipping, aircraft, openburning, NMVOC speciation, VOC speciation,
# CDR/EW CO2, and H2. Use these when no scenario-specific gridded output exists.
#
# Both strategies freeze the 2100 spatial pattern and repeat it across all
# extension years — the assumption being that spatial distributions do not change
# meaningfully beyond 2100. The resulting files are NetCDF4 with zlib compression.
#
# Key settings: `marker_to_run` (scenario marker), `VERSION_ESGF`, `SETTINGS_FILE`.

# %% [markdown]
# ## Imports

# %%
import xarray as xr
from pathlib import Path
import numpy as np
from dask.diagnostics import ProgressBar
from dask.utils import SerializableLock


# %% [markdown]
# ## Configuration

# %%
lock = SerializableLock()

# %%
SETTINGS_FILE: str = "config_cmip7_v0-4-0" # for second ESGF version
VERSION_ESGF: str = "1-1-1" # for second ESGF version

# Which scenario to run from the markers
marker_to_run: str = "vl" # options: h, hl, m, ml, l, ln, vl

GRIDDING_VERSION: str | None = f"{marker_to_run}_{VERSION_ESGF}"

# %% [markdown]
# ## Paths and output directories

# %%
# grid_file_location = "/Users/hoegner/Projects/CMIP7/input/gridding/"
# grid_file_location = "C:/Users/kikstra/IIASA/ECE.prog - Documents/Projects/CMIP7/IAM Data Processing/concordia_cmip7_esgf_v0_alpha/input/gridding/"
grid_file_location = "/Users/jarmo/Library/CloudStorage/OneDrive-SharedLibraries-IIASA/ECE.prog - Documents/Projects/CMIP7/IAM Data Processing/concordia_cmip7_v0-4-0/input/gridding/"

scenario_data_location = Path( "../../results", GRIDDING_VERSION)

proxies_location = Path(grid_file_location, "proxy_rasters")

new_proxies_location = Path(grid_file_location, "proxy_rasters_extensions")
new_proxies_location.mkdir(parents=True, exist_ok=True)

new_proxies_scenario_based = Path(grid_file_location, "proxy_rasters_extensions", marker_to_run)
new_proxies_scenario_based.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Sector mapping

# %%
sector_mapping = {0.0: 'AGR',
                  1.0: 'ENE',
                  2.0: 'IND',
                  3.0: 'TRA',
                  4.0: 'RCO',
                  5.0: 'SLV',
                  6.0: 'WST'}
# Note: shipping sector (7.0) is handled separately and renamed to 'SHP'

# %% [markdown]
# # Generate proxy rasters

# %% [markdown]
# ## Scenario-based proxies
#
# Spatial patterns derived from the 2100 slice of the scenario's own gridded output.
# Output: `proxy_rasters_extensions/<marker>/`

# %%
EXT_PROXY_YEARS = np.arange(2100, 2501, 5)

# %% [markdown]
# ### Anthro (sectors AGR / ENE / IND / TRA / RCO / SLV / WST)

# %%
# ANTHRO proxies - Scenario-based version

# Loop through all scenario files
for file in scenario_data_location.glob("*.nc"):

    if "em-anthro" in str(file):
        print(f"Processing {file}")

        # Extract species from filename
        species = file.stem.split("-")[0]

        # Open dataset with fixed numeric chunks (safe for object dtypes)
        ds = xr.open_dataset(
            file,
            engine="netcdf4",
            chunks={},  # adjust for your RAM / dataset
            lock=lock
        )

        # Drop variables we don't need
        drop_vars = ["lat_bnds", "lon_bnds", "time_bnds", "sector_bnds"]
        ds = ds.drop_vars([v for v in drop_vars if v in ds.data_vars])
        
        # Slice only year 2100 first to reduce size
        ds = ds.sel(time=ds["time"].dt.year == 2100)

      #  print(ds)

        # Rename main emissions variable
        var_name = f"{species}_em_anthro"
        if var_name in ds.data_vars:
            ds = ds.rename({var_name: "emissions"})
        else:
            # fallback to first variable
            ds = ds.rename({list(ds.data_vars)[0]: "emissions"})
            
        # Drop sectors >= 7
        ds = ds.sel(sector=ds["sector"] < 7)

        # Drop all attributes
        ds.attrs = {}
        for var in ds.data_vars:
            ds[var].attrs = {}

        # Add gas dimension
        ds = ds.expand_dims({"gas": [species]})
        
        # rename NMVOC to VOC    
        if species == "NMVOC":
            ds = ds.assign_coords(gas=["VOC"])
            species = "VOC"
            
        # rename SO2 to Sulfur
        if species == "SO2":
            ds = ds.assign_coords(gas=["Sulfur"])
            species = "Sulfur"
        
        # Split time into year/month and compute monthly mean
        ds = (
            ds.assign_coords(
                year=("time", ds["time"].dt.year.data),
                month=("time", ds["time"].dt.month.data)
            )
            .groupby(["year", "month"])
            .mean()
        )

        # Project to future years
        ds = ds.sel(year=2100).expand_dims({"year": EXT_PROXY_YEARS})
        
        # Rename sectors using mapping
        ds = ds.assign_coords(sector=ds["sector"].to_series().replace(sector_mapping).values)

        # Transpose to final dimension order and chunk lazily
        ds_reordered = ds.transpose("lat", "lon", "gas", "sector", "year", "month").chunk({"month": 12})

        # Output file
        outfile = new_proxies_scenario_based / f"anthro_{species}.nc"
        # Skip if already processed
        if outfile.exists():
            print(f"Skipping {file} (already exists)")
            continue
            
# or, if we want to override
#            outfile.unlink()


        # Encoding for NetCDF compression
        encoding = {var: {"zlib": True, "complevel": 4, "dtype": "float32"} for var in ds_reordered.data_vars}

        # Write lazily to NetCDF with progress bar
        with ProgressBar():
            ds_reordered.to_netcdf(outfile, mode="w", encoding=encoding, compute=True)

        # Free memory
        del ds, ds_reordered

# %% [markdown]
# ### Shipping (sector SHP)

# %%
# SHIPPING proxies - Scenario-based version

# Loop through all scenario files
for file in scenario_data_location.glob("*.nc"):

    if "em-anthro" in str(file):
        print(f"Processing {file}")

        # Extract species from filename
        species = file.stem.split("-")[0]

        # Open dataset with fixed numeric chunks (safe for object dtypes)
        ds = xr.open_dataset(
            file,
            engine="netcdf4",
            chunks={},  # adjust for your RAM / dataset
            lock=lock
        )

        # Drop variables we don't need
        drop_vars = ["lat_bnds", "lon_bnds", "time_bnds", "sector_bnds"]
        ds = ds.drop_vars([v for v in drop_vars if v in ds.data_vars])
        
        # Slice only year 2100 first to reduce size
        ds = ds.sel(time=ds["time"].dt.year == 2100)

        # Rename main emissions variable
        var_name = f"{species}_em_anthro"
        if var_name in ds.data_vars:
            ds = ds.rename({var_name: "emissions"})
        else:
            # fallback to first variable
            ds = ds.rename({list(ds.data_vars)[0]: "emissions"})
            
        # select shipping sector
        ds = ds.sel(sector=ds["sector"]==7)

        # Drop all attributes
        ds.attrs = {}
        for var in ds.data_vars:
            ds[var].attrs = {}

        # Add gas dimension
        ds = ds.expand_dims({"gas": [species]})

        # rename NMVOC to VOC    
        if species == "NMVOC":
            ds = ds.assign_coords(gas=["VOC"])
            species = "VOC"
            
        # rename SO2 to Sulfur
        if species == "SO2":
            ds = ds.assign_coords(gas=["Sulfur"])
            species = "Sulfur"
            
        # Split time into year/month and compute monthly mean
        ds = (
            ds.assign_coords(
                year=("time", ds["time"].dt.year.data),
                month=("time", ds["time"].dt.month.data)
            )
            .groupby(["year", "month"])
            .mean()
        )

        # Project to future years
        ds = ds.sel(year=2100).expand_dims({"year": EXT_PROXY_YEARS})
        
        # rename shipping sector and reorder dimensions
        ds = ds.assign_coords(sector=("sector", ["SHP" if v == 7 else v for v in ds["sector"].values]))    

        # Transpose to final dimension order and chunk lazily
        ds_reordered = ds.transpose("lat", "lon", "gas", "sector", "year", "month").chunk({"month": 12})

        # Output file
        outfile = new_proxies_scenario_based / f"shipping_{species}.nc"

        # Skip if already processed
        if outfile.exists():
            print(f"Skipping {file} (already exists)")
            continue
            
# or, if we want to override
#            outfile.unlink()

        # Encoding for NetCDF compression
        encoding = {var: {"zlib": True, "complevel": 4, "dtype": "float32"} for var in ds_reordered.data_vars}

        # Write lazily to NetCDF with progress bar
        with ProgressBar():
            ds_reordered.to_netcdf(outfile, mode="w", encoding=encoding, compute=True)

        # Free memory
        del ds, ds_reordered

# %% [markdown]
# ### Aircraft

# %%
# AIRCRAFT proxies - Scenario-based version

# loop through all CEDS em-AIR-anthro from input4MIP files
for file in scenario_data_location.glob("*.nc"):
    if "em-AIR-anthro" in str(file):
        # Process the file
        print(f"Processing {file}")
            
        # extract species information from filename
        species = file.stem.split("-")[0]
    
        # import file 
        ds = xr.open_dataset(
            file,
            engine="netcdf4",
            chunks={},
            lock=lock
        )
        
        # Drop variables we don't need
        drop_vars = ["lat_bnds", "lon_bnds", "time_bnds", "level_bnds"]
        ds = ds.drop_vars([v for v in drop_vars if v in ds.data_vars])
        
        # Slice only year 2100 first to reduce size
        ds = ds.sel(time=ds["time"].dt.year == 2100)

        # Rename main emissions variable
        var_name = f"{species}_em_AIR_anthro"
        if var_name in ds.data_vars:
            ds = ds.rename({var_name: "emissions"})
        else:
            # fallback to first variable
            ds = ds.rename({list(ds.data_vars)[0]: "emissions"})

        # Drop all attributes
        ds.attrs = {}
        for var in ds.data_vars:
            ds[var].attrs = {}

        # add gas dimension
        ds = ds.expand_dims(dim={"gas": [f"{species}"]})

        # rename NMVOC to VOC    
        if species == "NMVOC":
            ds = ds.assign_coords(gas=["VOC"])
            species = "VOC"
            
        # rename SO2 to Sulfur
        if species == "SO2":
            ds = ds.assign_coords(gas=["Sulfur"])
            species = "Sulfur"
            
        # Split time into year/month and compute monthly mean
        ds = (
            ds.assign_coords(
                year=("time", ds["time"].dt.year.data),
                month=("time", ds["time"].dt.month.data)
            )
            .groupby(["year", "month"])
            .mean()
        )

        # add sector dimension
        ds = ds.expand_dims(dim={"sector": ["AIR"]})
            
        # select 2100 data and project it onto future years
        ds = ds.sel(year=2100).expand_dims({"year": EXT_PROXY_YEARS})
    
        # Transpose to final dimension order and chunk lazily
        ds_reordered = ds.transpose("lat", "lon", "level", "gas", "sector", "year", "month").chunk({"month": 12})

        outfile = new_proxies_scenario_based / f"aircraft_{species}.nc"
   
        # Skip if already processed
        if outfile.exists():
            print(f"Skipping {file} (already exists)")
            continue
            
# or, if we want to override
#            outfile.unlink()

        encoding = {
            var: {"zlib": True, "complevel": 4, "dtype": "float32"}
            for var in ds_reordered.data_vars
        }
        
        with ProgressBar():
            ds_reordered.to_netcdf(outfile, mode="w", encoding=encoding)

        # Free memory
        del ds, ds_reordered

# %% [markdown]
# ## Proxy-based proxies
#
# Spatial patterns taken from the existing static proxy rasters at year 2100, held
# constant across all extension years. Output: `proxy_rasters_extensions/`

# %% [markdown]
# ### Anthro

# %%
# ANTHRO proxies - proxy-based version

# Loop through all proxy files
for file in proxies_location.glob("*.nc"):
    if "anthro" in str(file):
        print(f"Processing {file}")

        ds = xr.open_dataset(
            file,
            engine="netcdf4",
            chunks={},
            lock=lock
        )
        
        # Project to future years
        ds = ds.sel(year=2100).expand_dims({"year": EXT_PROXY_YEARS})
        
        # Transpose to final dimension order and chunk lazily
        ds_reordered = ds.transpose("lat", "lon", "gas", "sector", "year", "month").chunk({"month": 12})
    
        # Output file
        outfile = new_proxies_location / f"{file.stem + ".nc"}"
        print(outfile)
        # Skip if already processed
        if outfile.exists():
            print(f"Skipping {file} (already exists)")
            continue
            
    # or, if we want to override
    #            outfile.unlink()
    
    
        # Encoding for NetCDF compression
        encoding = {var: {"zlib": True, "complevel": 4, "dtype": "float32"} for var in ds_reordered.data_vars}
    
        # Write lazily to NetCDF with progress bar
        with ProgressBar():
            ds_reordered.to_netcdf(outfile, mode="w", encoding=encoding, compute=True)
    
        # Free memory
        del ds, ds_reordered

# %% [markdown]
# ### Shipping

# %%
# SHIPPING proxies - proxy-based version

# Loop through all proxy files
for file in proxies_location.glob("*.nc"):
    if "shipping" in str(file):
        print(f"Processing {file}")

        ds = xr.open_dataset(
            file,
            engine="netcdf4",
            chunks={},
            lock=lock
        )
        
        # Project to future years
        ds = ds.sel(year=2100).expand_dims({"year": EXT_PROXY_YEARS})
        
        # Transpose to final dimension order and chunk lazily
        ds_reordered = ds.transpose("lat", "lon", "gas", "sector", "year", "month").chunk({"month": 12})
    
        # Output file
        outfile = new_proxies_location / f"{file.stem + ".nc"}"
        print(outfile)
        # Skip if already processed
        if outfile.exists():
            print(f"Skipping {file} (already exists)")
            continue
            
    # or, if we want to override
    #            outfile.unlink()
    
        # Encoding for NetCDF compression
        encoding = {var: {"zlib": True, "complevel": 4, "dtype": "float32"} for var in ds_reordered.data_vars}
    
        # Write lazily to NetCDF with progress bar
        with ProgressBar():
            ds_reordered.to_netcdf(outfile, mode="w", encoding=encoding, compute=True)
    
        # Free memory
        del ds, ds_reordered

# %% [markdown]
# ### Aircraft

# %%
# AIRCRAFT proxies - proxy-based version

# Loop through all proxy files
for file in proxies_location.glob("*.nc"):
    if "aircraft" in str(file):
        print(f"Processing {file}")

        ds = xr.open_dataset(
            file,
            engine="netcdf4",
            chunks={},
            lock=lock
        )
        
        # Project to future years
        ds = ds.sel(year=2100).expand_dims({"year": EXT_PROXY_YEARS})
        
        # Transpose to final dimension order and chunk lazily
        ds_reordered = ds.transpose("lat", "lon", "level", "gas", "sector", "year", "month").chunk({"month": 12})
    
        # Output file
        outfile = new_proxies_location / f"{file.stem + ".nc"}"
        print(outfile)
        # Skip if already processed
        if outfile.exists():
            print(f"Skipping {file} (already exists)")
            continue
            
    # or, if we want to override
    #            outfile.unlink()
    
        # Encoding for NetCDF compression
        encoding = {var: {"zlib": True, "complevel": 4, "dtype": "float32"} for var in ds_reordered.data_vars}
    
        # Write lazily to NetCDF with progress bar
        with ProgressBar():
            ds_reordered.to_netcdf(outfile, mode="w", encoding=encoding, compute=True)
    
        # Free memory
        del ds, ds_reordered

# %% [markdown]
# ## Openburning proxies

# %%
# OPENBURNING proxies

# Loop through all proxy files
for file in proxies_location.glob("*.nc"):
    if "openburning" in str(file):
        print(f"Processing {file}")

        ds = xr.open_dataset(
            file,
            engine="netcdf4",
            chunks={},
            lock=lock
        )
        
        # Project to future years
        ds = ds.sel(year=2100).expand_dims({"year": EXT_PROXY_YEARS})
        
        # Transpose to final dimension order and chunk lazily
        ds_reordered = ds.transpose("lat", "lon", "gas", "sector", "year", "month").chunk({"month": 12})
    
        # Output file
        outfile = new_proxies_location / f"{file.stem + ".nc"}"
        print(outfile)
        # Skip if already processed
        if outfile.exists():
            print(f"Skipping {file} (already exists)")
            continue
            
    # or, if we want to override
    #            outfile.unlink()
    
        # Encoding for NetCDF compression
        encoding = {var: {"zlib": True, "complevel": 4, "dtype": "float32"} for var in ds_reordered.data_vars}
    
        # Write lazily to NetCDF with progress bar
        with ProgressBar():
            ds_reordered.to_netcdf(outfile, mode="w", encoding=encoding, compute=True)
    
        # Free memory
        del ds, ds_reordered

# %% [markdown]
# ## Openburning NMVOC speciation

# %%
nmvoc_proxy_location = Path(grid_file_location + "proxy_rasters" + "/NMVOC_speciation")
nmvoc_proxies_location = Path(grid_file_location, "proxy_rasters_extensions" + "/NMVOC_speciation")
nmvoc_proxies_location.mkdir(parents=True, exist_ok=True)

# %%
# Loop through all scenario files
for file in nmvoc_proxy_location.glob("*.nc"):

    print(f"Processing {file}")
    
    ds = xr.open_dataset(
        file,
        engine="netcdf4",
        chunks={},
        lock=lock
    )
    
    # Project to future years
    ds = ds.sel(year=2100).expand_dims({"year": EXT_PROXY_YEARS})
    
    # Transpose to final dimension order and chunk lazily
    ds_reordered = ds.transpose("lat", "lon", "gas", "sector", "year", "month").chunk({"month": 12})

    # Output file
    outfile = nmvoc_proxies_location / f"{file.stem + ".nc"}"
    print(outfile)
    # Skip if already processed
    if outfile.exists():
        print(f"Skipping {file} (already exists)")
        continue
        
# or, if we want to override
#            outfile.unlink()


    # Encoding for NetCDF compression
    encoding = {var: {"zlib": True, "complevel": 4, "dtype": "float32"} for var in ds_reordered.data_vars}

    # Write lazily to NetCDF with progress bar
    with ProgressBar():
        ds_reordered.to_netcdf(outfile, mode="w", encoding=encoding, compute=True)

    # Free memory
    del ds, ds_reordered

# %% [markdown]
# ## Anthro VOC speciation

# %%
voc_proxy_location = Path(grid_file_location + "proxy_rasters" + "/VOC_speciation")
voc_proxies_location = Path(grid_file_location, "proxy_rasters_extensions" + "/VOC_speciation")
voc_proxies_location.mkdir(parents=True, exist_ok=True)

# %%
# Loop through all scenario files
for file in voc_proxy_location.glob("*.nc"):

    print(f"Processing {file}")

    ds = xr.open_dataset(
        file,
        engine="netcdf4",
        chunks={},
        lock=lock
    )
    
    # Project to future years
    ds = ds.sel(year=2100).expand_dims({"year": EXT_PROXY_YEARS})
    
    # Transpose to final dimension order and chunk lazily
    ds_reordered = ds.transpose("lat", "lon", "gas", "sector", "year", "month").chunk({"month": 12})

    # Output file
    outfile = voc_proxies_location / f"{file.stem + ".nc"}"
    print(outfile)
    # Skip if already processed
    if outfile.exists():
        print(f"Skipping {file} (already exists)")
        continue
        
# or, if we want to override
#            outfile.unlink()


    # Encoding for NetCDF compression
    encoding = {var: {"zlib": True, "complevel": 4, "dtype": "float32"} for var in ds_reordered.data_vars}

    # Write lazily to NetCDF with progress bar
    with ProgressBar():
        ds_reordered.to_netcdf(outfile, mode="w", encoding=encoding, compute=True)

    # Free memory
    del ds, ds_reordered

# %% [markdown]
# ## CDR, EW, and H2

# %% [markdown]
# ### CDR CO2

# %%
# CDR proxy

# Loop through all proxy files
for file in proxies_location.glob("*.nc"):
    if "CDR_CO2" in str(file):
        print(f"Processing {file}")

        ds = xr.open_dataset(
            file,
            engine="netcdf4",
            chunks={},
            lock=lock
        )
        
        # Project to future years
        ds = ds.sel(year=2100).expand_dims({"year": EXT_PROXY_YEARS})
        
        # Transpose to final dimension order and chunk lazily
        ds_reordered = ds.transpose("lat", "lon", "gas", "sector", "year", "month").chunk({"month": 12})
    
        # Output file
        outfile = new_proxies_location / f"{file.stem + ".nc"}"
        print(outfile)
        # Skip if already processed
        if outfile.exists():
            print(f"Skipping {file} (already exists)")
            continue
            
    # or, if we want to override
    #            outfile.unlink()
    
        # Encoding for NetCDF compression
        encoding = {var: {"zlib": True, "complevel": 4, "dtype": "float32"} for var in ds_reordered.data_vars}
    
        # Write lazily to NetCDF with progress bar
        with ProgressBar():
            ds_reordered.to_netcdf(outfile, mode="w", encoding=encoding, compute=True)
    
        # Free memory
        del ds, ds_reordered

# %% [markdown]
# ### EW CO2

# %%
# EW proxy

# Loop through all proxy files
for file in proxies_location.glob("*.nc"):
    if "EW_CO2" in str(file):
        print(f"Processing {file}")

        ds = xr.open_dataset(
            file,
            engine="netcdf4",
            chunks={},
            lock=lock
        )
        
        # Project to future years
        ds = ds.sel(year=2100).expand_dims({"year": EXT_PROXY_YEARS})
        
        # Transpose to final dimension order and chunk lazily
        ds_reordered = ds.transpose("lat", "lon", "gas", "sector", "year", "month").chunk({"month": 12})
    
        # Output file
        outfile = new_proxies_location / f"{file.stem + ".nc"}"
        print(outfile)
        # Skip if already processed
        if outfile.exists():
            print(f"Skipping {file} (already exists)")
            continue
            
    # or, if we want to override
    #            outfile.unlink()
    
        # Encoding for NetCDF compression
        encoding = {var: {"zlib": True, "complevel": 4, "dtype": "float32"} for var in ds_reordered.data_vars}
    
        # Write lazily to NetCDF with progress bar
        with ProgressBar():
            ds_reordered.to_netcdf(outfile, mode="w", encoding=encoding, compute=True)
    
        # Free memory
        del ds, ds_reordered

# %% [markdown]
# ### H2

# %%
# H2 proxy

# Loop through all proxy files
for file in proxies_location.glob("*.nc"):
    if "EF_h2" in str(file):
        print(f"Processing {file}")

        ds = xr.open_dataset(
            file,
            engine="netcdf4",
            chunks={},
            lock=lock
        )
        
        # Project to future years
        ds = ds.sel(year=2100).expand_dims({"year": EXT_PROXY_YEARS})
        
        # Transpose to final dimension order and chunk lazily
        ds_reordered = ds.transpose("lat", "lon", "sector", "year", "month").chunk({"month": 12})
    
        # Output file
        outfile = new_proxies_location / f"{file.stem + ".nc"}"
        print(outfile)
        # Skip if already processed
        if outfile.exists():
            print(f"Skipping {file} (already exists)")
            continue
            
    # or, if we want to override
    #            outfile.unlink()
    
        # Encoding for NetCDF compression
        encoding = {var: {"zlib": True, "complevel": 4, "dtype": "float32"} for var in ds_reordered.data_vars}
    
        # Write lazily to NetCDF with progress bar
        with ProgressBar():
            ds_reordered.to_netcdf(outfile, mode="w", encoding=encoding, compute=True)
    
        # Free memory
        del ds, ds_reordered

# %%
