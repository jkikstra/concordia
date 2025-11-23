#!/usr/bin/env python
# %% [markdown]
# # Overview
#
# This creates percentage share proxies for openburning VOC species based on BB4CMIP7


# %% [markdown]
# ## Steps happening in this notebook
# - pre-process speciated data: calculate sectoral emissions from shares
# - re-grid to 0.5°
# - take 10 year average for each month
# - re-aggregate forest sectors
# - calculate speciated shares in relation to the sum of speciated emissions, sectorally resolved
# - adjust to CMIP7 file conventions
# - write out

# %%
# check later if we need all these imports 
import xarray as xr
from pathlib import Path
import os
from dask.utils import SerializableLock

from tqdm import tqdm

import concordia.cmip7.utils_futureproxy_ceds_bb4cmip as uprox
from concordia.cmip7.utils_futureproxy_ceds_bb4cmip import _normalize_time_slice

# %%
from concordia.cmip7.CONSTANTS import GASES_ESGF_BB4CMIP_VOC, CONFIG, PROXY_YEARS

# %%
VERSION = CONFIG

# %%
try:
    # when running the script from a terminal or otherwise
    cmip7_dir = Path(__file__).resolve()
    settings = uprox.get_settings(base_path=cmip7_dir, file = CONFIG)
except (FileNotFoundError, NameError):
    try:
        # when running the script from a terminal or otherwise
        cmip7_dir = Path(__file__).resolve().parent
        settings = uprox.get_settings(base_path=cmip7_dir, file = CONFIG)
    except (FileNotFoundError, NameError):
        # Fallback for interactive/Jupyter mode, where 'file location' does not exist
        cmip7_dir = Path().resolve()  # one up
        settings = uprox.get_settings(base_path=cmip7_dir, file = CONFIG)

# %%
settings.gridding_path

# %%
lock = SerializableLock()

# Workaround for HDF5 on Windows: disable file locking to avoid sporadic read errors
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

# %% [markdown]
# ## Settings

# %%
PROXY_TIME_RANGE = [2014,2023]

# %%
sectors = ["AGRI", "BORF", "DEFO", "PEAT", "SAVA", "TEMF"]
sectors_new = ["AWB", "PEAT", "GRSB", "FRTB"]

# %%
# what data to output
new_proxies_location = settings.proxy_path
# ensure output directory exists
new_proxies_location.mkdir(parents=True, exist_ok=True)

scenario_years = PROXY_YEARS


# %%
# load CEDS example file to get the right grid settings
template_file = (
    settings.gridding_path
    / "example_files" / "GCAM4-SSP4-34-SPA4-V25-Harmonized-DB-Sulfur-em-aircraft-anthro_input4MIPs_emissions_CMIP_IAMC-V1_gn_201501-210012.nc"
)

template = xr.open_dataset(template_file)


# %%
target_lat = template["lat"].values
target_lon = template["lon"].values

# %% [markdown]
# ## Functions

# %%
def get_bb4cmip7_location(variable, kind="speciated", data_type="total"):
    """
    Return the path to a BB4CMIP7 VOC dataset.

    Parameters
    ----------
    variable : str
        Species name (e.g. 'NMVOC', 'C2H2', etc.)
    kind : {'bulk', 'speciated'}
        Whether to load the bulk or speciated dataset.
    data_type : {'total', 'percentage'}
        Whether to load total emissions or percentage allocations.

    Returns
    -------
    Path
        Full path to the NetCDF dataset.
    """
    base = Path(settings.gridding_path)
    folder = "esgf/bb4cmip7" if kind == "bulk" else "esgf/bb4cmip7_voc"

    time_range = "190001-202312" if data_type == "total" else "175001-202312"

    filename = f"{variable}_input4MIPs_emissions_CMIP_DRES-CMIP-BB4CMIP7-2-1_gn_{time_range}.nc"

    return base / folder / variable / "gn" / "v20250612" / filename


# %%
def load_bb4cmip7_datasets(g, kind="speciated", lock=None, proxy_time_range=None):
    """
    Load BB4CMIP7 datasets for a given gas `g` and fixed kind.

    Loads both the total emissions file and all sectoral percentage files.

    Parameters
    ----------
    g : str
        Gas name (e.g., "CO", "VOC", etc.)
    kind : {"bulk", "speciated"}, optional
        Dataset kind (defaults to "speciated").
    lock : threading.Lock or dask.utils.SerializableLock, optional
        Lock to use during file access (recommended when using dask).
    proxy_time_range : slice or str, optional
        Time range passed to `.sel(time=...)`. Uses `_normalize_time_slice` if None.

    Returns
    -------
    dict
        {
            "total": <xarray.Dataset>,
            "percentage": {sector: <xarray.Dataset>, ...}
        }
    """
    # sectors for which we load percentage files
    sectors = ["AGRI", "BORF", "DEFO", "PEAT", "SAVA", "TEMF"]

    if proxy_time_range is None:
        proxy_time_range = PROXY_TIME_RANGE  # fallback to global

    # load total emissions
    ds_total = xr.open_dataset(
        get_bb4cmip7_location(variable=g, kind=kind, data_type="total"),
        engine="h5netcdf",
        chunks={},
        lock=lock
    ).sel(time=_normalize_time_slice(proxy_time_range)).drop_vars(
        ["lat_bnds", "lon_bnds", "time_bnds"]
    ).rename({f"{g}": "emissions"})

    # load percentage datasets per sector
    ds_percentages = {}
    for s in sectors:
        variable = f"{g}percentage{s}"
        ds = xr.open_dataset(
            get_bb4cmip7_location(variable=variable, kind=kind, data_type="percentage"),
            engine="h5netcdf",
            chunks={},
            lock=lock
        ).sel(time=_normalize_time_slice(proxy_time_range)).drop_vars(
            ["lat_bnds", "lon_bnds", "time_bnds"]
        ).rename({f"{variable}": "percentage"})

        ds_percentages[s] = ds

    return {"total": ds_total, "percentage": ds_percentages}


# %%
def compute_sectoral_emissions(datasets):
    """
    Compute sectoral emissions by multiplying total emissions with sector percentages,
    and interpolate both total and sector datasets from 0.25° to 0.5° grid.

    Parameters
    ----------
    datasets : dict
        Output from `load_bb4cmip7_datasets()`, containing:
            - datasets["total"] (xarray.Dataset with variable 'emissions')
            - datasets["percentage"][sector] (xarray.Dataset with variable 'percentage')

    Returns
    -------
    dict
        Dictionary of xarray.Datasets per sector, each containing variable 'emissions'
        interpolated onto the 0.5° grid.
    """
    ds_total = datasets["total"]
    ds_percentages = datasets["percentage"]

    # First interpolate the total emissions to the target grid
    ds_total_interp = ds_total.interp(
        latitude=target_lat,
        longitude=target_lon,
        method="linear"
    )

    sectoral_emissions = {}

    for sector, ds_perc in ds_percentages.items():
        # Interpolate the sector percentage dataset to target grid
        ds_perc_interp = ds_perc.interp(
            latitude=target_lat,
            longitude=target_lon,
            method="linear"
        )

        # Multiply totals by sector percentages (convert percentage to fraction)
        ds_sector = xr.Dataset({
            "emissions": ds_total_interp["emissions"] * (ds_perc_interp["percentage"] / 100)
        })

        sectoral_emissions[sector] = ds_sector

    return sectoral_emissions


# %%
def rename_and_aggregate_emissions(data, variable="emissions"):
    """
    Rename and aggregate sectors for either a single-level or nested dictionary.

    Supports both:
      - {sector: xarray.Dataset}
      - {species: {sector: xarray.Dataset}}

    Parameters
    ----------
    data : dict
        Either a dict of sectors, or a nested dict {species: {sector: Dataset}}.
    variable : str
        Name of the variable in each Dataset (e.g. "emissions").

    Returns
    -------
    dict
        Updated dictionary with renamed and aggregated sectors.
        - If input was {sector: Dataset}, output is {sector: Dataset}.
        - If input was {species: {sector: Dataset}}, output is {species: {sector: Dataset}}.
    """
    sector_mapping_singlesector = {
        "AGRI": "AWB",
        "PEAT": "PEAT",
        "SAVA": "GRSB"
    }

    gfed_sectors_forest = ["BORF", "DEFO", "TEMF"]

    def _aggregate_single(sectoral_emissions):
        """Aggregate and rename for one sector dictionary."""
        updated = {}

        # Rename individual sectors
        for old_sector, new_sector in sector_mapping_singlesector.items():
            if old_sector in sectoral_emissions:
                updated[new_sector] = sectoral_emissions[old_sector]

        # Aggregate forest sectors
        forest_datasets = [sectoral_emissions[s] for s in gfed_sectors_forest if s in sectoral_emissions]
        if forest_datasets:
            forest_sum = sum(ds[variable] for ds in forest_datasets)
            updated["FRTB"] = xr.Dataset({variable: forest_sum})

        # Copy remaining sectors (not renamed or aggregated)
        for s, ds in sectoral_emissions.items():
            if s not in sector_mapping_singlesector and s not in gfed_sectors_forest:
                updated[s] = ds

        return updated

    # detect if nested or not
    first_value = next(iter(data.values()))

    if isinstance(first_value, dict):
        # Nested dict case: {species: {sector: Dataset}}
        return {sp: _aggregate_single(sector_dict) for sp, sector_dict in data.items()}
    else:
        # Single-level dict case: {sector: Dataset}
        return _aggregate_single(data)


# %%
def time_average(data, variable="emissions", time_range=(2014, 2023)):
    """
    Compute monthly climatological means over a specified time range for
    emissions data, preserving the input dictionary structure.

    Works for both:
      - {sector: xarray.Dataset}
      - {species: {sector: xarray.Dataset}}

    Parameters
    ----------
    data : dict
        Dictionary of emissions data (sector-level or species→sector-level).
    variable : str
        Variable name in the Dataset (default: 'emissions').
    time_range : tuple(int, int)
        Start and end years, inclusive (default: (2014, 2023)).

    Returns
    -------
    dict
        Same structure as input, but each Dataset contains the monthly mean
        climatology over the selected years, with dimension 'month'.
    """
    start_year, end_year = time_range

    def _monthly_mean(ds):
        """Select time range and compute monthly climatology."""
        ds_sel = ds.sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-31"))
        monthly = ds_sel[variable].groupby("time.month").mean(dim="time", skipna=True)
        return monthly.to_dataset(name=variable)

    def _process_dict(d):
        """Apply to a single-level dict {sector: Dataset}."""
        return {sector: _monthly_mean(ds) for sector, ds in d.items()}

    # Detect whether nested structure or single level
    first_val = next(iter(data.values()))
    if isinstance(first_val, dict):
        # Nested structure: {species: {sector: Dataset}}
        return {sp: _process_dict(sector_dict) for sp, sector_dict in data.items()}
    else:
        # Single-level structure: {sector: Dataset}
        return _process_dict(data)


# %% [markdown]
# ## Obtain speciated shares for all sectors

# %%
# load and process speciated emissions
speciated_datasets = {}
speciated_sectoral_emissions = {}

for sp in GASES_ESGF_BB4CMIP_VOC:
    ds_sp = load_bb4cmip7_datasets(sp, kind="speciated", lock=lock, proxy_time_range=PROXY_TIME_RANGE)
    ds_sp_sectoral = compute_sectoral_emissions(ds_sp)
    
    speciated_datasets[sp] = ds_sp      # contains both total and percentage
    speciated_sectoral_emissions[sp] = ds_sp_sectoral  # contains emissions per sector

# %%
# rename single sectors, aggregate forest sectors
speciated_renamed = rename_and_aggregate_emissions(speciated_sectoral_emissions)

# %%
# time-average over PROXY_RANGES on emissions
speciated_average = time_average(speciated_renamed)

# %%
# because after processing the speciated emissions don't entirely add up to the bulk, 
# we have to calculate the shares relative to the sum of the speciated emissions
# such that the shares we use as proxies will preserve mass when applied to future emissions

species_shares = {}

for sp, sector_dict in speciated_average.items():
    species_shares[sp] = {}

    for sector, ds_sp in sector_dict.items():
        # species-specific emissions
        sp_emis = ds_sp["emissions"].fillna(0)

        # get all speciated emissions for this sector
        speciated_fields = [
            sp_dict[sector]["emissions"].fillna(0)
            for sp_dict in speciated_average.values()
            if sector in sp_dict
        ]

        # sum of all speciated emissions (denominator)
        speciated_sum = sum(speciated_fields)

        # compute normalized share (safe division)
        share = xr.where(speciated_sum > 0, sp_emis / speciated_sum, 0)

        # store dataset
        species_shares[sp][sector] = share.to_dataset(name="emissions_share")

# %%
# Precompute sectoral sums once
sector_sums = {}
for sector in {s for d in speciated_average.values() for s in d}:
    sector_sums[sector] = sum(
        sp_dict[sector]["emissions"].fillna(0)
        for sp_dict in speciated_average.values()
        if sector in sp_dict
    )

# Then compute shares per species using the precomputed sums
species_shares = {}
for sp, sector_dict in speciated_average.items():
    species_shares[sp] = {}
    for sector, ds_sp in sector_dict.items():
        sp_emis = ds_sp["emissions"].fillna(0)
        total = sector_sums[sector]
        share = xr.where(total > 0, sp_emis / total, 0)
        species_shares[sp][sector] = share.to_dataset(name="emissions_share")

# %% [markdown]
# ## Compute and export

# %%
outdir = new_proxies_location / "NMVOC_speciation"
outdir.mkdir(parents=True, exist_ok=True)

ysel_filename = "2014-23"
encoding = {"emissions_share": {"zlib": True, "complevel": 4}}

for sp in tqdm(GASES_ESGF_BB4CMIP_VOC, desc="Species"):

    # Build output filename
    outfile = outdir / f"{sp}_other_voc_em_speciated_NMVOC_openburning_{ysel_filename}.nc"

    # Skip if file already exists
    if outfile.exists():
        print(f"Skipping {outfile.name} (already exists)")
        continue

    # List to hold each sector's DataArray
    sector_arrays = []
    sector_names = []

    for sector, ds_share in species_shares[sp].items():
        # Rename lat/lon if necessary
        ds_share = ds_share.rename({
            "latitude": "lat" if "latitude" in ds_share.dims else "lat",
            "longitude": "lon" if "longitude" in ds_share.dims else "lon"
        })

        # Ensure variable is DataArray
        da = ds_share["emissions_share"]
        # Add 'sector' dimension
        da = da.expand_dims({"sector": [sector]})
        sector_arrays.append(da)
        sector_names.append(sector)

    # Combine all sectors along 'sector' dimension
    combined = xr.concat(sector_arrays, dim="sector")
    combined = combined.assign_coords(sector=sector_names)

    # Add gas dimension
    combined = combined.expand_dims({"gas": [sp]})

    # Broadcast to scenario years
    combined = combined.expand_dims({"year": scenario_years})

    # Reorder dimensions: lat, lon, gas, sector, year, month
    combined = combined.transpose("lat", "lon", "gas", "sector", "year", "month")

    # Write NetCDF
    combined.to_netcdf(outfile, encoding=encoding)


# %%
