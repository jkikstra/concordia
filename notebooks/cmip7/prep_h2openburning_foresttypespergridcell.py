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
import numpy as np
from pathlib import Path

# %% [markdown]
# # Prepare shares of BB4CMIP7 forest types
# This will enable the production of h2_openburning, from co_openburning
# The goal: 
# 1. calculate emissions weights (over the proxy period) for DEFO/TEMF/BORF
# 2. weight the EF_CO/EF_H2 relationships [of https://www.geo.vu.nl/~gwerf/GFED/GFED4/ancill/GFED4_Emission_Factors.xlsx] based on this for each gridcell; future versions could consider aligning with [Andrea 2019, Table](https://acp.copernicus.org/articles/19/8523/2019/acp-19-8523-2019-t01.xlsx) / GFED5

# Time to run: ~1 min


# %%
TIME = slice("2014-01-01", "2023-12-31")


# %% 
# locate data
bb_data_cmip7_location = Path("C:/Users/kikstra/IIASA/ECE.prog - Documents/Projects/CMIP7/IAM Data Processing/concordia_cmip7_v0-4-0/input/gridding/esgf/bb4cmip7")

ds_h2_total = xr.open_dataset(bb_data_cmip7_location / "H2_input4MIPs_emissions_CMIP_DRES-CMIP-BB4CMIP7-2-1_gn_190001-202312.nc")
ds_h2_perc_borf = xr.open_dataset(bb_data_cmip7_location / "H2percentageBORF_input4MIPs_emissions_CMIP_DRES-CMIP-BB4CMIP7-2-1_gn_175001-202312.nc")
ds_h2_perc_defo = xr.open_dataset(bb_data_cmip7_location / "H2percentageDEFO_input4MIPs_emissions_CMIP_DRES-CMIP-BB4CMIP7-2-1_gn_175001-202312.nc")
ds_h2_perc_temf = xr.open_dataset(bb_data_cmip7_location / "H2percentageTEMF_input4MIPs_emissions_CMIP_DRES-CMIP-BB4CMIP7-2-1_gn_175001-202312.nc")

# %% 
# datasets collection
dss = {
    # variable: dataset
    'BORF': ds_h2_perc_borf,
    'DEFO': ds_h2_perc_defo,
    'TEMF': ds_h2_perc_temf
}


# %% 
# select time, replace NAs with zeroes; collect in one ds

ds_h2_total_period = ds_h2_total.sel(time=TIME)
ds_h2_total_period['H2'] = ds_h2_total_period['H2'].fillna(0)
ds_h2_total_period = ds_h2_total_period['H2'].squeeze()

ds_h2_forest = None
for var, ds in dss.items():
    var_perc = f"H2percentage{var}"

    # select time period
    ds = ds.sel(time=TIME)
    # replace NAs with zeroes
    ds[var_perc] = ds[var_perc].fillna(0)

    # only keep values, discard metadata (to allow for easy merging)
    da = ds[var_perc].squeeze()
    da.attrs = {} # remove all remaining attributes
    
    # calculate sectoral emissions per gridcell for each month
    da = (
        ds_h2_total_period * da
    )

    # concat ds into ds_h2_forest, where we add a coordinate 'sector'
    if ds_h2_forest is None:
        ds_h2_forest = da.expand_dims({'sector': [var]})
    else:
        ds_h2_forest = xr.concat([ds_h2_forest, da.expand_dims({'sector': [var]})], dim='sector')

# %%
# calculate "(sectoral) forest emissions" across the years, for each month

# SECTORAL FOREST emissions [sum across years]
# -------------------------------------------
# split month and year
h2_forest_period = ds_h2_forest.assign_coords(
    year=("time", ds_h2_forest["time"].dt.year.data),
    month=("time", ds_h2_forest["time"].dt.month.data)
).groupby(["year", "month"]).mean()
# Do sum over the different years
h2_forest_period = h2_forest_period.sum(dim="year")

# %% 
# load template grid
# %%
import concordia.cmip7.utils_futureproxy_ceds_bb4cmip as uprox
from concordia.cmip7.CONSTANTS import CONFIG, PROXY_YEARS

VERSION = CONFIG
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

# load CEDS example file to get the right grid settings
template_file = (
    settings.gridding_path
    / "example_files" / "GCAM4-SSP4-34-SPA4-V25-Harmonized-DB-Sulfur-em-aircraft-anthro_input4MIPs_emissions_CMIP_IAMC-V1_gn_201501-210012.nc"
)

template = xr.open_dataset(template_file)
target_lat = template["lat"].values
target_lon = template["lon"].values

# %% 
# regrid `h2_forest_period` and `h2_forest_period_sum` to 0.5 degree
h2_forest_period_0p5 = h2_forest_period.interp(
        latitude=target_lat,
        longitude=target_lon,
        method="linear"
    )
# calculate sum for each 0.5 degree 
h2_forest_sum_period_0p5 = h2_forest_period_0p5.sum(
    dim='sector'
)

# %% 
# calculate multi-year shares; divide at the 0.5 degree grid level
ds_shares = None
for var in dss.keys():
    print(var)

    # collect as dataset
    ds_h2_forest_emissions_shares = xr.Dataset({
        "emissions_share" : (
        xr.where(
            h2_forest_sum_period_0p5 > 0, # if there was *any* burning in the proxy, else zero
            (
                h2_forest_period_0p5.sel(sector=var) /
                h2_forest_sum_period_0p5
            ),
            0
        )
    )
    })

    # ds_shares[var] = ds_h2_forest_emissions_shares

    # concat ds into ds_h2_forest, where we add a coordinate 'sector'
    if ds_shares is None:
        ds_shares = ds_h2_forest_emissions_shares.expand_dims({'sector': [var]})
    else:
        ds_shares = xr.concat([ds_shares, ds_h2_forest_emissions_shares.expand_dims({'sector': [var]})], dim='sector')


ds_shares

# %% 
# plot as a check: should all be *between* 0 and 1
# - all months, histogram
ds_shares.sel(sector='BORF').emissions_share.plot(bins=100) # hist
ds_shares.sel(sector='DEFO').emissions_share.plot(bins=100) # hist
ds_shares.sel(sector='TEMF').emissions_share.plot(bins=100) # hist
# - one month, plot
ds_shares.sel(sector='BORF', month=4).emissions_share.plot()
ds_shares.sel(sector='DEFO', month=4).emissions_share.plot()
ds_shares.sel(sector='TEMF', month=4).emissions_share.plot()

# plot as a check: should all be *exactly* 0 or 1
# - all months, histogram
ds_shares.sum(dim='sector').emissions_share.plot(bins=100)
# - one month, plot
ds_shares.sel(month=4).sum(dim='sector').emissions_share.plot()
# - check unique values
assert np.isclose(np.min(ds_shares.emissions_share.values), 0, atol=1e-5), "minimum should be zero"
assert np.isclose(np.max(ds_shares.emissions_share.values), 1, atol=1e-5), "maximum is expected to be 1 (is not explicitly required, but it is extremely likely that at least one gridcell only has one forest emissions type)"
np.unique(ds_shares.sum(dim='sector').emissions_share.values)


# %%
# multiply by Emission factor shares for each sector
EF_SOURCE = 'GFED4.1s'
# Emission factors (divide EF_h2 by EF_co)
EF_h2_div_EF_co = {
    'GFED4.1s': { # from summary table for emissions calculation; Akagi2011
        
        # forest sectors
        'BORF': 2.03 / 127,
        'TEMF': 2.03 / 88,
        'DEFO': 3.36 / 93,

        'Tropical Forest': np.nan, # not in summary table; not separately in CMIP7 historical emissions

        # other sectors (used directly)
        'SAVA': 1.7 / 63,
        'PEAT': 3.36 / 210,
        'AGRI': 2.59 / 102,
    },
    # not used
    # 'Andrea2019': {
    #     'BORF': 1.6 / 121,
    #     'TEMF': 2.1 / 113,
    #     'DEFO': np.nan, # 
    #     'Tropical Forest': 3.1 / 104 # not in CMIP7 historical/files
    # }
}

# example:
EF_h2_div_EF_co[EF_SOURCE]['SAVA']

# %%
# create prep proxy file, with values being (EF_H2 / EF_CO), because
# E_H2 = E_CO * (EF_H2 / EF_CO)

map_EF_h2_div_EF_co = None

for burning_sector in ["SAVA", "PEAT", "AGRI", "FRTB"]:

    if burning_sector != "FRTB": # no cumbersome weighting here
        map_EF_h2_div_EF_co_sector = xr.Dataset({
            "EF_h2_div_EF_co" : (
            xr.where(
                h2_forest_sum_period_0p5 > 0, # if there was *any* burning in the proxy, else zero
                EF_h2_div_EF_co[EF_SOURCE][burning_sector],
                0
            )
        )
        })
    if burning_sector == "FRTB": # use weights from ds_shares
        
        map_EF_h2_div_EF_co_sector = xr.Dataset({
            "EF_h2_div_EF_co" : (
            xr.where(
                h2_forest_sum_period_0p5 > 0, # if there was *any* burning in the proxy, else zero
                (
                    # EF * weight
                    EF_h2_div_EF_co[EF_SOURCE]['BORF'] * ds_shares.sel(sector='BORF').emissions_share.squeeze() +
                    EF_h2_div_EF_co[EF_SOURCE]['DEFO'] * ds_shares.sel(sector='DEFO').emissions_share.squeeze() +
                    EF_h2_div_EF_co[EF_SOURCE]['TEMF'] * ds_shares.sel(sector='TEMF').emissions_share.squeeze()
                ),
                0
            )
        )
        })
    
    # concat ds into ds_h2_forest, where we add a coordinate 'sector'
    if map_EF_h2_div_EF_co is None:
        map_EF_h2_div_EF_co = map_EF_h2_div_EF_co_sector.expand_dims({'sector': [burning_sector]})
    else:
        map_EF_h2_div_EF_co = xr.concat([map_EF_h2_div_EF_co,
                                         map_EF_h2_div_EF_co_sector.expand_dims({'sector': [burning_sector]})], dim='sector')


# %% 
# a bit of renaming to make it fit how we write out openburning emissions (CO)

co = xr.open_dataset((
    settings.out_path / 'test_paris_H' /
    'SO2-em-openburning_input4MIPs_emissions_CMIP6plus_IIASA-IAMC-esm-scen7-h-0-4-0_gn_202201-210012.nc'
)
)

# %%
# save out the file in the proxy location
# ... TBD ...
# 1. [ ] lat/lon instead of latitude/longitude
map_EF_h2_div_EF_co_formatted = map_EF_h2_div_EF_co.rename({
            "latitude": "lat" if "latitude" in map_EF_h2_div_EF_co.dims else "lat",
            "longitude": "lon" if "longitude" in map_EF_h2_div_EF_co.dims else "lon"
        })
# 2. [ ] sector full names (instead of short names)
sector_mapping = {
    'AGRI': 'Agricultural Waste Burning',
    'PEAT': 'Peat Burning',
    'SAVA': 'Grassland Burning',
    'FRTB': 'Forest Burning'
}

# Create rename dictionary: old_name -> new_name
rename_dict = {s: sector_mapping[s] for s in map_EF_h2_div_EF_co_formatted.sector.values}
map_EF_h2_div_EF_co_formatted['sector'] = [rename_dict[x] for x in map_EF_h2_div_EF_co_formatted.sector.values]


# 3. coordinate order; lat, lon, time, sector
# Broadcast to scenario years
map_EF_h2_div_EF_co_formatted = map_EF_h2_div_EF_co_formatted.expand_dims({"year": PROXY_YEARS})

# Reorder dimensions: lat, lon, gas, sector, year, month
map_EF_h2_div_EF_co_formatted = map_EF_h2_div_EF_co_formatted.transpose("lat", "lon", "sector", "year", "month")





# %% 
# calculate averages for each month across years
# Write NetCDF
encoding = {"EF_h2_div_EF_co": {"zlib": True, "complevel": 4}}
map_EF_h2_div_EF_co_formatted.to_netcdf(settings.proxy_path / 'EF_h2_div_EF_co.nc', 
                                        encoding=encoding)

