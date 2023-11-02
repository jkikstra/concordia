# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
from pathlib import Path

import xarray as xr
import yaml


# %%
with open("../config.yaml") as f:
    config = yaml.safe_load(f)
base_path = Path(config["base_path"])

# %% [markdown]
# # Shipping
#
# We use shipping patterns from MariTeam to generate shipping proxies. NOx, SOx,
# and CO2 are provided explicitly, and all other patterns are based on CO2, as
# NOx and SOx have specific pollution controls embedded.
#
# Citation: Kramel, D., Muri, H., Kim, Y., Lonka, R., Nielsen, J.B., Ringvold,
# A.L., Bouman, E.A., Steen, S. and Strømman, A.H., 2021. Global shipping
# emissions from a well-to-wake perspective: the MariTEAM model. Environmental
# science & technology, 55(22), pp.15040-15050.
# https://pubs.acs.org/doi/10.1021/acs.est.1c03937
#


# %%
def mariteam_shipping():
    gases = ["BC", "CH4", "CO", "CO2", "NH3", "NOx", "OC", "Sulfur", "VOC"]
    mari = list(
        Path(".").glob(
            base_path / "gridding_process_files/non_ceds_input/*MariTeam*.nc"
        )
    )
    mari = {f.stem.split("_")[-2]: f for f in mari}
    ceds_to_mari = {"Sulfur": "SO2"}  # maritime uses so2, ceds uses sulfur
    fallback = "CO2"  # if maritime doesn't provide data, use co2 as backup

    def convert_mariteam_to_ceds(mari, gas):
        # get path of file to use for this gas
        if gas in ceds_to_mari:
            pth = mari[ceds_to_mari[gas]]
        elif gas in mari:
            pth = mari[gas]
        else:
            pth = mari[fallback]

        # make sure gas name is aligned with gas arg
        print(f"For gas {gas}, using {pth}")
        with xr.open_dataarray(pth) as da:
            return da.drop_vars(["gas"]).assign_coords(gas=[gas])

    for gas in gases:
        da = convert_mariteam_to_ceds(mari, gas)
        fname = Path(
            base_path / f"gridding_process_files/proxy_rasters/shipping_{gas}.nc"
        )
        da.to_netcdf(fname, encoding={"emissions": dict(zlib=True, complevel=2)})


mariteam_shipping()

# %% [markdown]
# # Industry CDR
#
# **NOTE** Originally I hoped to use cement emission patterns, but it turns out that most industry CDR is from plastics, so fallback to simple industry CO2 emission patterns.

# %%
ind_cdr = (
    xr.open_dataset(
        base_path / "gridding_process_files/proxy_rasters/anthro_CO2.nc"
    ).sel(sector="IND")
).emissions
ind_cdr["sector"] = "IND_CDR"
ind_cdr

# %% [markdown]
# ## Regrid to Input4MIPs lat/lon

# %% [raw]
# da = xr.open_dataarray(base_path / 'gridding_process_files/non_ceds_input/GID_Cement_CO2_2019_v1.0_Grid.nc')
# with xr.open_dataset(base_path / 'iam_files/cmip6/REMIND-MAGPIE_SSP5-34-OS/BC-em-openburning_input4MIPs_emissions_CMIP_REMIND-MAGPIE-SSP5-34-OS-V1_gn_201501-210012.nc') as template:
#     da_regrid = da.interp(lon=template.lon, lat=template.lat, method="linear")
# da_regrid

# %% [raw]
# # add seasonality from industry CO2
# ind_co2 = (
#     xr.open_dataset(base_path / 'gridding_process_files/proxy_rasters/anthro_CO2.nc')
#     .sel(sector='IND')
# ).emissions
# ind_cdr = da_regrid * ind_co2 / ind_co2.sum('month')
# ind_cdr

# %% [markdown]
# ## Check country emissions - some are zero

# %% [raw]
# idxr = xr.open_dataarray(base_path / 'gridding_process_files/iso_mask.nc', chunks={'iso': 1})
# cell_area = xr.DataArray(pt.cell_area(lats=ind_cdr.lat, lons=da_regrid.lon), attrs=dict(unit="m2"))
# emissions_df = (
#     (ind_cdr.sel(year=2015) * cell_area * idxr)
#     .sum(["lat", "lon", "month"])
# ).compute().to_series()
# emissions_df.head()

# %% [raw]
# emissions_df[emissions_df > 0].head()

# %% [raw]
# emissions_df[emissions_df == 0].head()

# %% [markdown]
# ## For countries which are 0, use fallback industry proxies

# %% [raw]
# zero_isos = emissions_df[emissions_df == 0].index.get_level_values('iso')

# %% [raw]
# ind_cdr = ind_cdr + (idxr.sel(iso=zero_isos) * ind_co2).sum('iso')
# ind_cdr

# %% [markdown]
# ## Conform to standard proxy format and save

# %% [raw]
# ind_cdr['sector'] = 'IND_CDR'
# ind_cdr.name = 'emissions'

# %% [markdown]
# # OAE CDR and emissions

# %%
# add seasonality from industry CO2
oae_cdr = (
    xr.open_dataset(base_path / "gridding_process_files/proxy_rasters/shipping_CO2.nc")
).emissions.sel(sector="SHP")
oae_cdr["sector"] = "OAE_CDR"
oae_cdr

# %%
# add seasonality from industry CO2
oae_co2 = (
    xr.open_dataset(
        base_path / "gridding_process_files/proxy_rasters/anthro_CO2.nc"
    ).sel(sector="IND")
).emissions
oae_co2["sector"] = "OAE"
oae_co2.name = "emissions"
oae_co2

# %% [markdown]
# # DACCS Placeholder
#
# TODO:
# - Update this with renewable potential in grid cells with CO2 storage from Kearns et al.
# - check to make sure all countries have values

# %%
dac_cdr = (
    xr.open_dataset(
        base_path / "gridding_process_files/proxy_rasters/anthro_CO2.nc"
    ).sel(sector="ENE")
).emissions
dac_cdr["sector"] = "DAC_CDR"
dac_cdr

# %% [markdown]
# # Combine and Save

# %%
da = xr.concat(
    [
        ind_cdr,
        oae_cdr,
        oae_co2,
        dac_cdr,
    ],
    dim="sector",
)

# %%
da

# %%
comp = dict(zlib=True, complevel=5)
da.to_netcdf(
    base_path / "gridding_process_files/proxy_rasters/CDR_CO2.nc",
    encoding={da.name: comp},
)
