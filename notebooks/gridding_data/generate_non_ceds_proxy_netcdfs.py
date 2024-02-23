# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import textwrap
from pathlib import Path

import cartopy.crs as ccrs
import geoutils as gu
import matplotlib.pyplot as plt
import rasterio as rio
import rioxarray
import xarray as xr
import yaml
from ptolemy.raster import IndexRaster
from scipy.ndimage import gaussian_filter


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
    mari = {
        f.stem.split("_")[-2]: f
        for f in base_path.glob("gridding_process_files/non_ceds_input/*MariTeam*.nc")
    }
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
# # CDR
#
# We provide proxies for several CDR technologies:
# 1. OAE CDR re-uses shipping CO2 emissions
# 2. DACCS CDR incorporates renewable potentials and CO2 storage potentials
# 3. Industry CDR uses the composition of renewables, CO2 storage and industry co2 emissions
#

# %%
# ind co2 defines the exact grid and other dimensions
ind_co2 = (
    xr.open_dataset(
        base_path / "gridding_process_files/proxy_rasters/anthro_CO2.nc"
    ).sel(sector="IND")
).emissions
ind_co2

# %% [markdown]
# # OAE CDR and emissions

# %% [markdown]
# Use shipping CO2 for OAE CDR emissions

# %%
oae_cdr = (
    (
        xr.open_dataset(
            base_path / "gridding_process_files/proxy_rasters/shipping_CO2.nc"
        )
    )
    .emissions.sel(sector="SHP")
    .assign_coords(sector="OAE_CDR")
)

# %% [markdown]
# **TODO** We might want to try to give the OAE CDR negative emissions some seasonality that correlates with industry emissions. Unfortunately, the industry co2 seasonality is different between regions (compare `ind_co2.sel(lon=slice(0, 20), lat=slice(40, 20)).mean(["year", "gas", "lat", "lon"]).plot()` (Europe) to `ind_co2.sel(lon=slice(0, 20), lat=slice(-10, 30)).mean(["year", "gas", "lat", "lon"]).plot()` (Africa))

# %% [markdown]
# # DACCS and Industrial CDR
#
# Combine renewable potential from GaSP, Global Wind and Solar Atlas with CO2 storage potential

# %%
renewable_potential = gu.Raster(
    base_path / "gridding_process_files/renewable_potential/renewable_potential.tiff"
)


# %%
def read_co2_storage_potential(smooth=True):
    co2_storage_potential = gu.Raster(
        base_path / "gridding_process_files/co2_storage_potential/LOW_05.tif"
    )

    # Has no nodata value set, which defaults to 1e20. an explicit -1 is easier to track
    co2_storage_potential.set_nodata(-1)

    if not smooth:
        return co2_storage_potential

    # Transform into platecarree which is given in units of meter
    crs_platecarree = rio.CRS.from_authority("ESRI", 54001)

    # Calculate dst bbox of the transformation
    transform, width, height = rio.warp.calculate_default_transform(
        co2_storage_potential.crs,
        crs_platecarree,
        co2_storage_potential.width,
        co2_storage_potential.height,
        *co2_storage_potential.bounds,
    )
    bounds = rio.coords.BoundingBox(
        *rio.transform.array_bounds(height, width, transform)
    )
    co2_storage_potential_pc = co2_storage_potential.reproject(
        dst_crs=crs_platecarree, dst_size=(width, height), dst_bounds=bounds
    )
    co2_storage_potential_pc_smooth = gu.Raster.from_array(
        gaussian_filter(
            co2_storage_potential_pc.data,
            sigma=(
                200_000
                / abs(transform.a),  # use 200km stddev for gaussian kernel on both axes
                200_000 / abs(transform.e),
            ),
            mode="wrap",
        ),
        transform,
        crs_platecarree,
        nodata=-1,
    )
    co2_storage_potential_smooth = co2_storage_potential_pc_smooth.reproject(
        co2_storage_potential
    )
    return co2_storage_potential_smooth


co2_storage_potential_smooth = read_co2_storage_potential()
daccs_potential = renewable_potential * co2_storage_potential_smooth


# %%
def gu_to_xarray(raster, grid_ref=None, name=None):
    da = (
        rioxarray.open_rasterio(raster.to_rio_dataset(), band_as_variable=True)
        .band_1.where(lambda df: df != raster.nodata)
        .rename({"x": "lon", "y": "lat"})
        .drop(["spatial_ref"])
    )
    if grid_ref is not None:
        da = da.reindex_like(
            grid_ref, method="nearest"
        )  # the grid is not exactly the same
    if name is not None:
        da = da.rename(name)
    da.attrs.clear()
    return da


# %%
daccs_potential = gu_to_xarray(daccs_potential, ind_co2, "emissions")


# %%
def plot_map(da, title=None, robust=True, add_colorbar=None, **kwargs):
    fig, axis = plt.subplots(
        1, 1, subplot_kw=dict(projection=ccrs.Robinson()), figsize=(12, 6)
    )
    axis.set_global()
    # axis.stock_img()
    axis.coastlines()

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


# %%
plot_map(
    gu_to_xarray(renewable_potential, ind_co2, "emissions").assign_attrs(
        long_name="Renewable potential"
    )
)

# %%
plot_map(
    gu_to_xarray(co2_storage_potential_smooth, ind_co2, "emissions").assign_attrs(
        long_name="CO2 storage potential (in proximity)"
    )
)

# %%
plot_map(
    daccs_potential.assign_attrs(
        long_name="Renewable and CO2 storage potential",
        units="MW GtCO2 m-2",
    ),
    title="DACCS suitability",
)

# %%
ind_co2_dimensions = xr.ones_like(ind_co2.drop("sector"))
ind_co2_seasonality = ind_co2.sum(["gas", "year"])
ind_co2_seasonality /= ind_co2_seasonality.sum(["lat", "lon"]).mean("month")

# %%
# industry CDR is composition of daccs potential and availability of industrial co2 emissions
ind_cdr = (daccs_potential * ind_co2).assign_coords(sector="IND_CDR")

# %%
plot_map(
    ind_cdr.sel(year=2050, month=1).assign_attrs(
        long_name="Renewable, CO2 storage potential and Industry emissions",
    ),
    title="Industry CDR emissions",
)

# %%
indexraster = IndexRaster.from_netcdf(
    base_path / "gridding_process_files" / "ssp_comb_indexraster.nc"
)


def missing_countries(da, do_plot=True):
    missing = ~(indexraster.aggregate(da) > 0)
    print(textwrap.fill(f"Missing countries: {', '.join(indexraster.index[missing])}"))
    if do_plot:
        plot_map(
            indexraster.grid(missing.astype(float)).assign_attrs(long_name="Uncovered"),
            robust=False,
            add_colorbar=False,
        )


# %%
missing_countries(ind_cdr.sel(month=1, year=2050, gas="CO2"))

# %%
# dac cdr as non-seasonal daccs potential (if we find an easy way how to, we might want to add a renewable seasonality)
dac_cdr = (daccs_potential * ind_co2_dimensions).assign_coords(sector="DAC_CDR")

# %%
missing_countries(dac_cdr.sel(month=1, year=2050, gas="CO2"))

# %% [markdown]
# # Combine and Save

# %%
da = (
    xr.concat(
        [
            ind_cdr,
            oae_cdr,
            # oae_co2, # Part of other emissions
            dac_cdr,
        ],
        dim="sector",
    )
    .fillna(0.0)
    .transpose("lat", "lon", "gas", "sector", "year", "month")
)

# %%
da.to_netcdf(
    base_path / "gridding_process_files/proxy_rasters/CDR_CO2.nc",
    encoding={da.name: dict(zlib=True, complevel=2)},
)
