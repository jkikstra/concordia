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
import itertools
from functools import lru_cache

import xarray as xr
from pathlib import Path
import ptolemy as pt
import pyreadr
import pyogrio as pio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plts


# %%
from concordia.cmip7.utils import read_r_variable, read_r_to_da

# %%
try:
    SETTINGS_FILE = Path(__file__).parent.parent / "config_cmip7_v0_2.yaml"
except NameError:
    SETTINGS_FILE = Path().resolve().parent / "config_cmip7_v0_2.yaml"

# %%
HARMONIZATION_VERSION = ""
settings = Settings.from_config(version=HARMONIZATION_VERSION,
                                local_config_path=Path(Path.cwd(),
                                                       SETTINGS_FILE))

# %%
template_file = (
    settings.gridding_path
    / "example_files/GCAM4-SSP4-34-SPA4-V25-Harmonized-DB-Sulfur-em-aircraft-anthro_input4MIPs_emissions_CMIP_IAMC-V1_gn_201501-210012.nc"
)
template = xr.open_dataset(template_file)

# %%
settings.country_combinations


# %% [markdown]
# ## functions

# %%
def mask_to_ary(row):
    a = pyreadr.read_r(row.file)[f"{row.iso}_mask"]
    lat = template.lat[::-1][row.start_row - 1 : row.end_row]
    lon = template.lon[row.start_col - 1 : row.end_col]
    da = xr.DataArray(
        np.asarray(a, dtype="float32"), coords={"lat": lat, "lon": lon}
    ).reindex(lat=template.lat, lon=template.lon, fill_value=0)
    return da


# %%
def gen_mask():
    print("Generating Mask Raster")
    files = settings.gridding_path.glob("mask/*.Rd")
    df = pd.DataFrame(
        [[f.stem.split("_")[0], f] for f in files],
        columns=["iso", "file"],
    )
    idxs = pd.read_csv(settings.gridding_path / "country_location_index_05.csv")
    data = pd.merge(df, idxs, on="iso")
    mask = xr.concat([mask_to_ary(s) for i, s in data.iterrows()], data.iso)
    return mask


# %%
def recombine_mask(mask, include_world=True):
    country_combinations = settings.country_combinations
    for comb, (iso1, iso2) in country_combinations.items():
        mask = xr.concat(
            [
                mask,
                (
                    mask.sel(iso=iso1).fillna(0) + mask.sel(iso=iso2).fillna(0)
                ).assign_coords({"iso": comb}),
            ],
            dim="iso",
        )
    comb_mask = mask.drop_sel(iso=list(itertools.chain(*country_combinations.values())))
    if include_world:
        comb_mask = xr.concat(
            [
                comb_mask,
                comb_mask.sum(dim="iso").assign_coords({"iso": "World"}),
            ],
            dim="iso",
        )
    return comb_mask


# %%
def add_eez_to_mask(mask):
    print("Rasterize and add eez to mask raster")
    rasterize = pt.Rasterize(
        shape=(mask.sizes["lat"], mask.sizes["lon"]),
        coords={"lat": mask.coords["lat"], "lon": mask.coords["lon"]},
    )
    rasterize.read_shpf(
        pio.read_dataframe(
            settings.gridding_path / "non_ceds_input" / "eez_v12.gpkg",
            where="ISO_TER1 IS NOT NULL and POL_TYPE='200NM'",
        )
        .dissolve(by="ISO_TER1")
        .reset_index(names=["iso"]),
        idxkey="iso",
    )
    eez = rasterize.rasterize(strategy="weighted", normalize_weights=False).astype(
        mask.dtype
    )

    eez = eez.assign_coords(iso=eez.indexes["iso"].str.lower()).reindex_like(
        mask, fill_value=0.0
    )

    new_mask = mask + eez
    totals = new_mask.sum("iso")
    return (new_mask / totals).where(totals, 0)


# %%
def gen_comb_mask():
    mask = gen_mask()
    mask = add_eez_to_mask(mask)
    mask = recombine_mask(mask, include_world=False)
    return mask


# %%
def mask_to_indexraster(mask):
    indexraster = pt.IndexRaster.from_weighted_raster(mask.rename(iso="country"))
    return indexraster


# %%
def gen_indexraster(mask):
    print("Generating indexraster")
    indexraster = mask_to_indexraster(mask)
    indexraster.to_netcdf(settings.gridding_path / "ssp_comb_indexraster.nc")


# %% [markdown]
# ## create mask

# %%
mask = gen_comb_mask()

# %% [markdown]
# ## check and correct mask for border cell issues

# %%
# check whether the countrymask adds up to 1 everywhere

# sum across all isos to obtain total per grid cell
total = mask.sum(dim="iso")

# check where the total exceeds 1
overlap_mask = total > 1

# compute by how much it exceeds 1
excess = total - 1
excess = excess.where(excess > 0)

print("Number of overlapping cells that add up to > 1:", overlap_mask.sum().item())

# %%
excess = (total - 1).where(total > 1)

plt.figure(figsize=(9, 4))
excess.plot(cmap="magma_r")
plt.gca().set_aspect('auto')
plt.title("Excess amount above 1 per grid cell")
plt.show()

# %%
# correct the countrymask for these small deviations on the boundary cells between countries
# compute scaling factor (1 / total) for overlapping cells, else 1
scale = xr.where(overlap_mask, 1 / total, 1)

# apply scaling to each iso, i.e. preserves ratios between isos in overlapping cells
corrected = mask * scale

# replace original variable in dataset
mask = corrected.clip(min=0, max=1)

# save to a new file
mask.to_netcdf(settings.gridding_path / "ssp_comb_countrymask.nc")

# %% [markdown]
# ## create indexraster

# %%
gen_indexraster(mask)
