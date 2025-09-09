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
import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# %% [markdown]
# ## import data

# %%
grid_file_location = "/home/hoegner/Projects/CMIP7/input/gridding/"

# %%
countrymap_path = Path(grid_file_location)
f = "ssp_comb_countrymask.nc"
mask = xr.open_dataset(
    countrymap_path / f,
    engine="netcdf4",
)

# %% [markdown]
# ## run checks

# %%
# check whether the countrymask adds up to 1 everywhere

var = mask["__xarray_dataarray_variable__"]

# sum across all isos to obtain total per grid cell
total = var.sum(dim="iso")

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

# %% [markdown]
# ## write correction

# %%
# correct the countrymask for these small deviations on the boundary cells between countries

# compute scaling factor (1 / total) for overlapping cells, else 1
scale = xr.where(overlap_mask, 1 / total, 1)

# apply scaling to each iso, i.e. preserves ratios between isos in overlapping cells
corrected = var * scale

# replace original variable in dataset
mask["__xarray_dataarray_variable__"] = corrected.clip(min=0, max=1)

# save to a new file
mask.to_netcdf(countrymap_path / "ssp_comb_countrymask_corrected.nc")

# %% [markdown]
# ## test plots of (corrected) mask

# %%
# da_masked = ds["BC_em_anthro"].where(mask.sel(iso="chn")["__xarray_dataarray_variable__"], other=0)
# da_to_plot = da_masked.isel(time=1).sel(sector="Industrial")
da_to_plot =  mask.sel(iso="aus")["__xarray_dataarray_variable__"]

fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()}, figsize=(12,6))

da_to_plot.plot.pcolormesh(
    ax=ax,
    transform=ccrs.PlateCarree(),
    cmap="viridis",
    add_colorbar=True,
    cbar_kwargs={
        "shrink": 0.8
    }
)

ax.add_feature(cfeature.BORDERS, linewidth=0.8)
ax.add_feature(cfeature.COASTLINE, linewidth=0.8)

plt.show()

# %%
da_to_plot1 =  mask.sel(iso="ind")["__xarray_dataarray_variable__"]
da_to_plot2 =  mask.sel(iso="bgd")["__xarray_dataarray_variable__"]

fig, ax = plt.subplots(figsize=(12,6))

da_to_plot1.plot.pcolormesh(
    ax=ax,
    cmap="viridis",
    add_colorbar=True,
    alpha=.5,
    cbar_kwargs={
        "shrink": 0.8
    }
)

da_to_plot2.plot.pcolormesh(
    ax=ax,
    cmap="viridis",
    add_colorbar=True,
    alpha=.5,
    cbar_kwargs={
        "shrink": 0.8
    }
)

ax.set_xlim(60,100)
ax.set_ylim(0,40)

#ax.add_feature(cfeature.BORDERS, linewidth=0.8)
#ax.add_feature(cfeature.COASTLINE, linewidth=0.8)

plt.show()

# %%
da_to_plot1 =  mask.sel(iso="bgd")["__xarray_dataarray_variable__"]
da_to_plot2 =  mask.sel(iso="bgd")["__xarray_dataarray_variable__"]

fig, ax = plt.subplots(figsize=(12,6))

da_to_plot1.plot.pcolormesh(
    ax=ax,
    cmap="viridis",
    add_colorbar=True,
    cbar_kwargs={
        "shrink": 0.8
    }
)

ax.set_xlim(60,100)
ax.set_ylim(0,40)

#ax.add_feature(cfeature.BORDERS, linewidth=0.8)
#ax.add_feature(cfeature.COASTLINE, linewidth=0.8)

plt.show()
