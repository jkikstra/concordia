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
from concordia.rescue.proxy import plot_map

# %%
rescue_data = "C:/Users/kikstra/IIASA/RESCUE - WP 1/data_2024_09_16/gridding_process_files/proxy_rasters/CDR_CO2.nc"
cdr_ds = xr.open_dataset(rescue_data)
print(cdr_ds)

# %%
cdr_ds.sector

# %%
cdr_ds.sel(sector="DAC_CDR", year=2100, month=1).squeeze().emissions

# %%
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np

def plot_maps(ds, sectors, ncols=3, year=2100, month=1, proj=ccrs.Robinson()): 

    nrows = int(np.ceil(len(sectors) / ncols))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(6 * ncols, 4.5 * nrows),
        subplot_kw={"projection": ccrs.Robinson()}
    )

    # Flatten axes safely
    if isinstance(axes, np.ndarray):
        axes = axes.flatten() # make indexing easier
    else:
        axes = [axes]

    for i, sector in enumerate(sectors):
        da = (
            ds.sel(sector=sector, year=year, month=month)
            .squeeze()
            .emissions  # or whatever your variable is
        )

        # Plot directly with xarray's .plot.pcolormesh
        da.plot.pcolormesh(
            ax=axes[i],
            transform=ccrs.PlateCarree(),
            cmap="GnBu",
            robust=True,
            cbar_kwargs={"orientation": "horizontal", "shrink": 0.65},
        )
        axes[i].set_title(sector)
        axes[i].coastlines()

    # Remove any unused axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

# cdr_secs = ["DAC_CDR", "OAE_CDR", "IND_CDR", "BECCS", "A/R", "AGLAND", 'DEFOREST', 'NONURB', 'LAND']  # all CDR proxy file sectors
cdr_secs = ["DAC_CDR", "IND_CDR", "OAE_CDR"]  # non-land CDR proxy file sectors

plot_maps(cdr_ds, sectors=cdr_secs)

# %%
cdr_secs_additional = ["BECCS", "A/R", "AGLAND", 'DEFOREST', 'NONURB', 'LAND']  # all CDR proxy file sectors
# cdr_secs = ["DAC_CDR", "IND_CDR", "OAE_CDR"]  # non-land CDR proxy file sectors

plot_maps(cdr_ds, sectors=cdr_secs_additional)

# %%

# %%
ceds_shipping_data = "C:/Users/kikstra/IIASA/RESCUE - WP 1/data_2024_09_16/gridding_process_files/proxy_rasters/shipping_CO2.nc"
ceds_shipping_ds = xr.open_dataset(ceds_shipping_data)
print(ceds_shipping_ds)

# %%
plot_maps(ceds_shipping_ds, 
          sectors=['SHP'],ncols=3,
          year=2030, month=1)

# %%
ceds_anthro_data = "C:/Users/kikstra/IIASA/RESCUE - WP 1/data_2024_09_16/gridding_process_files/proxy_rasters/anthro_CO2.nc"
ceds_anthro_ds = xr.open_dataset(ceds_anthro_data)
print(ceds_anthro_ds)

# %%
ceds_anthro_ds.sector.to_numpy()

# %%
ceds_anthro_ds

# %%
ant_secs = ceds_anthro_ds.sector.to_numpy()
ant_secs_co2 = [x for x in ant_secs if x != 'AGR']
plot_maps(ceds_anthro_ds, 
          sectors=ant_secs_co2,ncols=3,
          year=2030, month=1)

# %%
pik_data = "C:/Users/kikstra/IIASA/ECE.prog - Documents/Projects/CMIP7/IAM Data Processing/Population/pik/pop-dens-SSP2_input4MIPs_population_CMIP_PIK-CMIP-1-0-0_gn_2022-2100.nc"
p_ds = xr.open_dataset(pik_data)
print(p_ds)


# %%
def plot_maps_population(ds, years=[2100], ncols=3, month=7, proj=ccrs.Robinson()): 

    nrows = int(np.ceil(len(years) / ncols))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(6 * ncols, 4.5 * nrows),
        subplot_kw={"projection": ccrs.Robinson()}
    )

    # Flatten axes safely
    if isinstance(axes, np.ndarray):
        axes = axes.flatten() # make indexing easier
    else:
        axes = [axes]

    for i, year in enumerate(years):
        da = (
            ds.sel(time=f"{year}-0{month}-01")
            .squeeze()
            .pop_dens_SSP2  # or whatever your variable is
        )

        # Plot directly with xarray's .plot.pcolormesh
        da.plot.pcolormesh(
            ax=axes[i],
            transform=ccrs.PlateCarree(),
            cmap="GnBu",
            robust=True,
            cbar_kwargs={"orientation": "horizontal", "shrink": 0.65},
        )
        axes[i].set_title(year)
        axes[i].coastlines()

    # Remove any unused axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

plot_maps_population(p_ds, years=[2023, 2100])

# %%
