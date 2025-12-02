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
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# %% [markdown]
# # Check that CMIP6 scenario files and their EFs are consistent with the auxiliary files

# %%
# load data
bb_data_cmip6_location = Path("C:/Users/kikstra/IIASA/ECE.prog - Documents/Projects/CMIP7/IAM Data Processing/concordia_cmip7_v0-4-0/input/gridding/esgf/cmip6")
h2_total = xr.open_dataset(bb_data_cmip6_location / "H2-em-openburning_input4MIPs_emissions_ScenarioMIP_IAMC-MESSAGE-GLOBIOM-ssp245-1-1_gn_201501-210012.nc")
h2_perc_allsectors = xr.open_dataset(bb_data_cmip6_location / "H2-openburning-share_input4MIPs_emissions_ScenarioMIP_IAMC-MESSAGE-GLOBIOM-ssp245-1-1_gn_201501-210012.nc")
co_total = xr.open_dataset(bb_data_cmip6_location / "CO-em-openburning_input4MIPs_emissions_ScenarioMIP_IAMC-MESSAGE-GLOBIOM-ssp245-1-1_gn_201501-210012.nc")
co_perc_allsectors = xr.open_dataset(bb_data_cmip6_location / "CO-openburning-share_input4MIPs_emissions_ScenarioMIP_IAMC-MESSAGE-GLOBIOM-ssp245-1-1_gn_201501-210012.nc")

# set variables to look at
SECTOR = 2 # Grassland Burning (SAVA)
TIME = "2015-01-16" # first year


# %%
# calculate savannah emissions
h2_sava_cmip6 = (
    h2_total.sel(time=TIME).H2_em_openburning *
    h2_perc_allsectors.sel(time=TIME,sector=SECTOR).H2_openburning_share
)
co_sava_cmip6 = (
    co_total.sel(time=TIME).CO_em_openburning *
    co_perc_allsectors.sel(time=TIME,sector=SECTOR).CO_openburning_share
)

# calculate the gridpoint ratios
co_div_h2_sava_cmip6 = co_sava_cmip6 / h2_sava_cmip6

# plot map
co_div_h2_sava_cmip6.plot()

# min/max
co_div_h2_sava_cmip6.min()
co_div_h2_sava_cmip6.max()
co_div_h2_sava_cmip6.mean()

# plot histogram of values
plt.figure(figsize=(10, 5))
plt.hist(co_div_h2_sava_cmip6.values.flatten(), bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('CO / H2 Ratio')
plt.ylabel('Frequency')
plt.title('Distribution of CO/H2 Ratios (Grassland Burning)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %%
# calculate agricultural waste burning emissions
h2_awb_cmip6 = (
    h2_total.sel(time=TIME).H2_em_openburning *
    h2_perc_allsectors.sel(time=TIME,sector=0).H2_openburning_share
)
co_awb_cmip6 = (
    co_total.sel(time=TIME).CO_em_openburning *
    co_perc_allsectors.sel(time=TIME,sector=0).CO_openburning_share
)

# calculate the gridpoint ratios
co_div_h2_awb_cmip6 = co_awb_cmip6 / h2_awb_cmip6

# plot map
co_div_h2_awb_cmip6.plot()

# min/max
co_div_h2_awb_cmip6.min()
co_div_h2_awb_cmip6.max()
co_div_h2_awb_cmip6.mean()

# plot histogram of values
plt.figure(figsize=(10, 5))
plt.hist(co_div_h2_awb_cmip6.values.flatten(), bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('CO / H2 Ratio')
plt.ylabel('Frequency')
plt.title('Distribution of CO/H2 Ratios (AWB)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# list of values - remove np.nan and order
co_div_h2_awb_cmip6_values = np.sort(co_div_h2_awb_cmip6.values.flatten()[~np.isnan(co_div_h2_awb_cmip6.values.flatten())])
print(co_div_h2_awb_cmip6_values)

# %% [markdown]
# # Check CMIP7 historical files and their EFs (consistent with GFED4.1s?)
bb_data_cmip7_location = Path("C:/Users/kikstra/IIASA/ECE.prog - Documents/Projects/CMIP7/IAM Data Processing/concordia_cmip7_v0-4-0/input/gridding/esgf/bb4cmip7")

ds_h2_total = xr.open_dataset(bb_data_cmip7_location / "H2_input4MIPs_emissions_CMIP_DRES-CMIP-BB4CMIP7-2-1_gn_190001-202312.nc")
ds_h2_perc_sava = xr.open_dataset(bb_data_cmip7_location / "H2percentageSAVA_input4MIPs_emissions_CMIP_DRES-CMIP-BB4CMIP7-2-1_gn_175001-202312.nc")
ds_co_total = xr.open_dataset(bb_data_cmip7_location / "CO" / "gn" / "v20250612" / "CO_input4MIPs_emissions_CMIP_DRES-CMIP-BB4CMIP7-2-1_gn_190001-202312.nc")
ds_co_perc_sava = xr.open_dataset(bb_data_cmip7_location / "COpercentageSAVA" / "gn" / "v20250612" / "COpercentageSAVA_input4MIPs_emissions_CMIP_DRES-CMIP-BB4CMIP7-2-1_gn_175001-202312.nc")

# %%
# calculate savannah emissions
h2_sava_cmip7 = (
    ds_h2_total.sel(time=TIME).H2.squeeze() *
    ds_h2_perc_sava.sel(time=TIME).H2percentageSAVA.squeeze()
)
co_sava_cmip7 = (
    ds_co_total.sel(time=TIME).CO.squeeze() *
    ds_co_perc_sava.sel(time=TIME).COpercentageSAVA.squeeze()
)


# %%
# calculate the gridpoint ratios
co_div_h2_sava_cmip7 = co_sava_cmip7 / h2_sava_cmip7.plot()

# plot map
co_div_h2_sava_cmip7.plot()

# %%
# plot histogram of values
plt.figure(figsize=(10, 5))
plt.hist(co_div_h2_sava_cmip7.values.flatten(), bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('CO / H2 Ratio')
plt.ylabel('Frequency')
plt.title('Distribution of CO/H2 Ratios (SAVA) CMIP7 DRES 2-1')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
# %%
