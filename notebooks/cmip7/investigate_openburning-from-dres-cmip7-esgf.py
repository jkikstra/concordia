#!/usr/bin/env python
# %% [markdown]
# # Overview
#
# This should create proxies for openburning based on BB4CMIP7
# 


# %% [markdown]
# ### Steps happening in this notebook, which starts from the PNNL server .Rd files
# 1. ...



# %% 

# data grid area:
# gridcellarea: "D:\ESGF\DRES-CMIP-BB4CMIP7-2-1\atmos\fx\gridcellarea\gn\v20250612\gridcellarea_input4MIPs_emissions_CMIP_DRES-CMIP-BB4CMIP7-2-1_gn.nc"
# data emissions:
# emissions: "D:\ESGF\DRES-CMIP-BB4CMIP7-2-1\atmos\mon\BC\gn\v20250612\BC_input4MIPs_emissions_CMIP_DRES-CMIP-BB4CMIP7-2-1_gn_190001-202312.nc"
# percentages: "D:\ESGF\DRES-CMIP-BB4CMIP7-2-1\atmos\mon\BCpercentageTEMF\gn\v20250612\BCpercentageTEMF_input4MIPs_emissions_CMIP_DRES-CMIP-BB4CMIP7-2-1_gn_175001-202312.nc"

from concordia.cmip7.CONSTANTS import GASES

# %%
# check later if we need all these imports 
import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt
import pandas_indexing as pix
import pandas as pd
import numpy as np
import os
import dask
from dask import delayed, compute
from dask.diagnostics import ProgressBar
from dask.utils import SerializableLock
import seaborn as sns

from concordia.cmip7 import utils as cmip7_utils

# %% [markdown]
# ### Unsmoothed data

# %%
gfed_sectors = ["AGRI", "BORF", "DEFO", "PEAT", "SAVA", "TEMF"]

perc_combinations = [f"{g}percentage{sec}" for g in GASES for sec in gfed_sectors]
totals_combinations = [f"{g}" for g in GASES]

# note: adjust the paths below as required by where you placed your downloaded data
perc_file_paths = [f"D:/ESGF/DRES-CMIP-BB4CMIP7-2-1/atmos/mon/{variable}/gn/v20250612/{variable}_input4MIPs_emissions_CMIP_DRES-CMIP-BB4CMIP7-2-1_gn_175001-202312.nc" for variable in perc_combinations]
totals_file_paths = [f"D:/ESGF/DRES-CMIP-BB4CMIP7-2-1/atmos/mon/{variable}/gn/v20250612/{variable}_input4MIPs_emissions_CMIP_DRES-CMIP-BB4CMIP7-2-1_gn_175001-202312.nc" for variable in totals_combinations]
print(len(totals_file_paths) + len(perc_file_paths))


# %% [markdown]
# ### First check that the percentages always add up to 100% (or nan)

# %%
# First check that the percentages always add up to 100% (or 0%?)

# load the datasets with chunking to enable dask
ds1 = xr.open_dataset(
        perc_file_paths[0],
        engine="netcdf4",
        chunks={"month": 12}  # Adjust chunk sizes as needed, e.g. {"month": 12}
    )
ds2 = xr.open_dataset(
        perc_file_paths[1],
        engine="netcdf4",
        chunks={"month": 12}
    )
ds3 = xr.open_dataset(
        perc_file_paths[2],
        engine="netcdf4",
        chunks={"month": 12}
    )
ds4 = xr.open_dataset(
        perc_file_paths[3],
        engine="netcdf4",
        chunks={"month": 12}
    )
ds5 = xr.open_dataset(
        perc_file_paths[4],
        engine="netcdf4",
        chunks={"month": 12}
    )
ds6 = xr.open_dataset(
        perc_file_paths[5],
        engine="netcdf4",
        chunks={"month": 12}
    )

# %%
# Get the variable names (first variable in each dataset, excluding coordinates)
var1 = list(ds1.data_vars)[3]  # Should be BCpercentageAGRI
var2 = list(ds2.data_vars)[3]  # Should be BCpercentageBORF
var3 = list(ds3.data_vars)[3]  # Should be BCpercentageDEFO
var4 = list(ds4.data_vars)[3]  # Should be BCpercentagePEAT
var5 = list(ds5.data_vars)[3]  # Should be BCpercentageSAVA
var6 = list(ds6.data_vars)[3]  # Should be BCpercentageTEMF

print(f"Variables: {var1}, {var2}, {var3}, {var4}, {var5}, {var6}")

# %%
# Sum using dask arrays - this creates a computation graph, doesn't compute yet
total_percentages = (ds1[var1] + ds2[var2] + ds3[var3] + ds4[var4] + ds5[var5] + ds6[var6])

# time_filter = (ds1.time.dt.year >= 1994) & (ds1.time.dt.year <= 2023)
time_filter = (ds1.time.dt.year >= 2014) & (ds1.time.dt.year <= 2023)
total_percentages = total_percentages.isel(time=time_filter)

print(f"Total percentages data type: {type(total_percentages.data)}")  # Should show dask array

# %%
# Use dask operations to check values efficiently
# First, let's check a small sample to see if the computation works, just for the last data point (decemebr 2023)
sample = total_percentages.isel(time=ds1.time.dt.month == 12)
sample_computed = sample.compute()

print(f"Sample values shape: {sample_computed.shape}")
print(f"Sample unique values: {np.unique(sample_computed.values)}")

# checked for BC, 2023-12:
# Sample values shape: (274, 720, 1440)
# Sample unique values: [  0.       99.99999 100.      100.00001       nan]
# thus: mostly correct

# %%
# For the full check, use dask delayed computation
@delayed
def check_percentages(data_array):
    values = data_array.values
    mask_not_nan = ~np.isnan(values)
    non_nan_values = values[mask_not_nan]
    
    unique_vals = np.unique(non_nan_values) if len(non_nan_values) > 0 else []
    
    n_nan = np.sum(np.isnan(values))
    n_not_nan = np.sum(mask_not_nan)

    # Count values that are not 0 or 100 (allowing for floating point precision)
    not_zero_or_hundred = np.sum((
        non_nan_values != 0
    ) & (
        ~np.isclose(non_nan_values, 100, rtol=1e-10, atol=1e-10)
    )) if len(non_nan_values) > 0 else 0

    share_of_issues = (not_zero_or_hundred) / (n_nan + n_not_nan)
    
    return share_of_issues, unique_vals, n_nan, n_not_nan

# Compute in chunks with progress bar
with ProgressBar():
    result = check_percentages(total_percentages).compute()

share_of_issues, unique_vals, n_nan, n_not_nan = result
print(f"Share of potentially problematic observations: {share_of_issues}")
print(f"Unique non-NaN values: {unique_vals}")
print(f"Number of NaN values: {n_nan}")
print(f"Number of non-NaN values: {n_not_nan}")

# For 2014-2023 it gives:
# [########################################] | 100% Completed | 40.13 s
# Share of potentially problematic observations: 0.0002432886445473251
# Unique non-NaN values: [  0.        99.999985  99.99999  100.       100.00001  100.000015]
# Number of NaN values: 93819720
# Number of non-NaN values: 30596280
# For 1994-2023 it gives:
# [########################################] | 100% Completed | 50.53 s
# All non-NaN values are 1.0: False
# Unique non-NaN values: [  0.        99.999985  99.99999  100.       100.00001  100.000015]
# Number of NaN values: 281459160
# Number of non-NaN values: 91788840
# For the full timeseries 1750-2023 it gives:
# [########################################] | 100% Completed | 342.73 s
# All non-NaN values are 1.0: False
# Unique non-NaN values: [  0.        99.999985  99.99999  100.       100.00001  100.000015]
# Number of NaN values: -1724306968 # note that this negative value is probably due to large data size, integer overflow
# Number of non-NaN values: 838338072




# %%
print("the end")
# %%
