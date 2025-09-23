#!/usr/bin/env python
# %% [markdown]
# # Overview
#
# This should create a proxy based on BB4CMIP7
# 


# %% [markdown]
# ### Steps happening in this notebook, which starts from the PNNL server .Rd files
# 1. ...
# %%
import itertools
from functools import lru_cache

# %%
import dask
import dask.array
import numpy as np
import pandas as pd
import ptolemy as pt
import pyogrio as pio
import pyreadr
import xarray as xr
from pathlib import Path
from typing import Callable, Dict, Tuple
import os

# %%
from concordia.settings import Settings

from concordia.cmip7.utils import read_r_variable, read_r_to_da
from concordia.cmip7.utils_rpy2 import save_da_as_rd

# %%
import concordia.cmip7.utils_futureproxy_ceds_bb4cmip as uprox
from concordia.cmip7.utils_futureproxy_ceds_bb4cmip import dim_order
from concordia.cmip7.CONSTANTS import CONFIG

# %%
config_file_name = CONFIG

# %%
try:
    # when running the script from a terminal or otherwise
    cmip7_dir = Path(__file__).resolve()
    settings = uprox.get_settings(base_path=cmip7_dir, file = config_file_name)
except (FileNotFoundError, NameError):
    try:
        # when running the script from a terminal or otherwise
        cmip7_dir = Path(__file__).resolve().parent
        settings = uprox.get_settings(base_path=cmip7_dir, file = config_file_name)
    except (FileNotFoundError, NameError):
        # Fallback for interactive/Jupyter mode, where 'file location' does not exist
        cmip7_dir = Path().resolve()  # one up
        settings = uprox.get_settings(base_path=cmip7_dir, file = config_file_name)

# %%
sector_mapping = {
    # "anthro": ["AGR", "ENE", "IND", "TRA", "RCO", "SLV", "WST"],
    "openburning": ["AWB", "FRTB", "GRSB", "PEAT"],
    # "aircraft": ["AIR"],  # NB: proxy issues here, need to address in hackathon
    # "shipping": ["SHP"],  # NB: this was not split out originally, but 1) we have proxy issues; 2) we will be redoing this in this project
}

# %%
all_sectors = [sector for sublist in sector_mapping.values() for sector in sublist]
all_sectors

# %%
template_file = (
    settings.gridding_path
    / "example_files" / "GCAM4-SSP4-34-SPA4-V25-Harmonized-DB-Sulfur-em-aircraft-anthro_input4MIPs_emissions_CMIP_IAMC-V1_gn_201501-210012.nc"
)
template = xr.open_dataset(template_file)

# %%
# NOTE: to still be worked on. 
# NOTE: base on deleted "generate_ceds_based_future_proxy_netcdfs-cmip7.py" or the rescue "generate_ceds_proxy_netcdfs.py", for `full_process("openburning")`
# # uprox.full_process("openburning")