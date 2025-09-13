#!/usr/bin/env python
# %% [markdown]
# # Overview
#
# Some information that is common to code processing CEDS and BB4CMIP data 
# 

# %%
from pathlib import Path


# %%
from concordia.settings import Settings

# %%
def get_settings(base_path: Path, 
                 file: str = "config_cmip7_v0_2.yaml"):
    settings = Settings.from_config(
        file, 
        base_path=base_path,
        version=None
    )
    return settings

# %%
# Constants

dim_order = ["gas", "sector", "level", "year", "month", "lat", "lon"]
