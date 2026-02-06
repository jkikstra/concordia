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


# %%
# Helper: normalize a time_slice argument into an xarray-friendly slice spanning full years
def _normalize_time_slice(time_slice):
    """
    Accepts:
      - int: a single year, e.g., 2023
      - list/tuple of two years: [min_year, max_year]
      - slice: passed through unchanged

    Returns a pandas-compatible slice covering full years: 'YYYY-01-01'..'YYYY-12-31'.
    """
    if isinstance(time_slice, int):
        y = int(time_slice)
        return slice(f"{y}-01-01", f"{y}-12-31")
    if isinstance(time_slice, (list, tuple)):
        if len(time_slice) != 2:
            raise ValueError("time_slice as list/tuple must have exactly two years: [min_year, max_year]")
        y0, y1 = sorted(int(y) for y in time_slice)
        return slice(f"{y0}-01-01", f"{y1}-12-31")
    if isinstance(time_slice, slice):
        return time_slice
    raise ValueError(
        "time_slice must be an int (year), a [min,max] list/tuple of years, or a slice"
    )

