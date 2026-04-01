# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
from pathlib import Path
import pandas as pd
import numpy as np
from concordia.settings import Settings

# %%
SETTINGS_FILE: str = "config_cmip7_v0-4-0.yaml" # for second ESGF version
VERSION_ESGF: str = "1-1-0" # for second ESGF version

# Which scenario to run from the markers
marker_to_run: str = "vl" # options: h, hl, m, ml, l, ln, vl

# What folder to save this run in
GRIDDING_VERSION: str | None = f"{marker_to_run}_{VERSION_ESGF}_ext"

# %%
# Get the directory of the current file, works in both script and notebook contexts
# When running through papermill, we need to find the original notebook location
try:
    # Try to get __file__ (works when running as script)
    HERE = Path(__file__).parent
    # Also check if HERE resolved to just current directory, which indicates path resolution failed
    if str(HERE) == "." or HERE == Path("."):
        raise NameError("HERE resolved to current directory, using fallback")
except NameError:
    # When running in notebook/papermill, use a more robust approach
    # Find the concordia repository root and navigate to notebooks/cmip7
    current_path = Path.cwd()
    
    # Look for the concordia root directory (contains pyproject.toml)
    concordia_root = None
    for parent in [current_path] + list(current_path.parents):
        if (parent / "pyproject.toml").exists() and (parent / "src" / "concordia").exists():
            concordia_root = parent
            break
    
    if concordia_root is None:
        raise RuntimeError("Could not find concordia repository root")
    
    HERE = concordia_root / "notebooks" / "cmip7"

settings = Settings.from_config(version=GRIDDING_VERSION,
                                local_config_path=Path(HERE,
                                                       SETTINGS_FILE))

# %%
settings.scenario_path

# %% [markdown]
# ## GDP extension

# %%
# New; updated SSP data from CMIP7 era (downloaded from: http://files.ece.iiasa.ac.at/ssp/downloads/ssp_basic_drivers_release_3.2.beta_full.xlsx, and then selected only the GDP|PPP variable)
gdp_new = pd.read_csv(
        settings.scenario_path / "ssp_basic_drivers_release_3.2.beta_full_gdp.csv",
        index_col=list(range(5)),
    )

# %%
EXT_PROXY_YEARS = [str(y) for y in range(2105, 2501, 5)]

# %%
gdp_new[EXT_PROXY_YEARS] = np.repeat(gdp_new[["2100"]].to_numpy(), len(EXT_PROXY_YEARS), axis=1)

# %%
gdp_new.to_csv("/Users/hoegner/Projects/CMIP7/input/scenarios/ssp_basic_drivers_release_3.2.beta_full_gdp-extensions.csv")
