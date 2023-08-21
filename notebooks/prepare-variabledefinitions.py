# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.6
#   kernelspec:
#     display_name: concordia
#     language: python
#     name: python3
# ---

# %%
from pathlib import Path

import pandas as pd
from pandas_indexing import extractlevel, isin


# %%
base_path = Path(
    "/Users/coroa/Library/CloudStorage/OneDrive-SharedLibraries-IIASA/RESCUE - WP 1/data"
)
data_path = Path("../data")

# %% [markdown]
# ## Variable definition files
#
# The variable definition file is a CSV or yaml file that needs to contain the `variable`-name, its `sector`, `gas` components and whether it is expected `global` (or regional instead).
#
# Here we generate one based on the cmip6 historical data we have that could be used as a basis but we would want to finetune this by hand.

# %% [markdown]
# ### Create new variable definition files from historical data

# %%
variables = (
    pd.read_csv(
        base_path / "historical/cmip6/history.csv",
        usecols=[3, 4],
    )
    .rename(columns=str.lower)
    .drop_duplicates()
    .set_index("variable")
)

# %%
variabledefs = extractlevel(
    variables, "CEDS+|9+ Sectors|Emissions|{gas}|{sector}|Unharmonized"
)

# %% [markdown]
# Mark variables as global or regional:
#
# Global variables are:
# - Variants of fluorinated gases,
# - N2O, as well as
# - Aggriculture and LUC, and
# - Aircraft and International Shipping

# %%
variabledefs["global"] = False
variabledefs.loc[
    isin(gas=["F-Gases", "N2O", "C2F6", "HFC", "SF6", "CF4"]), "global"
] = True
variabledefs.loc[
    isin(
        sector=["Aggregate - Agriculture and LUC", "Aircraft", "International Shipping"]
    ),
    "global",
] = True

# %%
variabledefs.to_csv(data_path / "variabledefs.csv")

# %%
