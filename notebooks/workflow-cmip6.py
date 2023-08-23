# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.6
#   kernelspec:
#     display_name: aneris2
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import re
from pathlib import Path

import pandas as pd
import xarray as xr
import pandas_indexing.accessors
from aneris.harmonize import Harmonizer
from aneris.downscaling import Downscaler
from aneris.grid import Gridder
from pandas import DataFrame
from pandas_indexing import isin, semijoin

from concordia import VariableDefinitions, RegionMapping, combine_countries

# %%
# Potentially better gridding performance??
# from dask.distributed import Client
# client = Client()

# %% [markdown]
# # Read model and historic data including overrides

# %%
base_path = Path(
    "/Users/coroa/Library/CloudStorage/OneDrive-SharedLibraries-IIASA/RESCUE - WP 1/data"
)
data_path = Path("../data")
base_year = 2015 # in which year scenario data should be harmonized to historical data
country_combinations = {
    "sdn_ssd": ["ssd", "sdn"],
    "isr_pse": ["isr", "pse"],
    "srb_ksv": ["srb", "srb (kosovo)"]
}

# %% [markdown]
# ## Variable definition files
#
# The variable definition file is a CSV or yaml file that needs to contain the `variable`-name, its `sector`, `gas` components and whether it is expected `global` (or regional instead).
#
# Here we generate one based on the cmip6 historical data we have that could be used as a basis but we would want to finetune this by hand.

# %%
variabledefs = VariableDefinitions.from_csv(data_path / "variabledefs-cmip6.csv")
variabledefs.data.head()

# %% [markdown]
# ## RegionMapping helps reading in a region definition file

# %%
regionmapping = RegionMapping.from_regiondef(
    base_path / "historical/cmip6/remind_region_mapping.csv"
)
regionmapping.data = combine_countries(regionmapping.data, **country_combinations, agg_func="first")

# %% [markdown]
# ## Model and historic data read in
#
# Can be read in and prepared using `read_iamc` or the `variabledefs`

# %%
hist = (
    pd.read_csv(base_path / "historical/cmip6/history.csv", index_col=list(range(5)))
    .rename_axis(index={"Region": "Country"})
    .pipe(variabledefs.load_data, levels=["country", "gas", "sector", "unit"])
    .pipe(combine_countries, **country_combinations)
)
hist.head()

# %%
model = (
    pd.read_excel(
        base_path / "iam_files/cmip6/REMIND-MAGPIE_SSP5-34-OS/ssp5-34-os.xlsx",
        sheet_name="data",
        index_col=list(range(5)),
    )
    .rename(index={"Mt CO2-equiv/yr": "Mt CO2-eq/yr"}, level="Unit")
    .pipe(
        variabledefs.load_data,
        levels=["model", "scenario", "region", "gas", "sector", "unit"],
        ignore_missing=True,
    )
)
model.head()

# %%
harm_overrides = (
    pd.read_excel(
        base_path / "iam_files/cmip6/REMIND-MAGPIE_SSP5-34-OS/ssp5-34-os.xlsx",
        sheet_name="harmonization",
        index_col=list(range(4)),
        usecols=list(range(5)),
    )
    .rename_axis(index=str.lower)
    .rename(columns=str.lower)
    .pipe(
        variabledefs.load_data,
        ignore_missing=True,
        levels=["region", "gas", "sector"],
        timeseries=False,
    )
    .method
)

harm_overrides.head()

# %%
model.head()

# %% [markdown]
# # Harmonization
#
# ## Preparation of input data

# %%
hist_agg = pd.concat(
    [
        hist.pix.semijoin(variabledefs.index_regional, how="inner").pipe(
            regionmapping.aggregate
        ),
        hist.pix.semijoin(variabledefs.index_global, how="inner")
        .loc[isin(country="World")]
        .rename_axis(index={"country": "region"}),
    ]
)

# %%
model_agg = pd.concat(
    [
        model.pix.semijoin(variabledefs.index_regional, how="inner").loc[
            isin(region=regionmapping.data.unique())
        ],
        model.pix.semijoin(variabledefs.index_global, how="inner").loc[
            isin(region="World")
        ],
    ]
)

# %% [markdown]
# ## Harmonize all model, scenarios combinations

# %%
harmonized = []
for m, s in model.index.pix.project(["model", "scenario"]).unique():
    scen = model_agg.loc[isin(model=m, scenario=s)].droplevel(["model", "scenario"])
    h = Harmonizer(
        scen, hist_agg.pix.semijoin(scen.index, how="right"), harm_idx=scen.index.names
    )
    result = h.harmonize(year=base_year, overrides=harm_overrides)
    harmonized.append(result.pix.assign(model=m, scenario=s))
harmonized = pd.concat(harmonized)

# TODO harmonization casts columns to str!!
harmonized = harmonized.rename(columns=int)
harmonized.head()

# %% [markdown]
# # Downscaling

# %% [markdown]
# ## Prepare GDP proxy
#
# Read in different GDP scenarios for SSP1 to SSP5 from SSP DB.

# %%
gdp = (
    pd.read_csv(
        base_path / "historical" / "SspDb_country_data_2013-06-12.csv",
        index_col=list(range(5)),
    )
    .rename_axis(index=str.lower)
    .loc[
        isin(
            model="OECD Env-Growth",
            scenario=[f"SSP{n+1}_v9_130325" for n in range(5)],
            variable="GDP|PPP",
        )
    ]
    .dropna(how="all", axis=1)
    .rename_axis(index={"scenario": "ssp", "region": "country"})
    .rename(index=str.lower, level="country")
    .rename(columns=int)
    .pix.project(["ssp", "country"])
    .pipe(combine_countries, **country_combinations)
)
gdp.head()

# %% [markdown]
# Determine likely SSP for each harmonized pathway from scenario string and create proxy data aligned with pathways

# %%
SSP_per_pathway = (
    harmonized.index.pix.project(["model", "scenario"])
    .unique()
    .to_frame()
    .scenario.str.extract("(SSP[1-5])")[0]
    .fillna("SSP2")
)
gdp = semijoin(
    gdp,
    SSP_per_pathway.index.pix.assign(ssp=SSP_per_pathway + "_v9_130325"),
    how="right",
).pix.project(["model", "scenario", "country"])
gdp.head()

# %%
downscaler = Downscaler(
    harmonized.pix.semijoin(variabledefs.index_regional, how="inner"),
    hist.pix.semijoin(variabledefs.index_regional, how="inner"),
    base_year,
    regionmapping.data,
    gdp=gdp,
)
results = downscaler.downscale()

# %%
downscaler.methods().value_counts()

# %%
results

# %% [markdown]
# # Gridding

# %%
idxr = xr.open_dataarray(base_path / "gridding_process_files" / "iso_mask.nc", chunks={"iso": 20}).rename({"iso": "country"})

# %%
proxy_dir = base_path / "gridding_process_files" / "proxy_rasters"
proxy_cfg = pd.concat(
    [
        # DataFrame(
        #     {
        #         "path": proxy_dir.glob("aircraft_*.nc"),
        #         "name": "em-AIR-anthro",
        #         "separate_shares": False,
        #     }
        # ),
        DataFrame(
            {
                "path": proxy_dir.glob("anthro_*.nc"),
                "name": "em-anthro",
                "separate_shares": False,
            }
        ),
        # DataFrame(
        #     {
        #         "path": proxy_dir.glob("openburning_*.nc"),
        #         "name": "em-openburning",
        #         "separate_shares": True,
        #     }
        # ),
        DataFrame({"path": proxy_dir.glob("shipping_*.nc"), "name": ..., "template": ..., "separate_shares": False})
    ]
).assign(
    name=lambda df: df.path.map(lambda p: p.stem.split("_")[1]) + "-" + df.name,
    template="{name}_emissions_{model}-{scenario}_201501-210012",
)
proxy_cfg.head()

# %%
sector_mapping = {
    "AIR": "Aircraft",
    "SHP": "International Shipping",
    "AWB": "Agricultural Waste Burning",
    "AGR": "Agriculture",
    "ENE": "Energy Sector",
    "FRTB": "Forest Burning",
    "GRSB": "Grassland Burning",
    "IND": "Industrial Sector",
    "PEAT": "Peat Burning",
    "RCO": "Residential Commercial Other",
    "SLV": "Solvents Production and Application",
    "TRA": "Transportation Sector",
    "WST": "Waste",
}

# %%
kg_per_mt = 1e9
s_per_yr = 365 * 24 * 60 * 60
results = (
    results.rename(index=lambda s: re.sub("Mt (.*)/yr", r"kg \1/s", s), level="unit")
    * kg_per_mt
    / s_per_yr
)

# %%
results = results.droplevel("region")
results.head()

# %%
gridder = Gridder(results, idxr, proxy_cfg, index_mappings=dict(sector=sector_mapping), output_dir="../results")

# %%
gridder.grid(skip_check=True)
