# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import logging
import re
from pathlib import Path

import pandas as pd
import pycountry
import xarray as xr
import yaml
from concordia import (
    CondordiaMagics,
    RegionMapping,
    VariableDefinitions,
    combine_countries,
)
from IPython import get_ipython
from pandas import DataFrame
from pandas_indexing import concat, isin, ismatch, semijoin
from pandas_indexing.units import set_openscm_registry_as_default

from aneris import logger
from aneris.downscaling import Downscaler
from aneris.grid import Gridder
from aneris.harmonize import Harmonizer


# %%
# %env HDF5_USE_FILE_LOCKING=FALSE

# %%
import os
os.environ["HDF5_USE_FILE_LOCKING"]

# %%
version = "2023-08-28"

# %%
fh = logging.FileHandler(f"debug_{version}.log", mode="w")
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
fh.setFormatter(formatter)
logger().addHandler(fh)

# %%
get_ipython().register_magics(CondordiaMagics)

# %%
ur = set_openscm_registry_as_default()

# %% [markdown]
# # Set which parts of the workflow you would like to execute and how the file names should be tagged

# %%
execute_harmonization = False
execute_downscaling = False
execute_gridding = True

# %% [markdown]
# # Read model and historic data including overrides
#
# To run this code, create a file called `config.yaml` in this directory pointing to the correct data file locations, e.g.,
#
# ```
# # config.yaml
# base_path: "/Users/{macuser}/Library/CloudStorage/OneDrive-SharedLibraries-IIASA/RESCUE - WP 1/data"
# data_path: "../data"
# country_combinations:
#   sdn_ssd: ["ssd", "sdn"]
#   isr_pse: ["isr", "pse"]
#   srb_ksv: ["srb", "srb (kosovo)"]
# ```
#

# %%
with open("config.yaml") as stream:
    config = yaml.safe_load(stream)

# %%
base_path = Path(config["base_path"]).expanduser()
data_path = Path(config["data_path"]).expanduser()
out_path = Path(config["out_path"]).expanduser() / version
out_path.mkdir(parents=True, exist_ok=True)

base_year = 2020  # in which year scenario data should be harmonized to historical data
country_combinations = config["country_combinations"]

# %%
base_path, os.path.exists(base_path)

# %% [markdown]
# ## Variable definition files
#
# The variable definition file is a CSV or yaml file that needs to contain the `variable`-name, its `sector`, `gas` components and whether it is expected `global` (or regional instead).
#
# Here we generate one based on the cmip6 historical data we have that could be used as a basis but we would want to finetune this by hand.
#

# %%
variabledefs = VariableDefinitions.from_csv(data_path / "variabledefs-rescue.csv")
variabledefs.data.tail()

# %% [markdown]
# ## RegionMapping helps reading in a region definition file
#

# %%
regionmapping = RegionMapping.from_regiondef(
    base_path / "iam_files/rescue/regionmappingH12.csv",
    country_column="CountryCode",
    region_column="RegionCode",
    sep=";",
)
regionmapping.data = combine_countries(
    regionmapping.data, **country_combinations, agg_func="last"
)

# %%
regionmapping.data.unique()

# %% [markdown]
# ## Model and historic data read in
#
# Can be read in and prepared using `read_iamc` or the `variabledefs`
#

# %%
hist_ceds = (
    pd.read_csv(
        base_path / "historical/rescue/ceds_2017_extended.csv", index_col=list(range(4))
    )
    .rename(index={"NMVOC": "VOC", "SO2": "Sulfur"}, level="gas")
    .rename(index={"Mt NMVOC/yr": "Mt VOC/yr"}, level="unit")
    .rename(columns=int)
    .pix.format(variable="CEDS+|9+ Sectors|Emissions|{gas}|{sector}", drop=True)
    .pix.assign(model="History", scenario="CEDS")
)

# %%
hist_global = (
    pd.read_excel(
        base_path / "historical/rescue/global_trajectories.xlsx",
        index_col=list(range(5)),
    )
    .rename_axis(index=str.lower)
    .rename_axis(index={"region": "country"})
    .rename(index=lambda s: s.removesuffix("|Unharmonized"), level="variable")
)

# %%
hist_gfed = pd.read_csv(
    base_path / "historical/rescue/gfed/GFED2015_extended.csv", index_col=list(range(5))
).rename(columns=int)

# %%
hist = (
    concat([hist_ceds, hist_global, hist_gfed])
    .droplevel(["model", "scenario"])
    .pipe(combine_countries, **country_combinations)
    .pipe(
        variabledefs.load_data,
        extend_missing=True,
        levels=["country", "gas", "sector", "unit"],
    )
)
hist.head()

# %%
with ur.context("AR4GWP100"):
    model = (
        pd.read_csv(
            base_path
            / "iam_files/rescue/REMIND-MAgPIE-CEDS-RESCUE-Tier1-Extension-2023-07-27.csv",
            index_col=list(range(5)),
            sep=";",
        )
        .drop(["Unnamed: 21"], axis=1)
        .rename(
            index={
                "Mt CO2-equiv/yr": "Mt CO2eq/yr",
                "Mt NOX/yr": "Mt NOx/yr",
                "kt HFC134a-equiv/yr": "kt HFC134a/yr",
            },
            level="Unit",
        )
        .pix.convert_unit({"kt HFC134a/yr": "Mt CO2eq/yr"}, level="Unit")
        .rename(index=lambda s: s.removesuffix("|Total"), level="Variable")
        .pipe(
            variabledefs.load_data,
            extend_missing=True,
            levels=["model", "scenario", "region", "gas", "sector", "unit"],
        )
    )
model.pix

# %%
harm_overrides = pd.read_excel(
    base_path / "iam_files" / "rescue" / "harmonization_overrides.xlsx",
    index_col=list(range(3)),
).method
harm_overrides

# %%
hist_available = hist.pix.unique(["gas", "sector"])

# %%
model.pix.unique(["gas", "sector"]).difference(hist_available)

# %% [markdown]
# # Harmonization
#
# ## Preparation of input data
#

# %%
hist_agg = pd.concat(
    [
        hist.pix.semijoin(variabledefs.index_regional, how="inner").pipe(
            regionmapping.aggregate, dropna=True
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
).pix.semijoin(hist_agg.index, how="inner")

# %% [markdown]
# ## Harmonize all model, scenarios combinations
#

# %%
luc_sectors = [
    "Agricultural Waste Burning",
    "Grassland Burning",
    "Forest Burning",
    "Peat Burning",
    "Agriculture",
    "Aggregate - Agriculture and LUC",
]


# %%
def harmonize(model_agg, hist_agg, config, overrides):
    harmonized = []
    for m, s in model_agg.index.pix.unique(["model", "scenario"]):
        scen = model_agg.loc[isin(model=m, scenario=s)].droplevel(["model", "scenario"])
        h = Harmonizer(
            scen,
            hist_agg.pix.semijoin(scen.index, how="right").loc[:, 2000:],
            harm_idx=scen.index.names,
            config=config,
        )
        result = h.harmonize(
            year=base_year, overrides=None if overrides.empty else overrides
        ).sort_index()
        methods = h.methods(year=base_year)
        result = result.pix.assign(
            method=methods.pix.semijoin(result.index, how="right")
        )
        harmonized.append(result.pix.assign(model=m, scenario=s))
    harmonized = pd.concat(harmonized)

    return harmonized


# %%
harmonized_path = out_path / f"harmonized-only-{version}.csv"

# %%
# %%execute_or_lazy_load execute_harmonization harmonized = pd.read_csv(harmonized_path, index_col=list(range(7))).rename(columns=int)
is_luc = isin(sector=luc_sectors)
harmonized = concat(
    [
        harmonize(
            model_agg.loc[is_luc],
            hist_agg.loc[is_luc],
            config=dict(),
            overrides=harm_overrides.loc[is_luc],
        ),
        harmonize(
            model_agg.loc[~is_luc],
            hist_agg.loc[~is_luc],
            config=dict(default_luc_method="reduce_ratio_2080"),
            overrides=harm_overrides.loc[~is_luc],
        ),
    ]
)
harmonized.to_csv(harmonized_path)

# %%
harmonized.loc[(harmonized < 0).any(axis=1)].loc[
    ~ismatch(sector="CDR*") & ~isin(method="aggregate")
]


# %%
def make_totals(df):
    original_levels = df.index.names
    if "method" in original_levels:  # need to process harm
        df = df.droplevel("method")
    level = df.index.names
    ret = concat(
        [
            (
                df.loc[~isin(region="World")]  # don"t count aviation
                .groupby(level.difference(["region"]))
                .agg(lambda s: s.sum(skipna=False))
                .pix.assign(region="World", order=level)
            ),
            (
                df.loc[~(isin(sector="Total") | isin(region="World"))]
                .groupby(level.difference(["sector"]))
                .agg(lambda s: s.sum(skipna=False))
                .pix.assign(sector="Total", order=level)
            ),
            (
                df.loc[~isin(sector="Total")]  # don"t count global totals
                .groupby(level.difference(["region", "sector"]))
                .agg(lambda s: s.sum(skipna=False))
                .pix.assign(region="World", sector="Total", order=level)
            ),
        ]
    )
    if "method" in original_levels:  # need to process harm
        ret = ret.pix.assign(method="aggregate", order=original_levels)
    return ret


model_agg = concat([model_agg, make_totals(model_agg)])
hist_agg = concat([hist_agg, make_totals(hist_agg)])
harmonized = concat([harmonized, make_totals(harmonized)])

# %%
data = concat(
    [
        model_agg.pix.format(
            variable="Emissions|{gas}|{sector}|Unharmonized", drop=True
        ),
        harmonized.pix.format(
            variable="Emissions|{gas}|{sector}|Harmonized|{method}", drop=True
        ),
        hist_agg.loc[:, 1990:].pix.format(
            model="Historic",
            scenario="Synthetic (GFED/CEDS/Global)",
            variable="Emissions|{gas}|{sector}",
            drop=True,
        ),
    ],
    order=["model", "scenario", "region", "variable", "unit"],
).sort_index(axis=1)
data.to_csv(out_path / f"harmonization-{version}.csv")

# %%
hfc_distribution = (
    pd.read_csv(
        base_path
        / "harmonization_postprocessing"
        / "rescue"
        / "rescue_hfc_scenario.csv",
        index_col=0,
    )
    .rename_axis("hfc")
    .rename(columns=int)
)


def split_hfc(df):
    return concat(
        [
            df.loc[~isin(gas="HFC")],
            df.pix.multiply(hfc_distribution.pix.assign(gas="HFC"), join="inner")
            .droplevel("gas")
            .rename_axis(index={"hfc": "gas"}),
        ]
    )


data = concat(
    [
        split_hfc(model_agg).pix.format(
            variable="Emissions|{gas}|{sector}|Unharmonized", drop=True
        ),
        split_hfc(harmonized).pix.format(
            variable="Emissions|{gas}|{sector}|Harmonized|{method}", drop=True
        ),
        split_hfc(hist_agg.loc[:, 1990:]).pix.format(
            model="Historic",
            scenario="Synthetic (GFED/CEDS/Global)",
            variable="Emissions|{gas}|{sector}",
            drop=True,
        ),
    ],
    order=["model", "scenario", "region", "variable", "unit"],
).sort_index(axis=1)
data.to_csv(out_path / f"harmonization-{version}-splithfc.csv")

# %% [markdown]
# # Downscaling
#

# %% [markdown]
# ## Prepare GDP proxy
#
# Read in different GDP scenarios for SSP1 to SSP5 from SSP DB.
#

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
#

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
downscaled_path = out_path / f"downscaled-only-{version}.csv"


# %% [markdown]
# The regionmapping has several countries which are not part of the original gdp data, therefore we remove those countries (and effectively split the regional emissions among fewer countries). The missing countries are:


# %%
def iso_to_name(x):
    cntry = pycountry.countries.get(alpha_3=x.upper())
    return cntry.name if cntry is not None else x


regionmapping.data.index.difference(gdp.pix.unique("country")).map(iso_to_name)

# %%
# Remove countries from regionmapping that the GDP proxy does not have
regionmapping_trimmed = RegionMapping(
    regionmapping.data.loc[isin(country=gdp.pix.unique("country"))]
)

# %%
# %%execute_or_lazy_load execute_downscaling downscaled = pd.read_csv(downscaled_path, index_col=list(range(8))).rename(columns=int)
downscaler = Downscaler(
    harmonized.pix.semijoin(variabledefs.index_regional, how="inner")
    .loc[~isin(region="World")]
    .droplevel("method"),
    hist.pix.semijoin(variabledefs.index_regional, how="inner"),
    base_year,
    regionmapping_trimmed.data,
    luc_sectors=luc_sectors,
    gdp=gdp,
)
methods = downscaler.methods()
downscaled = downscaler.downscale(methods).sort_index()
downscaled = downscaled.pix.assign(
    method=methods.pix.semijoin(downscaled.index, how="right")
)
downscaled.to_csv(downscaled_path)

# %%
downscaled.head()

# %% [markdown]
# # Gridding
#
# ## Configuration and Set Up

# %%
proxy_dir = base_path / "gridding_process_files" / "proxy_rasters"
proxy_cfg = pd.concat(
    [
        DataFrame(
            {
                "path": proxy_dir.glob("aircraft_*.nc"),
                "name": "em-AIR-anthro",
                "global_only": True,
            }
        ),
        DataFrame(
            {
                "path": proxy_dir.glob("shipping_*.nc"),
                "name": "em-SHP-anthro",
                "global_only": True,
            }
        ),
        DataFrame(
            {
                "path": proxy_dir.glob("anthro_*.nc"),
                "name": "em-anthro",
                "global_only": False,
            }
        ),
        DataFrame(
            {
                "path": proxy_dir.glob("openburning_*.nc"),
                "name": "em-openburning",
                "global_only": False,
            }
        ),
        DataFrame(
            {
                "path": proxy_dir.glob("CDR*.nc"),
                "name": "em-removal",
                "global_only": False,
            }
        ),
    ]
).assign(
    name=lambda df: df.path.map(lambda p: p.stem.split("_")[1]) + "-" + df.name,
    template="{name}_emissions_{model}-{scenario}_201501-210012",
)
_PROXY_CFG = proxy_cfg.copy()  # for debugging help not to overwrite name
proxy_cfg.tail()

# %% [raw]
# proxy_cfg = pd.concat([
#     proxy_cfg,
#     DataFrame(
#         {
#             "path": proxy_dir.glob("CDR_CO2.nc"),
#             "name": "CO2-em-removal",
#             "separate_shares": False,
#             "as_flux": True,
#             "template": "CO2-em-anthro_emissions_{model}-{scenario}_201501-210012"
#         }
#     ),
# ], ignore_index=True)

# %% [raw]
# sector_mapping = {
#     "Aircraft": "AIR",
#     "International Shipping": "SHP",
#     "Agricultural Waste Burning": "AWB",
#     "Agriculture": "AGR",
#     "Energy Sector": "ENE",
#     "Forest Burning": "FRTB",
#     "Grassland Burning": "GRSB",
#     "Industrial Sector": "IND",
#     "Peat Burning": "PEAT",
#     "Residential Commercial Other": "RCO",
#     "Solvents Production and Application": "SLV",
#     "Transportation Sector": "TRA",
#     "Waste": "WST",
# #    "CDR Afforestation",
# #    "CDR BECCS",
#     "CDR DACCS": "DAC_CDR",
# #    "CDR EW",
#     "CDR Industry": "IND_CDR",
#     "CDR OAE": "OAE_CDR",
#     "Emissions OAE": "OAE",
# }
#

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
    #    "CDR Afforestation",
    #    "CDR BECCS",
    "DAC_CDR": "CDR DACCS",
    #    "CDR EW",
    "IND_CDR": "CDR Industry",
    "OAE_CDR": "CDR OAE",
    "OAE": "Emissions OAE",
}

# %%
harmonized.pix.unique("sector").difference(
    sector_mapping.values()
)  # sectors not included in gridding

# %% [markdown]
# ## Getting Countries Right

# %%
# Starts a locally running dask distributed scheduler (and registers it as
# default) that comes with a web-frontend for introspection on
# http://localhost:8787, which seems to have slightly better scheduling
# characteristics
from dask.distributed import Client


client = Client()

# %%
idxr = xr.open_dataarray(
    base_path / "gridding_process_files" / "ssp_comb_iso_mask.nc", chunks={"iso": 3}
).rename({"iso": "country"})

# %%
# These are the countries in our downscaled data not in the index raster
iso_diff = downscaled.pix.unique("country").difference(idxr["country"])
print(iso_diff)
assert iso_diff.empty

# %% [markdown]
# ## Preparing Data for Gridding

# %%
data_for_gridding = downscaled.droplevel("region")

# %% [markdown]
# We also generate data for sectors only with global resolution, so we add those back in

# %%
global_sectors = variabledefs.data[
    variabledefs.data["global"] & variabledefs.data["gridded"]
].pix.unique("sector")
global_data = harmonized.loc[isin(sector=global_sectors, region="World")].rename_axis(
    index={"region": "country"}
)
data_for_gridding = concat([data_for_gridding, global_data])

# %% [markdown]
# Because proxy data is only provided at decadal values, we strip 5-year values out of the dataset

# %%
data_for_gridding = data_for_gridding[range(2020, 2101, 10)]
data_for_gridding.head()

# %% [markdown]
# Because we not all CDR options are currently supported, we remove those which are not

# %%
not_supported_sectors = [
    "CDR Afforestation",
    "CDR BECCS",
    "CDR EW",
]
data_for_gridding = data_for_gridding.loc[~isin(sector=not_supported_sectors)]

# %% [markdown]
# And finally we perform unit conversions so that gridded data is in terms of kg of emission species per second. Fluxes can then be computed by dividing by grid area, which is done as part of the `aneris.Gridder.grid()` operation.

# %%
kg_per_mt = 1e9
s_per_yr = 365 * 24 * 60 * 60
data_for_gridding = (
    data_for_gridding.rename(
        index=lambda s: re.sub("Mt (.*)/yr", r"kg \1/s", s), level="unit"
    )
    * kg_per_mt
    / s_per_yr
)

# %%
data_for_gridding_path = out_path / f"data_for_gridding-{version}.csv"
data_for_gridding.to_csv(data_for_gridding_path)

# %% [markdown]
# ## Execute Gridding

# %%
scen = data_for_gridding.pix.semijoin(
    data_for_gridding.pix.unique(["model", "scenario"])[2:4], how="right"
)  # TODO: Only 2nd and 3rd pathways
scen.pix.unique('scenario')

# %% [raw]
# _ = Gridder(
#     scen,
#     idxr,
#     proxy_cfg,
#     index_mappings=dict(sector=sector_mapping),
#     output_dir="../results",
# )
# _.check(strict_proxy_data=False)

# %% [raw]
# idx = [0, 11, 21, 32, 36]
# proxy_cfg_test = _PROXY_CFG.copy().iloc[idx]
# proxy_cfg_test

# %%
# cfg = proxy_cfg_test
cfg = _PROXY_CFG.copy()
cfg

# %%
gridder = Gridder(
    scen,
    idxr,
    cfg,
    index_mappings=dict(sector=sector_mapping),
    output_dir=out_path,
)

gridder.proxy_cfg
# skip checks when using this for testing

# %%
# %%execute_or_lazy_load execute_gridding
tasks = gridder.grid(
    skip_check=True,
    chunk_proxy_dims={"level": "auto"},
    iter_levels=["model", "scenario"],
    verify_output=True,
    skip_exists=True,
)

# %% [markdown]
# # Upload Data

# %%
from ftpsync.targets import FsTarget
from ftpsync.ftp_target import FTPTarget
from ftpsync.synchronizers import UploadSynchronizer

# %%
ftp = config["ftp"]
local = out_path
remote = ftp["path"] + '/' + version
opts = {"create_folder": True, "force": False, "delete_unmatched": True, "verbose": 3}
s = UploadSynchronizer(
    FsTarget(local), 
    FTPTarget(remote, ftp["server"], port=ftp["port"], username=ftp["user"], password=ftp["pass"]), 
    opts
)
s.run()
