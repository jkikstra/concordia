# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# # Workflow for CMIP7 ScenarioMIP emissions harmonization 

# %% [markdown]
# ## Specify input scenario data and project settings

# %% [markdown]
# Specify which scenario file to read in

# %%
HISTORY_FILE = "cmip7_history_0022.csv"

# %%
SCENARIO_FILE = "check_harmonisation_regions_REMIND.csv" # example (ALREADY HARMONIZED) COFFEE scenario

# SCENARIO_FILE = "scenarios_scenariomip_COFFEE 1.6_SSP2 - Low Overshoot.csv" # example COFFEE scenario
# SCENARIO_FILE = "scenarios_scenariomip_AIM 3.0_SSP2 - Low Emissions.csv" # example AIM scenario
# SCENARIO_FILE = "scenarios_scenariomip_REMIND-MAgPIE 3.5-4.10_SSP2 - Low Emissions.csv" # example REMIND scenario
# SCENARIO_FILE = "scenarios_scenariomip_MESSAGEix-GLOBIOM-GAINS 2.1-M-R12_SSP2 - Low Overshoot.csv" # example MESSAGE scenario
# SCENARIO_FILE = "scenarios_scenariomip_allmodels_2025-03-05-messagegains.csv" # TODO: update later for all models. Location for this file is specified in the yaml file read into the `settings` object later on

# %% [markdown]
# Specify settings

# %%
# Settings
SETTINGS_FILE = "config_cmip7_v0_testing_ukesm_remind.yaml" 

# versioning
# HARMONIZATION_VERSION = "config_cmip7_v0_testing_remind"
# HARMONIZATION_VERSION = "config_cmip7_v0_testing_aim"
HARMONIZATION_VERSION = "config_cmip7_v0_testing_ukesm_remind"

# %% [markdown]
# ## Importing packages

# %%
import aneris


aneris.__file__

# %%
import concordia


concordia.__file__

# %%
import logging
from pathlib import Path

import dask
import dask.dataframe as dd
import pandas as pd
from dask.distributed import Client
from pandas_indexing import concat, isin, ismatch, semijoin
from pandas_indexing.units import set_openscm_registry_as_default
from ptolemy.raster import IndexRaster

from aneris import logger
from concordia import (
    RegionMapping,
    VariableDefinitions,
)
from concordia.rescue import utils as rescue_utils # update to cmip7 utils (e.g. for dressing up netcdf)
from concordia.cmip7 import utils as cmip7_utils # update to cmip7 utils (e.g. for dressing up netcdf)
from concordia.settings import Settings
from concordia.utils import MultiLineFormatter, extend_overrides
from concordia.workflow import WorkflowDriver


# %% [markdown]
# Load unit registry from openSCM for translating units (e.g., to and from CO2eq)

# %%
ur = set_openscm_registry_as_default()

# %% [markdown]
# # Read Settings
#
# The key settings for this harmonization run are detailed in the file "config_cmip7_{VERSION}.yaml", which is located in this same folder.
#
# **NOTE: to use this workflow, you need to specify (at minimum) the `data_path` vairable in this "config_cmip7_{VERISON}.yaml" file.**
#
# This settings file for instance points to the data location, where e.g. the following files are hosted:
# - historical emissions data to harmonize the regional IAM trajectories to
# - gridding pattern files for gridding regional IAM trajectories
# - region-to-country mappings for all IAMs 
# - variable definitions: a list by species and sector, specifying units, gridding levels, and proxies
# - postprocessing: files for potential post-processing (current not used)
# - scenarios: input IAM trajectories

# %%
settings = Settings.from_config(version=HARMONIZATION_VERSION,
                                local_config_path=Path(Path.cwd(),
                                                       SETTINGS_FILE))

settings.base_year

# %% [markdown]
# Set logger (uses setting)

# %%
fh = logging.FileHandler(settings.out_path / f"debug_{settings.version}.log", mode="w")
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
fh.setFormatter(formatter)


streamhandler = logging.StreamHandler()
streamhandler.setFormatter(
    MultiLineFormatter(
        "%(log_color)s%(levelname)-8s%(reset)s %(message)s  (%(blue)s%(name)s%(reset)s)",
        datefmt=None,
        reset=True,
    )
)

logger().handlers = [streamhandler, fh]
logging.getLogger("flox").setLevel("WARNING")

# %% [markdown]
# Create output path for this version

# %%
version_path = settings.out_path / settings.version
version_path.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# # Read definitions

# %% [markdown]
# ## Read variable definitions
#
# The variable definition file is a CSV or yaml file that needs to contain the `variable`-name, its `sector`, `gas` components and whether it is expected `global` (or regional instead).
#
# Here we use a file based on the RESCUE variable definitions, but adapted to fit CMIP7 purposes.
#

# %%
settings.variabledefs_path

# %%
variabledefs = VariableDefinitions.from_csv(settings.variabledefs_path)
variabledefs.data.head()

# %%
variabledefs.data.loc[
    isin(sector="Energy Sector")
]

# %% [markdown]
# ## Read region definitions (using RegionMapping class)
#

# %%
settings.data_path

# %%
settings.regionmappings.items()

# %%
regionmappings = {}

for m, kwargs in settings.regionmappings.items():
    regionmapping = RegionMapping.from_regiondef(**kwargs)
    regionmapping.data = regionmapping.data.pix.aggregate(
        country=settings.country_combinations, agg_func="last"
    )
    regionmappings[m] = regionmapping

regionmappings

# %% [markdown]
# # IAM: Read and process IAM data

# %% [markdown]
# ### Define some useful functions

# %%
import pandas as pd

# Load IAMC data
def load_data(file_path):
    """
    Loads IAMC data from a CSV or Excel file and converts specific columns to lowercase.

    Parameters:
        file_path (str): The path to the input file (.csv or .xlsx).

    Returns:
        pd.DataFrame: The loaded and formatted dataframe.

    Raises:
        ValueError: If the file format is unsupported.
    """
    file_path_str = str(file_path)
    if file_path_str.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path_str.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Use .csv or .xlsx.")

    return iamc_to_lowercase(df)

# IAMC data to lower case
def iamc_to_lowercase(df):
    """
    Converts specific IAMC columns to lowercase.

    Parameters:
        df (pd.DataFrame): The dataframe to modify.

    Returns:
        pd.DataFrame: The modified dataframe with lowercased column names.
    """
    for col in ["Model", "Scenario", "Region", "Variable", "Unit"]:
        if col in df.columns:
            df.rename(columns={col: col.lower()}, inplace=True)
    return df


def sort_long_iamc_dataframe(df):
    """
    Sorts a long IAMC dataframe by model, scenario, region, variable, and year.

    Parameters:
        df (pd.DataFrame): The dataframe to sort.

    Returns:
        pd.DataFrame: The sorted dataframe.
    """
    sort_order = ["model", "scenario", "region", "variable", "year"]
    missing_cols = [col for col in sort_order if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns for sorting: {missing_cols}")
    return df.sort_values(by=sort_order).reset_index(drop=True)

def sort_iamc_dataframe(df, format="long"):

    if (format == "long"):
        return sort_long_iamc_dataframe(df)
    else:
        raise Exception("Formats other than 'long' not yet implemented.") 


def iamc_wide_to_long(df, iamc_cols=["model", "scenario", "variable", "region", "unit"]):
    """
    Converts IAMC data from wide to long format.

    Parameters:
        df (pd.DataFrame): The dataframe to transform.
        iamc_cols (list): List of IAMC-specific columns.

    Returns:
        pd.DataFrame: The transformed dataframe in long format.

    Raises:
        KeyError: If year columns cannot be identified.
    """

    # Convert all column names to string (sometimes, years may be integer)
    df.columns = df.columns.astype(str)
    # Convert all column names to lowercase
    df.columns = df.columns.str.strip().str.lower()

    # Identify year columns (assuming years are strings)
    year_columns = [str(col) for col in df.columns[len(iamc_cols):] if str(col).isdigit()]
    if not year_columns:
        raise KeyError("Year columns could not be identified. Ensure the dataframe has year columns after the basic IAMC columns.")



    # Melt the dataframe to long format
    long_df = df.melt(
        id_vars=iamc_cols,
        value_vars=year_columns,
        var_name="year",
        value_name="value",
    )

    # Convert year and value columns to numeric types
    long_df["year"] = pd.to_numeric(long_df["year"], errors="coerce")
    long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")
    long_df.dropna(subset=["year", "value"], inplace=True)
    long_df["year"] = long_df["year"].astype(int)

    long_df = sort_iamc_dataframe(long_df)

    return long_df

# Filter functions
def filter_scenario(df, scenarios):
    """
    Filters dataframe by scenarios.

    Parameters:
        df (pd.DataFrame): The dataframe to filter.
        scenarios (str or list): Scenario(s) to filter by.

    Returns:
        pd.DataFrame: The filtered dataframe.
    """
    if isinstance(scenarios, list):
        return df[df['scenario'].isin(scenarios)]
    return df[df['scenario'] == scenarios]

def filter_region(df, regions):
    """
    Filters dataframe by regions.

    Parameters:
        df (pd.DataFrame): The dataframe to filter.
        regions (str or list): Region(s) to filter by.

    Returns:
        pd.DataFrame: The filtered dataframe.
    """
    if isinstance(regions, list):
        return df[df['region'].isin(regions)]
    return df[df['region'] == regions]

def filter_variable(df, variables):
    """
    Filters dataframe by variables.

    Parameters:
        df (pd.DataFrame): The dataframe to filter.
        variables (str or list): Variable(s) to filter by.

    Returns:
        pd.DataFrame: The filtered dataframe.
    """
    if isinstance(variables, list):
        return df[df['variable'].isin(variables)]
    return df[df['variable'] == variables]

def filter_region_contains(df, substrings):
    """
    Filters dataframe by regions containing specific substrings.

    Parameters:
        df (pd.DataFrame): The dataframe to filter.
        substrings (str or list): Substring(s) to search for in region names.

    Returns:
        pd.DataFrame: The filtered dataframe.
    """
    if isinstance(substrings, list):
        return df[df['region'].str.contains('|'.join(substrings), case=False, na=False)]
    return df[df['region'].str.contains(substrings, case=False, na=False)]

def filter_emissions_data(df):
    """
    Filters dataframe for variables starting with "Emissions".

    Parameters:
        df (pd.DataFrame): The dataframe to filter.

    Returns:
        pd.DataFrame: The filtered dataframe.
    """
    return df[df['variable'].str.startswith("Emissions")]

# remove data with year > 2100; assumes a dataframe in long format 
def remove_data_after(df, yr = 2100):
    return df[df['year'] <= yr]

# Rename one variable explicitly
def rename_one_variable(df, old_string, new_string):
    df['variable'] = df['variable'].replace({old_string: new_string})
    return df

# Renaming variables
def rename_variable(df, rename_dict):
    """
    Renames variables in the dataframe using a dictionary mapping.

    Parameters:
        df (pd.DataFrame): The dataframe to modify.
        rename_dict (dict): A dictionary with old variable names as keys and new names as values.

    Returns:
        pd.DataFrame: The modified dataframe with updated variable names.
    """
    df['variable'] = df['variable'].replace(rename_dict)
    return df

# Custom function to select columns and drop duplicates
def select_and_distinct(dataframe, columns):
    return dataframe.loc[:, columns].drop_duplicates()

# Filter regions for CMIP7 data
def filter_regions_only_world_and_model_native(df, cmip7_iam_list=None):
    if cmip7_iam_list is None:
        cmip7_iam_list = ["MESSAGE", "AIM", "COFFEE", "GCAM", "IMAGE", "REMIND", "WITCH"]

    world_df = filter_region(df, regions="World")
    model_native_df = filter_region_contains(df, substrings=cmip7_iam_list)

    return pd.concat([world_df, model_native_df], axis=0)

# Reformatting; identify species in a separate column
def reformat_IAMDataframe_with_species_column(df, start_string="Emissions|", end_string=None):
    """
    Extracts species from e.g. "Emissions|" variable names, strips an optional `end_string` from the end,
    and reformats the dataframe.

    Parameters:
        df (pd.DataFrame): The dataframe to process.
        start_string (str): The string to strip from the start of the variable names.
        end_string (str): The string to strip from the end of the variable names (if provided).

    Returns:
        pd.DataFrame: The reformatted dataframe.
    """
    if end_string:
        df['variable'] = df['variable'].str.removesuffix(end_string)

    # Strip the start string
    if start_string:
        df['variable'] = df['variable'].str.removeprefix(start_string)

    # Extract species from the variable column (assuming it is the first element after the start string has been removed)
    df['species'] = df['variable'].str.split('|').str[0]

    
    # create a sector column
    df['sector'] = df['variable'].apply(lambda x: x.split('|', 1)[1] if '|' in x else 'Total')
    
    return df

# Sum values of selected variables
def sum_selected_variables(df, selected_variables, new_variable_name, group_cols=["model", "scenario", "region", "unit", "year"]):
    """
    Sums selected variables into a new variable.

    Parameters:
        df (pd.DataFrame): The dataframe to process.
        selected_variables (list): List of variables to sum.
        new_variable_name (str): Name of the new aggregated variable.
        group_cols (list): Columns to group by.

    Returns:
        pd.DataFrame: The modified dataframe with the new aggregated variable.

    Raises:
        ValueError: If inputs are of invalid types.
    """
    if not isinstance(selected_variables, list):
        raise ValueError("selected_variables must be a list.")
    if not isinstance(new_variable_name, str):
        raise ValueError("new_variable_name must be a string.")

    selected_df = df[df['variable'].isin(selected_variables)]
    summed = (
        selected_df
        .groupby(group_cols, as_index=False)
        .agg({"value": "sum"})
    )
    summed["variable"] = new_variable_name
    return pd.concat([df, summed], axis=0)

# Sectoral adjustments
def process_transportation_variables(
    df,
    group_cols=["model", "scenario", "region", "unit", "year"],
    new_aviation_variable="Aircraft",
    new_transportation_variable="Transportation Sector",
):
    av_dom_var = "Energy|Demand|Transportation|Domestic Aviation"
    av_int_var = "Energy|Demand|Bunkers|International Aviation"
    trp_var = "Energy|Demand|Transportation"

    aviation_df = df[df["variable"].isin([av_dom_var, av_int_var])]
    aggregated_aviation = (
        aviation_df.groupby(group_cols, as_index=False)
        .agg({"value": "sum"})
    )
    aggregated_aviation["variable"] = new_aviation_variable
    df = pd.concat([df, aggregated_aviation], axis=0)

    domestic_aviation = df[df["variable"] == av_dom_var][group_cols + ["value"]]
    domestic_aviation.rename(columns={"value": "value_dom"}, inplace=True)
    transportation = df[df["variable"] == trp_var]

    transportation = transportation.merge(domestic_aviation, on=group_cols, how="left")
    transportation["value"] -= transportation["value_dom"].fillna(0)
    transportation.drop(columns=["value_dom"], inplace=True)
    transportation["variable"] = new_transportation_variable

    df = pd.concat([df[df["variable"] != trp_var], transportation], axis=0)
    return df

def process_industrial_sector_variables(df, industry_variable_list=None, group_cols=["model", "scenario", "region", "unit", "year"]):
    if industry_variable_list is None:
        industry_variable_list = [
            "Energy|Supply",
            "Energy|Demand|Industry",
            "Energy|Demand|Other Sector",
            "Industrial Processes",
            "Other"
        ]

    df = sum_selected_variables(
        df,
        selected_variables=industry_variable_list,
        new_variable_name="Industrial Sector",
        group_cols=group_cols
    )
    return df

# Pipeline
def process_data(df, group_cols=["model", "scenario", "region", "unit", "year"]):
    """
    Main processing function for transportation and industrial sectors.

    Parameters:
        df (pd.DataFrame): The dataframe to process.
        group_cols (list): Columns to group by.

    Returns:
        pd.DataFrame: The processed dataframe.
    """
    df['variable'] = df['variable'].str.replace(r'^.*?\|', '', regex=True) # delete the species; i.e. everything before the first | character
    df = process_transportation_variables(df, group_cols=group_cols)
    df = process_industrial_sector_variables(df, group_cols=group_cols)
    return df

# Save processed data
def save_data(df, output_path):
    """
    Saves the dataframe to a CSV or Excel file.

    Parameters:
        df (pd.DataFrame): The dataframe to save.
        output_path (str): Path to the output file (.csv or .xlsx).

    Raises:
        ValueError: If the file format is unsupported.
    """
    if output_path.endswith('.csv'):
        df.to_csv(output_path, index=False)
    elif output_path.endswith('.xlsx'):
        df.to_excel(output_path, index=False)
    else:
        raise ValueError("Unsupported file format. Use .csv or .xlsx.")
    
# create a comparison
# Approach:
#     Canonical Form for Comparison:
#         Normalize the DataFrames by sorting their rows and resetting the index.
#         This allows you to compare rows in a consistent order.
#     Check for Matching Rows:
#         Use set operations to find exact matches and mismatches.
#         Convert the rows of each DataFrame into sets of tuples for comparison.
#     Identify Important Mismatches:
#         Rows in df1 but not in df2.
#         Rows in df2 but not in df1.
def compare_units(name_df1, df1, name_df2, df2, quiet = True):
    # to be applied to:
    # * historical
    # * iam data

    # Step 1: Convert rows to sets of tuples
    set_df1 = set([tuple(row) for row in df1.sort_values(by=df1.columns.tolist()).itertuples(index=False, name=None)])
    set_df2 = set([tuple(row) for row in df2.sort_values(by=df2.columns.tolist()).itertuples(index=False, name=None)])

    # Step 2: Find matches and mismatches
    matches = set_df1 & set_df2
    only_in_df1 = set_df1 - set_df2
    only_in_df2 = set_df2 - set_df1

    if not quiet:
        # Step 3: Display results
        print("Exact Matches:")
        print(matches)

        print("\nRows only in " + name_df1 + ":")
        print(only_in_df1)

        print("\nRows only in " + name_df2 + ":")
        print(only_in_df2)


    # Step 3: convert to dataframe
    all_tuples = list(matches) + list(only_in_df1) + list(only_in_df2)
    all_sources = (
        ['Exact Matches'] * len(matches) +
        [f'Only in {name_df1}'] * len(only_in_df1) +
        [f'Only in {name_df2}'] * len(only_in_df2)
    )


    matching_data = pd.DataFrame({
        'Sector': [x[0] for x in all_tuples],
        'Unit': [x[1] for x in all_tuples],
        'Source': all_sources
    })

    return matching_data

    


# %%
SCENARIO_FILE

# %%
iam_df.drop_duplicates()

# %% [markdown]
# ### Read in

# %%
# Read in already-harmonized data
iam_df = load_data(
    Path(settings.scenario_path, SCENARIO_FILE) 
)
iam_df = iam_df[iam_df['stage']=="harmonised"]
iam_df = iam_df.drop_duplicates() # aircraft and international shipping have duplicates right now

# filter only one scenario  # TODO: remove after test code is done
# iam_df = iam_df[iam_df['scenario']=="SSP1 - Very Low Emissions"]
iam_df = filter_scenario(iam_df, scenarios="SSP1 - Very Low Emissions")

iam_df[iam_df['variable']=="Emissions|BC|International Shipping"]

# # Filter data (should not do anything, as already taken care of in emissions_harmonization_historical workflow)
# iam_df = filter_emissions_data(iam_df) # only keep variable=="Emissions*" 
# iam_df = filter_regions_only_world_and_model_native(iam_df) # delete R10/R5

# %%
IAMC_COLS = ["model", "scenario", "region", "variable", "unit"]
HARMONIZED_YEAR_COLS = [col for col in iam_df.columns if col.isdigit() and 2023 <= int(col) <= 2100]

# %%
# keep only relevant columns
iam_df = iam_df.drop(columns=["stage"])[(IAMC_COLS + HARMONIZED_YEAR_COLS)]

# %%
iam_df

# %% [markdown]
# ### Process (using pix - formatting)

# %%
from pandas_indexing import extractlevel
# split the 'variable' column into the 'gas' and 'sector' columns
iam_df = extractlevel(iam_df.set_index(IAMC_COLS), variable="Emissions|{gas}|{sector}", drop=True)

# Reorder the MultiIndex of iam_df
iam_df = iam_df.reorder_levels(['model', 'scenario', 'region', 'gas', 'sector', 'unit'])
iam_df = iam_df.sort_index()

# Update column type and name
iam_df.columns = iam_df.columns.astype(int)
iam_df.columns.name = 'year'
iam_df


# %% [markdown]
# ## Save the processed IAM data

# %% [markdown]
# ### Basic checks

# %%
def check_na_in_columns(df):
    """
    Check all columns in the DataFrame for NA values.
    Raise a KeyError with the column names that contain one or more NA values.

    :param df: pandas DataFrame to check
    """
    # Find columns with NA values
    columns_with_na = df.columns[df.isna().any()].tolist()
    
    if columns_with_na:
        raise KeyError(f"The following column(s) contain NA values: {', '.join(columns_with_na)}")
    else:
        print("No NA values found in any column.")

check_na_in_columns(iam_df)

# %% [markdown]
# ### Save in wide format

# %%
save_data(df = iam_df.reset_index(),    
          output_path = str(Path(version_path, "scenarios_processed.csv" )))

# %% [markdown]
# # History: Read and process historical data
#
# Can be read in and prepared using `read_iamc` or the `variabledefs`
#

# %%
hist = (
    pd.read_csv(settings.history_path / HISTORY_FILE)
    .drop(columns=['model', 'scenario'])
    .rename(columns={"region": "country"})
)

hist = extractlevel(hist.set_index(['country', 'variable', 'unit']), variable="Emissions|{gas}|{sector}", drop=True)

# Reorder the MultiIndex of hist
hist = hist.reorder_levels(['country', 'gas', 'sector', 'unit'])
hist = hist.sort_index()

# Update column type and name
hist.columns = hist.columns.astype(int)
hist.columns.name = 'year'
hist

# %% [markdown]
# # Read Harmonization Overrides

# %%
settings.scenario_path

# %%
harm_overrides = pd.read_excel(
    settings.scenario_path / "harmonization_overrides.xlsx",
    index_col=list(range(3)),
).method
harm_overrides

# %%
harm_overrides = extend_overrides(
    harm_overrides,
    "constant_ratio",
    sector=[
        f"{sec} Burning"
        for sec in ["Agricultural Waste", "Forest", "Grassland", "Peat"]
    ],
    variables=variabledefs.data.index,
    regionmappings=regionmappings,
    model_baseyear=iam_df[str(settings.base_year)],
)

# %% [markdown]
# # Prepare GDP proxy
#
# Read in different GDP scenarios for SSP1 to SSP5 from SSP DB.
#

# %%
# New; updated SSP data from CMIP7 era
gdp = (
    pd.read_csv(
        settings.scenario_path / "gdp_sspv31_withallextradata_temporaryuse.csv", # only use until GDP data is finalised
        # index_col=list(range(5)),
    )
    .filter(["model", "scenario", "iso", "variable", "unit", "year", "value"]) # select only columns  "model", "scenario", "iso", "variable", "unit", "year"
    .query("year >= 2020") # keep only projections
    .rename(columns={'iso': 'region'})
    .pivot_table(index=["scenario", "region", "variable", "unit"],
                 values="value",
                 columns="year")
    .rename_axis(index=str.lower)
    .loc[
        isin(
            # model="OECD Env-Growth",
            scenario=[f"SSP{n+1}" for n in range(5)],
            variable="GDP|PPP",
        )
    ]
    .dropna(how="all", axis=1)
    .rename_axis(index={"scenario": "ssp", "region": "country"})
    .rename(index=str.lower, level="country")
    .rename(columns=int)
    .pix.project(["ssp", "country"])
    .pix.aggregate(country=settings.country_combinations)
)

# INTERPOLATE:
# Interpolate GDP DataFrame to annual data (fill all years in the column range)
# Assumes 'gdp' is a DataFrame with years as columns (integers) and a MultiIndex

# Get all years in the range (min to max)
all_years = range(min(gdp.columns), max(gdp.columns) + 1)

# Reindex to include all years, then interpolate
gdp = (
    gdp.reindex(columns=all_years)
       .interpolate(axis=1, method='linear', limit_direction='both')
)

gdp

# # Old; original SSP data from CMIP6 era
# gdp = (
#     pd.read_csv(
#         settings.scenario_path / "SspDb_country_data_2013-06-12.csv",
#         index_col=list(range(5)),
#     )
#     .rename_axis(index=str.lower)
#     .loc[
#         isin(
#             model="OECD Env-Growth",
#             scenario=[f"SSP{n+1}_v9_130325" for n in range(5)],
#             variable="GDP|PPP",
#         )
#     ]
#     .dropna(how="all", axis=1)
#     .rename_axis(index={"scenario": "ssp", "region": "country"})
#     .rename(index=str.lower, level="country")
#     .rename(columns=int)
#     .pix.project(["ssp", "country"])
#     # .pix.aggregate(country=settings.country_combinations)
# )
# gdp

# %% [markdown]
# Determine likely SSP for each harmonized pathway from scenario string and create proxy data aligned with pathways
#

# %%
def guess_ssp(df):
    ssp_guesses = (
    df.index.pix.project(["model", "scenario"])
    .unique()
    .to_frame()
    .scenario.str.extract("(SSP[1-5])")[0]
    .fillna("SSP2")
    )
    return ssp_guesses
def join_gdp_based_on_ssp(scenarios_with_ssp_mapping, gdp_per_ssp):
    gdp_for_each_scenario = semijoin(
            gdp_per_ssp,
            # SSP_per_pathway.index.pix.assign(ssp=SSP_per_pathway + "_v9_130325"), # CMIP6 era SSP data
            scenarios_with_ssp_mapping.index.pix.assign(ssp=scenarios_with_ssp_mapping), # CMIP7 era SSP data
            how="right",
        ).pix.project(["model", "scenario", "country"])
    return gdp_for_each_scenario


# %%
SSP_per_pathway = guess_ssp(iam_df)
GDP_per_pathway = join_gdp_based_on_ssp(
    scenarios_with_ssp_mapping=SSP_per_pathway,
    gdp_per_ssp=gdp
)

# %% [markdown]
# # Country coverage

# %%
# try to align with CEDS; but where necessary, aggregate to SSP coverage.

# what countries do we have in each data set?
countries_with_gdp_data = gdp.pix.unique("country") # as Index
countries_with_hist_data = hist.pix.unique("country") # as Index
countries_with_regionmapping = pd.Index(sorted(
    regionmapping.filter(countries_with_gdp_data).data.reset_index().country.unique() # as array
)) # as Index
countries_with_hist_and_gdp_and_regionmapping_data = pd.Index(sorted(( 
    set(countries_with_gdp_data) & set(countries_with_hist_data) & set(countries_with_regionmapping) # as set
))) # as Index

# show what we have
print(len(countries_with_gdp_data))
print(len(countries_with_hist_data))
print(len(countries_with_regionmapping))
print(countries_with_hist_and_gdp_and_regionmapping_data)

def select_only_countries_with_all_info(df,
                                        countries=countries_with_hist_and_gdp_and_regionmapping_data):
    df = (
        df
        .loc[
            isin(
                country=countries
            )
        ]
    )
    
    return df


# %%
# # Get unique countries from each dataframe
# hist_countries = set(hist.pix.unique("country"))
# gdp_countries = set(GDP_per_pathway.pix.unique("country"))

# # Countries in hist but not in GDP_per_pathway
# in_hist_not_gdp = hist_countries - gdp_countries
# print("Countries in hist but not in GDP_per_pathway:")
# print(sorted(in_hist_not_gdp))

# # Countries in GDP_per_pathway but not in hist
# in_gdp_not_hist = gdp_countries - hist_countries
# print("Countries in GDP_per_pathway but not in hist:")
# print(sorted(in_gdp_not_hist))

# # Display counts for reference
# print(f"\nTotal countries in hist: {len(hist_countries)}")
# print(f"Total countries in GDP_per_pathway: {len(gdp_countries)}")
# print(f"Countries in common: {len(hist_countries & gdp_countries)}")

# %% [markdown]
# # Set up technical bits for the workflow

# %%
client = Client()
# client.register_plugin(DaskSetWorkerLoglevel(logger().getEffectiveLevel()))
client.forward_logging()

# %%
dask.distributed.gc.disable_gc_diagnosis()

# %% [markdown]
# # Define workflow

# %%
# TODO: 
# - [ ] make this into a dataframe, and loop over models? --> right now the below section only works for 1 model at a time.

(model_name,) = iam_df.pix.unique("model")
regionmapping = regionmappings[model_name]

# scens_iam_wide.pix.unique("model")

# %%
# indexes for countries on a grid
indexraster = IndexRaster.from_netcdf(
    settings.gridding_path / "ssp_comb_indexraster.nc", # redo: notebooks\gridding_data\generate_ceds_proxy_netcdfs.py
    chunks={},
).persist()
indexraster_region = indexraster.dissolve(
    regionmapping.filter(indexraster.index).data.rename("country")
).persist()

# %%
iam_df.columns

# %%
workflow = WorkflowDriver( 
    # model
    # iam_df, # model
    # iam_df.loc[:, iam_df.columns.intersection(GDP_per_pathway.columns.tolist())], # model ; until GDP is interpolated, do only for years in GDP_per_pathway.columns.tolist()
    iam_df, 
    # hist
    hist, # select_only_countries_with_all_info(hist),
    # gdp
    GDP_per_pathway, #select_only_countries_with_all_info(GDP_per_pathway),
    # regionmapping
    regionmapping.filter(countries_with_hist_and_gdp_and_regionmapping_data[~countries_with_hist_and_gdp_and_regionmapping_data.isin(['myt','gum'])]), # mayotte and guam are missing some historical data for some sectors
    # indexraster_country
    indexraster,
    # indexraster_region
    indexraster_region,
    # variabledefs
    variabledefs,
    # harm_overrides
    harm_overrides,
    # settings
    settings
)

# %% [markdown]
# ## Add some checks on workflow

# %%
# save workflow info in easy-to-vet packets 
workflow.save_info(path = Path("..", "data", "compare_wfd_inputs"), prefix=settings.version)

# %%
# check regionmapping and scenarios
reg_model = iam_df.loc[~isin(region="World")].reset_index().region.unique() # all region names of the scenario
reg_mapped = regionmapping.data.reset_index().region.unique() # all region names of the scenario

def assert_strings_covered(array1, array2):
    assert all(s in array2 for s in array1), "Not all regions are covered in the regionmapping"
assert_strings_covered(reg_model, reg_mapped)

# %% [markdown]
# # Harmonize, downscale and grid everything
#

# %% [markdown]
# ## Alternative 2) Harmonize and downscale everything, but do not grid
#
# If you also want grids, use the gridding interface directly.
# For a 1 scenario, this takes about 50 seconds on Jarmo's DELL laptop.
#

# %%
# error: why?
# - maybe: column names are string/object, instead of int32 {other workflow} with name=year {SOLVED: COLUMNS TYPE CHANGED AND NAMES CHANGED}
# - probably: model is empty {SOLVED: GDP NOW INTERPOLATER}
# -> model has two times 'aircraft' and 'international shipping' for region=="World" {SOLVED: DUPLICATES IN MODEL DATA, i.e. Intl Shipping and Aicraft were duplicates}

downscaled = workflow.harmonize_and_downscale()

# %% [markdown]
# ### Export harmonized scenarios

# %%
print("Outputs will be placed in " + str(version_path.resolve()))
data = (
    workflow.harmonized_data.add_totals()
    .to_iamc(settings.variable_template, hist_scenario="Synthetic (CEDS/GFED/Global)")
    # .pipe(rename_alkalinity_addition)
    .rename_axis(index=str.capitalize)
)
print("File: " + f"harmonization-{settings.version}.csv")
data.to_csv(version_path / f"harmonization-{settings.version}.csv")

# %% [markdown]
# ### Export downscaled scenarios
#
# TODO: create a similar exporter to the Harmonized class for Downscaled which combines historic and downscaled data (maybe also harmonized?) and translates to iamc
#

# %%
# Do we also want to render this as IAMC?
workflow.downscaled.data.to_csv(
    version_path / f"downscaled-only-{settings.version}.csv"
)
print(
    "Countries covered (" + str(len(workflow.downscaled.data.loc[~isin(region="World")].reset_index().country.unique())) + "):"
)
print(workflow.downscaled.data.loc[~isin(region="World")].reset_index().country.unique())

# %% [markdown]
# ## Alternative 1) Run full processing and create netcdf files
#
# Latest test with 1 scenario was 25 minutes on Jarmo's DELL laptop.
# Output files are nearly 6GB for one scenario.

# %%
cmip7_utils.DS_ATTRS

# %%
res = workflow.grid(
    template_fn="{{name}}_{activity_id}_emissions_{target_mip}_{institution}-{{model}}-{{scenario}}_{grid_label}_{start_date}-{end_date}.nc".format(
        **cmip7_utils.DS_ATTRS | {"version": settings.version}
    ),
    callback=cmip7_utils.DressUp(version=settings.version),
    directory=version_path,
    skip_exists=True,
)

# %% [markdown]
# # END OF MAIN CODE

# %% [markdown]
# # ------------------------------------

# %% [markdown]
# # START OF SOME NOT-NECESSARY CODE SNIPPETS

# %% [markdown]
# ## Alternative 2) INPUT DIAGNOSTICS FOR Harmonize and downscale everything, but do not grid
#
# If you also want grids, use the gridding interface directly.
# For a handfull of scenarios, this takes less than a minute on a Dell laptop.
#

# %% [markdown]
# ### Looking around: Global

# %%
variables = workflow.variabledefs.globallevel.index
model = workflow.model.pix.semijoin(variables, how="right").loc[
            isin(region="World")
        ]
hist = (
            workflow.hist.pix.semijoin(variables, how="right")
            .loc[isin(country="World")]
            .rename_axis(index={"country": "region"})
        )

# %%
harmonized = concordia.harmonize.harmonize(
            model,
            hist,
            overrides=workflow.harm_overrides.pix.semijoin(variables, how="inner"),
            settings=workflow.settings,
        )
harmonized = concordia.utils.aggregate_subsectors(harmonized)
hist = concordia.utils.aggregate_subsectors(hist)


# %%
# workflow.history_aggregated.globallevel = hist
# self.harmonized.globallevel = harmonized
# self.downscaled.globallevel = harmonized.pix.format(
#     method="single", country="{region}"
# )
# harmonized.pix.format(
#     method="single", country="{region}"
# )
harmonized.pix.format(
    nlanal="{region}"
)

# %%
workflow.downscaled.globallevel

# %% [markdown]
# ### Looking around: Country

# %%
hist.loc[isin(country='nld')] # hist

# %%
# check for NLD, CO2, Energy Sector (for the 1 scenario that is loaded)
workflow.regionmapping.data.loc['nld'] # regionmapping

hist.loc[isin(country='nld', gas='CO2', sector='Energy Sector')] # hist
GDP_per_pathway.loc[isin(country='nld')] # gdp proxy
'nld' in countries_with_hist_and_gdp_and_regionmapping_data # regionmapping selection passed onto workflowdriver
indexraster.boundary.sel(country=indexraster.index.get_loc('nld')) # locate in indexraster
indexraster_region.boundary.sel(country=indexraster_region.index.get_loc(workflow.regionmapping.data.loc['nld'])) # locate in indexraster_region

variabledefs.data.loc[isin(gas='CO2', sector='Energy Sector')] # check variable definition for an important sector
settings

# %%
settings

# %%
nld_code = indexraster.index.get_loc('nld') + 1  # Assuming 1-based encoding in the raster
mas = indexraster.indicator == nld_code
mas
nld_code

# %%
for key, value in workflow.proxies.items():
    # print(f"Key: {key}, Value: {value}")
    # print(f"Attributes of Value: {dir(value)}")
    print(f"Key: {key}, Proxy Weight: {value.weight.sel(country='nld')}")

# %%
print(key)
print(type(value))
print(type(value.weight))
print(value.weight['country'].sel(country='ind',year=2030, sector='Peat Burning', gas='BC'))
print(value.weight['country'].sel(country='ind',year=2030, sector='Peat Burning', gas='BC').data)

# %%
workflow.proxies['BC_em_openburning'].data # example openburning proxy
workflow.proxies['CO2_em_anthro'].data # example anthropogenic emissions

# %%
workflow.proxies.items()
workflow.proxies.keys()

# %%
i = 1
for gr in workflow.country_groups(variabledefs):
    print(i)
    print(workflow.regionmapping.filter(gr.countries))

# %%
import concordia.downscale


print("Harmonizing and downscaling " + str(len(workflow.variabledefs.countrylevel.index)) + " variables to country level")

history_aggregated = []
harmonized = []
downscaled = []

history_aggregated = []
harmonized = []
downscaled = []
i = 1
while i < 2:
    for group in workflow.country_groups(variabledefs):
        regionmapping = workflow.regionmapping.filter(group.countries)
        missing_regions = set(workflow.regionmapping.data.unique()).difference(
            regionmapping.data.unique()
        )
        missing_countries = workflow.regionmapping.data.index.difference(
            group.countries
        )

        model = workflow.model.pix.semijoin(group.variables, how="right")
        hist = workflow.hist.pix.semijoin(group.variables, how="right")
        hist_agg = regionmapping.aggregate(hist, dropna=True)

        # log_uncovered_history(hist, hist_agg, base_year=self.settings.base_year)
        history_aggregated.append(
            concordia.utils.add_zeros_like(hist_agg, hist, region=missing_regions)
        )

        harm = concordia.harmonize.harmonize(
            model.loc[isin(region=regionmapping.data.unique())],
            hist_agg,
            overrides=workflow.harm_overrides.pix.semijoin(
                group.variables, how="inner"
            ),
            settings=workflow.settings,
        )
        harmonized.append(
            concordia.utils.add_zeros_like(harm, model, region=missing_regions, method=["all_zero"])
        )

        harm = concordia.utils.aggregate_subsectors(harm.droplevel("method"))
        hist = concordia.utils.aggregate_subsectors(hist)

        down = concordia.downscale.downscale(
            harm,
            hist,
            workflow.gdp,
            regionmapping,
            settings=workflow.settings,
        )
        downscaled.append(
            concordia.utils.add_zeros_like(
                down,
                harm,
                country=missing_countries,
                method=["all_zero"],
                derive=dict(region=workflow.regionmapping.index),
            )
        )
        i = i + 1 

# %%
'nld' in missing_countries # nld not in countries
'chn' in missing_countries # chn is in the countries

regionmapping.data # why only a few regions & countries!! --> all in this mapping don't feature in the harmonization output. All that 

# %% [markdown]
# ## Alternative 3) Investigations: Gridded

# %% [markdown]
# ### Look at a processed emissions file

# %%
import xarray as xr

# %%
result_grid = xr.open_dataset(Path("..", "results", "config_cmip7_v0_testing", "CH4-em-anthro_input4MIPs_emissions_RESCUE_IIASA-PIK-MESSAGEix-GLOBIOM-GAINS-2.1-M-R12-SSP2---Low-Overshoot_gn_201501-210012.nc"))

# %%
# inspect data set

# View variable names
print(result_grid.data_vars)
# View coordinates
print(result_grid.coords)
# Pick the variable (one per nc file)
print(result_grid['CH4_em_anthro'])
# What years?
import numpy as np
print(np.unique(result_grid.coords['time'].values))

# %%
result_grid['CH4_em_anthro'].sel(time = '2100-11-16 00:00:00', 
                                 sector = 'Energy').plot()

# %%
import numpy as np
# nicer color range
data = result_grid['CH4_em_anthro'].sel(time='2015-11-16 00:00:00', sector='Energy')
# Extract the data you're plotting
data = result_grid['CH4_em_anthro'].sel(time='2100-11-16 00:00:00', sector='Energy')
# Compute the 2.5th and 97.5th percentiles
vmin, vmax = np.nanpercentile(data, [0, 99.0])
# Plot using these as the color range
data.plot(vmin=vmin, vmax=vmax)

# %%
data.sel(time="2100-11-16 00:00:00")

# %%
# nicer color range and nicer projection
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

# Set up plot with Robinson projection
fig = plt.figure(figsize=(12, 6))
ax = plt.axes(projection=ccrs.Robinson())
ax.set_global()
ax.coastlines()

# Plot using pcolormesh with PlateCarree source projection
data.squeeze().plot.pcolormesh( # squeeze to remove size-1 dimensions
    ax=ax,
    transform=ccrs.PlateCarree(),  # Assumes data is in lat/lon (PlateCarree)
    vmin=vmin,
    vmax=vmax,
    cmap='viridis',  # You can change this to any color map
    add_colorbar=True,
    cbar_kwargs={'label': 'CH₄ Emissions (anthropogenic)'}
)

plt.title('CH₄ Emissions (Anthropogenic) - Energy Sector (2100-11-16)', fontsize=12)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Process single proxy
#
# `workflow.grid_proxy` returns an iterator of the gridded scenarios. We are looking at the first one in depth.

# %%
gridded = next(workflow.grid_proxy("CO2_em_anthro"))

# %%
ds = gridded.prepare_dataset(callback=rescue_utils.DressUp(version=settings.version))
ds

# %%
gridded.to_netcdf(
    template_fn="{{name}}_{activity_id}_emissions_{target_mip}_{institution}-{{model}}-{{scenario}}_{grid_label}_201501-210012.nc".format(
        **rescue_utils.DS_ATTRS | {"version": settings.version}
    ),
    callback=rescue_utils.DressUp(version=settings.version),
    directory=version_path,
)

# %%
ds["CO2_em_anthro"].sel(sector="CDR OAE", time="2015-09-16").plot()

# %%
# ds.isnull().any(["time", "lat", "lon"])["CO2_em_anthro"].to_pandas()

# %%
# reldiff, _ = dask.compute(
#     gridded.verify(compute=False),
#     gridded.to_netcdf(
#         template_fn=(
#             "{{name}}_{activity_id}_emissions_{target_mip}_{institution}-"
#             "{{model}}-{{scenario}}-{version}_{grid_label}_201501-210012.nc"
#         ).format(**rescue_utils.DS_ATTRS | {"version": settings.version}),
#         callback=rescue_utils.DressUp(version=settings.version),
#         encoding_kwargs=dict(_FillValue=1e20),
#         compute=False,
#         directory=version_path,
#     ),
# )
# reldiff

# %% [markdown]
# ### Regional proxy weights

# %%
# gridded.proxy.weight.regional.sel(
#     sector="Transportation Sector", year=2050, gas="CO2"
# ).compute().to_pandas().plot.hist(bins=100, logx=True, logy=True)


# %% [markdown]
# ## Alternative 4) Investigations: national Downscaled

# %%
# tbd

# %% [markdown]
# ## Export harmonized scenarios
#


# %%
data = (
    workflow.harmonized_data.add_totals()
    .to_iamc(settings.variable_template, hist_scenario="Synthetic (CEDS/GFED/Global)")
    # .pipe(rename_alkalinity_addition)
    .rename_axis(index=str.capitalize)
)
data.to_csv(version_path / f"harmonization-{settings.version}.csv")

# %% [markdown]
# ### Split HFC distributions
#

# %%
# hfc_distribution = (
#     pd.read_excel(
#         settings.postprocess_path / "rescue_hfc_scenario.xlsx",
#         index_col=0,
#         sheet_name="velders_2015",
#     )
#     .rename_axis("hfc")
#     .rename(columns=int)
# )

# data = (
#     workflow.harmonized_data.drop_method()
#     .add_totals()
#     .aggregate_subsectors()
#     .split_hfc(hfc_distribution)
#     .to_iamc(settings.variable_template, hist_scenario="Synthetic (GFED/CEDS/Global)")
#     # .pipe(rename_alkalinity_addition)
#     .rename_axis(index=str.capitalize)
# )
# data.to_csv(version_path / f"harmonization-{settings.version}-splithfc.csv")
