# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %load_ext autoreload
# %autoreload 2

# # Workflow for CMIP7 ScenarioMIP emissions harmonization 

# ## Specify input scenario data and project settings

# Specify which scenario file to read in

SCENARIO_FILE = "scenarios_scenariomipMESSAGE_20241127.csv" # TODO: update later for all models. Location for this file is specified in the yaml file read into the `settings` object later on

# Specify settings

# +
# Settings
SETTINGS_FILE = "config_cmip7_v0_testing.yaml" 

# versioning
HARMONIZATION_VERSION = "config_cmip7_v0_testing"
# -

# ## Importing packages

# +
import aneris


aneris.__file__

# +
import concordia


concordia.__file__

# +
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
from concordia.rescue import utils as rescue_utils
from concordia.settings import Settings
from concordia.utils import MultiLineFormatter, extend_overrides
from concordia.workflow import WorkflowDriver
# -


# Load unit registry from openSCM for translating units (e.g., to and from CO2eq)

ur = set_openscm_registry_as_default()

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

settings = Settings.from_config(version=HARMONIZATION_VERSION,
                                local_config_path=Path(Path.cwd(),
                                                       SETTINGS_FILE))

# Set logger (uses setting)

# +
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
# -

# Create output path for this version

version_path = settings.out_path / settings.version
version_path.mkdir(parents=True, exist_ok=True)

# # Read and process IAM data

# Define some useful functions

# +
import pandas as pd

# IAMC data to lower case
def iamc_to_lowercase(df):
    for col in ["Model", "Scenario", "Region", "Variable", "Unit"]:
        if col in df.columns:
            df.rename(columns={col: col.lower()}, inplace=True)
    return df

# Load IAMC data
def load_data(file_path):
    file_path_str = str(file_path)
    if file_path_str.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path_str.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Use .csv or .xlsx.")

    df = iamc_to_lowercase(df)
    return df

def iamc_wide_to_long(df, iamc_cols=["model", "scenario", "variable", "region", "unit"]):
    year_columns = [col for col in df.columns[len(iamc_cols):] if col.isdigit()]
    if not year_columns:
        raise KeyError("Year columns could not be identified. Ensure the dataframe has year columns after the basic IAMC columns.")

    long_df = df.melt(
        id_vars=iamc_cols,
        value_vars=year_columns,
        var_name="year",
        value_name="value",
    )

    long_df["year"] = pd.to_numeric(long_df["year"], errors="coerce")
    long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")
    long_df.dropna(subset=["year", "value"], inplace=True)
    long_df["year"] = long_df["year"].astype(int)

    return long_df

def filter_scenario(df, scenarios):
    if isinstance(scenarios, list):
        return df[df['scenario'].isin(scenarios)]
    return df[df['scenario'] == scenarios]

def filter_region(df, regions):
    if isinstance(regions, list):
        return df[df['region'].isin(regions)]
    return df[df['region'] == regions]

def filter_variable(df, variables):
    if isinstance(variables, list):
        return df[df['variable'].isin(variables)]
    return df[df['variable'] == variables]

def filter_region_contains(df, substrings):
    if isinstance(substrings, list):
        return df[df['region'].str.contains('|'.join(substrings), case=False, na=False)]
    return df[df['region'].str.contains(substrings, case=False, na=False)]

def filter_emissions_data(df):
    return df[df['variable'].str.startswith("Emissions")]

def rename_variable(df, old_string, new_string):
    df['variable'] = df['variable'].replace({old_string: new_string})
    return df

def filter_regions_only_world_and_model_native(df, cmip7_iam_list=None):
    if cmip7_iam_list is None:
        cmip7_iam_list = ["MESSAGE", "AIM", "COFFEE", "GCAM", "IMAGE", "REMIND", "WITCH"]

    world_df = filter_region(df, regions="World")
    model_native_df = filter_region_contains(df, substrings=cmip7_iam_list)

    return pd.concat([world_df, model_native_df], axis=0)

def reformat_IAMDataframe_with_species_column(df):
    df['species'] = df['variable'].str.lstrip("Emissions|").str.split('|').str[0]
    df['level'] = df['variable'].str.count(r'\|') + 1
    df = df[df['level'] >= 2]
    df['variable'] = df['variable'].where(df['level'] != 2, other="Total")
    df['variable'] = df['variable'].str.replace(r"^[^|]*\|[^|]*\|", "", regex=True)
    df.drop(columns=['level'], inplace=True)
    return df

def sum_selected_variables(df, selected_variables, new_variable_name, group_cols=["model", "scenario", "region", "unit", "year"]):
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
    df = pd.concat([df, summed], axis=0)
    return df

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

def process_data(df, group_cols=["model", "scenario", "region", "unit", "year", "species"]):
    df = process_transportation_variables(df, group_cols=group_cols)
    df = process_industrial_sector_variables(df, group_cols=group_cols)
    return df

def save_data(df, output_path):
    if output_path.endswith('.csv'):
        df.to_csv(output_path, index=False)
    elif output_path.endswith('.xlsx'):
        df.to_excel(output_path, index=False)
    else:
        raise ValueError("Unsupported file format. Use .csv or .xlsx.")

# -

# Define some useful functions (with dask)

# +
# # IAMC data to lower case
# def iamc_to_lowercase(df):
#     # Normalize column names to lowercase if they start with a capital letter
#     for col in ["Model", "Scenario", "Region", "Variable", "Unit"]:
#         if col in df.columns:
#             df.rename(columns={col: col.lower()}, inplace=True)

#     return df

# # Load IAMC data
# def load_data(file_path):
#     """Load IAMC data using pandas or dask."""
#     file_path_str = str(file_path)
#     if file_path_str.endswith('.csv'):
#         df = pd.read_csv(file_path)
#     elif file_path_str.endswith('.xlsx'):
#         df = pd.read_excel(file_path)
#     else:
#         raise ValueError("Unsupported file format. Use .csv or .xlsx.")
    
#     df = iamc_to_lowercase(df)

#     return df

# def iamc_wide_to_long(df, iamc_cols = ["model", "scenario", "variable", "region", "unit"]):
#     """Convert IAMC wide-format data to long format.

#     Assumes by default that all five basic IAMC columns are present (model, scenario, 
#     variable, unit, region), followed by year columns.

#     Assumes that all `iamc_cols` are the left-most columns of the dataframe.

#     Args:
#         df (DataFrame or Dask DataFrame): IAMC wide-format data.

#     Returns:
#         DataFrame or Dask DataFrame: IAMC data in long format with lowercase column names.
#     """
#     # Identify the year columns, assuming they start after the `iamc_cols` columns
#     iamc_col_number = len(iamc_cols)
#     try:
#         year_columns = df.columns[5:]
#         year_columns = [col for col in year_columns if col.isdigit()]
#     except AttributeError:
#         raise KeyError("Year columns could not be identified. Ensure the dataframe has year columns after the basic IAMC columns.")

#     # Melt the DataFrame to long format
#     long_df = df.melt(
#         id_vars=iamc_cols,
#         value_vars=year_columns,
#         var_name="year",
#         value_name="value",
#     )

#     # Convert year and value to numeric and drop rows with missing values
#     long_df["year"] = dd.to_numeric(long_df["year"], errors="coerce")
#     long_df["value"] = dd.to_numeric(long_df["value"], errors="coerce")
#     long_df = long_df.dropna(subset=["year", "value"]).map_partitions(
#         lambda partition: partition.astype({"year": "int64"})
#     )

#     return long_df

# def iamc_wide_to_long(df):
#     """Convert IAMC wide-format data to long format.

#     Assumes all five basic IAMC columns are present (model, scenario, variable, unit, region),
#     followed by year columns.

#     Args:
#         df (DataFrame or Dask DataFrame): IAMC wide-format data.

#     Returns:
#         DataFrame or Dask DataFrame: IAMC data in long format with lowercase column names.
#     """
#     # Identify the year columns, assuming they start after the fifth column
#     try:
#         year_columns = df.columns[5:]
#         year_columns = [col for col in year_columns if col.isdigit()]
#     except AttributeError:
#         raise KeyError("Year columns could not be identified. Ensure the dataframe has year columns after the basic IAMC columns.")

#     # Melt the DataFrame to long format
#     long_df = df.melt(
#         id_vars=["model", "scenario", "region", "variable", "unit"],
#         value_vars=year_columns,
#         var_name="year",
#         value_name="value",
#     )

#     # Convert year and value to numeric and drop rows with missing values
#     long_df["year"] = dd.to_numeric(long_df["year"], errors="coerce")
#     long_df["value"] = dd.to_numeric(long_df["value"], errors="coerce")
#     long_df = long_df.dropna(subset=["year", "value"]).map_partitions(
#         lambda partition: partition.astype({"year": "int64"})
#     )

#     return long_df

# def filter_scenario(df, scenarios):
#     """Filter the dataframe to only keep rows where 'scenario' matches the given value or list of values.

#     Args:
#         df (DataFrame): The input dataframe.
#         scenarios (str or list): The scenario(s) to filter by.

#     Returns:
#         DataFrame: The filtered dataframe.
#     """
#     if isinstance(scenarios, list):
#         return df[df['scenario'].isin(scenarios)]
#     return df[df['scenario'] == scenarios]

# def filter_region(df, regions):
#     """Filter the dataframe to only keep rows where 'region' matches the given value or list of values.

#     Args:
#         df (DataFrame): The input dataframe.
#         regions (str or list): The region(s) to filter by.

#     Returns:
#         DataFrame: The filtered dataframe.
#     """
#     if isinstance(regions, list):
#         return df[df['region'].isin(regions)]
#     return df[df['region'] == regions]

# def filter_variable(df, variables):
#     """Filter the dataframe to only keep rows where 'variable' matches the given value or list of values.

#     Args:
#         df (DataFrame): The input dataframe.
#         variables (str or list): The variable(s) to filter by.

#     Returns:
#         DataFrame: The filtered dataframe.
#     """
#     if isinstance(variables, list):
#         return df[df['variable'].isin(variables)]
#     return df[df['variable'] == variables]

# def filter_region_contains(df, substrings):
#     """Filter the dataframe to only keep rows where 'region' contains the given substring or list of substrings.

#     Args:
#         df (DataFrame): The input dataframe.
#         substrings (str or list): The substring(s) to match within 'region'.

#     Returns:
#         DataFrame: The filtered dataframe.
#     """
#     if isinstance(substrings, list):
#         return df[df['region'].str.contains('|'.join(substrings), case=False, na=False)]
#     return df[df['region'].str.contains(substrings, case=False, na=False)]

# def filter_emissions_data(df):
#     """Filter the dataframe to only keep rows where 'variable' starts with 'Emissions'."""
#     return df[df['variable'].str.startswith("Emissions")]


# def rename_variable(df, old_string, new_string):
#     """Rename rows in the 'variable' column that perfectly match the old_string with new_string.

#     Args:
#         df (DataFrame or Dask DataFrame): The input dataframe.
#         old_string (str): The exact string to be replaced.
#         new_string (str): The replacement string.

#     Returns:
#         DataFrame or Dask DataFrame: The dataframe with updated 'variable' column.
#     """
#     df['variable'] = df['variable'].map_partitions(
#         lambda partition: partition.replace({old_string: new_string})
#         if old_string in partition.unique()
#         else partition
#     )
#     return df


# def filter_regions_only_world_and_model_native(df, cmip7_iam_list=None):
#     """Filter the dataframe to include only rows where region is 'World' or contains specific IAM model substrings.

#     Args:
#         df (DataFrame): The input dataframe.
#         cmip7_iam_list (list): List of substrings to match for IAM models in the 'region' column. Defaults to a preset list.

#     Returns:
#         DataFrame: The filtered dataframe.
#     """
#     if cmip7_iam_list is None:
#         cmip7_iam_list = [
#             "MESSAGE",
#             "AIM",
#             "COFFEE",
#             "GCAM",
#             "IMAGE",
#             "REMIND",
#             "WITCH"
#         ]

#     world_df = filter_region(df, regions="World")
#     model_native_df = filter_region_contains(df, substrings=cmip7_iam_list)

#     return dd.concat([world_df, model_native_df], axis=0)

# def reformat_IAMDataframe_with_species_column(df):
#     """Add a 'species' column and strip everything before and including the second pipe ('|') from the 'variable' column.

#     Args:
#         df (DataFrame or Dask DataFrame): The input dataframe.

#     Returns:
#         DataFrame or Dask DataFrame: The reformatted dataframe.
#     """
#     # Add the species column
#     df['species'] = df['variable'].str.lstrip("Emissions|").str.split('|').str[0]

#     # Calculate the level count as the number of pipe characters + 1
#     df['level'] = df['variable'].str.count(r'\|') + 1

#     # Remove variables that do not have at least 2 levels
#     df = df[df['level'] >= 2]

#     # For variables with exactly 2 levels, rename to total
#     df['variable'] = df['variable'].where(df['level'] != 2, other="Total")

#     # Strip everything before and including the second pipe ('|')
#     df['variable'] = df['variable'].str.replace(r"^[^|]*\|[^|]*\|", "", regex=True)

#     # Drop the auxiliary 'level' column
#     df = df.drop(columns=['level'])

#     return df

# def sum_selected_variables(df, selected_variables, new_variable_name, group_cols = ["model", "scenario", "region", "unit", "year"]):
#     """Sum specific variables into a new row.

#     Args:
#         df (DataFrame): The input dataframe.
#         selected_variables (list): List of variables to sum.
#         new_variable_name (str): Name of the new aggregated variable.

#     Returns:
#         DataFrame: The dataframe with the new aggregated variable.
#     """
#     if not isinstance(selected_variables, list):
#         raise ValueError("selected_variables must be a list.")
#     if not isinstance(new_variable_name, str):
#         raise ValueError("new_variable_name must be a string.")

#     selected_df = df[df['variable'].isin(selected_variables)]
#     summed = (
#         selected_df
#         .groupby(group_cols)
#         .agg({"value": "sum"})
#         .reset_index()
#     )
#     summed["variable"] = new_variable_name
    
#     # Append the summed data to the dataframe
#     df = dd.concat([df, summed], axis=0)
#     return df

# def _subtract_domestic_from_transportation(partition):
#     """Subtract domestic aviation values from transportation values.

#     Args:
#         partition (DataFrame): A partition of the transportation DataFrame.

#     Returns:
#         DataFrame: The updated transportation partition with adjusted values.
#     """
#     # Ensure domestic aviation values exist before subtraction
#     if "value_dom" in partition.columns:
#         partition["value"] -= partition["value_dom"].fillna(0)
#     return partition.drop(columns=["value_dom"], errors="ignore")


# def process_transportation_variables(
#     df,
#     group_cols=["model", "scenario", "region", "unit", "year"],
#     new_aviation_variable="Aircraft",
#     new_transportation_variable="Transportation Sector",
# ):
#     """Process transportation variables to split between 'Aircraft' and 'Transportation'.

#     Args:
#         df (DataFrame or Dask DataFrame): Input IAMC DataFrame.
#         group_cols (list): Columns to group by for aggregation.
#         new_aviation_variable (str): Name for the new aggregated aviation variable.
#         new_transportation_variable (str): Name for the adjusted transportation variable.

#     Returns:
#         DataFrame or Dask DataFrame: Updated DataFrame with adjusted transportation variables.
#     """
#     av_dom_var = "Energy|Demand|Transportation|Domestic Aviation"
#     av_int_var = "Energy|Demand|Bunkers|International Aviation"
#     trp_var = "Energy|Demand|Transportation"

#     # Aggregate 'International Aviation' and 'Domestic Aviation' into 'Aircraft'
#     aviation_df = df[df["variable"].isin([av_dom_var, av_int_var])]
#     aggregated_aviation = (
#         aviation_df.groupby(group_cols)
#         .agg({"value": "sum"})
#         .reset_index()
#     )
#     aggregated_aviation["variable"] = new_aviation_variable

#     # Append the aggregated aviation data to the main DataFrame
#     df = dd.concat([df, aggregated_aviation], axis=0)

#     # Subtract domestic aviation from transportation
#     domestic_aviation = df[df["variable"] == av_dom_var][group_cols + ["value"]]
#     domestic_aviation = domestic_aviation.rename(columns={"value": "value_dom"})
#     transportation = df[df["variable"] == trp_var]

#     # Merge and adjust transportation values
#     transportation = transportation.merge(
#         domestic_aviation,
#         on=group_cols,
#         how="left"
#     )
#     transportation = transportation.map_partitions(_subtract_domestic_from_transportation)

#     # Update the variable name for transportation
#     transportation["variable"] = new_transportation_variable

#     # Combine back the adjusted transportation data
#     df = dd.concat([df[df["variable"] != trp_var], transportation], axis=0)

#     return df

# # def _subtract_domestic_from_transportation(partition, meta):
# #     """Subtract domestic aviation values from transportation values.

# #     Args:
# #         partition (DataFrame): A partition of the transportation DataFrame.
# #         meta (dict): Metadata to define the output structure of the DataFrame.

# #     Returns:
# #         DataFrame: The updated transportation partition.
# #     """
# #     # Merge domestic aviation and transportation data
# #     merged = partition.merge(
# #         meta['domestic_aviation'],
# #         on=meta['group_cols'],
# #         suffixes=("", "_dom"),
# #         how="left",
# #     )
# #     # Subtract domestic aviation values from transportation values
# #     merged["value"] = merged["value"] - merged["value_dom"].fillna(0)
# #     return merged.drop(columns=["value_dom"])

# # def process_transportation_variables(
# #     df,
# #     group_cols=["model", "scenario", "region", "unit", "year"],
# #     new_aviation_variable="Aircraft",
# #     new_transportation_variable="Transportation Sector",
# # ):
# #     """Process transportation variables to split between 'Aircraft' and 'Transportation'.

# #     Args:
# #         df (DataFrame or Dask DataFrame): Input IAMC DataFrame.
# #         group_cols (list): Columns to group by for aggregation.
# #         new_aviation_variable (str): Name for the new aggregated aviation variable.
# #         new_transportation_variable (str): Name for the adjusted transportation variable.

# #     Returns:
# #         DataFrame or Dask DataFrame: Updated DataFrame with adjusted transportation variables.
# #     """
# #     av_dom_var = "Energy|Demand|Transportation|Domestic Aviation"
# #     av_int_var = "Energy|Demand|Bunkers|International Aviation"
# #     trp_var = "Energy|Demand|Transportation"

# #     # Aggregate 'International Aviation' and 'Domestic Aviation' into 'Aircraft'
# #     aviation_df = df[df["variable"].isin([av_dom_var, av_int_var])]
# #     aggregated = (
# #         aviation_df.groupby(group_cols)
# #         .agg({"value": "sum"})
# #         .reset_index()
# #     )
# #     aggregated["variable"] = new_aviation_variable

# #     # Append the aggregated aviation data
# #     df = dd.concat([df, aggregated], axis=0)

# #     # Subtract domestic aviation from transportation
# #     domestic_aviation = df[df["variable"] == av_dom_var]
# #     transportation = df[df["variable"] == trp_var]

# #     meta = {
# #         "domestic_aviation": domestic_aviation.compute(),
# #         "group_cols": group_cols,
# #     }

# #     transportation = transportation.map_partitions(
# #         _subtract_domestic_from_transportation,
# #         meta=domestic_aviation,
# #     )

# #     transportation["variable"] = new_transportation_variable

# #     # Combine the adjusted transportation data back
# #     df = dd.concat([df[~(df["variable"] == trp_var)], transportation], axis=0)

# #     return df

# def process_industrial_sector_variables(df, 
#                                         industry_variable_list=None, 
#                                         group_cols=["model", "scenario", "region", "unit", "year"]):

#     if industry_variable_list is None:
#             industry_variable_list = [
#                 "Energy|Supply", 
#                 "Energy|Demand|Industry", 
#                 "Energy|Demand|Other Sector", 
#                 "Industrial Processes", 
#                 "Other"
#             ]

#     df = sum_selected_variables(df, 
#                                 selected_variables=industry_variable_list, 
#                                 new_variable_name="Industrial Sector",
#                                 group_cols=group_cols)

#     return df

# def process_data(df, group_cols = ["model", "scenario", "region", "unit", "year", "species"]):
#     """Process the IAMC data to aggregate aviation emissions, modify transportation variables, and sum industrial sector variables.

#     Args:
#         df (DataFrame or Dask DataFrame): The input IAMC DataFrame.

#     Returns:
#         DataFrame or Dask DataFrame: The processed IAMC DataFrame.
#     """
#     # # Ensure the dataframe is a Dask DataFrame for scalability
#     # if not isinstance(df, dd.DataFrame):
#     #     df = dd.from_pandas(df, npartitions=4) # TODO: automatically choose npartitions

#     # Modify transportation variables, to split between "Aircraft" and (other) "Transportation"
#     df = process_transportation_variables(df, group_cols=group_cols)

#     # Aggregate industrial sector emissions
#     df = process_industrial_sector_variables(df, group_cols=group_cols)
      
#     return df

# def save_data(df, output_path):
#     """Save the modified IAMC data to a file."""
#     if isinstance(df, dd.DataFrame):
#         df = df.compute()

#     if output_path.endswith('.csv'):
#         df.to_csv(output_path, index=False)
#     elif output_path.endswith('.xlsx'):
#         df.to_excel(output_path, index=False)
#     else:
#         raise ValueError("Unsupported file format. Use .csv or .xlsx.")

# # Example usage
# # df = load_data("scenario_timeseries.csv")
# # processed_df = process_data(df)
# # save_data(processed_df, "processed_timeseries.csv")

# -

# ### Read in

# read in
iam_df = load_data(
    Path(settings.scenario_path, SCENARIO_FILE) 
)


# ### Filter

iam_df = filter_emissions_data(iam_df)
iam_df = filter_scenario(iam_df, scenarios="SSP1 - Low Emissions") # TODO: remove after test code is done
iam_df = filter_regions_only_world_and_model_native(iam_df)

# ### Process

# prepare
iam_df = iamc_wide_to_long(iam_df)
iam_df = reformat_IAMDataframe_with_species_column(iam_df)

# process data
iam_df = process_data(iam_df) # do calculations for aviation, transportation, and industrial sector

# +
# rename variable to harmonization sectors
# aviation, transportation, and industrial sector are already handled inside `process_data`
iam_df = rename_variable(iam_df, old_string="Energy|Supply", new_string="Energy Sector")
iam_df = rename_variable(iam_df, old_string="Energy|Demand|Bunkers|International Shipping", new_string="International Shipping")
iam_df = rename_variable(iam_df, old_string="Residential Commercial Other", new_string="Energy|Demand|Residential and Commercial and AFOFI")
iam_df = rename_variable(iam_df, old_string="Solvents Production and Application", new_string="Product Use")
iam_df = rename_variable(iam_df, old_string="Agriculture", new_string="AFOLU Agriculture")
iam_df = rename_variable(iam_df, old_string="AFOLU|Agricultural Waste Burning", new_string="Agricultural Waste Burning")
iam_df = rename_variable(iam_df, old_string="AFOLU|Land|Fires|Forest Burning", new_string="Forest Burning")
iam_df = rename_variable(iam_df, old_string="AFOLU|Land|Fires|Grassland Burning", new_string="Grassland Burning")
iam_df = rename_variable(iam_df, old_string="AFOLU|Land|Fires|Peat Burning", new_string="Peat Burning")
# iam_df = rename_variable(iam_df, old_string="Waste", new_string="Waste")

# keep only harmonization sectors
scens_iam = filter_variable(iam_df,
                            variables=[
                                "Energy Sector",
                                "Industrial Sector",
                                "Aircraft",
                                "Transportation Sector",
                                "International Shipping",
                                "Residential Commercial Other",
                                "Solvents Production and Application",
                                "Agriculture",
                                "Agricultural Waste Burning",
                                "Forest Burning",
                                "Grassland Burning",
                                "Peat Burning",
                                "Waste"
                            ])


# # TODO: add CDR and gross emissions variables
# -

# ## Process units

# +
# def patch_model_variable(var):
#     if var == settings.alkalinity_variable:
#         var = settings.variable_template.format(gas="TA", sector="Alkalinity Addition")
#     elif var.endswith("|CO2|Aggregate - Agriculture and LUC"):
#         var = var.replace(
#             "|Aggregate - Agriculture and LUC", "|Deforestation and other LUC"
#         )
#     elif var.endswith("|Energy Sector"):
#         var += "|Modelled"
#     return var
# -

with ur.context("AR4GWP100"):
    model = (
        pd.read_csv(
            settings.scenario_path / "REMIND-MAgPIE-CEDS-RESCUE-Tier1-2024-08-19.csv",
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
        # .rename(index=patch_model_variable, level="Variable")
        .pipe(
            variabledefs.load_data,
            extend_missing=True,
            levels=["model", "scenario", "region", "gas", "sector", "unit"],
            settings=settings,
        )
    )
model.pix

#
model = model.fillna(0)

# ## Save the processed IAM data

save_data(df = scens_iam,    
          output_path = str(Path(version_path, "processed_scenarios.csv" )))

# # Read definitions

# ## Read variable definitions
#
# The variable definition file is a CSV or yaml file that needs to contain the `variable`-name, its `sector`, `gas` components and whether it is expected `global` (or regional instead).
#
# Here we use a file based on the RESCUE variable definitions, but adapted to fit CMIP7 purposes.
#

variabledefs = VariableDefinitions.from_csv(settings.variabledefs_path)
variabledefs.data.head()

# ## Read region definitions (using RegionMapping class)
#

settings.data_path

settings.regionmappings.items()

# +
regionmappings = {}

for model, kwargs in settings.regionmappings.items():
    regionmapping = RegionMapping.from_regiondef(**kwargs)
    regionmapping.data = regionmapping.data.pix.aggregate(
        country=settings.country_combinations, agg_func="last"
    )
    regionmappings[model] = regionmapping

regionmappings
# -

# # Read historic data
#
# Can be read in and prepared using `read_iamc` or the `variabledefs`
#

hist_ceds = (
    pd.read_csv(
        settings.history_path / "ceds_2017_extended.csv",
        index_col=list(range(4)),
    )
    .rename(index={"NMVOC": "VOC", "SO2": "Sulfur"}, level="gas")
    .rename(index={"Mt NMVOC/yr": "Mt VOC/yr"}, level="unit")
    .rename(columns=int)
    .pix.format(variable=settings.variable_template, drop=True)
    .pix.assign(model="History", scenario="CEDS")
)


# +
def patch_global_hist_variable(var):
    var = var.removesuffix("|Unharmonized")
    if any(
        var.endswith(s)
        for s in ("|Aggregate - Agriculture and LUC", "|CDR Afforestation")
    ):
        # TODO upstream into `global_trajectories.xlsx` once this is on main
        var = var.replace(
            "|Aggregate - Agriculture and LUC", "|Deforestation and other LUC"
        )
        return var

    return f"{var}|Total"


hist_global = (
    pd.read_excel(
        settings.history_path / "global_trajectories.xlsx",
        index_col=list(range(5)),
    )
    .rename_axis(index=str.lower)
    .rename_axis(index={"region": "country"})
    .rename(
        index=patch_global_hist_variable,
        level="variable",
    )
)
# -

hist_gfed = pd.read_csv(
    settings.history_path / "gfed/GFED2015_extended.csv",
    index_col=list(range(5)),
).rename(columns=int)

hist = (
    concat([hist_ceds, hist_global, hist_gfed])
    .droplevel(["model", "scenario"])
    .pix.aggregate(country=settings.country_combinations)
    .pipe(
        variabledefs.load_data,
        extend_missing=True,
        levels=["country", "gas", "sector", "unit"],
        settings=settings,
    )
)
hist.head()

# # Read Harmonization Overrides

harm_overrides = pd.read_excel(
    settings.scenario_path / "harmonization_overrides.xlsx",
    index_col=list(range(3)),
).method
harm_overrides

harm_overrides = extend_overrides(
    harm_overrides,
    "constant_ratio",
    sector=[
        f"{sec} Burning"
        for sec in ["Agricultural Waste", "Forest", "Grassland", "Peat"]
    ],
    variables=variabledefs.data.index,
    regionmappings=regionmappings,
    model_baseyear=model[settings.base_year],
)

# # Prepare GDP proxy
#
# Read in different GDP scenarios for SSP1 to SSP5 from SSP DB.
#

gdp = (
    pd.read_csv(
        settings.scenario_path / "SspDb_country_data_2013-06-12.csv",
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
    .pix.aggregate(country=settings.country_combinations)
)

# Determine likely SSP for each harmonized pathway from scenario string and create proxy data aligned with pathways
#

SSP_per_pathway = (
    model.index.pix.project(["model", "scenario"])
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

# Test with one scenario only
one_scenario = False
only_direct = True
if one_scenario:
    model = model.loc[ismatch(scenario="RESCUE-Tier1-Direct-*-PkBudg500-OAE_on")]
elif only_direct:
    model = model.loc[ismatch(scenario="RESCUE-Tier1-Direct-*")]
logger().info(
    "Running with %d scenario(s):\n- %s",
    len(model.pix.unique(["model", "scenario"])),
    "\n- ".join(model.pix.unique("scenario")),
)

client = Client()
# client.register_plugin(DaskSetWorkerLoglevel(logger().getEffectiveLevel()))
client.forward_logging()

dask.distributed.gc.disable_gc_diagnosis()

(model_name,) = model.pix.unique("model")
regionmapping = regionmappings[model_name]

indexraster = IndexRaster.from_netcdf(
    settings.gridding_path / "ssp_comb_indexraster.nc",
    chunks={},
).persist()
indexraster_region = indexraster.dissolve(
    regionmapping.filter(indexraster.index).data.rename("country")
).persist()

workflow = WorkflowDriver(
    model,
    hist,
    gdp,
    regionmapping.filter(gdp.pix.unique("country")),
    indexraster,
    indexraster_region,
    variabledefs,
    harm_overrides,
    settings,
)

# # Harmonize, downscale and grid everything
#
# Latest test with 2 scenarios was 70 minutes for everything on MacBook

# ## Alternative 1) Run full processing and create netcdf files

res = workflow.grid(
    template_fn="{{name}}_{activity_id}_emissions_{target_mip}_{institution}-{{model}}-{{scenario}}_{grid_label}_201501-210012.nc".format(
        **rescue_utils.DS_ATTRS | {"version": settings.version}
    ),
    callback=rescue_utils.DressUp(version=settings.version),
    directory=version_path,
    skip_exists=True,
)

# ## Alternative 2) Harmonize and downscale everything, but do not grid
#
# If you also want grids, use the gridding interface directly
#

downscaled = workflow.harmonize_and_downscale()

# ## Alternative 3) Investigations

# ### Process single proxy
#
# `workflow.grid_proxy` returns an iterator of the gridded scenarios. We are looking at the first one in depth.

gridded = next(workflow.grid_proxy("CO2_em_anthro"))

ds = gridded.prepare_dataset(callback=rescue_utils.DressUp(version=settings.version))
ds

gridded.to_netcdf(
    template_fn="{{name}}_{activity_id}_emissions_{target_mip}_{institution}-{{model}}-{{scenario}}_{grid_label}_201501-210012.nc".format(
        **rescue_utils.DS_ATTRS | {"version": settings.version}
    ),
    callback=rescue_utils.DressUp(version=settings.version),
    directory=version_path,
)

ds["CO2_em_anthro"].sel(sector="CDR OAE", time="2015-09-16").plot()

ds.isnull().any(["time", "lat", "lon"])["CO2_em_anthro"].to_pandas()

reldiff, _ = dask.compute(
    gridded.verify(compute=False),
    gridded.to_netcdf(
        template_fn=(
            "{{name}}_{activity_id}_emissions_{target_mip}_{institution}-"
            "{{model}}-{{scenario}}-{version}_{grid_label}_201501-210012.nc"
        ).format(**rescue_utils.DS_ATTRS | {"version": settings.version}),
        callback=rescue_utils.DressUp(version=settings.version),
        encoding_kwargs=dict(_FillValue=1e20),
        compute=False,
        directory=version_path,
    ),
)
reldiff

# ### Regional proxy weights

gridded.proxy.weight.regional.sel(
    sector="Transportation Sector", year=2050, gas="CO2"
).compute().to_pandas().plot.hist(bins=100, logx=True, logy=True)


# ## Export harmonized scenarios
#


def rename_alkalinity_addition(df):
    return df.rename(
        lambda v: v.replace(
            settings.variable_template.format(gas="TA", sector="Alkalinity Addition"),
            settings.alkalinity_variable,
        ),
        level="variable",
    )


data = (
    workflow.harmonized_data.add_totals()
    .to_iamc(settings.variable_template, hist_scenario="Synthetic (GFED/CEDS/Global)")
    .pipe(rename_alkalinity_addition)
    .rename_axis(index=str.capitalize)
)
data.to_csv(version_path / f"harmonization-{settings.version}.csv")

# ### Split HFC distributions
#

# +
hfc_distribution = (
    pd.read_excel(
        settings.postprocess_path / "rescue_hfc_scenario.xlsx",
        index_col=0,
        sheet_name="velders_2015",
    )
    .rename_axis("hfc")
    .rename(columns=int)
)

data = (
    workflow.harmonized_data.drop_method()
    .add_totals()
    .aggregate_subsectors()
    .split_hfc(hfc_distribution)
    .to_iamc(settings.variable_template, hist_scenario="Synthetic (GFED/CEDS/Global)")
    .pipe(rename_alkalinity_addition)
    .rename_axis(index=str.capitalize)
)
data.to_csv(version_path / f"harmonization-{settings.version}-splithfc.csv")
# -

# # Export downscaled results
#
# TODO: create a similar exporter to the Harmonized class for Downscaled which combines historic and downscaled data (maybe also harmonized?) and translates to iamc
#

# Do we also want to render this as IAMC?
workflow.downscaled.data.to_csv(
    version_path / f"downscaled-only-{settings.version}.csv"
)

# # Upload to BSC FTP
#

remote_path = Path("/forcings/emissions") / settings.version
rescue_utils.ftp_upload(settings.ftp, version_path, remote_path)


