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

# # IAM: Read and process IAM data

# ### Define some useful functions

# +
# import pandas as pd

# # IAMC data to lower case
# def iamc_to_lowercase(df):
#     for col in ["Model", "Scenario", "Region", "Variable", "Unit"]:
#         if col in df.columns:
#             df.rename(columns={col: col.lower()}, inplace=True)
#     return df

# # Load IAMC data
# def load_data(file_path):
#     file_path_str = str(file_path)
#     if file_path_str.endswith('.csv'):
#         df = pd.read_csv(file_path)
#     elif file_path_str.endswith('.xlsx'):
#         df = pd.read_excel(file_path)
#     else:
#         raise ValueError("Unsupported file format. Use .csv or .xlsx.")

#     df = iamc_to_lowercase(df)
#     return df

# def iamc_wide_to_long(df, iamc_cols=["model", "scenario", "variable", "region", "unit"]):
#     year_columns = [col for col in df.columns[len(iamc_cols):] if col.isdigit()]
#     if not year_columns:
#         raise KeyError("Year columns could not be identified. Ensure the dataframe has year columns after the basic IAMC columns.")

#     long_df = df.melt(
#         id_vars=iamc_cols,
#         value_vars=year_columns,
#         var_name="year",
#         value_name="value",
#     )

#     long_df["year"] = pd.to_numeric(long_df["year"], errors="coerce")
#     long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")
#     long_df.dropna(subset=["year", "value"], inplace=True)
#     long_df["year"] = long_df["year"].astype(int)

#     return long_df

# def filter_scenario(df, scenarios):
#     if isinstance(scenarios, list):
#         return df[df['scenario'].isin(scenarios)]
#     return df[df['scenario'] == scenarios]

# def filter_region(df, regions):
#     if isinstance(regions, list):
#         return df[df['region'].isin(regions)]
#     return df[df['region'] == regions]

# def filter_variable(df, variables):
#     if isinstance(variables, list):
#         return df[df['variable'].isin(variables)]
#     return df[df['variable'] == variables]

# def filter_region_contains(df, substrings):
#     if isinstance(substrings, list):
#         return df[df['region'].str.contains('|'.join(substrings), case=False, na=False)]
#     return df[df['region'].str.contains(substrings, case=False, na=False)]

# def filter_emissions_data(df):
#     return df[df['variable'].str.startswith("Emissions")]

# def rename_one_variable(df, old_string, new_string):
#     df['variable'] = df['variable'].replace({old_string: new_string})
#     return df

# def rename_variable(df, rename_dict):
#     """
#     Renames variables in a dataframe's 'variable' column based on a dictionary mapping.

#     Parameters:
#         df (pd.DataFrame): The dataframe to modify.
#         rename_dict (dict): A dictionary where keys are old strings and values are new strings.

#     Returns:
#         pd.DataFrame: The modified dataframe with updated variable names.
#     """
#     df['variable'] = df['variable'].replace(rename_dict)
#     return df

# def filter_regions_only_world_and_model_native(df, cmip7_iam_list=None):
#     if cmip7_iam_list is None:
#         cmip7_iam_list = ["MESSAGE", "AIM", "COFFEE", "GCAM", "IMAGE", "REMIND", "WITCH"]

#     world_df = filter_region(df, regions="World")
#     model_native_df = filter_region_contains(df, substrings=cmip7_iam_list)

#     return pd.concat([world_df, model_native_df], axis=0)

# def reformat_IAMDataframe_with_species_column(df):
#     df['species'] = df['variable'].str.lstrip("Emissions|").str.split('|').str[0]
#     df['level'] = df['variable'].str.count(r'\|') + 1
#     df = df[df['level'] >= 2]
#     df['variable'] = df['variable'].where(df['level'] != 2, other="Total")
#     df['variable'] = df['variable'].str.replace(r"^[^|]*\|[^|]*\|", "", regex=True)
#     df.drop(columns=['level'], inplace=True)
#     return df

# def sum_selected_variables(df, selected_variables, new_variable_name, group_cols=["model", "scenario", "region", "unit", "year"]):
#     if not isinstance(selected_variables, list):
#         raise ValueError("selected_variables must be a list.")
#     if not isinstance(new_variable_name, str):
#         raise ValueError("new_variable_name must be a string.")

#     selected_df = df[df['variable'].isin(selected_variables)]
#     summed = (
#         selected_df
#         .groupby(group_cols, as_index=False)
#         .agg({"value": "sum"})
#     )
#     summed["variable"] = new_variable_name
#     df = pd.concat([df, summed], axis=0)
#     return df

# def process_data(df, group_cols=["model", "scenario", "region", "unit", "year", "species"]):
#     df = process_transportation_variables(df, group_cols=group_cols)
#     df = process_industrial_sector_variables(df, group_cols=group_cols)
#     return df

# def save_data(df, output_path):
#     if output_path.endswith('.csv'):
#         df.to_csv(output_path, index=False)
#     elif output_path.endswith('.xlsx'):
#         df.to_excel(output_path, index=False)
#     else:
#         raise ValueError("Unsupported file format. Use .csv or .xlsx.")


# +
import pandas as pd

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

# Advanced processing
# def reformat_IAMDataframe_with_species_column(df, start_string = "Emissions|", end_string = ""):
#     """
#     Extracts species from e.g. "Emissions|" variable names and reformats the dataframe.

#     Parameters:
#         df (pd.DataFrame): The dataframe to process.

#     Returns:
#         pd.DataFrame: The reformatted dataframe.
#     """
#     df['species'] = df['variable'].str.lstrip(start_string).str.split('|').str[0]
#     df['level'] = df['variable'].str.count(r'\|') + 1
#     df = df[df['level'] >= 2]
#     df['variable'] = df['variable'].where(df['level'] != 2, other="Total")
#     df['variable'] = df['variable'].str.replace(r"^[^|]*\|[^|]*\|", "", regex=True)
#     df.drop(columns=['level'], inplace=True)
#     return df
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

    # Count levels based on the number of '|' characters
    df['level'] = df['variable'].str.count(r'\|') + 1

    # create the sector column
    df['sector'] = df['variable'].str.split('|').str[1].where(df['level'] == 2)
    # Replace sector names with "Total" if the variable only has the information of the species
    df['sector'] = df['variable'].where(df['variable'] != df['species'], other="Total")
    
    # Drop the 'level' column
    df.drop(columns=['level'], inplace=True)
    return df


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

iam_df

# process data
iam_df = process_data(iam_df,  group_cols=["model", "scenario", "region", "unit", "year", "species"]) # do calculations for aviation, transportation, and industrial sector

# +
# rename variable to harmonization sectors
# Dictionary for renaming variables
rename_dict = {
    # aviation, transportation, and industrial sector are already handled inside `process_data`
    "Energy|Supply": "Energy Sector",
    "Energy|Demand|Bunkers|International Shipping": "International Shipping",
    "Residential Commercial Other": "Energy|Demand|Residential and Commercial and AFOFI",
    "Solvents Production and Application": "Product Use",
    "Agriculture": "AFOLU Agriculture",
    "AFOLU|Agricultural Waste Burning": "Agricultural Waste Burning",
    "AFOLU|Land|Fires|Forest Burning": "Forest Burning",
    "AFOLU|Land|Fires|Grassland Burning": "Grassland Burning",
    "AFOLU|Land|Fires|Peat Burning": "Peat Burning",
    # Uncomment the following line if you want to rename "Waste" to "Waste" explicitly
    # "Waste": "Waste"
}

# Apply the renaming function
iam_df = rename_variable(iam_df, rename_dict)

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

# ## Units

# ### Check units and report on missing IAM data

# +
## NOTE: for now, let's try to stay with common-definitions units, as submitted, and change the variable defs config in line with that.

## TODO: automatically download common-definitions template, as the ultimate reference / source of truth for the variable template.
## - for now; since the IAM download is straight from the Scenario Explorer, this is pretty much the same.

# +
# units_IAM_variable = select_and_distinct(scens_iam, ["variable", "unit"])
# -

different_units_IAM = pd.DataFrame([]) # create dataframe
for m in scens_iam["model"].unique():
    # same dataframe structure
    units_IAM = select_and_distinct(scens_iam, ["variable", "unit"])
    units_defined = select_and_distinct(variabledefs.data.reset_index().rename(columns={"sector": "variable"}), ["variable", "unit"])
    # compare both
    different_units_onemodel = compare_units("IAM", units_IAM, "definitions", units_defined)
    # add back model column
    different_units_onemodel["model"] = m
    # append to dataframe
    different_units_IAM = pd.concat([different_units_IAM, different_units_onemodel], ignore_index=True)
    

save_data(df = different_units_IAM,    
          output_path = str(Path(version_path, "scenarios_missingdata.csv" )))

variabledefs.data

# ### Rename units

# +
# none for now.
# -

# ### Transform units to equivalents

# none for now.
# ...for an example, see here: https://github.com/IAMconsortium/concordia/blob/22bf33b0158839dd870eae019894e78d03fcf27d/notebooks/workflow-rescue.py#L219-L245 


# ### Dealing with NAs

# +
# none for now.
# ...one possible way: `model = model.fillna(0)`
# -

# ## Interpolate IAM data where necessary

# +
# none for now.
# ...likely will have to do (linear) interpolation for the 2020-2025 period for intermediary products
# ...and an option where changes follow historical data between 2020 and the latest year of historical data (noting this could create a larger underlying difference to the 2025 model year, compared to linear interpolation)
# -

# ## Save the processed IAM data

# ### Basic checks

# +
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

check_na_in_columns(scens_iam)
# -

# ### Long format

save_data(df = scens_iam,    
          output_path = str(Path(version_path, "scenarios_processed.csv" )))

# ### Wide format (and `model` variable)

model = (
    scens_iam
    .rename(columns={'species': 'gas', 'variable': 'sector'})
    .pivot_table(index=["model", "scenario", "region", "sector", "gas", "unit"],
                 values="value",
                 columns="year")
)

save_data(df = scens_iam,    
          output_path = str(Path(version_path, "scenarios_processed.csv" )))

# # History: Read and process historical data
#
# Can be read in and prepared using `read_iamc` or the `variabledefs`
#

# ## Read in

# ### CEDS

hist_ceds = (
    reformat_IAMDataframe_with_species_column(
        iamc_wide_to_long(
            pd.read_csv(
                settings.history_path / "ceds_cmip7_national_alpha.csv"
            )
        ),
        start_string="CMIP7 History|Emissions|"
    )
)
# Previous, faster example, using MultiIndex & pandas-indexing, is available here: https://github.com/jkikstra/concordia/blob/41880f678d082699738c515a9610f26a84768641/notebooks/workflow-cmip7.py#L909-L919
# ... this also has code for GFED, Other, and combining 

# #### Unit adjustments

# NMVOC (CEDS) -> VOC (IAM)
hist_ceds['variable'] = hist_ceds['variable'].replace(r'^NMVOC\|', 'VOC|', regex=True)
hist_ceds['unit'] = hist_ceds['unit'].replace('Mt NMVOC/yr', 'Mt VOC/yr', regex=False)
hist_ceds['species'] = hist_ceds['species'].replace('NMVOC', 'VOC', regex=False)
# Mt N2O (CEDS) -> kt N2O (IAM)
hist_ceds.loc[hist_ceds['unit'] == 'Mt N2O/yr', 'value'] *= 1000 # hist_ceds['value'] = hist_ceds['value'].where(hist_ceds['unit'] != 'Mt N2O/yr', other=hist_ceds['value']*1000)
hist_ceds['unit'] = hist_ceds['unit'].replace('Mt N2O/yr', 'kt N2O/yr', regex=False)

hist_ceds['variable'].unique()

# ### GFED

hist_gfed = (
    reformat_IAMDataframe_with_species_column(
        iamc_wide_to_long(
            pd.read_csv(
                settings.history_path / "GFED2015_extended.csv"
            ).rename(columns={'country': 'region'})
        ),
        start_string="CEDS+|9+ Sectors|Emissions|"
    )
)

# #### Unit adjustments

# Unit of NOx is NO2; Mt NOx (GFED) -> Mt NO2 (IAM)
hist_gfed['unit'] = hist_gfed['unit'].replace('Mt NOx/yr', 'Mt NO2/yr', regex=False)

# ### Other (global only)

hist_global = (
    reformat_IAMDataframe_with_species_column(
        iamc_wide_to_long(
            pd.read_excel(
                settings.history_path / "global_trajectories.xlsx"
            )
        ),
        start_string="CEDS+|9+ Sectors|Emissions|", end_string="|Unharmonized"
    )
)
# filter out N2O which now comes from CEDS
hist_global = hist_global[hist_global['species'] != 'N2O']

# ### Combine

hist_long = concat([hist_ceds, hist_global, hist_gfed])

# ## Units

# ### Check units and report on missing historical data

different_units_history = pd.DataFrame([]) # create dataframe
for m in hist_long["model"].unique():
    # same dataframe structure
    units_hist = select_and_distinct(hist_long, ["sector", "unit"])
    units_defined = select_and_distinct(variabledefs.data.reset_index(), ["sector", "unit"])
    # compare both
    different_units_onemodel = compare_units("Historical", units_hist, "definitions", units_defined)
    # add back model column
    different_units_onemodel["model"] = m
    # append to dataframe
    different_units_history = pd.concat([different_units_history, different_units_onemodel], ignore_index=True)

save_data(df = different_units_history,    
          output_path = str(Path(version_path, "history_missingdata.csv" )))

# ### Rename units and variables

# +
# none here; all done at the reading in of each data set.
# -

# ## Save

# ### Long format

save_data(df = hist_long,    
          output_path = str(Path(version_path, "history_processed_longformat.csv" )))

# ### Wide format (and `hist` variable)

hist_wide = (
    hist_long
    .rename(columns={'species': 'gas'})
    .pivot_table(index=["model", "scenario", "region", "variable", "sector", "gas", "unit"],
                 values="value",
                 columns="year")
    .droplevel(["model", "scenario", "variable"])
    # .pix.aggregate(country=settings.country_combinations)
)
save_data(df = hist_wide.reset_index(),    
          output_path = str(Path(version_path, "history_processed_wideformat.csv" )))

# # Read Harmonization Overrides

settings.scenario_path

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
