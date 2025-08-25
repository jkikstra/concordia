# %%
from __future__ import annotations

# %%
import datetime
import ftplib
import logging
from collections.abc import Sequence
from typing import Any

# %%
import cf_xarray  # noqa
import dateutil
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
import pyreadr
from attrs import define
from cattrs import structure, transform_error
from pandas_indexing import concat, isin, semijoin
from tqdm.auto import tqdm

# %%
# rpy2: Python <-> R bridge; needed for rewriting CEDS files in the same format
import rpy2.robjects as ro
from rpy2.robjects import default_converter
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import numpy2ri

# %%
r = ro.r

# %%
from ..settings import FtpSettings


# %%
logger = logging.getLogger(__name__)

# %% [markdown]
# ' TODO: check all these variable names and renamings
# ' - e.g., "Residential Commercial Other"
# ' - e.g., "Residential Commercial Other"


# %%
SECTOR_RENAMES = {
    "Energy Sector": "Energy",
    "Industrial Sector": "Industrial",
    "Transportation Sector": "Transportation",
    "Residential Commercial Other": "Residential, Commercial, Other",
}

# %%
SECTOR_ORDERING_DEFAULT = {
    "em_anthro": [
        "Agriculture",
        "Energy",
        "Industrial",
        "Transportation",
        "Residential, Commercial, Other",
        "Solvents Production and Application",
        "Waste",
        "International Shipping",
    ],
    "em_openburning": [
        "Agricultural Waste Burning",
        "Forest Burning",
        "Grassland Burning",
        "Peat Burning",
    ],
}

# %%
SECTOR_ORDERING_GAS = {
    "CO2_em_anthro": [
        "Agriculture",
        "Energy",
        "Industrial",
        "Transportation",
        "Residential, Commercial, Other",
        "Solvents Production and Application",
        "Waste",
        "International Shipping",
        # "Deforestation and other LUC",
        # "OAE Calcination Emissions",
        # "CDR Afforestation",
        # "CDR BECCS",
        # "CDR DACCS",
        # "CDR EW",
        # "CDR Industry",
        # "CDR OAE Uptake Ocean",
    ]
}

# %%
ATTRS = {
    "lat": {
        "units": "degrees_north",
        "long_name": "latitude",
        "axis": "Y",
        "realtopology": "linear",
        "standard_name": "latitude",
    },
    "lon": {
        "units": "degrees_east",
        "long_name": "longitude",
        "axis": "X",
        "modulo": 360.0,
        "realtopology": "circular",
        "standard_name": "longitude",
        "topology": "circular",
    },
    "time": {
        "long_name": "time",
        "axis": "T",
        "realtopology": "linear",
        "standard_name": "time",
    },
    "level": {
        "units": "km",
        "long_name": "altitude",
    },
}

# %%
DATA_HANDLES = {
    "em_anthro": "anthropogenic",
    "em_AIR_anthro": "aircraft",
    "em_openburning": "openburning",
    "em_removal": "cdr",
}

# %%
DS_ATTRS = dict(
    Conventions="CF-1.6",
    activity_id="input4MIPs",
    comment="Gridded emissions produced after harmonization and downscaling as part of the ScenarioMIP-CMIP7. See https://github.com/iiasa/emissions_harmonization_historical, https://github.com/IAMconsortium/concordia, and https://github.com/iiasa/aneris for documentation on the processes.",
    contact="Jarmo S. Kikstra (kikstra@iiasa.ac.at) and Zebedee Nicholls",
    data_structure="grid",
    dataset_category="emissions",
    external_variables="gridcell_area",
    frequency="mon",
    further_info_url="https://wcrp-cmip.org/mips/scenariomip/",
    grid="0.5x0.5 degree latitude x longitude",
    grid_label="gn",
    license="Creative Commons Attribution-ShareAlike 4.0 International License (https://creativecommons.org/licenses). Further information about this data, including some limitations, can be found via the further_info_url (recorded as a global attribute in this file). The data producers and data providers make no warranty, either express or implied, including, but not limited to, warranties of merchantability and fitness for a particular purpose. All liabilities arising from the supply of the information (including any liability arising in negligence) are excluded to the fullest extent permitted by law.",
    institution="IIASA",
    institution_id="IIASA",
    mip_era="CMIP7",
    nominal_resolution="50 km",
    realm="atmos",
    references="See: https://github.com/IAMconsortium/concordia and https://github.com/iiasa/emissions_harmonization_historical for references",
    source="Scenarios generated as part of the ScenarioMIP-CMIP7 project, see https://wcrp-cmip.org/mips/scenariomip/",
    table_id="input4MIPs",
    target_mip="CMIP7",
    product="primary-emissions-data",
    start_date="202301",
    end_date="210012",
)

# %%
ALKALINITY_ADDITION_LONGNAME = "Alkalinity Addition as part of OAE"


# %%
def convert_to_datetime(ds: xr.Dataset) -> xr.Dataset:
    ds = ds.stack(time=("year", "month"))
    dates = pd.DatetimeIndex(
        ds.indexes["time"].map(
            lambda t: datetime.date(t[0], t[1], 16 if t[1] != 2 else 15)
        )
    )
    return (
        ds.drop_vars(["time", "year", "month"])
        .assign_coords(
            time=xr.IndexVariable(
                "time",
                dates,
                encoding=dict(
                    units="days since 2023-1-1 0:0:0",
                    calendar="noleap",
                    dtype=np.dtype(float),
                ),
            )
        )
        .transpose("time", ...)
    )


# %%
def clean_coords(ds):
    return ds.squeeze(drop=True)


# %%
def add_bounds(ds, bounds=["lat", "lon", "time", "level"]):
    bounds = list(set(bounds) & set(ds.coords))
    ds = ds.cf.add_bounds(bounds, output_dim="bound")
    ds = ds.reset_coords([f"{b}_bounds" for b in bounds]).rename(
        {f"{b}_bounds": f"{b}_bnds" for b in bounds}
    )
    for b in bounds:
        ds.coords[b].attrs["bounds"] = f"{b}_bnds"
    return ds


# %%
def rename_sectors(ds, renames: dict):
    if "sector" not in ds.indexes:
        return ds

    return ds.assign_coords(
        sector=ds.indexes["sector"].map(lambda s: renames.get(s, s))
    )


# %%
def ensure_sector_ordering(ds, sector_ordering: Sequence | None):
    if sector_ordering is None:
        return ds

    return ds.reindex(sector=sector_ordering)


# %%
def add_sector_mapping(ds, keep_sector_names=True):
    if "sector" not in ds.indexes:
        return ds

    keys = ds.indexes["sector"]
    if not keep_sector_names:
        ds = ds.assign_coords(sector=pd.RangeIndex(len(keys)))

    ds["sector"].attrs.update(
        long_name="sector", ids="; ".join(f"{i}: {k}" for i, k in enumerate(keys))
    )

    return ds


# %%
def set_sector_encoding(ds):
    if "sector" not in ds.indexes:
        return ds

    # Saves strings as fixed-length character types (necessary for tools like CDO)
    ds["sector"].encoding["dtype"] = "S1"

    return ds


# %%
def update_attrs(ds, attrs):
    for k, v in attrs.items():
        if k in ds:
            ds[k].attrs.update(v)
    return ds


# %%
def clean_var(ds, name, gas, handle):
    long_name = (
        f"{gas} {handle} emissions"
        if name != "TA_em_anthro"
        else ALKALINITY_ADDITION_LONGNAME
    )
    ds[name].attrs.update({"cell_methods": "time: mean", "long_name": long_name})
    return ds


# %%
def set_var_encoding(ds, name):
    da = ds[name]
    da.encoding.update(
        {
            "zlib": True,
            "complevel": 2,
            "chunksizes": tuple((dict(da.sizes) | dict(time=1)).values()),
            "_FillValue": da.dtype.type(1e20),
        }
    )
    return ds


# %%
def ds_attrs(name, model, scenario, version, date):
    gas, rest = name.split("_", 1)
    handle = DATA_HANDLES[rest]
    title = (
        f"Future {handle} emissions of {gas} in {scenario}"
        if name != "TA_em_anthro"
        else f"{ALKALINITY_ADDITION_LONGNAME} in {scenario}"
    )

    extra_attrs = dict(
        source_version=version,
        source_id=f"{DS_ATTRS['institution']}-{model}-{scenario}".replace(" ", "-"),
        variable_id=name,
        creation_date=date,
        title=title,
        reporting_unit=f"Mass flux of {gas}",
    )
    attrs = DS_ATTRS | extra_attrs
    return attrs


# %%
class DressUp:
    def __init__(self, version) -> None:
        self.version = version
        self.date = str(datetime.datetime.today())

    def __call__(self, ds, model, scenario, **kwargs):
        vars = list(ds.data_vars)
        assert len(vars) == 1, vars

        name = vars[0]
        gas, rest = name.split("_", 1)
        handle = DATA_HANDLES[rest]

        return (
            ds.pipe(convert_to_datetime)
            .pipe(clean_coords)
            .pipe(add_bounds)
            .pipe(rename_sectors, SECTOR_RENAMES)
            .pipe(
                ensure_sector_ordering,
                SECTOR_ORDERING_GAS.get(name, SECTOR_ORDERING_DEFAULT.get(rest)),
            )
            .pipe(add_sector_mapping)
            .pipe(set_var_encoding, name)
            .pipe(set_sector_encoding)
            .pipe(update_attrs, ATTRS)
            .pipe(clean_var, name, gas, handle)
            .assign_attrs(ds_attrs(name, model, scenario, self.version, self.date))
        )


# %%
def ftp_upload(cfg: FtpSettings | dict[str, Any], local_path, remote_path):
    paths = list(local_path.iterdir())

    if not isinstance(cfg, FtpSettings):
        try:
            cfg = structure(cfg, FtpSettings)
        except Exception as exc:
            raise ValueError(", ".join(transform_error(exc, path="cfg"))) from None

    ftp = ftplib.FTP(timeout=30.0)
    ftp.connect(cfg.server, cfg.port)
    ftp.login(cfg.user, cfg.password)

    try:
        if remote_path.as_posix() not in ftp.nlst(remote_path.parent.as_posix()):
            ftp.mkd(remote_path.as_posix())

        ftp.cwd(remote_path.as_posix())
        remote_files = ftp.nlst(remote_path.as_posix())
        for lpath in paths:
            rpath = (remote_path / lpath.name).as_posix()
            lsize = lpath.stat().st_size

            msg = lpath.name
            if rpath in remote_files:
                rtimestamp = dateutil.parser.parse(ftp.voidcmd(f"MDTM {rpath}")[4:])
                ltimestamp = datetime.datetime.fromtimestamp(lpath.lstat().st_mtime)

                rsize = ftp.size(rpath)
                msg += f"\nalready on {remote_path.as_posix()}"
                if rtimestamp < ltimestamp:
                    msg += f",\nbut local file is newer ({rtimestamp} < {ltimestamp})"
                elif rsize != lsize:
                    msg += f",\nbut file size differs ({rsize} != {lsize})"
                else:
                    logger.info(msg + ",\nnot uploading")
                    continue
            else:
                msg += f"\nnot on {remote_path.as_posix()}"

            logger.info(msg + ", \nuploading")
            with (
                open(lpath, "rb") as fp,
                tqdm(total=lsize, unit="B", unit_scale=True, unit_divisor=1024) as pbar,
            ):

                def update_pbar(data):
                    pbar.update(len(data))

                try:
                    ftp.storbinary("STOR " + lpath.name, fp, callback=update_pbar)
                except TimeoutError:
                    # TODO determine whether we need to reconnect in this case
                    pass
                except OSError as exc:
                    if exc.args[0] != "cannot read from timed out object":
                        raise

    finally:
        ftp.close()


# %%
@define
class Variants:
    gas: str
    sectors: list[str]
    suffix: str
    variable_template: str

    def copy_from_default(self, data, on="variable"):
        if on == "variable":
            variables = [
                self.variable_template.format(gas=self.gas, sector=sector)
                for sector in self.sectors
            ]
            new_data = data.loc[isin(variable=variables)].rename(
                index=lambda s: s + f" ({self.suffix})", level="variable"
            )
        elif on == "sector":
            new_data = data.loc[isin(sector=self.sectors)].rename(
                index=lambda s: s + f" ({self.suffix})", level="sector"
            )
        else:
            raise ValueError(f"on needs to be either 'variable' or 'sector', not: {on}")

        return concat([data, new_data])

    def rename_suffix(self, data, from_, to, on="variable"):
        if on == "variable":
            renames = {}
            for sector in self.sectors:
                variable = self.variable_template.format(gas=self.gas, sector=sector)
                renames[variable + from_] = variable + to
            return data.rename(index=renames, level="variable")
        elif on == "sector":
            renames = {sector + from_: sector + to for sector in self.sectors}
            return concat(
                [
                    data.loc[~isin(gas=self.gas)],
                    data.loc[isin(gas=self.gas)].rename(index=renames, level=on),
                ]
            )
        else:
            raise ValueError(f"on needs to be either 'variable' or 'sector', not: {on}")

    def rename_from_subsector(self, data, on="variable"):
        return self.rename_suffix(data, f"|{self.suffix}", f" ({self.suffix})", on=on)

    def rename_to_subsector(self, data, on="sector"):
        return self.rename_suffix(data, f" ({self.suffix})", f"|{self.suffix}", on=on)


# %% [markdown]
# ## Define some useful functions

# %%
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

# %%
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


# %%
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

# %%
def sort_iamc_dataframe(df, format="long"):

    if (format == "long"):
        return sort_long_iamc_dataframe(df)
    else:
        raise Exception("Formats other than 'long' not yet implemented.") 


# %%
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

# %%
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

# %%
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

# %%
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

# %%
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

# %%
def filter_emissions_data(df):
    """
    Filters dataframe for variables starting with "Emissions".

    Parameters:
        df (pd.DataFrame): The dataframe to filter.

    Returns:
        pd.DataFrame: The filtered dataframe.
    """
    return df[df['variable'].str.startswith("Emissions")]

# %%
# remove data with year > 2100; assumes a dataframe in long format 
def remove_data_after(df, yr = 2100):
    return df[df['year'] <= yr]

# %%
# Rename one variable explicitly
def rename_one_variable(df, old_string, new_string):
    df['variable'] = df['variable'].replace({old_string: new_string})
    return df

# %%
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

# %%
# Custom function to select columns and drop duplicates
def select_and_distinct(dataframe, columns):
    return dataframe.loc[:, columns].drop_duplicates()

# %%
# Filter regions for CMIP7 data
def filter_regions_only_world_and_model_native(df, cmip7_iam_list=None):
    if cmip7_iam_list is None:
        cmip7_iam_list = ["MESSAGE", "AIM", "COFFEE", "GCAM", "IMAGE", "REMIND", "WITCH"]

    world_df = filter_region(df, regions="World")
    model_native_df = filter_region_contains(df, substrings=cmip7_iam_list)

    return pd.concat([world_df, model_native_df], axis=0)

# %%
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

# %%
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

# %%
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

# %%
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

# %%
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

# %%
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

# %%
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


# %%
def join_gdp_based_on_ssp(scenarios_with_ssp_mapping, gdp_per_ssp):
    gdp_for_each_scenario = semijoin(
            gdp_per_ssp,
            # SSP_per_pathway.index.pix.assign(ssp=SSP_per_pathway + "_v9_130325"), # CMIP6 era SSP data
            scenarios_with_ssp_mapping.index.pix.assign(ssp=scenarios_with_ssp_mapping), # CMIP7 era SSP data
            how="right",
        ).pix.project(["model", "scenario", "country"])
    return gdp_for_each_scenario


# %%
def read_r_variable(file, float_dtype: str = "float32"):
    file = Path(file)
    print(f"Reading in {file}\n")

    result = pyreadr.read_r(file)

    # If there's more than one, you can decide how to handle it
    if len(result) > 1:
        print(f"Warning: More than one variable found in {file.name}, using the first one.")

    # Get the first variable's name and value
    old_var_name, value = next(iter(result.items()))

    # Rename the variable to the filename
    if old_var_name != file.stem:
        print(f"Renaming variable '{old_var_name}' to '{file.stem}'")
        value.name = file.stem


    return np.asarray(value, dtype=float_dtype)[::-1]

# %%
def read_r_to_da(file, template, flipud=True, dtype="float32"):
    """
    Read an R .rds/.RData variable and return an xarray.DataArray
    on the same grid as `template` (expects template to have lat/lon coords).
    """
    res = pyreadr.read_r(str(file))
    _, value = next(iter(res.items()))           # take first object
    arr = np.asarray(value, dtype=dtype)
    if flipud:                                   # typical R->Python row order fix
        arr = arr[::-1, :]

    da = xr.DataArray(
        arr,
        coords={"lat": template["lat"].values, "lon": template["lon"].values},
        dims=("lat", "lon"),
        name=Path(file).stem,
    )
    return da

# %%
def save_da_as_rd(
    da: xr.DataArray,
    out_path: str | Path,
    *,
    object_name: str | None = None,
    undo_flip: bool = True,
    float_dtype: str = "float64"
):
    """
    Save a 2D xarray.DataArray (lat x lon) as an RData file containing a named R matrix.
    Reading with pyreadr.read_r(...) returns an OrderedDict with that name as the key
    and a pandas DataFrame of shape (lat, lon) as the value.

    Parameters
    ----------
    da : xr.DataArray        # 2D (lat x lon), numeric
    out_path : str | Path    # e.g. 'CO_2022_WST.Rd'
    object_name : str        # name of the object inside the RData (defaults to file stem)
    undo_flip : bool         # if you flipped with [::-1] at read, flip back here
    """
    out_path = Path(out_path)
    if object_name is None:
        object_name = out_path.stem

    if da.ndim != 2:
        raise ValueError(f"Expected 2D DataArray (lat x lon), got shape {da.shape}")
    if not np.issubdtype(da.dtype, np.number):
        raise TypeError(f"DataArray dtype must be numeric; got {da.dtype}")

    arr = np.asarray(da.data, dtype=float_dtype) # could also do np.float32 or np.float64
    if undo_flip:
        arr = arr[::-1].copy()

    nrow, ncol = arr.shape

    with localconverter(default_converter + numpy2ri.converter):
        r_vec = ro.FloatVector(arr.ravel(order="C"))
        r_mat = r["matrix"](r_vec, nrow=nrow, ncol=ncol, byrow=True)

        ro.globalenv[object_name] = r_mat
        r(f"save({object_name}, file={repr(str(out_path))})")


# %% [markdown]
# # -------- Example usage --------
# Assuming 'da' is your DataArray (lat: 360, lon: 720)
# and you want the re-read key to be 'CO_2022_WST'
# save_da_as_rd(da, "CO_2022_WST.Rd", object_name="CO_2022_WST", undo_flip=True)
# # da is your DataArray, e.g.:
# # <xarray.DataArray 'VOC-WST' (lat: 360, lon: 720)> float32 ...
# # Save as single-object .Rd (works like .Rds)
# save_da_as_rd(da, "VOC_2022_WST.Rd", container="RDS", undo_flip=True)
# Notes & tips
# RDS vs RData: If your originals are single-object files, use container="RDS" (simpler). If they are workspaces with a named symbol, use container="RDATA" and set object_name (often the stem, e.g., SO2_2021_WST).
# Validation round-trip (optional): you can re-read the saved file with pyreadr.read_r(...) and compare to your original NumPy array (after applying the same flip logic) to confirm equality.
