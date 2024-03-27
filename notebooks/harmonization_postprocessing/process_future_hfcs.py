# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
from pathlib import Path

import pandas as pd
import yaml


# %%
with open("../config.yaml") as f:
    config = yaml.safe_load(f)
base_path = Path(config["base_path"])

# # Show data layout of 2022 Velders
path_2022 = (
    base_path / "harmonization_postprocessing/rescue/HFC_Kigali2022_Scenario.xlsx"
)
df = pd.read_excel(path_2022, sheet_name="Upper")

df  # Emis_tot	Total emissions (ton/uyr)

label = "Emis_tot"
label_idx = 4
col = f"Unnamed: {label_idx}"

df[df[col] == label].index


def excel_to_df(df):
    cidx = [0, 1, label_idx]
    columns = df.iloc[3, cidx].values
    idxs = df[df[col] == label].index
    dfs = []
    for i in idxs:
        _df = df.iloc[i + 1 : i + 112, 0:3]
        _df.columns = columns
        dfs.append(_df)
    edf = pd.concat(dfs).set_index(["Species", "Year"])[label].unstack("Year")
    return edf


excel_to_df(df).head()

# # Do Data Processing

udf = excel_to_df(pd.read_excel(path_2022, sheet_name="Upper"))
ldf = excel_to_df(pd.read_excel(path_2022, sheet_name="Lower"))
df = (udf + ldf) / 2  # take the average

df

gwps = pd.Series(
    [14800, 675, 3500, 1430, 4470, 124, 3220, 9810, 1030, 794, 1640],
    index=[
        "HFC-23",
        "HFC-32",
        "HFC-125",
        "HFC-134a",
        "HFC-143a",
        "HFC-152a",
        "HFC-227ea",
        "HFC-236fa",
        "HFC-245fa",
        "HFC-365mfc",
        "HFC-43-10mee",
    ],
)
gwps

data = (df.T * gwps).dropna(axis="columns").T
data /= data.sum()
data

data.T.plot()

write_data = {}
write_data["velders_2022"] = data.copy()

# # Show data layout of old data
path_2015 = (
    base_path
    / "harmonization_postprocessing/cmip6/HFC_Total_Disaggregation_processed_decades_only.xlsx"
)
df = pd.read_excel(
    path_2015,
    sheet_name="mt-HFC-to-kt-species",
    header=1,
    usecols=range(1, 13),
    index_col=0,
)
df.head()

# ## Drop old HFC
#
# HFC-23 is not in the new data and is near-0 by 2020, so drop it

df = df.drop("HFC-23", axis=1)
df.head()

len(data), len(df.T)  # should be the same

newdata = df.T / df.T.sum()  # normalize

newdata.T.plot()

write_data["velders_2015"] = newdata.copy()

# # Write it out
#
# **NOTE** we take the 2015 scenario for our work to be as close to cmip6 as possible

write_data["rescue_hfc_scenario"] = write_data["velders_2015"]

with pd.ExcelWriter("rescue_hfc_scenario.xlsx") as writer:
    for sheet_name, _df in write_data.items():
        _df.to_excel(writer, sheet_name=sheet_name)
