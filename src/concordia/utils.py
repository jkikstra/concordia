from dataclasses import dataclass
from itertools import chain, repeat
from pathlib import Path
from typing import Optional

import pandas as pd
import pandas_indexing.accessors
from pandas import DataFrame
from pandas_indexing import concat, isin


@dataclass
class VariableDefinitions:
    data: DataFrame

    @classmethod
    def from_csv(cls, path):
        return cls(
            pd.read_csv(path, index_col=list(range(3))).loc[
                isin(variable=lambda s: ~s.str.startswith("#"))
            ]
        )

    @property
    def variable_index(self):
        return self.data.index

    @property
    def index_global(self):
        return self.data.index[self.data["global"]].idx.project(["gas", "sector"])

    @property
    def index_regional(self):
        return self.data.index[~self.data["global"]].idx.project(["gas", "sector"])

    def load_data(
        self,
        df: DataFrame,
        levels: Optional[list[str]] = None,
        ignore_undefined: bool = True,
        ignore_missing: bool = False,
        timeseries: bool = True,
    ):
        """Load data from dataframe

        Assigns sector/gas and checks correct units.

        Parameters
        ----------
        df : DataFrame
            data
        levels : list of str, optional
            levels to keep, or all if None
        ignore_undefined : bool, default True
            whether to fail if undefined variables exist in `df`
        ignore_missing : bool, default False
            whether to ignore defined variables missing from `df`
        timeseries : bool, default True
            whether data is a timeseries and columns should be cast to int

        Returns
        -------
        DataFrame
            data with sector/gas index levels

        Note
        ----
        Does not check regional availability yet! Also because it would have to
        understand about regionmappings and aggregations then.
        """

        df = df.rename_axis(index=str.lower)
        if timeseries:
            df = df.rename_axis(columns="year").rename(columns=int)
        else:
            df = df.rename(columns=str.lower)

        if ignore_undefined and ignore_missing:
            how = "inner"
        elif ignore_undefined:
            how = "right"
        else:
            how = "outer"
        index, li, ri = df.index.join(
            self.variable_index, how=how, return_indexers=True
        )

        def unique_variable_str(index):
            return "\n  " + ",\n  ".join(index.unique("variable"))

        if (li == -1).any():
            raise ValueError(
                "Variables missing from data:" + unique_variable_str(index[li == -1])
            )
        if (ri == -1).any():
            raise ValueError(
                "Undefined variables exist in data:"
                + unique_variable_str(index[ri == -1])
            )

        df = pd.DataFrame(df.values[li], index=index, columns=df.columns).__finalize__(
            df
        )
        if timeseries:
            data_units = self.data["unit"].values[ri]
            non_matching_units = df.index.idx.project("unit") != data_units
            if non_matching_units.any():
                errors = (
                    df.index.to_frame(index=False)
                    .loc[
                        non_matching_units,
                        lambda df: df.columns.intersection(
                            ["model", "scenario", "variable", "sector", "gas", "unit"]
                        ),
                    ]
                    .assign(**{"expected unit": data_units[non_matching_units]})
                    .drop_duplicates()
                )
                raise ValueError(
                    "Some variables in the data do not have the correct units:\n"
                    + errors.to_string(index=False)
                )

        if levels is not None:
            return df.idx.project(levels)
        return df


@dataclass
class RegionMapping:
    data: pd.Series

    @classmethod
    def concat(cls, rms):
        return cls(pd.concat(rm.data for rm in rms))

    @classmethod
    def from_regiondef(cls, path):
        path = Path(path)
        match path.suffix:
            case ".csv":
                df = pd.read_csv(path)
            case ".xlsx":
                df = pd.read_csv(path)
            case suffix:
                raise ValueError(f"Unknown file suffix: {suffix}")

        return cls(
            df.set_index("ISO Code")["Native Region Code"]
            .rename_axis("country")
            .rename(index=str.lower)
            .rename("region")
        )

    def prefix(self, s: str):
        return self.__class__(s + self.data)

    @property
    def index(self) -> pd.MultiIndex:
        return pd.MultiIndex.from_arrays(
            [self.data.index, self.data.values], names=["country", "region"]
        )

    def aggregate(self, df: DataFrame, level="country", agg_func="sum") -> DataFrame:
        if level != "country":
            df = df.rename_axis(index={level: "country"})
        return (
            df.idx.semijoin(self.index, how="left")
            .groupby(
                [n if n != "country" else "region" for n in df.index.names],
                dropna=False,
            )
            .agg(agg_func)
        )


def combine_countries(df, level="country", agg_func="sum", **countries):
    index = pd.MultiIndex.from_tuples(
        chain(
            *(
                zip(repeat(new_name), individual_countries)
                for new_name, individual_countries in countries.items()
            )
        ),
        names=[level, "old"],
    )

    new = (
        df.rename_axis(index={level: "old"})
        .idx.semijoin(index, how="right")
        .groupby(df.index.names)
        .agg(agg_func)
    )
    return concat(
        [df.loc[~isin(**{level: index.idx.project("old")})], new]
    ).sort_index()
