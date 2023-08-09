from dataclasses import dataclass
from itertools import chain, repeat
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import pandas_indexing.accessors
from pandas import DataFrame
from pandas_indexing import concat, isin
from IPython.core.magic import Magics, magics_class, cell_magic

@magics_class
class CondordiaMagics(Magics):

    @cell_magic
    def execute_or_lazy_load(self, line, cell):
        """
        Evaluates first line argument. If True, executes cell, otherwise executes remaining arguments
        """
        parts = line.split('#')[0].split()
        if self.shell.ev(parts[0]):
            self.shell.run_cell(cell)
        elif len(parts) > 1:
            print(f'Running: {cmd}')
            cmd = ' '.join(parts[1:])
            self.shell.run_cell(cmd)
        else:
            return print('Skipped')

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
    def history(self):
        return self.__class__(self.data.loc[self.data.has_history])

    @property
    def variable_index(self):
        return self.data.index

    @property
    def index_global(self):
        return self.data.index[self.data["global"]].pix.project(["gas", "sector"])

    @property
    def index_regional(self):
        return self.data.index[~self.data["global"]].pix.project(["gas", "sector"])

    def load_data(
        self,
        df: DataFrame,
        levels: Optional[list[str]] = None,
        ignore_undefined: bool = True,
        ignore_missing: bool = False,
        extend_missing: Union[bool, float] = False,
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
        extend_missing : bool or float, default False
            whether to extend_missing with a certain value
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

        if (li == -1).any() and extend_missing is False:
            raise ValueError(
                "Variables missing from data:" + unique_variable_str(index[li == -1])
            )
        if (ri == -1).any():
            raise ValueError(
                "Undefined variables exist in data:"
                + unique_variable_str(index[ri == -1])
            )

        if (li == -1).any():
            # Fix nan-values in levels
            index = index.pix.assign(
                unit=np.where(
                    li != -1, index.pix.project("unit"), self.data["unit"].values[ri]
                ),
            )

        fill_value = 0 if extend_missing is True else extend_missing
        df = pd.DataFrame(
            np.where((li != -1)[:, np.newaxis], df.values[li], fill_value),
            index=index,
            columns=df.columns,
        ).__finalize__(df)
        if timeseries:
            data_units = self.data["unit"].values[ri]
            non_matching_units = df.index.pix.project("unit") != data_units
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

        if (li == -1).any():
            # Need to expand nan values
            nanlevels = df.index.names.difference(self.variable_index.names).difference(["unit"])

            variations = pd.MultiIndex.from_product(
                chain(
                    (index.pix.project("variable")[li == -1],),
                    (df.pix.unique(level).dropna() for level in nanlevels),
                )
            )
            df = concat(
                [
                    df.loc[li != -1],
                    df.loc[li == -1].droplevel(nanlevels).pix.semijoin(variations),
                ]
            )

        if levels is not None:
            return df.pix.project(levels)
        return df


@dataclass
class RegionMapping:
    data: pd.Series

    @classmethod
    def concat(cls, rms):
        return cls(pd.concat(rm.data for rm in rms))

    @classmethod
    def from_regiondef(
        cls,
        path,
        country_column="ISO Code",
        region_column="Native Region Code",
        **kwargs,
    ):
        path = Path(path)
        match path.suffix:
            case ".csv":
                df = pd.read_csv(path, **kwargs)
            case ".xlsx":
                df = pd.read_excel(path, **kwargs)
            case suffix:
                raise ValueError(f"Unknown file suffix: {suffix}")

        return cls(
            df.set_index(country_column)[region_column]
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
            df.pix.semijoin(self.index, how="left")
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
        .pix.semijoin(index, how="right")
        .groupby(df.index.names)
        .agg(agg_func)
    )
    return concat(
        [df.loc[~isin(**{level: index.pix.project("old")})], new]
    ).sort_index()


def as_seaborn(
    df: DataFrame, meta: Optional[DataFrame] = None, value_name: str = "value"
):
    """Convert multi-indexed time-series dataframe to tidy dataframe

    Parameters
    ----------
    df : DataFrame
        data in time-series representation with years on columns
    meta : DataFrame, optional
        meta data that is joined before tidying up
    value_name : str
        column name for the values; default "value"

    Returns
    -------
    DataFrame
        Tidy dataframe without index
    """
    df = df.rename_axis(columns="year").stack().rename(value_name)
    if meta is not None:
        df = df.to_frame().join(meta, on=meta.index.names)
    return df.reset_index()
