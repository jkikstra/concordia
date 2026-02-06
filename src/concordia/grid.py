from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import xarray as xr
from attrs import define

from aneris.grid import Gridded, GriddingContext, Proxy  # noqa: F401

from .utils import VariableDefinitions


logger = logging.getLogger(__name__)


def sector_map(variables):
    return xr.DataArray(
        variables["proxy_sector"]
        .groupby(variables.index.pix.project("sector").str.split("|").str[0])
        .first()  # We assume that sector-splits fall back to the same sector
    )


@define(slots=False)  # cached_property's need __dict__
class ConcordiaProxy(Proxy):
    """
    A subclass of the `Proxy` class, representing a proxy for gridding operations
    in the Concordia workflow. This class provides methods for creating proxies
    from variable definitions and associated metadata.

    Attributes:
        None (inherits attributes from the `Proxy` base class).

    Methods:
        from_variables(cls, df, context, proxy_dir=None):
            Creates a `ConcordiaProxy` instance from variable definitions and
            gridding context.

    Usage:
        This class is used to handle gridding proxies for emissions data in the
        Concordia workflow. It maps variables to their corresponding gridding
        levels, sectors, and proxies.
    """

    @classmethod
    def from_variables(
        cls, df, context: GriddingContext, proxy_dir: Path | None = None
    ):
        """
        Creates a `ConcordiaProxy` instance from variable definitions.

        Parameters:
            df (pd.DataFrame or VariableDefinitions):
                The variable definitions or a DataFrame containing variable metadata.
                Must include columns such as `output_variable`, `proxy_path`,
                `griddinglevel`, and `proxy_sector`.

            context (GriddingContext):
                The gridding context used for creating the proxy.

            proxy_dir (Path, optional):
                The directory containing proxy files. Defaults to the current working
                directory if not provided.

        Returns:
            ConcordiaProxy:
                An instance of the `ConcordiaProxy` class.

        Notes:
            - If `df` is an instance of `VariableDefinitions`, its `data` attribute
              is used.
            - The `output_variable` column is used to determine the name of the proxy.
            - The `proxy_path` column specifies the paths to proxy files.
            - The `proxy_sector` column is used to map sectors to their proxies.
            - If the `gas` column contains only "TA", it is mapped to "CO2".

        Example:
            ```python
            from concordia.grid import ConcordiaProxy, GriddingContext
            from pathlib import Path

            # Example variable definitions
            variable_defs = VariableDefinitions.from_csv("variable_definitions.csv")
            context = GriddingContext()

            # Create a ConcordiaProxy instance
            proxy = ConcordiaProxy.from_variables(variable_defs, context, proxy_dir=Path("/path/to/proxies"))
            ```
        """
        if isinstance(df, VariableDefinitions):
            df = df.data
        if proxy_dir is None:
            proxy_dir = Path.getcwd()
        name = df["output_variable"].unique().item()
        paths = [proxy_dir / proxy_path for proxy_path in df["proxy_path"].unique()]
        index_mappings = {"sector": sector_map(df)}
        if df.pix.unique("gas").tolist() == ["TA"]:
            index_mappings["gas"] = pd.Series(["CO2"], ["TA"])

        return cls.from_files(
            name, paths, frozenset(df["griddinglevel"]), context, index_mappings
        )
