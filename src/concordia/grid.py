import logging
from collections import namedtuple
from functools import cached_property
from pathlib import Path
from typing import Optional

import dask
import pandas as pd
import ptolemy as pt
import xarray as xr
from attrs import define, field
from pandas_indexing import isin

from .utils import Pathy, VariableDefinitions


logger = logging.getLogger(__name__)


@dask.delayed
def verify_global_values(
    aggregated, tabular, proxy_name, index, abstol=1e-8, reltol=1e-4
):
    tab_df = tabular.groupby(level=index).sum().unstack("year")
    grid_df = aggregated.to_series().groupby(level=index).sum().unstack("year")
    grid_df, tab_df = grid_df.align(tab_df, join="inner")

    absdiff = abs(grid_df - tab_df)
    if (absdiff >= abstol + reltol * tab_df).any(axis=None):
        reldiff = (absdiff / tab_df).where(abs(tab_df) > 0, 0)
        logger.warning(
            f"Yearly global totals relative values between grids and global data for ({proxy_name}) not within {reltol}:\n"
            f"{reldiff}"
        )
        return reldiff
    else:
        logger.info(
            f"Yearly global totals relative values between grids and global data for ({proxy_name}) within tolerance"
        )
        return


def sector_map(variables):
    return dict(
        zip(
            variables["proxy_sector"],
            variables.index.pix.project("sector")
            .str.split("|")
            .str[0],  # We assume that sector-splits fall back to the same sector
        )
    )


Weight = namedtuple("Weight", ["global_", "regional"])


@define
class Gridded:
    gridded: xr.DataArray
    downscaled: pd.DataFrame
    proxy: "Proxy"
    meta: dict[str, str] = field(factory=dict)

    def verify(self, compute: bool = True):
        return self.proxy.verify_gridded(self.gridded, self.downscaled, compute=compute)

    def prepare_dataset(self, callback: Optional[callable] = None):
        name = self.proxy.name
        ds = self.gridded.to_dataset(name=name)

        if callback is not None:
            ds = callback(ds, name=name, **self.meta)

        return ds

    def to_netcdf(
        self,
        template_fn: str,
        callback: Optional[callable] = None,
        encoding_kwargs: Optional[dict] = None,
        directory: Optional[Pathy] = None,
        compute: bool = True,
    ):
        ds = self.prepare_dataset(callback)

        name = self.proxy.name
        meta = self.meta | dict(name=name)
        fn = template_fn.format(
            **{k: v.replace("_", "-").replace(" ", "-") for k, v in meta.items()}
        )
        if directory is not None:
            fn = Path(directory) / fn

        encoding_kwargs = {"zlib": True, "complevel": 2} | (encoding_kwargs or {})
        return ds.to_netcdf(
            fn,
            encoding={name: encoding_kwargs},
            compute=compute,
        )


class Proxy:
    def __init__(
        self,
        proxy,
        only_global,
        indexraster: pt.IndexRaster,
        name="unnamed",
        as_flux: bool | xr.DataArray = True,
    ):
        self.proxy = proxy
        self.only_global = only_global
        self.indexraster = indexraster
        self.name = name
        self.as_flux = as_flux

    @classmethod
    def from_variables(cls, df, indexraster=None, proxy_dir=None, **kwargs):
        if isinstance(df, VariableDefinitions):
            df = df.data
        if proxy_dir is None:
            proxy_dir = Path.getcwd()
        name = df["proxy_name"].unique().item()
        proxy = xr.concat(
            [
                xr.open_dataset(proxy_dir / proxy_path, chunks="auto").chunk(
                    {"lat": -1, "lon": -1}
                )["emissions"]
                for proxy_path in df["proxy_path"].unique()
            ],
            dim="sector",
        )
        sectors = proxy.indexes["sector"].map(sector_map(df))
        if sectors.isna().any():
            unused_sectors = proxy.indexes["sector"][sectors.isna()]
            logger.warn(
                "Proxy %s has unused sectors: %s",
                name,
                ", ".join(unused_sectors),
            )
            proxy = proxy.sel(sector=~sectors.isna())
            sectors = sectors.dropna()
        proxy["sector"] = sectors

        only_global = df["global"].all()
        if indexraster is None and not only_global:
            raise ValueError("indexraster needs to be given for regional variables")

        return cls(proxy, only_global, indexraster, name=name, **kwargs)

    @cached_property
    def proxy_with_flux(self):
        da = self.proxy
        if self.as_flux:
            da /= self.indexraster.cell_area
        return da

    def reduce_dimensions(self, da):
        da = da.mean("month")
        if "level" in da.dims:
            da = da.sum("level")
        return da

    @cached_property
    def weight(self):
        proxy_reduced = self.reduce_dimensions(self.proxy)

        global_weight = proxy_reduced.sum(["lat", "lon"]).assign_coords(country="World")
        if self.only_global:
            return Weight(global_weight.persist(), None)

        return Weight(
            *dask.persist(global_weight, self.indexraster.aggregate(proxy_reduced))
        )

    @property
    def countries(self):
        if self.only_global:
            return pd.Index(["World"], name="country")

        proxy_weight = self.weight.regional
        country_weight = proxy_weight.sum(
            [d for d in proxy_weight.dims if d != "country"]
        ).to_pandas()
        return country_weight.index[country_weight > 0]

    @staticmethod
    def assert_single_pathway(downscaled):
        pathways = downscaled.pix.unique(["model", "scenario"])
        assert (
            len(pathways) == 1
        ), "`downscaled` is needed as a single scenario, but there are: {pathways}"
        return dict(zip(pathways.names, pathways[0]))

    def prepare_downscaled(self, downscaled):
        meta = self.assert_single_pathway(downscaled)
        downscaled = (
            downscaled.stack("year")
            .pix.semijoin(
                pd.MultiIndex.from_product(
                    [self.proxy.indexes[d] for d in ["gas", "sector", "year"]]
                ),
                how="inner",
            )
            .pix.project(["gas", "sector", "country", "year"])
            .sort_index()
        )
        downscaled.attrs.update(meta)
        return downscaled

    def verify_gridded(self, gridded, downscaled, compute: bool = True):
        scen = self.prepare_downscaled(downscaled)

        global_gridded = self.reduce_dimensions(gridded)
        if self.as_flux:
            global_gridded *= self.indexraster.cell_area
        global_gridded = global_gridded.sum(["lat", "lon"])
        diff = verify_global_values(
            global_gridded, scen, self.name, ("sector", "gas", "year")
        )
        return diff.compute() if compute else diff

    def grid(self, downscaled: pd.DataFrame) -> Gridded:
        scen = self.prepare_downscaled(downscaled)

        def weighted(scen, weight):
            scen = xr.DataArray.from_series(scen)
            weight = weight.reindex_like(scen)
            return (scen / weight).where(weight, 0)

        is_global = isin(country="World")

        gridded_global = weighted(
            scen.loc[is_global].droplevel("country"), self.weight.global_
        ).drop_vars("country")
        if self.only_global:
            return Gridded(
                gridded_global * self.proxy_with_flux,
                downscaled.loc[is_global],
                self,
                scen.attrs,
            )

        gridded_regional = self.indexraster.grid(
            weighted(scen.loc[~is_global], self.weight.regional)
        ).drop_vars("country")

        gridded = (
            xr.concat([gridded_regional, gridded_global], dim="sector")
            if gridded_global.size > 0
            else gridded_regional
        )

        return Gridded(gridded * self.proxy_with_flux, downscaled, self, scen.attrs)
