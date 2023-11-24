import logging
import re
from functools import cached_property
from typing import Optional, Sequence

import pandas as pd
import ptolemy as pt
from attrs import define, field
from dask import compute
from pandas import isna
from pandas_indexing import concat, isin

from .downscale import downscale
from .grid import Gridded, Proxy
from .harmonize import Harmonized, harmonize
from .settings import Settings
from .utils import (
    Pathy,
    RegionMapping,
    VariableDefinitions,
    aggregate_subsectors,
    skipnone,
)


logger = logging.getLogger(__name__)


@define(slots=False)
class WorkflowDriver:
    model: pd.DataFrame
    hist: pd.DataFrame
    gdp: pd.DataFrame
    regionmapping: RegionMapping
    indexraster: pt.IndexRaster

    variabledefs: VariableDefinitions
    harm_overrides: pd.DataFrame
    settings: Settings

    history_aggregated: dict[str, pd.DataFrame] = field(factory=dict)
    harmonized: dict[str, pd.DataFrame] = field(factory=dict)
    downscaled: dict[str, pd.DataFrame] = field(factory=dict)

    @cached_property
    def harmdown_all_global(self) -> pd.DataFrame:
        variables = self.variabledefs.index_global

        model = self.model.pix.semijoin(variables, how="right").loc[
            isin(region="World")
        ]
        hist = (
            self.hist.pix.semijoin(variables, how="right")
            .loc[isin(country="World")]
            .rename_axis(index={"country": "region"})
        )

        harmonized = harmonize(
            model,
            hist,
            overrides=self.harm_overrides.pix.semijoin(variables, how="inner"),
            settings=self.settings,
        )

        harmonized = aggregate_subsectors(harmonized)
        hist = aggregate_subsectors(hist)

        self.history_aggregated["global"] = hist
        self.harmonized["global"] = harmonized
        self.downscaled["global"] = harmonized.pix.format(
            method="single", country="{region}"
        )

        return harmonized.droplevel("method").rename_axis(index={"region": "country"})

    def harmdown_global(self, variables: pd.MultiIndex):
        return self.harmdown_all_global.pix.semijoin(variables, how="right")

    def harmdown_regional(
        self,
        variables: pd.MultiIndex,
        regionmapping: RegionMapping,
        name: Optional[str] = None,
    ) -> pd.DataFrame:
        model = self.model.pix.semijoin(variables, how="right").loc[
            ~isin(region="World")
        ]
        hist = self.hist.pix.semijoin(variables, how="right")
        hist_agg = regionmapping.aggregate(hist, dropna=True)
        if name is not None:
            self.history_aggregated[name] = hist_agg

        harmonized = harmonize(
            model,
            hist_agg,
            overrides=self.harm_overrides.pix.semijoin(variables, how="inner"),
            settings=self.settings,
        )
        if name is not None:
            self.harmonized[name] = harmonized

        harmonized = aggregate_subsectors(harmonized.droplevel("method"))
        hist = aggregate_subsectors(hist)

        downscaled = downscale(
            harmonized,
            hist,
            self.gdp,
            regionmapping,
            settings=self.settings,
        )
        if name is not None:
            self.downscaled[name] = downscaled

        return downscaled.droplevel(["method", "region"])

    def harmdown_proxy(
        self,
        variables: pd.DataFrame,
        proxy: Optional[Proxy],
        include_global: bool = True,
    ) -> pd.DataFrame:
        regionmapping = (
            self.regionmapping
            if proxy is None
            else self.regionmapping.filter(proxy.countries)
        )

        downscaled = []
        if include_global:
            var_global = variables.index_global
            if not var_global.empty:
                downscaled.append(self.harmdown_global(var_global))
        var_regional = variables.index_regional
        if not var_regional.empty:
            downscaled.append(
                self.harmdown_regional(
                    var_regional, regionmapping, name=variables.proxies.item()
                )
            )

        return concat(downscaled) if downscaled else None

    def harmonize_and_downscale(
        self, proxy_name: Optional[str] = None, include_global: bool = True
    ):
        if proxy_name is None:
            return concat(
                skipnone(
                    *(
                        [
                            self.harmonize_and_downscale(
                                proxy_name, include_global=False
                            )
                            for proxy_name in self.variabledefs.proxies
                        ]
                        + ([self.harmdown_all_global] if include_global else [])
                    )
                )
            )

        logger.info(f"Harmonizing and downscaling variables for proxy {proxy_name}")
        variables = self.variabledefs.for_proxy(proxy_name)
        proxy = (
            Proxy.from_variables(
                variables, self.indexraster, proxy_dir=self.settings.proxy_path
            )
            if not isna(proxy_name)
            else None
        )

        return self.harmdown_proxy(variables, proxy, include_global=include_global)

    def grid_proxy(self, proxy_name: str):
        variables = self.variabledefs.for_proxy(proxy_name)
        proxy = Proxy.from_variables(
            variables, self.indexraster, proxy_dir=self.settings.proxy_path
        )

        downscaled = self.harmdown_proxy(variables, proxy)

        # Convert unit to kg/s of the repective gas
        downscaled = downscaled.pix.convert_unit(
            lambda s: re.sub("(?:Gt|Mt|kt|t|kg) (.*)/yr", r"kg \1/s", s)
        )

        for model, scenario in downscaled.pix.unique(["model", "scenario"]):
            yield proxy.grid(downscaled.loc[isin(model=model, scenario=scenario)])

    def grid_all(
        self,
        template_fn: str,
        directory: Optional[Pathy] = None,
        callback: Optional[callable] = None,
        encoding_kwargs: Optional[dict] = None,
        verify: bool = True,
    ):
        def verify_and_save(pathways: Sequence[Gridded]):
            return compute(
                (
                    gridded.to_netcdf(
                        template_fn,
                        callback,
                        directory=directory,
                        encoding_kwargs=encoding_kwargs,
                        compute=False,
                    ),
                    gridded.verify() if verify else None,
                )
                for gridded in pathways
            )

        res = {}
        for proxy_name in self.variabledefs.proxies:
            if isna(proxy_name):
                self.harmonize_and_downscale(proxy_name)
            else:
                res[proxy_name] = verify_and_save(self.grid_proxy(proxy_name))
        return res

    @property
    def harmonized_data(self):
        hist = concat(self.history_aggregated.values())
        model = self.model.pix.semijoin(hist.index, how="right")
        harmonized = concat(self.harmonized.values())

        return Harmonized(hist=hist, model=model, harmonized=harmonized)
