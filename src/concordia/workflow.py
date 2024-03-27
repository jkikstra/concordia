import logging
import re
import textwrap
from collections import namedtuple
from functools import cached_property
from typing import Iterator, Optional, Sequence

import dask
import pandas as pd
import ptolemy as pt
from attrs import define
from pandas_indexing import concat, isin
from pandas_indexing.utils import print_list
from tqdm.auto import tqdm

from .downscale import downscale
from .grid import Gridded, Proxy
from .harmonize import Harmonized, harmonize
from .settings import Settings
from .utils import (
    Pathy,
    RegionMapping,
    VariableDefinitions,
    add_regions_as_zero,
    aggregate_subsectors,
    skipnone,
)


logger = logging.getLogger(__name__)

CountryGroup = namedtuple("CountryGroup", ["countries", "variables"])


def log_uncovered_history(
    hist: pd.DataFrame, hist_agg: pd.DataFrame, threshold=0.01, base_year: int = 2020
) -> None:
    levels = ["gas", "sector", "unit"]
    hist_total = hist.loc[~isin(country="World"), base_year].groupby(levels).sum()
    hist_covered = hist_agg.loc[:, base_year].groupby(levels).sum()
    hist_uncovered = hist_total - hist_covered
    hist_stats = pd.DataFrame(
        dict(uncovered=hist_uncovered, rel=hist_uncovered / hist_total)
    )
    loglevel = (
        logging.WARN
        if (hist_uncovered > threshold * hist_total + 1e-6).any()
        else logging.INFO
    )
    logger.log(
        loglevel,
        "Historical emissions in countries missing from proxy:"
        + "".join(
            "\n"
            + "::".join(t.Index[:2])
            + f" - {t.uncovered:.02f} {t.Index[2]} ({t.rel * 100:.01f}%)"
            for t in hist_stats.sort_values("rel", ascending=False).itertuples()
        ),
    )


@define
class GlobalRegional:
    global_: Optional[pd.DataFrame] = None
    regional: Optional[pd.DataFrame] = None

    @property
    def data(self):
        return concat([self.global_, self.regional])


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

    history_aggregated: GlobalRegional = GlobalRegional()
    harmonized: GlobalRegional = GlobalRegional()
    downscaled: GlobalRegional = GlobalRegional()

    @cached_property
    def proxies(self):
        return {
            proxy_name: Proxy.from_variables(
                self.variabledefs.for_proxy(proxy_name),
                self.indexraster,
                self.settings.proxy_path,
            )
            for proxy_name in self.variabledefs.proxies
        }

    def country_groups(
        self, variabledefs: Optional[VariableDefinitions] = None
    ) -> Iterator[CountryGroup]:
        if variabledefs is None:
            variabledefs = self.variabledefs

        regional_proxies = set(
            variabledefs.data.loc[~variabledefs.data["global"], "proxy_name"]
        )
        variables = [
            w.to_series().loc[lambda s: abs(s) > 0].index
            for w in dask.compute(
                *[
                    proxy.weight.regional.sum("year")
                    for proxy_name, proxy in self.proxies.items()
                    if proxy_name in regional_proxies
                    and proxy.weight.regional is not None
                ]
            )
        ]

        variables = concat(variables) if variables else pd.DataFrame()
        if not variables.empty:
            sectors = variabledefs.index_regional.pix.unique("sector")
            variables = (
                variables.rename({"sector": "short_sector"})
                .join(
                    pd.MultiIndex.from_arrays(
                        [
                            sectors,
                            sectors.to_series()
                            .str.split("|")
                            .str[0]
                            .rename("short_sector"),
                        ]
                    )
                )
                .droplevel("short_sector")
            )
            countries = (
                variables.to_frame()
                .country.groupby(["gas", "sector"])
                .apply(lambda s: tuple(sorted(s)))
            )
            for cntries, variables in countries.index.groupby(countries).items():
                logger.info(
                    textwrap.fill(
                        print_list(cntries, n=40)
                        + " : "
                        + ", ".join(variables.map("::".join)),
                        width=88,
                    )
                )
                yield CountryGroup(countries=pd.Index(cntries), variables=variables)

    def harmdown_global(
        self, variabledefs: Optional[VariableDefinitions] = None
    ) -> pd.DataFrame:
        if variabledefs is None:
            variabledefs = self.variabledefs

        variables = variabledefs.index_global
        if variables.empty:
            return

        logger.info("Harmonizing and downscaling %d global variables", len(variables))
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

        self.history_aggregated.global_ = hist
        self.harmonized.global_ = harmonized
        self.downscaled.global_ = harmonized.pix.format(
            method="single", country="{region}"
        )

        return harmonized.droplevel("method").rename_axis(index={"region": "country"})

    def harmdown_regional(
        self, variabledefs: Optional[VariableDefinitions] = None
    ) -> pd.DataFrame:
        if variabledefs is None:
            variabledefs = self.variabledefs

        logger.info(
            "Harmonizing and downscaling %d regional variables",
            len(variabledefs.index_regional),
        )
        history_aggregated = []
        harmonized = []
        downscaled = []
        for group in self.country_groups(variabledefs):
            regionmapping = self.regionmapping.filter(group.countries)
            missing_regions = set(self.regionmapping.data.unique()).difference(
                regionmapping.data.unique()
            )

            model = self.model.pix.semijoin(group.variables, how="right").loc[
                isin(region=regionmapping.data.unique())
            ]
            hist = self.hist.pix.semijoin(group.variables, how="right")
            hist_agg = regionmapping.aggregate(hist, dropna=True)

            log_uncovered_history(hist, hist_agg, base_year=self.settings.base_year)
            history_aggregated.append(add_regions_as_zero(hist_agg, missing_regions))

            harm = harmonize(
                model,
                hist_agg,
                overrides=self.harm_overrides.pix.semijoin(
                    group.variables, how="inner"
                ),
                settings=self.settings,
            )
            harmonized.append(add_regions_as_zero(harm, missing_regions))

            harm = aggregate_subsectors(harm.droplevel("method"))
            hist = aggregate_subsectors(hist)

            down = downscale(
                harm,
                hist,
                self.gdp,
                regionmapping,
                settings=self.settings,
            )
            downscaled.append(down)

        if not downscaled:
            return

        self.history_aggregated.regional = concat(history_aggregated)
        self.harmonized.regional = concat(harmonized)
        downscaled = self.downscaled.regional = concat(downscaled)

        return downscaled.droplevel(["method", "region"])

    def harmonize_and_downscale(
        self, variabledefs: Optional[VariableDefinitions] = None
    ) -> pd.DataFrame:
        if variabledefs is None:
            variabledefs = self.variabledefs

        return concat(
            skipnone(
                self.harmdown_global(variabledefs), self.harmdown_regional(variabledefs)
            )
        )

    def grid_proxy(self, proxy_name: str, downscaled: Optional[pd.DataFrame] = None):
        proxy = self.proxies[proxy_name]

        variabledefs = self.variabledefs.for_proxy(proxy_name)
        if downscaled is None:
            downscaled = self.harmonize_and_downscale(variabledefs)
        else:
            downscaled = downscaled.pix.semijoin(
                variabledefs.variable_index, how="inner"
            )

        # Convert unit to kg/s of the repective gas
        downscaled = downscaled.pix.convert_unit(
            lambda s: re.sub("(?:Gt|Mt|kt|t|kg) (.*)/yr", r"kg \1/s", s)
        )

        for model, scenario in downscaled.pix.unique(["model", "scenario"]):
            yield proxy.grid(downscaled.loc[isin(model=model, scenario=scenario)])

    def grid(
        self,
        template_fn: str,
        directory: Optional[Pathy] = None,
        callback: Optional[callable] = None,
        encoding_kwargs: Optional[dict] = None,
        verify: bool = True,
    ):
        def verify_and_save(pathways: Sequence[Gridded]):
            return dask.compute(
                (
                    gridded.to_netcdf(
                        template_fn,
                        callback,
                        directory=directory,
                        encoding_kwargs=encoding_kwargs,
                        compute=False,
                    ),
                    gridded.verify(compute=False) if verify else None,
                )
                for gridded in pathways
            )

        downscaled = self.harmonize_and_downscale()

        return {
            proxy_name: verify_and_save(self.grid_proxy(proxy_name, downscaled))
            for proxy_name in tqdm(self.proxies.keys())
        }

    @property
    def harmonized_data(self):
        hist = self.history_aggregated.data
        model = self.model.pix.semijoin(hist.index, how="right")

        return Harmonized(hist=hist, model=model, harmonized=self.harmonized.data)
