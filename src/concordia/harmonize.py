from functools import partial

import pandas as pd
from attrs import define, evolve
from pandas_indexing import concat, isin

from aneris.harmonize import Harmonizer

from .settings import Settings
from .utils import add_totals, aggregate_subsectors, skipnone


def _harmonize(
    model_agg: pd.DataFrame,
    hist_agg: pd.DataFrame,
    config: dict,
    overrides: pd.DataFrame,
    settings: Settings,
) -> pd.DataFrame:
    harmonized = []
    for m, s in model_agg.index.pix.unique(["model", "scenario"]):
        scen = model_agg.loc[isin(model=m, scenario=s)].droplevel(["model", "scenario"])
        h = Harmonizer(
            scen,
            hist_agg.pix.semijoin(scen.index, how="right").loc[:, 2000:],
            harm_idx=scen.index.names,
            config=config,
        )
        result = h.harmonize(
            year=settings.base_year, overrides=None if overrides.empty else overrides
        ).sort_index()
        methods = h.methods(year=settings.base_year)
        result = result.pix.assign(
            method=methods.pix.semijoin(result.index, how="right")
        )
        harmonized.append(result.pix.assign(model=m, scenario=s))

    return concat(harmonized) if harmonized else None


def harmonize(
    model: pd.DataFrame, hist: pd.DataFrame, overrides: pd.DataFrame, settings: Settings
) -> pd.DataFrame:
    if model.empty:
        return model.loc[:, settings.base_year :].pix.assign(method=[])

    is_luc = isin(sector=settings.luc_sectors)
    harmonized = concat(
        skipnone(
            _harmonize(
                model.loc[is_luc],
                hist.loc[is_luc],
                config=dict(),
                overrides=overrides.loc[is_luc],
                settings=settings,
            ),
            _harmonize(
                model.loc[~is_luc],
                hist.loc[~is_luc],
                config=dict(default_luc_method="reduce_ratio_2080"),
                overrides=overrides.loc[~is_luc],
                settings=settings,
            ),
        )
    )

    return harmonized


@define
class Harmonized:
    hist: pd.DataFrame
    model: pd.DataFrame
    harmonized: pd.DataFrame
    skip_for_total: pd.MultiIndex

    def drop_method(self):
        return evolve(self, harmonized=self.harmonized.droplevel("method"))

    def pipe(self, func: callable, *args, **kwargs):
        f = partial(func, *args, **kwargs)
        return self.__class__(
            f(self.hist), f(self.model), f(self.harmonized), self.skip_for_total
        )

    def add_totals(self):
        return self.pipe(add_totals, skip_for_total=self.skip_for_total)

    def aggregate_subsectors(self):
        return self.pipe(aggregate_subsectors)

    def split_hfc(self, hfc_distribution):
        def split(df):
            return concat(
                [
                    df.loc[~isin(gas="HFC")],
                    df.pix.multiply(
                        hfc_distribution.pix.assign(gas="HFC"), join="inner"
                    )
                    .droplevel("gas")
                    .rename_axis(index={"hfc": "gas"}),
                ]
            )

        return self.pipe(split)

    def to_iamc(
        self,
        template: str = "Emissions|{gas}|{sector}",
        hist_model="Historic",
        hist_scenario="Historic",
    ) -> pd.DataFrame:
        harmonized_template = (
            f"{template}|Harmonized|{{method}}"
            if "method" in self.harmonized.index.names
            else f"{template}|Harmonized"
        )
        return concat(
            [
                self.model.pix.format(variable=f"{template}|Unharmonized", drop=True),
                self.harmonized.pix.format(variable=harmonized_template, drop=True),
                self.hist.pix.format(
                    variable=template,
                    model=hist_model,
                    scenario=hist_scenario,
                    drop=True,
                ),
            ],
            order=["model", "scenario", "region", "variable", "unit"],
        ).sort_index(axis=1)
