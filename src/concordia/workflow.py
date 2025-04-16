from __future__ import annotations

import logging
import re
import textwrap
from collections import namedtuple
from functools import cached_property
from itertools import chain
from typing import TYPE_CHECKING

import dask
import pandas as pd
import ptolemy as pt
#' pt.IndexRaster is important. 
#' pt.IndexRaster has:
#' - pt.IndexRaster.boundary example: (country: 234, spatial: 8070), where each row corresponds to a country (234 total), each column corresponds to a spatial cell (8070 grid cells in total). So, raster.boundary.sel(country=ix) would result in an array of shape = (spatial,) = (8070,). Where, spatial is a MultiIndex dimension that ties together. It gives a compact way to store only the relevant grid cells for each country, rather than a full 360×720 grid (which would be 259,200 cells).
from attrs import define
from pandas_indexing import concat, isin
from pandas_indexing.utils import print_list
from pathlib import Path
from tqdm.auto import tqdm

from .downscale import downscale
from .grid import ConcordiaProxy, GriddingContext
from .harmonize import Harmonized, harmonize
from .settings import Settings
from .utils import (
    Pathy,
    RegionMapping,
    VariableDefinitions,
    add_zeros_like,
    aggregate_subsectors,
    skipnone,
    indexraster_info_to_txt,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

    from .grid import Gridded


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
    globallevel: pd.DataFrame | None = None
    regionlevel: pd.DataFrame | None = None
    countrylevel: pd.DataFrame | None = None

    @property
    def data(self):
        return concat([self.globallevel, self.regionlevel, self.countrylevel])


@define(slots=False)
class WorkflowDriver:
    """
    A driver class for orchestrating the harmonization, downscaling, and gridding 
    of emissions data for Integrated Assessment Models (IAMs) and historical datasets.

    The `WorkflowDriver` class integrates various components of the Concordia framework 
    to process emissions data at global, regional, and country levels. It provides 
    methods for harmonizing IAM data with historical data, downscaling to finer spatial 
    resolutions, and generating gridded outputs suitable for climate modeling.

    Attributes:
        model (pd.DataFrame): The IAM scenario data, typically containing emissions 
            trajectories for various gases, sectors, and regions.
        hist (pd.DataFrame): Historical emissions data, used as a baseline for harmonization.
        gdp (pd.DataFrame): GDP data, used as a proxy for downscaling emissions to finer 
            spatial resolutions.
        regionmapping (RegionMapping): A mapping between IAM regions and countries, 
            used for aggregating and disaggregating data.
        indexraster_country (pt.IndexRaster): A raster object representing country-level 
            spatial indices for gridding.
        indexraster_region (pt.IndexRaster): A raster object representing region-level 
            spatial indices for gridding.
        variabledefs (VariableDefinitions): Definitions of variables, including their 
            units, sectors, and associated proxies for downscaling.
        harm_overrides (pd.Series[str]): Overrides for harmonization methods, allowing 
            customization of the harmonization process for specific variables or regions.
        settings (Settings): Configuration settings for the workflow, including paths 
            to input data, output directories, and processing options.
        history_aggregated (GlobalRegional): Aggregated historical data at global, 
            regional, and country levels.
        harmonized (GlobalRegional): Harmonized IAM data at global, regional, and 
            country levels.
        downscaled (GlobalRegional): Downscaled IAM data at global, regional, and 
            country levels.

    Methods:
        proxies:
            Cached property that generates proxy weights for downscaling based on 
            variable definitions and spatial context.
        country_groups(variabledefs=None):
            Generates groups of countries and variables for country-level harmonization 
            and downscaling, based on the availability of proxy weights.
        harmdown_globallevel(variabledefs=None):
            Harmonizes and downscales global-level variables.
        harmdown_regionlevel(variabledefs=None):
            Harmonizes and downscales region-level variables.
        harmdown_countrylevel(variabledefs=None):
            Harmonizes and downscales country-level variables.
        harmonize_and_downscale(variabledefs=None):
            Harmonizes and downscales data at all levels (global, regional, and country).
        grid_proxy(output_variable, downscaled=None):
            Applies a gridding proxy to downscaled data for a specific output variable.
        grid(template_fn, directory=None, callback=None, encoding_kwargs=None, 
             verify=True, skip_exists=False):
            Harmonizes, downscales, and grids all variables, saving the results as 
            NetCDF files.
        harmonized_data:
            Combines historical and harmonized data into a single dataset, formatted 
            for IAMC (Integrated Assessment Modeling Consortium) compatibility.

    Notes:
        - This class is designed to handle large datasets efficiently using Dask for 
          parallel computation.
        - The workflow is modular, allowing users to customize individual steps such 
          as harmonization, downscaling, and gridding.
        - The `WorkflowDriver` is a central component of the Concordia framework, 
          enabling end-to-end processing of emissions data for climate modeling.
    """
    model: pd.DataFrame
    hist: pd.DataFrame
    gdp: pd.DataFrame
    regionmapping: RegionMapping
    indexraster_country: pt.IndexRaster
    indexraster_region: pt.IndexRaster

    variabledefs: VariableDefinitions
    harm_overrides: pd.Series[str]
    settings: Settings

    history_aggregated: GlobalRegional = GlobalRegional()
    harmonized: GlobalRegional = GlobalRegional()
    downscaled: GlobalRegional = GlobalRegional()

    @cached_property
    def proxies(self):
        context = GriddingContext(
            indexraster_country=self.indexraster_country,
            indexraster_region=self.indexraster_region,
            cell_area=self.indexraster_country.cell_area,
        )

        return {
            output_variable: ConcordiaProxy.from_variables(
                self.variabledefs.for_proxy(output_variable),
                context,
                self.settings.proxy_path,
            )
            for output_variable in self.variabledefs.proxies
        }

    def country_groups(
        self, variabledefs: VariableDefinitions | None = None
    ) -> Iterator[CountryGroup]:
        """
        Generate groups of countries and variables for country-level harmonization and downscaling.

        This method divides the countries and variables into groups based on the availability 
        of proxy weights for the variables. It categorizes variables into three types:
        1. Variables without any proxy weights (no associated proxy data).
        2. Variables with proxy weights but no associated weight values.
        3. Variables with valid proxy weights for some countries.

        For each group, it yields a `CountryGroup` object containing:
        - A list of countries.
        - A list of variables to harmonize and downscale.

        Parameters:
            variabledefs (VariableDefinitions | None): 
                The variable definitions to use for grouping. If not provided, 
                `self.variabledefs` is used.

        Yields:
            CountryGroup: A named tuple containing:
                - `countries` (pd.Index): The list of countries in the group.
                - `variables` (pd.Index): The list of variables in the group.

        Notes:
            - Proxy weights are computed for variables using the `self.proxies` dictionary.
            - If no proxy weights are available, all variables are grouped together with all countries.
            - Logs information about the generated groups for debugging and tracking purposes.
        """
        if variabledefs is None:
            variabledefs = self.variabledefs

        all_countries = self.regionmapping.data.index

        # only regional
        variabledefs = variabledefs.countrylevel

        # determine proxy weights for all related proxy variables
        # Uses:
        # - `self.proxies`: A dictionary mapping output variables to `ConcordiaProxy` objects, 
        #   which contain proxy weights for downscaling.
        # - `dask.compute`: Executes computations in parallel for proxy weights.
        # - `proxy.weight["country"].sum("year")`: Sums proxy weights across the "year" dimension for each country.
        # - `concat`: Combines multiple proxy weight series into a single DataFrame.
        # - `variabledefs.countrylevel`: Filters variable definitions to include only country-level variables.
        # - `variabledefs.index.pix.assign`: Adds a `short_sector` column to variables for easier grouping.
        # - `variable_weights.pix.semijoin`: Aligns proxy weights with the variables to harmonize and downscale.
        # - `variable_weights.groupby`: Groups proxy weights by `["gas", "sector"]` to calculate total weights.
        # - `RegionMapping`: Filters and aggregates historical data based on region mappings.
        # - `CountryGroup`: A named tuple containing `countries` and `variables` for harmonization and downscaling.
        # - `logger.info`: Logs information about generated country groups for debugging and tracking.
        regional_proxies = variabledefs.proxies
        variable_weights = [
            w.to_series()
            for w in dask.compute(
                *[
                    proxy.weight["country"].sum("year")
                    for output_variable, proxy in self.proxies.items()
                    if output_variable in regional_proxies and "country" in proxy.weight
                ]
            )
        ]

        if not variable_weights:
            # No proxies, so all variables fall into one group with all countries
            country_groups = [(all_countries, variabledefs.index)]
        else:
            variable_weights = concat(variable_weights)

            # Add a short_sector (which is Energy Sector for Energy Sector|Modelled)
            variables = variabledefs.index.pix.assign(
                short_sector=variabledefs.index.pix.project("sector")
                .str.split("|")
                .str[0]
            )

            # Bring weights into the same form as the variables we want,
            # there are three different types of variables now:
            # 1. those that did not show up in the proxies (here with nan)
            # 2. those that did not have any associated weight
            # 3. those that had proxy weight for some countries
            variable_weights = variable_weights.rename_axis(
                index={"sector": "short_sector"}
            ).pix.semijoin(variables, how="right")

            total_weight = (
                abs(variable_weights).groupby(["gas", "sector"]).sum(min_count=1)
            )

            noproxy_vars = total_weight.index[total_weight.isna()]
            emptyproxy_vars = total_weight.index[total_weight == 0]
            weight_countries = (
                variable_weights.index[abs(variable_weights) > 0]
                # Only consider countries which we can harmonize and downscale
                .join(all_countries, how="inner")
                .to_frame()
                .country.groupby(["gas", "sector"])
                .apply(lambda s: tuple(sorted(s)))
            )

            country_groups = chain(
                [
                    (all_countries, noproxy_vars),  # type 1
                    ([], emptyproxy_vars),  # type 2
                ],
                weight_countries.index.groupby(weight_countries).items(),  # type 3
            )

        for countries, variables in country_groups:
            if variables.empty:
                continue
            logger.info(
                textwrap.fill(
                    print_list(countries, n=40)
                    + " : "
                    + ", ".join(variables.map("::".join)),
                    width=88,
                )
            )
            yield CountryGroup(countries=pd.Index(countries), variables=variables)

    def harmdown_globallevel(
        self, variabledefs: VariableDefinitions | None = None
    ) -> pd.DataFrame | None:
        if variabledefs is None:
            variabledefs = self.variabledefs

        variables = variabledefs.globallevel.index
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

        self.history_aggregated.globallevel = hist # add history_aggregated to the workflow object; global-level aggregated (over sub-sectors, if in the data) historical information 
        self.harmonized.globallevel = harmonized # add harmonized.globallevel dataframe to the workflow object
        self.downscaled.globallevel = harmonized.pix.format( # add 'downscaled'.'globallevel' dataframe to the workflow object
            method="single", # format (accessors.format index levels based on a template refering to other levels) uses core.formatlevel, which uses core._formatlevel, where it is still unclear what `method="single"`  means.
            country="{region}" # add 
        )

        return harmonized.droplevel("method").rename_axis(index={"region": "country"})

    def harmdown_regionlevel(
        self, variabledefs: VariableDefinitions | None = None
    ) -> pd.DataFrame | None:
        if variabledefs is None:
            variabledefs = self.variabledefs
        variabledefs = variabledefs.regionlevel

        if variabledefs.empty:
            return

        logger.info(
            "Harmonizing and downscaling %d variables to region level",
            len(variabledefs.index),
        )

        model = self.model.pix.semijoin(variabledefs.index, how="right")
        hist = self.hist.pix.semijoin(variabledefs.index, how="right")
        hist_agg = self.regionmapping.aggregate(hist, dropna=True)

        harmonized = harmonize(
            model.loc[isin(region=self.regionmapping.data.unique())],
            hist_agg,
            overrides=self.harm_overrides.pix.semijoin(variabledefs.index, how="inner"),
            settings=self.settings,
        )

        harmonized = aggregate_subsectors(harmonized)
        hist_agg = aggregate_subsectors(hist_agg)

        self.history_aggregated.regionlevel = hist_agg
        self.harmonized.regionlevel = harmonized
        self.downscaled.regionlevel = harmonized.pix.format(
            method="single", country="{region}"
        )

        return harmonized.droplevel("method").rename_axis(index={"region": "country"})

    def harmdown_countrylevel(
        self, variabledefs: VariableDefinitions | None = None
    ) -> pd.DataFrame | None:
        if variabledefs is None:
            variabledefs = self.variabledefs

        logger.info(
            "Harmonizing and downscaling %d variables to country level",
            len(variabledefs.countrylevel.index),
        )
        history_aggregated = []
        harmonized = []
        downscaled = []
        for group in self.country_groups(variabledefs):
            regionmapping = self.regionmapping.filter(group.countries)
            missing_regions = set(self.regionmapping.data.unique()).difference(
                regionmapping.data.unique()
            )
            missing_countries = self.regionmapping.data.index.difference(
                group.countries
            )

            model = self.model.pix.semijoin(group.variables, how="right")
            hist = self.hist.pix.semijoin(group.variables, how="right")
            hist_agg = regionmapping.aggregate(hist, dropna=True)

            log_uncovered_history(hist, hist_agg, base_year=self.settings.base_year)
            history_aggregated.append(
                add_zeros_like(hist_agg, hist, region=missing_regions)
            )

            harm = harmonize(
                model.loc[isin(region=regionmapping.data.unique())],
                hist_agg,
                overrides=self.harm_overrides.pix.semijoin(
                    group.variables, how="inner"
                ),
                settings=self.settings,
            )
            harmonized.append(
                add_zeros_like(harm, model, region=missing_regions, method=["all_zero"])
            )

            harm = aggregate_subsectors(harm.droplevel("method"))
            hist = aggregate_subsectors(hist)

            down = downscale(
                harm,
                hist,
                self.gdp,
                regionmapping,
                settings=self.settings,
            )
            downscaled.append(
                add_zeros_like(
                    down,
                    harm,
                    country=missing_countries,
                    method=["all_zero"],
                    derive=dict(region=self.regionmapping.index),
                )
            )

        if not downscaled:
            return

        self.history_aggregated.countrylevel = concat(history_aggregated)
        self.harmonized.countrylevel = concat(harmonized)
        downscaled = self.downscaled.countrylevel = concat(downscaled)

        return downscaled.droplevel(["method", "region"])

    def harmonize_and_downscale(
        self, variabledefs: VariableDefinitions | None = None
    ) -> pd.DataFrame:
        if variabledefs is None:
            variabledefs = self.variabledefs

        return concat(
            skipnone(
                self.harmdown_globallevel(variabledefs),
                self.harmdown_regionlevel(variabledefs),
                self.harmdown_countrylevel(variabledefs),
            )
        )

    def grid_proxy(self, output_variable: str, downscaled: pd.DataFrame | None = None):
        proxy = self.proxies[output_variable]

        variabledefs = self.variabledefs.for_proxy(output_variable)
        if downscaled is None:
            downscaled = self.harmonize_and_downscale(variabledefs)
        else:
            downscaled = downscaled.pix.semijoin(
                variabledefs.downscaling.index, how="inner"
            )

        hist_region = self.history_aggregated.regionlevel
        if hist_region is not None:
            hist_region = hist_region.rename_axis(index={"region": "country"})
        hist = aggregate_subsectors(
            concat(skipnone(self.hist, hist_region)).drop(
                self.settings.base_year, axis=1
            )
        )
        downscaled, hist = downscaled.align(hist, join="left", axis=0)
        tabular = concat([hist, downscaled], axis=1)

        # Convert unit to kg/s of the repective gas, and ?mol/yr to kmol/s
        tabular = tabular.pix.convert_unit(
            lambda s: re.sub(
                "(?:T|G|M|k|)mol (.*)/yr",
                r"kmol \1/s",
                re.sub("(?:Gt|Mt|kt|t|kg) (.*)/yr", r"kg \1/s", s),
            )
        ).rename(index=lambda s: s.rsplit(" ", 1)[0] + " s-1", level="unit")

        for model, scenario in tabular.pix.unique(["model", "scenario"]):
            yield proxy.grid(tabular.loc[isin(model=model, scenario=scenario)])

    def grid(
        self,
        template_fn: str,
        directory: Pathy | None = None,
        callback: Callable | None = None,
        encoding_kwargs: dict | None = None,
        verify: bool = True,
        skip_exists: bool = False,
    ):
    # ╔════════════════════════════╗
    # ║        grid()              ║
    # ╚════════════════════════════╝
    #              │
    #              ▼
    #   harmonize_and_downscale()
    #              │
    #              ▼
    # ┌────────────────────────────────┐
    # │ Loop over output_variable in   │
    # │ self.proxies.keys()            │
    # └────────────────────────────────┘
    #              │
    #              ▼
    # ┌───────────────────────────────┐
    # │  grid_proxy(variable, data)   │
    # │  ↳ Load proxy (e.g. raster)   │
    # │  ↳ Apply to downscaled data   │
    # └───────────────────────────────┘
    #              │
    #              ▼
    # ┌────────────────────────────────────┐
    # │ verify_and_save(gridded results)   │
    # │ ↳ skip if exists                   │
    # │ ↳ save NetCDF with Dask            │
    # └────────────────────────────────────┘

        def verify_and_save(pathways: Sequence[Gridded]):
            def skip(gridded, template_fn, directory):
                fname = gridded.fname(template_fn, directory)
                to_skip = skip_exists and fname.exists()
                if to_skip:
                    logger.log(
                        logging.INFO,
                        f"Skipping {fname} because the file already exists",
                    )
                return to_skip

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
                if not skip(gridded, template_fn, directory)
            )

        downscaled = self.harmonize_and_downscale()

        return {
            output_variable: verify_and_save(
                self.grid_proxy(output_variable, downscaled)
            )
            for output_variable in tqdm(self.proxies.keys())
        }

    @property
    def harmonized_data(self):
        hist = self.history_aggregated.data
        model = self.model.pix.semijoin(hist.index, how="right")

        return Harmonized(
            hist=hist,
            model=model,
            harmonized=self.harmonized.data,
            skip_for_total=self.variabledefs.skip_for_total,
        )
    

    def save_info(self, path: Pathy, prefix: str | None = None):
        """
        Save workflow input data and metadata to the specified directory.

        This method creates a directory (if it doesn't already exist) and writes
        various input data and metadata used in the workflow to separate files.
        The saved files include model data, historical data, GDP data, region
        mappings, index raster information, variable definitions, harmonization
        overrides, and settings.

        Parameters
        ----------
        path : Pathy
            The directory path where the input data and metadata will be saved.
        prefix : str | None, optional
            A prefix to prepend to each filename. If None, no prefix is added.

        Notes
        -----
        - The method ensures that the directory is created before saving the files.
        - Each type of data is saved in a separate file for clarity and organization.
        - The following files are generated:
            - `<prefix>model.csv`: Contains the model data.
            - `<prefix>hist.csv`: Contains the historical data.
            - `<prefix>gdp.csv`: Contains the GDP data.
            - `<prefix>regionmapping.csv`: Contains the region mapping data filtered by GDP countries.
            - `<prefix>indexraster_country_info.txt`: Contains metadata about the index raster.
            - `<prefix>indexraster_region_info.txt`: Contains metadata about the regional index raster.
            - `<prefix>variabledefs.csv`: Contains the variable definitions.
            - `<prefix>harm_overrides.csv`: Contains the harmonization overrides.
            - `<prefix>settings.txt`: Contains the workflow settings in text format.
        """
        # Create the directory
        path.mkdir(parents=True, exist_ok=True)

        # Define a helper function to prepend the prefix to filenames
        def prefixed_filename(filename: str) -> str:
            return f"{prefix}_{filename}" if prefix else filename

        # Write out the input data
        self.model.reset_index().to_csv(Path(path, prefixed_filename("model.csv")))
        self.hist.reset_index().to_csv(Path(path, prefixed_filename("hist.csv")))
        self.gdp.reset_index().to_csv(Path(path, prefixed_filename("gdp.csv")))
        self.regionmapping.filter(self.gdp.pix.unique("country")).data.to_csv(
            Path(path, prefixed_filename("regionmapping.csv"))
        )
        indexraster_info_to_txt(
            self.indexraster_country, Path(path, prefixed_filename("indexraster_country_info.txt"))
        )
        indexraster_info_to_txt(
            self.indexraster_region, Path(path, prefixed_filename("indexraster_region_info.txt"))
        )
        self.variabledefs.data.reset_index().to_csv(Path(path, prefixed_filename("variabledefs.csv")))
        self.harm_overrides.to_csv(Path(path, prefixed_filename("harm_overrides.csv")))
        self.settings.to_txt(Path(path, prefixed_filename("settings.txt")))
