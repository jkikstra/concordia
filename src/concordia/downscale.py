import pandas as pd
from pandas_indexing import isin

from aneris.downscaling import Downscaler

from .settings import Settings
from .utils import RegionMapping


def downscale(
    harmonized: pd.DataFrame,
    hist: pd.DataFrame,
    gdp: pd.DataFrame,
    regionmapping: RegionMapping,
    settings: Settings,
) -> pd.DataFrame:
    if harmonized.empty:
        return harmonized.pix.assign(country=[], method=[])

    downscaler = Downscaler(
        harmonized.loc[~isin(region="World")],
        hist,
        settings.base_year,
        regionmapping.data,
        luc_sectors=settings.luc_sectors,
        gdp=gdp,
    )
    methods = downscaler.methods()
    downscaled = downscaler.downscale(methods).sort_index()
    return downscaled.pix.assign(
        method=methods.pix.semijoin(downscaled.index, how="right")
    )
