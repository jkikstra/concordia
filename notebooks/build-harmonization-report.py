# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python [conda env:concordia]
#     language: python
#     name: conda-env-concordia-py
# ---

# %%
# %load_ext autoreload
# %autoreload 2

# %%
from subprocess import run

import pandas as pd

# %%
import plotly.express as px
import plotly.io as pio
import seaborn as sns
from dominate import document
from dominate.tags import div
from filelock import FileLock
from joblib import Parallel, delayed
from pandas_indexing import concat, isin, ismatch
from tqdm import tqdm

from concordia import add_sticky_toc, embed_image
from concordia.report import HEADING_TAGS, add_plotly_header
from concordia.settings import Settings
from concordia.utils import RegionMapping


pio.templates.default = "ggplot2"

# %%
version_old = None  # "2023-08-18"
version = "2023-12-08"

# %%
settings = Settings.from_config(version=version)

# %%
out_path = settings.out_path / settings.version

# %%
regionmapping = RegionMapping.from_regiondef(
    settings.shared_path / "iam_files/rescue/regionmappingH12.csv",
    country_column="CountryCode",
    region_column="RegionCode",
    sep=";",
)
regionmapping.data = regionmapping.data.pix.aggregate(
    country=settings.country_combinations, agg_func="last"
)


# %%
def read_version(version, variable_template):
    data = (
        pd.read_csv(
            out_path / f"harmonization-{version}.csv",
            index_col=list(range(5)),
            engine="pyarrow",
        )
        .rename(columns=int)
        .loc[
            ismatch(scenario=["*-Baseline", "*-PkBudg_cp2300-OAE_off", "*-PkBudg500-*"])
            | isin(model="Historic")
        ]
    )
    model = data.pix.extract(
        variable=variable_template + "|Unharmonized", drop=True
    ).dropna(how="all", axis=1)
    harm = data.pix.extract(
        variable=variable_template + "|Harmonized|{method}", drop=True
    ).dropna(how="all", axis=1)
    hist = (
        data.loc[isin(model="Historic")]
        .pix.extract(variable=variable_template, drop=True)
        .dropna(how="all", axis=1)
        .pix.dropna(subset="region")
    )

    def aggregate_subsectors(df):
        df_agg = (
            df.pix.assign(method="aggregated") if "method" in df.index.names else df
        )
        return concat(
            [
                df,
                df_agg.pix.aggregate(
                    sector=subsectors.index.groupby(subsectors.str.split("|").str[0]),
                    mode="return",
                ),
            ]
        )

    subsectors = (
        harm.pix.unique("sector")
        .to_series()
        .loc[lambda s: s.str.contains("|", regex=False)]
    )
    if not subsectors.empty:
        print(f"Aggregating subsectors {', '.join(subsectors)} in version {version}")
        model = aggregate_subsectors(model)
        harm = aggregate_subsectors(harm)
        hist = aggregate_subsectors(hist)

    # harm.droplevel("method").pix.aggregate(sector=)
    return model, harm, hist


# %%
model, harm, hist = read_version(version, settings.variable_template)

# %%
if version_old is not None:
    model_old, harm_old, _ = read_version(version_old)


# %%
def extract_sector_gas(df):
    return concat(
        [
            df.pix.extract(
                variable="CEDS+|9+ Sectors|Emissions|{gas}|{sector}|Unharmonized",
                drop=True,
            ),
            df.pix.extract(
                variable="CEDS+|9+ Sectors|Emissions|{gas}|Unharmonized", drop=True
            ).pix.assign(sector="Total"),
        ]
    )


# %%
cmip6_hist = regionmapping.aggregate(
    pd.read_csv(
        settings.shared_path / "historical" / "cmip6" / "history.csv",
        index_col=list(range(5)),
        engine="pyarrow",
    )
    .rename_axis(index=str.lower)
    .rename(index={"Mt CO2-eq/yr": "Mt CO2eq/yr"}, level="unit")
    .rename(columns=int)
    .pipe(extract_sector_gas)
    .pix.assign(
        model="CEDS",
        scenario="CMIP6",
    )
    .pix.aggregate(region=settings.country_combinations),
    level="region",
    keepworld=True,
    dropna=True,
)

# %%
# Recompute regional CO2 total (since they include wrongly fire emissions)
cmip6_hist = concat(
    [
        cmip6_hist.loc[~(isin(gas="CO2", sector="Total") & ~isin(region="World"))],
        cmip6_hist.loc[isin(gas="CO2") & ~isin(sector="Total") & ~isin(region="World")]
        .pix.assign(sector="Total")
        .groupby(cmip6_hist.index.names)
        .sum(),
    ]
).sort_index()

# %%
hist = concat([hist, cmip6_hist])


# %%
def plot_harm(sel, scenario=None, levels=["gas", "sector", "region"], useplotly=False):
    model_sel = sel if scenario is None else sel & ismatch(scenario=scenario)
    h = harm.loc[model_sel]

    data = {}

    data[version] = concat(
        dict(
            CEDS=hist.loc[sel & isin(scenario="Synthetic (GFED/CEDS/Global)")],
            CMIP6=hist.loc[sel & isin(scenario="CMIP6")],
            Unharmonized=model.loc[model_sel],
            Harmonized=h.droplevel("method"),
        ),
        keys="pathway",
    ).loc[:, 2000:]

    if version_old is not None:
        h_old = harm_old.loc[model_sel]
        data[version_old] = concat(
            dict(
                Unharmonized=model.loc[model_sel], Harmonized=h_old.droplevel("method")
            ),
            keys="pathway",
        ).loc[:, 2000:]

    data = concat(data, keys="version")

    non_uniques = [lvl for lvl in levels if len(h.pix.unique(lvl)) > 1]
    if not non_uniques:
        method = h.pix.unique("method").item()
        return data.T.plot(
            ylabel=data.pix.unique("unit").item(),
            title=" - ".join(data.pix.unique(["gas", "sector", "region"]).item())
            + f": {method}",
            legend=False,
        )

    (non_unique,) = non_uniques
    methods = pd.Series(h.index.pix.project("method"), h.index.pix.project(non_unique))

    if useplotly:
        g = px.line(
            data.pix.to_tidy(),
            x="year",
            y="value",
            color="pathway",
            style="version",
            facet_col=non_unique,
            facet_col_wrap=4,
            labels=dict(value=data.pix.unique("unit").item(), pathway="Trajectory"),
        )
        g.update_yaxes(matches=None)

        def add_method(text):
            name, label = text.split("=")
            return f"{name} = {label}, method = {methods[label]}"

        g.for_each_annotation(lambda a: a.update(text=add_method(a.text)))
        return g

    g = sns.relplot(
        data.rename_axis(columns="year").stack().to_frame("value").reset_index(),
        kind="line",
        x="year",
        y="value",
        col=non_unique,
        col_wrap=4,
        hue="pathway",
        style="version",
        facet_kws=dict(sharey=False),
        legend=True,
        height=2,
        aspect=1.5,
    ).set(ylabel=data.pix.unique("unit").item())
    for label, ax in g.axes_dict.items():
        ax.set_title(f"{non_unique} = {label}, method = {methods[label]}", fontsize=9)
    return g


# %%
import matplotlib.pyplot as plt


# %%
plot_harm(
    isin(region="CHA", sector="Energy Sector", gas="CH4"),
    scenario="RESCUE-Tier1-Direct-*-PkBudg500-OAE_off",
)
plt.legend(labels=["CEDS", "CMIP6", "Unharmonized", "Harmonized"], frameon=False)

# %%
g = plot_harm(
    isin(sector="Total", gas="CO2"),
    scenario="RESCUE-Tier1-Direct-*-PkBudg500-OAE_off",
    useplotly=False,
)

# %%
g = plot_harm(
    isin(sector="Aggregate - Agriculture and LUC", gas="CO2"),
    scenario="RESCUE-Tier1-Direct-*-PkBudg500-OAE_off",
    useplotly=False,
)

# %% [markdown]
# # Make Comparison Notebooks


# %%
def what_changed(next, prev):
    length = len(next)
    if prev is None:
        return range(length)
    for i in range(len(next)):
        if prev[i] != next[i]:
            return range(i, length)


# %%
def make_doc(order, scenario=None, compact=False, useplotly=False):
    if scenario is None:
        ((m, s),) = harm.pix.unique(["model", "scenario"])
    else:
        m = harm.pix.unique("model").item()
        s = scenario

    index = harm.index.pix.unique(order).sort_values()
    doc = document(title=f"Harmonization results: {m} - {s}")

    main = doc.add(div())
    prev_idx = None
    for idx in tqdm(index):
        main.add([HEADING_TAGS[i](idx[i]) for i in what_changed(idx, prev_idx)])

        try:
            ax = plot_harm(
                isin(**dict(zip(index.names, idx)), ignore_missing_levels=True),
                scenario=scenario,
                useplotly=useplotly,
            )
        except ValueError:
            print(
                f"During plot_harm(isin(**{dict(zip(index.names, idx))}, ignore_missing_levels=True), {scenario=})"
            )
            raise
        main.add(embed_image(ax, close=True))

        prev_idx = idx

    add_sticky_toc(doc, max_level=2, compact=compact)
    if useplotly:
        add_plotly_header(doc)
    return doc


# %%
def shorten(scenario):
    return scenario.removeprefix("RESCUE-Tier1-Direct-*-").removeprefix(
        "RESCUE-Tier1-Extension-*-"
    )


# %%
files = [
    out_path / f"harmonization-{version}.csv",
    out_path / f"harmonization-{version}-splithfc.csv",
]


# %%
# for scenario in harm.pix.unique("scenario").str.replace(version, "*"):
#     fn = out_path / f"harmonization-{version}-single-{shorten(scenario)}.html"
#     with open(fn, "w", encoding="utf-8") as f:
#         print(
#             make_doc(
#                 order=["gas", "sector", "region"], scenario=scenario, compact=False
#             ),
#             file=f,
#         )
#     files.append(fn)


# %%
def make_scenario_facets(scenario, useplotly=False):
    suffix = "-plotly" if useplotly else ""
    fn = out_path / f"harmonization-{version}-facet-{shorten(scenario)}{suffix}.html"

    lock = FileLock(out_path / ".lock")
    doc = make_doc(order=["gas", "sector"], scenario=scenario, useplotly=useplotly)

    with lock:
        with open(fn, "w", encoding="utf-8") as f:
            print(doc, file=f)
    return fn


files.extend(
    Parallel(n_jobs=min(10, len(harm.pix.unique("scenario"))), verbose=10)(
        delayed(make_scenario_facets)(scenario)
        for scenario in harm.pix.unique("scenario").str.replace(version, "*")
    )
)

# %%
for fn in files:
    run(["aws", "s3", "cp", fn, f"s3://rescue-task1.3/harmonization/{version}/"])

# %%
