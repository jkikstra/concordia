# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import pandas as pd
from pandas_indexing import isin, concat, ismatch
import pandas_indexing.accessors
from dominate.tags import div
from dominate import document
from concordia import embed_image, add_sticky_toc
from concordia.report import HEADING_TAGS, add_plotly_header
from concordia.utils import RegionMapping, combine_countries
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
from subprocess import run
from joblib import Parallel, delayed
import yaml

# %%
version = "2023-08-09"

# %%
with open("config.yaml") as f:
    config = yaml.safe_load(f)

# %%
base_path = Path(config["base_path"])
out_path = base_path.parent / "analysis" / "harmonization"
country_combinations = config["country_combinations"]

# %%
regionmapping = RegionMapping.from_regiondef(
    base_path / "iam_files/rescue/regionmappingH12.csv",
    country_column="CountryCode",
    region_column="RegionCode",
    sep=";",
)
regionmapping.data = combine_countries(
    regionmapping.data, **country_combinations, agg_func="last"
)

# %%
data = pd.read_csv(out_path / f"harmonization-{version}.csv", index_col=list(range(5)), engine="pyarrow").rename(
    columns=int
).loc[ismatch(scenario="*cp0400*") | isin(model="Historic")]
model = data.pix.extract(
    variable="Emissions|{gas}|{sector}|Unharmonized", drop=True
).dropna(how="all", axis=1)
harm = data.pix.extract(
    variable="Emissions|{gas}|{sector}|Harmonized|{method}", drop=True
).dropna(how="all", axis=1)
hist = (
    data.loc[isin(model="Historic")]
    .pix.extract(variable="Emissions|{gas}|{sector}", drop=True)
    .dropna(how="all", axis=1)
    .pix.dropna(subset="region")
)


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
        base_path / "historical" / "cmip6" / "history.csv",
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
    .pipe(combine_countries, level="region", **country_combinations),
    level="region",
    keepworld=True,
    dropna=True,
)

# %%
hist = concat([hist, cmip6_hist])

# %%
import plotly.express as px

# %%
import plotly.io as pio

# %%
pio.templates.default = "ggplot2"


# %%
def plot_harm(sel, scenario=None, levels=["gas", "sector", "region"], useplotly=False):
    model_sel = sel if scenario is None else sel & isin(scenario=scenario)
    h = harm.loc[model_sel]
    data = concat(
        dict(
            CEDS=hist.loc[sel & isin(scenario="Synthetic (GFED/CEDS/Global)")],
            CMIP6=hist.loc[sel & isin(scenario="CMIP6")],
            Unharmonized=model.loc[model_sel],
            Harmonized=h.droplevel("method"),
        ),
        keys="key",
    ).loc[:, 2000:]

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
            color="key",
            facet_col=non_unique,
            facet_col_wrap=4,
            labels=dict(value=data.pix.unique("unit").item(), key=None),
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
        hue="key",
        facet_kws=dict(sharey=False),
        legend=False,
        height=2,
        aspect=1.5,
    ).set(ylabel=data.pix.unique("unit").item())
    for label, ax in g.axes_dict.items():
        ax.set_title(f"{non_unique} = {label}, method = {methods[label]}", fontsize=9)
    return g


# %%
plot_harm(
    isin(region="World", sector="Total", gas="CO2"),
    scenario="RESCUE-Tier1-Extension-2023-07-27-PkBudg_cp0400-OAE_off",
)

# %%
g = plot_harm(
    isin(sector="Total", gas="CO2"),
    scenario="RESCUE-Tier1-Extension-2023-07-27-PkBudg_cp0400-OAE_off",
    useplotly=True
)

# %%
g.update_layout(height=700, width=1500)
g.update_xaxes(tick0=2000, dtick=20, tickmode="auto")

# %%
embed_image(g)


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
                useplotly=useplotly
            )
        except ValueError as e:
            print(f"During plot_harm(isin(**{dict(zip(index.names, idx))}, ignore_missing_levels=True), {scenario=})")
            raise
        main.add(embed_image(ax, close=True))

        prev_idx = idx

    add_sticky_toc(doc, max_level=2, compact=compact)
    add_plotly_header(doc)
    return doc


# %%
def shorten(scenario):
    return scenario.removeprefix("RESCUE-Tier1-Extension-2023-07-27-")


# %%
files = [out_path / f"harmonization-{version}.csv", out_path / f"harmonization-{version}-splithfc.csv"]

# %%
for scenario in harm.pix.unique("scenario"):
    fn = out_path / f"harmonization-{version}-single-{shorten(scenario)}.html"
    with open(fn, "w", encoding="utf-8") as f:
        print(
            make_doc(
                order=["gas", "sector", "region"], scenario=scenario, compact=False
            ),
            file=f,
        )
    files.append(fn)


# %%
def make_scenario_facets(scenario):
    fn = f"harmonization-{version}-facet-{shorten(scenario)}.html"
    with open(fn, "w", encoding="utf-8") as f:
        print(make_doc(order=["gas", "sector"], scenario=scenario, useplotly=False), file=f)
    return fn
files.extend(
    Parallel(n_jobs=1 #min(10, len(harm.pix.unique("scenario")))
            , verbose=10)(
        delayed(make_scenario_facets)(scenario)
        for scenario in harm.pix.unique("scenario")[:1]
    )
)   

# %%
scenario = harm.pix.unique("scenario")[0]
# !open "harmonization-{version}-facet-{shorten(scenario)}.html"

# %%
for fn in files:
    run(["aws", "s3", "cp", fn, "s3://rescue-task1.3/harmonization/"])

# %%
