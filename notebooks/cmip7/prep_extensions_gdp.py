# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # GDP Extension: SSP scenarios to 2500
#
# Extends SSP GDP|PPP scenario data from 2100 to 2500 by holding each region's
# value constant at its 2100 level. This flat extension is used as a proxy variable
# for post-2100 gridding weights in the CMIP7 emissions harmonization pipeline
# (concordia). No growth assumption is made — the 2100 value is simply repeated
# for all 5-year steps through 2500.
#
# **Input:** `ssp_basic_drivers_release_3.2.beta_full_gdp.csv` (IAMC wide format)
# from `settings.scenario_path`
#
# **Output:**
# - `ssp_basic_drivers_release_3.2.beta_full_gdp-extensions.csv` — same directory
# - `ssp_gdp_extensions_diagnostic.png` — faceted time-series by region

# %%
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from concordia.settings import Settings

# %%
SETTINGS_FILE: str = "config_cmip7_v0-4-0.yaml"
VERSION_ESGF: str = "1-1-1"
OUT_GDP_EXTENSION_FILENAME: str = "ssp_basic_drivers_release_3.2.beta_full_gdp-extensions.csv"

# Which scenario to run from the markers
marker_to_run: str = "vl" # options: h, hl, m, ml, l, ln, vl

# What folder to save this run in
GRIDDING_VERSION: str | None = f"{marker_to_run}_{VERSION_ESGF}_ext"

# %%
# Get the directory of the current file, works in both script and notebook contexts
# When running through papermill, we need to find the original notebook location
try:
    # Try to get __file__ (works when running as script)
    HERE = Path(__file__).parent
    # Also check if HERE resolved to just current directory, which indicates path resolution failed
    if str(HERE) == "." or HERE == Path("."):
        raise NameError("HERE resolved to current directory, using fallback")
except NameError:
    # When running in notebook/papermill, use a more robust approach
    # Find the concordia repository root and navigate to notebooks/cmip7
    current_path = Path.cwd()
    
    # Look for the concordia root directory (contains pyproject.toml)
    concordia_root = None
    for parent in [current_path] + list(current_path.parents):
        if (parent / "pyproject.toml").exists() and (parent / "src" / "concordia").exists():
            concordia_root = parent
            break
    
    if concordia_root is None:
        raise RuntimeError("Could not find concordia repository root")
    
    HERE = concordia_root / "notebooks" / "cmip7"

settings = Settings.from_config(version=GRIDDING_VERSION,
                                local_config_path=Path(HERE,
                                                       SETTINGS_FILE))

# %%
settings.scenario_path

# %% [markdown]
# ## GDP extension

# %%
# New; updated SSP data from CMIP7 era (downloaded from: http://files.ece.iiasa.ac.at/ssp/downloads/ssp_basic_drivers_release_3.2.beta_full.xlsx, and then selected only the GDP|PPP variable)
gdp_new = pd.read_csv(
        settings.scenario_path / "ssp_basic_drivers_release_3.2.beta_full_gdp.csv",
        index_col=list(range(5)),
    )

# %%
EXT_PROXY_YEARS = [str(y) for y in range(2105, 2501, 5)]

# %%
gdp_new[EXT_PROXY_YEARS] = np.repeat(gdp_new[["2100"]].to_numpy(), len(EXT_PROXY_YEARS), axis=1)

# %%
out_csv = settings.scenario_path / OUT_GDP_EXTENSION_FILENAME
gdp_new.to_csv(out_csv)
print(f"Saved: {out_csv}")

# %% [markdown]
# ## Diagnostic plot

# %%
gdp_plot = gdp_new.reset_index()

year_cols = [c for c in gdp_plot.columns if str(c).isdigit()]
id_cols = [c for c in gdp_plot.columns if c not in year_cols]

# Resolve region/scenario columns by name (IAMC standard) with positional fallback
col_lower = {c.lower(): c for c in id_cols}
region_col = col_lower.get("region", id_cols[2])
scenario_col = col_lower.get("scenario", id_cols[1])

gdp_long = gdp_plot.melt(id_vars=id_cols, value_vars=year_cols, var_name="year", value_name="value")
gdp_long["year"] = gdp_long["year"].astype(int)

unit_col = col_lower.get("unit", id_cols[4])
gdp_long = gdp_long[gdp_long[unit_col] == "billion USD_2017/yr"]

# %%
regions = sorted(gdp_long[region_col].unique())
scenarios = sorted(gdp_long[scenario_col].unique())

n_cols = min(6, len(regions))
n_rows = (len(regions) + n_cols - 1) // n_cols

colors = plt.cm.tab10(np.linspace(0, 0.9, len(scenarios)))
scenario_colors = dict(zip(scenarios, colors))

fig, axes = plt.subplots(n_rows, n_cols,
                         figsize=(n_cols * 3.5, n_rows * 2.8),
                         sharey=False)
axes_flat = np.array(axes).flatten() if n_rows * n_cols > 1 else [axes]

for ax, region in zip(axes_flat, regions):
    subset = gdp_long[gdp_long[region_col] == region]
    for scenario, grp in subset.groupby(scenario_col):
        grp_sorted = grp.sort_values("year")
        ax.plot(grp_sorted["year"], grp_sorted["value"],
                color=scenario_colors[scenario], lw=1.2, label=scenario)
    ax.axvline(2100, color="black", lw=0.8, ls="--", alpha=0.4)
    ax.set_title(region, fontsize=7)
    ax.tick_params(labelsize=6)

for ax in axes_flat[len(regions):]:
    ax.set_visible(False)

legend_handles = [
    plt.Line2D([0], [0], color=c, lw=1.5, label=s)
    for s, c in scenario_colors.items()
]
fig.legend(handles=legend_handles, title="Scenario", loc="upper right",
           ncol=1, fontsize=7, title_fontsize=8,
           framealpha=0.9)

fig.suptitle("GDP|PPP — SSP scenarios with post-2100 constant extension\n(dashed line = 2100)", fontsize=9)
fig.text(0.5, 0.0, "Year", ha="center", fontsize=8)
fig.text(0.01, 0.5, "GDP|PPP (billion USD_2017/yr)", va="center", rotation="vertical", fontsize=8)

plt.tight_layout(rect=[0.02, 0.02, 1, 0.97])

out_png = settings.scenario_path / "ssp_gdp_extensions_diagnostic.png"
plt.savefig(out_png, dpi=120, bbox_inches="tight")
plt.close()
print(f"Saved: {out_png}")

# %%
