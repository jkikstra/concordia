# -*- coding: utf-8 -*-
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
# # QC: Aircraft Emissions v1-1-2 — Monthly Aggregation Plots
#
# Loops over all 7 scenarios (h, hl, l, ln, m, ml, vl) × 10 species and produces two
# plot types per species:
# 1. **Annual totals** — all scenarios overlaid, Mt/yr vs year (2022–2100)
# 2. **Monthly seasonality** — absolute and normalised, for key years 2022, 2050, 2100
#
# Plots saved to `qc_output/aircraft_v1-1-2/` relative to the notebook directory.

# %%
from pathlib import Path

import matplotlib.cm as mplcm
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from concordia.cmip7.utils_plotting import ds_to_monthly_emissions_total

# %%
# --- Paths ---
AIRCRAFT_BASE = Path(
    "/Users/jarmo/Library/CloudStorage/OneDrive-SharedLibraries-IIASA/"
    "ECE.prog - Documents/Projects/CMIP7/IAM Data Processing/"
    "Shared emission fields data/v1_1/all_v1-1-2_aircraft"
)
AREACELLA_PATH = Path(
    "/Users/jarmo/Library/CloudStorage/OneDrive-SharedLibraries-IIASA/"
    "ECE.prog - Documents/Projects/CMIP7/IAM Data Processing/"
    "concordia_cmip7_v0-4-0/input/gridding/"
    "areacella_input4MIPs_emissions_CMIP_CEDS-CMIP-2025-04-18_gn.nc"
)
# HERE = Path(__file__).parent if "__file__" in dir() else Path(".")
HERE = Path(
    "/Users/jarmo/Library/CloudStorage/OneDrive-SharedLibraries-IIASA/"
    "ECE.prog - Documents/Projects/CMIP7/IAM Data Processing/"
    "Shared emission fields data/v1_1/all_v1-1-2_aircraft"
)
OUT_DIR = HERE / "qc_output" / "aircraft_v1-1-2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SCENARIOS = ["h", "hl", "l", "ln", "m", "ml", "vl"]
KEY_YEARS = [2022, 2050, 2100]

# %%
areacella = xr.open_dataset(AREACELLA_PATH)["areacella"]

# %%
# --- Discover files ---

def _species(f: Path) -> str:
    return f.name.split("-")[0]


all_files: dict[tuple[str, str], Path] = {}
for scen in SCENARIOS:
    folder = AIRCRAFT_BASE / f"{scen}_1-1-2"
    for nc in sorted(folder.glob("*.nc")):
        all_files[(_species(nc), scen)] = nc

species_list = sorted({sp for sp, _ in all_files})
print(f"Found {len(all_files)} files: {len(species_list)} species × {len(SCENARIOS)} scenarios")
print(f"Species: {species_list}")

# %%
# --- Helpers ---

def _varname(ds: xr.Dataset) -> str:
    return next(v for v in ds.data_vars if ds[v].dims == ("time", "level", "lat", "lon"))


COLORS = {s: c for s, c in zip(SCENARIOS, mplcm.tab10.colors)}
# Linestyles for the normalised (fraction) row, so scenarios stay distinguishable
# even where colours overlap on the seasonal-shape curves.
LINESTYLES = {s: ls for s, ls in zip(
    SCENARIOS,
    ["-", "--", ":", "-.", (0, (3, 1, 1, 1)), (0, (5, 1)), (0, (1, 1))],
)}

# %% [markdown]
# ## Per-species plots
#
# For each species we produce:
# - `annual_totals_{species}.png`     — trend overview across scenarios
# - `seasonal_cycle_{species}.png`    — absolute + normalised monthly cycles for key years

# %%
for species in species_list:
    print(f"\n--- {species} ---")

    annual: dict[str, xr.DataArray] = {}
    monthly: dict[str, xr.DataArray] = {}

    for scen in SCENARIOS:
        if (species, scen) not in all_files:
            print(f"  MISSING: {species} {scen}")
            continue
        # Chunk on open so the spatial+level reduction runs in parallel via dask.
        ds = xr.open_dataset(all_files[(species, scen)], chunks={"time": 12})
        var = _varname(ds)

        # Compute the monthly global total ONCE (the expensive spatial+level
        # reduction), then derive annual totals by summing months — avoids a
        # second full reduction, and materialises before close() so later
        # .sel()/.values don't re-trigger I/O on a closed dataset.
        mon = ds_to_monthly_emissions_total(ds, var, cell_area=areacella).compute()
        ann = mon.groupby("time.year").sum()
        ann.name = var
        ds.close()

        annual[scen] = ann
        monthly[scen] = mon

        print(
            f"  {scen}: years {int(ann.year[0])}–{int(ann.year[-1])}, "
            f"range {float(ann.min()):.2f}–{float(ann.max()):.2f} Mt/yr"
        )

    if not annual:
        continue

    # ── Figure 1: annual totals ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    for scen, ann in annual.items():
        ax.plot(ann.year.values, ann.values, marker="o", ms=4, lw=1.5,
                label=scen, color=COLORS[scen])
    ax.set(xlabel="year", ylabel="Mt/yr",
           title=f"{species} aircraft — annual global totals")
    ax.legend(title="scenario", fontsize=9, ncols=2)
    ax.grid(alpha=0.3)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"annual_totals_{species}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Figure 2: seasonal cycles for key years ────────────────────────────
    # Only plot years present in all loaded scenarios
    all_scen_years = [set(int(y) for y in mon["time"].dt.year.values)
                      for mon in monthly.values()]
    common_years = set.intersection(*all_scen_years) if all_scen_years else set()
    plot_years = [y for y in KEY_YEARS if y in common_years]

    if not plot_years:
        print(f"  No key years in common — skipping seasonal plot")
        continue

    n_years = len(plot_years)
    # Absolute row (Mt/month) is NOT y-shared: aircraft totals grow several-fold
    # across the key years, so a shared scale squashes the early years until
    # scenarios vanish. Each year autoscales to its own data. The fraction row
    # IS comparable across years, so we share one scale there (set below).
    fig, axes = plt.subplots(2, n_years, figsize=(5 * n_years, 8), sharex=True)
    # axes[0, :] = absolute;  axes[1, :] = normalised
    if n_years == 1:
        axes = axes.reshape(2, 1)

    norm_max = 0.0
    for col, yr in enumerate(plot_years):
        ax_abs  = axes[0, col]
        ax_norm = axes[1, col]

        for scen, mon in monthly.items():
            try:
                mon_yr = mon.sel(time=str(yr))
            except KeyError:
                continue
            months = mon_yr["time"].dt.month.values
            vals   = mon_yr.values

            ax_abs.plot(months, vals, marker="o", ms=4, lw=1.5,
                        label=scen, color=COLORS[scen])
            total = vals.sum()
            if total > 0:
                frac = vals / total
                norm_max = max(norm_max, float(frac.max()))
                ax_norm.plot(months, frac, marker="o", ms=4, lw=1.5,
                             label=scen, color=COLORS[scen],
                             linestyle=LINESTYLES[scen])

        for ax in (ax_abs, ax_norm):
            ax.set_xticks(range(1, 13))
            ax.set_xticklabels(["J","F","M","A","M","J","J","A","S","O","N","D"])
            ax.grid(alpha=0.3)
            ax.set_title(str(yr))

        # Absolute: keep floor at 0, let the top autoscale per year.
        ax_abs.set_ylim(bottom=0)

        ax_abs.set_ylabel("Mt/month") if col == 0 else None
        ax_norm.set_ylabel("fraction of annual total") if col == 0 else None

    # Fraction row: one shared scale across years for like-for-like comparison.
    norm_top = norm_max * 1.05 if norm_max > 0 else 1.0
    for col in range(n_years):
        axes[1, col].set_ylim(0, norm_top)

    axes[1, -1].legend(title="scenario", fontsize=9, loc="upper right")
    fig.suptitle(f"{species} aircraft — monthly seasonality", y=1.01)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"seasonal_cycle_{species}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  Plots saved: annual_totals_{species}.png, seasonal_cycle_{species}.png")

print(f"\nAll done. Output: {OUT_DIR}")

# %%
