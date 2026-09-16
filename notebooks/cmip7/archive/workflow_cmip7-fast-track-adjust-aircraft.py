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
# ## Notes of the issue:
# The cmip7 fast track workflow for aircraft emissions is based on the seasonality of the 2023 CEDS data, which is incorrect. 
# This means that the annual totals are correct, but the distribution of emissions across months (there is a time point representing the average emissions in each month) is incorrect. 
# This means that the seasonality of the future scenarios is also incorrect.
# We need to adjust this, and there are two possible solutions, which are described below.
# For now, only the 'Solution 2' is implemented, but we can also implement 'Solution 1' if needed (but it will take more time to run, and we don't have the correct seasonality file for that yet, so we would need to create that first).

# %% [markdown]
# ## Two possible solutions:
# 1. Rerun the Aircraft workflow, with adjusted seasonality file (i.e., create new proxies, and pull from those proxies).
# 2. Load files, and adjust seasonality following Steve's method ("Em(sen,em,x,y,z,t)_correct = Em(sen,em,x,y,z,2022) * E(sen,em,t)/(E(sen,em,2022)"), and save to new files (i.e., create new proxies, and pull from those proxies).

# %% [markdown]
# # Solution 1: base it on:
# a. proxies, aircraft bit from: notebooks/cmip7/prep_proxyfuture-anthro-from-ceds-cmip7-esgf.py
# b. workflow bit from: notebooks/cmip7/workflow_cmip7-fast-track.py

# %% [markdown]
# # Solution 2: adjust the seasonality of the existing gridded files ("Steve's method")
#
# **The problem in one line.** Each gridded file gives an aircraft emission *rate*
# (units `kg m-2 s-1`) for every month, altitude level, and grid cell. The yearly *totals*
# are correct, but the way each year's total is spread across the 12 months (the *seasonality*)
# was copied from the wrong reference year (CEDS 2023) and is therefore wrong.
#
# **The idea.** Any monthly value carries two separable pieces of information:
#
# - a **trend** — how the annual total rises or falls from year to year, and
# - a **seasonality** — how a single year's total is distributed over its 12 months.
#
# The scenario has the **trend** right but the **seasonality** wrong. CEDS 2022 (real
# observations, correctly resolved by month) has the **seasonality** right. So we keep the trend
# from the scenario, take the seasonality from CEDS 2022, and multiply the two back together.
#
# **The formula** (species *em*, grid point *(x,y,z)* = level/lat/lon, time *t* = year + month):
#
# $$ \mathrm{Em}_\text{correct}(x,y,z,t) \;=\; \underbrace{\frac{\mathrm{Em}_\text{CEDS}\big(x,y,z,\,2022,\,\mathrm{month}(t)\big)}{E_\text{CEDS}(2022)}}_{\text{seasonal template, integrates to 1}} \;\times\; \underbrace{E_\text{scen}\big(\mathrm{year}(t)\big)}_{\text{scenario annual total}} $$
#
# where $E_\text{scen}(y)$ is the scenario's **global annual total** in year $y$ (a single number
# per year) and $E_\text{CEDS}(2022)$ is the template's own global annual total. Dividing by
# $E_\text{CEDS}(2022)$ normalises the CEDS field so it integrates to exactly 1 over the year, so
# multiplying by $E_\text{scen}(y)$ reproduces the scenario's annual total **exactly**.
#
# **Step by step** (this matches the code cell below):
#
# 1. **Pick a species and load two files:** the future *scenario* file (correct totals, wrong
#    seasonality) and the historical *CEDS* file, sliced to the reference year 2022 (correct
#    seasonality).
# 2. **Compute the scenario's annual totals** $E_\text{scen}(y)$ by integrating the rate over
#    space and time: multiply by grid-cell area (`areacella`, in m²) and by the seconds in each
#    month (days-in-month × 86400), then sum over level, latitude, longitude **and** the 12
#    months. Summing the months collapses away the (wrong) monthly detail and keeps only the
#    (correct) yearly total.
# 3. **Form the trend factor** $\rho(y) = E_\text{scen}(y) / E_\text{CEDS}(2022)$, dividing the
#    scenario's annual total by the *template's* annual total (also computed in step 2, the same
#    way, on the CEDS 2022 slice). Because we divide by the CEDS total rather than the scenario's
#    own 2022 total, rescaling the template by $\rho(y)$ reproduces the scenario's yearly totals
#    **exactly**.
# 4. **Take the CEDS 2022 field as the seasonal template:** its 12 monthly gridded slices define
#    the correct month-to-month shape (and full spatial pattern).
# 5. **Rebuild every future month** by taking the matching CEDS-2022 *month* and multiplying it by
#    that *year's* trend factor $\rho(y)$. The result keeps the scenario's year-to-year trend but
#    carries CEDS 2022's seasonality everywhere.
#
# **Why the trend factor uses the *annual* total, not the monthly value.** Summing all 12 months
# into one yearly number deliberately throws away the scenario's monthly distribution — the part
# we do not trust — while preserving the annual total, which we do. Every bit of monthly structure
# in the output then comes from CEDS 2022, not from the scenario.
#
# **Why we do not re-weight the months by hand.** Seasonality is stored as a *rate* (`kg m-2 s-1`).
# The conversion to actual *mass* per month (rate × seconds-in-month) is applied later, at
# integration time, so a short February automatically receives proportionally less mass than a long
# July. Re-balancing the months explicitly would double-count the calendar.
#
# **Annual totals are preserved exactly.** Because the template is divided by its own annual total
# $E_\text{CEDS}(2022)$, the corrected *global* annual total equals $E_\text{scen}(y)$ to machine
# precision for every year (the sanity check in the code cell below prints a difference of ~0.000%).
#
# **Starting-point assumption — seasonality is treated as spatially uniform.** In principle aircraft
# seasonality *can* be spatially heterogeneous: the Northern Hemisphere has a boreal-summer travel
# peak, the Southern Hemisphere an austral-summer one, and different regions follow different flight
# patterns. **In this CEDS 2022 data, however, it is not** — the normalised monthly shape is
# identical across latitude bands (NH, SH, tropics, polar all peak in July with the same 12 fractions
# to four decimals), i.e. CEDS built these files by applying a single global seasonal curve to a
# spatial annual map. The method relies on this: it copies the full gridded template, so it would
# faithfully carry *any* spatial seasonality that were present, but here there is none to carry. If a
# future input file does have spatially-varying seasonality, this assumption should be revisited.
#
# **What is *not* preserved — the spatial/vertical pattern.** The trend factor $\rho(y)$ is a single
# global scalar, so the corrected field's spatial and vertical pattern is taken wholesale from CEDS
# 2022 and scaled uniformly each year. The scenario's regional distribution and any change in *where*
# emissions occur over time (e.g. faster aviation growth over Asia than Europe) are therefore lost —
# only the global totals and the (spatially-uniform) CEDS seasonality survive. Preserving the
# scenario's evolving geography would require rescaling per grid cell instead of with one global factor.
#
# **Calendar.** These files use a `noleap` (365-day) calendar, so every February has 28 days and the
# corrected totals are exact for all years. (With a real Gregorian calendar, leap-year Februaries
# would shift that year's total by ~0.3%.)


# %% [markdown]
# ## Identify all locations of the data files
# - The files to adjust are in (for instance): /Users/jarmo/Library/CloudStorage/OneDrive-SharedLibraries-IIASA/ECE.prog - Documents/Projects/CMIP7/IAM Data Processing/Shared emission fields data/v1_1/aircraft_seasonality_adjustment/cmip7_v111_files
# - The "correct '2022' data" is in (for instance): '/Users/jarmo/Library/CloudStorage/OneDrive-SharedLibraries-IIASA/ECE.prog - Documents/Projects/CMIP7/IAM Data Processing/concordia_cmip7_v0-4-0/input/gridding/esgf/ceds/CMIP7_AIR'

# %%
# Choose year with correct historical data:
year_with_correct_data = 2022

# %%
# Choose what to run: 
RUN_INTERACTIVE_TEST_ONE_SCENARIO = False  # run the single-file test of Steve's method, with diagnostic checks and plots

# %%
# List of files to adjust:
from pathlib import Path
import xarray as xr
# folder_with_air_files = Path('/Users/jarmo/Library/CloudStorage/OneDrive-SharedLibraries-IIASA/ECE.prog - Documents/Projects/CMIP7/IAM Data Processing/Shared emission fields data/v1_1/aircraft_seasonality_adjustment/cmip7_v111_files')
folder_with_air_files = Path('/Users/jarmo/Documents/GitHub/mozart/projects/mine/concordia/data/cmip7_v111_air')
scenario_air_files = list(folder_with_air_files.glob('*.nc'))

# folder_with_ceds_air_files = Path('/Users/jarmo/Library/CloudStorage/OneDrive-SharedLibraries-IIASA/ECE.prog - Documents/Projects/CMIP7/IAM Data Processing/concordia_cmip7_v0-4-0/input/gridding/esgf/ceds/CMIP7_AIR')
folder_with_ceds_air_files = Path('/Users/jarmo/Documents/GitHub/mozart/projects/mine/concordia/data/CMIP7_AIR')

ceds_air_files = list(folder_with_ceds_air_files.glob('*.nc'))

new_folder = Path('/Users/jarmo/Documents/GitHub/mozart/projects/mine/concordia/data/cmip7_v111_air_adjusted')


def file_ids(f):
    """Parse an input4MIPs filename into (species, source_id).

    e.g. 'CO2-em-AIR-anthro_input4MIPs_emissions_ScenarioMIP_IIASA-IAMC-m-1-1-1_gn_202201-210012.nc'
    -> ('CO2', 'IIASA-IAMC-m-1-1-1'). For ScenarioMIP files the source_id identifies the scenario;
    for CEDS files it is the CEDS version (e.g. 'CEDS-CMIP-2025-04-18').
    """
    parts = f.name.split("_")
    return parts[0].split("-")[0], parts[4]


# %% 
# load the gridcell area file, which is needed for the area-weighted integration to compute global totals.
# gridcell area (m2), shared by the scenario and CEDS grids — needed for true global totals.
# NB: aircraft emissions are kg m-2 s-1 per *horizontal* area, so the vertical reduction is a
# plain sum over 'level' (done inside the helpers), NOT a dz-weighted integral.
areacella = xr.open_dataset(
    Path("/Users/jarmo/Documents/GitHub/mozart/projects/mine/concordia/data/areacella_input4MIPs_emissions_CMIP_CEDS-CMIP-2025-04-18_gn.nc")
)["areacella"]


# %%
# validated global-total helpers (area-weighted, days-in-month weighting, sum over level)
from concordia.cmip7.utils_plotting import (
    ds_to_annual_emissions_total,
    ds_to_monthly_emissions_total,
)

# %%
# load one scenario and one CEDS file to work with.
# CEDS carries the *correct* historical seasonality: keep the full record (ds_ceds_full)
# for diagnostics and a single reference-year slice (ds_ceds) for Steve's adjustment.

if RUN_INTERACTIVE_TEST_ONE_SCENARIO:
    species = "CO2"
    scenario = "IIASA-IAMC-m-1-1-1"  # source_id of the scenario file to test
    ds_scen = xr.open_dataset(next(f for f in scenario_air_files if file_ids(f) == (species, scenario)))
    ds_ceds_full = xr.open_dataset(next(f for f in ceds_air_files if file_ids(f)[0] == species))
    ds_ceds = ds_ceds_full.sel(time=str(year_with_correct_data))

    # main gridded variable (the one carrying time, level, lat, lon)
    varname = next(v for v in ds_scen.data_vars if ds_scen[v].dims == ("time", "level", "lat", "lon"))


# %%
# do example calculation for one scenario and one file, following steve's method:
#   Em_correct(x,y,z,t) = Em_CEDS(x,y,z,2022,month(t)) / E_CEDS(2022) * E_scen(year(t))
# ...em    = emission species
# ...x,y,z = spatial dimensions (level, lat, lon)
# ...t     = time dimension (year, month)
# (Steve's note wrote the divisor as the scenario's own 2022 total; we divide by the CEDS-2022
#  template total E_CEDS(2022) instead, which preserves the scenario's annual totals *exactly* —
#  see the "Solution 2" markdown above for the reasoning.)
#
# Reading: take the *correct* gridded seasonality from CEDS 2022 (ds_ceds, 12 monthly slices),
# normalise it by its own global annual total E_CEDS(2022) so the template integrates to 1, then
# scale it year-by-year by the scenario's global annual total E_scen(year). All totals use proper
# area- and days-in-month weighting (ds_to_annual_emissions_total) — the same accounting that
# reproduces the files' reported global_total_emissions attributes to the decimal.

if RUN_INTERACTIVE_TEST_ONE_SCENARIO:
    # E_scen(year): scenario global annual totals [Mt/yr]. E_CEDS(2022): the template's annual total.
    scen_annual = ds_to_annual_emissions_total(ds_scen, varname, cell_area=areacella)
    ceds_ref_annual = float(ds_to_annual_emissions_total(ds_ceds, varname, cell_area=areacella).sel(year=year_with_correct_data))

    # trend factor E_scen(year) / E_CEDS(2022): rescales the CEDS-2022 template to the scenario's
    # annual total, so the corrected global annual totals equal the scenario's *exactly*.
    ratio = scen_annual / ceds_ref_annual

    # Em(...,2022): CEDS 2022 field re-indexed from time -> calendar month (the correct seasonality).
    ceds_season = ds_ceds[varname]
    ceds_season = (
        ceds_season.assign_coords(month=("time", ceds_season["time.month"].data))
        .swap_dims({"time": "month"})
        .drop_vars("time")
    )

    # pointwise-select CEDS month and scenario-year ratio onto the full scenario time axis, then multiply
    corrected = ceds_season.sel(month=ds_scen["time.month"]) * ratio.sel(year=ds_scen["time.year"])
    corrected = corrected.drop_vars(["month", "year"]).transpose("time", "level", "lat", "lon")
    corrected.name = varname
    corrected.attrs = dict(ds_scen[varname].attrs)  # carry units etc. for later saving

    # sanity check: corrected global annual totals now reproduce the scenario's *exactly* (~0.000%).
    # NB only the GLOBAL totals are preserved: the corrected field's spatial/vertical pattern is taken
    # wholesale from CEDS 2022 and scaled by one global factor per year, so the scenario's regional
    # distribution and any change in *where* emissions occur over time are not retained.
    corrected_ds = ds_scen.copy()
    corrected_ds[varname] = corrected
    ann_corr = ds_to_annual_emissions_total(corrected_ds, varname, cell_area=areacella)
    for yr in (year_with_correct_data, 2100):
        scen_y = float(scen_annual.sel(year=yr))
        corr_y = float(ann_corr.sel(year=yr))
        print(f"annual total {yr} [Mt]:  scenario = {scen_y:.3f}   corrected = {corr_y:.3f}   ({(corr_y / scen_y - 1) * 100:+.3f}%)")


# %% [markdown]
# ## Loop over all files, and apply the adjustment, and save to new files (i.e., create new proxies, and pull from those proxies).

# %%
# 1. function capturing the adjustment for a single (scenario, CEDS) file pair.
#    Builds a dataset identical to the scenario file except the emission field, which is replaced
#    by the seasonality-corrected one (CEDS reference-year shape, scenario annual totals preserved
#    exactly), writes it to out_dir, and returns the path + max annual-total error for a quick check.
#    Sequential / in-memory: one whole file (~10 GB) is held in RAM at a time, no dask.
import gc


def adjust_and_save(scen_file, ceds_file, ref_year, cell_area, out_dir):
    ds_s = xr.open_dataset(scen_file)
    ds_c = xr.open_dataset(ceds_file).sel(time=str(ref_year))
    var = next(v for v in ds_s.data_vars if ds_s[v].dims == ("time", "level", "lat", "lon"))

    # trend factor: E_scen(year) / E_CEDS(ref_year)  -> rescales the template to scenario totals
    scen_annual = ds_to_annual_emissions_total(ds_s, var, cell_area=cell_area)
    ceds_ref_annual = float(ds_to_annual_emissions_total(ds_c, var, cell_area=cell_area).sel(year=ref_year))
    ratio = scen_annual / ceds_ref_annual

    # CEDS reference-year field as a month-indexed seasonal template
    template = ds_c[var]
    template = (
        template.assign_coords(month=("time", template["time.month"].data))
        .swap_dims({"time": "month"})
        .drop_vars("time")
    )

    # rebuild on the scenario time axis: template[month(t)] * ratio[year(t)]
    corrected = template.sel(month=ds_s["time.month"]) * ratio.sel(year=ds_s["time.year"])
    corrected = corrected.drop_vars(["month", "year"]).transpose("time", "level", "lat", "lon")

    # copy the scenario file and replace only the emission field (keeps coords, bounds, attrs)
    out = ds_s.copy()
    out[var] = corrected
    out[var].attrs = dict(ds_s[var].attrs)        # keep units, long_name, cell_methods, ...
    out[var].encoding = dict(ds_s[var].encoding)  # keep dtype/_FillValue
    out.attrs["seasonality_adjustment"] = (
        f"Monthly seasonality replaced with CEDS {ref_year} pattern following Steve's method "
        f"(template normalised by its own annual total, so the scenario's global annual totals "
        f"are preserved exactly). Generated by workflow_cmip7-fast-track-adjust-aircraft.py."
    )

    out_path = out_dir / scen_file.name
    out.to_netcdf(out_path)

    # verify the corrected field reproduces the scenario annual totals
    corr_annual = ds_to_annual_emissions_total(out, var, cell_area=cell_area)
    max_err = float(abs(corr_annual / scen_annual - 1).max()) * 100

    ds_s.close()
    ds_c.close()
    del ds_s, ds_c, out, corrected, template
    gc.collect()
    return out_path, max_err


# 2. loop over all scenario files, match each to its CEDS counterpart, adjust, and save.
new_folder.mkdir(parents=True, exist_ok=True)

# match by the leading "{species}-em-AIR-anthro" token shared by both naming conventions
ceds_by_key = {f.name.split("_")[0]: f for f in ceds_air_files}

for scen_file in sorted(scenario_air_files):
    key = scen_file.name.split("_")[0]
    ceds_file = ceds_by_key.get(key)
    if ceds_file is None:
        print(f"SKIP {key}: no matching CEDS file")
        continue

    out_path, max_err = adjust_and_save(scen_file, ceds_file, year_with_correct_data, areacella, new_folder)
    print(f"{key:18s} -> {out_path.name}   (max annual-total error {max_err:.3f}%)")
# %% [markdown]
# # Other diagnostics:
# 1. Is 2022 scenario data different from CEDS 2022?
# 2. Is the seasonality of CEDS 2022 different from CEDS 2023?
#
# Required functionality (both live in `concordia.cmip7.utils_plotting`):
# 1. `ds_to_monthly_emissions_total` — vertical integration across 'level' -> global Mt/month
#    (keeps the monthly time axis; for inspecting seasonality).
# 2. `ds_to_annual_emissions_total` — vertical integration across 'level' AND correct
#    month aggregation (days-in-month weighting) -> global Mt/year.
# Both accept the full file or one already filtered to a single year.

# %% [markdown]
# ## Diagnostic 1: is 2022 scenario data different from CEDS 2022?
# Compare the global annual total for the reference year between the two sources.
# (functions, areacella, ds_scen, ds_ceds_full and varname come from the loading cell above)

# %%
scen_annual_tot = ds_to_annual_emissions_total(ds_scen, varname, cell_area=areacella)
ceds_annual_tot = ds_to_annual_emissions_total(ds_ceds_full, varname, cell_area=areacella)

scen_2022 = float(scen_annual_tot.sel(year=year_with_correct_data))
ceds_2022 = float(ceds_annual_tot.sel(year=year_with_correct_data))
print(f"annual total {species} {year_with_correct_data} [Mt]:  scenario ({scenario}) = {scen_2022:.3f}   CEDS = {ceds_2022:.3f}")
print(f"  difference: {scen_2022 - ceds_2022:+.3f} Mt  ({(scen_2022/ceds_2022 - 1)*100:+.2f}%)")

# %% [markdown]
# ## Diagnostic 2: is the seasonality of CEDS 2022 different from CEDS 2023?
# Compare the *normalised* monthly distribution (each year's months sum to 1) between 2022 and 2023.

# %%
import numpy as np

ceds_monthly = ds_to_monthly_emissions_total(ds_ceds_full, varname, cell_area=areacella)

# normalise within each year -> seasonal fractions (each year sums to 1, independent of annual level)
season = ceds_monthly.groupby("time.year") / ceds_monthly.groupby("time.year").sum()
season = season.assign_coords(month=season["time"].dt.month, yr=season["time"].dt.year)

# re-index each year's 12 points by calendar month so they line up for differencing
s2022 = season.where(season["yr"] == 2022, drop=True).swap_dims({"time": "month"}).drop_vars(["time", "yr"])
s2023 = season.where(season["yr"] == 2023, drop=True).swap_dims({"time": "month"}).drop_vars(["time", "yr"])
diff = s2023 - s2022

print(f"CEDS {species} seasonality, 2022 vs 2023 (fraction of annual per month):")
for m in range(1, 13):
    print(f"  month {m:2d}: 2022={float(s2022.sel(month=m)):.4f}  2023={float(s2023.sel(month=m)):.4f}  diff={float(diff.sel(month=m)):+.4f}")
print(f"max |monthly fraction difference|: {float(np.abs(diff).max()):.4f}")


# %% [markdown]
# ## Diagnostic plots: monthly seasonality, before vs after the adjustment
# Visual check for the current `species`, comparing the original (wrong) scenario seasonality, the
# corrected output, and the CEDS reference year:
# - **Seasonal shape** (left): the corrected curve should land on top of the CEDS reference, while
#   the original sits elsewhere.
# - **Monthly totals** (right): original vs corrected for one year — the distribution changes but the
#   area under the curve (the annual total) is unchanged.
#
# NB cftime dates are plotted as month numbers / decimal years on purpose, so the figures do not
# need the optional `nc_time_axis` package.

# %%
import matplotlib.pyplot as plt

plot_year = 2022  # which future year to inspect

# corrected output file written by the loop above, for the current (species, scenario)
adj_file = new_folder / next(f for f in scenario_air_files if file_ids(f) == (species, scenario)).name
ds_adj = xr.open_dataset(adj_file)

# monthly global totals [Mt/month]
m_scen = ds_to_monthly_emissions_total(ds_scen, varname, cell_area=areacella)   # original scenario
m_adj = ds_to_monthly_emissions_total(ds_adj, varname, cell_area=areacella)     # corrected output
m_ceds = ds_to_monthly_emissions_total(ds_ceds, varname, cell_area=areacella)   # CEDS reference year (12 pts)


def by_month(monthly, year=None):
    """Return (month numbers 1-12, values) for a monthly series, optionally one year."""
    sel = monthly if year is None else monthly.sel(time=str(year))
    return sel["time"].dt.month.values, sel.values


mo, ceds_v = by_month(m_ceds)
mo_s, scen_v = by_month(m_scen, plot_year)
mo_a, adj_v = by_month(m_adj, plot_year)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

# left: normalised seasonal shape (fraction of annual total)
ax1.plot(mo, ceds_v / ceds_v.sum(), "k--o", lw=2, label=f"CEDS {year_with_correct_data} (target shape)")
ax1.plot(mo_s, scen_v / scen_v.sum(), "C3-o", label=f"scenario {plot_year} (original)")
ax1.plot(mo_a, adj_v / adj_v.sum(), "C0-s", label=f"scenario {plot_year} (corrected)")
ax1.set(xlabel="month", ylabel="fraction of annual total", title=f"{species}: seasonal shape")
ax1.set_xticks(range(1, 13))
ax1.legend()
ax1.grid(alpha=0.3)

# right: absolute monthly totals (annual total preserved -> equal area under original vs corrected)
ax2.plot(mo_s, scen_v, "C3-o", label=f"scenario {plot_year} (original)")
ax2.plot(mo_a, adj_v, "C0-s", label=f"scenario {plot_year} (corrected)")
ax2.set(xlabel="month", ylabel="emissions [Mt/month]", title=f"{species}: monthly totals, {plot_year}")
ax2.set_xticks(range(1, 13))
ax2.legend()
ax2.grid(alpha=0.3)

fig.suptitle(f"Aircraft {species} — {scenario} — seasonality adjustment (annual totals preserved)", y=1.02)
fig.tight_layout()
fig.savefig(new_folder / f"diagnostic_adjustment-{plot_year}_{species}_{scenario}.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Monthly time-series over the last few years: the seasonal 'texture' before vs after.
# x-axis is decimal years (year + (month-0.5)/12) to avoid needing nc_time_axis.
win = slice("2022", "2031")
ts_scen = m_scen.sel(time=win)
ts_adj = m_adj.sel(time=win)
x = ts_scen["time"].dt.year.values + (ts_scen["time"].dt.month.values - 0.5) / 12

plt.figure(figsize=(13, 4))
plt.plot(x, ts_scen.values, "C3-o", ms=3, label=f"original {scenario} (wrong seasonality)")
plt.plot(x, ts_adj.values, "C0-s", ms=3, label=f"corrected {scenario} (CEDS{year_with_correct_data} seasonality)")
plt.xlabel("year")
plt.ylabel(f"{species} aircraft emissions [Mt/month]")
plt.title(f"Monthly aircraft {species} — {scenario}: {win.start}-{win.stop}, original vs corrected")
plt.ylim(bottom=0)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(new_folder / f"diagnostic_timeseries-window_{species}_{scenario}.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# ## Summary diagnostic: climatological monthly means for all (species, scenario) pairs
# For every (species, scenario) pair that has a matching original and adjusted file, plus a CEDS
# file for the species, plot:
#   - CEDS historical mean monthly cycle (averaged over all available historical years)
#   - original scenario mean monthly cycle (averaged over all scenario years, wrong seasonality)
#   - adjusted scenario mean monthly cycle (corrected seasonality)
# Left panel: normalised seasonal shape (fraction of annual total); right panel: absolute [Mt/month].
# Figure filenames carry both species and scenario (source_id), so runs over multiple scenarios
# do not overwrite each other.

adj_files_all = list(new_folder.glob("*.nc"))
orig_by_ids = {file_ids(f): f for f in sorted(scenario_air_files)}
adj_by_ids  = {file_ids(f): f for f in sorted(adj_files_all)}
ceds_by_sp  = {file_ids(f)[0]: f for f in sorted(ceds_air_files)}  # CEDS is per species only

# CEDS years to include in the climatological mean (set to None to use all available years)
ceds_clim_years = slice("2022", "2022")

# time-series window: first and last year shown in the monthly time-series plot
ts_year_first = "2015"
ts_year_last  = "2035"

common_ids = sorted(ids for ids in orig_by_ids if ids in adj_by_ids and ids[0] in ceds_by_sp)
print(f"(species, scenario) pairs with all three file types ({len(common_ids)}):")
for ids in common_ids:
    print(f"  {ids[0]:6s} {ids[1]}")

for sp, scen_id in common_ids:
    ds_o = xr.open_dataset(orig_by_ids[(sp, scen_id)])
    ds_a = xr.open_dataset(adj_by_ids[(sp, scen_id)])
    ds_c = xr.open_dataset(ceds_by_sp[sp])

    var = next(v for v in ds_o.data_vars if ds_o[v].dims == ("time", "level", "lat", "lon"))

    m_o = ds_to_monthly_emissions_total(ds_o, var, cell_area=areacella)
    m_a = ds_to_monthly_emissions_total(ds_a, var, cell_area=areacella)
    m_c = ds_to_monthly_emissions_total(ds_c, var, cell_area=areacella)

    # optionally restrict CEDS to the selected year window before averaging
    m_c_sel = m_c.sel(time=ceds_clim_years) if ceds_clim_years is not None else m_c
    ceds_label_years = (
        f"{ceds_clim_years.start}–{ceds_clim_years.stop}"
        if ceds_clim_years is not None
        else f"{int(ds_c['time.year'].min())}–{int(ds_c['time.year'].max())}"
    )

    # climatological mean: average each calendar month across all years in the dataset
    clim_o = m_o.groupby("time.month").mean()
    clim_a = m_a.groupby("time.month").mean()
    clim_c = m_c_sel.groupby("time.month").mean()

    months = clim_o["month"].values

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    # left: normalised seasonal shape
    ax1.plot(months, clim_c.values / clim_c.values.sum(), "k--o", lw=2,
             label=f"CEDS historical (mean {ceds_label_years})")
    ax1.plot(months, clim_o.values / clim_o.values.sum(), "C3-o",
             label=f"original {scen_id} (wrong seasonality)")
    ax1.plot(months, clim_a.values / clim_a.values.sum(), "C0-s",
             label=f"adjusted {scen_id} (CEDS{year_with_correct_data} seasonality)")
    ax1.set(xlabel="month", ylabel="fraction of annual total",
            title=f"{sp} — {scen_id}: seasonal shape (climatological mean)")
    ax1.set_xticks(months)
    ax1.set_ylim(bottom=0)
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    # right: absolute monthly totals
    ax2.plot(months, clim_c.values, "k--o", lw=2,
             label="CEDS historical mean")
    ax2.plot(months, clim_o.values, "C3-o",
             label=f"original {scen_id} mean")
    ax2.plot(months, clim_a.values, "C0-s",
             label=f"adjusted {scen_id} mean")
    ax2.set(xlabel="month", ylabel=f"{sp} aircraft emissions [Mt/month]",
            title=f"{sp} — {scen_id}: absolute monthly totals (climatological mean)")
    ax2.set_xticks(months)
    ax2.set_ylim(bottom=0)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.suptitle(
        f"Aircraft {sp} — {scen_id} — climatological monthly means\n"
        f"CEDS: {ceds_label_years}; "
        f"scenario: {int(ds_o['time.year'].min())}–{int(ds_o['time.year'].max())}",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(new_folder / f"diagnostic_monthly_means_{sp}_{scen_id}.png", dpi=150, bbox_inches="tight")
    plt.show()

    # time-series plot: monthly totals over the selected window, CEDS historical + orig vs adjusted
    ts_win = slice(ts_year_first, ts_year_last)
    ts_o = m_o.sel(time=ts_win)
    ts_a = m_a.sel(time=ts_win)
    ts_c = m_c.sel(time=ts_win)  # CEDS covers only its historical range; sel clips silently

    def _decimal_years(da):
        return da["time"].dt.year.values + (da["time"].dt.month.values - 0.5) / 12

    fig2, ax = plt.subplots(figsize=(13, 4))
    if ts_c.size:
        ax.plot(_decimal_years(ts_c), ts_c.values, "k-", lw=1.5, label="CEDS historical")
    ax.plot(_decimal_years(ts_o), ts_o.values, "C3-", lw=1, alpha=0.8,
            label=f"original {scen_id} (wrong seasonality)")
    ax.plot(_decimal_years(ts_a), ts_a.values, "C0-", lw=1, alpha=0.8,
            label=f"adjusted {scen_id} (CEDS{year_with_correct_data} seasonality)")
    ax.set_xlabel("year")
    ax.set_ylabel(f"{sp} aircraft emissions [Mt/month]")
    ax.set_title(f"Aircraft {sp} — {scen_id}: monthly time-series {ts_year_first}–{ts_year_last}")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(new_folder / f"diagnostic_timeseries_{sp}_{scen_id}.png", dpi=150, bbox_inches="tight")
    plt.show()

    ds_o.close()
    ds_a.close()
    ds_c.close()


# %%
