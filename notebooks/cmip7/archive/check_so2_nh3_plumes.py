# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # SO2 and NH3 plume time-series — v1-1-0
#
# Plots per-plume-region SO2 and NH3 emissions at specific lat/lon gridpoints
# for both the h (high) and vl (very-low) scenarios, using 1-1-0 ESGF files.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# %%
VERSION_ESGF = "1-1-0"
PATH_RESULTS = Path(
    "C:\\Users\\kikstra\\IIASA\\ECE.prog - Documents\\Projects\\CMIP7"
    "\\IAM Data Processing\\Shared emission fields data\\v1_1"
)

so2_h_file = (
    PATH_RESULTS
    / f"h_{VERSION_ESGF}"
    / f"SO2-em-anthro_input4MIPs_emissions_ScenarioMIP_IIASA-IAMC-h-{VERSION_ESGF}_gn_202201-210012.nc"
)
so2_vl_file = (
    PATH_RESULTS
    / f"vl_{VERSION_ESGF}"
    / f"SO2-em-anthro_input4MIPs_emissions_ScenarioMIP_IIASA-IAMC-vl-{VERSION_ESGF}_gn_202201-210012.nc"
)
nh3_h_file = (
    PATH_RESULTS
    / f"h_{VERSION_ESGF}"
    / f"NH3-em-anthro_input4MIPs_emissions_ScenarioMIP_IIASA-IAMC-h-{VERSION_ESGF}_gn_202201-210012.nc"
)
nh3_vl_file = (
    PATH_RESULTS
    / f"vl_{VERSION_ESGF}"
    / f"NH3-em-anthro_input4MIPs_emissions_ScenarioMIP_IIASA-IAMC-vl-{VERSION_ESGF}_gn_202201-210012.nc"
)

for f in [so2_h_file, so2_vl_file, nh3_h_file, nh3_vl_file]:
    if not f.exists():
        print(f"WARNING: file not found: {f}")

# %%
# Plume locations — structure mirrors workflow_cmip7-fast-track.py LOCATIONS dict
LOCATIONS = {
    "Europe":                   (50.08,   14.44), # prague
    "North America":            (39.77,  -86.16),  # indianapolis, -180/180 convention (277.5 - 360)
    "East Asia":                (30.59,  114.31), # wuhan
    "India":                    (21.15,   79.09), # nagpur
    "Northern Central Africa":  (15.60,   32.54), # Kharthoum
    "South America":            (-15.80, -47.89),  # brasilia, -180/180 convention (298.0 - 360)
    "South East Asia":               (-6.92,  107.62), # bandung
    "Southern Central Africa":  (-15.42,   28.27), # lusaka
    "Australia":                (-35.28, 149.1310), # canberra
}

# %%
# Load datasets
ds_so2_h  = xr.open_dataset(so2_h_file)
ds_so2_vl = xr.open_dataset(so2_vl_file)
ds_nh3_h  = xr.open_dataset(nh3_h_file)
ds_nh3_vl = xr.open_dataset(nh3_vl_file)

# %%
# Years to extract
years = np.array([
    2022, 2023, 2024, 2025, 2030, 2035, 2040, 2045, 2050,
    2055, 2060, 2065, 2070, 2075, 2080, 2085, 2090, 2095, 2100,
])


def extract_emissions(ds, var, lat, lon, years):
    """Sum over sectors, select nearest gridpoint, resample to annual sum."""
    da = ds[var].sum(dim="sector").sel(lat=lat, lon=lon, method="nearest")
    da_yearly = da.resample(time="1Y").sum(skipna=True)
    da_years = da_yearly["time"].dt.year.values
    idx = np.isin(da_years, years)
    return da_yearly.values[idx]


# %%
# Extract emissions for all locations
so2_h_loc  = {}
so2_vl_loc = {}
nh3_h_loc  = {}
nh3_vl_loc = {}

for place, (lat, lon) in LOCATIONS.items():
    so2_h_loc[place]  = extract_emissions(ds_so2_h,  "SO2_em_anthro", lat, lon, years)
    so2_vl_loc[place] = extract_emissions(ds_so2_vl, "SO2_em_anthro", lat, lon, years)
    nh3_h_loc[place]  = extract_emissions(ds_nh3_h,  "NH3_em_anthro", lat, lon, years)
    nh3_vl_loc[place] = extract_emissions(ds_nh3_vl, "NH3_em_anthro", lat, lon, years)

# %%
# Plot 3×3 grid
fig, axes = plt.subplots(3, 3, figsize=(18, 14), sharex=True)
axes = axes.flatten()

for i, (place, (lat, lon)) in enumerate(LOCATIONS.items()):
    ax = axes[i]

    ax.plot(years, so2_h_loc[place],  "o-",  color="tab:red",  label="SO2 h")
    ax.plot(years, so2_vl_loc[place], "x--", color="tab:red",  label="SO2 vl")
    ax.plot(years, nh3_h_loc[place],  "o-",  color="tab:blue", label="NH3 h")
    ax.plot(years, nh3_vl_loc[place], "x--", color="tab:blue", label="NH3 vl")

    ax.set_title(f"{place}\n({lat:.1f}°N, {lon:.1f}°E)", fontsize=10)
    ax.grid(True)
    if i % 3 == 0:
        ax.set_ylabel("Emissions [kg m$^{-2}$ s$^{-1}$]")
    if i >= 6:
        ax.set_xlabel("Year")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=10)
fig.suptitle(f"SO2 and NH3 emissions at plume locations — v{VERSION_ESGF}", fontsize=13, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.94])

out_dir = PATH_RESULTS / "plots_output"
out_dir.mkdir(parents=True, exist_ok=True)
out_file = out_dir / f"so2_nh3_plumes_v{VERSION_ESGF}.png"
plt.savefig(out_file, dpi=150, bbox_inches="tight")
print(f"Saved: {out_file}")
plt.show()

# %%
