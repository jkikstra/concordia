# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import xarray as xr
from pathlib import Path

import matplotlib.pyplot as plt
import cartopy.crs as ccrs

# %%
from concordia.settings import Settings
from concordia.cmip7.CONSTANTS import CONFIG

from concordia.cmip7.utils_plotting import ds_to_annual_emissions_total, plot_map

# %%
def get_settings(base_path: Path,
                 file = CONFIG):
    settings = Settings.from_config(
        file, 
        base_path=base_path,
        version=None
    )
    return settings

try:
    # when running the script from a terminal or otherwise
    notebook_dir = Path(__file__).resolve().parent
    settings = get_settings(base_path=notebook_dir)
except (FileNotFoundError, NameError):
    try:
        # when running the script from a terminal or otherwise
        notebook_dir = Path(__file__).resolve().parent.parent
        settings = get_settings(base_path=notebook_dir)
    except (FileNotFoundError, NameError):
        # Fallback for interactive/Jupyter mode, where 'file location' does not exist
        notebook_dir = Path().resolve().parent  # one up
        settings = get_settings(base_path=notebook_dir)

# %%
YEAR_CURRENT = 2023
YEAR_FUTURE = 2050

# %% 
path_fiedler = Path("C:/Users/kikstra/OneDrive - IIASA/_Other/Co-authored papers and reports/2025/fiedler_aerchemmip/submission/jsk-figures/data/")
folder_h = path_fiedler / "aerchemmip_pre0-4-0_H"
folder_vl = path_fiedler / "aerchemmip_pre0-4-0_VL"

# %%
areacella = xr.open_dataset(Path(settings.gridding_path, 
                                 "areacella_input4MIPs_emissions_CMIP_CEDS-CMIP-2025-04-18_gn.nc"))
cell_area = areacella["areacella"]

# H
# %%
# ======
# ANTHRO
# ======

# H 
# ------
# Load and process the data
anthro_h_raw = xr.open_dataset(
    folder_h / "Sulfur-em-anthro_input4MIPs_emissions_CMIP6plus_IIASA-GCAM-8s-SSP3---High-Emissions_gn_202301-210012.nc"
)
# Select the target years (using datetime selection)
anthro_h_subset = anthro_h_raw.sel(
    time=anthro_h_raw.time.dt.year.isin([YEAR_CURRENT, YEAR_FUTURE])
)
anthro_h_da = ds_to_annual_emissions_total(anthro_h_subset,
                                  "Sulfur_em_anthro",
                                  cell_area,
                                  keep_sectors=False,
                                  sum_dims=None)

anthro_h_da_aggregate = ds_to_annual_emissions_total(anthro_h_subset,
                                  "Sulfur_em_anthro",
                                  cell_area,
                                  keep_sectors=False,
                                  sum_dims=["lat", "lon"])

# VL
# ------
# Load and process the data
anthro_vl_raw = xr.open_dataset(
    folder_vl / "Sulfur-em-anthro_input4MIPs_emissions_CMIP6plus_IIASA-REMIND-MAgPIE-3.5-4.11-SSP1---Very-Low-Emissions_gn_202301-210012.nc"
)
# Select the target years (using datetime selection)
anthro_vl_subset = anthro_vl_raw.sel(
    time=anthro_vl_raw.time.dt.year.isin([YEAR_CURRENT, YEAR_FUTURE])
)
anthro_vl_da = ds_to_annual_emissions_total(anthro_vl_subset,
                                  "Sulfur_em_anthro",
                                  cell_area,
                                  keep_sectors=False,
                                  sum_dims=None)

# %%
# ======
# OPENBURNING
# ======

# H 
# ------
# Load and process the data
openburning_h_raw = xr.open_dataset(
    folder_h / "Sulfur-em-openburning_input4MIPs_emissions_CMIP6plus_IIASA-GCAM-8s-SSP3---High-Emissions_gn_202301-210012.nc"
)
# Select the target years (using datetime selection)
openburning_h_subset = openburning_h_raw.sel(
    time=openburning_h_raw.time.dt.year.isin([YEAR_CURRENT, YEAR_FUTURE])
)
openburning_h_da = ds_to_annual_emissions_total(openburning_h_subset,
                                  "Sulfur_em_openburning",
                                  cell_area,
                                  keep_sectors=False,
                                  sum_dims=None)

openburning_h_da_aggregate = ds_to_annual_emissions_total(openburning_h_subset,
                                  "Sulfur_em_openburning",
                                  cell_area,
                                  keep_sectors=False,
                                  sum_dims=["lat", "lon"])

# VL
# ------
# Load and process the data
openburning_vl_raw = xr.open_dataset(
    folder_vl / "Sulfur-em-openburning_input4MIPs_emissions_CMIP6plus_IIASA-REMIND-MAgPIE-3.5-4.11-SSP1---Very-Low-Emissions_gn_202301-210012.nc"
)
# Select the target years (using datetime selection)
openburning_vl_subset = openburning_vl_raw.sel(
    time=openburning_vl_raw.time.dt.year.isin([YEAR_CURRENT, YEAR_FUTURE])
)
openburning_vl_da = ds_to_annual_emissions_total(openburning_vl_subset,
                                  "Sulfur_em_openburning",
                                  cell_area,
                                  keep_sectors=False,
                                  sum_dims=None)


# %%
# ======
# AIR
# ======

# H 
# ------
# Load and process the data
air_h_raw = xr.open_dataset(
    folder_h / "Sulfur-em-AIR-anthro_input4MIPs_emissions_CMIP6plus_IIASA-GCAM-8s-SSP3---High-Emissions_gn_202301-210012.nc"
)
# Select the target years (using datetime selection)
air_h_subset = air_h_raw.sel(
    time=air_h_raw.time.dt.year.isin([YEAR_CURRENT, YEAR_FUTURE])
)
air_h_da = ds_to_annual_emissions_total(air_h_subset,
                                  "Sulfur_em_AIR_anthro",
                                  cell_area,
                                  keep_sectors=False,
                                  sum_dims=[])

# VL
# ------
# Load and process the data
air_vl_raw = xr.open_dataset(
    folder_vl / "Sulfur-em-AIR-anthro_input4MIPs_emissions_CMIP6plus_IIASA-REMIND-MAgPIE-3.5-4.11-SSP1---Very-Low-Emissions_gn_202301-210012.nc"
)
# Select the target years (using datetime selection)
air_vl_subset = air_vl_raw.sel(
    time=air_vl_raw.time.dt.year.isin([YEAR_CURRENT, YEAR_FUTURE])
)
air_vl_da = ds_to_annual_emissions_total(air_vl_subset,
                                  "Sulfur_em_AIR_anthro",
                                  cell_area,
                                  keep_sectors=False,
                                  sum_dims=[])


# %%
# Calculate shared colorbar range (0 to 99th percentile of all data)
all_data_arrays = [
    openburning_vl_da.sel(year=YEAR_CURRENT) + anthro_vl_da.sel(year=YEAR_CURRENT) + air_vl_da.sel(year=YEAR_CURRENT),
    openburning_vl_da.sel(year=YEAR_FUTURE) + anthro_vl_da.sel(year=YEAR_FUTURE) + air_vl_da.sel(year=YEAR_FUTURE),
    openburning_h_da.sel(year=YEAR_FUTURE) + anthro_h_da.sel(year=YEAR_FUTURE) + air_h_da.sel(year=YEAR_FUTURE),
    # anthro_vl_da.sel(year=YEAR_CURRENT),
    # openburning_h_da.sel(year=YEAR_CURRENT),
    # openburning_vl_da.sel(year=YEAR_CURRENT) + anthro_vl_da.sel(year=YEAR_CURRENT),
    # openburning_vl_da.sel(year=YEAR_CURRENT) + anthro_vl_da.sel(year=YEAR_CURRENT) + air_vl_da.sel(year=YEAR_CURRENT)
]

# Combine all data to find global range
combined_data = xr.concat(all_data_arrays, dim='temp')
vmin_shared = 0  # Start from zero
vmax_shared = float(combined_data.quantile(0.99))

print(f"Shared colorbar range: {vmin_shared:.2f} to {vmax_shared:.2f} Mt/year")

# %%
# Create multipanel plot
fig, axes = plt.subplots(
    1, 3, 
    figsize=(18, 6),
    subplot_kw={"projection": ccrs.Robinson()}
)

# Data for each panel
data_panels = [
    {
        'da': openburning_vl_da.sel(year=YEAR_CURRENT) + anthro_vl_da.sel(year=YEAR_CURRENT) + air_vl_da.sel(year=YEAR_CURRENT),
        'title': f"{YEAR_CURRENT} (vllo and h are the same)",
        'cmap': 'GnBu',
        'vmin': vmin_shared,
        'vmax': vmax_shared
    },
    {
        'da': openburning_vl_da.sel(year=YEAR_FUTURE) + anthro_vl_da.sel(year=YEAR_FUTURE) + air_vl_da.sel(year=YEAR_FUTURE),
        'title': f"{YEAR_FUTURE}, vllo",
        'cmap': 'GnBu',
        'vmin': vmin_shared,
        'vmax': vmax_shared
    },
    {
        'da': (
            openburning_h_da.sel(year=YEAR_FUTURE) + anthro_h_da.sel(year=YEAR_FUTURE) + air_h_da.sel(year=YEAR_FUTURE)
        ) - (
            openburning_vl_da.sel(year=YEAR_FUTURE) + anthro_vl_da.sel(year=YEAR_FUTURE) + air_vl_da.sel(year=YEAR_FUTURE)
        ),
        'title': f"{YEAR_FUTURE}, h minus vllo",
        'cmap': 'RdBu_r',
        'vmin': -vmax_shared,
        'vmax': vmax_shared
    }
]

# Plot each panel
for i, panel in enumerate(data_panels):
    ax = axes[i]
    
    # Set up the map
    ax.set_global()
    ax.coastlines()
    
    # Plot the data
    im = panel['da'].plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap=panel['cmap'],
        vmin=panel['vmin'],
        vmax=panel['vmax'],
        add_colorbar=False,  # We'll add colorbars separately
        robust=False  # Use explicit vmin/vmax instead of robust
    )
    
    ax.set_title(panel['title'], fontsize=12, pad=10)
    
    # Add colorbar for each panel
    cbar = plt.colorbar(
        im, 
        ax=ax, 
        orientation='horizontal', 
        shrink=0.8, 
        pad=0.05,
        aspect=30
    )
    cbar.set_label('kg/m2/year', fontsize=10)

# Add overall title
fig.suptitle('Illustrative Gridded SO2 Emissions', fontsize=16, y=0.8)

# Adjust layout
plt.tight_layout()

# Save the figure
output_base = path_fiedler / ".." / f"so2_emissions_comparison_{YEAR_CURRENT}_{YEAR_FUTURE}_v1"
plt.savefig(f"{output_base}.pdf", bbox_inches='tight', dpi=300)
plt.savefig(f"{output_base}.png", bbox_inches='tight', dpi=300)

print(f"Figures saved as:")
print(f"  {output_base}.pdf")
print(f"  {output_base}.png")

plt.show()


# # 2050, h
# plot_map(
#     da = openburning_h_da.sel(year=YEAR_FUTURE) + anthro_h_da.sel(year=YEAR_FUTURE) + air_h_da.sel(year=YEAR_FUTURE),
#     title= f"{YEAR_FUTURE}, h",
#     vmin=vmin_shared,
#     vmax=vmax_shared
# )

# plot_map(
#     da = (
#         openburning_h_da.sel(year=YEAR_CURRENT)
#     ),
#     title= f"{YEAR_CURRENT}, h  (openburning)",
#     cmap = "RdBu_r"
# )

# # 2050, h minus vllo (openburning)
# plot_map(
#     da = (
#         openburning_h_da.sel(year=YEAR_FUTURE)
#     ) - (
#         openburning_vl_da.sel(year=YEAR_FUTURE)
#     ),
#     title= f"{YEAR_FUTURE}, h minus vllo (openburning)",
#     cmap = "RdBu_r"
# )

# plot_map(
#     da = anthro_vl_da.sel(year=YEAR_CURRENT),
#     title= f"vllo: {YEAR_CURRENT}, anthropogenic",
#     vmin=vmin_shared,
#     vmax=vmax_shared
# )

# plot_map(
#     da = openburning_h_da.sel(year=YEAR_CURRENT),
#     title= f"h: {YEAR_CURRENT}, open burning",
#     vmin=vmin_shared,
#     vmax=vmax_shared
# )

# plot_map(
#     da = openburning_vl_da.sel(year=YEAR_CURRENT) + anthro_vl_da.sel(year=YEAR_CURRENT),
#     title= f"vl: {YEAR_CURRENT}, open burning + anthropogenic",
#     vmin=vmin_shared,
#     vmax=vmax_shared
# )

# plot_map(
#     da = openburning_vl_da.sel(year=YEAR_CURRENT) + anthro_vl_da.sel(year=YEAR_CURRENT) + air_vl_da.sel(year=YEAR_CURRENT),
#     title= f"vl: {YEAR_CURRENT}, open burning + anthropogenic + air",
#     vmin=vmin_shared,
#     vmax=vmax_shared
# )




# %%
