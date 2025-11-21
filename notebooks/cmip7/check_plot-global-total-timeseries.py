# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

from concordia.cmip7.utils_plotting import ds_to_annual_emissions_total


# %%
# load cell area
# areacella = xr.open_dataset(Path("C:/Users/kikstra/IIASA/ECE.prog - Documents/Projects/CMIP7/IAM Data Processing/concordia_cmip7_v0-4-0/input/gridding", 
#                                  "areacella_input4MIPs_emissions_CMIP_CEDS-CMIP-2025-04-18_gn.nc"))
# cell_area = areacella["areacella"]

# %% editable=true slideshow={"slide_type": ""} tags=["parameters"]
marker: str = "H"

species: list = [
    "SO2",
    "NH3"
]
species = ["BC", "CH4", "CO", "CO2", "N2O", "NH3", "NOx", "OC", "SO2", "NMVOC", "NMVOCbulk"]

filetypes: list = [
    "em-AIR-anthro",
    "em-anthro",
    "em-openburning"
]

# TODO:
# - [ ] replace paths based on Settings/config for defaults

# output path for plots
path_output_plot_results: Path = Path("C:/Users/kikstra/Documents/GitHub/concordia/results/plots")


# scenario data
# path_results: Path = Path("C:/Users/kikstra/IIASA/ECE.prog - Documents/Projects/CMIP7/IAM Data Processing/concordia_cmip7_esgf_v0_alpha/output/cmip7_esgf_v0_alpha_h/final")
# path_results: Path = Path("D:/concordia-results/temp_v03_fix/v0_3_files-rewrite")
# path_results: Path = Path("C:/Users/kikstra/Downloads/cmip7-uploaded/esgf")
path_results: Path = Path("C:/Users/kikstra/Documents/GitHub/concordia/results/prehandover_test_VL")


# ESGF history files:
path_history_ceds_anthro: Path = Path("C:/Users/kikstra/IIASA/ECE.prog - Documents/Projects/CMIP7/IAM Data Processing/concordia_cmip7_v0-4-0/input/gridding/esgf/ceds/CMIP7_anthro")
path_history_ceds_anthro_air: Path = Path("C:/Users/kikstra/IIASA/ECE.prog - Documents/Projects/CMIP7/IAM Data Processing/concordia_cmip7_v0-4-0/input/gridding/esgf/ceds/CMIP7_AIR")
path_history_ceds_anthro_voc: Path = Path("C:/Users/kikstra/IIASA/ECE.prog - Documents/Projects/CMIP7/IAM Data Processing/concordia_cmip7_v0-4-0/input/gridding/esgf/ceds/CMIP7_anthro_VOC")

path_history_bb4cmip: Path = Path("C:/Users/kikstra/IIASA/ECE.prog - Documents/Projects/CMIP7/IAM Data Processing/concordia_cmip7_v0-4-0/input/gridding/esgf/ceds/bb4cmip7")
path_history_bb4cmip_voc: Path = Path("C:/Users/kikstra/IIASA/ECE.prog - Documents/Projects/CMIP7/IAM Data Processing/concordia_cmip7_v0-4-0/input/gridding/esgf/ceds/bb4cmip7")

# history (harmonisation input):
path_history: Path = Path("C:/Users/kikstra/IIASA/ECE.prog - Documents/Projects/CMIP7/IAM Data Processing/concordia_cmip7_v0-4-0/input/historical/cmip7_history_countrylevel_250918.csv")

# history (from concordia downscaling output)
path_history_downscaled_results: Path = Path("")


# %%
# loop over species
for sp in species:
    print(f"Processing species: {sp}")
    for ft in filetypes:
        print(f"Processing file type: {ft}")
        
        # example (TO BE DELETED)
        # sp = "SO2"
        # ft = "em-anthro"

        # ==================
        # (1) SCENARIO data
        # ==================

        # load file
        pattern = f"{sp}-{ft}*"
        scenario_file = list(path_results.glob(pattern))
        
        # check if scenario_file is an empty list
        if not scenario_file:
            print(f"  No files found matching pattern '{pattern}' in {path_results}")
            print(f"  Skipping {sp}-{ft} combination...")
            continue
        
        print(f"  Found file: {scenario_file[0].name}")
        scenario_file

        scen = xr.open_dataset(scenario_file[0])

        # # read with netCDF4
        # import netCDF4
        # from netCDF4 import num2date
        # ds = netCDF4.Dataset(scenario_file[0])
        # time_var = ds.variables['time']
        # # Convert time values to readable dates
        # time_values = num2date(time_var[:], time_var.units, 
        #                       calendar=getattr(time_var, 'calendar', 'standard'))
        # ds.close() # Close the netCDF4 dataset

    
        # ==================
        # (2) HISTORY data
        # ==================
        
        # 
    
        # ==================
        # (3) COMBINING data
        # ==================
        
        # 
    
        # ==================
        # (4) PLOTTING data
        # ==================

        # Create figure for this species and filetype combination
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot scenario timeseries using annual data
        # Get the data variable (assumes first data variable is emissions)
        data_var_name = f"{sp}_{ft.replace("-", "_")}"

        # convert to global annual totals
        annual_emissions = ds_to_annual_emissions_total(
            gridded_data=scen,
            var_name=data_var_name,
            # cell_area=cell_area,
            keep_sectors=False
        )
        
        # Plot the timeseries
        annual_emissions.plot(
            ax=ax,
            marker='o',
            linewidth=2,
            markersize=4,
            label=f'Scenario ({ft})'
        )
        
        # Customize the plot
        ax.set_title(f'{sp} Global Annual Emissions Timeseries - {ft}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel(f'{sp} Emissions (Mt/yr)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Format y-axis with scientific notation if values are large
        if annual_emissions.max() > 1e6:
            ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        
        # Tight layout and show
        plt.tight_layout()
        plt.show()
        
        # Print some statistics
        print(f"\n{sp} {ft} Statistics:")
        print(f"Year range: {int(annual_emissions.year.min())} - {int(annual_emissions.year.max())}")
        print(f"Annual emissions range: {float(annual_emissions.min()):.2e} - {float(annual_emissions.max()):.2e}")
        print(f"Mean annual emissions: {float(annual_emissions.mean()):.2e}")
        print(f"Total years: {len(annual_emissions.year)}")
        
        print("\n" + "="*50 + "\n")  # Separator between plots

        # save the plot as png
        # Create filename with species and filetype
        filename = f"{sp}_{ft}_global_annual_timeseries_{marker}.png"
        output_file = path_output_plot_results / filename
        
        # Ensure output directory exists
        path_output_plot_results.mkdir(parents=True, exist_ok=True)
        
        # Save the figure
        fig.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Plot saved to: {output_file}")
        
        # Close the figure to free memory
        plt.close(fig)
# %%
