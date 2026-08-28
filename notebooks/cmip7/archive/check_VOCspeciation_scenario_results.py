# %% [markdown]
# # Check VOC speciation
# ## Tests performed:
# * Test: simple sum of the speciated files roughly adds up to the total VOC (openburning, anthro)
# ## Expected time for test:
# ~35 seconds per year per scenario per sector (on a laptop on battery).
# Meaning:
# Per scenario:
# Total: up to 4 minutes
# * up to 2 minutes for anthro (3 years)
# * up to 2 minutes for openburning (3 years)

# %%
# Do the necessary imports.

import xarray as xr
from pathlib import Path

from concordia.cmip7.CONSTANTS import GASES_ESGF_BB4CMIP_VOC, GASES_ESGF_CEDS_VOC
from concordia.settings import Settings

# %% editable=true slideshow={"slide_type": ""} tags=["parameters"]
SETTINGS_FILE: str = "config_cmip7_v0-4-0.yaml" # for second ESGF version

# Which scenario to run from the markers
# marker_to_run: str = "H" # options: H, HL, M, ML, L, LN, VL

GRIDDING_VERSION: str # will throw an error if not specified
# GRIDDING_VERSION: str = "prehandover_test_H"

FILE_NAME_ENDING: str # will throw an error if not specified
# FILE_NAME_ENDING: str = "input4MIPs_emissions_CMIP6plus_IIASA-IAMC-scen-h-0-4-0_gn_202201-210012.nc"

WHICH_SPECIATION: list[str] = ['anthro', 'openburning']
# WHICH_SPECIATION: list[str] = ['openburning']

WHICH_YEARS: list[str] = ["2023","2050","2100"]
# WHICH_YEARS: list[str] = ["2100"]

# %%
file_map = {"anthro": {
    'file-var': 'VOC-em-anthro',
    'var': 'VOC_em_anthro',
    'species': GASES_ESGF_CEDS_VOC
},
"openburning": {
    'file-var': 'VOC-em-openburning',
    'var': 'VOC_em_openburning',
    'species': GASES_ESGF_BB4CMIP_VOC
}
}


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



# %% [markdown]
# # Testing global totals (simple sum)
# Doing a simple sum becauase it is quicker

# %%
percentage_diff_allowed = 0.01 # note: doing the simple sum below misses information on (a) grid cell area, and (b) month lenght. Therefore, it is expected that this will not line up perfectly.

# %%
# loop over options
for sector in WHICH_SPECIATION:
    total = xr.open_dataset(settings.out_path / GRIDDING_VERSION / f"{file_map[sector]['file-var']}_{FILE_NAME_ENDING}")
    for t in WHICH_YEARS:
        print(f"Testing {sector}, for {FILE_NAME_ENDING}, year: {t}")

        total_t = total.sel(time=t).sum(dim=["lat","lon","time"])[file_map[sector]['var']].values

        parts = None
        for g in file_map[sector]['species']:
            ds = xr.open_dataset(settings.out_path / GRIDDING_VERSION / f"{g}_{FILE_NAME_ENDING}")
            da = ds.sel(time=t).sum(dim=["lat","lon","time"])
            da_vals = da[f'{g}'].values
            if parts is None:
                parts = da_vals
            else:
                parts = parts + da_vals

        # Allow 0.01% tolerance for floating point differences
        tolerance = percentage_diff_allowed / 100 * abs(total_t)
        assert (abs(total_t - parts) <= tolerance).all(), f"Difference: {abs(total_t - parts)}, Tolerance: {percentage_diff_allowed}%"


