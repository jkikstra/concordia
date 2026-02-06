# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: rescue_renewable
#     language: python
#     name: python3
# ---

# %%
from pathlib import Path

import numpy as np
import rioxarray
import yaml
from geoutils import Raster


# %%
with open("../config.yaml") as stream:
    config = yaml.safe_load(stream)

# %%
base_path = Path(config["base_path"]).expanduser()
co2_storage_potential = base_path / "gridding_process_files" / "co2_storage_potential"
out_path = base_path / "gridding_process_files" / "renewable_potential"
out_path.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# Used files are from Global Wind Atlas:
# - https://globalwindatlas.info/api/gis/global/IEC-class-fatigue-loads-incl-wake/
# - https://globalwindatlas.info/api/gis/global/capacity-factor_iec1/
# - https://globalwindatlas.info/api/gis/global/capacity-factor_iec2/
# - https://globalwindatlas.info/api/gis/global/capacity-factor_iec3/
#
# and the GlobalSolarAtlas:
# - [World_PVOUT_GISdata_LTAy_AvgDailyTotals_GlobalSolarAtlas-v2_GEOTIFF.zip](https://worldbank-atlas.s3.amazonaws.com/download/World/World_PVOUT_GISdata_LTAy_AvgDailyTotals_GlobalSolarAtlas-v2_GEOTIFF.zip?AWSAccessKeyId=ASIAS2HACIWTLW2RUUGX&Expires=1696456923&Signature=zsiiyJdwtol1x5HBCz6PBQnYEAY%3D&X-Amzn-Trace-Id=Root%3D1-651ddd57-74c8789978703ed93e7ab846%3BParent%3D73ab53be47ba27cc%3BSampled%3D0%3BLineage%3D3342b58b%3A0&x-amz-security-token=IQoJb3JpZ2luX2VjEEYaCWV1LXdlc3QtMSJIMEYCIQCR0F8j0vX2uMW2TXq%2BUAbISDkiWF1Y8emo5rBMDXwc8AIhAITmL0tlt4g899s4BaRaF%2BEN4rGznaT9GsUXNtiu7JqiKtUDCE8QBBoMMTkzNzQzNDM5MjcwIgwYj6JHhp4OiPdFNj8qsgOxWdoEKm7rKP5tJhHWWH1PXSm37FoEgSKCcDkLmtpMf9xqhgFqBpcfkI4G%2BGSTJqK1ZIXVUZwwOO2PMj3kUNX%2BviX4bsv57TkSCvztwAeFB%2BLVcHTcFmGkWtrFrD%2F6hjiJ4lIdVboMucC3GARG1cdlM3FJ%2B%2Bd93nae9kHa8D9BQD7an4u34FwD9qzd%2FNUgDXUbnpdA%2Fs%2BzySNipWirCIMKfxZWmsc0u%2Fe%2BC9tpceM41QlmBxlwaZzsvbbOZs4oiGcJl4WQ09CbnTI6s94kzpDFKxCMSjippQlCPNwe0J7QDCsyC6tRHgjUcDAvpyo4sP3KXEoYaKLZSfemW3orJ%2F8KGLkVgSknmtVe%2B4VssdB11sp37Qr0nb1Mr4%2B%2BQ2M5858gIT68z5Sf0mEbX90gNb6ysWvgLEW6NJkKP5iWGLA4voIKNpNBwrn0dq57jbrXZq9LCSRUWms0jN6rSgst1GTRrX6H49KPpNL7UggGKLU3aLDUY8Qp47YSNGieD6SbEfkiU7IiJwJoPkE0RI%2FZpOTIke0LfhD6l2cDEsAm6bp9Hhnr9edgkGsaAUMagn2gJltSCzDAuveoBjqdAYYlzW98hmWMb7MSpNLTG4Ak%2BA3A%2BF75XaWA23oYGsoHMK7qm6OQrudlRKUaOdqUcfHKW4Jwk1TLwpsD2wtbv1SM3YGPkQtqFu%2BsLzNnPyMfyKmZY433gLOtNX2W67TnE87N6RacgdMHKs3fH0zeABDj%2F7uEU494KBPoAanBvbOy%2FvFDIDPwH2f%2B0nDwQMtJIKSufl%2BWYjskjFXmRMU%3D) (extracted to sub-directory)
#

# %%
local_files = Path("/Users/coroa/projects/rescue/proxies/renewable_potential")

# %% [markdown]
# # Select iec classes per raster

# %%
reference = Raster(co2_storage_potential / "LOW_05.tif")

# %%
# From siting parameters database find the most often proposed fls class in each cell
gasp = Raster(local_files / "gasp_flsclasswake_100m.tif").reproject(
    reference, resampling="mode"
)

# %% [markdown]
# Map the detailed fls class to only the leading IEC1, IEC2 and IEC3 datasets
# ![Alt text](iec-class-mapping.png)

# %%
iec = [
    Raster(local_files / f"gwa3_250_capacityfactor_IEC{n}.tif").reproject(reference)
    for n in [1, 2, 3]
]


# %%
wind_cf = Raster.from_array(
    np.ma.choose(
        np.ma.masked_equal(np.digitize(gasp.data, [4, 8, 12]), 3),
        [ds.data for ds in iec],
    ),
    gasp.transform,
    gasp.crs,
)

# %%
pv_kWh_p_kWp_p_d = Raster(
    local_files
    / "./World_PVOUT_GISdata_LTAy_AvgDailyTotals_GlobalSolarAtlas-v2_GEOTIFF/PVOUT.tif"
).reproject(reference)
pv_cf = pv_kWh_p_kWp_p_d / 24  # convert to kW/kWp

# %% [markdown]
# Power densities
#
# | Solar     | Wind       |
# |-----------|------------|
# | 32 MWp/km2 | 3 MWp/km2  |
#
# from Appendix A in
#
# Maclaurin, Galen, Nick Grue, Anthony Lopez, Donna Heimiller, Michael Rossol, Grant Buster, and Travis Williams. 2019. The Renewable Energy Potential (reV) Model: A Geospatial Platform for Technical Potential and Supply Curve Modeling. Golden, CO: National Renewable Energy Laboratory. NREL/TP-6A20-73067. https://www.nrel.gov/docs/fy19osti/73067.pdf.

# %%
renewable_power_density = 32 * pv_cf + 3 * wind_cf

# %%
renewable_power_density.save(out_path / "renewable_potential.tiff")

# %%
rioxarray.open_rasterio(
    renewable_power_density.to_rio_dataset(), band_as_variable=True
).band_1.rename({"x": "lon", "y": "lat"}).rename("renewable_potential").assign_attrs(
    units="megawatt / squarekilometer"
).to_netcdf(
    out_path / "renewable_potential.nc"
)
