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
import dask
import dask.array
import numpy as np
import pandas as pd
import ptolemy as pt
import pyogrio as pio
import pyreadr
import xarray as xr
from pathlib import Path
import os

# %% [markdown]
# # Plot all .Rd files from CEDS that we use

# %%
base_gridding = Path("C:/Users/kikstra/IIASA/ECE.prog - Documents/Projects/CMIP7/IAM Data Processing/concordia_cmip7_v0_2/input/gridding")
example_file_backup = base_gridding / "ceds_input/population_proxy/population_2015.Rd"
example_file = base_gridding / "ceds_input/non-point_source_proxy_final_sector/CO_2022_WST.Rd"
plot_path = base_gridding / "ceds_input/plots"
all_files = pd.read_csv(plot_path / "df_files.csv")
files_to_visualise = all_files['file_year'].unique()

# %%
template_file = (
    base_gridding
    / "example_files" / "GCAM4-SSP4-34-SPA4-V25-Harmonized-DB-Sulfur-em-aircraft-anthro_input4MIPs_emissions_CMIP_IAMC-V1_gn_201501-210012.nc"
)
template = xr.open_dataset(template_file)


# %%
def read_r_variable(file):
    file = Path(file)
    print(f"Reading in {file}\n")

    result = pyreadr.read_r(file)

    # If there's more than one, you can decide how to handle it
    if len(result) > 1:
        print(f"Warning: More than one variable found in {file.name}, using the first one.")

    # Get the first variable's name and value
    old_var_name, value = next(iter(result.items()))

    # Rename the variable to the filename
    if old_var_name != file.stem:
        print(f"Renaming variable '{old_var_name}' to '{file.stem}'")
        value.name = file.stem


    return np.asarray(value, dtype="float32")[::-1]


# %%
def read_r_to_da(file, template, flipud=True, dtype="float32"):
    """
    Read an R .rds/.RData variable and return an xarray.DataArray
    on the same grid as `template` (expects template to have lat/lon coords).
    """
    res = pyreadr.read_r(str(file))
    _, value = next(iter(res.items()))           # take first object
    arr = np.asarray(value, dtype=dtype)
    if flipud:                                   # typical R->Python row order fix
        arr = arr[::-1, :]

    da = xr.DataArray(
        arr,
        coords={"lat": template["lat"].values, "lon": template["lon"].values},
        dims=("lat", "lon"),
        name=Path(file).stem,
    )
    return da



# %%
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import os

def plot_map(fname, ncols=1, proj=ccrs.Robinson(), save_path=None, dpi=300, show=True, template=template,
             name="unit",
             add_borders: bool = True,
             add_coastlines: bool = True):

    coords = {"lat": template.lat, "lon": template.lon}
    if "AIR" in os.path.basename(fname):
        coords["level"] = template.level
    a = read_r_variable(fname)
    da = xr.DataArray(a, coords=coords, name=name)

    
    
    if "AIR" in os.path.basename(fname):
        da.sel(level="")
        # not implemented right now
    
    nrows = 1
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(6 * ncols, 4.5 * nrows),
        subplot_kw={"projection": proj}
    )

    # Flatten axes safely
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    ax = axes[0]

    # Plot directly with xarray's .plot.pcolormesh
    da.plot.pcolormesh(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap="GnBu",
        robust=True,
        cbar_kwargs={"orientation": "horizontal", "shrink": 0.65},
    )

    if add_coastlines:
        ax.coastlines()
    if add_borders:
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor="black")

    
    ax.set_title(os.path.basename(fname)) # -> e.g. 'population_2015.Rd'
    ax.coastlines()

    plt.tight_layout()

    # --- Save if a path is provided ---
    if save_path is not None:
        save_path = Path(save_path)
        print(f'Save to {save_path}')
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())

    if show:
        plt.show()
    else:
        plt.close(fig)

    

# %%
# plot_map(example_file_backup)

# %%
os.path.basename(f)

# %%
for f in files_to_visualise:

    if "AIR" in os.path.basename(f):
        continue
    else:
        a = plot_map(f, 
                     # save_path=Path(f).with_suffix(".png"),
                     save_path=(Path(plot_path) / os.path.basename(Path(f))).with_suffix(".png"),
                     show=True)

# %% [markdown]
# # Combine proxy and backup proxy at the national level
# * especially for Waste (WST)
# * but also for other intermediary-sector-files

# %%
# pop_da = read_r_variable(example_file_backup)
# pop_da
pop_da = read_r_to_da(example_file_backup,template)
pop_da

# %%
# emis_da = read_r_variable(example_file)
# emis_da
emis_da = read_r_to_da(example_file, template)
emis_da

# %%
mask_v2 =  xr.open_dataset(base_gridding / "ssp_comb_indexraster.nc")
mask_v2

# %%
print(float(country_grid.min()), float(country_grid.max()))  # ~0 .. ~1.00036

# %%
mask =  xr.open_dataset(base_gridding / "ssp_comb_countrymask.nc").__xarray_dataarray_variable__
mask.sel(iso=['pak']).plot()

# %%
mask

# %%
xr.align(pop_da, emis_da, mask, join="exact")

# %%
# read mapping
d = pd.read_csv(
    base_gridding / "ceds_input" / "countries_using_population_backup" / "CO_proxy_substitution_mapping.csv"
).drop(columns=["sub_flag"]).rename(columns={"iso": "iso3"})
d["source"] = "pop"
d

# %%
# filter mapping: only selected gas & sector
dfil = d[(d['em']=='CO')&(d['sector']=='WST')]
# filter mapping: remove USA states
dfil = dfil[~dfil["iso3"].str.startswith("usa_")]
dfil.iso3.unique()

# %%
{
        str(row.iso3).upper(): str(row.source).lower()
        for _, row in dfil.iterrows()
        if str(row.source).lower() in {"pop", "emis"}
    }

# %%
# import pandas as pd
# import numpy as np
# import xarray as xr


# # Note:
# # template must be a DataArray (or Dataset) that already has lat/lon coords & dims
# # e.g., template = xr.DataArray(np.empty((nlat,nlon)), coords={"lat": lat, "lon": lon}, dims=("lat","lon"))

# def combine_by_country_v1(
#     pop_da: xr.DataArray,
#     emis_da: xr.DataArray,
#     country_grid: xr.DataArray,
#     # mapping_csv: str,
#     mapping_df: pd.DataFrame,
#     default_source: str = "emis",  # fallback if a country isn't in the CSV
# ) -> xr.DataArray:
#     """
#     Combine two rasters (population density vs emissions flux) into one,
#     choosing per-country which source to use based on a CSV mapping.

#     Parameters
#     ----------
#     pop_da : xr.DataArray
#         Population density grid (2D or 3D with a single time).
#     emis_da : xr.DataArray
#         Emissions flux grid (same grid as pop_da).
#     country_grid : xr.DataArray of dtype str/object/int
#         ISO3 code per pixel (same grid as pop_da/emis_da).
#     mapping_csv : str
#         Path to CSV with columns: iso3, source (values 'pop' or 'emis').
#     default_source : str
#         Which source to use for countries not listed in mapping ('pop' or 'emis').

#     Returns
#     -------
#     xr.DataArray
#         Combined grid where each country's pixels come from the chosen source.
#     """

#     # --- 1) Read mapping and standardize ---
#     # df = pd.read_csv(mapping_csv)
#     df = mapping_df
#     if not {"iso3", "source"}.issubset(df.columns):
#         raise ValueError("CSV must have columns: 'iso3' and 'source'")

#     valid_sources = {"pop", "emis"}
#     choice_map = {
#         str(row.iso3).upper(): str(row.source).lower()
#         for _, row in df.iterrows()
#         if str(row.source).lower() in valid_sources
#     }
#     default_source = default_source.lower()
#     if default_source not in valid_sources:
#         raise ValueError("default_source must be 'pop' or 'emis'")

#     # --- 2) Align all inputs on the exact same grid (dims + coords) ---
#     # If they don't match, consider regridding/resampling upstream.
#     pop_da, emis_da, country_grid = xr.align(pop_da, emis_da, country_grid, join="exact") # require coordinates to match exactly (raise error otherwise).

#     # --- 3) Build a boolean mask: True where we should use pop_da ---
#     def choose_pop(code: object) -> bool:
#         if code is None or (isinstance(code, float) and np.isnan(code)):
#             # No country: use default source
#             return default_source == "pop"
#         key = str(code).upper()
#         src = choice_map.get(key, default_source)  # csv or fallback
#         return src == "pop"

#     # vectorize for xarray/dask
#     mask_pop = xr.apply_ufunc(
#         np.vectorize(choose_pop, otypes=[bool]),
#         country_grid,
#         dask="parallelized",
#         output_dtypes=[bool]
#     )

#     # --- 4) Combine ---
#     combined = xr.where(mask_pop, pop_da, emis_da)

#     # Carry over attrs/name nicely
#     combined.name = combined.name or "combined" # TODO: change attribute name
#     combined.attrs.update({
#         "description": "Per-country combination of population and emissions grids",
#         "rule": "If mapping says 'pop' for ISO3 use population grid; else use emissions grid",
#         "default_source": default_source,
#     })

#     return combined


# %%
import pandas as pd
import numpy as np
import xarray as xr
from typing import Callable, Dict, Tuple


# ---------- 0) Helpers: mapping validation / normalization ----------
def validate_and_normalize_mapping(df: pd.DataFrame, fallback_source: str) -> Tuple[Dict[str, str], str]:
    """
    Return (choice_map, fallback_source), where choice_map maps iso3 -> 'pop'/'emis'.
    Both keys and values are normalized to lowercase.
    """
    if not {"iso3", "source"}.issubset(df.columns):
        raise ValueError("mapping_df must have columns: 'iso3' and 'source'")

    valid_sources = {"pop", "emis"}
    fallback_source = str(fallback_source).lower()
    if fallback_source not in valid_sources:
        raise ValueError("fallback_source must be 'pop' or 'emis'")

    choice_map = {
        str(row.iso3).strip().lower(): str(row.source).strip().lower()
        for _, row in df.iterrows()
        if str(row.source).strip().lower() in valid_sources
    }
    return choice_map, fallback_source


# ---------- 1) Align inputs ----------
def align_grids(pop_da: xr.DataArray, emis_da: xr.DataArray, country_grid: xr.DataArray) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """
    Require exact coordinate match; if they don't match, reproject / regrid upstream.
    """
    return xr.align(pop_da, emis_da, country_grid, join="exact")


# ---------- 3A) FAST mask path (set membership) ----------
def build_mask_fast(country_grid: xr.DataArray, choice_map: Dict[str, str], fallback_source: str) -> xr.DataArray:
    """
    True where the country in iso_grid maps to `source` in choice_map.
    Cells whose ISO is missing or not in the mapping are set according to `fallback_source`.
    """
    pop_set = {iso.lower() for iso, src in choice_map.items() if src == fallback_source}

    # True where ISO is explicitly in the 'pop' set
    # mask = country_grid.astype("object").isin(pop_set) # slower, generic type
    mask = country_grid.astype("U").isin(pop_set) # faster, unicode type

    # Handle missing/NaN country codes via the default
    mask = xr.where(country_grid.isnull(), True, mask) # instead of this, check if value is >0 and then do True, otherwise False

    mask.name = f"mask_{fallback_source}"
    return mask


# ---------- 4) Combine ----------
def combine_with_mask(pop_da: xr.DataArray, emis_da: xr.DataArray, mask_pop: xr.DataArray, default_source: str) -> xr.DataArray:
    """
    Use mask to select values per cell.
    """
    combined = xr.where(mask_pop, pop_da, emis_da) # instead of this; do sum
    combined.name = combined.name or "combined"
    combined.attrs.update({
        "description": "Per-country combination of population and emissions grids",
        "rule": "If mapping says 'pop' for ISO3 use population grid; else use emissions grid",
        "default_source": default_source,
    })
    return combined


# ---------- 5) Orchestrator ----------
def combine_by_country_v2(
    pop_da: xr.DataArray,
    emis_da: xr.DataArray,
    country_grid: xr.DataArray,
    mapping_df: pd.DataFrame,
    default_source: str = "emis",
    use_fast_mask: bool = True,   # prefer fast 'isin' path by default
) -> xr.DataArray:
    """
    Main function: produce a combined grid using per-country selection.
    """
    choice_map, default_source = validate_and_normalize_mapping(mapping_df, default_source)
    pop_da, emis_da, country_grid = align_grids(pop_da, emis_da, country_grid)

    mask_pop = build_mask_fast(country_grid, choice_map, default_source)
    
    return combine_with_mask(pop_da, emis_da, mask_pop, default_source)



# %%
# mask_pop = combine_by_country(
#     pop_da = pop_da, #: xr.DataArray,
#     emis_da = emis_da, #: xr.DataArray,
#     country_grid = mask, # xr.DataArray,
#     mapping_df = dfil, #: str,
#     default_source = "emis" #: str = "emis",  # fallback if a country isn't in the CSV
# )

# %%
# mask_pop.sel(iso=['ind']).plot()

# %%
# country_grid > 0
# country_grid.astype("U").isin(pop_set)
print(country_grid)

# %%
# validate_and_normalize_mapping(dfil, "emis") # works as expected
# align_grids(pop_da, emis_da, mask) # works as expected
choice_map, fallback_source = validate_and_normalize_mapping(dfil, "pop")
pop_da, emis_da, country_grid = align_grids(pop_da, emis_da, mask)
# m = build_mask_fast(country_grid, choice_map, fallback_source) # check if works; currently not, only 0 instead of 1. So we dive in
# # when building mask:

pop_set = {iso.lower() for iso, src in choice_map.items() if src == fallback_source}
# True where ISO is explicitly in the 'pop' set
# mask = country_grid.astype("object").isin(pop_set) # slower, generic type
mx = country_grid.astype("U").isin(pop_set) # faster, unicode type
# Handle missing/NaN country codes via the default
# mx = xr.where(country_grid.isnull(), True, mx)
mx

# %%
# mask.sel(iso="pak").plot()
# m.sel(iso="pak").plot()
# m.sum()
mx.sel(iso="pak").plot()
mx.sum()

# %%
new_example_da = combine_by_country_v2(
    pop_da = pop_da, #: xr.DataArray,
    emis_da = emis_da, #: xr.DataArray,
    country_grid = mask, # xr.DataArray,
    mapping_df = dfil, #: str,
    default_source = "emis" #: str = "emis",  # fallback if a country isn't in the CSV
)

# %%
new_example_da

# %%
encoding = {
    "combined": {
        "zlib": True,
        "complevel": 2
    }
}

# %%
new_example_da.to_netcdf(
            path = base_gridding / "test-recombine_fallbackmethod-proxies" / "example_file_co_waste.nc",
            encoding = encoding
        )


# %%
def plot_maps(ds, ncols=1, proj=ccrs.Robinson()): 

    nrows = 1

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(6 * ncols, 4.5 * nrows),
        subplot_kw={"projection": ccrs.Robinson()}
    )

    # Flatten axes safely
    if isinstance(axes, np.ndarray):
        axes = axes.flatten() # make indexing easier
    else:
        axes = [axes]

    da = (
        ds
        .squeeze()
    )

    # Plot directly with xarray's .plot.pcolormesh
    da.plot.pcolormesh(
        ax=axes[0],
        transform=ccrs.PlateCarree(),
        cmap="GnBu",
        robust=True,
        cbar_kwargs={"orientation": "horizontal", "shrink": 0.65},
    )
    axes[0].set_title('title')
    axes[0].coastlines()

    plt.tight_layout()
    plt.show()


# %%
# old
plot_maps(emis_da)
# mew
plot_maps(new_example_da.sum(dim="iso", keep_attrs=True))


# %%
# old
emis_da

# %%
# new
new_example_da.name = "CO_2022_WST"
new_example_da

# %%
# rename some variables (REMOVE LATER)
mapping_df = dfil

# --- 1) From (iso,lat,lon) mask stack -> 2D ISO grid -------------------------
# The dominant country per pixel (label, not index)
iso_grid = country_grid.idxmax(dim="iso")           # (lat, lon), dtype=str/object

# Mask out ocean/void: where all countries are zero (or below tiny eps)
maxval = country_grid.max(dim="iso")
iso_grid = iso_grid.where(maxval > 0, other=np.nan)  # or > 1e-12 for safety

# (Optional) normalize to lowercase to match your mapping convention
iso_grid = xr.apply_ufunc(
    np.char.lower, iso_grid.astype("U"),
    dask="parallelized", output_dtypes=[str]
)

# --- 2) Build choice set from mapping (all lowercase) -------------------------
def validate_and_normalize_mapping(df: pd.DataFrame, fallback_source: str):
    if not {"iso3", "source"}.issubset(df.columns):
        raise ValueError("mapping_df must have columns: 'iso3' and 'source'")
    valid = {"pop", "emis"}
    fallback_source = str(fallback_source).strip().lower()
    if fallback_source not in valid:
        raise ValueError("fallback_source must be 'pop' or 'emis'")
    choice_map = {
        str(row.iso3).strip().lower(): str(row.source).strip().lower()
        for _, row in df.iterrows()
        if str(row.source).strip().lower() in valid
    }
    return choice_map, fallback_source

choice_map, fallback = validate_and_normalize_mapping(mapping_df, "emis")
pop_set = {iso for iso, src in choice_map.items() if src == "pop"}
known_isos = set(choice_map.keys())

# --- 3) Boolean mask: True where we should use population ---------------------
mask_pop = iso_grid.isin(pop_set)

# If a pixel’s ISO isn’t in the CSV (or is NaN), use the fallback
known_mask = iso_grid.isin(known_isos)
if fallback == "pop":
    mask_pop = xr.where(~known_mask | xr.ufuncs.isnan(iso_grid), True, mask_pop)
else:  # fallback == "emis"
    mask_pop = xr.where(~known_mask | xr.ufuncs.isnan(iso_grid), False, mask_pop)

mask_pop.name = "mask_pop"

# --- 4) Combine (sum formulation avoids dtype surprises) ----------------------
maskf = mask_pop.astype(pop_da.dtype)
combined = pop_da * maskf + emis_da * (1.0 - maskf)
combined.name = "combined"
combined.attrs.update({
    "description": "Per-country combination of population and emissions grids",
    "rule": "pop where mask_pop True; emis elsewhere; mask from dominant ISO per pixel",
    "fallback_source": fallback,
})

# %%
dfil

# %%
maxval.plot()

# %%
mapping_df = dfil

# --- 1) Dominant ISO per pixel ---
maxval = country_grid.max(dim="iso")
iso_grid = country_grid.idxmax(dim="iso")                 # (lat, lon) of ISO labels
iso_grid = iso_grid.where(maxval > 0, other=np.nan)       # ocean/void -> NaN
# think about this combination more carefully for borders (which now might have some emissions already, which we could potentially be deleting here)
# normalize to lowercase strings
iso_grid = xr.apply_ufunc(np.char.lower, iso_grid.astype("U"),
                          dask="parallelized", output_dtypes=[str])

# --- 2) Build sets from mapping (lowercase) ---
choice_map = {str(r.iso3).strip().lower(): str(r.source).strip().lower()
              for _, r in mapping_df.iterrows()
              if str(r.source).strip().lower() in {"pop","emis"}}
fallback = "pop"  # or "pop"
pop_set = {iso for iso, src in choice_map.items() if src == "pop"}
unknown_isos = set(choice_map.keys()) # where we should use population

# --- 3) Mask: True where we should use population ---
# mask_pop = iso_grid.isin(pop_set) # This always goes wrong
mask_pop = iso_grid.isin(list(pop_set)) # make sure it is a list, not a set

# use fallback where ISO is missing or unknown
unknown_or_ocean = (iso_grid.isin(unknown_isos)) | (iso_grid.isnull())  # <-- use isnull(), not isnan
mask_pop = xr.where(unknown_or_ocean, (fallback == "pop"), mask_pop)
mask_pop.name = "mask_pop"

# --- 4) Combine via sum formulation (avoids dtype surprises) ---
maskf = mask_pop.astype(pop_da.dtype)
population_modifier = 1e-0
# think about this combination more carefully for borders (which now might have some emissions already, which we could potentially be deleting here)
combined = (pop_da * maskf * population_modifier) + emis_da * (1.0 - maskf) 
combined.name = "combined"


# %%
plot_maps(emis_da)
# pop_set # fine
# unknown_isos # fine
# np.unique(iso_grid.values) # fine
# np.unique(iso_grid.isin(pop_set)) # why only False?
mask_pop.plot()
# plot_maps(mask_pop)
plot_maps(combined)

# %%
mxx = iso_grid.isin(pop_set)
print(np.unique(mxx.values))
print("Matched pixels:", int(mxx.sum()))

# %%

# %%
pop_set

# %%
np.unique(iso_grid.values)

# %%
print("iso_grid dtype:", iso_grid.dtype)
print("sample:", iso_grid.values.ravel()[0:10])
print("np.char.lower(iso_grid.values) sample:", np.char.lower(iso_grid.values.ravel()[0:10]))


# %%
# What is the result of the mask?
mask = iso_grid.isin(pop_set)
print(np.unique(mask.values))  # should include True if any match


# %%
def normalize_iso_grid(iso_grid: xr.DataArray) -> xr.DataArray:
    """
    Normalize an ISO code grid:
    - Converts to lowercase
    - Replaces string 'nan' with actual np.nan
    - Ensures dtype is uniform unicode
    - Strips whitespace
    """
    # Force unicode dtype
    iso_grid = iso_grid.astype("U")

    # Apply string cleanup
    iso_grid = xr.apply_ufunc(
        lambda x: np.where(
            (x == "") | (x == "nan") | (x == "NaN") | (x == "None"),
            np.nan,
            np.char.strip(np.char.lower(x))
        ),
        iso_grid,
        dask="parallelized",
        output_dtypes=[str]
    )

    return iso_grid

iso_grid = normalize_iso_grid(country_grid)
iso_grid.isin(pop_set)

# %%
np.unique(iso_grid.isin(pop_set).values)

# %%
unique_isos_in_grid = set(np.unique(iso_grid.values))
intersection = unique_isos_in_grid & pop_set
intersection

# %%
