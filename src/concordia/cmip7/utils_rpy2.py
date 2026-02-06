# %%
from __future__ import annotations

# %%
import numpy as np
import xarray as xr
from pathlib import Path

# rpy2: Python <-> R bridge; needed for rewriting CEDS files in the same format
import rpy2.robjects as ro
from rpy2.robjects import default_converter
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import numpy2ri

# %%
r = ro.r

# %%
def save_da_as_rd(
    da: xr.DataArray,
    out_path: str | Path,
    *,
    object_name: str | None = None,
    undo_flip: bool = True,
    float_dtype: str = "float64"
):
    """
    Save a 2D xarray.DataArray (lat x lon) as an RData file containing a named R matrix.
    Reading with pyreadr.read_r(...) returns an OrderedDict with that name as the key
    and a pandas DataFrame of shape (lat, lon) as the value.

    Parameters
    ----------
    da : xr.DataArray        # 2D (lat x lon), numeric
    out_path : str | Path    # e.g. 'CO_2022_WST.Rd'
    object_name : str        # name of the object inside the RData (defaults to file stem)
    undo_flip : bool         # if you flipped with [::-1] at read, flip back here
    """
    out_path = Path(out_path)
    if object_name is None:
        object_name = out_path.stem

    if da.ndim != 2:
        raise ValueError(f"Expected 2D DataArray (lat x lon), got shape {da.shape}")
    if not np.issubdtype(da.dtype, np.number):
        raise TypeError(f"DataArray dtype must be numeric; got {da.dtype}")

    arr = np.asarray(da.data, dtype=float_dtype) # could also do np.float32 or np.float64
    if undo_flip:
        arr = arr[::-1].copy()

    nrow, ncol = arr.shape

    with localconverter(default_converter + numpy2ri.converter):
        r_vec = ro.FloatVector(arr.ravel(order="C"))
        r_mat = r["matrix"](r_vec, nrow=nrow, ncol=ncol, byrow=True)

        ro.globalenv[object_name] = r_mat
        r(f"save({object_name}, file={repr(str(out_path))})")


# %% [markdown]
# # -------- Example usage --------
# Assuming 'da' is your DataArray (lat: 360, lon: 720)
# and you want the re-read key to be 'CO_2022_WST'
# save_da_as_rd(da, "CO_2022_WST.Rd", object_name="CO_2022_WST", undo_flip=True)
# # da is your DataArray, e.g.:
# # <xarray.DataArray 'VOC-WST' (lat: 360, lon: 720)> float32 ...
# # Save as single-object .Rd (works like .Rds)
# save_da_as_rd(da, "VOC_2022_WST.Rd", container="RDS", undo_flip=True)
# Notes & tips
# RDS vs RData: If your originals are single-object files, use container="RDS" (simpler). If they are workspaces with a named symbol, use container="RDATA" and set object_name (often the stem, e.g., SO2_2021_WST).
# Validation round-trip (optional): you can re-read the saved file with pyreadr.read_r(...) and compare to your original NumPy array (after applying the same flip logic) to confirm equality.
