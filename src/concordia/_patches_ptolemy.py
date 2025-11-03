import pandas as pd
import xarray as xr
from flox.xarray import xarray_reduce
import ptolemy.raster

def patched_aggregate(self, ndraster, func="sum", interior_only=False):
    """Enhanced IndexRaster.aggregate with diagnostics and safe fill_value=0."""
    print(f"🧩 [aggregate] Called for index level '{getattr(self, 'dim', None)}'")
    print(f"    → number of indices: {len(self.index)}")
    print(f"    → proxy raster dims: {ndraster.dims}")

    # compute number of valid (non-NaN) pixels in the proxy
    try:
        valid_pixels = ndraster.notnull().sum().compute()
        print(f"    → valid proxy pixels: {valid_pixels.item() if hasattr(valid_pixels, 'item') else valid_pixels}")
    except Exception as e:
        print(f"    ⚠️ could not compute valid pixels: {e}")

    # --- core aggregation ---
    result = xarray_reduce(
        ndraster,
        self.indicator,
        expected_groups=pd.RangeIndex(len(self.index) + 1),
        func=func,
        fill_value=0,   # key fix
    ).isel({self.dim: slice(1, None)})

    # --- post-check: detect where fill_value was effectively used ---
    # Missing groups = indices that had no contribution (sum=0)
    try:
        zero_groups = result[self.dim][(result == 0).all(dim=[d for d in result.dims if d != self.dim])]
        if len(zero_groups) > 0:
            print(f"    ⚠️ [fill_value triggered] {len(zero_groups)} index groups had no data")
            if len(zero_groups) < 10:
                print("       missing indices:", zero_groups.values)
    except Exception as e:
        print(f"    ⚠️ could not inspect zero groups: {e}")

    if interior_only:
        return result.assign_coords({self.dim: self.index})
    return result

# replace the method at runtime
ptolemy.raster.IndexRaster.aggregate = patched_aggregate
print("✅ Patched ptolemy.IndexRaster.aggregate with fill_value=0 + debug logging")
