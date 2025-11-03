from flox.xarray import xarray_reduce as _original_xarray_reduce
import ptolemy.raster

# Patch xarray_reduce to use fill_value=0 by default
def patched_xarray_reduce(*args, **kwargs):
    """Patched xarray_reduce that defaults fill_value to 0 instead of None."""
    if 'fill_value' not in kwargs:
        kwargs['fill_value'] = 0
    return _original_xarray_reduce(*args, **kwargs)

# Replace the xarray_reduce function in the ptolemy.raster module
ptolemy.raster.xarray_reduce = patched_xarray_reduce
print("✅ Patched xarray_reduce to use fill_value=0 by default")
