# # Original fix, gives error?
# # ----------------------------

# from flox.xarray import xarray_reduce as _original_xarray_reduce
# import ptolemy.raster

# # Patch xarray_reduce to use fill_value=0 by default
# def patched_xarray_reduce(*args, **kwargs):
#     """Patched xarray_reduce that defaults fill_value to 0 instead of None."""
#     if 'fill_value' not in kwargs:
#         kwargs['fill_value'] = 0
#     return _original_xarray_reduce(*args, **kwargs)

# # Replace the xarray_reduce function in the ptolemy.raster module
# ptolemy.raster.xarray_reduce = patched_xarray_reduce
# print("✅ Patched xarray_reduce to use fill_value=0 by default")



# Updated (28.11.2025) fix
# ----------------------------

import flox.core

# Store the original reindex_ function
_original_reindex = flox.core.reindex_

def patched_reindex_(array, from_, to, fill_value=None, axis=-1, promote=False, **kwargs):
    """Wrapper that provides a default fill_value of 0 if None is passed."""
    if fill_value is None:
        fill_value = 0
    return _original_reindex(array, from_=from_, to=to, fill_value=fill_value, axis=axis, promote=promote, **kwargs)

# Replace the function in the flox.core module
flox.core.reindex_ = patched_reindex_