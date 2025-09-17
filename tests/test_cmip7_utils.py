# Goal: test 'save_da_as_rd'
# State of file: not finished. Just some draft code that has been run once but is not guaranteed to work. 
# from collections import OrderedDict
# import pandas as pd
# import numpy as np
# import pytest

# from concordia.cmip7.utils import read_r_variable
# from concordia.cmip7.utils_rpy2 import save_da_as_rd

# AN EXAMPLE TEST THAT ONCE WAS RUN
# def assert_ordereddict_equal(left: OrderedDict, right: OrderedDict):
#     assert isinstance(left, OrderedDict)
#     assert isinstance(right, OrderedDict)

#     # same keys in the same order
#     assert list(left.keys()) == list(right.keys())

#     # compare values
#     for k in left.keys():
#         lv, rv = left[k], right[k]
#         if isinstance(lv, pd.DataFrame) and isinstance(rv, pd.DataFrame):
#             # exact equality; relax with check_exact=False, rtol=1e-6 if needed
#             pd.testing.assert_frame_equal(lv, rv, check_dtype=False, check_exact=True)
#         elif isinstance(lv, pd.Series) and isinstance(rv, pd.Series):
#             pd.testing.assert_series_equal(lv, rv, check_dtype=False, check_exact=True)
#         elif isinstance(lv, np.ndarray) and isinstance(rv, np.ndarray):
#             assert np.array_equal(lv, rv)
#         else:
#             assert lv == rv

# save_da_as_rd(xr.DataArray(read_r_variable(in_test, float_dtype="float64"), 
#                  coords={"lat": template.lat, "lon": template.lon}, 
#                  name="emissions"), 
#               out_path=out_test, 
#               object_name="CO_2022_WST", 
#               undo_flip=True)

# in_test = settings.gridding_path / "ceds_input" / "non-point_source_proxy_final_sector" / "CO_2022_WST.Rd"
# out_test = settings.gridding_path / "ceds_input" / "non-point_source_proxy_final_sector" / "fixed_fallback" / "CO_2022_WST.Rd"

# assert_ordereddict_equal(pyreadr.read_r(in_test), 
#                          pyreadr.read_r(out_test))



# SKETCHES OF POSSIBLE OTHER TESTS:
# def test_ordereddict_with_dataframes_equal():
#     df1 = pd.DataFrame(np.zeros((3, 4)))
#     df2 = pd.DataFrame(np.zeros((3, 4)))
#     a = OrderedDict([("CO_2022_WST", df1)])
#     b = OrderedDict([("CO_2022_WST", df2)])
#     assert_ordereddict_equal(a, b)

# def test_ordereddict_with_dataframes_order_matters():
#     df = pd.DataFrame(np.zeros((2, 2)))
#     a = OrderedDict([("A", df), ("B", df)])
#     b = OrderedDict([("B", df), ("A", df)])
#     with pytest.raises(AssertionError):
#         assert_ordereddict_equal(a, b)
