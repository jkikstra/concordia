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

# Copied over from Zeb: https://github.com/climate-resource/input4mips-prep/blob/future-emms/scripts/fix-IIASA-IAMC-scendraft-0-3-0.py
# ...who provided this for the 'shared data' that Jarmo uploaded on 

# %%
from __future__ import annotations

import shutil
from pathlib import Path

import iris
import netCDF4
import numpy as np
from input4mips_prep import (
    copy_attributes,
    copy_dimension,
    copy_dimensions,
    copy_variable,
)
from input4mips_validation.cvs import load_cvs
from input4mips_validation.inference.from_data import (
    infer_time_start_time_end_for_filename,
)
from input4mips_validation.io import (
    generate_creation_timestamp,
    generate_tracking_id,
)
from input4mips_validation.validation.file import get_validate_file_result
from input4mips_validation.xarray_helpers.iris import ds_from_iris_cubes

# %%
def add_correct_time_bounds(
    ds: netCDF4.Dataset,
    ds_fixed: netCDF4.Dataset,
    bounds_time: str = "time_bnds",
) -> netCDF4.Dataset:
    """
    Add the correct time bounds

    Parameters
    ----------
    ds
        Dataset from which to get the original data

    ds_fixed
        Fixed dataset from which to get the correct time bounds

    bounds_time
        Name of the variable to use for time bounds

    Returns
    -------
    :
        `ds_fixed` with time bounds applied

        Note that the operation is in place,
        so returning this is just a convenience.
    """
    if "bound" not in ds.dimensions:
        raise AssertionError

    ds_fixed.createVariable(bounds_time, ds["time"].datatype, ("time", "bound"))

    time_int = ds["time"][:]
    time_date = netCDF4.num2date(
        time_int,
        units=ds["time"].getncattr("units"),
        calendar=ds["time"].getncattr("calendar"),
    )
    bounds_time_fixed_l = []
    # Loop, crazy slow, whatever
    n_months_in_year = 12
    for v_int, v_date in zip(time_int, time_date):
        start_of_bound_date = type(v_date)(v_date.year, v_date.month, 1)
        if v_date.month == n_months_in_year:
            end_of_bound_date = type(v_date)(v_date.year + 1, 1, 1)

        else:
            end_of_bound_date = type(v_date)(v_date.year, v_date.month + 1, 1)

        bounds_time_fixed_l.append(
            [
                netCDF4.date2num(
                    v,
                    units=ds["time"].getncattr("units"),
                    calendar=ds["time"].getncattr("calendar"),
                )
                for v in [start_of_bound_date, end_of_bound_date]
            ]
        )

    bounds_time_fixed = np.array(bounds_time_fixed_l)
    ds_fixed.variables[bounds_time][:] = bounds_time_fixed

    return ds_fixed


def rewrite_anthro_file(
    fp: Path,
    out_dir: Path,
    verbose: bool = True,
):
    """
    Re-write anthropogenic emissions file

    Parameters
    ----------
    fp
        File path to rewrite

    out_dir
        Directory in which to write the rewritten file

    verbose
        Should lots of information be printed?
    """
    ds = netCDF4.Dataset(fp)

    fixed_file = out_dir / fp.name.replace("_gn", "-0-3-0_gn")
    ds_fixed = netCDF4.Dataset(fixed_file, "w")

    if verbose:
        print(f"Rewriting {fp} to {fixed_file}")

    copy_attributes(ds, ds_fixed)
    ds_fixed.setncattr("source_version", "0.3.0")
    ds_fixed.setncattr("contact", "kikstra@iiasa.ac.at")
    ds_fixed.setncattr("creation_date", generate_creation_timestamp())
    ds_fixed.setncattr("tracking_id", generate_tracking_id())
    # Urgh out of date cf-checker
    ds_fixed.setncattr("Conventions", "CF-1.8")

    for dimension in ds.dimensions:
        if dimension != "sector":
            copy_dimension(ds, ds_fixed, dimension)

    in_sectors = list(ds["sector"][:])
    assert in_sectors == [  # noqa: S101
        "Agriculture",
        "Energy",
        "Industrial",
        "Transportation",
        "Residential, Commercial, Other",
        "Solvents Production and Application",
        "Waste",
        "International Shipping",
        "Other Capture and Removal",
    ], "Have to shuffle order too probably to avoid confusing people"

    ds_fixed.createDimension("sector", len(ds["sector"][:]))
    ds_fixed.createVariable("sector", int, ("sector",))
    ds_fixed["sector"][:] = np.arange(len(ds["sector"][:]))
    ds_fixed["sector"].setncattr("long_name", "sector")
    ds_fixed["sector"].setncattr("bounds", "sector_bnds")
    ds_fixed["sector"].setncattr(
        "ids", "; ".join(f"{i}: {sector}" for i, sector in enumerate(in_sectors))
    )

    ds_fixed.createVariable("sector_bnds", float, ("sector", "bound"))
    ds_fixed["sector_bnds"][:] = np.vstack(
        [
            np.arange(len(ds["sector"][:])) - 0.5,
            np.arange(len(ds["sector"][:])) + 0.5,
        ]
    ).T

    bounds_vars_except_time = [
        v for v in ds.variables if "bnds" in v and "time" not in v
    ]
    bounds_time = "time_bnds"
    other_vars = [
        v for v in ds.variables if v not in [*bounds_vars_except_time, bounds_time]
    ]

    for variable in bounds_vars_except_time:
        copy_variable(
            ds,
            ds_fixed,
            variable,
            # Value taken from variable instead rather than being repeated
            copy_fill_value=False,
            copy_attrs=False,
        )

    ds_fixed = add_correct_time_bounds(ds, ds_fixed)

    for variable in other_vars:
        if variable == "sector":
            continue

        if verbose:
            print(f"Copying {variable}")

        copy_variable(ds, ds_fixed, variable)

    v_name = fp.name.split("_")[0]
    v_name = ds.getncattr("variable_id")
    ds_fixed[v_name].setncattr(
        "long_name", f"{v_name.split('_')[0]} anthropogenic emissions"
    )
    # TODO: check this
    ds_fixed[v_name].setncattr("units", "kg m-2 s-1")

    ds.close()
    ds_fixed.close()


def rewrite_air_anthro_file(
    fp: Path,
    out_dir: Path,
    verbose: bool = True,
):
    """
    Re-write anthropogenic aircraft emissions file

    Parameters
    ----------
    fp
        File path to rewrite

    out_dir
        Directory in which to write the rewritten file

    verbose
        Should lots of information be printed?
    """
    ds = netCDF4.Dataset(fp)

    fixed_file = out_dir / fp.name.replace("_gn", "-0-3-0_gn")
    ds_fixed = netCDF4.Dataset(fixed_file, "w")

    if verbose:
        print(f"Rewriting {fp} to {fixed_file}")

    copy_attributes(ds, ds_fixed)
    ds_fixed.setncattr("source_version", "0.3.0")
    ds_fixed.setncattr("contact", "kikstra@iiasa.ac.at")
    ds_fixed.setncattr("creation_date", generate_creation_timestamp())
    ds_fixed.setncattr("tracking_id", generate_tracking_id())

    copy_dimensions(ds, ds_fixed)

    bounds_vars_except_time = [
        v for v in ds.variables if "bnds" in v and "time" not in v
    ]
    bounds_time = "time_bnds"
    other_vars = [
        v for v in ds.variables if v not in [*bounds_vars_except_time, bounds_time]
    ]

    for variable in bounds_vars_except_time:
        copy_variable(
            ds,
            ds_fixed,
            variable,
            # Value taken from variable instead rather than being repeated
            copy_fill_value=False,
            copy_attrs=False,
        )

    ds_fixed = add_correct_time_bounds(ds, ds_fixed)

    # Add level bounds
    if "bound" not in ds.dimensions:
        raise AssertionError

    level_diff = np.diff(ds["level"][:])
    # Check levels all same size,
    # if not, logic below needs updating
    np.testing.assert_allclose(level_diff, level_diff[0], atol=1e-10)
    level_lower = np.round(ds["level"] - level_diff[0] / 2, 8)
    level_lower[level_lower == 0.0] = np.abs(level_lower[level_lower == 0.0])
    level_upper = level_lower + level_diff[0]
    level_bnds_vals = np.vstack([level_lower, level_upper]).T

    ds_fixed.createVariable("level_bnds", ds["level"].datatype, ("level", "bound"))
    ds_fixed.variables["level_bnds"][:] = level_bnds_vals

    for variable in other_vars:
        if verbose:
            print(f"Copying {variable}")

        copy_variable(ds, ds_fixed, variable)

    ds.close()
    ds_fixed.close()


def rewrite_openburning_file(
    fp: Path,
    out_dir: Path,
    verbose: bool = True,
):
    """
    Re-write open biomass burning emissions file

    Parameters
    ----------
    fp
        File path to rewrite

    out_dir
        Directory in which to write the rewritten file

    verbose
        Should lots of information be printed?
    """
    ds = netCDF4.Dataset(fp)

    fixed_file = out_dir / fp.name.replace("_gn", "-0-3-0_gn")
    ds_fixed = netCDF4.Dataset(fixed_file, "w")

    if verbose:
        print(f"Rewriting {fp} to {fixed_file}")

    copy_attributes(ds, ds_fixed)
    ds_fixed.setncattr("source_version", "0.3.0")
    ds_fixed.setncattr("contact", "kikstra@iiasa.ac.at")
    ds_fixed.setncattr("creation_date", generate_creation_timestamp())
    ds_fixed.setncattr("tracking_id", generate_tracking_id())

    for dimension in ds.dimensions:
        if dimension != "sector":
            copy_dimension(ds, ds_fixed, dimension)

    in_sectors = list(ds["sector"][:])
    assert in_sectors == [  # noqa: S101
        "Agricultural Waste Burning",
        "Forest Burning",
        "Grassland Burning",
        "Peat Burning",
    ], "Have to shuffle order too probably to avoid confusing people"

    ds_fixed.createDimension("sector", len(ds["sector"][:]))
    ds_fixed.createVariable("sector", int, ("sector",))
    ds_fixed["sector"][:] = np.arange(len(ds["sector"][:]))
    ds_fixed["sector"].setncattr("long_name", "sector")
    ds_fixed["sector"].setncattr("bounds", "sector_bnds")
    ds_fixed["sector"].setncattr(
        "ids", "; ".join(f"{i}: {sector}" for i, sector in enumerate(in_sectors))
    )

    ds_fixed.createVariable("sector_bnds", float, ("sector", "bound"))
    ds_fixed["sector_bnds"][:] = np.vstack(
        [
            np.arange(len(ds["sector"][:])) - 0.5,
            np.arange(len(ds["sector"][:])) + 0.5,
        ]
    ).T

    bounds_vars_except_time = [
        v for v in ds.variables if "bnds" in v and "time" not in v
    ]
    bounds_time = "time_bnds"
    other_vars = [
        v for v in ds.variables if v not in [*bounds_vars_except_time, bounds_time]
    ]

    for variable in bounds_vars_except_time:
        copy_variable(
            ds,
            ds_fixed,
            variable,
            # Value taken from variable instead rather than being repeated
            copy_fill_value=False,
            copy_attrs=False,
        )

    ds_fixed = add_correct_time_bounds(ds, ds_fixed)

    for variable in other_vars:
        if variable == "sector":
            continue

        if verbose:
            print(f"Copying {variable}")

        copy_variable(ds, ds_fixed, variable)

    ds.close()
    ds_fixed.close()


def rewrite_file_in_drs(
    file: Path, output_root: Path, cv_source: str, verbose: bool = True
) -> Path:
    """
    Re-write files in the DRS - ensures that filenames are correct too

    Also does some other checks implicitly
    """
    cvs = load_cvs(cv_source=cv_source)

    ds = ds_from_iris_cubes(
        iris.load(file),
        # xr_variable_processor=xr_variable_processor,
        raw_file=file,
        time_dimension="time",
    )

    time_start, time_end = infer_time_start_time_end_for_filename(
        ds=ds,
        frequency_metadata_key="frequency",
        no_time_axis_frequency="fx",
        time_dimension="time",
    )

    full_file_path = cvs.DRS.get_file_path(
        root_data_dir=output_root,
        available_attributes=ds.attrs,
        time_start=time_start,
        time_end=time_end,
    )

    full_file_path.parent.mkdir(exist_ok=True, parents=True)
    shutil.copy(file, full_file_path)

    if verbose:
        print(f"File written according to the DRS in {full_file_path}")

    return full_file_path

# %%
def main():
    """
    Re-write the files
    """
    SOURCE_DIR = Path("../input4MIPs_CVs/iiasa-20250928")
    TMP_DIR = Path("../input4MIPs_CVs/iiasa-20250928-rewrite")
    OUT_DIR = Path("../input4MIPs_CVs/iiasa-20250928-drs")

    CV_SOURCE = "https://raw.githubusercontent.com/jkikstra/input4MIPs_CVs/refs/heads/source_id_scenariomip-emissions/CVs/"

    TMP_DIR.mkdir(exist_ok=True, parents=True)

    for f in SOURCE_DIR.glob("*em-anthro*.nc"):
        rewrite_anthro_file(f, TMP_DIR)

    for f in SOURCE_DIR.glob("*AIR-anthro*.nc"):
        rewrite_air_anthro_file(f, TMP_DIR)

    for f in SOURCE_DIR.glob("*em-openburning*.nc"):
        rewrite_openburning_file(f, TMP_DIR)

    OUT_DIR.mkdir(exist_ok=True, parents=True)
    for tmp_file in TMP_DIR.glob("*.nc"):
        out_file = rewrite_file_in_drs(
            tmp_file, output_root=OUT_DIR, cv_source=CV_SOURCE
        )

        validation_res = get_validate_file_result(
            out_file,
            cv_source=CV_SOURCE,
            # xr_variable_processor=xr_variable_processor,
            # frequency_metadata_keys=frequency_metadata_keys,
            # bounds_info=bounds_info,
            # Allow issues with units for sector
            allow_cf_checker_warnings=True,
        )
        validation_res.raise_if_errors()

if __name__ == "__main__":
    main()