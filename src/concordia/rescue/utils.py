import cf_xarray
import cftime
import datetime
import ftplib

import xarray as xr
import pandas as pd

from xarray.coding.times import convert_times

SECTOR_MAPPING = {
    # anthro
    "Agriculture": 0,
    "Energy Sector": 1,
    "Industrial Sector": 2,
    "International Shipping": 7,
    "Residential Commercial Other": 4,
    "Solvents Production and Application": 5,
    "Transportation Sector": 3,
    "Waste": 6,
    # aircraft
    "Aircraft": 8,
    # openburning
    "Agricultural Waste Burning": 0,
    "Forest Burning": 1,
    "Grassland Burning": 2,
    "Peat Burning": 3,
    # cdr
    "CDR DACCS": 0,
    "CDR Industry": 1,
}

ATTRS = {
    "lat": {
        "units": "degrees_north",
        "long_name": "latitude",
        "axis": "Y",
        "bounds": "lat_bnds",
        "realtopology": "linear",
        "standard_name": "latitude",
    },
    "lon": {
        "units": "degrees_east",
        "long_name": "longitude",
        "axis": "X",
        "bounds": "lon_bnds",
        "modulo": 360.0,
        "realtopology": "circular",
        "standard_name": "longitude",
        "topology": "circular",
    },
    "time": {
        "long_name": "time",
        "axis": "T",
        "bounds": "time_bnds",
        "realtopology": "linear",
        "standard_name": "time",
    },
    "level": {
        "units": "km",
        "long_name": "altitude",
    },
}

DATA_HANDLES = {
    "em_anthro": "anthropogenic",
    "em_AIR_anthro": "aircraft",
    "em_openburning": "openburning",
}

DS_ATTRS = dict(
    Conventions="CF-1.6",
    activity_id="input4MIPs",
    comment="Gridded emissions produced after harmonization and downscaling as part of the RESCUE project. See https://github.com/IAMconsortium/concordia and https://github.com/iiasa/aneris for documentation on the processes.",
    contact="Matthew J. Gidden (gidden@iiasa.ac.at)",
    data_structure="grid",
    dataset_category="emissions",
    external_variables="gridcell_area",
    frequency="mon",
    further_info_url="https://rescue-climate.eu/",
    grid="0.5x0.5 degree latitude x longitude",
    grid_label="gn",
    license="Creative Commons Attribution-ShareAlike 4.0 International License (https://creativecommons.org/licenses). Further information about this data, including some limitations, can be found via the further_info_url (recorded as a global attribute in this file). The data producers and data providers make no warranty, either express or implied, including, but not limited to, warranties of merchantability and fitness for a particular purpose. All liabilities arising from the supply of the information (including any liability arising in negligence) are excluded to the fullest extent permitted by law.",
    institution="IIASA-PIK",
    institution_id="IIASA",
    mip_era="CMIP7",
    nominal_resolution="50 km",
    realm="atmos",
    references="See: https://github.com/IAMconsortium/concordia and https://rescue-climate.eu/ for references",
    source="Scenarios generated as part of the RESCUE project, see https://rescue-climate.eu/",
    table_id="input4MIPs",
    target_mip="RESCUE",
    product="primary-emissions-data",
    start_date="202001",
    end_date="210012",
)


def convert_to_datetime(da):
    da = da.stack(time=("year", "month"))
    dates = pd.DatetimeIndex(
        da.indexes["time"].map(
            lambda t: datetime.date(t[0], t[1], 16 if t[1] != 2 else 15)
        )
    )
    return da.drop_vars(["time", "year", "month"]).assign_coords(time=dates)


def clean_coords(da, whitelist=["lat", "lon", "time", "sector", "level"]):
    return da.squeeze(dim=set(da.coords) - set(whitelist), drop=True)


def add_bounds(da, bounds=["lat", "lon", "time", "level"]):
    bounds = list(set(bounds) & set(da.coords))
    da = da.cf.add_bounds(bounds, output_dim="bound")
    da = (
        da.assign_coords(
            time=convert_times(da.time.data, cftime.DatetimeNoLeap),
            time_bounds=(
                ("time", "bound"),
                convert_times(
                    da.time_bounds.data.ravel(), cftime.DatetimeNoLeap
                ).reshape(-1, 2),
            ),
        )
        .reset_coords([f"{b}_bounds" for b in bounds])
        .rename({f"{b}_bounds": f"{b}_bnds" for b in bounds})
    )
    return da


def add_sector_mapping(da, sector_mapping):
    keys = da.indexes["sector"]
    vals = keys.map(sector_mapping)
    return da.assign_coords(
        sector=xr.DataArray(
            vals,
            attrs=dict(long_name="sector", id=str({v: k for v, k in zip(vals, keys)})),
        )
    )


def replace_attrs(da, attrs):
    for k, v in attrs.items():
        if k in da:
            da[k].attrs = v
    return da


def clean_var(da, name, gas, handle):
    attrs = {
        "units": "kg m-2 s-1",
        "cell_methods": "time: mean",
        "long_name": f"{gas} {handle} emissions",
    }
    da[name].attrs = attrs
    return da


def ds_attrs(name, model, scenario, version, date):
    split = name.split("_")
    gas = split[0]
    handle = DATA_HANDLES["_".join(split[1:])]

    extra_attrs = dict(
        source_version=version,
        source_id=f"{DS_ATTRS['institution']}-{model}-{scenario}-{version}".replace(
            " ", "__"
        ),
        variable_id=name,
        creation_date=date,
        title=f"Future {handle} emissions of {gas} in {scenario}",
        reporting_unit=f"Mass flux of {gas}",
    )
    attrs = DS_ATTRS | extra_attrs
    return attrs


class DressUp:
    def __init__(self, version) -> None:
        self.version = version
        self.date = str(datetime.datetime.today())

    def __call__(self, da, model, scenario, **kwargs):
        vars = list(da.data_vars)
        assert len(vars) == 1, vars

        name = vars[0]
        split = name.split("_")
        gas = split[0]
        handle = DATA_HANDLES["_".join(split[1:])]

        return (
            da.pipe(convert_to_datetime)
            .pipe(clean_coords)
            .pipe(add_bounds)
            .pipe(add_sector_mapping, SECTOR_MAPPING)
            .pipe(replace_attrs, ATTRS)
            .pipe(clean_var, name, gas, handle)
            .assign_attrs(ds_attrs(name, model, scenario, self.version, self.date))
        )


def ftp_upload(cfg, local_path, remote_path):
    paths = list(local_path.iterdir())

    ftp = ftplib.FTP()
    ftp.connect(cfg["server"], cfg["port"])
    ftp.login(cfg["user"], cfg["pass"])

    try:
        if not remote_path.as_posix() in ftp.nlst(remote_path.parent.as_posix()):
            ftp.mkd(remote_path.as_posix())

        ftp.cwd(remote_path.as_posix())
        for path in paths:
            print(f"Uploading {path.name} to {remote_path.as_posix()}")
            ftp.storbinary("STOR " + path.name, open(path, "rb"))

    finally:
        ftp.close()
