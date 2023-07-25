from __future__ import annotations

import datetime
import re
import warnings
from functools import lru_cache
from multiprocessing.pool import ThreadPool
from typing import Iterable

import numpy as np
import pandas as pd
import requests


def load_meta(*, cat: bool = False) -> pd.DataFrame:
    """Load the station metadata table.

    https://www.ncei.noaa.gov/pub/data/uscrn/products/stations.tsv
    """
    url = "https://www.ncei.noaa.gov/pub/data/uscrn/products/stations.tsv"

    now = datetime.datetime.now(datetime.timezone.utc)
    df = pd.read_csv(
        url,
        sep="\t",
        header=0,
        dtype={0: str, 13: str, 15: str},
        parse_dates=[10, 11],
        na_values=["NA", "UN"],
    )
    df.columns = df.columns.str.lower()

    if cat:
        for col in ["status", "operation", "network"]:
            df[col] = df[col].astype("category")

    df.attrs.update(created=now)

    return df


@lru_cache(1)
def get_daily_col_info() -> pd.DataFrame:
    """Read the column info file (the individual files don't have headers).

    https://www.ncei.noaa.gov/pub/data/uscrn/products/daily01/headers.txt
    """
    # "This file contains the following three lines: Field Number, Field Name and Unit of Measure."
    url = "https://www.ncei.noaa.gov/pub/data/uscrn/products/daily01/headers.txt"
    r = requests.get(url)
    r.raise_for_status()
    lines = r.text.splitlines()
    assert len(lines) == 3
    nums = lines[0].split()
    columns = lines[1].split()
    assert len(nums) == len(columns)
    assert nums == [str(i + 1) for i in range(len(columns))]

    # For consistency with meta, use 'wban' instead of 'wbanno'
    assert columns[0] == "WBANNO"
    columns[0] = "WBAN"

    # Lowercase better
    columns = [c.lower() for c in columns]

    # Based on
    # - https://www.ncei.noaa.gov/pub/data/uscrn/products/daily01/headers.txt
    # - https://www.ncei.noaa.gov/pub/data/uscrn/products/daily01/readme.txt

    attrs = {
        # vn: (long_name, units, description).
        "wban": ("WBAN number", "", "The station WBAN number."),
        "lst_date": (
            "LST date",
            "",
            "The Local Standard Time (LST) date of the observation.",
        ),
        "crx_vn": (
            "station datalogger version number",
            "",
            (
                "The version number of the station datalogger program that was in effect at the time of the observation. "
                "Note: This field should be treated as text (i.e. string)."
            ),
        ),
        "longitude": (
            "station longitude",
            "degree_east",
            "Station longitude, using WGS-84.",
        ),
        "latitude": (
            "station latitude",
            "degree_north",
            "Station latitude, using WGS-84.",
        ),
        "t_daily_max": (
            "daily maximum air temperature",
            "degree_Celsius",
            "Maximum air temperature, in degrees C. See Note F.",
        ),
        "t_daily_min": (
            "daily minimum air temperatuer",
            "degree_Celsius",
            "Minimum air temperature, in degrees C. See Note F.",
        ),
        "t_daily_mean": (
            "(t_daily_max + t_daily_min) / 2",
            "degree_Celsius",
            "Mean air temperature, in degrees C, calculated using the typical historical approach: (T_DAILY_MAX + T_DAILY_MIN) / 2. See Note F.",
        ),
        "t_daily_avg": (
            "daily average air temperature",
            "degree_Celsius",
            "Average air temperature, in degrees C. See Note F.",
        ),
        "p_daily_calc": (
            "daily total precip",
            "mm",
            "Total amount of precipitation, in mm. See Note F.",
        ),
        "solarad_daily": (
            "daily total solar radiation",
            "MJ m-2",
            "Total solar energy, in MJ/meter^2, calculated from the hourly average global solar radiation rates and converted to energy by integrating over time.",
        ),
        "sur_temp_daily_type": (
            "type of infrared surface temperature measurement",
            "",
            "Type of infrared surface temperature measurement. 'R' denotes raw measurements, 'C' denotes corrected measurements, and 'U' indicates unknown/missing. See Note G.",
        ),
        "sur_temp_daily_max": (
            "daily maximum infrared surface temperature",
            "degree_Celsius",
            "Maximum infrared surface temperature, in degrees C.",
        ),
        "sur_temp_daily_min": (
            "daily minimum infrared surface temperature",
            "degree_Celsius",
            "Minimum infrared surface temperature, in degrees C.",
        ),
        "sur_temp_daily_avg": (
            "daily average infrared surface temperature",
            "degree_Celsius",
            "Average infrared surface temperature, in degrees C.",
        ),
        "rh_daily_max": (
            "daily maximum relative humidity",
            "%",
            "Maximum relative humidity, in %. See Notes H and I.",
        ),
        "rh_daily_min": (
            "daily minimum relative humidity",
            "%",
            "Minimum relative humidity, in %. See Notes H and I.",
        ),
        "rh_daily_avg": (
            "daily average relative humidity",
            "%",
            "Average relative humidity, in %. See Notes H and I.",
        ),
        "soil_moisture_5_daily": (
            "daily average soil moisture at 5 cm depth",
            "1",
            "Average soil moisture, in fractional volumetric water content (m3 m-3), at 5 cm below the surface. See Notes I and J.",
        ),
        "soil_moisture_10_daily": (
            "daily average soil moisture at 10 cm depth",
            "1",
            "Average soil moisture, in fractional volumetric water content (m3 m-3), at 10 cm below the surface. See Notes I and J.",
        ),
        "soil_moisture_20_daily": (
            "daily average soil moisture at 20 cm depth",
            "1",
            "Average soil moisture, in fractional volumetric water content (m3 m-3), at 20 cm below the surface. See Notes I and J.",
        ),
        "soil_moisture_50_daily": (
            "daily average soil moisture at 50 cm depth",
            "1",
            "Average soil moisture, in fractional volumetric water content (m3 m-3), at 50 cm below the surface. See Notes I and J.",
        ),
        "soil_moisture_100_daily": (
            "daily average soil moisture at 100 cm depth",
            "1",
            "Average soil moisture, in fractional volumetric water content (m3 m-3), at 100 cm below the surface. See Notes I and J.",
        ),
        "soil_temp_5_daily": (
            "daily average soil temperature at 5 cm depth",
            "degree_Celsius",
            "Average soil temperature, in degrees C, at 5 cm below the surface. See Notes I and J.",
        ),
        "soil_temp_10_daily": (
            "daily average soil temperature at 10 cm depth",
            "degree_Celsius",
            "Average soil temperature, in degrees C, at 10 cm below the surface. See Notes I and J.",
        ),
        "soil_temp_20_daily": (
            "daily average soil temperature at 20 cm depth",
            "degree_Celsius",
            "Average soil temperature, in degrees C, at 20 cm below the surface. See Notes I and J.",
        ),
        "soil_temp_50_daily": (
            "daily average soil temperature at 50 cm depth",
            "degree_Celsius",
            "Average soil temperature, in degrees C, at 50 cm below the surface. See Notes I and J.",
        ),
        "soil_temp_100_daily": (
            "daily average soil temperature at 100 cm depth",
            "degree_Celsius",
            "Average  soil temperature, in degrees C, at 100 cm below the surface. See Notes I and J.",
        ),
    }

    # TODO: construct above a bit more programatically (using template strings)?

    assert set(attrs) == set(columns)

    # For xarray dataset with depth dim
    attrs["soil_moisture_daily"] = (
        "daily average soil moisture",
        "1",
        "Average soil moisture, in fractional volumetric water content (m3 m-3). See Notes I and J.",
    )
    attrs["soil_temp_daily"] = (
        "daily average soil temperature",
        "degree_Celsius",
        "Average soil temperature, in degrees C. See Notes I and J.",
    )

    return (columns, attrs)


(DAILY_COLS, DAILY_ATTRS) = get_daily_col_info()


def read_daily(fp, *, cat: bool = False) -> pd.DataFrame:
    """Read a daily CRN file.

    For example:
    https://www.ncei.noaa.gov/pub/data/uscrn/products/daily01/2010/CRND0103-2010-CO_Boulder_14_W.txt
    """
    # columns, = get_daily_col_info()  # joblib has issues with the cached fn result
    columns = DAILY_COLS

    df = pd.read_csv(
        fp,
        delim_whitespace=True,
        header=None,
        names=columns,
        dtype={0: str, 2: str},
        parse_dates=["lst_date"],
        date_format=r"%Y%m%d",
        na_values=[-99999, -9999],
    )

    # Set soil moisture -99 to NaN
    sm_cols = df.columns[df.columns.str.startswith("soil_moisture_")]
    df[sm_cols] = df[sm_cols].replace(-99, np.nan)

    # Unknown datalogger version
    df["crx_vn"] = df["crx_vn"].replace("-9.000", np.nan)

    # Category cols?
    if cat:
        for col in ["sur_temp_daily_type"]:
            df[col] = df[col].astype("category")

    return df


def get_crn(
    years: int | Iterable[int] | None = None,
    *,
    n_jobs: int | None = -2,
    cat: bool = False,
    dropna: bool = False,
) -> pd.DataFrame:
    """Get daily CRN data.

    Info: https://www.ncei.noaa.gov/access/crn/qcdatasets.html

    Data: https://www.ncei.noaa.gov/pub/data/uscrn/products/daily01/
    """
    from itertools import chain

    from joblib import Parallel, delayed

    base_url = "https://www.ncei.noaa.gov/pub/data/uscrn/products/daily01"

    now = datetime.datetime.now(datetime.timezone.utc)

    # Get available years from the main page
    # e.g. `>2000/<`
    r = requests.get(f"{base_url}/")
    r.raise_for_status()
    available_years: list[int] = [int(s) for s in re.findall(r">([0-9]{4})/?<", r.text)]

    if isinstance(years, int):
        years = [years]

    if years is None:
        years = available_years[:]

    # Discover files
    print("Discovering files...")

    def get_year_urls(year):
        if year not in available_years:
            raise ValueError(
                f"year {year} not in detected available CRN years {available_years}"
            )

        # Get filenames from the year page
        # e.g. `>CRND0103-2020-TX_Palestine_6_WNW.txt<`
        url = f"{base_url}/{year}/"
        r = requests.get(url)
        r.raise_for_status()
        fns = re.findall(r">(CRN[a-zA-Z0-9\-_]*\.txt)<", r.text)
        if not fns:
            warnings.warn(
                f"no CRN files found for year {year} (url {url})", stacklevel=2
            )

        return (f"{base_url}/{year}/{fn}" for fn in fns)

    pool = ThreadPool(processes=min(len(years), 10))
    urls = list(chain.from_iterable(pool.imap(get_year_urls, years)))
    pool.close()

    print(f"{len(urls)} files found")
    print(urls[0])
    print("...")
    print(urls[-1])

    print("Reading files...")
    dfs = Parallel(n_jobs=n_jobs, verbose=10)(delayed(read_daily)(url) for url in urls)

    df = pd.concat(dfs, axis="index", ignore_index=True, copy=False)

    # Drop rows where all data cols are missing data?
    site_cols = [
        "wban",
        "lst_date",
        "crx_vn",
        "longitude",
        "latitude",
    ]
    assert set(site_cols) < set(df)
    data_cols = [c for c in df.columns if c not in site_cols]
    if dropna:
        df = df.dropna(subset=data_cols, how="all").reset_index(drop=True)
        if df.empty:
            warnings.warn(
                "CRN dataframe empty after dropping missing data rows", stacklevel=2
            )

    # Category cols?
    if cat:
        for col in ["sur_temp_daily_type"]:
            df[col] = df[col].astype("category")

    df.attrs.update(created=now)

    return df


if __name__ == "__main__":
    import xarray as xr

    # meta = load_meta(cat=True)
    # df = get_crn(2020, cat=True)

    fn = "crn_2020.parquet.gz"
    # df.to_parquet(fn, engine="fastparquet", compression="gzip")
    dfr = pd.read_parquet(fn, engine="fastparquet")

    #
    # xarray
    #

    df = dfr
    ds = (
        df.set_index(["wban", "lst_date"])
        .to_xarray()
        .swap_dims(wban="site")
        .set_coords(["latitude", "longitude"])
        .rename(lst_date="time")
    )
    # Combine vertically resolved variables
    for pref in ["soil_moisture_", "soil_temp_"]:
        vns = list(df.columns[df.columns.str.startswith(pref)])
        depths = sorted(float(vn.split("_")[2]) for vn in vns)
        vn_new_parts = vns[0].split("_")
        del vn_new_parts[2]
        vn_new = "_".join(vn_new_parts)

        if "depth" not in ds:
            ds["depth"] = (
                "depth",
                depths,
                {"long_name": "depth below surface", "units": "cm"},
            )

        vns_ = [f"{pref}{d:.0f}_daily" for d in depths]  # ensure sorted correctly
        assert set(vns) == set(vns_)

        # New var
        ds[vn_new] = xr.concat([ds[vn] for vn in vns_], dim="depth")
        ds = ds.drop_vars(vns_)

    # float32
    for vn in ds.data_vars:  # leave coords
        if pd.api.types.is_float_dtype(ds[vn].dtype) and ds[vn].dtype != np.float32:
            ds[vn] = ds[vn].astype(np.float32)

    # attrs
    for vn in ds.variables:
        attrs = DAILY_ATTRS.get(vn)
        if attrs is None:
            if vn not in {"time", "depth"}:
                warnings.warn(f"no attrs for {vn}")
            continue
        (long_name, units, description) = attrs
        ds[vn].attrs.update(long_name=long_name, units=units, description=description)
    ds["time"].attrs.update(description=DAILY_ATTRS["lst_date"][2])

    # lat/lon don't vary in time
    lat0 = ds["latitude"].isel(time=0)
    lon0 = ds["longitude"].isel(time=0)
    assert (ds["latitude"] == lat0).all()
    assert (ds["longitude"] == lon0).all()
    ds["latitude"] = lat0
    ds["longitude"] = lon0

    # save
    encoding = {vn: {"zlib": True, "complevel": 1} for vn in ds.data_vars if pd.api.types.is_float_dtype(ds[vn].dtype)}
    ds.to_netcdf("crn_2020.nc", encoding=encoding)
