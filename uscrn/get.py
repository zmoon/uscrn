from __future__ import annotations

import datetime
import re
import warnings
from collections.abc import Iterable
from multiprocessing.pool import ThreadPool

import numpy as np
import pandas as pd
import requests
import xarray as xr

from .attrs import DEFAULT_WHICH as _DEFAULT_WHICH
from .attrs import VALID_WHICHS as _VALID_WHICHS
from .attrs import get_col_info

_HOURLY = get_col_info("hourly")
_DAILY = get_col_info("daily")


def load_meta(*, cat: bool = False) -> pd.DataFrame:
    """Load the station metadata table.

    Parameters
    ----------
    cat
        Convert some columns to pandas categorical type.

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


def read_hourly(fp, *, cat: bool = False) -> pd.DataFrame:
    """Read an hourly CRN file.

    For example:
    https://www.ncei.noaa.gov/pub/data/uscrn/products/hourly02/2019/CRNH0203-2019-CO_Boulder_14_W.txt

    Parameters
    ----------
    cat
        Convert some columns to pandas categorical type.
    """
    df = pd.read_csv(
        fp,
        delim_whitespace=True,
        header=None,
        names=_HOURLY.names,
        dtype=_HOURLY.dtypes,
        parse_dates={"utc_time_": ["utc_date", "utc_time"], "lst_time_": ["lst_date", "lst_time"]},
        date_format=r"%Y%m%d %H%M",
        na_values=["-99999", "-9999"],
    )
    df = df.rename(columns={"utc_time_": "utc_time", "lst_time_": "lst_time"})

    # Set soil moisture -99 to NaN
    sm_cols = df.columns[df.columns.str.startswith("soil_moisture_")]
    df[sm_cols] = df[sm_cols].replace(-99, np.nan)

    # Unknown datalogger version
    df["crx_vn"] = df["crx_vn"].replace("-9.000", np.nan)

    # Category cols?
    if cat:
        for col in ["sur_temp_type"]:
            df[col] = df[col].astype(pd.CategoricalDtype(categories=["R", "C", "U"], ordered=False))
        for col in df.columns:
            if col.endswith("_flag"):
                df[col] = df[col].astype(pd.CategoricalDtype(categories=["0", "3"], ordered=False))

    df.attrs.update(which="hourly")

    return df


def read_daily(fp, *, cat: bool = False) -> pd.DataFrame:
    """Read a daily CRN file.

    For example:
    https://www.ncei.noaa.gov/pub/data/uscrn/products/daily01/2019/CRND0103-2019-CO_Boulder_14_W.txt

    Parameters
    ----------
    cat
        Convert some columns to pandas categorical type.
    """
    df = pd.read_csv(
        fp,
        delim_whitespace=True,
        header=None,
        names=_DAILY.names,
        dtype=_DAILY.dtypes,
        parse_dates=["lst_date"],
        date_format=r"%Y%m%d",
        na_values=["-99999", "-9999"],
    )

    # Set soil moisture -99 to NaN
    sm_cols = df.columns[df.columns.str.startswith("soil_moisture_")]
    df[sm_cols] = df[sm_cols].replace(-99, np.nan)

    # Unknown datalogger version
    df["crx_vn"] = df["crx_vn"].replace("-9.000", np.nan)

    # Category cols?
    if cat:
        for col in ["sur_temp_daily_type"]:
            df[col] = df[col].astype(pd.CategoricalDtype(categories=["R", "C", "U"], ordered=False))

    df.attrs.update(which="daily")

    return df


_which_to_reader = {
    "hourly": read_hourly,
    "daily": read_daily,
}


def get_crn(
    years: int | Iterable[int] | None = None,
    which: _VALID_WHICHS = _DEFAULT_WHICH,
    *,
    n_jobs: int | None = -2,
    cat: bool = False,
    dropna: bool = False,
) -> pd.DataFrame:
    """Get daily CRN data.

    * Home page: https://www.ncei.noaa.gov/access/crn/
    * Info: https://www.ncei.noaa.gov/access/crn/qcdatasets.html
    * Data: https://www.ncei.noaa.gov/pub/data/uscrn/products/daily01/

    Parameters
    ----------
    years
        Year(s) to get data for. If ``None`` (default), get all available years.
    n_jobs
        Number of parallel joblib jobs to use for loading the individual files.
        The default is ``-2``, which means to use one less than joblib's detected max.
    cat
        Convert some columns to pandas categorical type.
    dropna
        Drop rows where all data cols are missing data.
    """
    from itertools import chain

    from joblib import Parallel, delayed

    from .attrs import load_attrs, validate_which

    validate_which(which)

    attrs = load_attrs()

    base_url = attrs[which]["base_url"]

    # Get available years from the main page
    # e.g. `>2000/<`
    r = requests.get(f"{base_url}/")
    r.raise_for_status()
    available_years: list[int] = [int(s) for s in re.findall(r">([0-9]{4})/?<", r.text)]

    years_: list[int]
    if isinstance(years, int):
        years_ = [years]
    elif years is None:
        years_ = available_years[:]
    else:
        years = list(years)

    # Discover files
    print("Discovering files...")

    def get_year_urls(year):
        if year not in available_years:
            raise ValueError(f"year {year} not in detected available CRN years {available_years}")

        # Get filenames from the year page
        # e.g. `>CRND0103-2020-TX_Palestine_6_WNW.txt<`
        url = f"{base_url}/{year}/"
        r = requests.get(url)
        r.raise_for_status()
        fns = re.findall(r">(CRN[a-zA-Z0-9\-_]*\.txt)<", r.text)
        if not fns:
            warnings.warn(f"no CRN files found for year {year} (url {url})", stacklevel=2)

        return (f"{base_url}/{year}/{fn}" for fn in fns)

    pool = ThreadPool(processes=min(len(years_), 10))
    urls = list(chain.from_iterable(pool.imap(get_year_urls, years_)))
    pool.close()

    print(f"{len(urls)} files found")
    print(urls[0])
    print("...")
    print(urls[-1])

    print("Reading files...")
    read = _which_to_reader[which]
    dfs = Parallel(n_jobs=n_jobs, verbose=10)(delayed(read)(url) for url in urls)

    df = pd.concat(dfs, axis="index", ignore_index=True, copy=False)

    # Drop rows where all data cols are missing data?
    non_data_col_cands = [
        "wban",
        "lst_date",
        "lst_time",
        "utc_time",
        "crx_vn",
        "longitude",
        "latitude",
    ]
    non_data_cols = [c for c in non_data_col_cands if c in df]
    assert set(non_data_cols) < set(df)
    data_cols = [c for c in df.columns if c not in non_data_cols]
    if dropna:
        df = df.dropna(subset=data_cols, how="all").reset_index(drop=True)
        if df.empty:
            warnings.warn("CRN dataframe empty after dropping missing data rows", stacklevel=2)

    # Category cols?
    if cat:
        for col in ["sur_temp_daily_type"]:
            df[col] = df[col].astype("category")

    now = datetime.datetime.now(datetime.timezone.utc)
    df.attrs.update(which=which, created=now, source=base_url)

    return df


def to_xarray(df: pd.DataFrame, which: _VALID_WHICHS | None = None) -> xr.Dataset:
    """Convert to an xarray dataset."""
    from .attrs import load_attrs, validate_which

    if which is None:
        if "which" not in df.attrs:
            raise NotImplementedError(
                "Guessing `which` when not present in the df attr dict is not implemented. "
                "Please specify."
            )
        which = df.attrs["which"]

    validate_which(which)

    info = load_attrs()
    var_attrs = info[which]["columns"]
    base_url = info[which]["base_url"]
    time_var = info[which]["time_var"]

    ds = (
        df.set_index(["wban", time_var])
        .to_xarray()
        .swap_dims(wban="site")
        .set_coords(["latitude", "longitude"])
        .rename({time_var: "time"})
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

    # var attrs
    for vn in ds.variables:
        assert isinstance(vn, str)
        attrs = var_attrs.get(vn)
        if attrs is None:
            if vn not in {"time", "depth"}:
                warnings.warn(f"no attrs for {vn}")
            continue
        attrs_ = {
            k: attrs[k] for k in ["long_name", "units", "description"] if attrs[k] is not None
        }
        ds[vn].attrs.update(attrs_)
    ds["time"].attrs.update(description=var_attrs[time_var]["description"])

    # lat/lon don't vary in time
    lat0 = ds["latitude"].isel(time=0)
    lon0 = ds["longitude"].isel(time=0)
    assert (ds["latitude"] == lat0).all()
    assert (ds["longitude"] == lon0).all()
    ds["latitude"] = lat0
    ds["longitude"] = lon0

    # ds attrs
    now = datetime.datetime.now(datetime.timezone.utc)
    unique_years = sorted(df[time_var].dt.year.unique())
    if len(unique_years) == 1:
        s_years = str(unique_years[0])
    else:
        s_years = f"{unique_years[0]}--{unique_years[-1]}"
    ds.attrs["title"] = f"U.S. Climate Reference Network (USCRN) | {which} | {s_years}"
    ds.attrs["created"] = str(now)
    ds.attrs["source"] = base_url

    return ds
