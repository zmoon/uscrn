"""
Load CRN data from NCEI.
"""
from __future__ import annotations

import datetime
import warnings
from collections.abc import Iterable
from typing import Literal, NamedTuple

import numpy as np
import pandas as pd
import xarray as xr

_GET_CAP: int | None = None
"""Restrict how many files to load, for testing purposes."""


def load_meta(*, cat: bool = False) -> pd.DataFrame:
    """Load the station metadata table.

    https://www.ncei.noaa.gov/pub/data/uscrn/products/stations.tsv

    Parameters
    ----------
    cat
        Convert some columns to pandas categorical type.
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


class _ParseRes(NamedTuple):
    fp: str
    which: str
    group: str
    state: str
    location: str
    vector: str


def parse_fp(fp: str) -> _ParseRes:
    """Parse CRN file path."""
    from pathlib import Path

    p = Path(fp)

    if p.name.startswith("CRNS0"):
        which = "subhourly"
    elif p.name.startswith("CRNH0"):
        which = "hourly"
    elif p.name.startswith("CRND0"):
        which = "daily"
    elif p.name.startswith("CRNM0"):
        which = "monthly"
    else:
        raise ValueError(
            "Unknown CRN file type. Expected the name to start with `CRN{S,H,D,M}0`. "
            f"Got: {p.name!r}."
        )

    parts = p.stem.split("_")
    group = parts[0]
    state = group.split("-")[-1]
    location = " ".join(parts[1:-2])
    vector = " ".join(parts[-2:])

    return _ParseRes(fp=fp, which=which, group=group, state=state, location=location, vector=vector)


def parse_url(url: str) -> _ParseRes:
    """Parse CRN file path from URL."""
    from urllib.parse import urlsplit

    return parse_fp(urlsplit(url).path)


def read_hourly(fp, *, cat: bool = False) -> pd.DataFrame:
    """Read an hourly CRN file.

    For example:
    https://www.ncei.noaa.gov/pub/data/uscrn/products/hourly02/2019/CRNH0203-2019-CO_Boulder_14_W.txt

    Parameters
    ----------
    cat
        Convert some columns to pandas categorical type.
    """
    from .attrs import get_col_info

    col_info = get_col_info("hourly")
    df = pd.read_csv(
        fp,
        delim_whitespace=True,
        header=None,
        names=col_info.names,
        dtype=col_info.dtypes,
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
        for col, cats in col_info.categorical.items():
            df[col] = df[col].astype(pd.CategoricalDtype(categories=cats, ordered=False))

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
    from .attrs import get_col_info

    col_info = get_col_info("daily")
    df = pd.read_csv(
        fp,
        delim_whitespace=True,
        header=None,
        names=col_info.names,
        dtype=col_info.dtypes,
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
        for col, cats in col_info.categorical.items():
            df[col] = df[col].astype(pd.CategoricalDtype(categories=cats, ordered=False))

    df.attrs.update(which="daily")

    return df


def read_monthly(fp, *, cat: bool = False) -> pd.DataFrame:
    """Read a monthly CRN file.

    For example:
    https://www.ncei.noaa.gov/pub/data/uscrn/products/monthly01/CRNM0102-CO_Boulder_14_W.txt


    Note: Unlike the other datasets, for monthly there is only one file per site.

    Parameters
    ----------
    cat
        Convert some columns to pandas categorical type.
    """
    from .attrs import get_col_info

    col_info = get_col_info("monthly")
    df = pd.read_csv(
        fp,
        delim_whitespace=True,
        header=None,
        names=col_info.names,
        dtype=col_info.dtypes,
        parse_dates=["lst_yrmo"],
        date_format=r"%Y%m",
        na_values=["-99999", "-9999"],
    )

    # Set soil moisture -99 to NaN
    sm_cols = df.columns[df.columns.str.startswith("soil_moisture_")]
    df[sm_cols] = df[sm_cols].replace(-99, np.nan)

    # Unknown datalogger version
    df["crx_vn"] = df["crx_vn"].replace("-9.000", np.nan)

    # Category cols?
    if cat:
        for col, cats in col_info.categorical.items():
            df[col] = df[col].astype(pd.CategoricalDtype(categories=cats, ordered=False))

    df.attrs.update(which="monthly")

    return df


_which_to_reader = {
    "hourly": read_hourly,
    "daily": read_daily,
    "monthly": read_monthly,
}


def read(fp, *, cat: bool = False) -> pd.DataFrame:
    """Read a CRN file, auto-detecting which reader to use based on file name."""
    from .attrs import validate_which

    res = parse_url(fp)
    validate_which(res.which)

    return _which_to_reader[res.which](fp, cat=cat)


def get_data(
    years: int | Iterable[int] | None = None,
    which: Literal["hourly", "daily", "monthly"] = "daily",
    *,
    n_jobs: int | None = -2,
    cat: bool = False,
    dropna: bool = False,
) -> pd.DataFrame:
    """Get CRN data.

    * Home page: https://www.ncei.noaa.gov/access/crn/
    * Info: https://www.ncei.noaa.gov/access/crn/qcdatasets.html
    * Data: https://www.ncei.noaa.gov/pub/data/uscrn/products/

    Parameters
    ----------
    years
        Year(s) to get data for. If ``None`` (default), get all available years.
    which
        Which dataset.
    n_jobs
        Number of parallel joblib jobs to use for loading the individual files.
        The default is ``-2``, which means to use one less than joblib's detected max.
    cat
        Convert some columns to pandas categorical type.
    dropna
        Drop rows where all data cols are missing data.
    """
    import re
    from itertools import chain

    import requests
    from joblib import Parallel, delayed

    from .attrs import get_col_info, load_attrs, validate_which

    validate_which(which)

    if which == "monthly" and years is not None:
        warnings.warn("`years` ignored for monthly data.")

    stored_attrs = load_attrs()
    col_info = get_col_info(which)

    base_url = stored_attrs[which]["base_url"]

    # Get available years from the main page
    # e.g. `>2000/<`
    print("Discovering files...")
    r = requests.get(f"{base_url}/")
    r.raise_for_status()
    urls: list[str]
    if which == "monthly":
        # No year subdirectories
        fns = re.findall(r">(CRN[a-zA-Z0-9\-_]*\.txt)<", r.text)
        urls = [f"{base_url}/{fn}" for fn in fns]
    else:
        # Year subdirectories
        from multiprocessing.pool import ThreadPool

        available_years: list[int] = [int(s) for s in re.findall(r">([0-9]{4})/?<", r.text)]

        years_: list[int]
        if isinstance(years, int):
            years_ = [years]
        elif years is None:
            years_ = available_years[:]
        else:
            years_ = list(years)
            if len(years_) == 0:
                raise ValueError("years should be not be empty")

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
                warnings.warn(f"no CRN files found for year {year} (url {url})", stacklevel=2)

            return (f"{base_url}/{year}/{fn}" for fn in fns)

        pool = ThreadPool(processes=min(len(years_), 10))
        urls = list(chain.from_iterable(pool.imap(get_year_urls, years_)))
        pool.close()

    print(f"{len(urls)} files found")
    print(urls[0])
    print("...")
    print(urls[-1])

    if _GET_CAP is not None:
        urls = urls[:_GET_CAP]
        # TODO: random selection?
        print(f"Using the first {_GET_CAP} files only")

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
            warnings.warn("CRN dataframe empty after dropping missing data rows")

    # Category cols?
    if cat:
        for col, cats in col_info.categorical.items():
            df[col] = df[col].astype(pd.CategoricalDtype(categories=cats, ordered=False))

    now = datetime.datetime.now(datetime.timezone.utc)
    title = f"U.S. Climate Reference Network (USCRN) | {which}"
    if which == "monthly":
        unique_years = sorted(df[stored_attrs[which]["time_var"]].dt.year.unique())
        title += f" | {unique_years[0]}--{unique_years[-1]}"
    else:
        if len(years_) == 1:
            title += f" | {years_[0]}"
        else:
            title += f" | {years_[0]}--{years_[-1]}"
    df.attrs.update(
        which=which,
        title=title,
        created=str(now),
        source=base_url,
        attrs=col_info.attrs,  # NOTE: nested, may not survive storage roundtrip
        notes=col_info.notes,
    )

    return df


def to_xarray(
    df: pd.DataFrame,
    which: Literal["hourly", "daily", "monthly"] | None = None,
) -> xr.Dataset:
    """Convert to an xarray dataset.

    Parameters
    ----------
    df
        Input dataframe, created using :func:`get_data` or one of the readers.
    which
        Which dataset. Will attempt to guess by default. Specify to override this.
    """
    from .attrs import get_col_info, load_attrs, validate_which

    if which is None:
        if "which" not in df.attrs:
            raise NotImplementedError(
                "Guessing `which` when not present in the df attr dict is not implemented. "
                "Please specify."
            )
        which = df.attrs["which"]

    validate_which(which)

    stored_attrs = load_attrs()
    var_attrs = stored_attrs[which]["columns"]
    base_url = stored_attrs[which]["base_url"]
    time_var = stored_attrs[which]["time_var"]

    col_info = get_col_info(which)
    notes = col_info.notes

    ds = (
        df.set_index(["wban", time_var])
        .to_xarray()
        .swap_dims(wban="site")
        .set_coords(["latitude", "longitude"])
        .rename({time_var: "time"})
    )
    # Combine vertically resolved variables
    if which in {"hourly", "daily"}:
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

            # Ensure sorted correctly
            vns.sort(key=lambda vn: int(vn.split("_")[2]))

            # New var
            ds[vn_new] = xr.concat([ds[vn] for vn in vns], dim="depth")
            ds = ds.drop_vars(vns)

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

    # lat/lon shouldn't vary in time
    # But it seems like they are NaN if station not operating or something
    def first(da):
        def func(x):
            assert x.ndim == 1
            inds = np.where(~np.isnan(x))[0]
            if len(inds) == 0:
                warnings.warn(f"{da.name} all NaN for certain site")
                return np.nan
            else:
                i = inds[0]
                return x[i]

        return xr.apply_ufunc(func, da, input_core_dims=[["time"]], vectorize=True)

    lat0 = first(ds["latitude"])
    lon0 = first(ds["longitude"])
    ds["latitude"] = lat0
    ds["longitude"] = lon0

    # ds attrs
    now = datetime.datetime.now(datetime.timezone.utc)
    unique_years_init = sorted(df[time_var].dt.year.unique())
    unique_years = []
    for y in unique_years_init:
        t = ds.time.sel(time=str(y))
        if t.size == 1 and t == pd.Timestamp(year=y, month=1, day=1):
            continue
        unique_years.append(y)
    if len(unique_years) == 0:
        s_years = "?"
    elif len(unique_years) == 1:
        s_years = str(unique_years[0])
    else:
        s_years = f"{unique_years[0]}--{unique_years[-1]}"
    ds.attrs["title"] = f"U.S. Climate Reference Network (USCRN) | {which} | {s_years}"
    ds.attrs["created"] = str(now)
    ds.attrs["source"] = base_url
    ds.attrs["notes"] = notes

    return ds
