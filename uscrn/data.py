"""
Load USCRN data from NCEI.
"""

from __future__ import annotations

import datetime
import warnings
from collections.abc import Iterable
from typing import Any, Literal, NamedTuple

import numpy as np
import pandas as pd
import xarray as xr

from ._util import retry

_GET_CAP: int | None = None
"""Restrict how many files to load, for testing purposes."""


@retry
def load_meta(*, cat: bool = False) -> pd.DataFrame:
    """Load the station metadata table.

    https://www.ncei.noaa.gov/pub/data/uscrn/products/stations.tsv

    Parameters
    ----------
    cat
        Convert some columns to pandas categorical type.

    See Also
    --------
    :doc:`/examples/stations`
        Notebook example demonstrating using this function and examining its results.
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
    nrt: bool
    group: str
    state: str | None
    location: str | None
    vector: str | None
    time: datetime.datetime | None


def parse_fp(fp: str) -> _ParseRes:
    """Parse USCRN file path."""
    from pathlib import Path

    p = Path(fp)

    nrt: bool = False
    state: str | None = None
    location: str | None = None
    vector: str | None = None
    time: datetime.datetime | None = None
    if p.name.startswith("CRNS0"):
        which = "subhourly"
    elif p.name.startswith("CRNH0"):
        which = "hourly"
    elif p.name.startswith("CRN60H"):
        which = "hourly"
        nrt = True
        group, s_dt = p.stem.split("-")
        time = datetime.datetime.strptime(s_dt, r"%Y%m%d%H%M")
    elif p.name.startswith("CRND0"):
        which = "daily"
        try:
            group, s_dt = p.stem.split("-")
            time = datetime.datetime.strptime(s_dt, r"%Y%m%d%H%M")
        except ValueError:
            pass
        else:
            nrt = True
    elif p.name.startswith("CRNM0"):
        which = "monthly"
    else:
        raise ValueError(
            "Unknown USCRN file type. Expected the name to start with `CRN{S,H,D,M}0` or `CRN60H`. "
            f"Got: {p.name!r}."
        )

    if not nrt:
        parts = p.stem.split("_")
        group = parts[0]
        state = group.split("-")[-1]
        location = " ".join(parts[1:-2])
        vector = " ".join(parts[-2:])

    return _ParseRes(
        fp=fp,
        which=which,
        nrt=nrt,
        group=group,
        state=state,
        location=location,
        vector=vector,
        time=time,
    )


def parse_url(url: str) -> _ParseRes:
    """Parse USCRN file path from URL."""
    from urllib.parse import urlsplit

    return parse_fp(urlsplit(url).path)


@retry
def read_subhourly(fp, *, cat: bool = False) -> pd.DataFrame:
    """Read a subhourly USCRN file.

    For example:
    https://www.ncei.noaa.gov/pub/data/uscrn/products/subhourly01/2019/CRNS0101-05-2019-CO_Boulder_14_W.txt

    Parameters
    ----------
    cat
        Convert some columns to pandas categorical type.
    """
    from .attrs import get_col_info

    col_info = get_col_info("subhourly")
    dtype = col_info.dtypes.copy()
    for col in ["utc_date", "utc_time", "lst_date", "lst_time"]:
        dtype[col] = str
    df = pd.read_csv(
        fp,
        sep=r"\s+",
        header=None,
        names=col_info.names,
        dtype=dtype,
        na_values=["-99999", "-9999"],
    )
    df["utc_time"] = pd.to_datetime(df["utc_date"] + df["utc_time"], format=r"%Y%m%d%H%M")
    df["lst_time"] = pd.to_datetime(df["lst_date"] + df["lst_time"], format=r"%Y%m%d%H%M")
    df = df.drop(columns=["utc_date", "lst_date"])

    # Set soil moisture -99 to NaN
    sm_cols = df.columns[df.columns.str.startswith("soil_moisture_")]
    df[sm_cols] = df[sm_cols].replace(-99, np.nan)

    # Unknown datalogger version
    df["crx_vn"] = df["crx_vn"].replace("-9.000", np.nan)

    # Lower precision floats
    cols = ["wind_1_5"]
    df[cols] = df[cols].replace(-99, np.nan)

    # Category cols?
    if cat:
        for col, cats in col_info.categorical.items():
            df[col] = df[col].astype(pd.CategoricalDtype(categories=cats, ordered=False))

    df.attrs.update(which="subhourly")

    return df


@retry
def read_hourly(fp, *, cat: bool = False, **kwargs) -> pd.DataFrame:
    """Read an hourly USCRN file.

    For example:
    https://www.ncei.noaa.gov/pub/data/uscrn/products/hourly02/2019/CRNH0203-2019-CO_Boulder_14_W.txt

    Parameters
    ----------
    cat
        Convert some columns to pandas categorical type.
    **kwargs
        Additional keyword arguments to pass to :func:`pandas.read_csv`.
    """
    from .attrs import get_col_info

    col_info = get_col_info("hourly")
    dtype = col_info.dtypes.copy()
    for col in ["utc_date", "utc_time", "lst_date", "lst_time"]:
        dtype[col] = str
    df = pd.read_csv(
        fp,
        sep=r"\s+",
        header=None,
        names=col_info.names,
        dtype=dtype,
        na_values=["-99999", "-9999"],
        **kwargs,
    )
    df["utc_time"] = pd.to_datetime(df["utc_date"] + df["utc_time"], format=r"%Y%m%d%H%M")
    df["lst_time"] = pd.to_datetime(df["lst_date"] + df["lst_time"], format=r"%Y%m%d%H%M")
    df = df.drop(columns=["utc_date", "lst_date"])

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


def read_hourly_nrt(fp, *, cat: bool = False) -> pd.DataFrame:
    """Read an hourly NRT USCRN file.

    For example:
    https://www.ncei.noaa.gov/pub/data/uscrn/products/hourly02/updates/2024/CRN60H0203-202402082100.txt

    Parameters
    ----------
    cat
        Convert some columns to pandas categorical type.
    """
    # Two of the five header lines use `\r\r\n` ending instead of `\n`.
    # The last data line has normal `\n` ending, but then the file ends with `\r\r\n\x03`.
    # `\x03` is the end-of-text control character (^C).
    # The 'c' engine does not support `skipfooter`.
    return read_hourly(fp, cat=cat, skiprows=5, skipfooter=1, engine="python")


@retry
def read_daily(fp, *, cat: bool = False, **kwargs) -> pd.DataFrame:
    """Read a daily USCRN file.

    For example:
    https://www.ncei.noaa.gov/pub/data/uscrn/products/daily01/2019/CRND0103-2019-CO_Boulder_14_W.txt

    Parameters
    ----------
    cat
        Convert some columns to pandas categorical type.
    **kwargs
        Additional keyword arguments to pass to :func:`pandas.read_csv`.
    """
    from .attrs import get_col_info

    col_info = get_col_info("daily")
    df = pd.read_csv(
        fp,
        sep=r"\s+",
        header=None,
        names=col_info.names,
        dtype=col_info.dtypes,
        parse_dates=["lst_date"],
        date_format=r"%Y%m%d",
        na_values=["-99999", "-9999"],
        **kwargs,
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


def read_daily_nrt(fp, *, cat: bool = False) -> pd.DataFrame:
    """Read a daily NRT USCRN file.

    For example:
    https://www.ncei.noaa.gov/pub/data/uscrn/products/daily01/updates/2024/CRND0103-202402072359.txt

    Parameters
    ----------
    cat
        Convert some columns to pandas categorical type.
    """
    # See notes in `read_hourly_nrt`.
    return read_daily(fp, cat=cat, skiprows=5, skipfooter=1, engine="python")


@retry
def read_monthly(fp, *, cat: bool = False) -> pd.DataFrame:
    """Read a monthly USCRN file.

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
        sep=r"\s+",
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
    "subhourly": read_subhourly,
    "hourly": read_hourly,
    "daily": read_daily,
    "monthly": read_monthly,
}


def read(fp, *, cat: bool = False) -> pd.DataFrame:
    """Read a USCRN file, auto-detecting which reader to use based on file name.

    See Also
    --------
    read_daily
    read_hourly
    read_monthly
    read_subhourly
    """
    from .attrs import validate_which

    res = parse_url(fp)
    validate_which(res.which)

    return _which_to_reader[res.which](fp, cat=cat)


def get_data(
    years: int | Iterable[int] | None = None,
    which: Literal["subhourly", "hourly", "daily", "monthly"] = "daily",
    *,
    n_jobs: int | None = -2,
    cat: bool = False,
    dropna: bool = False,
) -> pd.DataFrame:
    """Get USCRN data.

    * Home page: https://www.ncei.noaa.gov/access/crn/
    * Info: https://www.ncei.noaa.gov/access/crn/qcdatasets.html
    * Data: https://www.ncei.noaa.gov/pub/data/uscrn/products/

    Variable and dataset metadata are included in the ``.attrs`` dict.
    These will be preserved if you have pandas v2.1+
    and save the dataframe to Parquet format with the PyArrow engine.

    >>> df.to_parquet('crn.parquet', engine='pyarrow')

    Parameters
    ----------
    years
        Year(s) to get data for. If ``None`` (default), get all available years.
        If `which` is ``'monthly'``, `years` is ignored and you always get all available years.
    which
        Which dataset.
    n_jobs
        Number of parallel joblib jobs to use for loading the individual files.
        The default is ``-2``, which means to use one less than joblib's detected max.
    cat
        Convert some columns to pandas categorical type.
    dropna
        Drop rows where all data cols are missing data.

    See Also
    --------
    :doc:`/examples/daily`
        Notebook example demonstrating using this function to get a year of daily data.
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
    # TODO: could cache available years and files like the docs pages
    print("Discovering files...")

    @retry
    def get_main_list_page():
        r = requests.get(f"{base_url}/", timeout=10)
        r.raise_for_status()
        return r.text

    main_list_page = get_main_list_page()

    urls: list[str]
    if which == "monthly":
        # No year subdirectories
        fns = re.findall(r">(CRN[a-zA-Z0-9\-_]*\.txt)<", main_list_page)
        urls = [f"{base_url}/{fn}" for fn in fns]
    else:
        # Year subdirectories
        from multiprocessing.pool import ThreadPool

        available_years: list[int] = [int(s) for s in re.findall(r">([0-9]{4})/?<", main_list_page)]

        years_: list[int]
        if isinstance(years, int):
            years_ = [years]
        elif years is None:
            years_ = available_years[:]
        else:
            years_ = list(years)
            if len(years_) == 0:
                raise ValueError("years should not be empty")

        @retry
        def get_year_urls(year):
            if year not in available_years:
                raise ValueError(
                    f"year {year} not in detected available USCRN years {available_years}"
                )

            # Get filenames from the year page
            # e.g. `>CRND0103-2020-TX_Palestine_6_WNW.txt<`
            url = f"{base_url}/{year}/"
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            fns = re.findall(r">(CRN[a-zA-Z0-9\-_]*\.txt)<", r.text)
            if not fns:  # pragma: no cover
                warnings.warn(f"no USCRN files found for year {year} (url {url})", stacklevel=2)

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
        if df.empty:  # pragma: no cover
            warnings.warn("USCRN dataframe empty after dropping missing data rows")

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


def get_nrt_data(
    period: tuple[Any, Any],
    which: Literal["hourly", "daily"] = "hourly",
    *,
    n_jobs: int | None = -2,
    cat: bool = False,
) -> pd.DataFrame:
    """Get USCRN NRT data.

    Parameters
    ----------
    period
        2-tuple expressing the (inclusive) time bounds of the period of interest (UTC).
        Use ``None`` to indicate an open-ended bound.
    which
        Which dataset.
        Only hourly and daily are supported for NRT.
    n_jobs
        Number of parallel joblib jobs to use for loading the individual files.
        The default is ``-2``, which means to use one less than joblib's detected max.
    cat
        Convert some columns to pandas categorical type.
    """
    import re
    from urllib.parse import urlsplit

    import requests
    from joblib import Parallel, delayed

    from .attrs import get_col_info, load_attrs

    if which == "hourly":
        read = read_hourly_nrt
    elif which == "daily":
        read = read_daily_nrt
    else:
        raise ValueError(
            f"Invalid dataset identifier: {which!r}. Valid identifiers are: ('hourly', 'daily')."
        )

    a, b = period
    try:
        if a is not None:
            a = pd.to_datetime(a)
        if b is not None:
            b = pd.to_datetime(b)
        assert (a is None or isinstance(a, pd.Timestamp)) and (
            b is None or isinstance(b, pd.Timestamp)
        )
    except Exception as e:
        raise TypeError(
            "Expected period bounds to be None or coercible to pandas.Timestamp, "
            f"got ({a!r}, {b!r})."
        ) from e

    if not isinstance(a, pd.Timestamp):
        raise NotImplementedError("Unbounded left not implemented yet.")
    if b is None:
        b = pd.Timestamp(year=3000, month=1, day=1)

    stored_attrs = load_attrs()
    col_info = get_col_info(which)

    base_url = stored_attrs[which]["base_url"]

    # Get available years from the main page
    # e.g. `>2024/<`
    print("Discovering files...")

    @retry
    def get_main_list_page():
        r = requests.get(f"{base_url}/updates/", timeout=10)
        r.raise_for_status()
        return r.text

    main_list_page = get_main_list_page()

    available_years: list[int] = [int(s) for s in re.findall(r">([0-9]{4})/?<", main_list_page)]

    @retry
    def get_year_urls(year):
        if year not in available_years:
            raise ValueError(f"year {year} not in detected available USCRN years {available_years}")

        # Get filenames from the year page
        # e.g. `>CRN60H0203-202402082100.txt<`
        url = f"{base_url}/updates/{year}/"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        fns = re.findall(r">(CRN[a-zA-Z0-9\-_]*\.txt)<", r.text)
        if not fns:  # pragma: no cover
            warnings.warn(f"no USCRN files found for year {year} (url {url})", stacklevel=2)

        return (f"{base_url}/updates/{year}/{fn}" for fn in fns)

    # Get available files for years in period
    urls = []
    for year in available_years:
        if year < a.year or (b is not None and year > b.year):
            continue
        urls.extend(get_year_urls(year))

    # Remove files outside of period
    urls_ = []
    for url in urls:
        parts = urlsplit(url)
        path = parts.path
        _, s = path.split("-")
        t = pd.to_datetime(s, format=r"%Y%m%d%H%M.txt")
        if a <= t <= b:
            urls_.append(url)
    urls = urls_

    # TODO: warn if period bounds are outside what is available?

    print(f"Found {len(urls)} file(s) to load")
    if len(urls) > 0:
        print(urls[0])
    if len(urls) > 2:
        print("...")
    if len(urls) > 1:
        print(urls[-1])

    print("Reading files...")
    dfs = Parallel(n_jobs=n_jobs, verbose=10)(delayed(read)(url) for url in urls)

    df = pd.concat(dfs, axis="index", ignore_index=True, copy=False)

    # Category cols?
    if cat:
        for col, cats in col_info.categorical.items():
            df[col] = df[col].astype(pd.CategoricalDtype(categories=cats, ordered=False))

    now = datetime.datetime.now(datetime.timezone.utc)
    title = f"U.S. Climate Reference Network (USCRN) | {which} | NRT"
    s_a = "" if a is None else str(a)
    s_b = "" if b is None else str(b)
    s_period = f"{s_a}--{s_b}"
    title += f" | {s_period}"

    df.attrs.update(
        which=which,
        title=title,
        created=str(now),
        source=base_url,
        attrs=col_info.attrs,
        notes=col_info.notes,
    )

    return df


def to_xarray(
    df: pd.DataFrame,
    which: Literal["subhourly", "hourly", "daily", "monthly"] | None = None,
) -> xr.Dataset:
    """Convert to xarray representation.

    Soil variables will be combined, with a soil depth dimension added,
    if applicable (hourly, daily).

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
