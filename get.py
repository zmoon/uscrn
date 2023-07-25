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

    # TODO: descriptions

    return (columns,)


DAILY_COLS, = get_daily_col_info()


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
        parse_dates=["LST_DATE"],
        na_values=[-99999, -9999],
    )
    df.columns = df.columns.str.lower()
    df = df.rename(columns={"wbanno": "wban"})  # consistency with meta file

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
            raise ValueError(f"year {year} not in detected available CRN years {available_years}")

        # Get filenames from the year page
        # e.g. `>CRND0103-2020-TX_Palestine_6_WNW.txt<`
        url = f"{base_url}/{year}/"
        r = requests.get(url)
        r.raise_for_status()
        fns = re.findall(r">(CRN[a-zA-Z0-9\-_]*\.txt)<", r.text)
        if not fns:
            warnings.warn(f"no CRN files found for year {year} (url {url})", stacklevel=2)

        return (
            f"{base_url}/{year}/{fn}"
            for fn in fns
        )

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
            warnings.warn("CRN dataframe empty after dropping missing data rows", stacklevel=2)

    # Category cols?
    if cat:
        for col in ["sur_temp_daily_type"]:
            df[col] = df[col].astype("category")

    return df


if __name__ == "__main__":
    meta = load_meta(cat=True)
    df = get_crn(2020, cat=True)
