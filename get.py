import datetime
import re
import warnings

import numpy as np
import pandas as pd
import requests


def load_meta(*, cat=False):
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


def get_crn(days, *, use_cache=True):
    """Get daily soil (and vegetation?) CRN data for `days`.

    Info: https://www.ncei.noaa.gov/access/crn/qcdatasets.html

    Data: https://www.ncei.noaa.gov/pub/data/uscrn/products/daily01/
    """
    days = pd.DatetimeIndex(days)

    # Get metadata
    # "This file contains the following three lines: Field Number, Field Name and Unit of Measure."
    base_url = "https://www.ncei.noaa.gov/pub/data/uscrn/products/daily01"
    r = requests.get(
        f"{base_url}/headers.txt",
    )
    r.raise_for_status()
    lines = r.text.splitlines()
    assert len(lines) == 3
    nums = lines[0].split()
    columns = lines[1].split()
    assert len(nums) == len(columns)
    assert nums == [str(i + 1) for i in range(len(columns))]

    # Get available years from the main page
    # e.g. `>2000/<`
    r = requests.get(f"{base_url}/")
    r.raise_for_status()
    available_years: list[str] = re.findall(r">([0-9]{4})/?<", r.text)

    # Get files
    # TODO: parallelize using Dask or joblib
    dfs_per_year = []
    years = days.year.astype(str).unique()
    for year in years:
        if year not in available_years:
            raise ValueError(f"year {year} not in detected available CRN years {available_years}")

        cached_fp = CACHE_DIR / f"CRN_{year}.csv.gz"
        is_cached = cached_fp.is_file()

        if not is_cached or not use_cache:
            # Get filenames from the year page
            # e.g. `>CRND0103-2020-TX_Palestine_6_WNW.txt<`
            url = f"{base_url}/{year}/"
            r = requests.get(url)
            r.raise_for_status()
            fns = re.findall(r">(CRN[a-zA-Z0-9\-_]*\.txt)<", r.text)
            if not fns:
                warnings.warn(f"no CRN files found for year {year} (url {url})", stacklevel=2)

            dfs_per_file = []
            for fn in fns:
                url = f"{base_url}/{year}/{fn}"
                print(url)
                df = pd.read_csv(
                    url,
                    delim_whitespace=True,
                    names=columns,
                    parse_dates=["LST_DATE"],
                    infer_datetime_format=True,
                    na_values=[-99999, -9999.0],
                )
                dfs_per_file.append(df)
            df = pd.concat(dfs_per_file)
        else:
            # Read from cache
            df = pd.read_csv(cached_fp, index_col=0, parse_dates=["LST_DATE"])

        if not is_cached:
            df.to_csv(cached_fp)  # ~ 2 MB

        dfs_per_year.append(df)

    # Combined df
    site_cols = [
        "WBANNO",
        "LST_DATE",
        "CRX_VN",
        "LONGITUDE",
        "LATITUDE",
    ]
    assert set(site_cols) < set(df)
    data_cols = [c for c in df.columns if c not in site_cols]
    df = pd.concat(dfs_per_year).dropna(subset=data_cols, how="all").reset_index(drop=True)
    if df.empty:
        warnings.warn("CRN dataframe empty after dropping missing data rows", stacklevel=2)

    # Set soil moisture -99 to NaN
    sm_cols = df.columns[df.columns.str.startswith("SOIL_MOISTURE_")]
    df[sm_cols] = df[sm_cols].replace(-99, np.nan)

    # Select data at days
    df = df[df.LST_DATE.isin(days.floor("D"))].reset_index(drop=True)

    return df


if __name__ == "__main__":
    meta = load_meta(cat=True)
