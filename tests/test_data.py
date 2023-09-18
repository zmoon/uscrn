from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import uscrn
from uscrn.data import (
    _which_to_reader,
    get_data,
    load_meta,
    parse_url,
    read_daily,
    read_hourly,
    to_xarray,
)

HERE = Path(__file__).parent
DATA = HERE / "data"

EXAMPLE_URL = {
    "subhourly": "https://www.ncei.noaa.gov/pub/data/uscrn/products/subhourly01/2019/CRNS0101-05-2019-CO_Boulder_14_W.txt",
    "hourly": "https://www.ncei.noaa.gov/pub/data/uscrn/products/hourly02/2019/CRNH0203-2019-CO_Boulder_14_W.txt",
    "daily": "https://www.ncei.noaa.gov/pub/data/uscrn/products/daily01/2019/CRND0103-2019-CO_Boulder_14_W.txt",
    "monthly": "https://www.ncei.noaa.gov/pub/data/uscrn/products/monthly01/CRNM0102-CO_Boulder_14_W.txt",
}

N = 2
uscrn.data._GET_CAP = N


def test_example_xr():
    # Parquet file created with:
    # df = get_data(2020, cat=True)
    # df.to_parquet(fn, engine="fastparquet", compression="gzip")

    fn = DATA / "crn_2020.parquet.gz"
    df = pd.read_parquet(fn, engine="fastparquet")
    if "which" not in df.attrs:
        import warnings

        warnings.warn("No 'which' attribute found on dataframe. Adding.")
        df.attrs["which"] = "daily"

    #
    # xarray
    #

    ds = to_xarray(df)  # noqa: F841

    # # save
    # encoding = {
    #     vn: {"zlib": True, "complevel": 1}
    #     for vn in ds.data_vars
    #     if pd.api.types.is_float_dtype(ds[vn].dtype)
    # }
    # ds.to_netcdf("crn_2020.nc", encoding=encoding)


def test_load_meta():
    meta = load_meta()
    assert len(meta) > 0
    assert meta[["country", "state"]].isnull().sum().sum() == 0


def test_get_hourly():
    df = read_hourly(EXAMPLE_URL["hourly"])
    assert len(df) > 0
    assert "soil_moisture_10" in df

    ds = to_xarray(df)
    assert {"soil_moisture", "soil_temp"} < set(ds.data_vars)
    assert "soil_moisture_10" not in ds


def test_get_daily():
    df = read_daily(EXAMPLE_URL["daily"])
    assert len(df) > 0
    assert "soil_moisture_10_daily" in df

    ds = to_xarray(df)
    assert {"soil_moisture_daily", "soil_temp_daily"} < set(ds.data_vars)
    assert "soil_moisture_10_daily" not in ds


def test_which_to_reader():
    from uscrn.attrs import WHICHS

    assert _which_to_reader.keys() == set(WHICHS)


@pytest.mark.parametrize("which, url", EXAMPLE_URL.items())
def test_parse_url(which, url):
    res = parse_url(url)
    assert url.endswith(res.fp)
    assert res.which == which
    assert res.state == "CO"
    assert res.location == "Boulder"
    assert res.vector == "14 W"


@pytest.mark.parametrize("which", uscrn.attrs.WHICHS)
def test_get(which):
    df = get_data(2019, which=which, n_jobs=N)
    assert df.wban.nunique() == N

    ds = to_xarray(df)
    if which == "monthly":
        assert ds.title.startswith(f"U.S. Climate Reference Network (USCRN) | {which}")
    else:
        assert ds.title == f"U.S. Climate Reference Network (USCRN) | {which} | 2019"
    assert set(np.unique(ds.time.dt.year)) >= {2019}
