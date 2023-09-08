from pathlib import Path

import pandas as pd

from uscrn.get import _which_to_reader, load_meta, read_daily, read_hourly, to_xarray

HERE = Path(__file__).parent
DATA = HERE / "data"


def test_example_xr():
    # Parquet file created with:
    # df = get_crn(2020, cat=True)
    # df.to_parquet(fn, engine="fastparquet", compression="gzip")

    fn = DATA / "crn_2020.parquet.gz"
    dfr = pd.read_parquet(fn, engine="fastparquet")

    #
    # xarray
    #

    ds = to_xarray(dfr)  # noqa: F841

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
    df = read_hourly(
        "https://www.ncei.noaa.gov/pub/data/uscrn/products/hourly02/2019/CRNH0203-2019-CO_Boulder_14_W.txt"
    )
    assert len(df) > 0


def test_get_daily():
    df = read_daily(
        "https://www.ncei.noaa.gov/pub/data/uscrn/products/daily01/2019/CRND0103-2019-CO_Boulder_14_W.txt"
    )
    assert len(df) > 0


def test_which_to_reader():
    from uscrn.attrs import WHICHS

    assert _which_to_reader.keys() == set(WHICHS)
