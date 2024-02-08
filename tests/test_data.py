import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from packaging.version import Version

import uscrn
from uscrn.data import _which_to_reader, get_data, load_meta, parse_fp, parse_url, read, to_xarray

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


def test_load_meta_cat():
    meta = load_meta(cat=True)
    cats = meta.select_dtypes("category")
    assert set(cats) == {"status", "operation", "network"}
    assert set(meta.status.cat.categories) == {
        "Commissioned",
        "Experimental",
        "Non-comissioned",
        "Test-site",
    }
    assert set(meta.operation.cat.categories) == {
        "Abandoned",
        "Closed",
        "Non-operational",
        "Operational",
    }
    assert set(meta.network.cat.categories) >= {"USCRN", "USRCRN"}


@pytest.mark.parametrize("which, url", EXAMPLE_URL.items())
def test_read(which, url):
    # NOTE: subhourly takes a bit
    df = getattr(uscrn, f"read_{which}")(url)
    assert len(df) > 0

    df_cat = getattr(uscrn, f"read_{which}")(url, cat=True)
    assert len(df_cat.select_dtypes("category").columns) > 0

    df_auto = read(url)
    assert df_auto.equals(df)

    ds = to_xarray(df)

    if which == "subhourly":
        assert "depth" not in ds.dims
    elif which == "hourly":
        assert "soil_moisture_10" in df
        assert "depth" in ds.dims
        assert {"soil_moisture", "soil_temp"} < set(ds.data_vars)
        assert "soil_moisture_10" not in ds
    elif which == "daily":
        assert "soil_moisture_10_daily" in df
        assert "depth" in ds.dims
        assert {"soil_moisture_daily", "soil_temp_daily"} < set(ds.data_vars)
        assert "soil_moisture_10_daily" not in ds
    elif which == "monthly":
        assert "depth" not in ds.dims
    else:
        raise AssertionError


def test_read_hourly_nrt():
    url = "https://www.ncei.noaa.gov/pub/data/uscrn/products/hourly02/updates/2024/CRN60H0203-202402082100.txt"
    df = uscrn.read_hourly_nrt(url)
    assert df.wban.iloc[0] == "03047"
    assert df.wban.iloc[-1] == "96409"


def test_read_daily_nrt():
    url = "https://www.ncei.noaa.gov/pub/data/uscrn/products/daily01/updates/2024/CRND0103-202402072359.txt"
    df = uscrn.read_daily_nrt(url)
    assert df.wban.iloc[0] == "03047"
    assert df.wban.iloc[-1] == "96409"


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


def test_parse_fp_bad():
    with pytest.raises(ValueError, match="^Unknown USCRN file type."):
        parse_fp("asdf")


@pytest.mark.parametrize("which", uscrn.attrs.WHICHS)
def test_get(which):
    df = get_data(2019 if which != "monthly" else None, which=which, n_jobs=N)
    assert df.wban.nunique() == N

    ds = to_xarray(df)
    if which == "monthly":
        assert ds.title.startswith(f"U.S. Climate Reference Network (USCRN) | {which} | 2018--")
    else:
        assert ds.title == f"U.S. Climate Reference Network (USCRN) | {which} | 2019"
    assert set(np.unique(ds.time.dt.year)) >= {2019}

    # Check no -99s and such remain
    for vn in ds.data_vars:
        da = ds[vn]
        is_float = pd.api.types.is_float_dtype(da.dtype)
        is_string = pd.api.types.is_string_dtype(da.dtype)
        if is_float:
            if da.units == "degree_Celsius":
                with xr.set_options(keep_attrs=True):
                    da += 273.15
                    da.attrs.update(units="K")
            if which == "subhourly" and vn == "relative_humidity":
                da = da.where(ds.rh_flag == "0")
            da_min = da.min()
            if np.isnan(da_min):
                warnings.warn(f"All NaN {which}.{vn}")
                continue
            assert da_min > -99
            if da_min < 0:
                warnings.warn(
                    f"Negative values found in {which}.{vn}:\n"
                    f"{da.where(da < 0).to_series().value_counts()}"
                )
        elif is_string:
            assert da.to_series().str.startswith("-9").sum() == 0


def test_get_bad_year():
    with pytest.raises(ValueError, match="^year 1900 not in detected available USCRN years"):
        get_data(1900)

    with pytest.raises(ValueError, match="^years should not be empty"):
        get_data([])


def test_get_years_default():
    df = get_data(None, "daily", n_jobs=N)
    # With get-cap set as we have it, should get just first year data (2000)
    unique_years = df["lst_date"].dt.year.unique()
    assert unique_years.size == 1
    assert unique_years[0] == 2000


def test_to_xarray_no_which_attr():
    with pytest.raises(NotImplementedError, match="^Guessing `which`"):
        to_xarray(pd.DataFrame())


@pytest.mark.parametrize("engine", ["pyarrow", "fastparquet"])
def test_df_parquet_roundtrip(tmp_path, engine):
    df = get_data(2019, which="daily", n_jobs=N, cat=True)
    assert df.attrs != {}

    fp = tmp_path / "test.parquet"
    df.to_parquet(fp, index=False)
    df2 = pd.read_parquet(fp, engine=engine)

    assert df.equals(df2), "data same"

    if Version(pd.__version__) < Version("2.1"):
        assert df2.attrs == {}, "no preservation before pandas 2.1"
    else:
        assert df.attrs is not df2.attrs
        if engine == "fastparquet":
            assert df2.attrs == {}
        else:
            assert df.attrs == df2.attrs
