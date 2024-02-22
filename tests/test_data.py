import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from packaging.version import Version

import uscrn
from uscrn.data import (
    _which_to_reader,
    auto_title,
    get_data,
    get_nrt_data,
    load_meta,
    parse_fp,
    parse_url,
    read,
    to_xarray,
)

HERE = Path(__file__).parent
DATA = HERE / "data"

EXAMPLE_URL = {
    "subhourly": "https://www.ncei.noaa.gov/pub/data/uscrn/products/subhourly01/2019/CRNS0101-05-2019-CO_Boulder_14_W.txt",
    "hourly": "https://www.ncei.noaa.gov/pub/data/uscrn/products/hourly02/2019/CRNH0203-2019-CO_Boulder_14_W.txt",
    "daily": "https://www.ncei.noaa.gov/pub/data/uscrn/products/daily01/2019/CRND0103-2019-CO_Boulder_14_W.txt",
    "monthly": "https://www.ncei.noaa.gov/pub/data/uscrn/products/monthly01/CRNM0102-CO_Boulder_14_W.txt",
    "hourly_nrt": "https://www.ncei.noaa.gov/pub/data/uscrn/products/hourly02/updates/2024/CRN60H0203-202402082100.txt",
    "daily_nrt": "https://www.ncei.noaa.gov/pub/data/uscrn/products/daily01/updates/2024/CRND0103-202402072359.txt",
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

    assert {"long_name", "units"} <= ds.latitude.attrs.keys()
    assert {"long_name", "units"} <= ds.longitude.attrs.keys()

    auto_xr_title = ds.title
    if which.endswith("_nrt"):
        which_ = which[: -len("_nrt")]
        if which_ == "hourly":
            assert (
                auto_xr_title
                == f"U.S. Climate Reference Network (USCRN) | {which_} | 2024-02-08 20"
            )
        elif which_ == "daily":
            assert (
                auto_xr_title == f"U.S. Climate Reference Network (USCRN) | {which_} | 2024-02-07"
            )
        else:
            raise AssertionError
    elif which == "monthly":
        assert auto_xr_title.startswith(
            f"U.S. Climate Reference Network (USCRN) | {which} | 2003-09--20"
        )
    else:
        assert auto_xr_title == f"U.S. Climate Reference Network (USCRN) | {which} | 2019"

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
    elif which == "hourly_nrt":
        assert df.wban.iloc[0] == "03047"
        assert df.wban.iloc[-1] == "96409"
        #
        assert "soil_moisture_10" in df
        assert "depth" in ds.dims
        assert {"soil_moisture", "soil_temp"} < set(ds.data_vars)
        assert "soil_moisture_10" not in ds
    elif which == "daily_nrt":
        assert df.wban.iloc[0] == "03047"
        assert df.wban.iloc[-1] == "96409"
        #
        assert "soil_moisture_10_daily" in df
        assert "depth" in ds.dims
        assert {"soil_moisture_daily", "soil_temp_daily"} < set(ds.data_vars)
        assert "soil_moisture_10_daily" not in ds
    else:
        raise AssertionError


def test_which_to_reader():
    from uscrn.attrs import WHICHS

    assert _which_to_reader.keys() == set(WHICHS)


@pytest.mark.parametrize("which, url", EXAMPLE_URL.items())
def test_parse_url(which, url):
    nrt = which.endswith("_nrt")
    res = parse_url(url)
    assert res.nrt == nrt
    if nrt:
        assert res.time.year == 2024
    else:
        assert url.endswith(res.name)
        assert res.which == which
        assert res.state == "CO"
        assert res.location == "Boulder"
        assert res.vector == "14 W"
        assert res.time is None


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


def test_get_nrt_m1_hourly():
    now = pd.Timestamp.now("UTC").tz_localize(None)
    df = get_nrt_data(-1, "hourly")

    time_counts = df["utc_time"].value_counts()

    if now.minute >= 47:
        # Probably new file has been uploaded, but there is some variation in upload times
        # https://www.ncei.noaa.gov/pub/data/uscrn/products/hourly02/updates/2024/
        dh = 1
    else:
        dh = 2
    try:
        assert time_counts.index[0] == now.floor("h") - pd.Timedelta(hours=dh)
    except AssertionError:
        assert time_counts.index[0] == now.floor("h") - pd.Timedelta(hours=dh + 1)
        warnings.warn("Seems new hourly NRT file hasn't been uploaded yet")


def test_get_nrt_m1_daily():
    now = pd.Timestamp.now("EST").tz_localize(None)
    df = get_nrt_data(-1, "daily")

    unique_times = df["lst_date"].unique()
    assert len(unique_times) == 1
    time = unique_times[0]

    if (now.hour, now.minute) >= (0, 40):
        # Probably yesterday file has been uploaded
        # https://www.ncei.noaa.gov/pub/data/uscrn/products/daily01/updates/2024/
        dd = 1
    else:
        dd = 2
    try:
        assert time == now.floor("d") - pd.Timedelta(days=dd)
    except AssertionError:
        assert time == now.floor("d") - pd.Timedelta(days=dd + 1)
        warnings.warn("Seems new daily NRT file hasn't been uploaded yet")


def test_get_nrt_mslice():
    df = get_nrt_data((-2, None), "daily")
    assert df.lst_date.nunique() == 2
    assert "--" in df.attrs["title"].split(" | ")[-1]


def test_get_nrt_ts_hourly():
    with pytest.warns(
        UserWarning, match="Timestamp 2023-08-31 17:00:00 has no timezone, assuming UTC."
    ):
        df = uscrn.get_nrt_data("2023-08-31 17", "hourly")
    time_counts = df["utc_time"].value_counts()
    assert time_counts.index[0] == pd.Timestamp("2023-08-31 16:00")
    assert time_counts.iloc[0] / len(df) > 0.75
    assert (
        df.attrs["title"]
        == "U.S. Climate Reference Network (USCRN) | hourly | updates | 2023-08-31 17"
    )

    df2 = uscrn.get_nrt_data(pd.Timestamp("2023-08-31 12", tz="EST"), "hourly")
    assert df2.equals(df)


def test_get_nrt_bad_which():
    with pytest.raises(ValueError, match="^Invalid dataset identifier."):
        _ = get_nrt_data(-1, "asdf")


def test_get_nrt_bad_period_element():
    with pytest.raises(TypeError):
        _ = get_nrt_data("asdf", "hourly")


@pytest.mark.parametrize("year", [2019, 3000])
def test_get_nrt_bad_year(year):
    with pytest.raises(RuntimeError, match="^No files"):
        _ = get_nrt_data(pd.Timestamp(year=year, month=1, day=1, tz="UTC"), "hourly")


def test_get_nrt_single_daily_auto_floor():
    with pytest.warns(
        UserWarning, match="Timestamp 2023-08-31 01:00:00 has no timezone, assuming UTC."
    ):
        df = get_nrt_data("2023-08-31 01", "daily")
    assert df.lst_date.eq("2023-08-31").all()


def test_get_nrt_zero_hourly():
    df = get_nrt_data(0, "hourly")
    time_counts = df["utc_time"].value_counts()
    assert time_counts.index[0] == pd.Timestamp("2020-10-06 20") - pd.Timedelta(hours=1)


def test_get_nrt_zero_daily():
    df = get_nrt_data(0, "daily")
    assert df.lst_date.eq("2020-10-06").all()
    assert (
        df.attrs["title"] == "U.S. Climate Reference Network (USCRN) | daily | updates | 2020-10-06"
    )


def test_get_nrt_cat():
    df = get_nrt_data(-1, "hourly", cat=True)
    assert len(df.select_dtypes("category").columns) > 0


def test_to_xarray_no_which_attr():
    with pytest.raises(NotImplementedError, match="^Guessing `which`"):
        to_xarray(pd.DataFrame())


def test_auto_title_bad_input():
    with pytest.raises(TypeError, match="^Failed to convert a and b to pandas datetime"):
        auto_title(("asdf", "asdf"), "daily")

    with pytest.raises(TypeError, match="^Expected a and b to be coercible to pandas.Timestamp"):
        auto_title((["2023-08-01", "2023-08-02"], "2023-08-04"), "daily")

    with pytest.raises(ValueError, match="^Expected b >= a"):
        auto_title(("2023-01-01 01", "2023-01-01 00"), "hourly")


def test_df_parquet_roundtrip(tmp_path):
    import fastparquet

    df = get_data(2019, which="daily", n_jobs=N, cat=True)
    assert df.attrs != {}

    # Write with both engines
    p_pa = tmp_path / "test_pa.parquet"
    df.to_parquet(p_pa, index=False, engine="pyarrow")
    p_fp = tmp_path / "test_fp.parquet"
    df.to_parquet(p_fp, index=False, engine="fastparquet")

    # Read with both engines
    cases = {
        "pa-pa": pd.read_parquet(p_pa, engine="pyarrow"),
        "pa-fp": pd.read_parquet(p_pa, engine="fastparquet"),
        "fp-pa": pd.read_parquet(p_fp, engine="pyarrow"),
        "fp-fp": pd.read_parquet(p_fp, engine="fastparquet"),
    }

    # For all cases, the data should be the same
    for case, df_ in cases.items():
        assert df_.index.equals(df.index), f"index same for {case}"
        assert df_.columns.equals(df.columns), f"columns same for {case}"
        if case == "fp-pa":
            # Categorical type not preserved, but data same
            assert not df_.equals(df), f"equals fails for {case}"
            diff = df.compare(df_)
            assert diff.empty, f"data same for {case}"
            assert not isinstance(
                df_.sur_temp_daily_type.dtype, pd.CategoricalDtype
            ), f"cat dtype not rt for {case}"
        else:
            assert df_.equals(df), f"data same for {case}"
            assert isinstance(
                df_.sur_temp_daily_type.dtype, pd.CategoricalDtype
            ), f"cat dtype rt for {case}"

    if Version(pd.__version__) < Version("2.1"):
        for case, df_ in cases.items():
            assert df_.attrs == {}, f"no preservation before pandas 2.1, case {case}"
    else:  # pandas 2.1+
        for case, df_ in cases.items():
            if case == "pa-pa":
                assert df.attrs == df_.attrs, f"attrs roundtrip for {case}"
            else:  # fastparquet involved
                if Version(fastparquet.__version__) < Version("2024.2.0"):
                    assert (
                        df_.attrs == {}
                    ), f"no attrs roundtrip before fastparquet 2024.2.0, case {case}"
                else:
                    assert df_.attrs == df.attrs, f"attrs roundtrip for {case}"
