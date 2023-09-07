from pathlib import Path

import pandas as pd

from uscrn import to_xarray

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
