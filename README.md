# uscrn

Easily load [U.S. Climate Reference Network](https://www.ncei.noaa.gov/access/crn/) (USCRN) data.

[![Version on PyPI](https://img.shields.io/pypi/v/uscrn.svg)](https://pypi.org/project/uscrn/)
[![CI status](https://github.com/zmoon/uscrn/actions/workflows/ci.yml/badge.svg)](https://github.com/zmoon/uscrn/actions/workflows/ci.yml)
[![Documentation status](https://readthedocs.org/projects/uscrn/badge/?version=latest)](https://uscrn.readthedocs.io/en/latest/)
[![Test coverage](https://codecov.io/gh/zmoon/uscrn/branch/main/graph/badge.svg)](https://app.codecov.io/gh/zmoon/uscrn)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/zmoon/uscrn/main.svg)](https://results.pre-commit.ci/latest/github/zmoon/uscrn/main)
[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)

With `uscrn`, fetching and loading years of [data](https://www.ncei.noaa.gov/access/crn/qcdatasets.html) for all USCRN sites[^a] takes just one line of code[^b].

Example:

```python
import uscrn

df = uscrn.get_data(2019, "hourly", n_jobs=6)  # pandas.DataFrame

ds = uscrn.to_xarray(df)  # xarray.Dataset, with soil depth dimension if applicable (hourly, daily)
```

Both `df` (pandas) and `ds` (xarray) include dataset and variable metadata.
For `df`, these are in `df.attrs` and can be preserved by
writing to Parquet with the PyArrow engine[^d] with
[pandas v2.1+](https://pandas.pydata.org/docs/whatsnew/v2.1.0.html#other-enhancements).

```python
df.to_parquet("uscrn_2019_hourly.parquet", engine="pyarrow")
```

Conda install example[^c]:

```sh
conda create -n crn -c conda-forge python=3.10 joblib numpy pandas pyyaml requests xarray pyarrow netcdf4
conda activate crn
pip install --no-deps uscrn
```

[^a]: Use `uscrn.load_meta()` to load the site metadata table.
[^b]: Not counting the `import` statement...
[^c]: `uscrn` is not yet on conda-forge.
[^d]: Or the fastparquet engine with [fastparquet v2024.2.0+](https://github.com/dask/fastparquet/commit/9d7ee90e38103fef3dd1bd2f5eb0654b8bd3fdff).
