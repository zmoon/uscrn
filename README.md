# uscrn

Easily load [U.S. Climate Reference Network data](https://www.ncei.noaa.gov/access/crn/).

[![Version on PyPI](https://img.shields.io/pypi/v/uscrn.svg)](https://pypi.org/project/uscrn/)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/zmoon/uscrn/main.svg)](https://results.pre-commit.ci/latest/github/zmoon/uscrn/main)
[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)

Example:

```python
import uscrn as crn

df = crn.get_data(2019, "hourly", n_jobs=6)  # pandas.DataFrame

ds = crn.to_xarray(df)  # xarray.Dataset
```

Both `df` (pandas) and `ds` (xarray) include dataset and variable metadata.
For `df`, these are in `df.attrs` and can be preserved by
writing to Parquet with the PyArrow engine with pandas v2.1+.

```python
df.to_parquet("crn_2019_hourly.parquet", engine="pyarrow")
```

Mamba install example:

```sh
mamba create -n crn -c conda-forge python=3.10 joblib numpy pandas pyyaml requests xarray pyarrow netcdf4
mamba activate crn
pip install --no-deps uscrn
```
