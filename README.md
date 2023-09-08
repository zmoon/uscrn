# uscrn

Easily load [U.S. CRN data](https://www.ncei.noaa.gov/access/crn/).

[![Version on PyPI](https://img.shields.io/pypi/v/uscrn.svg)](https://pypi.org/project/uscrn/)
[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)

Example:

```python
import uscrn as crn

df = crn.get_data(2019, "hourly", n_jobs=6)

ds = crn.to_xarray(df)
```

Mamba install example:

```sh
mamba create -n crn -c conda-forge python=3.10 joblib numpy pandas pyyaml requests xarray fastparquet netcdf4
mamba activate crn
pip install --no-deps uscrn
```
