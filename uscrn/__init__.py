"""
Load `U.S. Climate Reference Network <https://www.ncei.noaa.gov/access/crn/>`__ (USCRN) data.

* Home page: https://www.ncei.noaa.gov/access/crn/
* Info: https://www.ncei.noaa.gov/access/crn/qcdatasets.html
* Data: https://www.ncei.noaa.gov/pub/data/uscrn/products/
"""

__version__ = "0.1.0b3"

from .attrs import load_attrs
from .data import (
    get_data,
    load_meta,
    read,
    read_daily,
    read_hourly,
    read_monthly,
    read_subhourly,
    to_xarray,
)

ATTRS = load_attrs()
"""Dataset and variable attributes, mainly taken from the respective readmes.
The relevant top-level keys are ``hourly``, ``daily``, etc.
"""

del load_attrs

__all__ = [
    "ATTRS",
    "get_data",
    "load_meta",
    "read",
    "read_daily",
    "read_hourly",
    "read_monthly",
    "read_subhourly",
    "to_xarray",
    "__version__",
]
