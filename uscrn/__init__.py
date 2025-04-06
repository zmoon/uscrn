"""
Load `U.S. Climate Reference Network <https://www.ncei.noaa.gov/access/crn/>`__ (USCRN) data.

* Home page: https://www.ncei.noaa.gov/access/crn/
* Info: https://www.ncei.noaa.gov/access/crn/qcdatasets.html
* Data: https://www.ncei.noaa.gov/pub/data/uscrn/products/

.. currentmodule:: ~

Get data
--------

Download selected data for all sites and return as a :class:`pandas.DataFrame`.
In the returned :class:`~pandas.DataFrame`,
and those from `the readers <#read>`__ as well,
variable attributes such as long name and units are stored in :attr:`~pandas.DataFrame.attrs`.

.. autosummary::
   :toctree: api/

   uscrn.get_data
   uscrn.get_nrt_data

Convert to xarray
-----------------

Convert to :class:`xarray.Dataset`,
automatically adding a soil depth dimension if applicable.

.. autosummary::
   :toctree: api/

   uscrn.to_xarray


Read
----

Read data from a single file and return as a :class:`pandas.DataFrame`.

.. autosummary::
   :toctree: api/

   uscrn.read
   uscrn.read_daily
   uscrn.read_daily_nrt
   uscrn.read_hourly
   uscrn.read_hourly_nrt
   uscrn.read_monthly
   uscrn.read_subhourly


Metadata
--------

Load site metadata as a :class:`pandas.DataFrame`.

.. autosummary::
   :toctree: api/

   uscrn.load_meta
"""

__version__ = "0.2.0b0"

from .attrs import load_attrs
from .data import (
    get_data,
    get_nrt_data,
    load_meta,
    read,
    read_daily,
    read_daily_nrt,
    read_hourly,
    read_hourly_nrt,
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
    "get_nrt_data",
    "load_meta",
    "read",
    "read_daily",
    "read_daily_nrt",
    "read_hourly",
    "read_hourly_nrt",
    "read_monthly",
    "read_subhourly",
    "to_xarray",
    "__version__",
]
