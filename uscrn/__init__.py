"""
Load U.S. CRN data.
"""

__version__ = "0.1.0.dev4"

from .attrs import load_attrs
from .data import get_data, load_meta, read, read_daily, read_hourly, read_monthly, to_xarray

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
    "to_xarray",
    "__version__",
]
