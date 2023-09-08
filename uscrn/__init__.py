"""
Load US CRN data.
"""

__version__ = "0.1.0.dev0"

from .attrs import load_attrs
from .get import get_data, load_meta, read, read_daily, read_hourly, to_xarray

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
    "to_xarray",
    "__version__",
]
