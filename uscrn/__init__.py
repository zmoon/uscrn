"""
Load US CRN data.
"""

__version__ = "0.1.0.dev0"

from .attrs import load_attrs
from .get import get_crn, load_meta, read_daily, to_xarray

ATTRS = load_attrs()
del load_attrs

__all__ = ["load_meta", "read_daily", "get_crn", "to_xarray", "ATTRS", "__version__"]
