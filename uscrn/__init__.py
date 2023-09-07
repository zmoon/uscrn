"""
Load US CRN data.
"""

from .attrs import load_attrs
from .get import get_crn, load_meta, read_daily, to_xarray

ATTRS = load_attrs()
del load_attrs

__all__ = ["load_meta", "read_daily", "get_crn", "to_xarray", "ATTRS"]
