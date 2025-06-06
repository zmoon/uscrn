from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any, Final, Literal, NamedTuple, Union

import numpy as np

HERE = Path(__file__).parent
_CACHE_DIR = HERE / "cache"


def expand_str(s: str) -> list[str]:
    """For example:
    "hi there, I'm a {cat,dog}. {woof,meow}!"
    => ["hi there, I'm a cat. woof!", "hi there, I'm a dog. meow!"]
    """
    import re
    from ast import literal_eval

    repls: dict[str, list[str]] = {}
    to_repl = re.findall(r"\{.*?\}", s)
    for braced in to_repl:
        # TODO: could be improved, issues if quotes within the quoted string
        opts = [
            s.strip() for s in re.split(r",(?=(?:[^'\"]*['\"][^'\"]*['\"])*[^'\"]*$)", braced[1:-1])
        ]
        for i, opt in enumerate(opts):
            # Maybe remove quotes
            if opt.startswith(("'", '"')):
                try:
                    opt_ = literal_eval(opt)
                except (ValueError, SyntaxError):
                    continue
                else:
                    opts[i] = opt_
        repls[braced] = opts

    if not repls:
        return [s]

    # Check counts
    counts = {k: len(v) for k, v in repls.items()}
    n0 = counts[to_repl[0]]
    if not all(n == n0 for n in counts.values()):
        raise ValueError(f"Number of options should be same in all cases, but got: {counts}.")

    # Do replacements
    s_news = []
    for i in range(n0):
        s_new = s
        for braced, opts in repls.items():
            s_new = s_new.replace(braced, opts[i])
        s_news.append(s_new)

    return s_news


def expand_strs(d: Mapping[str, str | None]) -> list[dict[str, str | None]]:
    """Apply :func:`expand_str` to all values in dict, generating new dicts."""

    opts: dict[str, list[str] | list[None]] = {}
    for k, v in d.items():
        if isinstance(v, str):
            opts[k] = expand_str(v)
        else:
            opts[k] = [v]

    # NOTE: Number of opts for each key will be 1 or n (which may itself be 1)
    n = max(len(v) for v in opts.values())
    d_news = []
    for i in range(n):
        d_new = {}
        for k, vals in opts.items():
            if len(vals) == 1:
                d_new[k] = vals[0]
            else:
                d_new[k] = vals[i]
        d_news.append(d_new)

    return d_news


WHICHS: Final = ("subhourly", "hourly", "daily", "monthly")
"""Identifiers for the datasets that have been implemented."""

_ALL_WHICHS: Final = ("subhourly", "hourly", "daily", "monthly")
"""All dataset identifiers, including those that may have not yet been implemented."""

DEFAULT_WHICH: Final = "daily"


def validate_which(which: str) -> None:
    if which not in _ALL_WHICHS:
        msg = f"Invalid dataset identifier: {which!r}. Valid identifiers are: {_ALL_WHICHS}."
        if len(WHICHS) < len(_ALL_WHICHS):  # pragma: no cover
            msg += f" These have been implemented: {WHICHS}."
        raise ValueError(msg)

    if which not in WHICHS:  # pragma: no cover
        raise NotImplementedError(
            f"Dataset {which!r} not yet implemented. These have been implemented: {WHICHS}."
        )


@lru_cache(1)
def load_attrs() -> dict[str, dict[str, Any]]:
    """Load information from the attrs YAML file.

    * expanding templating using :func:`expand_strs`
    * filling in defaults for fields that haven't been specified in individual entries

    The variable metadata are based on the individual dataset ``readme.txt`` files,
    but may differ slightly, e.g.

    * trying to follow CF conventions for units

    This is cached for the lifetime of the process.
    """
    import itertools

    import yaml

    with open(HERE / "attrs.yml") as f:
        attrs = yaml.full_load(f)

    # Expand column entries
    for which in WHICHS:
        if which not in attrs:  # pragma: no cover
            continue
        var_attrs = list(
            itertools.chain.from_iterable(expand_strs(d) for d in attrs[which]["columns"])
        )
        names = [a["name"] for a in var_attrs]
        assert len(names) == len(set(names)), "Names should be unique"
        attrs[which]["columns"] = {a["name"]: a for a in var_attrs}  # list -> dict

    # dtype defaults to float32
    # NOTE: Floats in the text files are represented with 7 chars only, little precision
    for which in WHICHS:
        for _, v in attrs[which]["columns"].items():
            if "dtype" not in v:
                v["dtype"] = "float32"

    # xarray-only defaults to false
    for which in WHICHS:
        for _, v in attrs[which]["columns"].items():
            if "xarray_only" not in v:
                v["xarray_only"] = False

    # categories defaults to false
    for which in WHICHS:
        for _, v in attrs[which]["columns"].items():
            if "categories" not in v:
                v["categories"] = False

    # QC flag name defaults to None
    # (only for selected variables in subhourly and hourly data)
    pref = "QC flag for "
    for which in WHICHS:
        for _, v in attrs[which]["columns"].items():
            v["qc_flag_name"] = None

        long_name_to_name = {v["long_name"]: k for k, v in attrs[which]["columns"].items()}
        flag_names = [name for name in attrs[which]["columns"] if name.endswith("_flag")]
        for flag_name in flag_names:
            flag_long_name = attrs[which]["columns"][flag_name]["long_name"]
            assert flag_long_name.startswith(pref)
            flag_for = long_name_to_name[flag_long_name[len(pref) :]]
            attrs[which]["columns"][flag_for]["qc_flag_name"] = flag_name

    # type flag name defaults to None
    # (only for infrared surface temperature)
    for which in WHICHS:
        for _, v in attrs[which]["columns"].items():
            v["type_flag_name"] = None

        if which == "subhourly":
            flag_names = ["st_type"]
            flag_fors = ["surface_temperature"]
        else:
            pref = "sur_temp"
            flag_fors = [
                name
                for name in attrs[which]["columns"]
                if name.startswith(pref) and not name.endswith(("_flag", "_type"))
            ]
            assert len(flag_fors) == 3
            flag_name = pref + ("" if which == "hourly" else f"_{which}") + "_type"
            flag_names = [flag_name] * len(flag_fors)

        for flag_name, flag_for in zip(flag_names, flag_fors):
            assert flag_name in attrs[which]["columns"]
            assert flag_for in attrs[which]["columns"]
            attrs[which]["columns"][flag_for]["type_flag_name"] = flag_name

    return attrs


_DTypes = Union[type[np.float32], type[np.float64], type[str]]


class _DsetVarInfo(NamedTuple):
    names: list[str]
    """Column (variable) names, in the correct order."""

    dtypes: dict[str, _DTypes]
    """Maps column names to dtypes, for use in pandas ``read_csv``."""

    attrs: dict[str, dict[str, str | None]]
    """Maps column names to attribute dicts with, e.g., ``long_name`` and ``units``."""

    notes: str
    """Labeled notes associated with the dataset, mentioned in the readme."""

    categorical: dict[str, list[Any]]
    """Maps applicable column names to list of categories."""


_DTYPE_MAP: dict[str, _DTypes | None] = {
    "string": str,
    "float": np.float32,
    "float32": np.float32,
    "float64": np.float64,
    "ignore": None,
}


def _map_dtype(dtype: str) -> _DTypes | None:
    if dtype not in _DTYPE_MAP:
        raise ValueError(f"Unknown dtype: {dtype!r}. Expected one of: {list(_DTYPE_MAP)}")
    return _DTYPE_MAP[dtype]


@lru_cache(len(WHICHS))
def get_col_info(
    which: Literal["subhourly", "hourly", "daily", "monthly"] = "daily",
) -> _DsetVarInfo:
    """Column (variable) info (the individual data files don't have headers),
    intended for use in ``read_csv``.

    Derived based on:

    * stored attribute data (based on the individual datasets' ``readme.txt`` files)
    * the dataset ``headers.txt`` file (to ensure correct column name order)
    * the dataset ``readme.txt`` file (for the notes)

    This is cached for the lifetime of the process.
    """
    from textwrap import dedent

    validate_which(which)

    stored_attrs = load_attrs()
    headers_txt, readme_txt = _get_docs(which)

    # "This file contains the following three lines: Field Number, Field Name and Unit of Measure."
    readme_lines = headers_txt.splitlines()
    assert len(readme_lines) == 3
    nums = readme_lines[0].split()
    columns = readme_lines[1].split()
    assert len(nums) == len(columns)
    assert nums == [str(i + 1) for i in range(len(columns))]

    # For consistency with meta, we use 'wban' instead of 'wbanno'
    assert columns[0] == "WBANNO"
    columns[0] = "WBAN"

    if which == "monthly":
        # Consistent CRX version number var name
        assert columns[2] == "CRX_VN_MONTHLY"
        columns[2] = "CRX_VN"

        # Consistent lat/lon names
        assert columns[3] == "PRECISE_LONGITUDE"
        assert columns[4] == "PRECISE_LATITUDE"
        columns[3] = "LONGITUDE"
        columns[4] = "LATITUDE"

    # Lowercase is better
    columns = [c.lower() for c in columns]

    # Check consistency with attrs YAML file
    assert len(columns) == len(set(columns)), "Column names should be unique"
    attrs = stored_attrs[which]["columns"]
    in_table_var_attrs = {k: v for k, v in attrs.items() if v["xarray_only"] is False}
    assert in_table_var_attrs.keys() == set(columns)

    # Notes
    readme_url = f"{stored_attrs[which]['base_url']}/readme.txt"
    readme_lines = readme_txt.splitlines()
    notes = f"Notes from {readme_url}:\n"
    for i, line in enumerate(readme_lines):
        line_ = line.strip()
        if line_.startswith("IMPORTANT NOTES:"):
            notes += dedent("\n".join(readme_lines[i + 1 :]))
            break
    else:  # pragma: no cover
        raise AssertionError("Expected readme line starting with 'IMPORTANT NOTES:'")

    # Construct dtype dict (for ``read_csv``)
    dtypes: dict[str, _DTypes] = {}
    for k, v in attrs.items():
        assert isinstance(k, str)
        dtype = _map_dtype(v["dtype"])
        if dtype is not None:
            dtypes[k] = dtype

    # Categorical dtype categories
    categorical = {k: v["categories"] for k, v in attrs.items() if v["categories"] is not False}

    return _DsetVarInfo(
        names=columns,
        dtypes=dtypes,
        attrs=attrs,
        notes=notes,
        categorical=categorical,
    )


def _get_docs(
    which: Literal["subhourly", "hourly", "daily", "monthly"] = "daily",
) -> tuple[str, str]:
    """Get the header and readme docs as strings.

    The files are downloaded if they:

    * are not already present in the cache location
    * are present but their timestamp is older than the remote version

    Otherwise, the cached file is used (read).
    """
    import pandas as pd
    import requests

    from ._util import logger, retry

    validate_which(which)

    stored_attrs = load_attrs()
    base_url = stored_attrs[which]["base_url"]

    @retry
    def get(url: str, fp: Path) -> str:
        needs_update: bool
        if fp.is_file():
            r = requests.head(url, timeout=10)
            r.raise_for_status()
            last_modified_url = pd.Timestamp(r.headers["Last-Modified"])
            last_modified_local = pd.Timestamp(fp.stat().st_mtime, unit="s", tz="UTC")
            needs_update = last_modified_url > last_modified_local
        else:
            needs_update = True

        if needs_update:
            logger.info(f"downloading {url} to {fp}")
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            with open(fp, "w") as f:
                f.write(r.text)

        with open(fp) as f:
            text = f.read()

        return text

    cache_dir = _CACHE_DIR
    cache_dir.mkdir(exist_ok=True)

    headers_txt = get(f"{base_url}/headers.txt", cache_dir / f"{which}_headers.txt")
    readme_txt = get(f"{base_url}/readme.txt", cache_dir / f"{which}_readme.txt")

    return headers_txt, readme_txt
