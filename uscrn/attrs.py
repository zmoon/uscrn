from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache, partial
from pathlib import Path
from typing import Any, NamedTuple

HERE = Path(__file__).parent


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
        if v is not None:
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


@lru_cache(1)
def load_attrs():
    import itertools

    import yaml

    with open(HERE / "attrs.yml") as f:
        attrs = yaml.full_load(f)

    # Expand column entries
    for which in ["hourly", "daily", "monthly"]:
        if which not in attrs:
            continue
        var_attrs = list(
            itertools.chain.from_iterable(expand_strs(d) for d in attrs[which]["columns"])
        )
        names = [a["name"] for a in var_attrs]
        assert len(names) == len(set(names)), "Names should be unique"
        attrs[which]["columns"] = {a["name"]: a for a in var_attrs}

    return attrs


class _DsetVarInfo(NamedTuple):
    names: list[str]
    dtypes: dict[str, Any]  # TODO: better typing
    attrs: dict[str, dict[str, str | None]]
    notes: dict[str, str]


def get_col_info(which: str = "daily") -> _DsetVarInfo:
    """Read the column info file (the individual data files don't have headers)
    and stored attribute data.

    For example:
    https://www.ncei.noaa.gov/pub/data/uscrn/products/daily01/headers.txt
    """
    import numpy as np
    import requests

    dtype_overrides = {
        "wban": str,
        "crx_vn": str,
        "longitude": np.float64,  # since coords
        "latitude": np.float64,
    }
    if which == "hourly":
        url = "https://www.ncei.noaa.gov/pub/data/uscrn/products/hourly02/headers.txt"
        for vn in [
            "solarad_flag",
            "solarad_max_flag",
            "solarad_min_flag",
            "sur_temp_type",
            "sur_temp_flag",
            "sur_temp_max_flag",
            "sur_temp_min_flag",
            "rh_hr_avg_flag",
        ]:
            dtype_overrides[vn] = str
        dtype_removes = [
            "utc_date",
            "utc_time",
            "lst_date",
            "lst_time",
        ]
        # NOTE: reader combines the DATE and TIME columns of original dataset together to make TIMEs
    elif which == "daily":
        url = "https://www.ncei.noaa.gov/pub/data/uscrn/products/daily01/headers.txt"
        dtype_overrides.update(
            {
                "sur_temp_daily_type": str,
            }
        )
        dtype_removes = ["lst_date"]
    else:
        raise ValueError(f"Unknown/unsupported which: {which!r}")

    # "This file contains the following three lines: Field Number, Field Name and Unit of Measure."
    r = requests.get(url)
    r.raise_for_status()
    lines = r.text.splitlines()
    assert len(lines) == 3
    nums = lines[0].split()
    columns = lines[1].split()
    assert len(nums) == len(columns)
    assert nums == [str(i + 1) for i in range(len(columns))]

    # For consistency with meta, use 'wban' instead of 'wbanno'
    assert columns[0] == "WBANNO"
    columns[0] = "WBAN"

    # Lowercase better
    columns = [c.lower() for c in columns]

    # Check consistency with attrs YAML file
    assert len(columns) == len(set(columns)), "Column names should be unique"
    stored_attrs = load_attrs()
    attrs = stored_attrs[which]["columns"]
    notes = stored_attrs[which]["notes"]
    in_table_var_attrs = {
        k: v
        for k, v in attrs.items()
        if k
        not in {
            "soil_moisture_daily",
            "soil_temp_daily",
            "soil_temp",
            "soil_moisture",
        }
    }
    assert in_table_var_attrs.keys() == set(columns)

    # NOTE: Floats in the text files are represented with 7 chars only, little precision
    dtypes: dict[str, Any] = {c: np.float32 for c in columns}
    for k, v in dtype_overrides.items():
        dtypes[k] = v
    for k in dtype_removes:
        del dtypes[k]

    return _DsetVarInfo(names=columns, dtypes=dtypes, attrs=attrs, notes=notes)


get_daily_col_info = partial(get_col_info, which="daily")
