import inspect
from contextlib import contextmanager
from pathlib import Path
from typing import get_args

import pytest

import uscrn
from uscrn.attrs import (
    _ALL_WHICHS,
    DEFAULT_WHICH,
    WHICHS,
    _get_docs,
    _map_dtype,
    expand_str,
    expand_strs,
    get_col_info,
    load_attrs,
    validate_which,
)


@pytest.mark.parametrize(
    "s, expected",
    [
        ("hi no opts", ["hi no opts"]),
        ("hi {only-one-opt}", ["hi only-one-opt"]),
        ("{one,two}", ["one", "two"]),
        ("{one,'two'}", ["one", "two"]),
        ("{one, 'two'}", ["one", "two"]),
        ("{one, ' two'}", ["one", " two"]),
        (
            "Hi there, \"{asdf, 'name, with, commas, in, it'}\"!",
            ['Hi there, "asdf"!', 'Hi there, "name, with, commas, in, it"!'],
        ),
        (
            'Hi there, "{asdf, "name, with, commas, in, it"}"!',
            ['Hi there, "asdf"!', 'Hi there, "name, with, commas, in, it"!'],
        ),
        (
            "Hi there, I'm a {cat,dog}. {Meow,Woof}!",
            ["Hi there, I'm a cat. Meow!", "Hi there, I'm a dog. Woof!"],
        ),
        ("a{b, 'c\"}", ["ab", "a'c\""]),
    ],
)
def test_expand_str(s, expected):
    assert expand_str(s) == expected


def test_expand_str_error():
    with pytest.raises(ValueError, match="^Number of options"):
        expand_str("a{'1', '2'} + b{'3', '4', '5'}")


def test_expand_strs():
    d = {"greeting": "Hi there, I'm a {🐱,🐶}. {Meow,Woof}!", "type": "{cat,dog}"}
    assert expand_strs(d) == [
        {"greeting": "Hi there, I'm a 🐱. Meow!", "type": "cat"},
        {"greeting": "Hi there, I'm a 🐶. Woof!", "type": "dog"},
    ]


def test_validate_which():
    with pytest.raises(ValueError, match="^Invalid dataset identifier: 'asdf'"):
        validate_which("asdf")


def test_invalid_dtype():
    with pytest.raises(ValueError, match="^Unknown dtype: 'asdf'"):
        _map_dtype("asdf")


def test_load_attrs():
    attrs = load_attrs()

    assert len(attrs["daily"]["columns"]) == 28 + 2, "2 extra for the xarray depth dim ones"
    assert len(attrs["hourly"]["columns"]) == 38 + 2, "2 extra for the xarray depth dim ones"

    assert set(WHICHS) <= set(_ALL_WHICHS)
    whichs_guess = [k for k in attrs if not k.startswith("_")]
    assert set(whichs_guess) == set(WHICHS)

    for which in WHICHS:
        d = attrs[which]
        assert set(d) == {"base_url", "time_var", "columns"}
        assert d["time_var"] in d["columns"]
        assert not d["base_url"].endswith("/")
        for _, variable_dict in d["columns"].items():
            assert set(variable_dict) == {
                "name",
                "long_name",
                "units",
                "description",
                "dtype",
                "categories",
                "xarray_only",
                "qc_flag_name",
                "type_flag_name",
            }


@contextmanager
def change_cache_dir(p: Path):
    default = uscrn.attrs._CACHE_DIR
    uscrn.attrs._CACHE_DIR = p
    try:
        yield
    finally:
        uscrn.attrs._CACHE_DIR = default


def test_get_docs_dl(tmp_path, caplog):
    which = "daily"
    d = tmp_path

    p_headers = d / f"{which}_headers.txt"
    p_readme = d / f"{which}_readme.txt"
    assert not p_headers.exists()
    assert not p_readme.exists()

    with change_cache_dir(d), caplog.at_level("INFO"):
        _get_docs(which)
        assert "downloading" in caplog.text

    assert p_headers.is_file()
    assert p_readme.is_file()

    # Now should load from disk instead of web
    caplog.clear()
    with change_cache_dir(d), caplog.at_level("INFO"):
        _get_docs(which)
        assert "downloading" not in caplog.text


def test_load_col_info():
    get_col_info("subhourly")
    get_col_info("daily")
    get_col_info("hourly")
    get_col_info("monthly")


@pytest.mark.skipif(
    not hasattr(inspect, "get_annotations"), reason="Requires inspect.get_annotations (Python 3.10)"
)
def test_check_which_args():
    from uscrn.data import get_data, to_xarray

    for fn in [get_col_info, get_data]:
        which_anno = inspect.get_annotations(fn, eval_str=True)["which"]
        assert get_args(which_anno) == WHICHS
        sig = inspect.signature(fn)
        assert sig.parameters["which"].default == DEFAULT_WHICH

    fn = to_xarray
    which_anno = inspect.get_annotations(fn, eval_str=True)["which"]
    assert get_args(get_args(which_anno)[0]) == WHICHS
    sig = inspect.signature(fn)
    assert sig.parameters["which"].default in {None, DEFAULT_WHICH}
