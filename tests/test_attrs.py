from typing import get_args

import pytest

from uscrn.attrs import (
    _ALL_WHICHS,
    VALID_WHICHS,
    WHICHS,
    expand_str,
    expand_strs,
    get_col_info,
    load_attrs,
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
    ],
)
def test_expand_str(s, expected):
    assert expand_str(s) == expected


def test_expand_strs():
    d = {"greeting": "Hi there, I'm a {🐱,🐶}. {Meow,Woof}!", "type": "{cat,dog}"}
    assert expand_strs(d) == [
        {"greeting": "Hi there, I'm a 🐱. Meow!", "type": "cat"},
        {"greeting": "Hi there, I'm a 🐶. Woof!", "type": "dog"},
    ]


def test_load_attrs():
    attrs = load_attrs()

    assert len(attrs["daily"]["columns"]) == 28 + 2, "2 extra for the xarray depth dim ones"
    assert len(attrs["hourly"]["columns"]) == 38 + 2, "2 extra for the xarray depth dim ones"

    assert set(WHICHS) <= set(_ALL_WHICHS)
    whichs_guess = [k for k in attrs if not k.startswith("_")]
    assert set(whichs_guess) == set(WHICHS)
    assert get_args(VALID_WHICHS) == WHICHS

    for which in WHICHS:
        d = attrs[which]
        assert set(d) == {"base_url", "time_var", "columns", "notes"}
        assert d["time_var"] in d["columns"]


def test_load_col_info():
    get_col_info("daily")
    get_col_info("hourly")
