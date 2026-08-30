"""Shared aligner helpers (alignment/base.py) — key extraction.

`key_value` is what every keyed aligner pairs on, so its three answers have to
stay distinct: the value, `None` for a field the element does not carry, and a
sentinel for an element that carries no fields at all.
"""

from typing import Any

import pytest

from structured_eval.alignment.base import _MISSING_KEY, key_value

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("element", "key", "value"),
    [
        ({"sku": "A-1"}, "sku", "A-1"),
        ({"who": {"first": "Ann"}}, "who.first", "Ann"),
        ({"sku": "A-1"}, "qty", None),
        ({"sku": "A-1"}, None, {"sku": "A-1"}),
        ("A-1", None, "A-1"),
        ("A-1", "sku", _MISSING_KEY),
    ],
    ids=[
        "field",
        "nested-field",
        "absent-field",
        "no-key",
        "no-key-scalar",
        "not-a-dict",
    ],
)
def test_the_key_of_an_element(element: Any, key: str | None, value: Any) -> None:
    # No key at all means the element *is* the key; a named key on something
    # that has no fields is neither the element nor a missing field — pairing on
    # it would make every scalar look alike.
    assert key_value(element, key) == value
