"""extract_paths — structural skeleton (containers + indices + leaves)."""

from typing import Any

import pytest

from structured_eval.utils.flatten import extract_paths

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("value", "paths"),
    [
        ({"a": 1, "b": 2}, {"a", "b"}),  # flat dict
        ({"a": {"b": 1}}, {"a", "a.b"}),  # nested dict → container + leaf
        ({"a": [1, 2]}, {"a", "a[0]", "a[1]"}),  # list indices
        ({"a": [{"b": 1}]}, {"a", "a[0]", "a[0].b"}),  # mixed nesting
        ({}, set()),  # empty
        (42, set()),  # scalar root has no paths
    ],
    ids=["flat", "nested", "list", "mixed", "empty", "scalar"],
)
def test_extract_paths(value: Any, paths: set[str]) -> None:
    assert extract_paths(value) == paths
