"""TypeMatch — equality of the JSON value *type*, value ignored.

Booleans are not numbers; null matches null.
"""

from typing import Any

import pytest

from structured_eval.metrics import TypeMatch

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("actual", "expected", "score"),
    [
        (100, 200, 1.0),  # both int
        ("100", 100, 0.0),  # str vs int
        (True, 1, 0.0),  # bool is not number
        (None, None, 1.0),  # null matches null
        ([1], [2, 3], 1.0),  # both list (length irrelevant)
        ({"a": 1}, {}, 1.0),  # both dict
    ],
    ids=["int-int", "str-vs-int", "bool-vs-int", "null-null", "list-list", "dict-dict"],
)
def test_score(actual: Any, expected: Any, score: Any) -> None:
    assert TypeMatch().score(actual, expected) == score
