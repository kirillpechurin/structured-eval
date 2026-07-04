"""ArrayExactMatch — strict, order-sensitive whole-array equality (1.0 / 0.0)."""

from typing import Any

import pytest

from structured_eval.metrics import ArrayExactMatch
from structured_eval.models import ArrayFieldConfig, EvalConfig
from tests.conftest import build_tree

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("actual", "expected", "score"),
    [
        ([1, 2, 3], [1, 2, 3], 1.0),  # identical
        ([], [], 1.0),  # both empty
        ([1, 2], [2, 1], 0.0),  # order matters
        ([1, 2], [1, 2, 3], 0.0),  # length differs
        ([{"a": 1}], [{"a": 1}], 1.0),  # deep equality of nested objects
        ([{"a": 1}], [{"a": 2}], 0.0),  # nested value differs
        ([1, 2], "1,2", 0.0),  # non-list side
        ([1], [1.0], 0.0),  # type-strict: int vs float
    ],
    ids=[
        "identical",
        "both-empty",
        "order",
        "length",
        "nested-equal",
        "nested-diff",
        "non-list",
        "type-strict",
    ],
)
def test_array_exact_match(actual: Any, expected: Any, score: float) -> None:
    assert ArrayExactMatch().score(actual, expected) == score


def test_runs_on_array_node() -> None:
    """As an ArrayMetric it cascades onto the array node and scores the whole list."""
    config = EvalConfig(fields={"tags": ArrayFieldConfig(metrics=[ArrayExactMatch()])})
    root = build_tree({"tags": ["a", "b"]}, {"tags": ["a", "b"]}, config)
    tags = next(c for c in root.children_nodes() if c.path == "tags")
    assert float(tags.metric_results["array_exact_match"]) == 1.0
