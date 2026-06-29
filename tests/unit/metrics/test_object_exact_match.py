"""ObjectExactMatch — strict deep equality of whole objects (1.0 / 0.0)."""

from typing import Any

import pytest

from structured_eval import EvalConfig, FieldConfig, ObjectExactMatch, ObjectFieldConfig
from tests.conftest import build_tree

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("actual", "expected", "score"),
    [
        ({"a": 1, "b": 2}, {"a": 1, "b": 2}, 1.0),  # identical
        ({"b": 2, "a": 1}, {"a": 1, "b": 2}, 1.0),  # key order irrelevant
        ({}, {}, 1.0),  # both empty
        ({"a": 1}, {"a": 2}, 0.0),  # value differs
        ({"a": 1}, {"a": 1, "b": 2}, 0.0),  # missing key
        ({"a": {"x": [1, 2]}}, {"a": {"x": [1, 2]}}, 1.0),  # deep nested equal
        ({"a": {"x": [1, 2]}}, {"a": {"x": [2, 1]}}, 0.0),  # nested list order
        ({"a": 1}, [["a", 1]], 0.0),  # non-dict side
    ],
    ids=[
        "identical",
        "key-order",
        "both-empty",
        "value-diff",
        "missing-key",
        "nested-equal",
        "nested-order",
        "non-dict",
    ],
)
def test_object_exact_match(actual: Any, expected: Any, score: float) -> None:
    assert ObjectExactMatch().score(actual, expected) == score


def test_runs_on_object_node() -> None:
    """As an ObjectMetric it cascades onto the object node and scores the whole dict."""
    config = EvalConfig(
        fields={
            "addr": ObjectFieldConfig(
                fields={"city": FieldConfig()}, metrics=[ObjectExactMatch()]
            )
        }
    )
    root = build_tree({"addr": {"city": "NY"}}, {"addr": {"city": "LA"}}, config)
    addr = next(c for c in root.children_nodes() if c.path == "addr")
    assert float(addr.metric_results["object_exact_match"]) == 0.0
