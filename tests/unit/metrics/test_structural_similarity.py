"""StructuralSimilarity — Jaccard over flattened JSON paths (a root metric).

Compares the *shape* of two documents (which paths exist), ignoring values.
"""

from typing import Any

import pytest

from structured_eval import EvalConfig, StructuralSimilarity
from tests.conftest import build_tree

pytestmark = pytest.mark.unit

CONFIG = EvalConfig(metrics=[StructuralSimilarity()])


@pytest.mark.parametrize(
    ("actual", "expected", "score"),
    [
        ({"a": 1, "b": 2}, {"a": 9, "b": 8}, 1.0),  # same shape, different values
        ({"a": 1, "b": 2}, {"a": 9, "c": 3}, 1 / 3),  # paths a,b vs a,c → ∩=1 ∪=3
        ({"a": {"x": 1}}, {"a": {"x": 1}}, 1.0),  # nested paths match
        ({"a": {"x": 1}}, {"a": {"y": 1}}, 1 / 3),  # {a,a.x} vs {a,a.y} → ∩=1 ∪=3
        ({"a": 1}, {"b": 1}, 0.0),  # no shared path
        ({}, {}, 1.0),  # both empty → 1.0
    ],
    ids=["same-shape", "partial", "nested-equal", "nested-diff", "disjoint", "empty"],
)
def test_structural_similarity(actual: Any, expected: Any, score: Any) -> None:
    root = build_tree(actual, expected, CONFIG)
    assert float(root.metric_results["structural_similarity"]) == pytest.approx(score)
