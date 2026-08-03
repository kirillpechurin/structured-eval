"""ArrayJaccardSimilarity — set-overlap ``|A∩B|/|A∪B|`` over arrays.

Order- and count-insensitive; membership is exact equality. Built for arrays of
scalars, but does not crash on object/list elements (keyed by canonical JSON).
"""

from typing import Any

import pytest

from structured_eval.metrics import ArrayJaccardSimilarity
from structured_eval.models import ArrayFieldConfig, EvalConfig
from tests.conftest import build_tree

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("actual", "expected", "score"),
    [
        (["a", "b", "c"], ["b", "c", "d"], 2 / 4),  # |∩|=2, |∪|=4
        (["a", "b"], ["b", "a"], 1.0),  # order ignored
        (["a", "a", "b"], ["a", "b"], 1.0),  # duplicates collapse
        ([], [], 1.0),  # both empty → vacuously 1.0
        (["a"], [], 0.0),  # one side empty
        (["a"], ["b"], 0.0),  # disjoint
        ([1, 2, 3], [1, 2, 3], 1.0),  # numeric scalars
        ([{"a": 1}], [{"a": 1}], 1.0),  # object elements keyed by JSON
        ([{"a": 1}], [{"a": 2}], 0.0),  # differing objects don't overlap
        (None, ["a"], 0.0),  # null → empty set
        ("a", ["a"], 1.0),  # a bare scalar is a one-element set
        ({"a", "b"}, ["b", "a"], 1.0),  # an actual set is taken as-is
    ],
    ids=[
        "partial",
        "order",
        "duplicates",
        "both-empty",
        "one-empty",
        "disjoint",
        "numeric",
        "objects-equal",
        "objects-diff",
        "null",
        "bare-scalar",
        "set",
    ],
)
def test_jaccard(actual: Any, expected: Any, score: Any) -> None:
    assert ArrayJaccardSimilarity().score(actual, expected) == pytest.approx(score)


def test_runs_on_array_node() -> None:
    """As an ArrayMetric it cascades onto the array node and scores the whole list."""
    config = EvalConfig(
        fields={"tags": ArrayFieldConfig(metrics=[ArrayJaccardSimilarity()])}
    )
    root = build_tree({"tags": ["a", "b"]}, {"tags": ["b", "c"]}, config)
    tags = next(c for c in root.children_nodes() if c.path == "tags")
    assert float(tags.metric_results["array_jaccard_similarity"]) == pytest.approx(
        1 / 3
    )
