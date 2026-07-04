"""ArrayJaccardSimilarity — set-overlap ``|A∩B|/|A∪B|`` over arrays.

Order- and count-insensitive; membership is exact equality. Built for arrays of
scalars, but does not crash on object/list elements (keyed by canonical JSON).
"""

from typing import Any

import pytest

from structured_eval.metrics import ArrayJaccardSimilarity

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
    ],
)
def test_jaccard(actual: Any, expected: Any, score: Any) -> None:
    assert ArrayJaccardSimilarity().score(actual, expected) == pytest.approx(score)
