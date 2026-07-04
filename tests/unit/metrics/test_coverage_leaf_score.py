"""CoverageLeafScore — fraction of expected leaves actually present (non-null).

Measures completeness, not correctness: a present-but-wrong value still counts
as covered; a missing or null value does not.
"""

from collections.abc import Callable

import pytest

from structured_eval.metrics import CoverageLeafScore
from structured_eval.models import EvalNode

pytestmark = pytest.mark.unit


def test_full_coverage(tree_factory: Callable[..., EvalNode]) -> None:
    root = tree_factory({"a": 1, "b": 2}, {"a": 1, "b": 2})
    assert CoverageLeafScore().compute(root) == 1.0


def test_missing_leaf_not_covered(tree_factory: Callable[..., EvalNode]) -> None:
    root = tree_factory({"a": 1}, {"a": 1, "b": 2})
    assert CoverageLeafScore().compute(root) == pytest.approx(0.5)


def test_null_leaf_not_covered(tree_factory: Callable[..., EvalNode]) -> None:
    root = tree_factory({"a": 1, "b": None}, {"a": 1, "b": 2})
    assert CoverageLeafScore().compute(root) == pytest.approx(0.5)


def test_no_expected_is_vacuously_one(tree_factory: Callable[..., EvalNode]) -> None:
    root = tree_factory({"a": 1}, {})
    assert CoverageLeafScore().compute(root) == 1.0
