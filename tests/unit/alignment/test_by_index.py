"""ByIndexAligner — position-wise pairing; surplus on either side is unmatched.

Aligners only pair indices; value scoring happens later in the array metrics.
"""

import pytest

from structured_eval.alignment import ByIndexAligner
from structured_eval.model.config import ArrayStrategy

pytestmark = pytest.mark.unit


def test_equal_length_pairs_positionally() -> None:
    r = ByIndexAligner().align([1, 2, 3], [1, 2, 3])
    assert r.matched == [(0, 0), (1, 1), (2, 2)]
    assert r.missed == [] and r.spurious == []


def test_more_actual_is_spurious() -> None:
    r = ByIndexAligner().align([1, 2], [1, 2, 3])
    assert r.spurious == [2]
    assert r.missed == []


def test_more_expected_is_missed() -> None:
    r = ByIndexAligner().align([1, 2, 3], [1, 2])
    assert r.missed == [2]
    assert r.spurious == []


def test_strategy_recorded() -> None:
    assert ByIndexAligner().align([], []).strategy == ArrayStrategy.BY_INDEX
