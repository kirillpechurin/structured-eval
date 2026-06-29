"""ByKeyAligner — match by key, globally greedy best-first.

A whole-element key reduces to first-match; a named key reorders; a soft
(graded) key picks the strongest free partner, order-independently.
"""

import pytest

from structured_eval.alignment import ByKeyAligner
from structured_eval.model.config import ArrayStrategy

pytestmark = pytest.mark.unit


def test_whole_element_key() -> None:
    r = ByKeyAligner().align([1, 2], [2, 1])  # (expected, actual)
    # expected 1 pairs actual idx1; expected 2 pairs actual idx0
    assert sorted(r.matched) == [(0, 1), (1, 0)]
    assert r.missed == [] and r.spurious == []


def test_named_key_reorders() -> None:
    expected = [{"id": "a"}, {"id": "b"}]
    actual = [{"id": "b"}, {"id": "a"}]
    r = ByKeyAligner(key="id").align(expected, actual)
    assert sorted(r.matched) == [(0, 1), (1, 0)]


def test_missing_and_spurious() -> None:
    expected = [{"id": "a"}, {"id": "b"}]
    actual = [{"id": "a"}, {"id": "c"}]
    r = ByKeyAligner(key="id").align(expected, actual)
    assert r.matched == [(0, 0)]
    assert r.missed == [1]  # b
    assert r.spurious == [1]  # c


def test_duplicate_keys_paired_once() -> None:
    # one expected "a", two actual "a": one pairs, the other is spurious
    r = ByKeyAligner(key="id").align([{"id": "a"}], [{"id": "a"}, {"id": "a"}])
    assert r.matched == [(0, 0)]
    assert r.spurious == [1]


def test_soft_key_picks_best_not_first() -> None:
    # both actuals clear the threshold, but pairing is greedy best-first —
    # the closer id wins, not the first one seen.
    expected = [{"id": 10}]
    actual = [{"id": 5}, {"id": 9}]  # closeness 0.5 vs 0.9
    r = ByKeyAligner(key="id", key_metric="numeric_closeness", threshold=0.4).align(
        expected, actual
    )
    assert r.matched == [(0, 1)]  # 9 (best), not 5 (first)
    assert r.spurious == [0]


def test_soft_key_global_greedy_is_order_independent() -> None:
    # two expected each best-match a different actual; global best-first resolves
    # the assignment without an order bias.
    expected = [{"id": 10}, {"id": 8}]
    actual = [{"id": 8}, {"id": 10}]
    r = ByKeyAligner(key="id", key_metric="numeric_closeness", threshold=0.5).align(
        expected, actual
    )
    assert sorted(r.matched) == [(0, 1), (1, 0)]  # 10↔10, 8↔8
    assert r.missed == [] and r.spurious == []


def test_strategy_recorded() -> None:
    assert ByKeyAligner().align([], []).strategy == ArrayStrategy.BY_KEY
