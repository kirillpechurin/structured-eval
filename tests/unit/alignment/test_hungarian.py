"""HungarianAligner — optimal one-to-one assignment (scipy).

Owns its element-similarity logic and a ``Scorer`` type: a single
metric/name/callable, or a per-field ``dict[str, Scorer]``.
"""

import pytest

from structured_eval.alignment import HungarianAligner
from structured_eval.models.config import ArrayStrategy

pytestmark = pytest.mark.unit


def test_strategy_recorded() -> None:
    assert HungarianAligner().align([1], [1]).strategy == ArrayStrategy.HUNGARIAN


def test_optimal_reorder() -> None:
    # numeric closeness pairs equal values regardless of order
    r = HungarianAligner().align([10, 20, 30], [30, 10, 20])
    assert sorted(r.matched) == [(0, 1), (1, 2), (2, 0)]
    assert r.missed == []
    assert r.spurious == []


def test_below_threshold_unmatched() -> None:
    # 100 vs 5 similarity is far below 0.8 → missed + spurious, no match
    r = HungarianAligner(threshold=0.8).align([100], [5])
    assert r.matched == []
    assert r.missed == [0]
    assert r.spurious == [0]


def test_empty_actual_side() -> None:
    r = HungarianAligner().align([1, 2], [])
    assert r.matched == []
    assert r.missed == [0, 1]
    assert r.spurious == []


def test_dict_elements_mean_agreement() -> None:
    expected = [{"id": "a", "v": 1}, {"id": "b", "v": 2}]
    actual = [{"id": "b", "v": 2}, {"id": "a", "v": 1}]
    r = HungarianAligner(threshold=0.9).align(expected, actual)
    assert sorted(r.matched) == [(0, 1), (1, 0)]


def test_scorer_on_key_field() -> None:
    expected = [{"id": "acme"}, {"id": "globex"}]
    actual = [{"id": "globex"}, {"id": "acme"}]
    r = HungarianAligner(key="id", threshold=0.6).align(expected, actual)
    assert sorted(r.matched) == [(0, 1), (1, 0)]


def test_per_field_scorer_dict() -> None:
    # a dict scorer compares arrays of objects field by field
    expected = [{"name": "acme", "amount": 100}, {"name": "globex", "amount": 200}]
    actual = [{"name": "globe", "amount": 205}, {"name": "acme", "amount": 99}]
    r = HungarianAligner(
        scorer={"name": "fuzzy", "amount": "numeric_closeness"}, threshold=0.7
    ).align(expected, actual)
    assert sorted(r.matched) == [(0, 1), (1, 0)]
