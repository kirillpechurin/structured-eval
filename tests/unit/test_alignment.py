"""Unit tests for array alignment strategies (the only role left of a matcher).

Aligners only pair indices; value scoring happens later in the array metrics.
"""

from __future__ import annotations

import pytest

from structured_eval.alignment import (
    ByIndexAligner,
    ByKeyAligner,
    HungarianAligner,
    make_aligner,
)
from structured_eval.model.config import ArrayStrategy

pytestmark = pytest.mark.unit


class TestByIndex:
    def test_equal_length(self):
        r = ByIndexAligner().align([1, 2, 3], [1, 2, 3])
        assert r.matched == [(0, 0), (1, 1), (2, 2)]
        assert r.missed == [] and r.spurious == []

    def test_more_actual_is_spurious(self):
        r = ByIndexAligner().align([1, 2], [1, 2, 3])
        assert r.spurious == [2]
        assert r.missed == []

    def test_more_expected_is_missed(self):
        r = ByIndexAligner().align([1, 2, 3], [1, 2])
        assert r.missed == [2]
        assert r.spurious == []

    def test_strategy_recorded(self):
        assert ByIndexAligner().align([], []).strategy == ArrayStrategy.BY_INDEX


class TestByKey:
    def test_whole_element_key(self):
        r = ByKeyAligner().align([1, 2], [2, 1])  # expected, actual
        # greedy: expected 1 pairs actual idx1; expected 2 pairs actual idx0
        assert sorted(r.matched) == [(0, 1), (1, 0)]
        assert r.missed == [] and r.spurious == []

    def test_named_key_reorders(self):
        expected = [{"id": "a"}, {"id": "b"}]
        actual = [{"id": "b"}, {"id": "a"}]
        r = ByKeyAligner(key="id").align(expected, actual)
        assert sorted(r.matched) == [(0, 1), (1, 0)]

    def test_missing_and_spurious(self):
        expected = [{"id": "a"}, {"id": "b"}]
        actual = [{"id": "a"}, {"id": "c"}]
        r = ByKeyAligner(key="id").align(expected, actual)
        assert r.matched == [(0, 0)]
        assert r.missed == [1]  # b
        assert r.spurious == [1]  # c

    def test_strategy_recorded(self):
        assert ByKeyAligner().align([], []).strategy == ArrayStrategy.BY_KEY

    def test_duplicate_keys_paired_once(self):
        # one expected "a", two actual "a": only one pairs, the other is spurious
        r = ByKeyAligner(key="id").align([{"id": "a"}], [{"id": "a"}, {"id": "a"}])
        assert r.matched == [(0, 0)]
        assert r.spurious == [1]


class TestHungarian:
    def test_strategy_recorded(self):
        assert HungarianAligner().align([1], [1]).strategy == ArrayStrategy.HUNGARIAN

    def test_optimal_reorder(self):
        # numeric closeness pairs equal values regardless of order
        r = HungarianAligner().align([10, 20, 30], [30, 10, 20])
        assert sorted(r.matched) == [(0, 1), (1, 2), (2, 0)]
        assert r.missed == [] and r.spurious == []

    def test_below_threshold_unmatched(self):
        # 100 vs 5 similarity is far below 0.8 → missed + spurious, no match
        r = HungarianAligner(threshold=0.8).align([100], [5])
        assert r.matched == []
        assert r.missed == [0] and r.spurious == [0]

    def test_empty_sides(self):
        r = HungarianAligner().align([1, 2], [])
        assert r.matched == [] and r.missed == [0, 1] and r.spurious == []

    def test_dict_elements_mean_agreement(self):
        expected = [{"id": "a", "v": 1}, {"id": "b", "v": 2}]
        actual = [{"id": "b", "v": 2}, {"id": "a", "v": 1}]
        r = HungarianAligner(threshold=0.9).align(expected, actual)
        assert sorted(r.matched) == [(0, 1), (1, 0)]

    def test_scorer_on_key_field(self):
        expected = [{"id": "acme"}, {"id": "globex"}]
        actual = [{"id": "globe"}, {"id": "acme"}]
        r = HungarianAligner(key="id", threshold=0.6).align(expected, actual)
        assert sorted(r.matched) == [(0, 1), (1, 0)]

    def test_per_field_scorer_dict(self):
        # a dict scorer compares arrays of objects field by field
        expected = [{"name": "acme", "amount": 100}, {"name": "globex", "amount": 200}]
        actual = [{"name": "globe", "amount": 205}, {"name": "acme", "amount": 99}]
        r = HungarianAligner(
            scorer={"name": "fuzzy", "amount": "numeric_closeness"}, threshold=0.7
        ).align(expected, actual)
        assert sorted(r.matched) == [(0, 1), (1, 0)]


class TestMakeAligner:
    def test_by_index(self):
        assert isinstance(make_aligner(ArrayStrategy.BY_INDEX), ByIndexAligner)

    def test_hungarian(self):
        aligner = make_aligner(ArrayStrategy.HUNGARIAN, {"threshold": 0.9})
        assert isinstance(aligner, HungarianAligner)

    def test_by_key(self):
        aligner = make_aligner(ArrayStrategy.BY_KEY, {"key": "id"})
        assert isinstance(aligner, ByKeyAligner)

    def test_metric_by_name(self):
        # key_metric given as a registered name string is resolved to an instance
        aligner = make_aligner(ArrayStrategy.BY_KEY, {"key": "id", "key_metric": "exact_match"})
        assert isinstance(aligner, ByKeyAligner)
