"""Unit tests for array alignment strategies (the only role left of a matcher).

Aligners only pair indices; value scoring happens later in the array metrics.
"""

from __future__ import annotations

import pytest

from structured_eval.alignment import (
    ByIndexAligner,
    ByKeyAligner,
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


class TestMakeAligner:
    def test_by_index(self):
        assert isinstance(make_aligner(ArrayStrategy.BY_INDEX), ByIndexAligner)

    def test_by_key(self):
        aligner = make_aligner(ArrayStrategy.BY_KEY, {"key": "id"})
        assert isinstance(aligner, ByKeyAligner)

    def test_metric_by_name(self):
        # key_metric given as a registered name string is resolved to an instance
        aligner = make_aligner(ArrayStrategy.BY_KEY, {"key": "id", "key_metric": "exact_match"})
        assert isinstance(aligner, ByKeyAligner)
