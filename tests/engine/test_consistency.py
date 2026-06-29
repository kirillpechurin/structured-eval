"""Tests for evaluate_consistency → ConsistencyReport (run-to-run stability)."""

from typing import Any

import pytest

from structured_eval import EvalConfig, ObjectF1, Sample, evaluate_consistency

pytestmark = pytest.mark.engine


def _runs(actuals: list[Any], expected: Any) -> list[Sample]:
    return [Sample(actual=a, expected=expected) for a in actuals]


def test_all_stable() -> None:
    runs = _runs([{"a": 1}, {"a": 1}, {"a": 1}], {"a": 1})
    r = evaluate_consistency(runs)
    assert r.stable_fields == ["a"]
    assert r.unstable_fields == []
    assert r.field_variance["a"] == 0.0


def test_unstable_field_detected() -> None:
    runs = _runs([{"a": 1}, {"a": 9}, {"a": 1}], {"a": 1})
    r = evaluate_consistency(runs)
    assert "a" in r.unstable_fields


def test_variance_threshold_respected() -> None:
    runs = _runs([{"a": 1}, {"a": 9}], {"a": 1})
    # huge threshold → everything stable
    r = evaluate_consistency(runs, variance_threshold=1.0)
    assert "a" in r.stable_fields


def test_score_variance() -> None:
    runs = _runs([{"a": 1}, {"a": 9}], {"a": 1})
    r = evaluate_consistency(runs, config=EvalConfig(key_metric=ObjectF1()))
    assert r.mean_score == pytest.approx(0.5)
    assert r.score_variance is not None
    assert r.score_variance > 0.0


def test_parse_errors_excluded() -> None:
    runs = [
        Sample(actual="{bad", expected={"a": 1}),
        Sample(actual={"a": 1}, expected={"a": 1}),
    ]
    r = evaluate_consistency(runs)
    # only the parseable run contributes
    assert r.field_variance["a"] == 0.0
