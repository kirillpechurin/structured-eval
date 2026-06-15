"""Tests for batch aggregation via evaluate(list[Sample]) → BatchEvalReport."""

from __future__ import annotations

import pytest

from structured_eval import (
    BatchEvalReport,
    EvalConfig,
    ObjectF1,
    Sample,
    evaluate,
)

pytestmark = pytest.mark.engine


def _batch(samples, cfg=None) -> BatchEvalReport:
    report = evaluate(samples, config=cfg)
    assert isinstance(report, BatchEvalReport)
    return report


def test_batch_mean_metrics():
    samples = [
        Sample(actual={"a": 1}, expected={"a": 1}),   # f1 1.0
        Sample(actual={"a": 9}, expected={"a": 1}),   # f1 0.0
    ]
    r = _batch(samples, EvalConfig(metrics=[ObjectF1()]))
    assert r.metrics["object_f1"] == pytest.approx(0.5)
    assert len(r.per_sample) == 2


def test_perfect_response_rate():
    samples = [
        Sample(actual={"a": 1}, expected={"a": 1}),
        Sample(actual={"a": 9}, expected={"a": 1}),
    ]
    r = _batch(samples, EvalConfig(metrics=[ObjectF1()]))
    assert r.perfect_response_rate == pytest.approx(0.5)


def test_parse_error_rate():
    samples = [
        Sample(actual="{bad", expected={"a": 1}),
        Sample(actual={"a": 1}, expected={"a": 1}),
    ]
    r = _batch(samples, EvalConfig(metrics=[ObjectF1()]))
    assert r.parse_error_rate == pytest.approx(0.5)


def test_score_from_key_metric():
    samples = [
        Sample(actual={"a": 1}, expected={"a": 1}),
        Sample(actual={"a": 9}, expected={"a": 1}),
    ]
    r = _batch(samples, EvalConfig(key_metric=ObjectF1()))
    assert r.score == pytest.approx(0.5)
    assert r.score_label == "object_f1"


def test_field_breakdown_across_batch():
    samples = [
        Sample(actual={"a": 1}, expected={"a": 1}),
        Sample(actual={"a": 9}, expected={"a": 1}),
    ]
    r = _batch(samples, EvalConfig(metrics=[ObjectF1()]))
    bd = r.field_breakdown()
    assert bd["a"]["mean"] == pytest.approx(0.5)
    assert bd["a"]["fail_rate"] == pytest.approx(0.5)


def test_positional_config_tolerated():
    # evaluate(samples, cfg) positionally — cfg lands in `expected` slot
    samples = [Sample(actual={"a": 1}, expected={"a": 1})]
    r = evaluate(samples, EvalConfig(metrics=[ObjectF1()]))
    assert isinstance(r, BatchEvalReport)
    assert r.metrics["object_f1"] == 1.0


def test_empty_batch():
    r = _batch([], EvalConfig(metrics=[ObjectF1()]))
    assert r.per_sample == []
    assert r.parse_error_rate == 0.0
