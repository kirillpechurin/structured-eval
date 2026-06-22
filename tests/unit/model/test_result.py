"""Unit tests for the result models: EvalReport queries, asserts, diff,
serialization, plus BatchEvalReport / ConsistencyReport aggregates.

Reports are constructed directly (no engine) to isolate the model behaviour.
"""

from __future__ import annotations

import pytest

from structured_eval import (
    BatchEvalReport,
    ConsistencyReport,
    EvalReport,
    FieldScore,
    MetricCollection,
    MetricResult,
)
from structured_eval.model.result import NodeType

pytestmark = pytest.mark.unit


def _coll(name, value, extra=None):
    """A single-node MetricCollection rooted at "$" (document-level)."""
    return MetricCollection(name=name, by_path={"$": MetricResult(value, extra)})


def _fs(path, score, threshold=1.0, metrics=None, actual=None, expected=None):
    return FieldScore(
        path=path,
        node_type=NodeType.SCALAR,
        actual=actual,
        expected=expected,
        metrics=metrics or {"exact_match": score},
        score=score,
        threshold=threshold,
    )


def _report(score=None, metrics=None, fields=None, **kwargs):
    return EvalReport(
        score=score,
        metrics={k: _coll(k, v) for k, v in (metrics or {}).items()},
        field_scores={fs.path: fs for fs in (fields or [])},
        **kwargs,
    )


class TestFailedFields:
    def test_below_own_threshold(self):
        r = _report(fields=[_fs("a", 1.0), _fs("b", 0.0)])
        assert list(r.failed_fields()) == ["b"]
        assert r.failed_fields()["b"].path == "b"

    def test_explicit_threshold(self):
        r = _report(fields=[_fs("a", 0.8, threshold=0.5)])
        assert r.failed_fields() == {}
        assert list(r.failed_fields(threshold=0.9)) == ["a"]

    def test_skips_scoreless(self):
        fs = FieldScore(path="x", node_type=NodeType.OBJECT, score=None)
        r = EvalReport(field_scores={"x": fs})
        assert r.failed_fields() == {}


class TestAssertions:
    def test_assert_score_pass_and_fail(self):
        r = _report(score=0.9)
        r.assert_score(0.8)
        with pytest.raises(AssertionError):
            r.assert_score(0.95)

    def test_assert_score_no_score(self):
        with pytest.raises(AssertionError, match="no score"):
            _report(score=None).assert_score(0.5)

    def test_assert_no_parse_errors(self):
        EvalReport().assert_no_parse_errors()
        with pytest.raises(AssertionError):
            EvalReport(parse_error=True, parse_error_message="bad").assert_no_parse_errors()

    def test_assert_field(self):
        r = _report(fields=[_fs("a", 1.0)])
        r.assert_field("a", 0.9)
        with pytest.raises(AssertionError):
            r.assert_field("a", 1.5)
        with pytest.raises(AssertionError, match="no field"):
            r.assert_field("zzz", 0.1)

    def test_assert_metric(self):
        r = _report(metrics={"object_f1": 0.7})
        r.assert_metric("object_f1", 0.5)
        with pytest.raises(AssertionError):
            r.assert_metric("object_f1", 0.9)
        with pytest.raises(AssertionError, match="not computed"):
            r.assert_metric("missing", 0.1)

    def test_assert_schema_valid(self):
        EvalReport(metrics={"schema_validity": _coll("schema_validity", 1.0)}).assert_schema_valid()
        with pytest.raises(AssertionError):
            bad = _coll("schema_validity", 0.0, {"schema_errors": ["type: total"]})
            EvalReport(metrics={"schema_validity": bad}).assert_schema_valid()


class TestDiff:
    def test_metric_deltas(self):
        a = _report(metrics={"object_f1": 0.9})
        b = _report(metrics={"object_f1": 0.7})
        diff = a.diff_from(b)
        assert diff.deltas["object_f1"] == pytest.approx(0.2)

    def test_field_deltas(self):
        a = _report(fields=[_fs("x", 1.0)])
        b = _report(fields=[_fs("x", 0.0)])
        diff = a.diff_from(b)
        assert diff.field_deltas["x"]["score"] == pytest.approx(1.0)

    def test_metric_subset(self):
        a = _report(metrics={"object_f1": 0.9, "coverage_leaf_score": 1.0})
        b = _report(metrics={"object_f1": 0.7, "coverage_leaf_score": 0.5})
        diff = a.diff_from(b, metrics=["coverage_leaf_score"])
        assert set(diff.deltas) == {"coverage_leaf_score"}


class TestSerialization:
    def test_to_dict_jsonable(self):
        r = _report(score=0.5, metrics={"object_f1": 0.5}, fields=[_fs("a", 0.5)])
        d = r.to_dict()
        assert d["score"] == 0.5
        assert d["metrics"]["object_f1"]["by_path"]["$"] == 0.5

    def test_json_roundtrip(self, tmp_path):
        r = _report(score=0.5, metrics={"object_f1": 0.5}, fields=[_fs("a", 0.5)])
        path = tmp_path / "report.json"
        r.to_json(str(path))
        loaded = EvalReport.from_json(str(path))
        assert loaded.score == 0.5
        assert loaded.metrics["object_f1"].root() == 0.5

    def test_from_dict(self):
        r = _report(score=1.0)
        assert EvalReport.from_dict(r.to_dict()).score == 1.0


class TestBatch:
    def test_field_breakdown(self):
        reports = [
            _report(fields=[_fs("a", 1.0)]),
            _report(fields=[_fs("a", 0.0)]),
        ]
        batch = BatchEvalReport(per_sample=reports)
        bd = batch.field_breakdown()
        assert bd["a"]["mean"] == pytest.approx(0.5)
        assert bd["a"]["min"] == 0.0
        assert bd["a"]["max"] == 1.0
        assert bd["a"]["fail_rate"] == pytest.approx(0.5)

    def test_breakdown_skips_parse_errors(self):
        reports = [
            EvalReport(parse_error=True),
            _report(fields=[_fs("a", 1.0)]),
        ]
        bd = BatchEvalReport(per_sample=reports).field_breakdown()
        assert bd["a"]["mean"] == 1.0


class TestConsistency:
    def test_stable_vs_unstable(self):
        report = ConsistencyReport(
            field_variance={"a": 0.0, "b": 0.5},
            stable_fields=["a"],
            unstable_fields=["b"],
        )
        assert report.stable_fields == ["a"]
        assert report.unstable_fields == ["b"]


def test_percentile_helper():
    from structured_eval.model.result import _percentile

    assert _percentile([1.0], 0.95) == 1.0
    assert _percentile([0.0, 1.0], 0.5) == pytest.approx(0.5)
    assert _percentile([0.0, 10.0], 0.95) == pytest.approx(9.5)
