"""Result models (model/result.py): EvalReport queries, asserts, diff,
serialization, plus BatchEvalReport / ConsistencyReport aggregates.

Reports are constructed directly (no engine) to isolate the model behaviour.
"""

from pathlib import Path
from typing import Any

import pytest

from structured_eval.models import (
    BatchEvalReport,
    ConsistencyReport,
    EvalReport,
    EvalWarning,
    FieldScore,
    MetricCollection,
    MetricResult,
    NodeType,
    WarningType,
)
from structured_eval.models.result import _percentile

pytestmark = pytest.mark.unit


def _coll(
    name: str, value: float, extra: dict[str, Any] | None = None
) -> MetricCollection:
    """A single-node MetricCollection rooted at "$" (document-level)."""
    return MetricCollection(name=name, by_path={"$": MetricResult(value, extra)})


def _fs(
    path: str,
    score: float,
    threshold: float = 1.0,
    metrics: dict[str, MetricResult] | None = None,
    actual: Any = None,
    expected: Any = None,
) -> FieldScore:
    return FieldScore(
        path=path,
        node_type=NodeType.SCALAR,
        actual=actual,
        expected=expected,
        metrics=metrics or {"exact_match": MetricResult(score)},
        score=score,
        threshold=threshold,
    )


def _report(
    score: float | None = None,
    metrics: dict[str, float] | None = None,
    fields: list[FieldScore] | None = None,
    **kwargs: Any,
) -> EvalReport:
    return EvalReport(
        score=score,
        metrics={k: _coll(k, v) for k, v in (metrics or {}).items()},
        field_scores={fs.path: fs for fs in (fields or [])},
        **kwargs,
    )


# ── failed_fields ────────────────────────────────────────────────────────────


def test_failed_fields_below_own_threshold() -> None:
    r = _report(fields=[_fs("a", 1.0), _fs("b", 0.0)])
    assert list(r.failed_fields()) == ["b"]
    assert r.failed_fields()["b"].path == "b"


def test_failed_fields_explicit_threshold() -> None:
    r = _report(fields=[_fs("a", 0.8, threshold=0.5)])
    assert r.failed_fields() == {}
    assert list(r.failed_fields(threshold=0.9)) == ["a"]


def test_failed_fields_skips_scoreless() -> None:
    fs = FieldScore(path="x", node_type=NodeType.OBJECT, score=None)
    assert EvalReport(field_scores={"x": fs}).failed_fields() == {}


def test_a_field_without_a_threshold_must_be_perfect() -> None:
    # No bar configured and none passed → the implicit bar is 1.0, the same
    # default every node carries.
    r = EvalReport(
        field_scores={
            p: FieldScore(path=p, node_type=NodeType.SCALAR, score=s, threshold=None)
            for p, s in (("a", 1.0), ("b", 0.99))
        }
    )
    assert list(r.failed_fields()) == ["b"]


# ── warnings ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("message", "rendered"),
    [
        ("'extra' not in expected", "[EXTRA_KEY] 'extra' not in expected"),
        ("", "[EXTRA_KEY] vendor.extra"),
    ],
    ids=["with-message", "bare"],
)
def test_a_warning_reads_as_a_line(message: str, rendered: str) -> None:
    warning = EvalWarning(
        type=WarningType.EXTRA_KEY, path="vendor.extra", message=message
    )
    assert str(warning) == rendered


# ── assertions ───────────────────────────────────────────────────────────────


def test_assert_score_pass_and_fail() -> None:
    r = _report(score=0.9)
    r.assert_score(0.8)
    with pytest.raises(AssertionError):
        r.assert_score(0.95)


def test_assert_score_without_score() -> None:
    with pytest.raises(AssertionError, match="no score"):
        _report(score=None).assert_score(0.5)


def test_assert_no_parse_errors() -> None:
    EvalReport().assert_no_parse_errors()
    with pytest.raises(AssertionError):
        EvalReport(parse_error=True, parse_error_message="bad").assert_no_parse_errors()


def test_assert_field() -> None:
    r = _report(fields=[_fs("a", 1.0)])
    r.assert_field("a", 0.9)
    with pytest.raises(AssertionError):
        r.assert_field("a", 1.5)
    with pytest.raises(AssertionError, match="no field"):
        r.assert_field("zzz", 0.1)


def test_assert_field_on_a_scoreless_field() -> None:
    # An object node with no key metric applied has no number to assert on, and
    # saying "it scored 0" would be a different claim entirely.
    fs = FieldScore(path="vendor", node_type=NodeType.OBJECT, score=None)
    with pytest.raises(AssertionError, match="has no score"):
        EvalReport(field_scores={"vendor": fs}).assert_field("vendor", 0.5)


def test_assert_metric() -> None:
    r = _report(metrics={"object_f1": 0.7})
    r.assert_metric("object_f1", 0.5)
    with pytest.raises(AssertionError):
        r.assert_metric("object_f1", 0.9)
    with pytest.raises(AssertionError, match="not computed"):
        r.assert_metric("missing", 0.1)


def test_assert_schema_valid() -> None:
    EvalReport(
        metrics={"schema_validity": _coll("schema_validity", 1.0)}
    ).assert_schema_valid()
    bad = _coll("schema_validity", 0.0, {"schema_errors": ["type: total"]})
    with pytest.raises(AssertionError):
        EvalReport(metrics={"schema_validity": bad}).assert_schema_valid()


def test_assert_schema_valid_without_the_metric_passes() -> None:
    # Nobody asked for schema validation, so there is nothing to fail — as
    # opposed to validation that ran and found nothing wrong.
    EvalReport().assert_schema_valid()


# ── diff ─────────────────────────────────────────────────────────────────────


def test_diff_metric_deltas() -> None:
    diff = _report(metrics={"object_f1": 0.9}).diff_from(
        _report(metrics={"object_f1": 0.7})
    )
    assert diff.deltas["object_f1"] == pytest.approx(0.2)


def test_diff_field_deltas() -> None:
    diff = _report(fields=[_fs("x", 1.0)]).diff_from(_report(fields=[_fs("x", 0.0)]))
    assert diff.field_deltas["x"]["score"] == pytest.approx(1.0)


def test_diff_skips_a_field_the_other_run_does_not_have() -> None:
    # The two documents disagree about which fields exist; there is no baseline
    # to subtract, and inventing one would report a delta from nothing.
    diff = _report(fields=[_fs("x", 1.0), _fs("new", 1.0)]).diff_from(
        _report(fields=[_fs("x", 0.0)])
    )
    assert set(diff.field_deltas) == {"x"}


def test_diff_of_two_scoreless_fields_is_no_entry_at_all() -> None:
    # Neither the score nor any metric can be subtracted, so the field carries
    # no delta rather than an empty one.
    scoreless = FieldScore(path="v", node_type=NodeType.OBJECT, score=None, metrics={})
    a = EvalReport(field_scores={"v": scoreless})
    assert a.diff_from(a).field_deltas == {}


def test_diff_metric_subset() -> None:
    a = _report(metrics={"object_f1": 0.9, "coverage_leaf_score": 1.0})
    b = _report(metrics={"object_f1": 0.7, "coverage_leaf_score": 0.5})
    diff = a.diff_from(b, metrics=["coverage_leaf_score"])
    assert set(diff.deltas) == {"coverage_leaf_score"}


# ── serialization ────────────────────────────────────────────────────────────


def test_to_dict_is_jsonable() -> None:
    d = _report(score=0.5, metrics={"object_f1": 0.5}, fields=[_fs("a", 0.5)]).to_dict()
    assert d["score"] == 0.5
    assert d["metrics"]["object_f1"]["by_path"]["$"] == 0.5


def test_json_file_roundtrip(tmp_path: Path) -> None:
    r = _report(score=0.5, metrics={"object_f1": 0.5}, fields=[_fs("a", 0.5)])
    path = tmp_path / "report.json"
    r.to_json(str(path))
    loaded = EvalReport.from_json(str(path))
    assert loaded.score == 0.5
    assert loaded.metrics["object_f1"].root() == 0.5


def test_from_dict() -> None:
    assert EvalReport.from_dict(_report(score=1.0).to_dict()).score == 1.0


# ── batch / consistency aggregates ───────────────────────────────────────────


def test_batch_field_breakdown() -> None:
    batch = BatchEvalReport(
        per_sample=[_report(fields=[_fs("a", 1.0)]), _report(fields=[_fs("a", 0.0)])]
    )
    bd = batch.field_breakdown()
    assert bd["a"]["mean"] == pytest.approx(0.5)
    assert bd["a"]["min"] == 0.0
    assert bd["a"]["max"] == 1.0
    assert bd["a"]["fail_rate"] == pytest.approx(0.5)


def test_batch_breakdown_skips_parse_errors() -> None:
    reports = [EvalReport(parse_error=True), _report(fields=[_fs("a", 1.0)])]
    bd = BatchEvalReport(per_sample=reports).field_breakdown()
    assert bd["a"]["mean"] == 1.0


def test_batch_breakdown_skips_scoreless_fields() -> None:
    scoreless = FieldScore(path="v", node_type=NodeType.OBJECT, score=None)
    batch = BatchEvalReport(
        per_sample=[
            EvalReport(field_scores={"v": scoreless, "a": _fs("a", 1.0)}),
        ]
    )
    assert set(batch.field_breakdown()) == {"a"}


def test_batch_breakdown_defaults_the_bar_to_one() -> None:
    # Same rule as `failed_fields`: no threshold anywhere means perfection.
    unbarred = FieldScore(
        path="a", node_type=NodeType.SCALAR, score=0.99, threshold=None
    )
    batch = BatchEvalReport(per_sample=[EvalReport(field_scores={"a": unbarred})])
    assert batch.field_breakdown()["a"]["fail_rate"] == 1.0


@pytest.mark.parametrize(
    "report",
    [
        BatchEvalReport(per_sample=[]),
        ConsistencyReport(per_run=[]),
    ],
    ids=["batch", "consistency"],
)
def test_aggregate_reports_print_a_summary(
    report: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    report.print_summary()
    assert capsys.readouterr().out.strip()


def test_consistency_stable_vs_unstable() -> None:
    report = ConsistencyReport(
        field_variance={"a": 0.0, "b": 0.5},
        stable_fields=["a"],
        unstable_fields=["b"],
    )
    assert report.stable_fields == ["a"]
    assert report.unstable_fields == ["b"]


@pytest.mark.parametrize(
    ("values", "pct", "expected"),
    [([1.0], 0.95, 1.0), ([0.0, 1.0], 0.5, 0.5), ([0.0, 10.0], 0.95, 9.5)],
    ids=["single", "median", "p95"],
)
def test_percentile_helper(values: Any, pct: Any, expected: Any) -> None:
    assert _percentile(values, pct) == pytest.approx(expected)
