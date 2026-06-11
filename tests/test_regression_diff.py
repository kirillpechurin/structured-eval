import pytest

from structured_eval.core.config import EvalConfig, MatchMode
from structured_eval.core.result import (
    EvalReport,
    FieldScore,
    FieldStatus,
    RegressionDiff,
)


def _make_report(
    f1: float | None = None,
    precision: float | None = None,
    recall: float | None = None,
    field_scores: dict[str, float] | None = None,
) -> EvalReport:
    fs: dict[str, FieldScore] = {}
    for key, score in (field_scores or {}).items():
        fs[key] = FieldScore(
            key=key,
            expected="x",
            actual="x",
            score=score,
            matcher=str(MatchMode.EXACT),
            status=FieldStatus.MATCH if score == 1.0 else FieldStatus.PARTIAL,
        )
    return EvalReport(f1=f1, precision=precision, recall=recall, field_scores=fs)


# ── diff_from: aggregate metrics ──────────────────────────────────────────────


class TestDiffFromAggregates:
    def test_improvement(self):
        new = _make_report(f1=0.9, precision=0.95, recall=0.85)
        old = _make_report(f1=0.8, precision=0.85, recall=0.75)
        diff = new.diff_from(old)
        assert diff.f1_delta == pytest.approx(0.1)
        assert diff.precision_delta == pytest.approx(0.1)
        assert diff.recall_delta == pytest.approx(0.1)

    def test_regression(self):
        new = _make_report(f1=0.7, precision=0.7, recall=0.7)
        old = _make_report(f1=0.9, precision=0.9, recall=0.9)
        diff = new.diff_from(old)
        assert diff.f1_delta == pytest.approx(-0.2)

    def test_no_change(self):
        r = _make_report(f1=0.85, precision=0.85, recall=0.85)
        diff = r.diff_from(r)
        assert diff.f1_delta == pytest.approx(0.0)
        assert diff.precision_delta == pytest.approx(0.0)
        assert diff.recall_delta == pytest.approx(0.0)

    def test_both_f1_none_returns_none(self):
        new = _make_report(f1=None)
        old = _make_report(f1=None)
        diff = new.diff_from(old)
        assert diff.f1_delta is None
        assert diff.precision_delta is None
        assert diff.recall_delta is None

    def test_f1_gained(self):
        # old had no f1 (schema-only), new has f1
        new = _make_report(f1=0.8, precision=0.8, recall=0.8)
        old = _make_report(f1=None, precision=None, recall=None)
        diff = new.diff_from(old)
        assert diff.f1_delta == pytest.approx(0.8)

    def test_f1_lost(self):
        new = _make_report(f1=None)
        old = _make_report(f1=0.8)
        diff = new.diff_from(old)
        assert diff.f1_delta == pytest.approx(-0.8)


# ── diff_from: field_deltas ───────────────────────────────────────────────────


class TestDiffFromFieldDeltas:
    def test_improved_field(self):
        new = _make_report(field_scores={"vendor": 1.0, "total": 0.9})
        old = _make_report(field_scores={"vendor": 0.5, "total": 0.9})
        diff = new.diff_from(old)
        assert diff.field_deltas["vendor"] == pytest.approx(0.5)
        assert diff.field_deltas["total"] == pytest.approx(0.0)

    def test_regressed_field(self):
        new = _make_report(field_scores={"date": 0.2})
        old = _make_report(field_scores={"date": 0.8})
        diff = new.diff_from(old)
        assert diff.field_deltas["date"] == pytest.approx(-0.6)

    def test_field_added_in_new(self):
        # field present only in new → delta = score - 0.0
        new = _make_report(field_scores={"extra": 0.7})
        old = _make_report(field_scores={})
        diff = new.diff_from(old)
        assert diff.field_deltas["extra"] == pytest.approx(0.7)

    def test_field_missing_in_new(self):
        # field present only in old → delta = 0.0 - score
        new = _make_report(field_scores={})
        old = _make_report(field_scores={"removed": 0.6})
        diff = new.diff_from(old)
        assert diff.field_deltas["removed"] == pytest.approx(-0.6)

    def test_field_deltas_sorted(self):
        new = _make_report(field_scores={"z": 1.0, "a": 0.5, "m": 0.8})
        old = _make_report(field_scores={"z": 0.5, "a": 0.5, "m": 0.5})
        diff = new.diff_from(old)
        assert list(diff.field_deltas.keys()) == sorted(diff.field_deltas.keys())

    def test_empty_field_scores(self):
        new = _make_report(f1=0.9)
        old = _make_report(f1=0.8)
        diff = new.diff_from(old)
        assert diff.field_deltas == {}

    def test_union_of_fields(self):
        new = _make_report(field_scores={"a": 1.0, "b": 0.5})
        old = _make_report(field_scores={"b": 0.8, "c": 0.3})
        diff = new.diff_from(old)
        assert set(diff.field_deltas.keys()) == {"a", "b", "c"}


# ── failed_fields and to_dict with None ───────────────────────────────────────


class TestFailedFields:
    def test_returns_only_failed(self):
        report = _make_report(f1=0.5, field_scores={"ok": 1.0, "bad": 0.3, "partial": 0.7})
        failed = report.failed_fields()
        keys = {f.key for f in failed}
        assert keys == {"bad", "partial"}
        assert "ok" not in keys

    def test_empty_when_all_perfect(self):
        report = _make_report(f1=1.0, field_scores={"a": 1.0, "b": 1.0})
        assert report.failed_fields() == []

    def test_empty_field_scores(self):
        report = _make_report(f1=None)
        assert report.failed_fields() == []


class TestToDict:
    def test_none_fields_serialized(self):
        report = _make_report(f1=None, precision=None, recall=None)
        d = report.to_dict()
        assert d["f1"] is None
        assert d["precision"] is None
        assert d["recall"] is None
        assert d["perfect"] is None
        assert d["faithfulness_score"] is None

    def test_float_fields_serialized(self):
        report = _make_report(f1=0.8, precision=0.9, recall=0.72)
        d = report.to_dict()
        assert d["f1"] == pytest.approx(0.8)
        assert d["precision"] == pytest.approx(0.9)
        assert d["recall"] == pytest.approx(0.72)

    def test_returns_dict_type(self):
        report = _make_report()
        assert isinstance(report.to_dict(), dict)
