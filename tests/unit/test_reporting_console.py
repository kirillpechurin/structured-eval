"""ConsoleRenderer — structural checks on the rendered report text.

Not a byte snapshot: we assert the rendered text *contains* the pieces a reader
relies on (overall verdict, score, every field row, pass/fail marks), so layout
drops are caught without pinning exact whitespace.
"""

from typing import Any

import pytest

from structured_eval import evaluate, evaluate_batch
from structured_eval.metrics import ObjectF1
from structured_eval.models import (
    BatchEvalReport,
    ConsistencyReport,
    EvalConfig,
    EvalReport,
    FieldScore,
    MetricCollection,
    MetricResult,
    NodeType,
    Sample,
)
from structured_eval.reporting.console import ConsoleRenderer

pytestmark = pytest.mark.unit


def _render(actual: Any, expected: Any, **kw: Any) -> str:
    """The console rendering of one evaluation."""
    report = evaluate(actual, expected, config=EvalConfig(key_metric=ObjectF1()), **kw)
    return ConsoleRenderer().render(report)


def test_header_and_score_present() -> None:
    out = _render({"id": "INV-1", "total": 99.0}, {"id": "INV-1", "total": 100.0})
    assert "OVERALL" in out
    assert "object_f1" in out  # the key-metric label
    assert "0.50" in out  # 1/2 fields correct


def test_every_field_has_a_row() -> None:
    out = _render(
        {"id": "INV-1", "total": 99.0, "vendor": "Acme"},
        {"id": "INV-1", "total": 100.0, "vendor": "Acme"},
    )
    for field in ("id", "total", "vendor"):
        assert field in out


def test_pass_and_fail_marks_rendered() -> None:
    out = _render({"a": 1, "b": 2}, {"a": 1, "b": 99})
    assert "✓" in out  # a correct
    assert "✗" in out  # b wrong


def test_perfect_report_has_no_fail_mark() -> None:
    out = _render({"a": 1}, {"a": 1})
    assert "✗" not in out
    assert "✓" in out


def test_parse_error_render() -> None:
    out = ConsoleRenderer().render(
        EvalReport(parse_error=True, parse_error_message="unexpected token")
    )
    assert "PARSE ERROR" in out
    assert "unexpected token" in out


def test_render_is_nonempty_str() -> None:
    assert _render({"a": 1}, {"a": 1}).strip()


def test_batch_report_renders() -> None:
    report = evaluate_batch(
        [Sample(actual={"a": 1}, expected={"a": 1})], EvalConfig(key_metric=ObjectF1())
    )
    assert ConsoleRenderer().render(report)


def test_print_summary_writes_to_stdout(capsys: pytest.CaptureFixture[str]) -> None:
    evaluate(
        {"a": 1}, {"a": 1}, config=EvalConfig(key_metric=ObjectF1())
    ).print_summary()
    assert capsys.readouterr().out


# ── nothing to show ──────────────────────────────────────────────────────────


def test_a_report_without_ground_truth_says_so() -> None:
    # Source-only mode: there is no expected document, so no overall score —
    # and a bare "0.00" would read as "everything is wrong".
    out = ConsoleRenderer().render(EvalReport())
    assert "no ground truth" in out
    assert "OVERALL" in out


def test_an_empty_report_renders_without_a_table() -> None:
    out = ConsoleRenderer().render(EvalReport())
    assert "Field" not in out  # no rows → no header either


def test_a_scoreless_field_is_left_out_of_the_table() -> None:
    report = EvalReport(
        field_scores={"vendor": FieldScore(path="vendor", node_type=NodeType.OBJECT)}
    )
    assert "vendor" not in ConsoleRenderer().render(report)


def test_a_field_without_a_threshold_gets_no_mark() -> None:
    # Nothing to compare the score against, so neither ✓ nor ✗ would be honest.
    report = EvalReport(
        field_scores={
            "a": FieldScore(
                path="a", node_type=NodeType.SCALAR, score=0.5, threshold=None
            )
        }
    )
    out = ConsoleRenderer().render(report)
    assert "a" in out
    assert "✓" not in out
    assert "✗" not in out


def test_the_headline_metric_is_not_repeated_in_the_grid() -> None:
    # It is already on the OVERALL line; the grid is for everything *else*, and
    # with nothing else there is no grid at all.
    report = EvalReport(
        score=1.0,
        score_label="object_f1",
        metrics={
            "object_f1": MetricCollection(
                name="object_f1", by_path={"$": MetricResult(1.0)}
            )
        },
    )
    out = ConsoleRenderer().render(report)
    assert out.count("object_f1") == 1


# ── batch ────────────────────────────────────────────────────────────────────


def test_an_empty_batch_renders() -> None:
    out = ConsoleRenderer().render(BatchEvalReport())
    assert "BATCH   0 samples" in out
    assert "Field breakdown" not in out  # nothing to rank


def test_a_batch_shows_its_mean_and_worst_fields_first() -> None:
    report = evaluate_batch(
        [
            Sample(actual={"a": 1, "b": 1}, expected={"a": 1, "b": 2}),
            Sample(actual={"a": 1, "b": 1}, expected={"a": 1, "b": 3}),
        ],
        EvalConfig(key_metric=ObjectF1()),
    )
    out = ConsoleRenderer().render(report)
    assert "mean object_f1" in out
    assert "Field breakdown (worst first)" in out
    assert out.index("b") < out.index("a", out.index("Field breakdown"))


# ── consistency ──────────────────────────────────────────────────────────────


def test_a_consistency_report_renders_its_stable_and_unstable_fields() -> None:
    report = ConsistencyReport(
        per_run=[EvalReport(), EvalReport()],
        mean_score=0.75,
        score_variance=0.0625,
        field_variance={"a": 0.0, "b": 0.5},
        stable_fields=["a"],
        unstable_fields=["b"],
    )
    out = ConsoleRenderer().render(report)
    assert "CONSISTENCY   2 runs" in out
    assert "0.75" in out  # mean score
    assert "0.0625" in out  # variance, four decimals
    assert "stable" in out
    assert "0.5000" in out  # the per-field variance table


def test_a_consistency_report_with_nothing_measured_still_renders() -> None:
    out = ConsoleRenderer().render(ConsistencyReport())
    assert "CONSISTENCY   0 runs" in out
    assert "—" in out  # no stable and no unstable fields
    assert "variance" not in out  # no per-field table


def test_an_unknown_report_type_is_refused() -> None:
    with pytest.raises(NotImplementedError):
        ConsoleRenderer().render("not a report")  # type: ignore[arg-type]
