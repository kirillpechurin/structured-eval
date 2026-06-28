"""ConsoleRenderer — structural checks on the rendered report text.

Not a byte snapshot: we assert the rendered text *contains* the pieces a reader
relies on (overall verdict, score, every field row, pass/fail marks), so layout
drops are caught without pinning exact whitespace.
"""

import pytest

from structured_eval import EvalConfig, EvalReport, ObjectF1, evaluate
from structured_eval.reporting.console import ConsoleRenderer

pytestmark = pytest.mark.unit


def _render(actual, expected, **kw):
    report = evaluate(actual, expected, config=EvalConfig(key_metric=ObjectF1()), **kw)
    return ConsoleRenderer().render(report)


def test_header_and_score_present():
    out = _render({"id": "INV-1", "total": 99.0}, {"id": "INV-1", "total": 100.0})
    assert "OVERALL" in out
    assert "object_f1" in out  # the key-metric label
    assert "0.50" in out  # 1/2 fields correct


def test_every_field_has_a_row():
    out = _render(
        {"id": "INV-1", "total": 99.0, "vendor": "Acme"},
        {"id": "INV-1", "total": 100.0, "vendor": "Acme"},
    )
    for field in ("id", "total", "vendor"):
        assert field in out


def test_pass_and_fail_marks_rendered():
    out = _render({"a": 1, "b": 2}, {"a": 1, "b": 99})
    assert "✓" in out  # a correct
    assert "✗" in out  # b wrong


def test_perfect_report_has_no_fail_mark():
    out = _render({"a": 1}, {"a": 1})
    assert "✗" not in out
    assert "✓" in out


def test_parse_error_render():
    out = ConsoleRenderer().render(
        EvalReport(parse_error=True, parse_error_message="unexpected token")
    )
    assert "PARSE ERROR" in out
    assert "unexpected token" in out


def test_render_is_nonempty_str():
    assert _render({"a": 1}, {"a": 1}).strip()


def test_batch_report_renders():
    from structured_eval import Sample, evaluate_batch

    report = evaluate_batch(
        [Sample(actual={"a": 1}, expected={"a": 1})], EvalConfig(key_metric=ObjectF1())
    )
    assert ConsoleRenderer().render(report)


def test_print_summary_writes_to_stdout(capsys):
    evaluate({"a": 1}, {"a": 1}, config=EvalConfig(key_metric=ObjectF1())).print_summary()
    assert capsys.readouterr().out
