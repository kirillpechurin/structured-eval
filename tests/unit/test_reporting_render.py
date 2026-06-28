"""Structural checks on the console renderer.

Not a pixel/byte snapshot — we assert the report's rendered text *contains* the
pieces a reader relies on (the overall verdict, the score, every field row with
its mark), so accidental drops in the layout are caught without pinning exact
whitespace.
"""

import pytest

from structured_eval import EvalConfig, EvalReport, ObjectF1, evaluate
from structured_eval.reporting.console import ConsoleRenderer

pytestmark = pytest.mark.unit


def _render(actual, expected, **kw):
    report = evaluate(actual, expected, config=EvalConfig(key_metric=ObjectF1()), **kw)
    return report, ConsoleRenderer().render(report)


class TestRenderStructure:
    def test_header_and_score_present(self):
        _, out = _render(
            {"id": "INV-1", "total": 99.0, "vendor": "Acme"},
            {"id": "INV-1", "total": 100.0, "vendor": "Acme"},
        )
        assert "OVERALL" in out
        assert "object_f1" in out  # the key-metric label
        assert "0.67" in out  # 2/3 fields correct

    def test_every_field_has_a_row(self):
        _, out = _render(
            {"id": "INV-1", "total": 99.0, "vendor": "Acme"},
            {"id": "INV-1", "total": 100.0, "vendor": "Acme"},
        )
        for field in ("id", "total", "vendor"):
            assert field in out

    def test_pass_and_fail_marks_rendered(self):
        _, out = _render({"a": 1, "b": 2}, {"a": 1, "b": 99})
        # a correct (✓), b wrong (✗) — both marks must appear.
        assert "✓" in out
        assert "✗" in out

    def test_perfect_report_has_no_fail_mark(self):
        _, out = _render({"a": 1}, {"a": 1})
        assert "✗" not in out
        assert "✓" in out

    def test_parse_error_render(self):
        out = ConsoleRenderer().render(
            EvalReport(parse_error=True, parse_error_message="unexpected token")
        )
        assert "PARSE ERROR" in out
        assert "unexpected token" in out

    def test_render_is_nonempty_str(self):
        _, out = _render({"a": 1}, {"a": 1})
        assert isinstance(out, str) and out.strip()
