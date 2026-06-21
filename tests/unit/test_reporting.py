"""Smoke tests for console rendering and the integration adapter.

The renderer is pure stdlib; the adapter maps a report to (score, success,
reason) and is tested without any host library installed.
"""

from __future__ import annotations

import pytest

from structured_eval import EvalConfig, EvalReport, ObjectF1, evaluate
from structured_eval.integrations._adapter import reason_text, verdict
from structured_eval.reporting.console import ConsoleRenderer

pytestmark = pytest.mark.unit


def _report(actual, expected, **kw):
    return evaluate(actual, expected, config=EvalConfig(key_metric=ObjectF1()), **kw)


class TestConsole:
    def test_eval_report_renders(self):
        out = ConsoleRenderer().render(_report({"a": 1}, {"a": 1}))
        assert isinstance(out, str) and out

    def test_parse_error_renders(self):
        out = ConsoleRenderer().render(EvalReport(parse_error=True, parse_error_message="bad"))
        assert "PARSE ERROR" in out

    def test_batch_renders(self):
        from structured_eval import Sample, evaluate_batch

        report = evaluate_batch(
            [Sample(actual={"a": 1}, expected={"a": 1})], EvalConfig(key_metric=ObjectF1())
        )
        assert ConsoleRenderer().render(report)

    def test_print_summary_no_crash(self, capsys):
        _report({"a": 1}, {"a": 1}).print_summary()
        assert capsys.readouterr().out


class TestAdapter:
    def test_verdict_success(self):
        score, success, reason = verdict(_report({"a": 1}, {"a": 1}), threshold=0.8)
        assert score == 1.0 and success is True
        assert "passed" in reason

    def test_verdict_failure(self):
        score, success, reason = verdict(_report({"a": 9}, {"a": 1}), threshold=0.8)
        assert success is False
        assert "failed" in reason

    def test_reason_parse_error(self):
        text = reason_text(EvalReport(parse_error=True, parse_error_message="boom"))
        assert "parse error" in text

    def test_verdict_none_score_is_failure(self):
        # No key-metric score (e.g. no ground truth) → the threshold can't be
        # applied → not a pass.
        _score, success, _reason = verdict(EvalReport(score=None), threshold=0.5)
        assert success is False
