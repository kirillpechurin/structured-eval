"""integrations/_adapter — maps a report to (score, success, reason).

Pure mapping shared by the host adapters; tested without any host library.
"""

import pytest

from structured_eval import EvalConfig, EvalReport, ObjectF1, evaluate
from structured_eval.integrations._adapter import reason_text, verdict

pytestmark = pytest.mark.unit


def _report(actual, expected):
    return evaluate(actual, expected, config=EvalConfig(key_metric=ObjectF1()))


def test_verdict_success():
    score, success, reason = verdict(_report({"a": 1}, {"a": 1}), threshold=0.8)
    assert score == 1.0 and success is True
    assert "passed" in reason


def test_verdict_failure():
    _score, success, reason = verdict(_report({"a": 9}, {"a": 1}), threshold=0.8)
    assert success is False
    assert "failed" in reason


def test_reason_text_for_parse_error():
    text = reason_text(EvalReport(parse_error=True, parse_error_message="boom"))
    assert "parse error" in text


def test_none_score_is_failure():
    # no key-metric score (e.g. no ground truth) → threshold can't apply → not a pass
    _score, success, _reason = verdict(EvalReport(score=None), threshold=0.5)
    assert success is False
