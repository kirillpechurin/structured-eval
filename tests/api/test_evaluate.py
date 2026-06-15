"""Tests for the public ``evaluate`` entrypoint: the various call shapes."""

from __future__ import annotations

import pytest

from structured_eval import (
    BatchEvalReport,
    EvalConfig,
    EvalReport,
    ObjectF1,
    Sample,
    evaluate,
)

pytestmark = pytest.mark.engine


def test_dict_shorthand():
    r = evaluate({"a": 1}, {"a": 1}, config=EvalConfig(metrics=[ObjectF1()]))
    assert isinstance(r, EvalReport)
    assert r.metrics["object_f1"] == 1.0


def test_sample_input():
    r = evaluate(Sample(actual={"a": 1}, expected={"a": 1}),
                 config=EvalConfig(metrics=[ObjectF1()]))
    assert isinstance(r, EvalReport)


def test_json_string_input():
    r = evaluate('{"a": 1}', '{"a": 1}', config=EvalConfig(metrics=[ObjectF1()]))
    assert r.metrics["object_f1"] == 1.0


def test_list_of_samples_is_batch():
    r = evaluate([Sample(actual={"a": 1}, expected={"a": 1})])
    assert isinstance(r, BatchEvalReport)


def test_bare_list_is_single_document():
    # a bare list is one array-root document, NOT a batch
    r = evaluate([1, 2], [1, 2])
    assert isinstance(r, EvalReport)


def test_source_kwarg_enables_faithfulness():
    from structured_eval import Faithfulness

    r = evaluate(
        {"vendor": "Globex"}, None,
        config=EvalConfig(metrics=[Faithfulness()]),
        source="Invoice from Acme Corp",
    )
    assert r.metrics["faithfulness"] == 0.0


def test_no_config_runs_with_defaults():
    r = evaluate({"a": 1}, {"a": 1})
    assert isinstance(r, EvalReport)
    assert not r.parse_error
