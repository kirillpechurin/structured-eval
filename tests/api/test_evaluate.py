"""Tests for the public ``evaluate`` entrypoint: the various call shapes."""

import pytest

from structured_eval import (
    BatchEvalReport,
    EvalConfig,
    EvalReport,
    ObjectF1,
    Sample,
    evaluate,
    evaluate_batch,
)

pytestmark = pytest.mark.engine


def test_dict_shorthand() -> None:
    r = evaluate({"a": 1}, {"a": 1}, config=EvalConfig(metrics=[ObjectF1()]))
    assert isinstance(r, EvalReport)
    assert r.metrics["object_f1"].representative() == 1.0


def test_sample_input() -> None:
    r = evaluate(
        Sample(actual={"a": 1}, expected={"a": 1}),
        config=EvalConfig(metrics=[ObjectF1()]),
    )
    assert isinstance(r, EvalReport)


def test_json_string_input() -> None:
    r = evaluate('{"a": 1}', '{"a": 1}', config=EvalConfig(metrics=[ObjectF1()]))
    assert r.metrics["object_f1"].representative() == 1.0


def test_evaluate_batch() -> None:
    r = evaluate_batch([Sample(actual={"a": 1}, expected={"a": 1})])
    assert isinstance(r, BatchEvalReport)


def test_evaluate_rejects_list_of_samples() -> None:
    # a batch must go through evaluate_batch, not evaluate
    with pytest.raises(TypeError, match="evaluate_batch"):
        evaluate([Sample(actual={"a": 1}, expected={"a": 1})])


def test_bare_list_is_single_document() -> None:
    # a bare list is one array-root document, NOT a batch
    r = evaluate([1, 2], [1, 2])
    assert isinstance(r, EvalReport)


def test_source_kwarg_enables_faithfulness() -> None:
    from structured_eval import FieldFaithfulness

    r = evaluate(
        {"vendor": "Globex"},
        None,
        config=EvalConfig(metrics=[FieldFaithfulness()]),
        source="Invoice from Acme Corp",
    )
    assert r.metrics["field_faithfulness"].representative() == 0.0


def test_no_config_runs_with_defaults() -> None:
    r = evaluate({"a": 1}, {"a": 1})
    assert isinstance(r, EvalReport)
    assert not r.parse_error
