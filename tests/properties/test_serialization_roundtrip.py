"""Case E — report serialization round-trips losslessly.

A report is the framework's output contract (CI artifacts, dashboards, the diff
machinery). ``to_json`` → ``from_json`` must preserve the headline score, the
per-field scores, and the structured ``.extra`` side-channels (schema errors,
rule results). We generate varied documents and assert the round-trip is a
fixed point at the level callers actually read.
"""

import random

import pytest
from pydantic import BaseModel

from structured_eval import (
    EvalConfig,
    EvalReport,
    ObjectF1,
    OverallLeafScore,
    RulePassRate,
    SchemaValidity,
    evaluate,
)
from structured_eval.metrics.rule_pass_rate.dsl import Rule

from .conftest import SEEDS, random_document

pytestmark = pytest.mark.property


def _assert_report_equivalent(a: EvalReport, b: EvalReport) -> None:
    assert a.score == pytest.approx(b.score) if a.score is not None else b.score is None
    assert a.score_label == b.score_label
    assert a.parse_error == b.parse_error
    assert set(a.field_scores) == set(b.field_scores)
    for path, fs in a.field_scores.items():
        assert fs.score == pytest.approx(b.field_scores[path].score)
    assert set(a.metrics) == set(b.metrics)


@pytest.mark.parametrize("seed", SEEDS)
def test_roundtrip_preserves_scores_and_fields(seed):
    rng = random.Random(seed)
    expected = random_document(rng, depth=3)
    actual = random_document(rng, depth=3)
    report = evaluate(actual, expected, config=EvalConfig(metrics=[OverallLeafScore()]))

    restored = EvalReport.from_dict(report.to_dict())
    _assert_report_equivalent(report, restored)


class _Invoice(BaseModel):
    id: str
    total: float


def test_roundtrip_preserves_schema_errors_extra():
    """The ``schema_errors`` side-channel survives a JSON round-trip."""
    cfg = EvalConfig(metrics=[SchemaValidity(_Invoice)])
    report = evaluate({"id": "1"}, None, config=cfg)  # missing total → invalid

    restored = EvalReport.from_dict(report.to_dict())
    assert restored.metrics["schema_validity"].extra_values("schema_errors")
    assert (
        restored.metrics["schema_validity"].representative()
        == report.metrics["schema_validity"].representative()
    )


def test_roundtrip_preserves_rule_results_extra():
    """The ``rule_results`` side-channel survives a JSON round-trip."""
    cfg = EvalConfig(
        metrics=[
            ObjectF1(),
            RulePassRate(rules=[Rule("$.total").eq("$.subtotal + $.tax")]),
        ]
    )
    doc = {"total": 99.0, "subtotal": 100.0, "tax": 10.0}  # rule fails
    report = evaluate(doc, None, config=cfg)

    restored = EvalReport.from_dict(report.to_dict())
    assert len(restored.metrics["rule_pass_rate"].extra_values("rule_results")) == 1
    assert (
        restored.metrics["rule_pass_rate"].representative()
        == report.metrics["rule_pass_rate"].representative()
    )
