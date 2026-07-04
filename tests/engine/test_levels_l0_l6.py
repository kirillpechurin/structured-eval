"""The L0→L6 thesis test: one document, every level reported independently.

This is the project's reason to exist (CLAUDE.md / RFC): structural validity
(L0–L3) does not imply value correctness (L4), faithfulness to a source (L5),
or business-rule consistency (L6). A document can be perfectly parseable,
schema-valid and correctly typed while still being *wrong*.

We build exactly such a document and assert each level surfaces its own verdict:

  L0 parse        — valid JSON, no parse error.
  L1 schema       — matches the pydantic schema (SchemaValidity == 1.0).
  L2/L3 types/req — all required fields present with correct types.
  L4 values       — a wrong value (`total`) drags the value score below 1.0.
  L5 faithfulness — a field absent from `source` is flagged as a hallucination.
  L6 rules        — `total == subtotal + tax` is violated and reported.

If a future refactor lets a higher level "leak" into a lower one (e.g. a bad
value silently failing the schema, or a rule failure not surfacing), this test
breaks — which is the whole point.
"""

import pytest
from pydantic import BaseModel

from structured_eval import evaluate
from structured_eval.metrics import (
    FieldFaithfulness,
    ObjectF1,
    OverallLeafScore,
    RulePassRate,
    SchemaValidity,
)
from structured_eval.metrics.rule_pass_rate.dsl import Rule
from structured_eval.models import EvalConfig, EvalReport

pytestmark = pytest.mark.engine


class Invoice(BaseModel):
    id: str
    vendor: str
    subtotal: float
    tax: float
    total: float
    status: str


# Structurally flawless, semantically broken:
#   * total is 999.0 but should be 110.0  (L4 wrong value)
#   * total != subtotal + tax            (L6 rule violation)
#   * vendor "Globex" does not appear in the source text (L5 hallucination)
ACTUAL = {
    "id": "INV-001",
    "vendor": "Globex",
    "subtotal": 100.0,
    "tax": 10.0,
    "total": 999.0,
    "status": "paid",
}
EXPECTED = {
    "id": "INV-001",
    "vendor": "Acme Corp",
    "subtotal": 100.0,
    "tax": 10.0,
    "total": 110.0,
    "status": "paid",
}
SOURCE = (
    "Invoice INV-001 from Acme Corp, subtotal 100.0, tax 10.0, total 110.0, status paid"
)


@pytest.fixture
def report() -> EvalReport:
    cfg = EvalConfig(
        metrics=[
            ObjectF1(),
            OverallLeafScore(),
            SchemaValidity(Invoice),
            FieldFaithfulness(),
            RulePassRate(rules=[Rule("$.total").eq("$.subtotal + $.tax")]),
        ]
    )
    return evaluate(ACTUAL, EXPECTED, config=cfg, source=SOURCE)


def test_l0_parses_clean(report: EvalReport) -> None:
    assert report.parse_error is False


def test_l1_schema_valid(report: EvalReport) -> None:
    # Document conforms to the schema even though values are wrong.
    assert report.metrics["schema_validity"].representative() == 1.0
    # A valid document carries the schema_errors channel but every bucket is empty.
    root = report.metrics["schema_validity"].root()
    assert root is not None
    errors = root.extra["schema_errors"]
    assert all(not bucket for bucket in errors.values())


def test_l4_values_imperfect(report: EvalReport) -> None:
    # Structural levels pass, yet the value score is dragged down by `total`.
    assert report.metrics["overall_leaf_score"].representative() < 1.0
    assert report.field_scores["total"].score == 0.0
    # ... and the correct fields are unaffected.
    assert report.field_scores["subtotal"].score == 1.0


def test_l5_faithfulness_flags_hallucination(report: EvalReport) -> None:
    ff = report.metrics["field_faithfulness"].by_path
    hallucinated = [p for p, v in ff.items() if float(v) == 0.0]
    assert "vendor" in hallucinated  # "Globex" is absent from the source
    assert ff["id"] == pytest.approx(1.0)  # "INV-001" is grounded


def test_l6_rule_violation_reported(report: EvalReport) -> None:
    rpr = report.metrics["rule_pass_rate"]
    assert rpr.representative() == 0.0  # 999 != 100 + 10
    results = rpr.extra_values("rule_results")
    assert len(results) == 1


def test_levels_are_independent(report: EvalReport) -> None:
    # The thesis in one assert block: valid structure (L1) coexists with broken
    # values (L4), an unfaithful field (L5), and a violated rule (L6).
    assert report.metrics["schema_validity"].representative() == 1.0
    assert report.metrics["overall_leaf_score"].representative() < 1.0
    assert report.metrics["rule_pass_rate"].representative() == 0.0
