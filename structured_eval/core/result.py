from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from structured_eval.core.config import ArrayStrategy, EvalConfig

# ── Enums ─────────────────────────────────────────────────────────────────────


class FieldStatus(StrEnum):
    MATCH = "match"  # score == 1.0
    PARTIAL = "partial"  # 0 < score < 1.0
    MISS = "miss"  # field expected but absent in actual
    EXTRA = "extra"  # field present in actual but not in expected
    NULL_MATCH = "null_match"  # both sides null/missing (lenient mode)


# ── Result types ──────────────────────────────────────────────────────────────


class FieldScore(BaseModel):
    """Evaluation result for a single field.

    Produced by the field accuracy evaluator for every comparable field.
    Nested objects produce a FieldScore with children populated recursively.
    """

    key: str = Field(description="Field name or JSONPath expression.")
    expected: Any = Field(description="Expected value from ground truth.")
    actual: Any = Field(description="Actual value from the LLM output.")
    score: float = Field(ge=0.0, le=1.0, description="Match score: 1.0 = perfect, 0.0 = no match.")
    matcher: str = Field(description="Name of the matcher that produced this score.")
    status: FieldStatus = Field(description="Categorical match status.")
    children: dict[str, FieldScore] | None = Field(
        default=None,
        description="Nested FieldScores for object-type fields. Score is the bottom-up F1 of children.",
    )


FieldScore.model_rebuild()


class ArrayMatchResult(BaseModel):
    """Alignment result for a pair of object arrays."""

    strategy: ArrayStrategy = Field(description="Strategy used: by_index, by_key, best_match.")
    matched: list[tuple[int, int]] = Field(
        description="Pairs of (expected_index, actual_index) that were aligned (TP)."
    )
    missed: list[int] = Field(
        description="Expected indices not found in actual (FN)."
    )
    spurious: list[int] = Field(
        description="Actual indices not present in expected (FP)."
    )
    precision: float = Field(ge=0.0, le=1.0, description="Matched / total actual.")
    recall: float = Field(ge=0.0, le=1.0, description="Matched / total expected.")
    f1: float = Field(ge=0.0, le=1.0, description="Harmonic mean of precision and recall.")


class RuleResult(BaseModel):
    """Result of evaluating a single business rule."""

    name: str = Field(description="Rule name as declared in the DSL or Rule.custom(name=...).")
    passed: bool = Field(description="Whether the rule evaluated to True.")
    message: str = Field(default="", description="Human-readable explanation for failures.")


class RegressionDiff(BaseModel):
    """Metric delta between two EvalReport runs.

    Positive deltas indicate improvement; negative indicate regression.
    """

    f1_delta: float = Field(description="Change in F1 score.")
    precision_delta: float = Field(description="Change in precision.")
    recall_delta: float = Field(description="Change in recall.")
    field_deltas: dict[str, float] = Field(
        description="Per-field score deltas. Only includes fields present in at least one run."
    )


class EvalReport(BaseModel):
    """Full evaluation result for one document or a batch.

    Top-level output of evaluate(). Contains aggregate metrics, per-field
    breakdown, rule results, and optional schema validation details.

    The score property is the recommended single-number summary for CI thresholding.
    It returns the best available signal: f1 > rule_pass_rate > coverage_score.
    When expected is not provided (schema-only mode), f1/precision/recall are None.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Core metrics — None when expected was not provided (schema-only mode)
    f1: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Harmonic mean of precision and recall. None in schema-only mode.",
    )
    precision: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Fraction of actual fields that match expected. None in schema-only mode.",
    )
    recall: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Fraction of expected fields found in actual. None in schema-only mode.",
    )

    # Per-sample flags
    perfect: bool | None = Field(
        default=None,
        description="True iff f1 == 1.0 (all fields correct). None in schema-only mode.",
    )

    # Schema
    schema_valid: bool | None = Field(
        default=None, description="Whether actual passes the provided schema."
    )
    coverage_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Fraction of schema fields that are non-null in actual.",
    )
    path_recall: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Fraction of expected JSONPaths present in actual.",
    )
    path_precision: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Fraction of actual JSONPaths that are present in expected.",
    )
    type_error_rate: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Fraction of fields where the value type differs from expected.",
    )

    # Rules
    rule_pass_rate: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Fraction of rules that passed.",
    )
    rule_results: list[RuleResult] = Field(default_factory=list)

    # Faithfulness
    faithfulness_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Fraction of leaf fields whose value is found in the source text (L1 substring).",
    )
    hallucinated_fields: list[str] = Field(
        default_factory=list,
        description="Field paths where the value was not found in the source text.",
    )

    # Breakdown
    field_scores: dict[str, FieldScore] = Field(default_factory=dict)
    array_matches: dict[str, ArrayMatchResult] | None = None
    root_array_match: ArrayMatchResult | None = Field(
        default=None,
        description="Set when the top-level input is a list. Not supported in v0.1.",
    )

    # Meta
    config: EvalConfig | None = None
    warnings: list[str] = Field(
        default_factory=list,
        description="Non-fatal issues encountered during evaluation.",
    )

    @property
    def score(self) -> float | None:
        """Best available single-number summary for CI thresholding.

        Returns the first non-None value from: f1 → rule_pass_rate → coverage_score.
        Returns None when no metric could be computed (e.g. no expected, no schema, no rules).
        """
        if self.f1 is not None:
            return self.f1
        if self.rule_pass_rate is not None:
            return self.rule_pass_rate
        if self.coverage_score is not None:
            return self.coverage_score
        return None

    def failed_fields(self) -> list[FieldScore]:
        """Return all top-level fields with score < 1.0."""
        return [fs for fs in self.field_scores.values() if fs.score < 1.0]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the report to a plain Python dict."""
        return self.model_dump()

    def print_summary(self) -> None:
        """Print a Rich-formatted table to stdout."""
        raise NotImplementedError

    def save_html(self, path: str) -> None:
        """Write a self-contained HTML report to path."""
        raise NotImplementedError

    def to_junit_xml(self, path: str) -> None:
        """Write a JUnit XML report to path (compatible with GitHub Actions / Jenkins)."""
        raise NotImplementedError

    def diff_from(self, other: EvalReport) -> RegressionDiff:
        """Compute metric deltas relative to other (self is the newer run)."""
        raise NotImplementedError
