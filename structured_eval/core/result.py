from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from structured_eval.nodes.array_node import ArrayMatchResult

NodeType = Literal["scalar", "object", "array"]


# ── Rules ─────────────────────────────────────────────────────────────────────


@dataclass
class RuleResult:
    """Result of evaluating a single business rule."""

    name: str
    passed: bool
    message: str = ""


# ── Per-path score ────────────────────────────────────────────────────────────


@dataclass
class FieldScore:
    """Evaluation result for one node of the tree (flat, dot-notation path).

    ``metrics`` holds only the metrics that were requested and applied to this
    node (e.g. ``{"exact_match": 0.0, "token_f1": 0.62}``). ``score`` is the
    value of the key metric at this path, ``threshold`` the bar applied to it.
    """

    path: str
    node_type: NodeType
    actual: Any = None
    expected: Any = None
    metrics: dict[str, float] = field(default_factory=dict)
    score: float | None = None
    threshold: float | None = None


# ── Regression diff ─────────────────────────────────────────────────────────


@dataclass
class RegressionDiff:
    """Metric deltas between two EvalReports (self minus other).

    ``deltas`` are per-metric changes in the aggregate; ``field_deltas`` maps
    each field path to its own per-metric changes. Positive means improvement.
    """

    deltas: dict[str, float] = field(default_factory=dict)
    field_deltas: dict[str, dict[str, float]] = field(default_factory=dict)


# ── Eval report ───────────────────────────────────────────────────────────────


@dataclass
class EvalReport:
    """Full evaluation result for a single document.

    ``metrics`` contains only the requested metrics. ``field_scores`` is a flat
    map of every tree node keyed by its path. On a parse error, ``parse_error``
    is True and the metrics are left empty.
    """

    score: float | None = None
    score_label: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    field_scores: dict[str, FieldScore] = field(default_factory=dict)
    array_matches: dict[str, ArrayMatchResult] = field(default_factory=dict)
    rule_results: list[RuleResult] = field(default_factory=list)
    parse_error: bool = False
    parse_error_message: str | None = None
    schema_errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    # ── Queries ───────────────────────────────────────────────────────────

    def failed_fields(self, threshold: float | None = None) -> list[FieldScore]:
        """Return fields whose score falls below the applicable threshold.

        Precedence per field: the ``threshold`` argument, else the field's own
        ``threshold``, else a perfect-match bar of 1.0. Fields without a score
        (no key metric applied) are skipped.
        """
        failed: list[FieldScore] = []
        for fs in self.field_scores.values():
            if fs.score is None:
                continue
            bar = threshold if threshold is not None else fs.threshold
            if bar is None:
                bar = 1.0
            if fs.score < bar:
                failed.append(fs)
        return failed

    # ── Reporting / serialization (later stages) ──────────────────────────

    def print_summary(self) -> None:
        """Print a Rich-formatted summary table to stdout (Stage 11)."""
        raise NotImplementedError

    def to_json(self, path: str) -> None:
        """Serialize the report to JSON (Stage 11)."""
        raise NotImplementedError

    @classmethod
    def from_json(cls, path: str) -> EvalReport:
        """Load a report from JSON (Stage 11)."""
        raise NotImplementedError

    def diff_from(
        self, other: EvalReport, metrics: list[str] | None = None
    ) -> RegressionDiff:
        """Compute metric deltas relative to ``other`` (Stage 11)."""
        raise NotImplementedError

    # ── Assertions (later stages) ─────────────────────────────────────────

    def assert_score(self, min_score: float) -> None:
        raise NotImplementedError

    def assert_field(self, path: str, min_score: float) -> None:
        raise NotImplementedError

    def assert_metric(self, metric_name: str, min_value: float) -> None:
        raise NotImplementedError

    def assert_no_parse_errors(self) -> None:
        raise NotImplementedError

    def assert_schema_valid(self) -> None:
        raise NotImplementedError
