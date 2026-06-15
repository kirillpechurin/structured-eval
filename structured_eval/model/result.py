from __future__ import annotations

from enum import StrEnum
from statistics import mean
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from structured_eval.model.nodes.array_node import ArrayMatchResult
from structured_eval.reporting.console import render


class NodeType(StrEnum):
    """The kind of tree node a ``FieldScore`` describes."""

    SCALAR = "scalar"
    OBJECT = "object"
    ARRAY = "array"


def _percentile(values: list[float], q: float) -> float:
    """Linear-interpolation percentile (q in [0, 1]) over a non-empty list."""
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = q * (len(ordered) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * frac


# ── Rules ─────────────────────────────────────────────────────────────────────


class RuleResult(BaseModel):
    """Result of evaluating a single business rule."""

    name: str
    passed: bool
    message: str = ""


# ── Per-path score ────────────────────────────────────────────────────────────


class FieldScore(BaseModel):
    """Evaluation result for one node of the tree (flat, dot-notation path).

    ``metrics`` holds only the metrics that were requested and applied to this
    node (e.g. ``{"exact_match": 0.0, "token_f1": 0.62}``). ``score`` is the
    value of the key metric at this path, ``threshold`` the bar applied to it.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    path: str
    node_type: NodeType
    actual: Any = None
    expected: Any = None
    metrics: dict[str, float] = Field(default_factory=dict)
    score: float | None = None
    threshold: float | None = None


# ── Regression diff ─────────────────────────────────────────────────────────


class RegressionDiff(BaseModel):
    """Metric deltas between two EvalReports (self minus other).

    ``deltas`` are per-metric changes in the aggregate; ``field_deltas`` maps
    each field path to its own per-metric changes. Positive means improvement.
    """

    deltas: dict[str, float] = Field(default_factory=dict)
    field_deltas: dict[str, dict[str, float]] = Field(default_factory=dict)


# ── Eval report ───────────────────────────────────────────────────────────────


class EvalReport(BaseModel):
    """Full evaluation result for a single document.

    ``metrics`` contains only the requested metrics. ``field_scores`` is a flat
    map of every tree node keyed by its path. On a parse error, ``parse_error``
    is True and the metrics are left empty.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    score: float | None = None
    score_label: str | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    field_scores: dict[str, FieldScore] = Field(default_factory=dict)
    array_matches: dict[str, ArrayMatchResult] = Field(default_factory=dict)
    rule_results: list[RuleResult] = Field(default_factory=list)
    parse_error: bool = False
    parse_error_message: str | None = None
    schema_errors: list[str] = Field(default_factory=list)
    hallucinated_fields: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

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

    # ── Reporting / serialization ─────────────────────────────────────────

    def print_summary(self) -> None:
        """Print a field-level summary table to stdout."""
        print(render(self))

    def to_dict(self) -> dict:
        """Return a JSON-friendly dict of the full report."""
        return self.model_dump(mode="json")

    def to_json(self, path: str) -> None:
        """Serialize the report to a JSON file."""
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.model_dump_json(indent=2))

    @classmethod
    def from_dict(cls, data: dict) -> EvalReport:
        """Reconstruct a report from a dict produced by ``to_dict``."""
        return cls.model_validate(data)

    @classmethod
    def from_json(cls, path: str) -> EvalReport:
        """Load a report from a JSON file."""
        with open(path, encoding="utf-8") as fh:
            return cls.model_validate_json(fh.read())

    def diff_from(
        self, other: EvalReport, metrics: list[str] | None = None
    ) -> RegressionDiff:
        """Compute metric deltas relative to ``other`` (self minus other).

        ``deltas`` covers document-level metrics present in both reports (or the
        subset named in ``metrics``); ``field_deltas`` covers per-field metrics
        for paths present in both. Positive means improvement.
        """
        names = metrics if metrics is not None else sorted(self.metrics)
        deltas = {
            name: self.metrics[name] - other.metrics[name]
            for name in names
            if name in self.metrics and name in other.metrics
        }

        field_deltas: dict[str, dict[str, float]] = {}
        for path, fs in self.field_scores.items():
            other_fs = other.field_scores.get(path)
            if other_fs is None:
                continue
            per: dict[str, float] = {
                m: fs.metrics[m] - other_fs.metrics[m]
                for m in fs.metrics
                if m in other_fs.metrics
            }
            if fs.score is not None and other_fs.score is not None:
                per["score"] = fs.score - other_fs.score
            if per:
                field_deltas[path] = per

        return RegressionDiff(deltas=deltas, field_deltas=field_deltas)

    # ── Assertions (pytest-style: raise AssertionError, else None) ────────

    def assert_no_parse_errors(self) -> None:
        """Fail if the document could not be parsed."""
        if self.parse_error:
            raise AssertionError(
                f"parse error: {self.parse_error_message or 'could not parse document'}"
            )

    def assert_score(self, min_score: float) -> None:
        """Fail if the key-metric score is below ``min_score``."""
        self.assert_no_parse_errors()
        if self.score is None:
            raise AssertionError(
                "no score available (no key metric configured); "
                "use assert_metric() instead"
            )
        if self.score < min_score:
            label = self.score_label or "score"
            raise AssertionError(
                f"{label} {self.score:.4g} < required {min_score:.4g}"
            )

    def assert_field(self, path: str, min_score: float) -> None:
        """Fail if the field at ``path`` scores below ``min_score``."""
        fs = self.field_scores.get(path)
        if fs is None:
            raise AssertionError(f"no field at path {path!r}")
        if fs.score is None:
            raise AssertionError(f"field {path!r} has no score (no key metric applied)")
        if fs.score < min_score:
            raise AssertionError(
                f"field {path!r} scored {fs.score:.4g} < required {min_score:.4g} "
                f"(actual={fs.actual!r}, expected={fs.expected!r})"
            )

    def assert_metric(self, metric_name: str, min_value: float) -> None:
        """Fail if metric ``metric_name`` is missing or below ``min_value``."""
        if metric_name not in self.metrics:
            available = ", ".join(sorted(self.metrics)) or "none"
            raise AssertionError(
                f"metric {metric_name!r} not computed (available: {available})"
            )
        value = self.metrics[metric_name]
        if value < min_value:
            raise AssertionError(
                f"metric {metric_name!r} {value:.4g} < required {min_value:.4g}"
            )

    def assert_schema_valid(self) -> None:
        """Fail if schema validation produced errors."""
        if self.metrics.get("schema_validity") == 0.0 or self.schema_errors:
            errors = "; ".join(self.schema_errors) or "schema validation failed"
            raise AssertionError(f"schema invalid: {errors}")


# ── Batch / consistency reports ───────────────────────────────────────────────


class BatchEvalReport(BaseModel):
    """Aggregate result over a list of documents (``evaluate(list[Sample])``).

    ``metrics`` is the mean of each document-level metric across successfully
    parsed samples; ``score`` is the mean key-metric score. ``perfect_response_rate``
    is the fraction of samples that parsed and had no failing field;
    ``parse_error_rate`` the fraction that failed to parse.
    """

    per_sample: list[EvalReport] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)
    score: float | None = None
    score_label: str | None = None
    perfect_response_rate: float = 0.0
    parse_error_rate: float = 0.0

    def field_breakdown(
        self, threshold: float | None = None
    ) -> dict[str, dict[str, float]]:
        """Per-path statistics across the batch: mean/min/max/p95/fail_rate.

        Only nodes with a score (a key metric applied) are counted. ``fail_rate``
        is the fraction of samples where the field scored below its bar (the
        ``threshold`` argument, else the field's own threshold, else 1.0).
        """
        scores: dict[str, list[float]] = {}
        fails: dict[str, int] = {}
        for r in self.per_sample:
            if r.parse_error:
                continue
            for path, fs in r.field_scores.items():
                if fs.score is None:
                    continue
                scores.setdefault(path, []).append(fs.score)
                bar = threshold if threshold is not None else fs.threshold
                if bar is None:
                    bar = 1.0
                if fs.score < bar:
                    fails[path] = fails.get(path, 0) + 1

        return {
            path: {
                "mean": mean(vals),
                "min": min(vals),
                "max": max(vals),
                "p95": _percentile(vals, 0.95),
                "fail_rate": fails.get(path, 0) / len(vals),
            }
            for path, vals in scores.items()
        }

    def print_summary(self) -> None:
        """Print a batch summary (aggregate metrics + field breakdown)."""
        print(render(self))


class ConsistencyReport(BaseModel):
    """Stability of repeated runs of one prompt (``evaluate_consistency``).

    ``field_variance`` is the variance of each field's score across runs;
    fields with variance at or below ``variance_threshold`` are ``stable_fields``,
    the rest ``unstable_fields``. ``score_variance`` is the variance of the
    document-level key-metric score.
    """

    per_run: list[EvalReport] = Field(default_factory=list)
    field_variance: dict[str, float] = Field(default_factory=dict)
    stable_fields: list[str] = Field(default_factory=list)
    unstable_fields: list[str] = Field(default_factory=list)
    mean_score: float | None = None
    score_variance: float | None = None

    def print_summary(self) -> None:
        """Print a consistency summary (stable/unstable fields + variance)."""
        print(render(self))
