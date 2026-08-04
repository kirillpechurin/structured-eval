"""The report models an evaluation returns.

- `EvalReport` — one document: `score` as the headline number, `field_scores`
  for per-path detail, `metrics` across the tree, plus any `warnings`.
- `BatchEvalReport` — the aggregate `evaluate_batch` returns.
- `ConsistencyReport` — the aggregate `evaluate_consistency` returns.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from statistics import mean
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from structured_eval.models.metrics import (  # noqa: TC001
    MetricCollection,
    MetricResult,
)
from structured_eval.models.nodes.array_node import ArrayMatchResult  # noqa: TC001
from structured_eval.models.nodes.base import NodeType  # noqa: TC001


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


# ── Warnings ────────────────────────────────────────────────────────────────


class WarningType(StrEnum):
    """The kind of structural warning the engine raised while building the tree.

    Attributes:
        EXTRA_KEY: Key present in actual but not expected, under
            `ExtraKeysPolicy.IGNORE`.
        MISSING_FIELD: Key present in expected but absent from actual.
    """

    EXTRA_KEY = "extra_key"
    MISSING_FIELD = "missing_field"


class EvalWarning(BaseModel):
    """A structural warning, typed by `WarningType` and located by `path`.

    Attributes:
        type: What kind of warning this is.
        path: The node it concerns.
        message: A human-readable note; the path is shown when it is empty.
    """

    type: WarningType
    path: str
    message: str = ""

    def __str__(self) -> str:
        """The warning as one tagged line, for a console report."""
        tag = f"[{self.type.name}]"
        return f"{tag} {self.message}" if self.message else f"{tag} {self.path}"


# ── Rules ─────────────────────────────────────────────────────────────────────


class RuleResult(BaseModel):
    """Result of evaluating a single business rule.

    Attributes:
        name: The rule's name.
        passed: Whether the document satisfied it.
        message: Why it failed, when the rule explains itself.
    """

    name: str
    passed: bool
    message: str = ""


# ── Per-path score ────────────────────────────────────────────────────────────


class FieldScore(BaseModel):
    """Evaluation result for one node of the tree (flat, dot-notation path).

    Attributes:
        path: The node's dot-and-bracket path.
        node_type: Whether the node is a scalar, an object or an array.
        actual: The value found in the actual document.
        expected: The value the reference document asked for.
        metrics: Only the metrics requested and applied here, e.g.
            `{"exact_match": 0.0, "token_f1": 0.62}`. Each value is a
            `MetricResult` — a `float` that also carries `.extra`.
        score: The key metric's value at this path.
        threshold: The bar applied to that score.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    path: str
    node_type: NodeType
    actual: Any = None
    expected: Any = None
    metrics: dict[str, MetricResult] = Field(default_factory=dict)
    # TODO: Should be required by default key metric
    score: float | None = None
    # TODO: Reconsider default arguments - all possible should be defined in model as defaults
    threshold: float | None = None


# ── Regression diff ─────────────────────────────────────────────────────────


class RegressionDiff(BaseModel):
    """Metric deltas between two `EvalReport`s (self minus other).

    Positive means improvement.

    Attributes:
        deltas: Per-metric changes in the document-level aggregate.
        field_deltas: Each field path mapped to its own per-metric changes.
    """

    deltas: dict[str, float] = Field(default_factory=dict)
    field_deltas: dict[str, dict[str, float]] = Field(default_factory=dict)


# ── Eval report ───────────────────────────────────────────────────────────────


class EvalReport(BaseModel):
    """Full evaluation result for a single document.

    Attributes:
        score: The root node's key-metric value — the headline number.
        score_label: Which metric that was.
        metrics: Each metric name mapped to a `MetricCollection` — its value at
            every node that produced it, plus that metric's structured detail
            (schema errors, hallucinated paths, per-rule outcomes, …) on each
            value's `.extra`.
        field_scores: Every tree node, flat, keyed by its path.
        array_matches: Each array node's alignment result, keyed by path.
        parse_error: True when the document could not be parsed; the metrics
            are then left empty.
        parse_error_message: What went wrong parsing it.
        warnings: Structural warnings raised while building the tree.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    score: float | None = None
    score_label: str | None = None
    metrics: dict[str, MetricCollection] = Field(default_factory=dict)
    field_scores: dict[str, FieldScore] = Field(default_factory=dict)
    array_matches: dict[str, ArrayMatchResult] = Field(default_factory=dict)
    parse_error: bool = False
    parse_error_message: str | None = None
    warnings: list[EvalWarning] = Field(default_factory=list)

    # ── Queries ───────────────────────────────────────────────────────────

    def failed_fields(self, threshold: float | None = None) -> dict[str, FieldScore]:
        """Return fields whose score falls below the applicable threshold.

        Args:
            threshold: One bar for every field; when omitted each field's own
                `threshold` applies, falling back to a perfect-match 1.0.

        Returns:
            The failing fields, keyed by path as in `field_scores`. Fields with
            no score (no key metric applied) are skipped.
        """
        failed: dict[str, FieldScore] = {}
        for path, fs in self.field_scores.items():
            if fs.score is None:
                continue
            bar = threshold if threshold is not None else fs.threshold
            if bar is None:
                bar = 1.0
            if fs.score < bar:
                failed[path] = fs
        return failed

    # ── Reporting / serialization ─────────────────────────────────────────

    def print_summary(self) -> None:
        """Print a field-level summary table to stdout."""
        from structured_eval.reporting import render

        print(render(self))  # noqa: T201

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly dict of the full report."""
        return self.model_dump(mode="json")

    def to_json(self, path: str) -> None:
        """Serialize the report to a JSON file."""
        with Path(path).open("w", encoding="utf-8") as fh:
            fh.write(self.model_dump_json(indent=2))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvalReport:
        """Reconstruct a report from a dict produced by ``to_dict``."""
        return cls.model_validate(data)

    @classmethod
    def from_json(cls, path: str) -> EvalReport:
        """Load a report from a JSON file."""
        with Path(path).open(encoding="utf-8") as fh:
            return cls.model_validate_json(fh.read())

    def diff_from(
        self, other: EvalReport, metrics: list[str] | None = None
    ) -> RegressionDiff:
        """Compute metric deltas relative to `other` (self minus other).

        Args:
            other: The report to compare against — typically the baseline.
            metrics: Which document-level metrics to diff; all shared ones by
                default.

        Returns:
            Deltas for the document-level metrics present in both reports, and
            per-field deltas for the paths present in both.
        """
        names = metrics if metrics is not None else sorted(self.metrics)
        deltas = {
            name: self.metrics[name].representative()
            - other.metrics[name].representative()
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
                "no score available (no key metric configured); use assert_metric() instead"
            )
        if self.score < min_score:
            label = self.score_label or "score"
            raise AssertionError(f"{label} {self.score:.4g} < required {min_score:.4g}")

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
        """Fail if metric `metric_name` is missing or below `min_value`.

        Compares the metric's document-level value: the root's, else its mean
        across the tree.

        Args:
            metric_name: The metric to check.
            min_value: The value it must reach.

        Raises:
            AssertionError: If the metric was never computed, or falls short.
        """
        if metric_name not in self.metrics:
            available = ", ".join(sorted(self.metrics)) or "none"
            raise AssertionError(
                f"metric {metric_name!r} not computed (available: {available})"
            )
        value = self.metrics[metric_name].representative()
        if value < min_value:
            raise AssertionError(
                f"metric {metric_name!r} {value:.4g} < required {min_value:.4g}"
            )

    def assert_schema_valid(self) -> None:
        """Fail if schema validation produced errors."""
        coll = self.metrics.get("schema_validity")
        if coll is None:
            return
        errors = coll.extra_values("schema_errors")
        if coll.representative() == 0.0 or errors:
            message = "; ".join(errors) or "schema validation failed"
            raise AssertionError(f"schema invalid: {message}")


# ── Batch / consistency reports ───────────────────────────────────────────────


class BatchEvalReport(BaseModel):
    """Aggregate result over a list of documents (`evaluate_batch`).

    Attributes:
        per_sample: Each document's own report, in input order.
        metrics: The mean of each document-level metric across the samples that
            parsed.
        score: The mean key-metric score.
        score_label: Which metric that was.
        perfect_response_rate: Fraction of samples that parsed with no failing
            field.
        parse_error_rate: Fraction that failed to parse.
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

        Args:
            threshold: One bar for every field; when omitted each field's own
                `threshold` applies, falling back to 1.0.

        Returns:
            Each path mapped to its statistics, where `fail_rate` is the
            fraction of samples scoring below the bar. Only nodes with a score
            (a key metric applied) are counted.
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
        from structured_eval.reporting import render

        print(render(self))  # noqa: T201


class ConsistencyReport(BaseModel):
    """Stability of repeated runs of one prompt (`evaluate_consistency`).

    Attributes:
        per_run: Each run's own report, in input order.
        field_variance: The variance of each field's score across runs.
        stable_fields: Paths whose variance stayed at or below the threshold.
        unstable_fields: The paths that did not.
        mean_score: The mean document-level key-metric score across runs.
        score_variance: The variance of that score.
    """

    per_run: list[EvalReport] = Field(default_factory=list)
    field_variance: dict[str, float] = Field(default_factory=dict)
    stable_fields: list[str] = Field(default_factory=list)
    unstable_fields: list[str] = Field(default_factory=list)
    mean_score: float | None = None
    score_variance: float | None = None

    def print_summary(self) -> None:
        """Print a consistency summary (stable/unstable fields + variance)."""
        from structured_eval.reporting import render

        print(render(self))  # noqa: T201
