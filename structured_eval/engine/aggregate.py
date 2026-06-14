"""Aggregation over multiple EvalReports: batch and consistency statistics."""

from __future__ import annotations

from statistics import mean, pvariance

from structured_eval.core.result import (
    BatchEvalReport,
    ConsistencyReport,
    EvalReport,
)


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


def _mean_metrics(reports: list[EvalReport]) -> dict[str, float]:
    """Mean of each metric across reports that carry it (errors excluded)."""
    buckets: dict[str, list[float]] = {}
    for r in reports:
        if r.parse_error:
            continue
        for name, value in r.metrics.items():
            buckets.setdefault(name, []).append(value)
    return {name: mean(vals) for name, vals in buckets.items() if vals}


def batch_report(reports: list[EvalReport]) -> BatchEvalReport:
    """Aggregate a list of single-document reports into a BatchEvalReport."""
    n = len(reports)
    errors = sum(1 for r in reports if r.parse_error)
    ok = [r for r in reports if not r.parse_error]

    scores = [r.score for r in ok if r.score is not None]
    score = mean(scores) if scores else None
    label = next((r.score_label for r in ok if r.score_label is not None), None)

    perfect = sum(1 for r in ok if not r.failed_fields())

    return BatchEvalReport(
        per_sample=reports,
        metrics=_mean_metrics(reports),
        score=score,
        score_label=label,
        perfect_response_rate=(perfect / n) if n else 0.0,
        parse_error_rate=(errors / n) if n else 0.0,
    )


def field_breakdown(
    reports: list[EvalReport], threshold: float | None = None
) -> dict[str, dict[str, float]]:
    """Per-path mean/min/max/p95/fail_rate across reports (scored nodes only)."""
    scores: dict[str, list[float]] = {}
    fails: dict[str, int] = {}
    for r in reports:
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

    breakdown: dict[str, dict[str, float]] = {}
    for path, vals in scores.items():
        breakdown[path] = {
            "mean": mean(vals),
            "min": min(vals),
            "max": max(vals),
            "p95": _percentile(vals, 0.95),
            "fail_rate": fails.get(path, 0) / len(vals),
        }
    return breakdown


def consistency_report(
    reports: list[EvalReport], variance_threshold: float = 0.05
) -> ConsistencyReport:
    """Measure run-to-run stability across repeated outputs of one prompt."""
    ok = [r for r in reports if not r.parse_error]

    by_path: dict[str, list[float]] = {}
    for r in ok:
        for path, fs in r.field_scores.items():
            if fs.score is None:
                continue
            by_path.setdefault(path, []).append(fs.score)

    variance: dict[str, float] = {}
    stable: list[str] = []
    unstable: list[str] = []
    for path, vals in by_path.items():
        var = pvariance(vals) if len(vals) > 1 else 0.0
        variance[path] = var
        (stable if var <= variance_threshold else unstable).append(path)

    scores = [r.score for r in ok if r.score is not None]
    return ConsistencyReport(
        per_run=reports,
        field_variance=variance,
        stable_fields=stable,
        unstable_fields=unstable,
        mean_score=mean(scores) if scores else None,
        score_variance=(pvariance(scores) if len(scores) > 1 else 0.0)
        if scores
        else None,
    )
