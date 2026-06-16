"""Aggregation over multiple EvalReports: batch and consistency statistics."""

from __future__ import annotations

from statistics import mean, pvariance

from structured_eval.model.result import (
    BatchEvalReport,
    ConsistencyReport,
    EvalReport,
)


class BatchAggregator:
    """Combines per-document reports into batch / consistency summaries."""

    def batch(self, reports: list[EvalReport]) -> BatchEvalReport:
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
            metrics=self._mean_metrics(reports),
            score=score,
            score_label=label,
            perfect_response_rate=(perfect / n) if n else 0.0,
            parse_error_rate=(errors / n) if n else 0.0,
        )

    def consistency(
        self, reports: list[EvalReport], variance_threshold: float = 0.05
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
            score_variance=(pvariance(scores) if len(scores) > 1 else 0.0) if scores else None,
        )

    @staticmethod
    def _mean_metrics(reports: list[EvalReport]) -> dict[str, float]:
        """Mean of each metric across reports that carry it (errors excluded)."""
        buckets: dict[str, list[float]] = {}
        for r in reports:
            if r.parse_error:
                continue
            for name, value in r.metrics.items():
                buckets.setdefault(name, []).append(value)
        return {name: mean(vals) for name, vals in buckets.items() if vals}
