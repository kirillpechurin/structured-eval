"""Aggregation over multiple EvalReports: batch and consistency statistics."""

from __future__ import annotations

from statistics import mean, pvariance

from structured_eval.models import (
    BatchEvalReport,
    ConsistencyReport,
    EvalReport,
    NodeType,
)


class BatchAggregator:
    """Combines per-document reports into batch / consistency summaries.

    The two summaries read the same reports along different axes: `batch` treats
    them as independent documents and averages across them, `consistency` treats
    them as repeated runs of one prompt and measures how much each field wobbles
    between them.

    A report that failed to parse stays in the result but contributes no scores.

    Example:
        >>> from structured_eval import evaluate
        >>> from structured_eval.engine import BatchAggregator
        >>> reports = [evaluate({"status": status}, {"status": "paid"})
        ...            for status in ("paid", "due")]
        >>> batch = BatchAggregator().batch(reports)
        >>> batch.perfect_response_rate, batch.score
        (0.5, 0.5)
        >>> BatchAggregator().consistency(reports).unstable_fields
        ['status']
    """

    def batch(self, reports: list[EvalReport]) -> BatchEvalReport:
        """Aggregate a list of single-document reports.

        Args:
            reports: One report per document, parse errors included.

        Returns:
            The batch report: the reports themselves, the mean of each metric
            over those that parsed, the mean document score, and the
            `perfect_response_rate` / `parse_error_rate` shares over all of them.

        Example:
            >>> from structured_eval import evaluate
            >>> from structured_eval.engine import BatchAggregator
            >>> reports = [evaluate({"status": status}, {"status": "paid"})
            ...            for status in ("paid", "due")]
            >>> batch = BatchAggregator().batch(reports)
            >>> batch.parse_error_rate, batch.metrics["exact_match"]
            (0.0, 0.5)
        """
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
        """Measure run-to-run stability across repeated outputs of one prompt.

        Args:
            reports: One report per run of the same prompt.
            variance_threshold: The bar a field's score variance must stay at
                or below to count as stable.

        Returns:
            The stability report: the reports themselves, the score variance of
            each leaf field, the split into stable and unstable fields, and the
            mean and variance of the document score.

        Example:
            >>> from structured_eval import evaluate
            >>> from structured_eval.engine import BatchAggregator
            >>> reports = [evaluate({"status": status}, {"status": "paid"})
            ...            for status in ("paid", "paid", "due")]
            >>> summary = BatchAggregator().consistency(reports)
            >>> summary.stable_fields, summary.unstable_fields
            ([], ['status'])
        """
        ok = [r for r in reports if not r.parse_error]

        by_path: dict[str, list[float]] = {}
        for r in ok:
            for path, fs in r.field_scores.items():
                # Field-level stability currently tracks leaf fields only: object/
                # array nodes carry an aggregate representative score whose variance
                # is just a function of its children's, so including it here would
                # be redundant (double-counting the same wobble), non-actionable
                # (a parent path doesn't point at a concrete field to fix) and
                # noisy (an F1-over-children varies for different reasons than a
                # single atomic value). Hence the leaf filter.
                #
                # TODO: support per-node stability regardless of node type. Some
                # users want block-level wobble ("the whole `address` object is
                # unstable") without drilling into leaves. The fix is NOT to drop
                # this filter (that mixes scales) but to expose a separate,
                # parallel view computed over non-scalar nodes (e.g.
                # ConsistencyReport.object_variance / block_variance), keeping the
                # leaf map clean and adding the aggregate one alongside it.
                if fs.score is None or fs.node_type != NodeType.SCALAR:
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

    @staticmethod
    def _mean_metrics(reports: list[EvalReport]) -> dict[str, float]:
        """Mean of each metric across reports that carry it (errors excluded)."""
        buckets: dict[str, list[float]] = {}
        for r in reports:
            if r.parse_error:
                continue
            for name, coll in r.metrics.items():
                buckets.setdefault(name, []).append(coll.representative())
        return {name: mean(vals) for name, vals in buckets.items() if vals}
