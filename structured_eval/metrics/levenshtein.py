"""The `levenshtein` metric — normalized edit-distance ratio."""

from __future__ import annotations

from structured_eval.metrics.fuzzy import Fuzzy, FuzzyMethod


class Levenshtein(Fuzzy):
    """Normalized Levenshtein ratio — a thin alias over `Fuzzy(RATIO)`.

    RapidFuzz's `ratio` *is* the normalized Levenshtein similarity; this class
    exists only for discoverability. All arithmetic lives in `Fuzzy`.

    Example:
        >>> from structured_eval import evaluate
        >>> from structured_eval.metrics import Levenshtein
        >>> from structured_eval.models import EvalConfig
        >>> round(Levenshtein().score("kitten", "sitting"), 3)
        0.615
        >>> report = evaluate({"word": "kitten"}, {"word": "sitting"},
        ...                   EvalConfig(metrics=[Levenshtein()]))
        >>> round(float(report.field_scores["word"].metrics["levenshtein"]), 3)
        0.615
    """

    name = "levenshtein"

    def __init__(
        self,
        method: FuzzyMethod = FuzzyMethod.RATIO,
        ignore_case: bool = True,
        ignore_whitespace: bool = True,
        name: str | None = None,
    ):
        """Configure the comparison, defaulting to the Levenshtein ratio.

        Args:
            method: Which RapidFuzz scorer to use.
            ignore_case: Compare case-insensitively.
            ignore_whitespace: Collapse runs of whitespace before comparing.
            name: Per-instance report key.
        """
        super().__init__(
            method=method,
            ignore_case=ignore_case,
            ignore_whitespace=ignore_whitespace,
            name=name,
        )
