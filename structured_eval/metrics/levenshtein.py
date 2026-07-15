from __future__ import annotations

from structured_eval.metrics.fuzzy import Fuzzy, FuzzyMethod


class Levenshtein(Fuzzy):
    """Normalized Levenshtein ratio — a thin alias over ``Fuzzy(RATIO)``.

    RapidFuzz's ``ratio`` *is* the normalized Levenshtein similarity; this class
    exists only for discoverability. All arithmetic lives in ``Fuzzy``.
    """

    name = "levenshtein"

    def __init__(
        self,
        method: FuzzyMethod = FuzzyMethod.RATIO,
        ignore_case: bool = True,
        ignore_whitespace: bool = True,
        name: str | None = None,
    ):
        super().__init__(
            method=method,
            ignore_case=ignore_case,
            ignore_whitespace=ignore_whitespace,
            name=name,
        )
