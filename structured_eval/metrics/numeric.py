"""The `numeric` metric — numeric equality within a tolerance band."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from structured_eval.metrics.base import FieldMetric
from structured_eval.metrics.utils.null import both_null
from structured_eval.metrics.utils.number import parse_number


class NumericMode(StrEnum):
    """Tolerance band for the single-band form of `Numeric`.

    Attributes:
        RELATIVE: Measures `|a - e| / |e|`.
        ABSOLUTE: Measures `|a - e|`.
    """

    RELATIVE = "relative"
    ABSOLUTE = "absolute"


class Numeric(FieldMetric):
    """Numeric equality within a tolerance band → 1.0, otherwise 0.0.

    Values are parsed leniently, so `"$1,234.50"`, `"(123)"` and `"1e3"` are all
    read as numbers. Two `None`s agree; a one-sided `None` is 0.0.

    Example:
        >>> from structured_eval import evaluate
        >>> from structured_eval.metrics import Numeric
        >>> from structured_eval.models import EvalConfig
        >>> Numeric(tolerance=0.01).score(100.5, 100)       # within ±1%
        1.0
        >>> Numeric(tolerance=0.01).score(110, 100)
        0.0
        >>> Numeric(absolute_tolerance=5).score(103, 100)   # within ±5 units
        1.0
        >>> Numeric().score("$1,234.50", 1234.5)
        1.0
        >>> report = evaluate({"total": "$1,234.50", "tax": 110},
        ...                   {"total": 1234.5, "tax": 100},
        ...                   EvalConfig(metrics=[Numeric(tolerance=0.01)]))
        >>> float(report.metrics["numeric"].representative())
        0.5
    """

    name = "numeric"

    def __init__(
        self,
        tolerance: float = 0.01,
        mode: NumericMode = NumericMode.RELATIVE,
        relative_tolerance: float | None = None,
        absolute_tolerance: float | None = None,
        name: str | None = None,
    ):
        """Set the tolerance band, in either of the two forms it accepts.

        Args:
            tolerance: Width of the single band; `0` means exact equality.
            mode: Whether that band is relative or absolute.
            relative_tolerance: Explicit relative band.
            absolute_tolerance: Explicit absolute band.
            name: Per-instance report key.

        Either explicit band takes precedence over `tolerance`/`mode`, and a
        value matches when it falls within *either* of them.
        """
        super().__init__(name=name)
        self.tolerance = tolerance
        self.mode = NumericMode(mode)
        self.relative_tolerance = relative_tolerance
        self.absolute_tolerance = absolute_tolerance

    def score(self, actual: Any, expected: Any) -> float:
        """1.0 when the parsed values fall inside the tolerance band, else 0.0."""
        if both_null(actual, expected):
            return 1.0
        a = parse_number(actual)
        e = parse_number(expected)
        if a is None or e is None:
            return 0.0
        return 1.0 if self._within_tolerance(a, e) else 0.0

    def _within_tolerance(self, a: float, e: float) -> bool:
        if a == e:
            return True

        # Explicit bands take precedence; match within either.
        if self.relative_tolerance is not None or self.absolute_tolerance is not None:
            # Equality is already settled above, so an expected zero can never be
            # matched relatively — only the absolute band can still save it.
            if (
                self.relative_tolerance is not None
                and e != 0
                and abs(a - e) / abs(e) <= self.relative_tolerance
            ):
                return True
            return (
                self.absolute_tolerance is not None
                and abs(a - e) <= self.absolute_tolerance
            )

        # Single-band form (tolerance + mode).
        if self.mode == NumericMode.RELATIVE:
            if e == 0:
                deviation = 0.0 if a == 0 else float("inf")
            else:
                deviation = abs(a - e) / abs(e)
        else:
            deviation = abs(a - e)
        return deviation <= self.tolerance
