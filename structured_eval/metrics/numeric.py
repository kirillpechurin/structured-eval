from __future__ import annotations

import re
from enum import StrEnum
from typing import Any

from structured_eval.metrics.base import FieldMetric


class NumericMode(StrEnum):
    """Tolerance band for the single-band form of :class:`Numeric`."""

    RELATIVE = "relative"  # |a - e| / |e|
    ABSOLUTE = "absolute"  # |a - e|


# Everything that is not part of a (possibly scientific) number — currency
# symbols, thousands separators, whitespace, percent signs, etc. are dropped.
# Kept: digits, decimal point, signs, and the exponent marker e/E, so float()
# parses scientific notation ("1e3" → 1000.0, "1.5e-3" → 0.0015). A "%" is only
# stripped, never interpreted: "50%" parses to 50, not 0.5.
_NON_NUMERIC = re.compile(r"[^0-9eE.+\-]")


class Numeric(FieldMetric):
    """Numeric equality within a tolerance band → 1.0, otherwise 0.0.

    Values are parsed leniently: currency symbols and thousands separators are
    stripped (``"$1,234.50"`` → ``1234.50``), accounting notation is honored
    (``"(123)"`` → ``-123``), and scientific notation is supported
    (``"1e3"`` → ``1000``). A percent sign is only stripped, **not** interpreted
    (``"50%"`` → ``50``, not ``0.5``). US format is assumed (``,`` = thousands,
    ``.`` = decimal); other shapes that don't parse cleanly yield 0.0.

    Tolerance can be given two ways:

    * ``tolerance`` + ``mode`` (``"relative"`` | ``"absolute"``) — the original
      single-band form; ``relative`` measures ``|a - e| / |e|``, ``absolute``
      measures ``|a - e|``. A tolerance of 0 means exact numeric equality.
    * ``relative_tolerance`` and/or ``absolute_tolerance`` — explicit bands; a
      value matches if it falls within *either* band. When either is supplied it
      takes precedence over ``tolerance``/``mode``.
    """

    name = "numeric"

    def __init__(
        self,
        tolerance: float = 0.01,
        mode: NumericMode = NumericMode.RELATIVE,
        relative_tolerance: float | None = None,
        absolute_tolerance: float | None = None,
    ):
        self.tolerance = tolerance
        self.mode = NumericMode(mode)
        self.relative_tolerance = relative_tolerance
        self.absolute_tolerance = absolute_tolerance

    def score(self, actual: Any, expected: Any) -> float:
        a = self._to_number(actual)
        e = self._to_number(expected)
        if a is None or e is None:
            return 0.0
        return 1.0 if self._within_tolerance(a, e) else 0.0

    def _within_tolerance(self, a: float, e: float) -> bool:
        if a == e:
            return True

        # Explicit bands take precedence; match within either.
        if self.relative_tolerance is not None or self.absolute_tolerance is not None:
            if self.relative_tolerance is not None:
                if e == 0:
                    if a == 0:
                        return True
                elif abs(a - e) / abs(e) <= self.relative_tolerance:
                    return True
            if self.absolute_tolerance is not None:
                if abs(a - e) <= self.absolute_tolerance:
                    return True
            return False

        # Single-band form (tolerance + mode).
        if self.mode == NumericMode.RELATIVE:
            if e == 0:
                deviation = 0.0 if a == 0 else float("inf")
            else:
                deviation = abs(a - e) / abs(e)
        else:
            deviation = abs(a - e)
        return deviation <= self.tolerance

    @staticmethod
    def _to_number(value: Any) -> float | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if not isinstance(value, str):
            return None

        text = value.strip()
        negative = False
        # Accounting notation: "(123)" means -123.
        if text.startswith("(") and text.endswith(")"):
            text = text[1:-1]
            negative = True

        text = _NON_NUMERIC.sub("", text)
        if text in ("", "-", ".", "-."):
            return None
        try:
            number = float(text)
        except ValueError:
            return None
        return -number if negative else number
