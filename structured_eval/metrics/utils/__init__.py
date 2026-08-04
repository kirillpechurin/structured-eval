"""Utilities shared across metric implementations (metric-layer only).

- `calculate` — the precision / recall / F1 arithmetic.
- `object_utils` — an object's matched fields as verdicts for `calculate`.
- `array` — the same for an array's aligned items.
- `number` — the lenient numeric parsing the numeric field metrics share.
- `null` — the `(None, None) → 1.0` rule the comparison field metrics share.
"""
