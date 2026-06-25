# Numeric

|            |                       |
|------------|-----------------------|
| **Class**  | `Numeric`             |
| **Key**    | `numeric`             |
| **Branch** | field (scalar leaves) |
| **Needs**  | `expected`            |

## What it measures

Numeric equality **within a tolerance**: `1.0` when the two numbers are close enough,
else `0.0`. It parses leniently first — `"$1,299.00"`, `"(50)"`, `"1e3"` all become
numbers — then checks the difference against a tolerance band. Use it for numeric
fields where small rounding differences shouldn't count as wrong.

By **default the tolerance is ±1% relative**, not exact: `12` and `12.1` match, `12`
and `13` don't.

## Parameters

| Param                | Default      | Meaning                                              |
|----------------------|--------------|------------------------------------------------------|
| `tolerance`          | `0.01`       | size of the single band; `0` means exact equality    |
| `mode`               | `"relative"` | `"relative"` = `abs(a−e)/abs(e)`, `"absolute"` = `abs(a−e)` |
| `relative_tolerance` | `None`       | explicit relative band                               |
| `absolute_tolerance` | `None`       | explicit absolute band                               |

Two ways to set the band: the simple `tolerance` + `mode`, or the explicit
`relative_tolerance` / `absolute_tolerance`. If either explicit band is given it
**takes precedence**, and a value matches if it falls within *either* one.

## How it's computed

```text
a, e = parse(actual), parse(expected)        # lenient; non-numeric → 0.0
match if a == e
   or (relative band given and |a−e|/|e| ≤ relative_tolerance)
   or (absolute band given and |a−e|     ≤ absolute_tolerance)
   else fall back to tolerance + mode
score = 1.0 if match else 0.0
```

## Example

The default band is relative ±1%, so a tiny difference still matches but a real one
doesn't:

```python
from structured_eval import evaluate, EvalConfig, FieldConfig, Numeric

config = EvalConfig(fields={"duration_hours": FieldConfig(metrics=[Numeric()])})

# 12.1 is within 1% of 12; 13 is not
float(evaluate({"duration_hours": 12.1}, {"duration_hours": 12}, config)
      .field_scores["duration_hours"].metrics["numeric"])   # 1.0
float(evaluate({"duration_hours": 13}, {"duration_hours": 12}, config)
      .field_scores["duration_hours"].metrics["numeric"])   # 0.0
```

Use an **absolute** band when you want "within N", regardless of magnitude — here,
±1 hour:

```python
from structured_eval import evaluate, EvalConfig, FieldConfig, Numeric

config = EvalConfig(fields={
    "duration_hours": FieldConfig(metrics=[Numeric(absolute_tolerance=1)]),
})

float(evaluate({"duration_hours": 13}, {"duration_hours": 12}, config)
      .field_scores["duration_hours"].metrics["numeric"])   # 1.0 — off by 1
float(evaluate({"duration_hours": 14}, {"duration_hours": 12}, config)
      .field_scores["duration_hours"].metrics["numeric"])   # 0.0 — off by 2
```

Strings are parsed leniently, so messy numeric text still compares (here exactly, with
`tolerance=0`):

```python
from structured_eval import evaluate, EvalConfig, FieldConfig, Numeric

config = EvalConfig(fields={"price": FieldConfig(metrics=[Numeric(tolerance=0)])})

float(evaluate({"price": "$1,299.00"}, {"price": 1299.0}, config)
      .field_scores["price"].metrics["numeric"])   # 1.0 — currency & separators stripped
float(evaluate({"price": "(50)"}, {"price": -50}, config)
      .field_scores["price"].metrics["numeric"])   # 1.0 — accounting notation = −50
float(evaluate({"price": "1e3"}, {"price": 1000}, config)
      .field_scores["price"].metrics["numeric"])   # 1.0 — scientific notation
```

## Edge cases

- **Default isn't exact** — it's ±1% relative. Pass `tolerance=0` for strict numeric
  equality (`100.0 == 100`, but unlike [`ExactMatch`](exact-match.md), `"100"` parses
  and matches too).
- **Lenient parsing** — strips currency, thousands separators, and whitespace; honors
  accounting `"(123)"` → `−123`; supports scientific notation `"1e3"` → `1000`.
- **Percent is stripped, not interpreted** — `"50%"` parses to `50`, **not** `0.5`. So
  `"50%"` matches `50`, not `0.5`. Store percentages the way you'll compare them.
- **US number format** — `,` = thousands, `.` = decimal. European `"1.234,50"` doesn't
  parse and scores `0.0`.
- **Non-numeric → `0.0`** — anything that doesn't parse (including `bool`: `True` is
  *not* `1` here).
- **All-or-nothing** — it's a pass/fail band, no partial credit. For a *graded* closeness
  score use [`NumericCloseness`](numeric-closeness.md).

## See also

- [`NumericCloseness`](numeric-closeness.md) — graded closeness instead of pass/fail.
- [`ExactMatch`](exact-match.md) — strict equality without parsing.
- [The metric catalog](../index.md) — all metrics and the return-shape model.
