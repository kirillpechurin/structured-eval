# NumericCloseness

|            |                       |
|------------|-----------------------|
| **Class**  | `NumericCloseness`    |
| **Key**    | `numeric_closeness`   |
| **Branch** | field (scalar leaves) |
| **Needs**  | `expected`            |

## What it measures

A **graded** numeric similarity in `[0, 1]` — not a pass/fail verdict. It scores how
close two numbers are by their ratio: identical values score `1.0`, and the score falls
off smoothly as they diverge. Use it for numeric fields where you want partial credit
(being off by a little should beat being off by a lot), e.g. as a soft signal in
dashboards or array matching.

Contrast with [`Numeric`](numeric.md), which is all-or-nothing against a tolerance.
`NumericCloseness` is also the default element scorer for numbers under Hungarian array
alignment, where a graded cost matters.

## Parameters

None — `NumericCloseness()`.

## How it's computed

```text
score = 1 − |a − e| / max(|a|, |e|)        (clamped to [0, 1])
```

For same-sign values this is exactly `min(|a|, |e|) / max(|a|, |e|)` — the **ratio
similarity**, the smaller magnitude over the larger. It is symmetric in `a` and `e`,
equals `1.0` at equality, and reaches `0.0` when the values have opposite signs. Both
values are read with the same lenient parser as [`Numeric`](numeric.md).

## Example

Each field gets partial credit proportional to how close it is:

```python
from structured_eval import evaluate
from structured_eval.models import EvalConfig, FieldConfig
from structured_eval.metrics import NumericCloseness

config = EvalConfig(fields={
    "duration_hours": FieldConfig(metrics=[NumericCloseness()]),
    "experience_years": FieldConfig(metrics=[NumericCloseness()]),
})
report = evaluate(
    {"duration_hours": 10, "experience_years": 7},
    {"duration_hours": 12, "experience_years": 8},
    config,
)

float(report.field_scores["duration_hours"].metrics["numeric_closeness"])    # 0.833 — 10/12
float(report.field_scores["experience_years"].metrics["numeric_closeness"])  # 0.875 — 7/8
```

Numeric strings are parsed before grading, so messy values still get a graded score:

```python
from structured_eval import evaluate
from structured_eval.models import EvalConfig, FieldConfig
from structured_eval.metrics import NumericCloseness

config = EvalConfig(fields={"price": FieldConfig(metrics=[NumericCloseness()])})

float(evaluate({"price": "$120"}, {"price": 100}, config)
      .field_scores["price"].metrics["numeric_closeness"])   # 0.833 — parsed, then 100/120
```

## Edge cases

- **Numbers only** — if either side isn't numeric the score is `0.0`. That includes
  `None` (a null makes the metric inapplicable → `0.0`, never a match), non-numeric
  strings, and `bool` (`True` is *not* `1`).
- **Opposite signs → `0.0`** — `5` vs `-5` scores `0.0` (the formula is clamped at 0).
- **`0` vs `0` → `1.0`** — equal values always score `1.0`, including both-zero.
- **Lenient parsing** — same as [`Numeric`](numeric.md): strips currency/separators,
  honors `"(123)"` → `−123`, supports `"1e3"`; `"%"` is stripped not interpreted.
- **Graded, not pass/fail** — there's no threshold here. For a hard 0/1 verdict against
  a tolerance use [`Numeric`](numeric.md).

## See also

- [`Numeric`](numeric.md) — pass/fail numeric equality within a tolerance.
- [`ExactMatch`](exact-match.md) — strict equality without parsing.
- [The metric catalog](../index.md) — all metrics and the return-shape model.
