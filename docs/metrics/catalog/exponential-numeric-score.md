# ExponentialNumericScore

|            |                       |
|------------|-----------------------|
| **Class**  | `ExponentialNumericScore` |
| **Key**    | `exponential_numeric_score` |
| **Branch** | field (scalar leaves) |
| **Needs**  | `expected`            |

## What it measures

A **graded** numeric similarity that decays exponentially with the *absolute* error. An
exact match scores `1.0`; the score falls off smoothly as the values diverge, never
quite reaching `0.0`. Unlike the ratio-based [`NumericCloseness`](numeric-closeness.md)
(scale-free), this is **scale-aware** — you pick how many units of error halve the score
— which suits fields with a natural unit (minutes, dollars, counts) where "off by 2" is
meaningful regardless of magnitude.

## Parameters

| Param   | Default | Meaning                                                          |
|---------|---------|------------------------------------------------------------------|
| `scale` | `1.0`   | error (in field units) of the decay constant; larger = more tolerant. Must be > 0. |

## How it's computed

```text
score = exp(−|a − e| / scale)
```

So the score is `1.0` at zero error and ≈ `0.37` when the error equals `scale`. Both
values are read with the same lenient parser as [`Numeric`](numeric.md), so numeric
strings (`"$120"`) are graded too.

## Example

```python
from structured_eval import evaluate
from structured_eval.models import EvalConfig, FieldConfig
from structured_eval.metrics import ExponentialNumericScore

config = EvalConfig(fields={
    "duration_hours": FieldConfig(metrics=[ExponentialNumericScore(scale=5)]),
})

# off by 2 hours, scale 5 → exp(-2/5) ≈ 0.670
report = evaluate({"duration_hours": 10}, {"duration_hours": 12}, config)
report.field_scores["duration_hours"].metrics["exponential_numeric_score"]   # 0.670
```

## Edge cases

- **Numbers only** — if either side isn't numeric the score is `0.0`: that includes
  non-numeric strings, `bool` (`True` is *not* `1`), and a `None` facing a number,
  matching [`Numeric`](numeric.md) / [`NumericCloseness`](numeric-closeness.md).
- **Both `null` → `1.0`** — a null expectation met by a null value is a correct answer,
  not a type mismatch. Only both sides `None` count; one-sided `None` stays `0.0`.
- **Never exactly `0.0`** — the decay is asymptotic; the score is always in `(0, 1]` for
  two numbers. Pair it with a `threshold` if you need a pass/fail gate.
- **`scale` must be > 0** — a non-positive `scale` raises `ValueError`.
- **Absolute, not relative** — `100` vs `102` and `1` vs `3` both have error `2` and so
  score the same. Use [`NumericCloseness`](numeric-closeness.md) for ratio-based grading.

## See also

- [`NumericCloseness`](numeric-closeness.md) — ratio (scale-free) graded closeness.
- [`Numeric`](numeric.md) — pass/fail equality within a tolerance.
- [The metric catalog](../index.md) — all metrics and the return-shape model.
