# DateDistanceScore

|            |                       |
|------------|-----------------------|
| **Class**  | `DateDistanceScore`   |
| **Key**    | `date_distance_score` |
| **Branch** | field (scalar leaves) |
| **Needs**  | `expected`            |

## What it measures

A **graded** similarity for date fields that decreases linearly with the number of days
between two dates. Identical dates score `1.0`, the score drops in proportion to the gap,
and once the gap reaches `max_days` it is `0.0`. Use it where a near-miss date deserves
partial credit (an extracted invoice date off by a day or two) instead of the all-or-nothing
[`ExactMatch`](exact-match.md).

## Parameters

| Param      | Default | Meaning                                                       |
|------------|---------|---------------------------------------------------------------|
| `max_days` | `30`    | gap (in days) at which the score reaches `0.0`. Must be > 0.  |

## How it's computed

```text
score = max(0, 1 − |days(a) − days(e)| / max_days)
```

`date`, `datetime`, and ISO-8601 strings (`"2026-06-29"`) are accepted — strings are
coerced via pydantic. Datetimes are compared by **calendar date only** (time-of-day is
ignored).

## Example

```python
from structured_eval import evaluate
from structured_eval.models import EvalConfig, FieldConfig
from structured_eval.metrics import DateDistanceScore

config = EvalConfig(fields={
    "due_date": FieldConfig(metrics=[DateDistanceScore(max_days=30)]),
})

# 5 days apart, max_days 30 → 1 − 5/30 ≈ 0.833
report = evaluate({"due_date": "2026-06-29"}, {"due_date": "2026-07-04"}, config)
report.field_scores["due_date"].metrics["date_distance_score"]   # 0.833
```

## Edge cases

- **Unparseable → `0.0`** — if either side can't be read as a date (`None`, a non-date
  type, or a string pydantic can't parse) the score is `0.0`.
- **Beyond `max_days` → `0.0`** — the linear score is clamped at `0`; it never goes
  negative.
- **Time ignored** — two datetimes on the same calendar day score `1.0` regardless of
  the clock.
- **`max_days` must be > 0** — a non-positive `max_days` raises `ValueError`.

## See also

- [`ExactMatch`](exact-match.md) — strict date equality (no partial credit).
- [`ExponentialNumericScore`](exponential-numeric-score.md) — the analogous graded decay
  for plain numbers.
- [The metric catalog](../index.md) — all metrics and the return-shape model.
