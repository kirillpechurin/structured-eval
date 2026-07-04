# ExactMatch

|            |                       |
|------------|-----------------------|
| **Class**  | `ExactMatch`          |
| **Key**    | `exact_match`         |
| **Branch** | field (scalar leaves) |
| **Needs**  | `expected`            |

## What it measures

Strict equality of a leaf value: `1.0` when `actual == expected`, else `0.0`. No
tolerance, no normalization — `"Beginner"` and `"beginner"` differ, and so do `"12"`
and `12`. It's the strictest field metric and the right default when a value must be
reproduced verbatim (codes, enums, ids).

`ExactMatch` is the **default scalar metric**: a scalar leaf you don't configure gets
it automatically. It is also the default **key** comparison used by `by_key` array
alignment (matching elements on a key field).

## Parameters

None — `ExactMatch()`.

## How it's computed

```text
score = 1.0 if actual == expected else 0.0
```

It is a plain Python `==`, so equality follows Python's rules: type usually matters
(`"12" != 12`), but `12 == 12.0` is `True`, and two `None`s compare equal.

## Example

Put `ExactMatch` on the fields you want compared verbatim:

```python
from structured_eval import evaluate
from structured_eval.models import EvalConfig, FieldConfig
from structured_eval.metrics import ExactMatch

config = EvalConfig(fields={
    "level": FieldConfig(metrics=[ExactMatch()]),
    "duration_hours": FieldConfig(metrics=[ExactMatch()]),
})
report = evaluate(
    {"level": "beginner", "duration_hours": 10},
    {"level": "beginner", "duration_hours": 12},
    config,
)

float(report.field_scores["level"].metrics["exact_match"])           # 1.0 — identical
float(report.field_scores["duration_hours"].metrics["exact_match"])  # 0.0 — 10 ≠ 12
```

It is strict about casing and string-vs-number — both of these score `0.0`:

```python
from structured_eval import evaluate
from structured_eval.models import EvalConfig, FieldConfig
from structured_eval.metrics import ExactMatch

config = EvalConfig(fields={
    "level": FieldConfig(metrics=[ExactMatch()]),
    "duration_hours": FieldConfig(metrics=[ExactMatch()]),
})
report = evaluate(
    {"level": "Beginner", "duration_hours": "12"},
    {"level": "beginner", "duration_hours": 12},
    config,
)

float(report.field_scores["level"].metrics["exact_match"])           # 0.0 — casing differs
float(report.field_scores["duration_hours"].metrics["exact_match"])  # 0.0 — str ≠ int
```

## Edge cases

- **Casing / whitespace** — significant. Use [`RegexMatch`](regex-match.md)
  to ignore them.
- **Numbers** — `12 == 12.0` (both `1.0`), but a number as a string does **not** match
  (`"12" != 12`), and there's no tolerance. Use [`Numeric`](numeric.md) for tolerant
  numeric equality that also strips currency/separators.
- **Both `None`** — compare equal → `1.0`.
- **Near-misses** — no partial credit; a one-character typo is `0.0`. Use
  [`Fuzzy`](fuzzy.md) or [`TokenF1`](token-f1.md) for graded text similarity.
- **Object / array fields** — `ExactMatch` is a *field* metric; it only attaches to
  scalar leaves. Configured on an object or array node it is ignored, and the node
  keeps its type default ([`ObjectAccuracy`](object-accuracy.md) /
  [`ArrayAccuracy`](array-accuracy.md)).

## See also

- [`RegexMatch`](regex-match.md) — equality ignoring casing and spacing.
- [`Numeric`](numeric.md) — numeric equality within a tolerance.
- [`Fuzzy`](fuzzy.md) / [`TokenF1`](token-f1.md) — graded string similarity.
- [The metric catalog](../index.md) — all metrics and the return-shape model.
