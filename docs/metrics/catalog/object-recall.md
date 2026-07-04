# ObjectRecall

|            |                       |
|------------|-----------------------|
| **Class**  | `ObjectRecall`        |
| **Key**    | `object_recall`       |
| **Branch** | object (`ObjectNode`) |
| **Needs**  | `expected`            |

## What it measures

**Recall** over an object's fields: of the fields that *should* be there, how many were
produced correctly — `TP / (TP + FN)`. A missing or wrong expected field is a FN.
Answers "how much of what we expected did the model actually get right?"

Shares the [slot-filling mechanism](object-f1.md#how-its-computed) of `ObjectF1`.

## Parameters

Same as [`ObjectF1`](object-f1.md): `score_policy`, `threshold`, `mode`
(default `"hard"`), `weight_mode` (default `"proportional"`).

## How it's computed

```text
matched field → TP if its score clears its threshold (HARD), else not
missing field → FN     

score = TP / (TP + FN)        # expected-and-correct over all expected fields
```

Default `mode="hard"` with field threshold `1.0`; `mode="soft"` counts fractional
scores. With `mode="soft"` this is exactly what [`ObjectAccuracy`](object-accuracy.md)
computes.

## Example

`name` is right, `experience_years` is wrong, `title` is missing — one of three expected
fields is correct:

```python
from structured_eval import evaluate
from structured_eval.models import EvalConfig
from structured_eval.metrics import ObjectRecall

config = EvalConfig(metrics=[ObjectRecall()])
report = evaluate(
    {"name": "Sarah Johnson", "experience_years": 5},
    {"name": "Sarah Johnson", "experience_years": 8, "title": "PhD"},
    config,
)

# expected 3 (name✓, experience✗, title missing)  →  TP=1 over 3 expected
float(report.metrics["object_recall"].root())   # 0.333
```

## Edge cases

- **Extra fields don't lower recall** — they're FP (a [precision](object-precision.md)
  concern).
- **Strict default** — `hard` / threshold `1.0`; `soft` gives partial credit (and equals
  [`ObjectAccuracy`](object-accuracy.md)).
- **No expected fields** → vacuously `1.0`.

## See also

- [`ObjectPrecision`](object-precision.md) — the complementary half.
- [`ObjectF1`](object-f1.md) — their harmonic mean.
- [`ObjectAccuracy`](object-accuracy.md) — the soft version of recall.
- [The metric catalog](../index.md) — all metrics and the return-shape model.
