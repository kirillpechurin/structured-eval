# ObjectPrecision

|            |                       |
|------------|-----------------------|
| **Class**  | `ObjectPrecision`     |
| **Key**    | `object_precision`    |
| **Branch** | object (`ObjectNode`) |
| **Needs**  | `expected`            |

## What it measures

**Precision** over an object's fields: of the fields the model produced, how many are
correct — `TP / (TP + FP)`. A produced field is a TP when its value clears its
threshold, a FP when it's extra (not expected) or wrong. Answers "when the model fills a
field, can I trust it?"

Shares the [slot-filling mechanism](object-f1.md#how-its-computed) of `ObjectF1`.

## Parameters

Same as [`ObjectF1`](object-f1.md): `score_policy`, `threshold`, `mode`
(default `"hard"`), `weight_mode` (default `"proportional"`).

## How it's computed

```text
matched field → TP if its score clears its threshold (HARD), else not
extra field → FP

score = TP / (TP + FP)        # produced-and-correct over all produced fields
```

Default `mode="hard"` with field threshold `1.0` (a field must score a perfect match to
count); `mode="soft"` counts fractional scores. An object that produced no fields is
vacuously `1.0`.

## Example

`name` is right, `experience_years` is wrong — both were produced, so half are correct:

```python
from structured_eval import evaluate, EvalConfig, ObjectPrecision

config = EvalConfig(metrics=[ObjectPrecision()])
report = evaluate(
    {"name": "Sarah Johnson", "experience_years": 5},
    {"name": "Sarah Johnson", "experience_years": 8, "title": "PhD"},
    config,
)

# produced 2 fields (name✓, experience✗), 0 extra  →  TP=1, FP=0 over 2 produced
float(report.metrics["object_precision"].root())   # 0.5
```

## Edge cases

- **Missing fields don't lower precision** — they're FN (a [recall](object-recall.md)
  concern), not FP. In the example, the missing `title` doesn't affect precision.
- **Strict default** — `hard` / threshold `1.0`; use `soft` or lower thresholds for
  partial credit.
- **No produced fields** → vacuously `1.0`.

## See also

- [`ObjectRecall`](object-recall.md) — the complementary half.
- [`ObjectF1`](object-f1.md) — their harmonic mean.
- [`ObjectPRF1`](object-prf1.md) — all three at once.
- [The metric catalog](../index.md) — all metrics and the return-shape model.
