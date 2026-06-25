# ObjectPRF1

|            |                                              |
|------------|----------------------------------------------|
| **Class**  | `ObjectPRF1`                                 |
| **Key**    | `object_precision` / `object_recall` / `object_f1` |
| **Branch** | object (`ObjectNode`)                        |
| **Needs**  | `expected`                                   |

## What it measures

All three at once — **precision, recall, and F1** of an object's fields — in a single
metric, computed in one pass. Same numbers as running
[`ObjectPrecision`](object-precision.md), [`ObjectRecall`](object-recall.md), and
[`ObjectF1`](object-f1.md) separately, but cheaper and tidier when you want the full
picture.

## Parameters

Same as [`ObjectF1`](object-f1.md): `score_policy`, `threshold`, `mode`
(default `"hard"`), `weight_mode` (default `"proportional"`).

## How it's computed

The [slot-filling counts](object-f1.md#how-its-computed) once, then all three formulas.
It returns a **dict**, so the engine writes three keys into the report —
`object_precision`, `object_recall`, `object_f1` — instead of one under the metric's own
name.

## Example

```python
from structured_eval import evaluate, EvalConfig, ObjectPRF1

config = EvalConfig(metrics=[ObjectPRF1()])
report = evaluate(
    {"name": "Sarah Johnson", "experience_years": 5},
    {"name": "Sarah Johnson", "experience_years": 8, "title": "PhD"},
    config,
)

# the one metric populates three keys:
float(report.metrics["object_precision"].root())   # 0.5
float(report.metrics["object_recall"].root())      # 0.333
float(report.metrics["object_f1"].root())          # 0.4
```

## Edge cases

- **Three keys, not one** — there's no `object_prf1` entry in `report.metrics`; read the
  three component keys.
- Everything else matches [`ObjectF1`](object-f1.md) (strict `hard`/`1.0` default,
  weights, all child kinds counted).

## See also

- [`ObjectPrecision`](object-precision.md) / [`ObjectRecall`](object-recall.md) /
  [`ObjectF1`](object-f1.md) — the individual metrics.
- [The metric catalog](../index.md) — all metrics and the return-shape model.
