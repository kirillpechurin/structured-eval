# ArrayPRF1

|            |                                          |
|------------|------------------------------------------|
| **Class**  | `ArrayPRF1`                              |
| **Key**    | `array_precision` / `array_recall` / `array_f1` |
| **Branch** | array (`ArrayNode`)                      |
| **Needs**  | `expected`                               |

## What it measures

All three at once — **precision, recall, and F1** of an array's elements — in a single
metric, computed in one pass over the [alignment](../core-concepts/array-alignment.md).
Same numbers as [`ArrayPrecision`](array-precision.md) + [`ArrayRecall`](array-recall.md)
+ [`ArrayF1`](array-f1.md), but tidier when you want the whole picture.

## Parameters

| Param       | Default  | Meaning                                                     |
|-------------|----------|-------------------------------------------------------------|
| `threshold` | `1.0`    | the score an aligned element must clear to count as correct |
| `mode`      | `"hard"` | `hard` = threshold gate; `soft` = add element scores fractionally |

## How it's computed

The [slot-filling counts](array-f1.md#how-its-computed) once, then all three formulas. It
returns a **dict**, so the engine writes three keys — `array_precision`, `array_recall`,
`array_f1` — instead of one under the metric's own name.

## Example

```python
from structured_eval import evaluate
from structured_eval.models import ArrayFieldConfig, EvalConfig
from structured_eval.metrics import ArrayPRF1

config = EvalConfig(fields={"tags": ArrayFieldConfig(metrics=[ArrayPRF1()])})
report = evaluate(
    {"tags": ["python", "data", "xxx"]},
    {"tags": ["python", "data", "web", "ml"]},
    config,
)

fs = report.field_scores["tags"]
float(fs.metrics["array_precision"])   # 0.667
float(fs.metrics["array_recall"])      # 0.5
float(fs.metrics["array_f1"])          # 0.571
```

## Edge cases

- **Three keys, not one** — there's no `array_prf1` entry; read the three component keys.
- Everything else matches [`ArrayF1`](array-f1.md) (strict `hard`/`1.0` default,
  count-based, alignment-dependent).

## See also

- [`ArrayPrecision`](array-precision.md) / [`ArrayRecall`](array-recall.md) /
  [`ArrayF1`](array-f1.md) — the individual metrics.
- [The metric catalog](../index.md) — all metrics and the return-shape model.
