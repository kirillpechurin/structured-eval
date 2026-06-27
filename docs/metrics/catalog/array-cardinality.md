# ArrayCardinality

|            |                       |
|------------|-----------------------|
| **Class**  | `ArrayCardinality`    |
| **Key**    | `array_cardinality`   |
| **Branch** | array (`ArrayNode`)   |
| **Needs**  | `expected`            |

## What it measures

How close the **lengths** of two arrays are, in `[0, 1]` — a cheap count-agreement check,
**independent of element correctness**. It only asks "did the model produce about the
right number of items?", not whether they're right. Useful as a structural signal
alongside a value metric.

## Parameters

None — `ArrayCardinality()`.

## How it's computed

```text
score = min(|actual|, |expected|) / max(|actual|, |expected|)
```

The ratio of the shorter length to the longer (the same shape as
[`NumericCloseness`](numeric-closeness.md) on counts). Two empty arrays are vacuously
`1.0`.

## Example

Four expected items, three produced — regardless of whether those three are correct:

```python
from structured_eval import evaluate, EvalConfig, ArrayFieldConfig, ArrayCardinality

config = EvalConfig(fields={"tags": ArrayFieldConfig(metrics=[ArrayCardinality()])})
report = evaluate(
    {"tags": ["python", "data", "xxx"]},        # 3 items
    {"tags": ["python", "data", "web", "ml"]},  # 4 items
    config,
)

float(report.field_scores["tags"].metrics["array_cardinality"])   # 0.75 — min(3,4)/max(3,4)
```

## Edge cases

- **Value-blind** — only counts matter; the wrong `xxx` doesn't change it. Pair with
  [`ArrayF1`](array-f1.md) / [`ArrayAccuracy`](array-accuracy.md) to judge content.
- **Counts come from the alignment** — `|actual| = matched + spurious`,
  `|expected| = matched + missed`.
- **Both empty** → vacuously `1.0`.

## See also

- [`ArrayF1`](array-f1.md) / [`ArrayAccuracy`](array-accuracy.md) — element correctness,
  not just count.
- [`NumericCloseness`](numeric-closeness.md) — the same ratio shape on numbers.
- [The metric catalog](../index.md) — all metrics and the return-shape model.
