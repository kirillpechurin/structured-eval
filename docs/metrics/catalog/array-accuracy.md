# ArrayAccuracy

|            |                     |
|------------|---------------------|
| **Class**  | `ArrayAccuracy`     |
| **Key**    | `array_accuracy`    |
| **Branch** | array (`ArrayNode`) |
| **Needs**  | `expected`          |

> The **default array metric**: an array node you don't configure gets it.

## What it measures

The **mean correctness of an array's expected elements** — a soft score in `[0, 1]`.
After [alignment](../core-concepts/array-alignment.md), each matched element contributes
its own score (its `representative`); missing expected elements count as `0.0`. Like
[`ArrayAccuracy`'s object cousin](object-accuracy.md), it's **recall-flavored**: extra
(spurious) elements are *not* penalized.

Use it as a forgiving "how good are the elements we expected?"; use
[`ArrayF1`](array-f1.md) when extra elements should count against the score.

## Parameters

None — `ArrayAccuracy()`. (Soft by nature; no threshold/mode. Alignment is configured on
the field's `ArrayFieldConfig`.)

## How it's computed

```text
align → matched pairs, missing (FN), spurious (FP)
matched element → TP if its score ≥ threshold (HARD)

score = Σ representative(matched) / (#matched + #missing)
```

The denominator is the **expected** count (matched + missing), so spurious elements don't
change it. Equivalent to soft recall.

## Example

Four expected tags, three produced — one wrong (`xxx`), one missing (`ml`):

```python
from structured_eval import evaluate, EvalConfig, ArrayFieldConfig, ArrayAccuracy

config = EvalConfig(fields={"tags": ArrayFieldConfig(metrics=[ArrayAccuracy()])})
report = evaluate(
    {"tags": ["python", "data", "xxx"]},
    {"tags": ["python", "data", "web", "ml"]},
    config,
)

# matched scores: python 1.0, data 1.0, (xxx↔web) 0.0 ; ml missing → 0.0
# mean over 4 expected = (1 + 1 + 0 + 0) / 4
float(report.field_scores["tags"].metrics["array_accuracy"])   # 0.5
```

## Edge cases

- **Recall-flavored** — spurious/extra elements aren't penalized (only the expected count
  is in the denominator). Use [`ArrayF1`](array-f1.md) for a precision-aware score.
- **Soft** — graded item metrics flow through (no threshold).
- **Count-based** — every element weighs 1.0.
- **Empty / fully-missed array** → vacuously `1.0`.

## See also

- [`ArrayF1`](array-f1.md) — precision-aware (penalizes extra elements).
- [`ArrayRecall`](array-recall.md) — the hard-threshold recall this softens.
- [`ArrayCardinality`](array-cardinality.md) — count agreement only.
- [The metric catalog](../index.md) — all metrics and the return-shape model.
