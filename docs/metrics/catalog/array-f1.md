# ArrayF1

|            |                     |
|------------|---------------------|
| **Class**  | `ArrayF1`           |
| **Key**    | `array_f1`          |
| **Branch** | array (`ArrayNode`) |
| **Needs**  | `expected`          |

## What it measures

The **F1** of an array's elements, treating the list like a set to recover: the array is
first [aligned](../core-concepts/array-alignment.md) (each actual element paired with an
expected one), then an aligned-and-correct element is a true positive, a missing expected
element is a false negative, and an extra element is a false positive. F1 is the harmonic
mean of the resulting precision and recall.

"Correct" comes from each element's own score (its `representative`), so `ArrayF1`
composes with whatever metric the items use. Elements share one config, so all count
equally (no per-element weights).

## Parameters

| Param       | Default  | Meaning                                                     |
|-------------|----------|-------------------------------------------------------------|
| `threshold` | `1.0`    | the score an aligned element must clear to count as correct |
| `mode`      | `"hard"` | `hard` = threshold gate; `soft` = add element scores fractionally |

(Alignment strategy/key are set on the field's `ArrayFieldConfig`, not the metric.)

## How it's computed

```text
align → matched pairs, missing (FN), spurious (FP)
matched element → TP if its score ≥ threshold (HARD)

precision = TP / (TP + FP)
recall = TP / (TP + FN)

score = 2 · precision · recall / (precision + recall)
```

Default **`threshold=1.0`, `mode="hard"`** — an element must be a perfect match to count;
`mode="soft"` adds fractional scores instead of gating.

## Example

Four expected tags, three produced — one of them wrong (`xxx`), one expected tag missing
(`ml`):

```python
from structured_eval import evaluate, EvalConfig, ArrayFieldConfig, ArrayF1

config = EvalConfig(fields={"tags": ArrayFieldConfig(metrics=[ArrayF1()])})
report = evaluate(
    {"tags": ["python", "data", "xxx"]},
    {"tags": ["python", "data", "web", "ml"]},
    config,
)

# aligned: python✓ data✓ (xxx↔web)✗ ; ml missing
# TP=2, FP=0, FN=2 → p=2/3=0.667, r=2/4=0.5
float(report.field_scores["tags"].metrics["array_f1"])   # 0.571
```

`report.array_matches["tags"]` complements this with the **structural** breakdown —
`matched` / `missed` / `spurious` (here 3 / 1 / 0) and the `strategy`. It's value-blind
(by_index aligned `xxx` to `web` by position); the `array_f1` *metric* is what adds value
correctness. Read the metric for quality, `array_matches` for what aligned with what.

## Edge cases

- **Strict default** — `hard` / `1.0`; use `soft` or a lower `threshold` for partial
  credit.
- **Count-based** — every element weighs 1.0 (elements share one config; no weights).
- **Empty arrays** — both empty → vacuously `1.0`.
- **Alignment matters** — which elements are "matched" depends on the strategy
  (`by_index` / `by_key` / `hungarian`); see
  [array alignment](../core-concepts/array-alignment.md).

## See also

- [`ArrayPrecision`](array-precision.md) / [`ArrayRecall`](array-recall.md) — the halves.
- [`ArrayPRF1`](array-prf1.md) — all three at once.
- [`ArrayAccuracy`](array-accuracy.md) — soft recall-style mean.
- [`ArrayCardinality`](array-cardinality.md) — count agreement only.
- [The metric catalog](../index.md) — all metrics and the return-shape model.
