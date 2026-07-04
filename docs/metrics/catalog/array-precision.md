# ArrayPrecision

|            |                     |
|------------|---------------------|
| **Class**  | `ArrayPrecision`    |
| **Key**    | `array_precision`   |
| **Branch** | array (`ArrayNode`) |
| **Needs**  | `expected`          |

## What it measures

**Precision** over an array's elements: of the elements the model produced, how many are
correct — `TP / (TP + FP)`. After [alignment](../core-concepts/array-alignment.md), an
aligned element is a TP when its score clears the threshold; an extra (spurious) element
is a FP. Answers "of what the model emitted, how much is right?"

Shares the [slot-filling mechanism](array-f1.md#how-its-computed) of `ArrayF1`,
count-based.

## Parameters

| Param       | Default  | Meaning                                                     |
|-------------|----------|-------------------------------------------------------------|
| `threshold` | `1.0`    | the score an aligned element must clear to count as correct |
| `mode`      | `"hard"` | `hard` = threshold gate; `soft` = add element scores fractionally |

## How it's computed

```text
align → matched pairs, missing (FN), spurious (FP)
matched element → TP if its score ≥ threshold (HARD)

score = TP / (TP + FP)        # aligned-and-correct over all produced elements
```

Default `threshold=1.0`, `mode="hard"` (perfect match required); `mode="soft"` counts
fractional scores. An array that produced no elements is vacuously `1.0`.

## Example

Three produced, one wrong (`xxx`); no extras, so precision is "2 of 3 produced":

```python
from structured_eval import evaluate
from structured_eval.models import ArrayFieldConfig, EvalConfig
from structured_eval.metrics import ArrayPrecision

config = EvalConfig(fields={"tags": ArrayFieldConfig(metrics=[ArrayPrecision()])})
report = evaluate(
    {"tags": ["python", "data", "xxx"]},
    {"tags": ["python", "data", "web", "ml"]},
    config,
)

# produced 3 (python✓ data✓ xxx✗), 0 spurious → TP=2 over 3 produced
float(report.field_scores["tags"].metrics["array_precision"])   # 0.667
```

## Edge cases

- **Missing elements don't lower precision** — they're FN (a [recall](array-recall.md)
  concern). The missing `ml` doesn't affect precision here.
- **Strict default** — `hard` / `1.0`; use `soft` for partial credit.
- **No produced elements** → vacuously `1.0`.

## See also

- [`ArrayRecall`](array-recall.md) — the complementary half.
- [`ArrayF1`](array-f1.md) — their harmonic mean.
- [`ArrayPRF1`](array-prf1.md) — all three at once.
- [The metric catalog](../index.md) — all metrics and the return-shape model.
