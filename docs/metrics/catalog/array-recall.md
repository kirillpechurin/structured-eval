# ArrayRecall

|            |                     |
|------------|---------------------|
| **Class**  | `ArrayRecall`       |
| **Key**    | `array_recall`      |
| **Branch** | array (`ArrayNode`) |
| **Needs**  | `expected`          |

## What it measures

**Recall** over an array's elements: of the elements that *should* be there, how many
were produced correctly — `TP / (TP + FN)`. After
[alignment](../core-concepts/array-alignment.md), a missing or wrong expected element is
a FN. Answers "how much of what we expected did the model actually get right?"

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

score = TP / (TP + FN)        # aligned-and-correct over all expected elements
```

Default `threshold=1.0`, `mode="hard"`; `mode="soft"` counts fractional scores. With
`mode="soft"` this is exactly [`ArrayAccuracy`](array-accuracy.md).

## Example

Four expected, one wrong (`xxx`), one missing (`ml`) — one of four expected is correct...
plus `data`:

```python
from structured_eval import evaluate, EvalConfig, ArrayFieldConfig, ArrayRecall

config = EvalConfig(fields={"tags": ArrayFieldConfig(metrics=[ArrayRecall()])})
report = evaluate(
    {"tags": ["python", "data", "xxx"]},
    {"tags": ["python", "data", "web", "ml"]},
    config,
)

# expected 4 (python✓ data✓ web✗ ml missing) → TP=2 over 4 expected
float(report.field_scores["tags"].metrics["array_recall"])   # 0.5
```

## Edge cases

- **Extra elements don't lower recall** — they're FP (a [precision](array-precision.md)
  concern).
- **Strict default** — `hard` / `1.0`; `soft` gives partial credit (and equals
  [`ArrayAccuracy`](array-accuracy.md)).
- **No expected elements** → vacuously `1.0`.

## See also

- [`ArrayPrecision`](array-precision.md) — the complementary half.
- [`ArrayF1`](array-f1.md) — their harmonic mean.
- [`ArrayAccuracy`](array-accuracy.md) — the soft version of recall.
- [The metric catalog](../index.md) — all metrics and the return-shape model.
