# ObjectF1

|            |                       |
|------------|-----------------------|
| **Class**  | `ObjectF1`            |
| **Key**    | `object_f1`           |
| **Branch** | object (`ObjectNode`) |
| **Needs**  | `expected`            |

## What it measures

The **F1** of an object's fields, treating extraction like slot filling: a field that's
present and correct is a true positive, a missing expected field is a false negative,
and an extra field is a false positive. F1 is the harmonic mean of the resulting
precision and recall — a single score that punishes both wrong/missing values and
spurious fields.

A field's "correct" verdict comes from the child's own score (its `representative`), so
`ObjectF1` composes with whatever metrics the fields use — including nested objects and
arrays, which contribute their own representative.

## Parameters

| Param          | Default          | Meaning                                                        |
|----------------|------------------|----------------------------------------------------------------|
| `score_policy` | `None`           | per-field `{name: metric}` override of the match criterion     |
| `threshold`    | `None`           | override the per-field pass threshold (single value or dict)   |
| `mode`         | `"hard"`         | `hard` = threshold gate; `soft` = use field scores fractionally |
| `weight_mode`  | `"proportional"` | weight fields by their configured `weight`; `none` = plain counts |

## How it's computed

```text
matched field → TP if its score clears its threshold (HARD), else not
missing field → FN     
extra field → FP

precision = TP / (TP + FP)     
recall = TP / (TP + FN)

score = 2 · precision · recall / (precision + recall)
```

By default **`mode="hard"` with a field threshold of `1.0`** — a field counts as a TP
only when its score is a perfect `1.0`. Lower a field's `threshold`, or use
`mode="soft"` (which adds each field's fractional score instead of gating), to give
partial credit.

## Example

`name` is right, `experience_years` is wrong, and `title` is missing entirely:

```python
from structured_eval import evaluate, EvalConfig, ObjectF1

config = EvalConfig(metrics=[ObjectF1()])
report = evaluate(
    {"name": "Sarah Johnson", "experience_years": 5},                 # title missing
    {"name": "Sarah Johnson", "experience_years": 8, "title": "PhD"},
    config,
)

# matched={name✓, experience✗}, missing={title}
# TP=1, FP=0, FN=1  →  p=1/2=0.5, r=1/3=0.333
float(report.metrics["object_f1"].root())   # 0.4
```

## Edge cases

- **Strict default** — `mode="hard"`, threshold `1.0`: a field needs a perfect score to
  count. Switch to `mode="soft"` or lower thresholds for partial credit.
- **All child kinds count** — a nested object/array contributes its representative just
  like a scalar field (not only scalars).
- **Empty object** — no expected and no actual fields → vacuously `1.0`.
- **Weights** — with non-uniform `weight` on fields, precision/recall are weighted; the
  default uniform weights reduce to plain counts.

## See also

- [`ObjectPrecision`](object-precision.md) / [`ObjectRecall`](object-recall.md) — the
  two halves on their own.
- [`ObjectPRF1`](object-prf1.md) — precision + recall + F1 in one metric.
- [`ObjectAccuracy`](object-accuracy.md) — soft recall-style mean (extra fields not
  penalized).
- [The metric catalog](../index.md) — all metrics and the return-shape model.
