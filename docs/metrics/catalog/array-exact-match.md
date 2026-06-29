# ArrayExactMatch

|            |                     |
|------------|---------------------|
| **Class**  | `ArrayExactMatch`   |
| **Key**    | `array_exact_match` |
| **Branch** | array (`ArrayNode`) |
| **Needs**  | `expected`          |

## What it measures

Strict, **order-sensitive** equality of a whole list: `1.0` if the two arrays are
identical element-for-element (recursively, including nested dicts/lists), else `0.0`. No
alignment and no partial credit — the array is either exactly right or wrong. Use it when
element order is part of correctness (a course's ordered module list, a ranked syllabus)
and the aligned [`ArrayF1`](array-f1.md) family's partial credit would be misleading.

## Parameters

None — `ArrayExactMatch()`. (It does **not** align elements, so the field's `strategy` /
`params` are irrelevant to it.)

## How it's computed

```text
equal  <=>  same length  ∧  a[i] deep-equals e[i] for all i
score = 1.0 if equal else 0.0
```

Deep equality is type-strict: `1` ≠ `1.0`, and dicts must have the same keys with
deep-equal values.

## Example

```python
from structured_eval import evaluate, EvalConfig, ArrayFieldConfig, ArrayExactMatch

config = EvalConfig(fields={
    "modules": ArrayFieldConfig(metrics=[ArrayExactMatch()]),
})

# identical, in order → 1.0
report = evaluate(
    {"modules": ["Intro", "Variables", "Loops"]},
    {"modules": ["Intro", "Variables", "Loops"]},
    config,
)
report.field_scores["modules"].metrics["array_exact_match"]   # 1.0

# same modules, wrong order → 0.0
report = evaluate(
    {"modules": ["Variables", "Intro", "Loops"]},
    {"modules": ["Intro", "Variables", "Loops"]},
    config,
)
report.field_scores["modules"].metrics["array_exact_match"]   # 0.0
```

## Edge cases

- **Both empty → `1.0`** — two empty lists are equal.
- **Non-list side → `0.0`** — if either value isn't a list.
- **Type-strict** — `[1]` vs `[1.0]` scores `0.0`.
- **No alignment** — unlike the `Array*` P/R/F1 metrics, this ignores the field's
  alignment strategy entirely; it compares the raw lists.

## See also

- [`ArrayJaccardSimilarity`](array-jaccard-similarity.md) — order/count-insensitive set overlap.
- [`ArrayF1`](array-f1.md) / [`ArrayAccuracy`](array-accuracy.md) — aligned, value-aware, partial credit.
- [`ObjectExactMatch`](object-exact-match.md) — the same idea for objects.
- [The metric catalog](../index.md) — all metrics and the return-shape model.
