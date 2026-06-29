# ObjectExactMatch

|            |                      |
|------------|----------------------|
| **Class**  | `ObjectExactMatch`   |
| **Key**    | `object_exact_match` |
| **Branch** | object (`ObjectNode`)|
| **Needs**  | `expected`           |

## What it measures

Strict **deep equality** of a whole object: `1.0` if `actual` and `expected` have the
same keys with deep-equal values (recursively, nested dicts and lists included), else
`0.0`. No partial credit and no coercion — the object is either exactly right or wrong.
Use it for sub-objects that must match verbatim (a course's certificate block, a
schedule record) where the per-field partial credit of
[`ObjectAccuracy`](object-accuracy.md) / [`ObjectF1`](object-f1.md) would hide a single
wrong value.

## Parameters

None — `ObjectExactMatch()`.

## How it's computed

```text
equal  <=>  same key set  ∧  a[k] deep-equals e[k] for all k
score = 1.0 if equal else 0.0
```

Key order is irrelevant; nested lists are compared order-sensitively. Equality is
type-strict (`1` ≠ `1.0`).

## Example

```python
from structured_eval import (
    evaluate, EvalConfig, FieldConfig, ObjectFieldConfig, ObjectExactMatch,
)

config = EvalConfig(fields={
    "certificate": ObjectFieldConfig(
        fields={"title": FieldConfig(), "credits": FieldConfig()},
        metrics=[ObjectExactMatch()],
    ),
})

# one field off → the whole object is wrong
report = evaluate(
    {"certificate": {"title": "Python 101", "credits": 3}},
    {"certificate": {"title": "Python 101", "credits": 4}},
    config,
)
report.field_scores["certificate"].metrics["object_exact_match"]   # 0.0
```

## Edge cases

- **Both empty → `1.0`** — two empty objects are equal.
- **Key order irrelevant** — `{"a":1,"b":2}` equals `{"b":2,"a":1}`.
- **Missing/extra key → `0.0`** — the key sets must match exactly.
- **Non-dict side → `0.0`**; **type-strict** values.
- **Whole-object verdict** — this is *not* an aggregation over children; for per-field
  credit use [`ObjectAccuracy`](object-accuracy.md) / [`ObjectF1`](object-f1.md).

## See also

- [`ObjectAccuracy`](object-accuracy.md) / [`ObjectF1`](object-f1.md) — per-field partial credit.
- [`ArrayExactMatch`](array-exact-match.md) — the same idea for arrays.
- [The metric catalog](../index.md) — all metrics and the return-shape model.
