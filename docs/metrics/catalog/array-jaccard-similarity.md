# ArrayJaccardSimilarity

|            |                            |
|------------|----------------------------|
| **Class**  | `ArrayJaccardSimilarity`   |
| **Key**    | `array_jaccard_similarity` |
| **Branch** | array (`ArrayNode`)        |
| **Needs**  | `expected`                 |

## What it measures

Set-overlap (**Jaccard**) similarity between two lists treated as **sets** — order- and
count-insensitive. It answers "how much do these two collections share", ignoring
position and duplicates: ideal for unordered tag/label/category lists where
[`ArrayExactMatch`](array-exact-match.md) is too strict and the aligned
[`ArrayF1`](array-f1.md) family is more than you need.

## Parameters

None — `ArrayJaccardSimilarity()`. (No alignment, so the field's `strategy` / `params`
don't apply.)

## How it's computed

```text
score = |A ∩ B| / |A ∪ B|
```

where `A`, `B` are the two lists as sets. Membership is **exact equality**; duplicates
collapse and order is ignored. Object/list elements don't crash — they're keyed by their
canonical JSON — but the metric is designed for scalar elements.

## Example

```python
from structured_eval import evaluate, EvalConfig, ArrayFieldConfig, ArrayJaccardSimilarity

config = EvalConfig(fields={
    "tags": ArrayFieldConfig(metrics=[ArrayJaccardSimilarity()]),
})

# {a,b,c} vs {b,c,d}: ∩={b,c}, ∪={a,b,c,d} → 2/4 = 0.5
report = evaluate({"tags": ["a", "b", "c"]}, {"tags": ["b", "c", "d"]}, config)
report.field_scores["tags"].metrics["array_jaccard_similarity"]   # 0.5
```

## Edge cases

- **Order/duplicates ignored** — `["a","a","b"]` and `["b","a"]` score `1.0`.
- **Both empty → `1.0`**, **exactly one empty → `0.0`**, **disjoint → `0.0`**.
- **`None` → empty set** — a null list contributes nothing.
- **Exact membership** — there's no partial credit per element; near-miss values count as
  different. For value-aware element matching use [`ArrayF1`](array-f1.md) with a graded
  item metric.

## See also

- [`ArrayExactMatch`](array-exact-match.md) — strict, order-sensitive whole-list equality.
- [`ArrayF1`](array-f1.md) / [`ArrayAccuracy`](array-accuracy.md) — aligned, value-aware scoring.
- [The metric catalog](../index.md) — all metrics and the return-shape model.
