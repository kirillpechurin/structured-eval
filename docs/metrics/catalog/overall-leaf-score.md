# OverallLeafScore

|            |                       |
|------------|-----------------------|
| **Class**  | `OverallLeafScore`    |
| **Key**    | `overall_leaf_score`  |
| **Branch** | root (whole document) |
| **Needs**  | `expected`            |

## What it measures

A single **headline accuracy** for the whole document: the weighted mean of every scalar
leaf's score, no matter how deeply nested. Each leaf contributes its own correctness (its
`representative`) times its `weight`. It answers "across all the fields that carry a value,
how right is the document on average?"

It is **leaf-flattened**: it pools every scalar leaf into one flat weighted average and
ignores the object/array nesting (unlike [`ObjectF1`](object-f1.md), which aggregates level
by level). A subtree with more leaves therefore weighs more in the total.

## Parameters

None — `OverallLeafScore()`. Per-leaf weight comes from each field's
`FieldConfig(weight=...)`; equal weights reduce to a plain mean.

## How it's computed

```text
for every scalar leaf in the tree:
    total_weight += leaf.weight
    weighted     += leaf.weight * leaf.representative   # leaf's own score

score = weighted / total_weight        # 1.0 if the document has no leaves
```

## Example

To make it the headline `report.score`, pass it as `key_metric` (it distributes to the
root). Here `title` is weighted 2×, and one nested leaf is wrong:

```python
from structured_eval import evaluate, EvalConfig, FieldConfig, ExactMatch
from structured_eval.metrics.overall_leaf_score import OverallLeafScore

config = EvalConfig(
    key_metric=OverallLeafScore(),
    fields={"title": FieldConfig(metrics=[ExactMatch()], weight=2.0)},
)
report = evaluate(
    {"title": "ML", "meta": {"level": "beginner", "hours": 40}},
    {"title": "ML", "meta": {"level": "advanced", "hours": 40}},
    config,
)

# leaves: title(w2, 1.0) + meta.hours(w1, 1.0) + meta.level(w1, 0.0)
# weighted = 2·1 + 1·1 + 1·0 = 3 ; total weight = 4
report.score                                              # 0.75
report.score_label                                        # "overall_leaf_score"
float(report.metrics["overall_leaf_score"].root())        # 0.75
```

If you only add it to `metrics=[...]` (not `key_metric`), it's still computed — read it via
`report.metrics["overall_leaf_score"]` — but `report.score` stays the default `mean_score`.

## Edge cases

- **Leaf-flattened, not hierarchical** — no per-object normalization; a deeper subtree
  contributes more total weight. For a balanced per-level view use the object metrics.
- **Array `missed` elements aren't counted** — an expected element that was never produced
  has no leaf node, so it doesn't appear here. Use [`CoverageLeafScore`](coverage-leaf-score.md)
  or the array metrics for completeness.
- **Missing scalar leaves score 0** — an expected scalar with no actual value gets a 0
  representative and drags the mean down.
- **No leaves** → vacuously `1.0`.

## See also

- [`MeanScore`](mean-score.md) — the default representative; what `report.score` shows
  unless you override `key_metric`.
- [`CoverageLeafScore`](coverage-leaf-score.md) — completeness (presence) instead of value.
- [`ObjectF1`](object-f1.md) — hierarchical, level-by-level aggregation.
- [The metric catalog](../index.md) — all metrics and the return-shape model.
