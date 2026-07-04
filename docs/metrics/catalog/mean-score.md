# MeanScore

|            |                              |
|------------|------------------------------|
| **Class**  | `MeanScore`                  |
| **Key**    | `mean_score`                 |
| **Branch** | any-node (runs on every node)|
| **Needs**  | nothing of its own           |

## What it measures

`MeanScore` is the **default `key_metric`** — the metric a node uses as its representative
score when you don't pick another. It's the arithmetic mean of the node's *own* already-computed
metrics. That's all it is.

The important idea is the role, not this metric: **any metric can be a node's
`key_metric`/representative.** A node's representative is the single number that bubbles up to
its parent's aggregation and, at the root, becomes `report.score`. You choose it with
`key_metric=...` (e.g. `ObjectF1()`, `OverallLeafScore()`); when you don't, the node falls
back to `MeanScore`.

## Parameters

None — `MeanScore()`. Metrics are averaged equally (weighting lives in the metrics that do it —
`OverallLeafScore`, the object metrics).

## How it's computed

```text
score = mean(node's own metric values, excluding mean_score itself)
        # 0.0 if the node has no other metric

# computed LAST (so the node's other metrics already have values)
# does NOT recurse into children — cross-child aggregation is the job of the
# node's own object/array metric (ObjectF1 / ArrayAccuracy / …)
```

## Example

The root carries two object metrics and uses `MeanScore` as its `key_metric`, so
`report.score` is their mean:

```python
from structured_eval import evaluate
from structured_eval.models import EvalConfig
from structured_eval.metrics import MeanScore, ObjectAccuracy, ObjectF1

config = EvalConfig(metrics=[ObjectF1(), ObjectAccuracy()], key_metric=MeanScore())
report = evaluate({"a": 1, "b": 9}, {"a": 1, "b": 2}, config)

report.score_label                                   # "mean_score"
report.score == (
    report.metrics["object_f1"].representative()
    + report.metrics["object_accuracy"].representative()
) / 2                                                # True
```

Pick a specific representative instead, and `MeanScore` steps aside:

```python
config = EvalConfig(metrics=[ObjectF1()], key_metric=ObjectF1())
evaluate({"a": 1, "b": 9}, {"a": 1, "b": 2}, config).score_label   # "object_f1"
```

## Edge cases

- **Computed last** — its value depends on the node's other metrics, so it always runs after
  them.
- **No other metric → `0.0`** — a node whose only metric opted out (returned `None`) still
  gets a representative; `0.0` is the honest floor (every node always has one).
- **No child recursion** — averaging a container's children is the job of that container's
  object/array metric, not `MeanScore`.

## See also

- [Representative score](../core-concepts/evaluation-model.md#the-representative-score-key_metric)
  — how `key_metric` works and why every node has one.
- [`OverallLeafScore`](overall-leaf-score.md) — a common explicit `key_metric` for the root.
- [The metric catalog](../index.md) — all metrics and the return-shape model.
