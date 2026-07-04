# Writing a custom metric

When the [catalog](index.md) doesn't have the metric you need, write your own. A
metric is a small class: give it a `name`, implement how it scores, and it plugs into
exactly the same machinery as the built-ins — usable by instance or by name, on any
node its branch fits.

## A field metric

The common case: a leaf comparison. Subclass `FieldMetric` and implement
`score(actual, expected)` returning a `float` in `[0, 1]`. Here a metric that treats
two course codes as equal when they share the prefix before the dash
(`COURSE-101` ≈ `COURSE-202`), which `ExactMatch` would score `0.0`:

```python
from typing import Any
from structured_eval.metrics.base import FieldMetric

class CodePrefixMatch(FieldMetric):
    name = "code_prefix_match"          # the key in report.metrics[...]
    
    # Override `score` to return only float value
    def score(self, actual: Any, expected: Any) -> float:
        head = lambda v: str(v).split("-")[0]
        return 1.0 if head(actual) == head(expected) else 0.0
```

Declaring the class **registers** it (via `name`), so you can use it either as an
instance or by that name string:

```python
from structured_eval import evaluate
from structured_eval.models import EvalConfig, FieldConfig

config = EvalConfig(fields={
    "course_id": FieldConfig(metrics=["code_prefix_match"]),   # by name…
    # "course_id": FieldConfig(metrics=[CodePrefixMatch()]),   # …or by instance
})
report = evaluate({"course_id": "COURSE-101"}, {"course_id": "COURSE-202"}, config)

float(report.field_scores["course_id"].metrics["code_prefix_match"])   # 1.0
```

`name` must be unique across all metrics — it's the registry key and the column in
the report.

## Returning structured detail

A score can carry findings alongside it. Return a `(value, extra)` tuple and the
engine puts `extra` on the result's `.extra` (the same channel `schema_validity` and
`faithfulness` use). To reach the node and attach detail, override `compute(node)`
instead of `score`:

```python
from structured_eval.metrics.base import FieldMetric, MetricOutput
from structured_eval.models.nodes.scalar import ScalarNode

class CodePrefixMatch(FieldMetric):
    name = "code_prefix_match"

    # Override `compute` to build a full MetricResult
    def compute(self, node: ScalarNode) -> MetricOutput:
        head = lambda v: str(v).split("-")[0]
        ok = head(node.actual) == head(node.expected)
        return (1.0 if ok else 0.0), {"compared_prefix": head(node.actual)}
```

```python
res = report.field_scores["course_id"].metrics["code_prefix_match"]
float(res)                       # 1.0
res.extra["compared_prefix"]     # 'COURSE'
```

A metric returns one of: a `float`, a `dict[str, float]` of sub-scores, a
`(value, extra)` tuple, a ready `MetricResult`, or `None` — the engine normalizes any
of them into a `MetricResult` (see [the catalog](index.md#what-a-metric-returns)).

## Choosing a branch

The branch you subclass decides **which nodes the metric runs on** — that's its job.
Pick by the shape you want to score:

| Subclass        | Runs on                          |
|-----------------|----------------------------------|
| `FieldMetric`   | scalar leaves                    |
| `ObjectMetric`  | objects                          |
| `ArrayMetric`   | arrays                           |
| `RootMetric`    | the root node only (`$`)         |
| `AnyNodeMetric` | every node, one uniform compute  |

*How* you score is then your choice between the two entry points every `Metric` has:

- `compute(node)` — sees the whole node, including its children. Aggregating metrics
  need this: an `ObjectMetric` walks `node.children`, an `ArrayMetric` walks
  `node.items`, and each child's representative is already computed (metrics run
  post-order), so `child.representative` is available.
- `score(actual, expected)` — a pure value comparison, with no node. Array alignment
  reuses it to compare candidate elements before any node exists.

`compute` defaults to delegating to `score`, so for a leaf comparison implementing
`score` alone is enough (as in [the field metric above](#a-field-metric)) — but
nothing stops you overriding `compute` instead, or both. An `AnyNodeMetric` runs the
*same* `compute` on every node regardless of kind — that's where `MeanScore` lives.

## A metric across node kinds

When one metric should behave differently per node kind, subclass `GenericMetric` and
implement only the `compute_<kind>` methods you need — `compute_scalar` /
`compute_object` / `compute_array`. The engine **dispatches by kind** and admits the
metric onto a node only where the matching method exists. Here, the fraction of a
node's immediate children that are non-null — meaningful for objects and arrays, but
not for a scalar (so no `compute_scalar`):

```python
from structured_eval.metrics.base import GenericMetric
from structured_eval.models.nodes.object_node import ObjectNode
from structured_eval.models.nodes.array_node import ArrayNode

class FilledRate(GenericMetric):
    name = "filled_rate"

    def compute_object(self, node: ObjectNode) -> float:
        kids = list(node.children.values())
        return sum(k.actual is not None for k in kids) / len(kids) if kids else 1.0

    def compute_array(self, node: ArrayNode) -> float:
        items = node.items
        return sum(k.actual is not None for k in items) / len(items) if items else 1.0
```

```python
from structured_eval import evaluate
from structured_eval.models import EvalConfig

report = evaluate(
    {"title": "X", "level": None, "tags": ["a", None]},
    {"title": "X", "level": "b",  "tags": ["a", "c"]},
    EvalConfig(metrics=[FilledRate()]),
)

float(report.field_scores["$"].metrics["filled_rate"])      # 0.667 — 2 of 3 keys filled
float(report.field_scores["tags"].metrics["filled_rate"])   # 0.5   — 1 of 2 elements filled
"filled_rate" in report.field_scores["title"].metrics       # False — no compute_scalar
```

## Notes

- **Don't call `compute` / `score` yourself** — the engine runs every metric through
  its invoker, post-order, and normalizes the result. You only implement the method;
  the framework calls it.
- **`name` is the contract** — unique, stable, and the key everywhere
  (`report.metrics[name]`, config-by-name, the summary table).
- Keep `score` a **pure value comparison** — array alignment reuses it to compare
  candidate elements before any node exists.

## See also

- [The metric catalog](index.md) — the built-ins and the return-shape model.
- [The evaluation model](../core-concepts/evaluation-model.md) — nodes, branches, and
  how a metric is run.
