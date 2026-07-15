# The evaluation model

[Getting started](../getting-started.md) showed *what* `evaluate` does: pass an
`actual` and an `expected`, get a field-level report. This page is about *how* —
the machinery you tune when you assign metrics, read `report.score`, or wonder why
a nested object has a score of its own.

One idea holds it together: **`evaluate` is a pipeline of four phases.** The same
course-extraction example from getting-started runs through all of them:

```
┌─ INPUTS ───────┐       ┌─ BUILD TREE ──────────────────┐       ┌─ RUN METRICS ─┐       ┌─ REPORT ──────────────────────────┐
                                                                  (post-order)                                                
 actual ─┐                $ (object)                              ExactMatch, ...          report.score               | 0.27  
 expected├───parse ────▶  ├─ course_id  (scalar)           ────▶                   ────▶   field_scores["title"]      | 0.0   
 source ─┘                ├─ title       (scalar)                 per node                 field_scores["instructor"] | 0.33  
                          └─ instructor  (object)                 child first              metrics["mean_score"]      |  ...  
                             ├─ name             (scalar)         then parent                                                 
                             └─ experience_years (scalar)                                                                     
└────────────────┘       └───────────────────────────────┘       └───────────────┘       └───────────────────────────────────┘ 
 parse             ────▶  build the node tree              ────▶  run metrics      ────▶  flatten into a report
                          (+ assign each its metrics)             in post-order           
```

The rest of this page walks each phase.

## Inputs

`evaluate` takes **one document** plus an optional config. The document is given
through three arguments:

| Argument   | Required | Purpose                                                   |
|------------|----------|-----------------------------------------------------------|
| `actual`   | yes      | the output being evaluated                                |
| `expected` | no       | the reference; omit it for some metrics                   |
| `source`   | no       | the grounding text for some metrics, such as faithfulness |

Each of `actual` / `expected` can be a `dict`, a `list`, or a string. Strings are
parsed as JSON, falling back to YAML — see
[input forms](../evaluation/evaluate.md) for the details. A single document can
also be bundled as a `Sample` (that's what batch and consistency take a *list*
of — see [the evaluation functions](../evaluation/index.md)).

```python
from structured_eval import evaluate
from structured_eval.models import Sample

# the document can be given in several equivalent forms:
# Python dicts
evaluate({"total": 100}, {"total": 100})

# JSON (or YAML) strings
evaluate('{"total": 100}', '{"total": 100}')

# With source
evaluate(actual, expected, source="...")

# One document as a Sample. Batch and consistency evaluation take a list of Sample
evaluate(Sample(actual=actual, expected=expected))
```

The optional second argument is the `EvalConfig` — the declarative description of
*how* to score. With no config you get sensible defaults (every field compared by
exact match). The config mirrors the **shape of your data**: one config type per
node type.

```python
from structured_eval.models import EvalConfig, FieldConfig
from structured_eval.metrics import ExactMatch, MeanScore, ObjectF1, TokenF1

# how to score — these are the knobs you'll reach for most:
config = EvalConfig(
    metrics=[
      # metrics defined at the config root cascade to every applicable node
      ObjectF1(),
      ExactMatch(),
      ...
    ],
    # cascade to every node, by node type
    fields={
        # per-field overrides, keyed by path
        "title": FieldConfig(metrics=[TokenF1()]),
    },
    # the root's representative metric → report.score
    key_metric=MeanScore(),
)
```

| Data   | Config              | Key fields                                   |
|--------|---------------------|----------------------------------------------|
| scalar | `FieldConfig`       | `metrics`, `threshold`, `weight`             |
| object | `ObjectFieldConfig` | `fields={…}` (nested configs)                |
| array  | `ArrayFieldConfig`  | `item=…`, `strategy=…`                       |

> By default the document root is an object, and `EvalConfig(fields={...})`
> describes its fields directly. The `root=` parameter is only needed when the
> root is itself an array (or otherwise needs an explicit type). Weights,
> thresholds, and alignment strategies are covered in
> [the evaluation functions](../evaluation/index.md) and
> [array alignment](array-alignment.md).

## The node tree

Internally the document becomes a tree of `EvalNode`s mirroring its shape:

| Node         | Wraps                               | Children        |
|--------------|-------------------------------------|-----------------|
| `ScalarNode` | a leaf (string, number, bool, null) | none            |
| `ObjectNode` | an object                           | one per field   |
| `ArrayNode`  | an array                            | one per element |

Each node knows its own `actual`, `expected`, path, and metrics:

```
$ : ObjectNode                          metrics = [ObjectAccuracy]
├─ course_id  : ScalarNode              metrics = [ExactMatch]
├─ title      : ScalarNode              metrics = [TokenF1]
└─ instructor : ObjectNode              metrics = [ObjectAccuracy]
   ├─ name             : ScalarNode     metrics = [ExactMatch]
   └─ experience_years : ScalarNode     metrics = [NumericCloseness]
```

Although the model is a tree, **nodes are addressed by a flat string path** in
dot-and-bracket notation (this is `flatten`): 
- `$` is the root;
- dots descend into objects (`instructor.name`);
- brackets index arrays (`items[0].price`)

These same paths are the keys you read in the report.

## Comparison is a metric

The core design choice: there is **no separate "comparison" step.** A comparison
*is* a metric. A metric takes a node's `(actual, expected)` and returns a score in
`[0, 1]` — and a node *owns* a list of metrics, so it can carry several at once
(cheap and expensive), each scoring the very same pair:

```python
from structured_eval import evaluate
from structured_eval.models import EvalConfig, FieldConfig
from structured_eval.metrics import ExactMatch, TokenF1

# one field, two metrics scoring the same (actual, expected):
report = evaluate(
    {"title": "Intro to Python"},
    {"title": "Introduction to Python"},
    EvalConfig(fields={"title": FieldConfig(metrics=[ExactMatch(), TokenF1()])}),
)

report.field_scores["title"].metrics["exact_match"]  # 0.0  — the strings differ
report.field_scores["title"].metrics["token_f1"]  # 0.67 — they share "to python"
```

Where a node's metrics come from: the config's global `metrics` cascade by type,
any per-field `metrics` are merged in, and if a scalar is left with none it falls
back to its structural default (`ExactMatch` / `ObjectAccuracy` / `ArrayAccuracy`).

> So every node is guaranteed at least one metric of its own kind.

### The representative score (`key_metric`)

A node may carry many metrics but reports **one score per node** — its
*representative*. Which metric provides it is the node's `key_metric`. By default
that's `MeanScore`, the mean of the node's own metrics — but **any metric can be
the `key_metric`**, so you can pick one to speak for the node:

```python
from structured_eval import evaluate
from structured_eval.models import EvalConfig, FieldConfig
from structured_eval.metrics import ExactMatch, TokenF1

a, e = {"title": "Intro to Python"}, {"title": "Introduction to Python"}
metrics = [ExactMatch(), TokenF1()]   # exact_match = 0.0, token_f1 = 0.67

# default: representative = MeanScore, the mean of the node's metrics
r = evaluate(a, e, EvalConfig(fields={"title": FieldConfig(metrics=metrics)}))
r.field_scores["title"].score    # 0.33   — mean(0.0, 0.67)

# override: let one metric be the representative
r = evaluate(a, e, EvalConfig(
    fields={"title": FieldConfig(metrics=metrics, key_metric="token_f1")}))
r.field_scores["title"].score    # 0.67   — token_f1 speaks for the node
```

The same `key_metric` override is available on `ObjectFieldConfig` and
`ArrayFieldConfig`, so an object or array node can also elect one metric — e.g.
its per-object faithfulness verdict — to speak for it.

`report.score` is simply the root node's representative.

> Metrics run **post-order** — children before parents — so an aggregating parent
> metric (say `ObjectF1`) reads its children's already-computed representatives, and
> the node's own `key_metric` runs last. See
> [comparison is a metric](comparison-is-a-metric.md) for `key_metric` in depth.

## The report

The final phase flattens the computed tree into an `EvalReport` — the same model,
exposed as a flat, path-addressable view:

```python
report = evaluate(actual, expected)

report.score  # 0.27   — root's key_metric value
report.field_scores["instructor"].score  # 0.33   — the nested object's own representative
report.field_scores["title"].score  # 0.0    — one scalar's score
report.metrics["mean_score"].mean()  # one metric's value across all nodes
report.array_matches["items"]  # matched / missed / spurious + P/R/F1 (array nodes)
```

- `report.score` / `report.score_label` — the root's representative value and the
  metric's name.
- `report.field_scores[path]` — a `FieldScore` per node, keyed by the same flat
  paths: `.score`, `.metrics[name]`, `.actual` / `.expected`.
- `report.metrics[name]` — a `MetricCollection`: one metric's value across *all*
  nodes (`.by_path`, `.mean()`, `.root()`).
- Convenience: `report.print_summary()`, `report.failed_fields()`,
  `report.to_dict()` / `to_json()`.

## Next

- [The evaluation functions](../evaluation/index.md) — `evaluate`,
  `evaluate_batch`, `evaluate_consistency`, and when to use each.
- [Metric catalog](../metrics/index.md) — what metrics exist and how to write your own.
- [Comparison is a metric](comparison-is-a-metric.md) — `key_metric` and
  representatives in depth.
