# The metric catalog

A **metric** is a comparison: it looks at a node and returns a score in `[0, 1]`.
Nodes own their metrics, and "comparison is a metric" is the core design idea — see
[the evaluation model](../core-concepts/evaluation-model.md). This page is the map:
every built-in metric, what it measures, and what each one needs.

A metric's **key** is its `name` — the string under which its score lands in
`report.metrics[name]` and `report.field_scores[path].metrics[name]`. You select a
metric by passing the instance (`ExactMatch()`) or its name (`"exact_match"`) into a
config.

## How metrics attach to nodes

Each metric belongs to a **branch** of the hierarchy, and that branch decides which
nodes it can run on:

| Branch        | Runs on                                  | Examples                          |
|---------------|------------------------------------------|-----------------------------------|
| `field`       | scalar leaves (`ScalarNode`)             | `ExactMatch`, `TokenF1`, `Numeric`|
| `object`      | objects (`ObjectNode`)                   | `ObjectF1`, `ObjectAccuracy`      |
| `array`       | arrays (`ArrayNode`)                     | `ArrayF1`, `ArrayAccuracy`        |
| `root`        | the root node only (`$`)                 | `SchemaValidity`, `Faithfulness`  |
| `any-node`    | every node, one uniform computation      | `MeanScore`                       |
| `generic`     | several node kinds (per-kind dispatch)   | *(custom — see below)*            |

A metric in `EvalConfig(metrics=[...])` **cascades** to every node its branch fits;
a metric in a node's own `FieldConfig`/`ObjectFieldConfig`/`ArrayFieldConfig` is
*added* just to that node. When you configure none of a node's type, the engine
falls back to a default so every node always carries one metric:

| Node kind | Default metric    |
|-----------|-------------------|
| scalar    | `ExactMatch`      |
| object    | `ObjectAccuracy`  |
| array     | `ArrayAccuracy`   |
| key_metric (any node) | `MeanScore` |

> The **`generic` branch** has no built-ins — every shipped metric belongs to one of
> the others. It exists for custom metrics that span node kinds; see
> [custom metrics](custom-metric.md).

## Field metrics

Leaf comparisons of `actual` vs `expected` — all need `expected`, except
`Presence` (looks only at `actual`) and `FieldFaithfulness` (needs a `source`).

| Class             | Key                 | Measures                                                        |
|-------------------|---------------------|----------------------------------------------------------------|
| [`ExactMatch`](catalog/exact-match.md)         | `exact_match`       | exact equality (1.0 / 0.0); the default scalar metric          |
| [`RegexMatch`](catalog/regex-match.md) | `regex_match`  | string equality after optional lower/strip + a regex rewrite (string-only) |
| [`Numeric`](catalog/numeric.md)             | `numeric`           | numeric equality within a tolerance (strips currency/separators)|
| [`NumericCloseness`](catalog/numeric-closeness.md) | `numeric_closeness` | graded numeric closeness `1 − |a−e| / max(|a|,|e|)`            |
| [`TokenF1`](catalog/token-f1.md)             | `token_f1`          | token-overlap F1 — rewards partial phrase matches              |
| [`Fuzzy`](catalog/fuzzy.md)               | `fuzzy`             | fuzzy string similarity (rapidfuzz; configurable method)       |
| [`Levenshtein`](catalog/levenshtein.md)         | `levenshtein`       | edit-distance ratio (thin alias of `Fuzzy(RATIO)`)             |
| [`Presence`](catalog/presence.md)            | `presence`          | whether the field is present / non-null                        |
| [`TypeMatch`](catalog/type-match.md)           | `type_match`        | whether actual and expected share a JSON type                  |
| [`FieldFaithfulness`](catalog/field_faithfulness.md) | `field_faithfulness` | whether each value is grounded in the `source` text (needs `source`, not `expected`) |

## Object metrics

Aggregate over an object's children, weighting each by `child.weight`
(`WeightMode.PROPORTIONAL` by default; `NONE` for plain counts). All need `expected`.

| Class             | Key                  | Measures                                            |
|-------------------|----------------------|-----------------------------------------------------|
| [`ObjectAccuracy`](catalog/object-accuracy.md)    | `object_accuracy`    | fraction of children that pass; the default object metric |
| [`ObjectPrecision`](catalog/object-precision.md)   | `object_precision`   | of the keys present, how many are right             |
| [`ObjectRecall`](catalog/object-recall.md)      | `object_recall`      | of the expected keys, how many are right            |
| [`ObjectF1`](catalog/object-f1.md)          | `object_f1`          | harmonic mean of object precision & recall          |
| [`ObjectPRF1`](catalog/object-prf1.md)        | `object_prf1`        | precision + recall + F1 together (one metric, three keys) |
| [`ObjectTypeValidity`](catalog/object-type-validity.md) | `object_type_validity` | fraction of children with the expected JSON type (count-based) |

## Array metrics

Aggregate over aligned array elements (matched / missed / spurious — see
[array alignment](../core-concepts/array-alignment.md)). Count-based (elements share
one `item` config, so they carry no per-item weight). All need `expected`.

| Class             | Key                | Measures                                          |
|-------------------|--------------------|---------------------------------------------------|
| [`ArrayAccuracy`](catalog/array-accuracy.md)     | `array_accuracy`    | fraction of elements correctly matched; the default array metric |
| [`ArrayPrecision`](catalog/array-precision.md)    | `array_precision`   | of the produced elements, how many matched        |
| [`ArrayRecall`](catalog/array-recall.md)       | `array_recall`      | of the expected elements, how many were found     |
| [`ArrayF1`](catalog/array-f1.md)           | `array_f1`          | harmonic mean of array precision & recall         |
| [`ArrayPRF1`](catalog/array-prf1.md)         | `array_prf1`        | precision + recall + F1 together                  |
| [`ArrayCardinality`](catalog/array-cardinality.md)   | `array_cardinality` | how close the element count is, ignoring matching |

## Root metrics

Run once, on the whole document. They take their **own** input — a schema or a rule
set — rather than comparing against `expected`.

| Class             | Key                  | Needs        | Measures                                          |
|-------------------|----------------------|--------------|---------------------------------------------------|
| [`OverallLeafScore`](catalog/overall-leaf-score.md)   | `overall_leaf_score`  | `expected`   | weighted mean over every scalar leaf of the tree  |
| [`CoverageLeafScore`](catalog/coverage-leaf-score.md)  | `coverage_leaf_score` | `expected`   | fraction of expected leaves that are present      |
| [`SchemaValidity`](catalog/schema-validity.md)     | `schema_validity`     | a schema     | validity against a JSON Schema / pydantic model   |
| [`RulePassRate`](catalog/rule-pass-rate.md)       | `rule_pass_rate`      | a rule set   | fraction of business rules that hold              |

## Representative metric

| Class             | Key          | Branch     | Measures                                  |
|-------------------|--------------|------------|-------------------------------------------|
| [`MeanScore`](catalog/mean-score.md)         | `mean_score` | any-node   | mean of a node's *own* metrics (its default `key_metric`) |

`MeanScore` is the only built-in `any-node` metric: it runs on every node and is each
node's default representative — the single number that bubbles up to a parent's
aggregation and, at the root, to `report.score`. It is computed **last**, so it
averages the node's other already-computed metrics (without recursing into children).
See [representative score](../core-concepts/evaluation-model.md#the-representative-score-key_metric).

## What a metric returns

What a metric *returns* is variable — a bare `float`, a `dict` of sub-scores, a
`(value, extra)` tuple, a ready `MetricResult`, or `None`. The engine
(`MetricRunner`) normalizes any of these into a **`MetricResult`**: a `float` that
also carries structured detail on `.extra`. So whatever the metric produced, what you
*read back* is always a `MetricResult` — and because it *is* a float, it drops into
arithmetic and comparisons directly.

**A score.** The everyday case — `report.field_scores[path].metrics[name]` is a
`MetricResult`, usable as a number:

```python
from structured_eval import evaluate, EvalConfig, FieldConfig, TokenF1

# course title scored with TokenF1: "Intro to Python" vs "Introduction to Python"
report = evaluate(
    {"title": "Intro to Python"},
    {"title": "Introduction to Python"},
    EvalConfig(fields={"title": FieldConfig(metrics=[TokenF1()])}),
)

m = report.field_scores["title"].metrics["token_f1"]
type(m).__name__   # 'MetricResult'
float(m)           # 0.667 — it's a float; round(m, 3), m < 0.8, … all work
```

**A score with `.extra`.** The same float also carries structured detail — root
metrics use this for their findings (`schema_validity` → `"schema_errors"`,
`rule_pass_rate` → `"rule_results"`):

```python
from structured_eval import evaluate, EvalConfig, SchemaValidity

schema = {
    "type": "object",
    "properties": {"duration_hours": {"type": "number"}},
    "required": ["title", "duration_hours"],
}
report = evaluate(
    {"duration_hours": "oops"},                       # wrong type, and title missing
    config=EvalConfig(metrics=[SchemaValidity(schema)]),
)

res = report.metrics["schema_validity"].root()   # the root node's MetricResult
float(res)                    # 0.0
res.extra["schema_errors"]    # the finding rides on .extra
# {'type_errors': ['duration_hours'], 'missing_required': ['title'], 'extra_fields': []}
```

**Across all nodes.** `report.metrics[name]` is a `MetricCollection` — the same metric
seen on every node it ran on, ready to reduce:

```python
from structured_eval import evaluate, EvalConfig, FieldConfig, ObjectFieldConfig, TokenF1

# TokenF1 placed on three fields, so it lands on three nodes of the tree
config = EvalConfig(fields={
    "title": FieldConfig(metrics=[TokenF1()]),
    "instructor": ObjectFieldConfig(fields={
        "name": FieldConfig(metrics=[TokenF1()]),
        "bio": FieldConfig(metrics=[TokenF1()]),
    }),
})
report = evaluate(
    {"title": "Intro to Python", "instructor": {"name": "Sarah Johnson", "bio": "Data engineer"}},
    {"title": "Introduction to Python", "instructor": {"name": "Sarah Johnson", "bio": "Senior data engineer"}},
    config,
)

mc = report.metrics["token_f1"]
mc.by_path     # one entry per node the metric ran on:
               # {'instructor.bio': 0.8, 'instructor.name': 1.0, 'title': 0.667}
mc.mean()      # 0.822 — also .root() / .representative() / .min() / .max()
```

Each metric page documents which of these it produces.

## How to read a metric page

Every metric page follows the same shape, so you can scan to the part you need:

1. **Name & key** — the class and its `name` (the `report.metrics` key).
2. **Branch** — field / object / array / root / any-node / generic.
3. **What it measures** — one or two plain sentences.
4. **Parameters** — the constructor (e.g. `tolerance`, `mode`, `weight_mode`).
5. **How it's computed** — the formula or rule (P/R/F1, weights, thresholds).
6. **Example** — a minimal custom-domain snippet and what shows up in the report.
7. **Edge cases** — vacuous 1.0, nulls, missing `expected`, and the like.
8. **See also** — related metrics.

## Next

- [Custom metrics](custom-metric.md) — write your own comparison.
- [`ExactMatch`](catalog/exact-match.md) — the default scalar metric, a good first page.
- [The evaluation model](../core-concepts/evaluation-model.md) — how nodes, metrics,
  and the report fit together.
