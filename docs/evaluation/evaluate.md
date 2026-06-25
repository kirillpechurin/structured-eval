# evaluate

`evaluate` scores **one document** and returns an [`EvalReport`](#the-report).
It's the function behind getting-started and the one you'll reach for most.

The signature — only `actual` is required:

```python
from structured_eval import evaluate

evaluate(actual, expected=None, config=None, *, source=None)
```

## Input forms

`actual` and `expected` can each be a `dict`, a `list`, or a string, and there are
two call shapes — loose arguments or a single `Sample`.

The everyday shape — `actual` and `expected` as Python values:

```python
from structured_eval import evaluate

report = evaluate({"total": 100}, {"total": 99})
report.score   # 0.0 — exact match, the values differ
```

Strings are accepted too; they're parsed as JSON, falling back to YAML:

```python
from structured_eval import evaluate

report = evaluate('{"total": 100}', '{"total": 100}')
report.score   # 1.0
```

A `Sample` bundles `actual` / `expected` / `source` / `id` into one object — handy
when you carry an identifier or build a list for batch evaluation:

```python
from structured_eval import evaluate, Sample

sample = Sample(actual={"total": 100}, expected={"total": 100}, id="invoice-1")
report = evaluate(sample)
report.score   # 1.0
```

A bare `list` is one document whose **root is an array** — not a batch. `evaluate`
scores it as a single document; to score many, pass `list[Sample]` to
[`evaluate_batch`](evaluate-batch.md):

```python
from structured_eval import evaluate

# a list is one array-root document, NOT a batch of two
actual = [{"id": "A", "qty": 2}, {"id": "B", "qty": 5}]
expected = [{"id": "A", "qty": 2}, {"id": "B", "qty": 4}]
report = evaluate(actual, expected)
```

> Full element-level scoring of an **array-root** document is still on the
> roadmap; today, evaluate arrays as a field of an object (see
> [array alignment](../core-concepts/array-alignment.md)).

## `expected` and `source`

Most metrics compare `actual` against `expected`, so you pass both. But `expected`
is optional: some metrics don't need a reference, and a few need a `source`
instead. You choose which metrics run through the `config`.

Without `expected`, `SchemaValidity` checks the structure against a JSON Schema —
the offending paths come back on the metric's `.extra`, grouped by kind of error:

```python
from structured_eval import evaluate, EvalConfig, SchemaValidity

schema = {
    "type": "object",
    "properties": {"total": {"type": "number"}},
    "required": ["total"],
}
report = evaluate({"total": "oops"}, config=EvalConfig(metrics=[SchemaValidity(schema)]))
report.score   # 0.0
report.metrics["schema_validity"].root().extra["schema_errors"]
# {'type_errors': ['total'], 'missing_required': [], 'extra_fields': []}
```

A `source` enables `Faithfulness` — it flags values that aren't grounded in the
original text:

```python
from structured_eval import evaluate, EvalConfig, Faithfulness

report = evaluate(
    {"name": "Sarah", "city": "Berlin"},
    config=EvalConfig(metrics=[Faithfulness()]),
    source="Sarah lives in Munich.",
)
report.metrics["faithfulness"].extra_values("hallucinated_fields")   # ['city'] — Berlin isn't in the source
```

`Faithfulness` **requires** `source` and raises `ValueError` without one. See the
[metric catalog](../metrics/index.md) for which metrics need what.

## The report

`evaluate` always returns a detailed `EvalReport` that exposes the computed tree as
a flat, path-addressable view (see
[the evaluation model](../core-concepts/evaluation-model.md)). The snippets below
all run against this one course-extraction example:

```python
from structured_eval import (
    evaluate, EvalConfig, FieldConfig, ObjectFieldConfig,
    RegexMatch, NumericCloseness, TokenF1, ObjectF1,
)

expected = {
    "course_id": "COURSE-101",
    "title": "Introduction to Python",
    "level": "beginner",
    "duration_hours": 12,
    "instructor": {"name": "Sarah Johnson", "experience_years": 8},
}
actual = {
    "course_id": "COURSE-101",
    "title": "Intro to Python",     # paraphrased
    "level": "Beginner",            # different casing
    "duration_hours": 10,           # off by 2
    "instructor": {"name": "Sarah Johnson", "experience_years": 7},  # off by 1
}
config = EvalConfig(fields={
    "title": FieldConfig(metrics=[TokenF1()]),
    "level": FieldConfig(metrics=[RegexMatch()]),
    "duration_hours": FieldConfig(metrics=[NumericCloseness()]),
    "instructor": ObjectFieldConfig(
        metrics=[ObjectF1()],
        fields={"experience_years": FieldConfig(metrics=[NumericCloseness()])},
    ),
})

report = evaluate(actual, expected, config)
```

The headline score is the root node's representative (here the mean across fields):

```python
report.score          # 0.8
report.score_label    # 'mean_score'
```

`field_scores` holds one `FieldScore` per node, keyed by flat path (`"$"` is the
root). Walk it to see every node — parents and leaves alike — with its score:

```python
for path, fs in report.field_scores.items():
    print(f"{path:<28} {round(fs.score, 3)}")

# the whole tree, flattened to one row per node:
# $                            0.8     ← the object root (object_accuracy)
# course_id                    1.0
# duration_hours               0.833
# instructor                   0.5     ← the nested object's own score
# instructor.experience_years  0.875
# instructor.name              1.0
# level                        1.0
# title                        0.667
```

A single `FieldScore` exposes the node's metrics and the compared values:

```python
fs = report.field_scores["title"]

print(fs.score)
# 0.667

print(list(fs.metrics.keys()))
# ['token_f1', 'mean_score']

print(float(fs.metrics["token_f1"]))
# 0.667 — a float carrying .extra

print(fs.actual, "|", fs.expected)
# Intro to Python | Introduction to Python
```

`report.metrics[name]` is the same metric seen across **all** nodes — a
`MetricCollection` you can reduce:

```python
mean_score = report.metrics["mean_score"]

print(round(mean_score.mean(), 3))   
# 0.834 — averaged over every node

print(mean_score.by_path)
# {'$': 0.8, 'course_id': 1.0, 'title': 0.667, ...}
```

`failed_fields()` returns just the nodes below their pass threshold (keyed by
path), so you can iterate over what actually went wrong:

```python
for path, fs in report.failed_fields().items():
    print(f"{path:<28} {round(fs.score, 3)} (threshold {fs.threshold})")

# only the fields that missed their threshold:
# $                            0.8 (threshold 1.0)
# duration_hours               0.833 (threshold 1.0)
# instructor                   0.5 (threshold 1.0)
# instructor.experience_years  0.875 (threshold 1.0)
# title                        0.667 (threshold 1.0)
```

`print_summary()` renders the whole thing as a table:

```python
report.print_summary()
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#   OVERALL   0.80   ✗ FAIL        mean_score
#
#   Field                        Metric             Score  Threshold  Mark
#   $                            object_accuracy     0.80       1.00   ✗
#   course_id                    exact_match         1.00       1.00   ✓
#   duration_hours               numeric_closeness   0.83       1.00   ✗
#   instructor                   object_f1           0.50       1.00   ✗
#   instructor.experience_years  numeric_closeness   0.88       1.00   ✗
#   instructor.name              exact_match         1.00       1.00   ✓
#   level                        regex_match         1.00       1.00   ✓
#   title                        token_f1            0.67       1.00   ✗
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

For array nodes the report also fills `report.array_matches[path]` (matched /
missed / spurious + P/R/F1) — see [array alignment](../core-concepts/array-alignment.md).

A parse failure is **not** an exception — instead the report records it and leaves
the metrics empty:

```python
report = evaluate("{not valid json", {"total": 100})
report.parse_error            # True
report.parse_error_message    # the parser's message
report.score                  # None
```

### Assertions for tests / CI

Each `assert_*` raises `AssertionError` on failure, so a report drops straight into
a test:

```python
report.assert_no_parse_errors()
report.assert_score(0.8)             # ok: 0.8 >= 0.8
report.assert_field("course_id", 1.0)
report.assert_metric("object_accuracy", 0.5)
```

`to_dict()` returns a plain, JSON-serializable structure (and `to_json(path)`
writes the same to a file) — useful for storing a report and comparing runs later
(e.g. regression with `diff_from`):

```python
data = report.to_dict()

# top-level shape mirrors the report fields:
# {
#   "score": 0.8,
#   "score_label": "mean_score",
#   "metrics": {...},                       # per-metric, across nodes
#   "field_scores": {
#       "title": {
#           "path": "title",
#           "node_type": "scalar",
#           "actual": "Intro to Python",
#           "expected": "Introduction to Python",
#           "metrics": {"token_f1": 0.667, "mean_score": 0.667},
#           "score": 0.667,
#           "threshold": 1.0,
#       },
#       ...
#   },
#   "array_matches": {...},
#   "parse_error": False,
#   "parse_error_message": None,
#   "warnings": [...],
# }

report.to_json("report.json")   # same structure, written to disk
```

## Next

- [`evaluate_batch`](evaluate-batch.md) — score a dataset and read aggregates.
- [`evaluate_consistency`](evaluate-consistency.md) — run-to-run stability.
- [Metric catalog](../metrics/index.md) — which metrics need `expected` / `source`.
- [The evaluation model](../core-concepts/evaluation-model.md) — the pipeline and
  the report's shape.
