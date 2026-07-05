# structured-eval

[![CI](https://github.com/kirillpechurin/structured-eval/actions/workflows/ci.yml/badge.svg)](https://github.com/kirillpechurin/structured-eval/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/kirillpechurin/structured-eval/graph/badge.svg)](https://codecov.io/gh/kirillpechurin/structured-eval)
[![PyPI](https://img.shields.io/pypi/v/structured-eval.svg)](https://pypi.org/project/structured-eval/)
[![Python versions](https://img.shields.io/pypi/pyversions/structured-eval.svg)](https://pypi.org/project/structured-eval/)
[![License](https://img.shields.io/pypi/l/structured-eval.svg)](https://github.com/kirillpechurin/structured-eval/blob/main/LICENSE)

**The LLM Structured Output Evaluation Framework**

When interacting with an LLM, returning structured data is a common task.
And in any such integration, the quality of the returned data deserves attention —
it should be evaluated with the appropriate tools.
However, quality evaluation of structured data is often limited to the following:

- Parses correctly
- Data types are correct
- Required fields are present and there are no extra fields

That is not enough to be fully confident the data is correct, because these
mechanisms target structural validation, not the values themselves.

**structured-eval** focuses on evaluating the quality of an LLM's structured output,
such as JSON or YAML. The library makes it possible to compare
values, objects, and arrays at every level of nesting, which lets you check

- Which fields matched or are close to the expected ones, and which are not
- Whether values are grounded in a source
- Whether fields are logically consistent with one another

This kind of field-level evaluation gives you a systematic way to target improvements.
A few examples of what you can learn using structured-eval:

- How close an output value is to the expected one
- Which scalars, objects, and arrays have problems most often
- At dataset scale — how stable the LLM is

## What it checks

Correctness is a set of checks, where each check builds on the ones below it:

| Level | Check                                     | How you can check it         |
|-------|-------------------------------------------|------------------------------|
| L0    | Parses correctly                          | structured-eval / jsonschema |
| L1    | Data types are correct                    | structured-eval / jsonschema |
| L2    | Required fields are present               | structured-eval / jsonschema |
| L3    | No extra fields                           | structured-eval / jsonschema |
| L4    | Values match or are close to the expected | structured-eval              |
| L5    | Values are grounded in a source           | structured-eval              |
| L6    | Fields are logically consistent           | structured-eval              |

**The value of structured-eval is to provide a complete quality evaluation of
structured data.**

## Install

```bash
pip install structured-eval          # core depends only on Pydantic
pip install "structured-eval[all]"   # + YAML, fuzzy, schema, rules, scipy alignment…
```

By default only `pydantic` is required, but to extend functionality there are
[extras](docs/getting-started.md#install) — install only what you need.

## Quick start

The example uses a course record. The expected and the actual are structurally
identical, but several values differ. A small evaluation config says *how* to
judge each field, and the report shows where the data diverges:

```python
from structured_eval import evaluate
from structured_eval.models import EvalConfig, FieldConfig
from structured_eval.metrics import Numeric, TokenF1

expected = {
    "course_id": "COURSE-101",
    "title": "Introduction to Python",
    "published": True,
    "duration_hours": 12,
    "rating": 4.8,
    "modules": [
        {"name": "Basics", "lessons": 5},
        {"name": "Functions", "lessons": 4},
        {"name": "Classes", "lessons": 3},
    ],
}

actual = {
    "course_id": "COURSE-101",
    "title": "Intro to Python",  # paraphrased
    "published": True,
    "duration_hours": 10,  # off by 2
    "rating": 4.5,  # off by 0.3
    "modules": [
        {"name": "Basics", "lessons": 5},
        {"name": "Functions", "lessons": 4},
    ],  # the "Classes" module is missing
}

config = EvalConfig(fields={
    "title": FieldConfig(metrics=[TokenF1()]),  # reward paraphrases
    "duration_hours": FieldConfig(metrics=[Numeric(tolerance=2)]),
    "rating": FieldConfig(metrics=[Numeric(tolerance=0.5)]),  # close enough is fine
})

report = evaluate(actual, expected, config)

report.score  # 0.8889 — close, with the gaps pinpointed
report.field_scores["title"].score  # 0.6667 — a paraphrase gets partial credit
report.field_scores["duration_hours"].score  # 1.0 — within tolerance
report.field_scores["modules"].score  # 0.6667 — 2 of 3 modules recovered
report.field_scores["modules[0]"].score  # 1.0 — the first module is spot-on
```

You can assign several metrics to a single field and make the field's
representative (`key_metric`) — the score that lands in the total — a separate
aggregating metric over them. For example, `CompositeScore` blends metrics with
given weights, while `MeanScore` takes their plain mean; metrics not included in
the representative are still computed alongside, for detail:

```python
from structured_eval import evaluate
from structured_eval.models import EvalConfig, FieldConfig
from structured_eval.metrics import (
    CompositeScore, ExactMatch, Fuzzy, MeanScore, Numeric, NumericCloseness, TokenF1,
)

config = EvalConfig(fields={
    # the field's representative — a weighted blend of token_f1 and fuzzy (exact_match is for detail only)
    "title": FieldConfig(
        metrics=[ExactMatch(), TokenF1(), Fuzzy()],
        key_metric=CompositeScore(weights={"exact_match": 0.1, "token_f1": 0.6, "fuzzy": 0.3}),
    ),
    # the field's representative — the plain mean of two numeric metrics
    "rating": FieldConfig(
        metrics=[Numeric(tolerance=0.5), NumericCloseness()],
        key_metric=MeanScore(),
    ),
})

report = evaluate(
    {"title": "Intro to Python", "rating": 4.5},
    {"title": "Introduction to Python", "rating": 4.8},
    config,
)

report.field_scores["title"].score  # 0.6432  — CompositeScore: 0.1·exact_match + 0.6·token_f1 + 0.3·fuzzy
report.field_scores["title"].metrics["exact_match"]  # 0.0 — computed alongside, for detail
report.field_scores["rating"].score  # 0.96875 — MeanScore: mean of numeric and numeric_closeness
```

This is the key idea: *comparison is a metric*, not a separate "matcher" with a
pre-computed similarity. Learn more —
[comparison is a metric](docs/core-concepts/comparison-is-a-metric.md).

Every field is scored — nested objects and array elements of any depth included —
so you see not a single pass/fail, but exactly which fields match and which
don't.

### Sensible default metrics

**structured-eval** ships a default metric for every node type, so you only
configure the fields where the default isn't what you want. With no config at
all, the same data is scored by these default metrics:

```python
report = evaluate(actual, expected)  # no config

report.score  # 0.4444 — scored entirely by the defaults
report.field_scores["title"].score  # 0.0  — exact match: "Intro to Python" ≠ "Introduction to Python"
```

Each node type gets a structural default, and every node's headline score (its
*representative*) defaults to the mean of its own metrics:

| Node                | Default metric   | What it does                                    |
|---------------------|------------------|-------------------------------------------------|
| scalar (leaf)       | `ExactMatch`     | the value must match exactly                    |
| object              | `ObjectAccuracy` | mean correctness of its fields                  |
| array               | `ArrayAccuracy`  | mean correctness of its aligned elements        |
| any node (headline) | `MeanScore`      | the node's representative = mean of its metrics |

Exact match is a strict baseline. Tuning metrics per field, as in the first
example, is how you tell the evaluator what "close enough" means for *your* data.
The defaults and the representative score are covered in
[the evaluation model](docs/core-concepts/evaluation-model.md) and the
[metric catalog](docs/metrics/index.md).

## Explore — every level of correctness

structured-eval covers the whole ladder, L0 through L6. Behind each level there
is a tool and a concept page:

### Structural checks

Levels L0–L3 let you check successful parsing, data types, required fields, and
the absence of extra fields.

Validate against a Pydantic model or JSON Schema, with no ground-truth answer:

```python
from pydantic import BaseModel
from structured_eval import evaluate
from structured_eval.models import EvalConfig
from structured_eval.metrics import SchemaValidity


class Course(BaseModel):
    title: str
    duration_hours: int


report = evaluate(
    actual={"title": "ML", "duration_hours": "twelve"},
    expected=None,
    config=EvalConfig(key_metric=SchemaValidity(Course))
)
report.score  # 0.0
report.metrics["schema_validity"].root().extra["schema_errors"]
#   {'type_errors': ['duration_hours'], 'missing_required': [], 'extra_fields': []}
```

### Value correctness

Level L4 lets you choose how flexibly to judge values,
on your own criteria — for leaves as well as objects and arrays.

**structured-eval** provides a large set of metrics, see more in the
[metric catalog](docs/metrics/index.md).

```python
from structured_eval import evaluate
from structured_eval.models import EvalConfig, FieldConfig
from structured_eval.metrics import Numeric, TokenF1

report = evaluate(
    actual={"title": "Intro to Python", "duration_hours": 11},
    expected={"title": "Introduction to Python", "duration_hours": 12},
    config=EvalConfig(fields={
        "title": FieldConfig(metrics=[TokenF1()]),  # token overlap
        "duration_hours": FieldConfig(metrics=[Numeric(tolerance=2)]),
    }),
)
report.field_scores["title"].score  # 0.6667 — partial credit for a paraphrase
report.field_scores["duration_hours"].score  # 1.0 — within tolerance
```

Leaf scores can be rolled up into a metric on the object or array. The base
metrics for such structures are precision, recall, F1 — see more in the full
[metric catalog](docs/metrics/index.md).

```python
from structured_eval import evaluate
from structured_eval.models import EvalConfig
from structured_eval.metrics import ObjectF1

report = evaluate(
    actual={"a": 1, "b": 9},
    expected={"a": 1, "b": 2, "c": 3},
    config=EvalConfig(metrics=[ObjectF1()])
)

report.metrics["object_f1"].root()  # 0.4
```

For an array of objects the elements are first *aligned* (which actual element
corresponds to which expected one), and then each pair is evaluated field by
field. You set the strategy on `ArrayFieldConfig` — for example, `by_key` matches
elements by a key, and array order stops mattering:

```python
from structured_eval import evaluate
from structured_eval.models import ArrayFieldConfig, ArrayStrategy, EvalConfig
from structured_eval.metrics import ArrayF1

config = EvalConfig(fields={"items": ArrayFieldConfig(
    strategy=ArrayStrategy.BY_KEY,  # match elements by the sku key
    params={"key": "sku"},
    metrics=[ArrayF1()],
)})

report = evaluate(
    {"items": [{"sku": "B", "qty": 5}, {"sku": "A", "qty": 2}]},
    {"items": [{"sku": "A", "qty": 2}, {"sku": "B", "qty": 3}]},
    config,
)

report.array_matches["items"].matched  # [(0, 1), (1, 0)] — A↔A, B↔B despite the order
report.field_scores["items"].score  # 0.5  — A matched (qty 2), B didn't (qty 5 ≠ 3)
```

Learn more about strategies in
[array alignment](docs/core-concepts/array-alignment.md).

### Source grounding

Level L5 lets you catch hallucinations by checking each value against a source.
Note that `expected` is not required for the computation.

Learn more — [field faithfulness](docs/metrics/catalog/field_faithfulness.md).

```python
from structured_eval import evaluate
from structured_eval.models import EvalConfig
from structured_eval.metrics import FieldFaithfulness

report = evaluate(
    actual={"title": "Introduction to Python", "duration_hours": 40},
    expected=None,
    config=EvalConfig(metrics=[FieldFaithfulness()]),
    source="Course: Introduction to Python. Duration: 12 hours.",
)

report.metrics["field_faithfulness"].by_path  # {'title': 1.0, 'duration_hours': 0.0 ← 40 ≠ 12}
```

### Logical consistency of values

Level L6 offers an interface for describing cross-field business rules with a
small DSL.

Learn more — [rule pass rate](docs/metrics/catalog/rule-pass-rate.md).

```python
from structured_eval import evaluate
from structured_eval.models import EvalConfig
from structured_eval.metrics import Rule, RulePassRate

report = evaluate(
    actual={"subtotal": 100, "tax": 20, "total": 130},
    expected=None,
    config=EvalConfig(key_metric=RulePassRate([Rule("$.total").eq("$.subtotal + $.tax")]))
)

report.score  # 0.0  — 130 ≠ 120
```

## Scale it

**structured-eval** offers ways to evaluate

- A whole dataset
- Prompt stability across repeated runs

```python
from structured_eval import evaluate_batch, evaluate_consistency
from structured_eval.models import Sample

# one report per sample + dataset aggregates
batch = evaluate_batch([
    Sample(actual={"x": 1}, expected={"x": 1}),
    Sample(actual={"x": 1}, expected={"x": 2}),
])
batch.score  # 0.5
batch.perfect_response_rate  # 0.5

# repeated runs of the same prompt → which fields drift?
runs = [
    Sample(actual={"sentiment": "positive", "score": 0.9}, expected={"sentiment": "positive", "score": 0.9}),
    Sample(actual={"sentiment": "positive", "score": 0.9}, expected={"sentiment": "positive", "score": 0.9}),
    Sample(actual={"sentiment": "neutral", "score": 0.9}, expected={"sentiment": "positive", "score": 0.9}),
]
report = evaluate_consistency(runs, variance_threshold=0.05)
report.stable_fields  # ['score']
report.unstable_fields  # ['sentiment'] — flipped on one run
```

## Documentation

- **[Introduction](docs/introduction.md)** — the L0–L6 ladder and why values matter.
- **[Getting started](docs/getting-started.md)** — install → first evaluation →
  reading and tuning the report.
- **Core concepts** — [the evaluation model](docs/core-concepts/evaluation-model.md) ·
  [comparison is a metric](docs/core-concepts/comparison-is-a-metric.md) ·
  [array alignment](docs/core-concepts/array-alignment.md)
- **[Evaluation functions](docs/evaluation/index.md)** — `evaluate`,
  `evaluate_batch`, `evaluate_consistency`.
- **[Metric catalog](docs/metrics/index.md)** — every metric, plus how to write your own.

## License

Apache-2.0 — see [LICENSE](LICENSE).
