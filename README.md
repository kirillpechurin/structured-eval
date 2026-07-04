# structured-eval

**A declarative, field-level evaluation framework for LLM structured outputs (JSON/YAML).**

Getting an LLM to return well-formed JSON is mostly a solved problem — it parses,
it fits the schema, the types line up. But well-formed isn't the same as *right*.
The shape can be flawless while a price is wrong, a date is invented, or a status
quietly contradicts the rest of the record. Structural checks wave all of that
through.

structured-eval looks at what those checks skip: **the values themselves**. It
scores your output field by field, so you don't just learn *that* a response is
off — you see *which* fields matched and which didn't, where to look first, and,
across a dataset, which fields your model keeps getting wrong.

structured-eval lets you check not just that the JSON is valid, but that the
data itself is correct.

## The gap it closes

Correctness is a ladder — each level assumes the ones below it:

|           |                                                      |
|-----------|------------------------------------------------------|
| **L0–L3** | structure: parses · types · required · no extras     |
| **L4**    | values are close to expected                         |
| **L5**    | values are grounded in the source (no hallucination) |
| **L6**    | fields are logically consistent with one another     |

L0–L3 is where most tools stop. **L4–L6 is where structured-eval earns its keep.**
See [the introduction](docs/introduction.md) for the full ladder.

## Install

```bash
pip install structured-eval          # core depends only on Pydantic
pip install "structured-eval[all]"   # + YAML, fuzzy, schema, rules, scipy alignment…
```

Optional features live behind [extras](docs/getting-started.md#install) — install
only what you need.

## Quick start

A model extracted a course record; you have the canonical one to check it against.
The two are *structurally* identical — same keys, mixed types, a nested array of
objects — but several values are off. A small config says *how* to judge each
field, and the report tells you exactly where the output stands:

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
    ],  # "Classes" module missing
}

config = EvalConfig(fields={
    "title": FieldConfig(metrics=[TokenF1()]),                # reward paraphrases
    "duration_hours": FieldConfig(metrics=[Numeric(tolerance=2)]),
    "rating": FieldConfig(metrics=[Numeric(tolerance=0.5)]),  # close enough is fine
})

report = evaluate(actual, expected, config)

report.score  # 0.8889  — close, with the gaps pinpointed
report.field_scores["title"].score  # 0.6667  — paraphrase gets partial credit
report.field_scores["duration_hours"].score  # 1.0     — within tolerance
report.field_scores["modules"].score  # 0.6667  — 2 of 3 modules recovered
report.field_scores["modules[0]"].score  # 1.0     — first module is spot-on
```

Every field is scored — nested objects and array elements included — so you see
not a single pass/fail but exactly which fields hold up and which don't.

### Sensible default metrics

The config is optional. structured-eval ships a default metric for every node
type, so you only configure the fields where the default isn't what you want —
the rest just work. With no config at all, the same data is scored by those
defaults:

```python
report = evaluate(actual, expected)  # no config

report.score  # 0.4444  — scored entirely by the defaults
report.field_scores["title"].score  # 0.0  — exact match: "Intro to Python" ≠ "Introduction to Python"
```

Each node type gets a structural default, and every node's headline score (its
*representative*) defaults to the mean of its own metrics:

| Node                 | Default metric   | What it does                                   |
|----------------------|------------------|------------------------------------------------|
| scalar (leaf)        | `ExactMatch`     | the value must match exactly                   |
| object               | `ObjectAccuracy` | mean correctness of its fields                 |
| array                | `ArrayAccuracy`  | mean correctness of its aligned elements       |
| any node (headline)  | `MeanScore`      | the node's representative = mean of its metrics |

Exact match is a strict baseline — it punishes every paraphrase and rounded value
as wrong, which is why the no-config score is low. Tuning metrics per field, as in
the first example, is how you tell the evaluator what "close enough" means for
*your* data. The defaults and the representative score are covered in
[the evaluation model](docs/core-concepts/evaluation-model.md) and the
[metric catalog](docs/metrics/index.md).

## Explore — every level of correctness

structured-eval covers the whole ladder, L0 through L6. Each level has a tool and
a concept page behind it:

| Level               | The question                        | Reach for                                                   | Learn more                                                             |
|---------------------|-------------------------------------|-------------------------------------------------------------|------------------------------------------------------------------------|
| **L0–L3** structure | does it parse / fit the schema?     | `SchemaValidity`                                            | [schema validity](docs/metrics/catalog/schema-validity.md)             |
| **L4** values       | is each value right?                | field metrics — `ExactMatch`, `Numeric`, `TokenF1`, `Fuzzy` | [comparison is a metric](docs/core-concepts/comparison-is-a-metric.md) |
| **L4** roll-up      | how do fields & elements aggregate? | `ObjectF1` / `ArrayF1`, alignment, weights                  | [array alignment](docs/core-concepts/array-alignment.md)               |
| **L5** faithfulness | is it grounded in the source?       | `FieldFaithfulness(source=…)`                               | [field faithfulness](docs/metrics/catalog/field_faithfulness.md)       |
| **L6** logic        | are fields mutually consistent?     | `RulePassRate` + `Rule` DSL                                 | [rule pass rate](docs/metrics/catalog/rule-pass-rate.md)               |

**L0–L3 — structure.** Validate against a Pydantic model or JSON Schema, with no
ground-truth answer:

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

**L4 — values.** Pick *how* each field is judged — exact match is just the default
(*comparison is a metric*):

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
report.field_scores["title"].score  # 0.6667  — partial credit for a paraphrase
report.field_scores["duration_hours"].score  # 1.0     — within tolerance
```

Fields roll up into objects and arrays with precision / recall / F1, and arrays are
[aligned](docs/core-concepts/array-alignment.md) by index, key, or optimally:

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

**L5 — faithfulness.** Catch hallucinations by checking each value against its
source — no `expected` required:

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

**L6 — logic.** Assert cross-field business rules with a small DSL:

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

Evaluate a whole dataset, or measure how stable a prompt is across repeated runs:

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
report.unstable_fields  # ['sentiment']  — flipped on one run
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
