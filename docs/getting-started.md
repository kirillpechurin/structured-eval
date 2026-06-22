# Getting started

This page takes you from install to a real evaluation: run a model's structured
output against a reference, read the field-level report, and tune how each field
is compared.

## Install

```bash
pip install structured-eval
```

The core depends only on Pydantic. Optional features live behind extras — install
them all at once:

```bash
pip install "structured-eval[all]"
```

Each extra enables a specific capability, so you can install only what you need:

| Extra | Enables |
|-------|---------|
| `yaml` | parsing YAML inputs |
| `fuzzy` | the `Fuzzy` / `Levenshtein` metrics (rapidfuzz) |
| `jsonschema` | `SchemaValidity` against a JSON Schema |
| `rules` | the `Rule` DSL / `RulePassRate` |
| `diff` | the `structured_diff` utility (deepdiff) |
| `report` | richer console rendering |
| `align` | the `hungarian` array alignment strategy (scipy) |

The examples on this page use only core metrics, so plain `pip install
structured-eval` is enough.

## Your first evaluation

Say an LLM extracted a course record from a catalog page, and you have the
canonical record to check it against. The two are *structurally* identical —
same keys, same types, a nested `instructor` object — but several values differ
in small, realistic ways:

```python
from structured_eval import evaluate

expected = {
    "course_id": "COURSE-101",
    "title": "Introduction to Python",
    "level": "beginner",
    "duration_hours": 12,
    "instructor": {
        "name": "Sarah Johnson",
        "experience_years": 8,
        "specialization": "Data Engineering",
    },
}

actual = {
    "course_id": "COURSE-101",
    "title": "Intro to Python",        # paraphrased
    "level": "Beginner",               # different casing
    "duration_hours": 10,              # off by 2
    "instructor": {
        "name": "Sarah Johnson",
        "experience_years": 7,         # off by 1
        "specialization": "Data Eng",  # abbreviated
    },
}

report = evaluate(actual, expected)    # no config — sensible defaults
print(report.score)                    # 0.27
```

With no configuration, every field is compared by **exact match** and the
document's headline score is the mean of those verdicts. Here that is `0.27` —
and that low number is the point: the JSON is perfectly well-formed, yet exact
match punishes every paraphrase, every bit of casing, every rounded number as if
it were completely wrong.

## Reading the report

The report is always detailed. `field_scores` is keyed by path (`"$"` is the
root document, dots descend into nested objects):

```python
report.field_scores["course_id"].score                    # 1.0
report.field_scores["title"].score                         # 0.0  ← "Intro to Python"
report.field_scores["level"].score                         # 0.0  ← only the casing differs
report.field_scores["duration_hours"].score               # 0.0  ← 10 vs 12
report.field_scores["instructor"].score                    # 0.33 ← the nested object's own score
report.field_scores["instructor.name"].score              # 1.0
report.field_scores["instructor.experience_years"].score  # 0.0  ← 7 vs 8
report.field_scores["instructor.specialization"].score    # 0.0  ← "Data Eng"
```

The nested `instructor` object is evaluated in its own right and gets its own
representative score (`0.33` — one of its three fields matched), which then bubbles
up into the document score.

To jump straight to what fell below its pass threshold, use `failed_fields()`. It
returns a dict keyed by path:

```python
sorted(report.failed_fields())
# ['$', 'duration_hours', 'instructor', 'instructor.experience_years',
#  'instructor.specialization', 'level', 'title']
```

(`'$'` is the root — the whole document scored below a perfect 1.0.) For a
human-readable table of every field and metric, call:

```python
report.print_summary()
```

## Tuning the comparison

Exact match is the wrong tool for most of these fields: a title can be
paraphrased, a level differs only in casing, a duration is approximately right.
Assign each field the comparison that actually fits it via `EvalConfig`:

```python
from structured_eval import (
    EvalConfig,
    FieldConfig,
    ObjectFieldConfig,
    NormalizedMatch,
    NumericCloseness,
    TokenF1,
)

config = EvalConfig(
    fields={
        "title": FieldConfig(metrics=[TokenF1()]),               # word overlap
        "level": FieldConfig(metrics=[NormalizedMatch()]),       # case-insensitive
        "duration_hours": FieldConfig(metrics=[NumericCloseness()]),  # graded closeness
        "instructor": ObjectFieldConfig(
            fields={
                "experience_years": FieldConfig(metrics=[NumericCloseness()]),
                "specialization": FieldConfig(metrics=[TokenF1()]),
            },
        ),
    },
)

report = evaluate(actual, expected, config)
print(report.score)    # 0.86
```

The same errors, scored with the right metrics, now read as *mostly correct*
instead of mostly wrong — `0.27` → `0.86`. Each field tells a graded story:

```python
report.field_scores["title"].score                         # 0.67  ← shares "to python"
report.field_scores["level"].score                         # 1.0   ← casing normalized away
report.field_scores["duration_hours"].score               # 0.83  ← 10 is close to 12
report.field_scores["instructor.experience_years"].score  # 0.875 ← 7 is close to 8
report.field_scores["instructor.specialization"].score    # 0.5   ← "Data Eng" shares "data"
report.field_scores["instructor"].score                    # 0.79
```

`course_id` stays on exact match (it should — an id is right or it isn't), and so
the framework gives partial credit exactly where partial credit makes sense.

A couple of notes:

- `specialization` scores `0.5` because `TokenF1` compares whole words and
  `"Eng"` ≠ `"Engineering"`. For abbreviations and typos, `Fuzzy` (character-level,
  needs the `fuzzy` extra) gives smoother credit — see the [metric catalog](metrics/index.md).
- `report.score` comes from the field's **key metric** (by default, the mean of a
  node's metrics). You can pick a different headline metric — see
  [the evaluation model](core-concepts/evaluation-model.md).

## Next steps

- **[Core concepts](core-concepts/evaluation-model.md)** — how the evaluation tree,
  metrics, and scores fit together.
- **[Metrics](metrics/index.md)** — the full catalog and how to write your own.
