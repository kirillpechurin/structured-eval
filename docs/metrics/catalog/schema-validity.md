# SchemaValidity

|            |                                          |
|------------|------------------------------------------|
| **Class**  | `SchemaValidity(schema)`                 |
| **Key**    | `schema_validity`                        |
| **Branch** | root (whole document)                    |
| **Needs**  | nothing (reads `actual`; no `expected`)  |

## What it measures

Does the document **validate against a schema** — `1.0` if it does, `0.0` if it doesn't.
A purely structural check (types, required fields, unexpected fields), with no opinion on
whether the *values* are correct. Use it as the L0–L3 gate before the value metrics weigh
in. It works in schema-only mode (`expected=None`), so you can check well-formedness even
without a ground-truth answer.

`schema` is either a **Pydantic model class** or a **JSON Schema dict**. Both are checked
the same way: a model is converted to its JSON Schema and validated with `jsonschema` (the
canonical validator, not a homegrown checker), so the verdict depends on the schema rather
than on how you expressed it.

That makes it **stricter than `model_validate`**, which coerces: `"100"` for a `float`
field is a type error here, not a valid `100.0`. Reporting a coerced value as valid would
hide exactly the failure this metric exists to catch — see [`TypeMatch`](type-match.md) for
the field-level version of the same question.

## Parameters

| Param    | Meaning                                                              |
|----------|---------------------------------------------------------------------|
| `schema` | a Pydantic `BaseModel` subclass **or** a JSON Schema `dict` (required) |

Needs the optional `jsonschema` extra (`pip install 'structured-eval[jsonschema]'`).

## How it's computed

```text
score = 1.0 if schema accepts actual else 0.0

failures are grouped into schema_errors = {
    type_errors,        # wrong-typed fields
    missing_required,   # required fields that are absent
    extra_fields,       # fields the schema forbids
}
```

Every entry names the field it is about, spelled the way `report.field_scores` spells
it — `lines[0].sku`, not the enclosing object.

## Example

A document with a wrong-typed `hours` and a missing `title`:

```python
from pydantic import BaseModel
from structured_eval import evaluate
from structured_eval.models import EvalConfig
from structured_eval.metrics.schema_validity import SchemaValidity

class Course(BaseModel):
    title: str
    hours: int

config = EvalConfig(key_metric=SchemaValidity(Course))
report = evaluate({"hours": "forty"}, None, config)      # expected=None: schema-only mode

report.score                                             # 0.0
report.score_label                                       # "schema_validity"

# why it failed — read the grouped detail off the root metric result:
report.metrics["schema_validity"].root().extra["schema_errors"]
# {'type_errors': ['hours'], 'missing_required': ['title'], 'extra_fields': []}
```

A JSON Schema dict works the same way:

```python
schema = {
    "type": "object",
    "properties": {"title": {"type": "string"}, "hours": {"type": "integer"}},
    "required": ["title"],
}
config = EvalConfig(key_metric=SchemaValidity(schema))
evaluate({"title": "ML", "hours": 40}, None, config).score   # 1.0
```

## Edge cases

- **Binary, not graded** — any single violation makes it `0.0`; it never reports "80% of
  fields valid". Read `schema_errors` to see *what* failed.
- **Structural only** — it checks shape, not values. Pair it with value metrics
  (`ExactMatch`, `ObjectF1`, …) for correctness.
- **Read detail via `.root().extra`** — `schema_errors` is a dict, so
  `extra_values("schema_errors")` would iterate its keys; use
  `report.metrics["schema_validity"].root().extra["schema_errors"]`.
- **Wrong `schema` type** → `TypeError`; the dict path without the `jsonschema` extra →
  `ImportError` with an install hint.

## See also

- [`CoverageLeafScore`](coverage-leaf-score.md) — presence of expected fields (completeness).
- [`OverallLeafScore`](overall-leaf-score.md) — value accuracy once the shape is valid.
- [`ObjectTypeValidity`](object-type-validity.md) — per-object type agreement vs `expected`.
- [The metric catalog](../index.md) — all metrics and the return-shape model.
