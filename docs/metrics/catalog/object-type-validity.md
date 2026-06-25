# ObjectTypeValidity

|            |                          |
|------------|--------------------------|
| **Class**  | `ObjectTypeValidity`     |
| **Key**    | `object_type_validity`   |
| **Branch** | object (`ObjectNode`)    |
| **Needs**  | `expected`               |

## What it measures

A **structural sanity check**: of an object's present fields, what fraction carry the
right JSON type — independent of whether the values are correct. It catches the common
LLM slips of emitting `"199"` (string) where `199` (number) is expected, or a `list`
where an object was expected, across a whole object at once.

Every present field is checked with [`TypeMatch`](type-match.md), which covers all JSON
types — so scalars **and** containers (object/array) are validated. It's a basic type
check, not a deep one (a nested object's *own* fields are graded at that node). The score
is the fraction of present fields that pass.

## Parameters

None — `ObjectTypeValidity()`.

## How it's computed

```text
present = fields in both actual and expected (any kind)
score = (number of present fields with the right JSON type) / len(present)
```

Count-based (no weights). **All present fields** are considered — scalars and containers
alike; missing fields are out of scope. An object with no present fields is vacuously
`1.0`.

## Example

`duration_hours` has the right type; `price` came back as a string — one of two valid:

```python
from structured_eval import evaluate, EvalConfig, ObjectTypeValidity

config = EvalConfig(metrics=[ObjectTypeValidity()])
report = evaluate(
    {"duration_hours": 99, "price": "199"},     # price is a string, should be number
    {"duration_hours": 12, "price": 199},
    config,
)

# duration_hours: number✓, price: string vs number ✗  →  1 of 2
float(report.metrics["object_type_validity"].root())   # 0.5
```

## Edge cases

- **Value-blind** — `duration_hours` is `99` vs `12` (wrong value) but the *type* is
  right, so it passes. Pair with a value metric to judge correctness.
- **Containers count too** — a `list` where an object is expected (or vice versa) is
  type-invalid; it's a shallow check, not a deep one (the nested node's own fields are
  graded separately).
- **Present fields only** — missing fields don't count, against or for.
- **No present fields** → vacuously `1.0`.
- **`int`/`float` are both `number`; `bool` is its own type** — see
  [`TypeMatch`](type-match.md).

## See also

- [`TypeMatch`](type-match.md) — the per-field type check this aggregates.
- [`ObjectF1`](object-f1.md) / [`ObjectAccuracy`](object-accuracy.md) — value
  correctness, not type.
- [The metric catalog](../index.md) — all metrics and the return-shape model.
