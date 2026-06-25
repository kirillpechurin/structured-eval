# TypeMatch

|            |                       |
|------------|-----------------------|
| **Class**  | `TypeMatch`           |
| **Key**    | `type_match`          |
| **Branch** | field (scalar leaves) |
| **Needs**  | `expected`            |

## What it measures

Whether `actual` and `expected` have the **same JSON type**: `1.0` if they do, else
`0.0` — regardless of the value. It catches a common LLM error, emitting `"100"` (a
string) where `100` (a number) is expected, independently of whether the number is
right.

## Parameters

None — `TypeMatch()`.

## How it's computed

Each side is mapped to a JSON type, then compared:

| Python value        | JSON type  |
|---------------------|------------|
| `bool`              | `bool`     |
| `int` / `float`     | `number`   |
| `str`               | `string`   |
| `list`              | `array`    |
| `dict`              | `object`   |
| `None`              | `null`     |

```text
score = 1.0 if json_type(actual) == json_type(expected) else 0.0
```

## Example

Same type passes even when the value is wrong; a string-vs-number mismatch fails:

```python
from structured_eval import evaluate, EvalConfig, FieldConfig, TypeMatch

config = EvalConfig(fields={
    "duration_hours": FieldConfig(metrics=[TypeMatch()]),
    "price": FieldConfig(metrics=[TypeMatch()]),
})
report = evaluate(
    {"duration_hours": 99, "price": "199"},     # price came back as a string
    {"duration_hours": 12, "price": 199},
    config,
)

float(report.field_scores["duration_hours"].metrics["type_match"])   # 1.0 — both numbers
float(report.field_scores["price"].metrics["type_match"])            # 0.0 — string vs number
```

## Edge cases

- **`int` and `float` are both `number`** — `100` matches `100.0` (JSON has one number
  type).
- **`bool` is its own type** — `True` is *not* a `number`, so `True` vs `1` → `0.0`.
- **`null`** — `None` vs `None` → `1.0`; `None` vs anything else → `0.0`.
- **Value-blind** — it never looks at the value, only the type. Pair it with a value
  metric (e.g. [`Numeric`](numeric.md)) to also judge correctness.

## See also

- [`Numeric`](numeric.md) — judge the numeric *value* once the type is right.
- [`Presence`](presence.md) — whether the field is populated at all.
- [The metric catalog](../index.md) — all metrics and the return-shape model.
