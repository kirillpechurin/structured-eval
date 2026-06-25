# Presence

|            |                       |
|------------|-----------------------|
| **Class**  | `Presence`            |
| **Key**    | `presence`            |
| **Branch** | field (scalar leaves) |
| **Needs**  | nothing (one-sided)   |

## What it measures

Whether a field came back **populated**: `1.0` if `actual` is present and non-null,
else `0.0`. It's a one-sided check — it ignores `expected` and looks only at the actual
value. Use it to measure coverage/completeness: did the model emit this field at all?

A missing key and an explicit `null` both score `0.0` — for this signal, a null is as
good as missing.

## Parameters

None — `Presence()`.

## How it's computed

```text
score = 1.0 if actual is not None else 0.0
```

`expected` is not consulted.

## Example

Only `null` (or a missing key) counts as absent — falsy-but-present values still pass:

```python
from structured_eval import evaluate, EvalConfig, FieldConfig, Presence

config = EvalConfig(fields={
    "title":   FieldConfig(metrics=[Presence()]),
    "price":   FieldConfig(metrics=[Presence()]),
    "summary": FieldConfig(metrics=[Presence()]),
})
report = evaluate(
    {"title": "", "price": 0, "summary": None},
    {"title": "Intro", "price": 199, "summary": "..."},
    config,
)

float(report.field_scores["title"].metrics["presence"])     # 1.0 — "" is present
float(report.field_scores["price"].metrics["presence"])     # 1.0 — 0 is present
float(report.field_scores["summary"].metrics["presence"])   # 0.0 — null
```

## Edge cases

- **Non-null, not key-exists** — it measures non-null-ness. A present-but-`null` field
  scores `0.0`, the same as a missing key.
- **Falsy values count as present** — `""`, `0`, `False` all score `1.0` (they aren't
  `None`).
- **Ignores `expected`** — it never compares; a field present in `actual` scores `1.0`
  even if it's the wrong value. Pair it with a value metric to judge correctness.

## See also

- [`ExactMatch`](exact-match.md) / [`TokenF1`](token-f1.md) — judge the *value* once
  it's present.
- [`CoverageLeafScore`](coverage-leaf-score.md) — coverage of expected leaves across
  the whole document.
- [The metric catalog](../index.md) — all metrics and the return-shape model.
