# ObjectAccuracy

|            |                       |
|------------|-----------------------|
| **Class**  | `ObjectAccuracy`      |
| **Key**    | `object_accuracy`     |
| **Branch** | object (`ObjectNode`) |
| **Needs**  | `expected`            |

> The **default object metric**: an object node you don't configure gets it.

## What it measures

The **mean correctness of an object's expected fields** — a soft, continuous score in
`[0, 1]`. Each expected field contributes its own score (its `representative`); missing
fields count as `0.0`. Unlike [`ObjectF1`](object-f1.md), it rewards partial matches
directly (no threshold) and **does not penalize extra fields** — it's a
**recall-flavored** mean (formally, *soft recall*).

Use it as a forgiving "how right are the fields we expected?" summary; use `ObjectF1`
when spurious fields should count against the score.

## Parameters

| Param          | Default          | Meaning                                                  |
|----------------|------------------|----------------------------------------------------------|
| `score_policy` | `None`           | per-field `{name: metric}` override of the match criterion |
| `weight_mode`  | `"proportional"` | weight fields by their `weight`; `none` = plain mean      |

(No `threshold`/`mode` — it's inherently soft.)

## How it's computed

```text
score = Σ weight · field_score(matched) / (Σ weight(matched) + Σ weight(missing))
```

The denominator is the **expected** side only (matched + missing), so adding an extra
field doesn't change the score. With uniform weights this is just the average field
score, counting each missing field as 0.

## Example

`name` is right, `experience_years` is wrong, `title` is missing:

```python
from structured_eval import evaluate, EvalConfig, ObjectAccuracy

config = EvalConfig(metrics=[ObjectAccuracy()])
report = evaluate(
    {"name": "Sarah Johnson", "experience_years": 5},                 # title missing
    {"name": "Sarah Johnson", "experience_years": 8, "title": "PhD"},
    config,
)

# scores: name 1.0, experience 0.0, title (missing) 0.0  →  mean over 3 expected
float(report.metrics["object_accuracy"].root())   # 0.333
```

## Edge cases

- **Recall-flavored** — extra/spurious fields are not penalized (only expected fields
  are in the denominator). Use [`ObjectF1`](object-f1.md) for a precision-aware score.
- **Soft by design** — no threshold; each field's fractional score counts directly, so
  graded field metrics (e.g. [`NumericCloseness`](numeric-closeness.md),
  [`TokenF1`](token-f1.md)) flow through.
- **All child kinds count** — nested objects/arrays contribute via their representative.
- **Empty object** — no expected fields → vacuously `1.0`.

## See also

- [`ObjectF1`](object-f1.md) — precision-aware (penalizes extra fields).
- [`ObjectRecall`](object-recall.md) — the hard-threshold recall this softens.
- [The metric catalog](../index.md) — all metrics and the return-shape model.
