# Introduction

**structured-eval** is a field-level evaluation framework for LLM structured outputs (JSON/YAML).

## Why structured-eval

LLMs increasingly return structured data — JSON or YAML shaped by a schema. Checking
the *structure* is easy: did it parse, are the types right, are the required fields
present, are there no extra keys.

But structural validity does not mean the values are right. The JSON can parse cleanly,
match the schema perfectly, and still report a `total` of `99` when it should be `100`,
or extract a name that never appeared in the source document. Most checks stop at the
structure and never look at the values.

structured-eval is built for exactly that gap. It evaluates your output **field by
field** — not just *is this valid JSON*, but *is this the right value, in the right
place, consistent with everything else*.

## Levels of correctness

It helps to think about correctness as a ladder. Each level assumes the ones below it:

| Level | What it checks |
|-------|----------------|
| L0 | The output parses as JSON/YAML at all |
| L1 | Field types match |
| L2 | Required fields are present |
| L3 | No extra fields — the structure matches exactly |
| L4 | Values are close to what's expected |
| L5 | Values are grounded in the source (no hallucination) |
| L6 | Fields are logically consistent with one another |

L0–L3 are about *structure*. They're necessary, but a model that scores 100% here can
still be wrong about everything that matters. L4–L6 are where structured-eval earns its
keep: the value-level correctness, faithfulness, and cross-field logic that structural
checks never reach.

## A first look

Here's the gap in a few lines. The structure is flawless — every key is present, every
type is right — but one value is off:

```python
from structured_eval import evaluate

actual   = {"id": "INV-1", "total": 99.0, "status": "paid"}
expected = {"id": "INV-1", "total": 100.0, "status": "paid"}

report = evaluate(actual, expected)

report.field_scores["total"].score   # 0.0  ← valid JSON, wrong value
report.field_scores["id"].score      # 1.0
```

The JSON is valid. The schema would pass. But the `total` is wrong, and structured-eval
sees it — right down to the field that broke.

## Next steps

- **[Getting started](getting-started.md)** — install structured-eval and run your first evaluation.
- **[Core concepts](core-concepts/evaluation-model.md)** — how the evaluation tree, metrics, and scores fit together.
