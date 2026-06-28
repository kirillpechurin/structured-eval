# FieldFaithfulness

|            |                                              |
|------------|----------------------------------------------|
| **Class**  | `FieldFaithfulness()`                        |
| **Key**    | `field_faithfulness`                         |
| **Branch** | field (each scalar leaf)                     |
| **Needs**  | a grounding `source` (no `expected` needed)  |

## What it measures

Whether each value is **grounded in a source text** — it catches *hallucinations*, fields
the model emitted that aren't supported by the input it was extracted from. No ground-truth
answer is required; you check the output against its source.

It's a **per-field** metric: every scalar leaf scores `1.0` if its value appears in the
`source`, else `0.0`. Because it's an ordinary field metric, the usual leaf roll-up
(`MeanScore` / [`OverallLeafScore`](overall-leaf-score.md)) gives you a document-level
faithfulness number, and the hallucinated fields are simply the leaves that scored `0.0`.

The current check is **L1: case-insensitive substring matching** — a value is grounded if
its string form appears verbatim in the source. It's a cheap, deterministic floor; deeper
semantic checks (token/embedding overlap, entailment, an LLM judge) are on the roadmap as
separate metrics.

## Parameters

None — `FieldFaithfulness()`. The grounding text is passed per call as
`evaluate(..., source=...)`.

## How it's computed

```text
for each scalar leaf:
    grounded = str(value).lower() is a substring of source.lower()
    score    = 1.0 if grounded else 0.0          # a null value is skipped

document score = mean of the leaf scores         # via MeanScore / OverallLeafScore
hallucinations = the leaves scoring 0.0
```

A missing `source` is a configuration error — `FieldFaithfulness` raises `ValueError` rather
than silently skipping, because faithfulness is undefined without something to ground against.

## Example

`level` says `beginner`, but the source says the course is *advanced* — a hallucination:

```python
from structured_eval import evaluate, EvalConfig, FieldFaithfulness

source = "Intro to Machine Learning. Duration: 40 hours. Level: advanced."
config = EvalConfig(metrics=[FieldFaithfulness()])
report = evaluate(
    {"title": "Machine Learning", "hours": 40, "level": "beginner"},
    None,                                            # no expected
    config,
    source=source,
)

mc = report.metrics["field_faithfulness"]
mc.mean()                                            # 0.6667 — 2 of 3 grounded
{p: float(v) for p, v in mc.by_path.items()}         # {'title': 1.0, 'hours': 1.0, 'level': 0.0}

# the hallucinated fields are the leaves scoring 0.0:
[p for p, v in mc.by_path.items() if float(v) == 0.0]   # ['level']
```

## Edge cases

- **Requires `source=`** — omitting it raises `ValueError`.
- **Substring, so imperfect** — `"40"` matches any "40" in the text (false positive);
  paraphrases or different casing of long values may be missed (false negative). It's a
  baseline signal, not semantic verification.
- **Null values are skipped** — a `null` leaf has nothing to ground, so it isn't counted.
- **Array elements need `expected`** — in source-only mode (`expected=None`) list items
  aren't aligned into nodes, so faithfulness doesn't reach them yet; scalars and
  nested-object fields are checked normally. (Materializing list items without `expected` is
  on the roadmap.)
- **No checkable leaves** → the metric simply doesn't appear in `report.metrics`.

## See also

- [`RulePassRate`](rule-pass-rate.md) — the other no-`expected` check: business logic.
- [`OverallLeafScore`](overall-leaf-score.md) — roll the per-field scores into one headline.
- [`Presence`](presence.md) — another value-on-actual field metric.
- [The metric catalog](../index.md) — all metrics and the return-shape model.
