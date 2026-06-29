# CompositeScore

|            |                       |
|------------|-----------------------|
| **Class**  | `CompositeScore`      |
| **Key**    | `composite_score`     |
| **Branch** | any-node              |
| **Needs**  | other metrics on the node |

## What it measures

A **normalized weighted blend** of other metrics already computed on the same node. Where
the default [`MeanScore`](mean-score.md) averages a node's metrics equally,
`CompositeScore` lets you say *how much* each one counts — e.g. "the title is 70 % token
overlap, 30 % character overlap". Use it as a node's `key_metric` to shape the single
representative score that bubbles up to `report.score`.

## Parameters

| Param     | Default | Meaning                                                          |
|-----------|---------|------------------------------------------------------------------|
| `weights` | —       | `dict[str, float]` mapping a metric's **key** to its weight. Required, non-empty, positive sum. Weights are normalized to sum `1.0`. |

## How it's computed

```text
score = Σ wᵢ · node.metric_results[nameᵢ]        (weights normalized; clamped to [0, 1])
```

It reads the named metrics out of the node's already-computed results, so list those
metrics on the node **and** set `CompositeScore` as the `key_metric` (the engine runs the
`key_metric` **last**, after every other metric on the node is in place).

## Example

```python
from structured_eval import (
    evaluate, EvalConfig, FieldConfig, TokenF1, CharacterF1, CompositeScore,
)

config = EvalConfig(fields={
    "title": FieldConfig(
        metrics=[TokenF1(), CharacterF1()],
        key_metric=CompositeScore({"token_f1": 0.7, "character_f1": 0.3}),
    ),
})

report = evaluate(
    {"title": "Intro to Python"},
    {"title": "Introduction to Python"},
    config,
)
# token_f1 and character_f1 blended 0.7 / 0.3 → the field's representative score
report.field_scores["title"].score                          # ≈ 0.703
report.field_scores["title"].metrics["composite_score"]     # ≈ 0.703
```

## Edge cases

- **Unknown metrics ignored** — a metric on the node but absent from `weights` does not
  contribute (only the named ones do).
- **Absent named metric → `0`** — a weighted name that isn't on the node contributes `0`
  (the weights are still normalized over all the names you listed).
- **Invalid weights raise** — an empty dict, or weights summing to `≤ 0`, raise
  `ValueError`.
- **Inputs should be in `[0, 1]`** — the blend is clamped to `[0, 1]`; feed it metrics
  that return scores in that range.
- **Best as `key_metric`** — set it as the representative so it runs after its inputs. As
  an ordinary cascaded metric it only sees results computed before it.

## See also

- [`MeanScore`](mean-score.md) — the unweighted default representative.
- [The representative score](../core-concepts/evaluation-model.md#the-representative-score-key_metric) — how `key_metric` works.
- [The metric catalog](../index.md) — all metrics and the return-shape model.
