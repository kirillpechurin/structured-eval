# RulePassRate

|            |                                          |
|------------|------------------------------------------|
| **Class**  | `RulePassRate(rules)`                    |
| **Key**    | `rule_pass_rate`                         |
| **Branch** | root (whole document)                    |
| **Needs**  | nothing (reads `actual`; no `expected`)  |

## What it measures

The fraction of **business rules** that hold for the document — your L6 logic checks. Each
rule is an assertion about the document (a field's value, a relationship between fields, a
membership constraint); the score is simply how many of them pass. It needs no ground-truth
answer, so you can enforce invariants on any output.

Rules are written with the `Rule` DSL (JSONPath + a comparator) or wrapped as an arbitrary
function with `Rule.custom`.

## Parameters

| Param   | Meaning                                                         |
|---------|-----------------------------------------------------------------|
| `rules` | a list of `Rule(...)` / `Rule.custom(...)` objects (required)   |

The path DSL needs the optional `rules` extra (`pip install 'structured-eval[rules]'`).

## How it's computed

```text
score = (number of rules that pass) / (number of rules)     # 1.0 if the list is empty

each rule's outcome is recorded as a RuleResult(name, passed, message)
```

A rule that can't be evaluated (missing path, type-mismatched comparison, a raising custom
function) counts as **failed**, with the reason in its `message` — the run never crashes on
a bad rule.

## Rule DSL

```python
Rule("$.status").eq("paid")                 # eq / lt / gt / lte / gte
Rule("$.hours").gte(0)
Rule("$.total").eq("$.subtotal + $.tax")    # cross-field arithmetic (+ - * /)
Rule("$.level").in_(["beginner", "advanced"])
Rule.custom(lambda doc: doc["hours"] > 0, name="positive_hours")
```

## Example

Three rules, one document that violates two of them:

```python
from structured_eval import evaluate, EvalConfig
from structured_eval.metrics.rule_pass_rate import RulePassRate
from structured_eval.metrics.rule_pass_rate.dsl import Rule

rules = [
    Rule("$.total").eq("$.subtotal + $.tax"),     # 130 != 120  → fails
    Rule("$.hours").gte(0),                        # 40 >= 0     → passes
    Rule("$.level").in_(["beginner", "advanced"]), # 'expert'    → fails
]
config = EvalConfig(key_metric=RulePassRate(rules))
report = evaluate(
    {"subtotal": 100, "tax": 20, "total": 130, "hours": 40, "level": "expert"},
    None,                                          # no expected needed
    config,
)

report.score                                       # 0.3333 — 1 of 3
report.score_label                                 # "rule_pass_rate"

for rr in report.metrics["rule_pass_rate"].root().extra["rule_results"]:
    print(rr["name"], rr["passed"], rr["message"])
# $.total eq $.subtotal + $.tax  False  '$.total' (130) does not satisfy eq(120)
# $.hours gte 0                  True
# $.level in ['beginner', ...]   False  '$.level' ('expert') does not satisfy in([...])
```

## Edge cases

- **No `expected` required** — rules judge the document on its own; pass `expected=None`.
- **Bad rule = failed, not crash** — read `message` to tell a genuinely violated rule from a
  rule that couldn't run (e.g. a path that doesn't exist).
- **Safe arithmetic** — cross-field expressions allow only `+ - * /` and constants; there's
  no arbitrary code execution.
- **Empty rule list** → vacuously `1.0`.

## See also

- [`SchemaValidity`](schema-validity.md) — structural constraints (types/required) vs logic.
- [`FieldFaithfulness`](field_faithfulness.md) — another no-`expected` check: grounding in a source.
- [The metric catalog](../index.md) — all metrics and the return-shape model.
