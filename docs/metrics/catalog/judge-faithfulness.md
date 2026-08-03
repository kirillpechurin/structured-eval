# JudgeFaithfulness

|            |                                                                                                     |
|------------|-----------------------------------------------------------------------------------------------------|
| **Class**  | `JudgeFaithfulness(criteria=None, *, client=None, verdict_scores=None)`                             |
| **Key**    | `judge_faithfulness`                                                                                |
| **Branch** | any-node (attach it to the node you want judged)                                                    |
| **Needs**  | a grounding `source` and an [LLM client](../../core-concepts/llm-clients.md) (no `expected` needed) |

## What it measures

Whether each field beneath a node is **grounded in a source text**, as decided by an LLM. Same question as [
`FieldFaithfulness`](field_faithfulness.md), asked of a model instead of a substring check: it catches the fabrications
a string comparison cannot see — a value that paraphrases the source, contradicts it in meaning rather than in
characters, or is simply plausible and absent.

Attach it to a node and it takes every leaf beneath that node, asks **one** LLM call whether the source backs each
value, and rules on each:

| Verdict        | Meaning                                               | Default score |
|----------------|-------------------------------------------------------|---------------|
| `supported`    | the source states the value, or it follows directly   | `1.0`         |
| `contradicted` | the source states something incompatible              | `0.0`         |
| `not_stated`   | nothing in the source backs the value — a fabrication | `0.0`         |

The node scores the mean of those verdicts, weighted by each field's `weight`. The verdicts themselves travel with the
score in `.extra["verdict"]`, each naming the field it is about — that detail is the reason to pay a model rather than
compare strings.

> **It costs money and it is not deterministic.** It never runs by default; it runs only
> where you attach it explicitly. Where you hang it is a budget decision — see
> [Where to attach it](#where-to-attach-it).

## Parameters

| Param            | Meaning                                                                                                                                                                               |
|------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `criteria`       | what "faithful" means: one string for every field, or `{relative_path: rule}`. `None` uses a generic default                                                                          |
| `client`         | the model — a `"provider/model"` string, a chat model, a callable, or an `LlmClient`. `None` reads `STRUCTURED_EVAL_LLM_MODEL`. See [LLM clients](../../core-concepts/llm-clients.md) |
| `verdict_scores` | overrides for the verdict → score table above (e.g. `{"not_stated": 1.0}`)                                                                                                            |

The client is resolved on **first use**, so building a config costs no credentials.

### Criteria

A criterion is the judge's most valuable input — it says what the field is supposed to be, which the model otherwise has
to guess from the field's name. Either one rule for everything:

```python
JudgeFaithfulness("the value must be stated in the syllabus, not inferred from it")
```

or a rule per field, keyed by paths **relative to the judged node**, with `[*]` for array elements — the same wildcard
the rest of the config uses:

```python
JudgeFaithfulness({
    "instructor": "must be the person teaching, not the author of the reading list",
    "modules[*].title": "the module must be one the syllabus actually lists",
})
```

Because the keys are relative, the same configuration works wherever the judge is hung. Fields with no entry fall back
to the generic default. A mapping on a *single field* is refused — a leaf has nothing beneath it for the keys to
address; pass one string instead.

## How it's computed

```text
fields  = every leaf beneath the judged node that the output actually produced
          (an explicit null counts; a key only `expected` has does not)

one LLM call for all of them together, each listed as (path, value, criterion)

score   = Σ verdict_scores[verdict_i] · weight_i  /  Σ weight_i

the verdicts land in .extra["verdict"] = {score, reason, verdicts: [
    {path, verdict, reason},                    # path is absolute
]}
```

Verdicts are asked for by path **relative** to the judged node and reported by the **absolute** one: the prompt speaks
the language of the subtree, the report speaks the language of the document.

A field the judge said nothing about is left out of the mean rather than counted as a zero — silence is missing
evidence, not a failed check — and a path the model invented is dropped.

## Example

A course record extracted from a syllabus blurb. `level` contradicts the source and
`certificate` is invented:

```python
import json

from structured_eval import evaluate
from structured_eval.metrics import JudgeFaithfulness
from structured_eval.models import EvalConfig, ObjectFieldConfig

SOURCE = (
    "Applied Statistics runs for 30 hours across 6 modules. "
    "It is taught by Dr. Rivera and is aimed at intermediate students."
)


# A stub model, so this page's numbers are reproducible. In real use pass a
# model instead: JudgeFaithfulness(client="qwen/qwen3-235b-a22b-2507").
def stub_judge(prompt: str, *, system: str | None = None) -> str:
    return json.dumps({"verdicts": [
        {"path": "title", "verdict": "supported"},
        {"path": "hours", "verdict": "supported"},
        {"path": "level", "verdict": "contradicted",
         "reason": "the source says intermediate"},
        {"path": "instructor", "verdict": "supported"},
        {"path": "certificate", "verdict": "not_stated",
         "reason": "certificates are never mentioned"},
    ]})


config = EvalConfig(
    root=ObjectFieldConfig(metrics=[JudgeFaithfulness(client=stub_judge)])
)
report = evaluate(
    {
        "title": "Applied Statistics",
        "hours": 30,
        "level": "beginner",
        "instructor": "Dr. Rivera",
        "certificate": True,
    },
    None,  # no expected — only the source
    config,
    source=SOURCE,
)

result = report.field_scores["$"].metrics["judge_faithfulness"]
float(result)  # 0.6 — 3 of 5 fields grounded

for v in result.extra["verdict"]["verdicts"]:
    print(v["path"], v["verdict"], v["reason"])
# title       supported
# hours       supported
# level       contradicted  the source says intermediate
# instructor  supported
# certificate not_stated    certificates are never mentioned

# the fabrications are the verdicts that aren't `supported`:
[v["path"] for v in result.extra["verdict"]["verdicts"]
 if v["verdict"] != "supported"]  # ['level', 'certificate']
```

## Where to attach it

The judge is an **any-node** metric, so it fits everywhere — which makes *where* a real decision, and the one that sets
your bill. One attached judge is one LLM call, however deep the subtree beneath it.

| Attachment                                            | What you get                                     | Cost                            |
|-------------------------------------------------------|--------------------------------------------------|---------------------------------|
| `root=ObjectFieldConfig(metrics=[...])`               | one score for the document, verdicts as evidence | 1 call                          |
| `fields={"modules": ArrayFieldConfig(metrics=[...])}` | one score for that subtree                       | 1 call                          |
| `fields={"level": FieldConfig(metrics=[...])}`        | the judge's verdict **is** that field's score    | 1 call per field                |
| `item=` inside an `ArrayFieldConfig`                  | one judge per element                            | 1 call **per element**          |
| `EvalConfig(metrics=[...])`                           | cascades onto every node                         | 1 call **per node** — see below |

Judging a whole subtree in one call is not only cheaper: the fields are judged *together*, so a value that only makes
sense beside its neighbours is not ruled on in isolation.

```python
from structured_eval.models import ArrayFieldConfig

config = EvalConfig(fields={
    "modules": ArrayFieldConfig(metrics=[JudgeFaithfulness(
        {"[*].title": "the module must be one the syllabus lists"},
        client=stub_judge,
    )])
})
# → report.field_scores["modules"].metrics["judge_faithfulness"] == 0.5
#   verdicts name 'modules[0].title' / 'modules[1].title'; `course` is never judged
```

**A verdict is not the score of the field it names.** A judge hung on the root rules on
`level`, but `level` keeps its own metrics — the verdict is evidence inside the judged node's result. To make the
judge's opinion a field's own score, hang a judge on that field.

> ⚠️ **Temporary warning!** **Do not cascade it via `EvalConfig(metrics=[...])`.** That attaches it to *every* node
> its branch fits, and its branch fits all of them — a 4-node document becomes 4 model calls.
> Nothing forbids it; the bill does.

## Edge cases

- **Requires `source=`** — omitting it raises `ValueError`, as faithfulness is undefined without something to ground
  against.
- **Only what the output produced is judged.** An explicit `null` **is** a claim ("the source gives no value here") and
  is judged like any other. A field that appears only in `expected`
  claims nothing and is skipped — that gap is recall, measured by
  [`Presence`](presence.md) / [`CoverageLeafScore`](coverage-leaf-score.md).
- **Containers aren't judged, their leaves are.** An object or an array makes no claim of its own; every claim it makes
  is one of the values underneath it.
- **No leaves to judge** → no call is made and the metric doesn't appear in the report.
- **A sloppy reply can't invent numbers.** A field the judge ignored is absent from the mean; a path it invented is
  dropped; a repeated path keeps the first verdict.
- **`not_stated` scores 0.0** — this is where we part with deepeval, whose `idk` counts as faithful. In RAG "the context
  doesn't mention it" is benign; in a *structured extraction* a value nobody stated is exactly the fabrication the
  metric exists to catch. Pass
  `verdict_scores={"not_stated": 1.0}` for the other reading.
- **Non-deterministic** — two runs of the same document can differ. Pin a model and, where the provider allows it, a
  temperature; see [LLM clients](../../core-concepts/llm-clients.md).

## See also

- [`FieldFaithfulness`](field_faithfulness.md) — the deterministic, free L1 floor: does the value appear verbatim in the
  source? Run it first; pay the model only where it can't decide.
- [LLM clients](../../core-concepts/llm-clients.md) — choosing and configuring the model.
- [`RulePassRate`](rule-pass-rate.md) — the other no-`expected` check: business logic.
- [The metric catalog](../index.md) — all metrics and the return-shape model.
