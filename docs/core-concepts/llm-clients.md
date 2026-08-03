# LLM clients

Some metrics ask a model instead of comparing values — [`JudgeFaithfulness`](../metrics/catalog/judge-faithfulness.md)
is the first. This page is how you give them one.

## The core ships no provider SDK

structured-eval depends on pydantic and nothing else. There is no `openai`, no `anthropic`, no `litellm` in the core —
instead there is a **seam**, `structured_eval.llm`, and every LLM-backed feature talks to that and nothing else.

```text
metric  →  LlmClient.generate(prompt, system=...)         -> str
           LlmClient.generate_with_schema(prompt, Model)  -> Model
```

Two methods, one abstract. That is the whole contract, and it is why *your* model — behind your proxy, with your
retries, your tracing and your rate limits — is a first-class client rather than something to work around.

## Choosing one

Whatever you pass as `client=` is coerced by `resolve_client`, so you never pick an adapter class by hand. Pick the row
that describes what you already have:

| You have                              | Pass                          | You get                                        |
|---------------------------------------|-------------------------------|------------------------------------------------|
| nothing yet                           | `client=None` (the default)   | the model named by `STRUCTURED_EVAL_LLM_MODEL` |
| a model name                          | `"qwen/qwen3-235b-a22b-2507"` | `LiteLlmClient` — needs the `litellm` extra    |
| a LangChain-style chat model          | the model object              | `ChatModelClient` — duck-typed on `.invoke()`  |
| one function that calls your provider | the function                  | `CallableClient`                               |
| something else entirely               | your own `LlmClient` subclass | itself, untouched                              |

Anything else raises `TypeError` naming what was expected.

> The string form names a **model**, not a registered class — unlike `resolve_metric("numeric")`,
> whose strings are registry keys.

### Zero code: an environment variable

The cheapest way to try an LLM-backed metric is not to write a client at all:

```bash
export STRUCTURED_EVAL_LLM_MODEL="anthropic/claude-opus-5"
export ANTHROPIC_API_KEY="..."         # each provider's own variable
```

```python
JudgeFaithfulness()  # no client= — reads the variable
```

Only the *model* is ours; credentials stay in each provider's own variable, which LiteLLM reads itself. So switching
providers is a one-string change.

Two deliberate non-behaviours:

- **`.env` is not read for you.** A library quietly loading files from the working directory would be a surprise;
  calling `load_dotenv()` (or exporting) stays your decision.
- **An unset variable is an error, not a default.** Nobody should discover which model graded their data by reading a
  bill.

The client is resolved on **first use**, so building a config costs no credentials — a config is constructible on a
machine with no keys at all, and only running the metric needs them.

### A model name — `LiteLlmClient`

Behind the `litellm` extra (`pip install 'structured-eval[litellm]'`). Every provider LiteLLM speaks, addressed as one
string:

```python
from structured_eval.llm import LiteLlmClient

JudgeFaithfulness(client="openai/gpt-5.5")  # the short form
JudgeFaithfulness(client=LiteLlmClient(  # ...or configured
    "ollama/llama3",
    temperature=0.0,
    timeout=30,
    api_base="http://localhost:11434",  # any extra kwarg is
))  # forwarded to litellm
```

`temperature` / `max_tokens` / `timeout` are passed through **only when set**, so each provider's own defaults apply
otherwise. For a judge, pinning `temperature=0.0` is usually what you want — it will not make the metric deterministic,
but it narrows the spread.

### A chat model you already configured

Anything exposing `.invoke()` — nothing is imported from LangChain:

```python
from langchain_anthropic import ChatAnthropic

JudgeFaithfulness(client=ChatAnthropic(model="claude-opus-5", temperature=0))
```

Preferred over wrapping the same model in a callable: `system` becomes a real message turn instead of prompt text,
replies arriving as content blocks are flattened, and schema requests go through `.with_structured_output()` — the
provider's native structured outputs instead of prompt-and-parse. Your proxy, retries, callbacks and tracing come along.

### A plain function

The smallest client there is — a function with `generate`'s own signature:

```python
def my_model(prompt: str, *, system: str | None = None) -> str:
    return my_sdk.complete(system=system, user=prompt).text


JudgeFaithfulness(client=my_model)
```

Schema support comes from the base class, so this one function is enough for both methods. It is also the **test seam**:
a function returning canned JSON makes an LLM metric free and deterministic in unit tests, with no network and no
mocking library — see
[`JudgeFaithfulness`](../metrics/catalog/judge-faithfulness.md#example).

### Your own client

Subclass `LlmClient` and implement `generate`. That is genuinely all:

```python
from structured_eval.llm import LlmClient


class MyClient(LlmClient):
    model_name = "internal/reviewer-v3"  # descriptive; records what graded what

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        return internal_gateway.chat(system, prompt)
```

## Structured replies

Judges need a *shape* back, not prose. `generate_with_schema(prompt, Model)` returns a validated pydantic instance, and
it has a **working default** built on `generate`: the JSON Schema is appended to the prompt, and the reply is extracted
(tolerating code fences and narration) and validated on our side.

So a ten-line wrapper is already a complete client. A provider that can constrain generation natively overrides the
method and gets a format *guarantee* instead of a request — both shipped clients do:

- `LiteLlmClient` uses `response_format` when LiteLLM reports the model supports it, and falls back to prompt-and-parse
  when it doesn't. An unrecognised model is not an error.
- `ChatModelClient` uses `.with_structured_output()` when present.

To customise without reimplementing the method, override one of its three steps:
`schema_prompt` (reword the instruction), `parse_reply` (change how a reply is sliced), or
`validate_reply` (hand over an already-parsed payload).

## Errors

Everything this layer raises descends from `LlmError`, so one `except` catches the lot:

| Exception                | Means                                                                                                                |
|--------------------------|----------------------------------------------------------------------------------------------------------------------|
| `LlmInvocationError`     | the provider call failed — timeout, auth, rate limit, transport. The provider's own exception is kept as `__cause__` |
| `LlmResponseFormatError` | the reply could not be read as the requested schema                                                                  |

```python
from structured_eval.llm import LlmError

try:
    report = evaluate(actual, None, config, source=source)
except LlmError as exc:
    ...  # the provider's exception is exc.__cause__
```

**This layer owns no retry policy.** No backoff, no timeout of its own, no fallback model — those belong to the client
you inject, where you can see and tune them. LiteLLM has
`num_retries`; a LangChain model has `.with_retry()`; your own client has whatever you put there.

An `LlmError` propagates out of `evaluate()` rather than being swallowed into a score, so a failed call never reads as a
bad model output. Over a batch that means one broken reply stops the run — if you would rather lose a sample than the
run, catch it per sample around
`evaluate()`.

## Cost

An LLM metric is the only thing in structured-eval that costs money and wall time, and it is the only one that never
runs unless you attach it explicitly. Two habits keep the bill honest:

- **Judge subtrees, not fields.** One attached judge is one call, however wide the subtree.
  See [where to attach it](../metrics/catalog/judge-faithfulness.md#where-to-attach-it).
- **Let the free metrics go first.** `FieldFaithfulness`, `Numeric`, `Fuzzy` and the rest decide most fields for
  nothing. Pay the model only where nothing cheaper can.

Calls are made **serially**, one per attached judge per sample — there is no batching, no caching, and no concurrency
inside `evaluate_batch`. Size a dataset accordingly.

## See also

- [`JudgeFaithfulness`](../metrics/catalog/judge-faithfulness.md) — the first metric that uses this.
- [The evaluation model](evaluation-model.md) — where metrics sit in the pipeline.
- [Custom metrics](../metrics/custom-metric.md) — writing your own, LLM-backed or not.
