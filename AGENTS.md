# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`structured-eval` — a declarative, **field-level** evaluation framework for LLM
structured outputs (JSON/YAML). It scores each field of an output against an
expected value (or a schema/source/rules), rather than a single pass/fail.
Positioning: correctness is a ladder L0–L6 (L0–L3 structure → L4 values → L5
faithfulness → L6 logic); the project's value is L4–L6.

## Canonical docs — read these first

- **`docs/`** — user-facing documentation: `core-concepts/` explains the evaluation
  model, "comparison is a metric", and array alignment; `metrics/catalog/` documents
  every metric one page each. The closest thing to an architecture map.
- **`tests/README.md`** — test architecture and conventions; read before writing tests.
- **`CONTRIBUTING.md`** — workflow, PR expectations.
- **When the docs and the code disagree, the code wins.** The source is the only
  authority on current behaviour; treat prose as intent, not as a spec.

## Commands

```bash
make check       # lintcheck + format-check + typecheck (must be green before a PR)
make test        # uv run pytest, then make doctest
make doctest     # pytest --doctest-modules structured_eval (examples in docstrings)
make test-cov    # pytest with coverage (html + xml + terminal, gated at 90%)
make lintcheck   # uv run ruff check
make typecheck   # uv run mypy --strict
make format      # ruff format + ruff --fix-only (writes changes)

uv run pytest tests/unit/metrics/test_numeric.py            # single file
uv run pytest tests/unit/metrics/test_numeric.py::test_name # single test
uv run pytest -m unit            # by marker: unit / engine / integration / golden / property
```

Environment: Python 3.12, `uv` / `.venv`. Setup
with `uv sync --extra all`. `make check && make test` must both be green before a
PR; mypy is `--strict`.

CI (`.github/workflows/ci.yml`) runs these same `make` targets; the same targets
also back the pre-commit hooks.

## Architecture essentials

Layered, dependencies point **downward only** — never import upward:

```
models / llm ← metrics / alignment / formats / utils ← engine / reporting ← integrations / api
```

Three engine phases (`engine/`): **parse → build tree & resolve each node's metric
list → compute every node's metrics post-order → build report**. The key ideas:

- **"Comparison is a metric."** There is no separate matcher with a precomputed
  similarity. A field metric itself compares `(node.actual, node.expected) → score`,
  and a node can carry several metrics at once.
- **Every node owns its metrics.** `TreeBuilder` cascades config metrics by node
  type and merges per-node `cfg.metrics`. Every node also gets a `key_metric` — its
  **representative score** (default `MeanScore` = mean of the node's own metrics,
  no recursion into children). `report.score` = the root node's `key_metric`.
- **`MetricInvoker(metric)` is the one way to run a metric** (`on_node` / `on_values`).
  Never call a metric's `compute` / `score` directly.

## Conventions (non-obvious)

- **Public API is intentionally narrow.** Top-level `structured_eval` exports **only**
  `evaluate` / `evaluate_batch` / `evaluate_consistency`. Everything else imports one
  level down: models via `structured_eval.models`, metrics/base classes/rule DSL via
  `structured_eval.metrics`, helpers via `structured_eval.utils`. Do **not** add names
  to the top-level `__init__.py`. Results are accessed via `report.field_scores[path]`,
  `report.metrics[name]`, `report.score` — **not** `report.f1` (that style does not exist).
- **One metric = one module** in `structured_eval/metrics/<snake>.py` (a metric with
  helper code becomes a package). Declaring the class auto-registers its `name`; also
  add the import **and** `__all__` entry in `metrics/__init__.py`. Never group metrics
  by node type.
- **A metric defining `__init__` must accept a trailing `name: str | None = None` and
  forward it via `super().__init__(name=name)`.** That is the per-instance name
  override (`Numeric(tolerance=0.01, name="strict")`), which lets two configurations
  of one metric coexist on a node under distinct report keys. The class-level `name`
  stays the registry key. `tests/unit/metrics/test_metric_contracts.py` enforces this
  across the whole registry.
- **Data models are pydantic v2** — use `model_dump` / `model_validate`.
- **Optional features are lazy-imported behind extras** (`yaml`, `fuzzy`, `jsonschema`,
  `rules`, `diff`, `report`, `litellm`, `deepeval`, `langsmith`, `all`). Guard any new
  optional import so the core stays installable without the extra.
- **LLM access goes through `structured_eval.llm`, never a provider SDK.** The core
  ships no provider dependency: features call `LlmClient.generate` /
  `generate_with_schema` and take whatever `resolve_client` returns (a client, a
  LangChain-style chat model, a callable, or a `"provider/model"` string routed to the
  `litellm` extra). Do not import a provider package outside `llm/`.
- **Tests mirror the source tree** one-to-one, one file per cohesive unit. Style:
  flat parametrized functions, no test classes, table-driven; `pytestmark` set once
  per file. New behaviour needs a test; coverage is gated in `pyproject.toml`.

## Docstrings — Google style

Docstrings are the source of the generated API reference, so how much a docstring
owes is decided by **import path and visibility**, not by taste:

| Tier | What | Owes |
|------|------|------|
| **A** | Named in some subpackage's `__all__`, plus that symbol's public methods and properties | summary + prose + `Args:` + `Returns:`/`Yields:` + `Raises:` + `Attributes:` (classes) + **`Example:`** |
| **B** | Public but not exported (`metrics/utils/*`, `rule_pass_rate/dsl.py`, …) | summary + the sections that apply; `Example:` optional |
| **C** | `_`-prefixed, plus anything a private symbol owns — nested functions included | one-line summary; sections only when the signature is not self-evident |

- **`Example:` uses doctest** (`>>>`) and is executed by `make doctest`, so an
  example cannot drift from the code. Non-deterministic ones (anything reaching an
  LLM) end in `# doctest: +SKIP`.
- **A one-line docstring needs no `Returns:`** — Google lets you drop the section
  when the summary already describes the return value, and ruff's
  `ignore-one-line-docstrings` encodes exactly that. Say it in the summary and
  stop; add the section only once the docstring has grown a body.
- **Document a field once, on the class that declares it.** An `Attributes:`
  entry per model field, and no trailing `# comment` restating it — subclasses
  do not repeat what the base class already documented.
- Tier A exempts pure pydantic models and `StrEnum`s from `Example:` — they owe an
  `Attributes:` section instead.
- **Markdown, not reST**, inside docstrings: the reference is rendered by
  mkdocstrings. Cross-reference as `[Numeric][structured_eval.metrics.Numeric]`;
  write `` `code` ``, never ``` ``code`` ``` or `:class:`/`:func:` roles.
- **Enumerate with a bullet list, explain with prose.** A package `__init__.py`
  listing its exports or submodules is an enumeration — write it as a Markdown
  `-` list, one line per entry, never flattened into a semicolon-joined
  sentence. A docstring that explains how something works is prose.
- **A summary line is one physical line of at most 80 characters** ending in
  `.`, `?` or `!`. Nothing enforces the 80 (`E501` is off and the formatter
  wraps at 88) — it is on you.
- **Keep it short.** A module docstring is a summary plus at most one short
  paragraph or list saying what is in the module and how it is used. Design
  rationale, algorithm detail and per-argument behaviour belong on the class or
  function they describe — not in the module docstring, and not restated in two
  places. The one thing worth its length is a `Usage`/`Example` block, which
  Google's guide explicitly wants.
- **Four lines is the ceiling for a paragraph.** Docstrings are read by people.
  Past four lines, break the thought into a bullet list or cut it. Never write
  a run-on of clauses strung together with semicolons, colons and dashes just to
  fit an explanation into one block — that is the shape to refactor, not to
  reflow. The ceiling is not a target: a connected thought stays one paragraph
  rather than being chopped into one-line fragments.
- Overridden `score` / `compute` carry their own docstring rather than inheriting
  the base one — each states what *this* metric does with the values.
- Test docstrings are per-module, not per-test: `D100`/`D104` are enforced in
  `tests/`, `D101`/`D102`/`D103`/`D107` are not. A test's name and its
  `parametrize` ids are its documentation; add a docstring only where they aren't
  enough (fixtures, builders, golden/property tests).

Enforced by `ruff` (`D` with `convention = "google"`, plus pydoclint
`DOC201`/`DOC402`/`DOC501` checking sections against the signature). The
`per-file-ignores` block named "Docstring migration" in `pyproject.toml` is the
remaining todo list — one line per layer, deleted as that layer is converted.
