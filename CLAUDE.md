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
make test        # uv run pytest
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
models ← metrics / alignment / formats / utils ← engine / reporting ← integrations / api
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
  `rules`, `diff`, `report`, `deepeval`, `langsmith`, `all`). Guard any new optional
  import so the core stays installable without the extra.
- **Tests mirror the source tree** one-to-one, one file per cohesive unit. Style:
  flat parametrized functions, no test classes, table-driven; `pytestmark` set once
  per file. New behaviour needs a test; coverage is gated in `pyproject.toml`.
