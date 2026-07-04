# Contributing

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)

## Setup

```bash
git clone https://github.com/your-org/structured-eval
cd structured-eval
uv sync --extra dev
```

## Daily workflow

The `Makefile` wraps the common commands (run `make help` to list them):

```bash
make test        # uv run pytest
make lintcheck   # uv run ruff check structured_eval tests
make typecheck   # uv run mypy structured_eval tests
make check       # lintcheck + typecheck
make format      # ruff format + ruff --fix-only
```

Before opening a PR, `make check && make test` must be green. mypy runs in
`--strict` mode and the test suite is the source of truth for behaviour.

## Adding a dependency

```bash
uv add <package>          # runtime dependency
uv add --dev <package>    # dev-only
```

Optional features are lazy-imported behind extras (`yaml`, `fuzzy`,
`jsonschema`, `rules`, `diff`, `report`, `deepeval`, `langsmith`, `all`). Guard
any new optional import so the core stays installable without the extra.

## Project conventions

These are the essentials a contributor needs; each is self-contained here.

- **Layered dependencies point downward only:**
  `models ← metrics / alignment / formats / utils ← engine / reporting ←
  integrations / api`. A layer may import from layers to its left, never to its
  right. Never import upward.

- **Import structure.** The package root (`structured_eval/__init__.py`) exports
  **only** the three entrypoints (`evaluate` / `evaluate_batch` /
  `evaluate_consistency`). Everything else is imported one level down:
  - data models → `from structured_eval.models import ...`
  - metrics, base classes, the rule DSL → `from structured_eval.metrics import ...`
  - helpers → `from structured_eval.utils import ...`

  Do not add new names to the top-level `structured_eval/__init__.py`.

- **One metric = one module.** A metric lives in its own
  `structured_eval/metrics/<snake_name>.py` (a metric that needs helper code
  becomes a package instead). Never group metrics by node type. Declaring the
  class auto-registers it by `name` — the only extra step is registering it in
  `structured_eval/metrics/__init__.py`: add both the import **and** the
  `__all__` entry.

- **"Comparison is a metric."** There is no separate matcher with a precomputed
  similarity. A field metric compares `(node.actual, node.expected) → score`, and
  a node may carry several metrics at once. To run any metric, go through
  `MetricInvoker(metric)` (`on_node` / `on_values`) — never call a metric's
  `compute` / `score` directly.

- **Every node owns its metrics**, plus a `key_metric` that produces its
  representative score (default `MeanScore`). The root node's `key_metric` is what
  `report.score` reports. Access results via `report.field_scores[path]`,
  `report.metrics[name]`, and `report.score` — there is no `report.f1`.

- **Optional features are lazy-imported behind extras** (`yaml`, `fuzzy`,
  `jsonschema`, `rules`, `diff`, `report`, `deepeval`, `langsmith`, `all`). Guard
  any new optional import so the core stays installable without the extra.

- **Data models are pydantic v2** — use `model_dump` / `model_validate` for
  serialization.

- **Tests mirror the source tree** one-to-one, one file per cohesive unit. Style:
  flat parametrized functions, no test classes, table-driven, with `pytestmark`
  set once per file. New behaviour needs a test; coverage is gated in
  `pyproject.toml`.
