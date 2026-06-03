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

**Run tests:**
```bash
uv run pytest
```

**Lint:**
```bash
uv run ruff check .
```

**Format:**
```bash
uv run ruff format .
```

**Type check:**
```bash
uv run mypy structured_eval
```

## Adding a dependency

```bash
uv add <package>          # runtime dependency
uv add --dev <package>    # dev-only
```
