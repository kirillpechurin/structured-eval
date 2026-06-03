.PHONY: fmt check

fmt:
	uv run ruff format .
	uv run ruff check . --fix-only

lint:
	uv run ruff check .
	uv run mypy structured_eval
