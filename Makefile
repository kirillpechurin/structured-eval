.PHONY: fmt check

fmt:
	uv run ruff format .
	uv run ruff check . --fix

lint:
	uv run ruff check .
	uv run mypy structured_eval
