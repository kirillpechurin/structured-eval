.DEFAULT_GOAL := help
.PHONY: format typecheck lint-check check test

.PHONY: format
format: ## Format code with ruff
	uv run ruff format structured_eval tests
	uv run ruff check structured_eval tests --fix-only

.PHONY: lintcheck
lintcheck: ## Check code with ruff
	uv run ruff check structured_eval tests

.PHONY: typecheck
typecheck: ## Check types with mypy
	uv run mypy structured_eval tests

.PHONY: check
check: ## Quick command to check code
	make lintcheck
	make typecheck

.PHONY: test
test: ## Run tests
	uv run pytest

.PHONY: help
help: ## Display this help screen
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)