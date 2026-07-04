.DEFAULT_GOAL := help
.PHONY: format format-check lintcheck typecheck check test test-cov help

.PHONY: format
format: ## Format code with ruff
	uv run ruff format structured_eval tests
	uv run ruff check structured_eval tests --fix-only

.PHONY: format-check
format-check: ## Check formatting without modifying files
	uv run ruff format --check structured_eval tests

.PHONY: lintcheck
lintcheck: ## Check code with ruff
	uv run ruff check structured_eval tests

.PHONY: typecheck
typecheck: ## Check types with mypy
	uv run mypy structured_eval tests

.PHONY: check
check: ## Quick command to check code
	make lintcheck
	make format-check
	make typecheck

.PHONY: test
test: ## Run tests
	uv run pytest

.PHONY: test-cov
test-cov: ## Run tests with coverage (html, xml for Codecov, terminal)
	uv run pytest --cov=structured_eval --cov-report=html:htmlcov --cov-report=xml:coverage.xml --cov-report=term-missing

.PHONY: help
help: ## Display this help screen
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)
