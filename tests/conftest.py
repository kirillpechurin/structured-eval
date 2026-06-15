"""Shared fixtures and helpers for the structured_eval test suite.

The suite mirrors the package layers (model ← metrics/alignment/formats/utils ←
engine/reporting ← integrations/api). Helpers here keep the engine-level tests
free of boilerplate; pure unit tests construct their objects directly.
"""

from __future__ import annotations

from typing import Any

import pytest

from structured_eval import EvalConfig, EvalReport, evaluate
from structured_eval.engine.tree_builder import TreeBuilder
from structured_eval.model.context import EvalContext
from structured_eval.model.nodes.base import EvalNode
from structured_eval.utils.flatten import flatten

# ── engine helpers ──────────────────────────────────────────────────────────


def run(
    actual: Any,
    expected: Any = None,
    config: EvalConfig | None = None,
    *,
    source: str | None = None,
) -> EvalReport:
    """Evaluate one document through the public API (single-document shape)."""
    report = evaluate(actual, expected, config=config, source=source)
    assert isinstance(report, EvalReport)
    return report


def make_context(
    actual: Any,
    expected: Any = None,
    config: EvalConfig | None = None,
    *,
    source: str | None = None,
) -> EvalContext:
    """Build an EvalContext directly — for unit-testing nodes and metrics."""
    cfg = config or EvalConfig()
    return EvalContext(
        actual=actual,
        expected=expected,
        source=source,
        flat_actual=flatten(actual) if isinstance(actual, (dict, list)) else {},
        flat_expected=flatten(expected) if isinstance(expected, (dict, list)) else {},
        config=cfg,
    )


def build_tree(
    actual: Any,
    expected: Any = None,
    config: EvalConfig | None = None,
    *,
    source: str | None = None,
) -> EvalNode:
    """Build the EvalNode tree (with leaf metrics) — for object/array metric tests."""
    ctx = make_context(actual, expected, config, source=source)
    root, _warnings = TreeBuilder(ctx).build()
    return root


@pytest.fixture
def tree_factory():
    """Fixture exposing ``build_tree`` for object/array/root metric unit tests."""
    return build_tree


@pytest.fixture
def evaluate_one():
    """Fixture exposing the ``run`` helper to tests that prefer injection."""
    return run


@pytest.fixture
def context_factory():
    """Fixture exposing ``make_context`` for node/metric unit tests."""
    return make_context


# ── data fixtures ───────────────────────────────────────────────────────────


@pytest.fixture
def invoice_pair() -> tuple[dict, dict]:
    """A small invoice document with one wrong field (total)."""
    actual = {
        "id": "INV-001",
        "vendor": "Acme Corp",
        "total": 99.0,
        "status": "paid",
    }
    expected = {
        "id": "INV-001",
        "vendor": "Acme Corp",
        "total": 100.0,
        "status": "paid",
    }
    return actual, expected


@pytest.fixture
def invoice_source() -> str:
    return "Invoice INV-001 from Acme Corp, total amount 100.0 USD, status paid"
