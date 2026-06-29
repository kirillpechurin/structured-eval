"""Shared fixtures and helpers for the structured_eval test suite.

The suite mirrors the package layers (model ← metrics/alignment/formats/utils ←
engine/reporting ← integrations/api). Helpers here keep the engine-level tests
free of boilerplate; pure unit tests construct their objects directly.
"""

from collections.abc import Callable
from typing import Any

import pytest

from structured_eval import EvalConfig, EvalReport, evaluate
from structured_eval.engine.metric_runner import MetricRunner
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
    """Build and compute the EvalNode tree — for object/array/root metric tests.

    Runs ``MetricRunner`` so every node's ``metric_results`` (and hence each
    child's ``representative``) is populated, exactly as in the real engine; an
    aggregating metric called on the returned tree can read its children.
    """
    ctx = make_context(actual, expected, config, source=source)
    root, _warnings = TreeBuilder(ctx).build()
    MetricRunner().run(root)
    return root


@pytest.fixture
def tree_factory() -> Callable[..., EvalNode]:
    """Fixture exposing ``build_tree`` for object/array/root metric unit tests."""
    return build_tree


@pytest.fixture
def evaluate_one() -> Callable[..., EvalReport]:
    """Fixture exposing the ``run`` helper to tests that prefer injection."""
    return run


@pytest.fixture
def context_factory() -> Callable[..., EvalContext]:
    """Fixture exposing ``make_context`` for node/metric unit tests."""
    return make_context


# ── semantic assertions (exposed as fixtures; see tests/README.md) ───────────


def _assert_metric(report: EvalReport, name: str, value: float) -> None:
    """Assert a metric's representative value across the tree (``report.metrics``)."""
    actual = report.metrics[name].representative()
    assert actual == pytest.approx(value), f"metric {name!r}: expected {value}, got {actual}"


def _assert_field(report: EvalReport, path: str, score: float) -> None:
    """Assert one field's representative score (``report.field_scores[path].score``)."""
    fs = report.field_scores[path]
    assert fs.score == pytest.approx(score), f"field {path!r}: expected {score}, got {fs.score}"


@pytest.fixture
def assert_metric() -> Callable[[EvalReport, str, float], None]:
    """Semantic assertion: ``assert_metric(report, "object_f1", 0.5)``."""
    return _assert_metric


@pytest.fixture
def assert_field() -> Callable[[EvalReport, str, float], None]:
    """Semantic assertion: ``assert_field(report, "total", 0.0)``."""
    return _assert_field


# ── domain data builders ─────────────────────────────────────────────────────


def make_invoice(**overrides: Any) -> dict[str, Any]:
    """A canonical invoice document; pass overrides for the fields under test.

    Builders keep tests focused: ``make_invoice(total=99.0)`` shows *only* what
    differs from the baseline, instead of repeating the full literal.
    """
    base = {"id": "INV-001", "vendor": "Acme Corp", "total": 100.0, "status": "paid"}
    base.update(overrides)
    return base


INVOICE_SOURCE = "Invoice INV-001 from Acme Corp, total amount 100.0 USD, status paid"


@pytest.fixture
def invoice_builder() -> Callable[..., dict[str, Any]]:
    """Fixture exposing ``make_invoice`` for tests that prefer injection."""
    return make_invoice


@pytest.fixture
def invoice_pair() -> tuple[dict[str, Any], dict[str, Any]]:
    """A small invoice document with one wrong field (total)."""
    return make_invoice(total=99.0), make_invoice()


@pytest.fixture
def invoice_source() -> str:
    return INVOICE_SOURCE
