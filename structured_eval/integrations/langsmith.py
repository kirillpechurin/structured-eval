"""LangSmith adapter: structured-eval as an evaluator function.

Usage (requires ``structured-eval[langsmith]``)::

    from langsmith import evaluate
    from structured_eval.integrations.langsmith import structured_evaluator

    evaluator = structured_evaluator(config=cfg, threshold=0.85)
    evaluate(target, data=dataset, evaluators=[evaluator])

The returned callable follows LangSmith's ``(run, example) -> dict`` contract and
emits a single feedback key with ``report.score`` plus a ``comment`` summarising
failures. By default the actual output is read from ``run.outputs`` and the
reference from ``example.outputs``; pass ``extract_actual``/``extract_expected``
to point at a nested field or adapt a different object shape.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from structured_eval.api import evaluate
from structured_eval.integrations._adapter import verdict
from structured_eval.model.config import EvalConfig

Extractor = Callable[[Any], Any]


def _outputs(obj: Any) -> Any:
    """Default extraction: the ``outputs`` payload of a run/example."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get("outputs", obj)
    return getattr(obj, "outputs", obj)


class StructuredEvaluator:
    """A LangSmith evaluator that scores structured outputs field-by-field.

    Instances are callable with LangSmith's ``(run, example) -> dict`` contract.
    ``key`` is the feedback key recorded in LangSmith; ``threshold`` decides the
    boolean only for the ``comment`` — LangSmith stores the numeric
    ``report.score`` itself. ``extract_actual`` / ``extract_expected`` adapt the
    run/example shape (default: their ``outputs`` payload).
    """

    def __init__(
        self,
        config: EvalConfig | None = None,
        *,
        key: str = "structured_eval",
        threshold: float = 0.5,
        extract_actual: Extractor | None = None,
        extract_expected: Extractor | None = None,
    ) -> None:
        self.config = config or EvalConfig()
        self.key = key
        self.threshold = threshold
        self._get_actual = extract_actual or _outputs
        self._get_expected = extract_expected or _outputs
        self.__name__ = key

    def __call__(self, run: Any, example: Any) -> dict:
        report = evaluate(self._get_actual(run), self._get_expected(example), self.config)
        score, _success, reason = verdict(report, self.threshold)
        return {"key": self.key, "score": score, "comment": reason}


def structured_evaluator(
    config: EvalConfig | None = None,
    *,
    key: str = "structured_eval",
    threshold: float = 0.5,
    extract_actual: Extractor | None = None,
    extract_expected: Extractor | None = None,
) -> StructuredEvaluator:
    """Convenience factory returning a ``StructuredEvaluator`` instance."""
    return StructuredEvaluator(
        config,
        key=key,
        threshold=threshold,
        extract_actual=extract_actual,
        extract_expected=extract_expected,
    )
