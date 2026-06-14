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

from typing import Any, Callable

from structured_eval._evaluate import evaluate
from structured_eval.core.config import EvalConfig

Extractor = Callable[[Any], Any]


def _outputs(obj: Any) -> Any:
    """Default extraction: the ``outputs`` payload of a run/example."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get("outputs", obj)
    return getattr(obj, "outputs", obj)


def structured_evaluator(
    config: EvalConfig | None = None,
    *,
    key: str = "structured_eval",
    threshold: float = 0.5,
    extract_actual: Extractor | None = None,
    extract_expected: Extractor | None = None,
) -> Callable[[Any, Any], dict]:
    """Build a LangSmith evaluator that scores structured outputs field-by-field.

    ``key`` is the feedback key recorded in LangSmith; ``threshold`` decides the
    boolean ``score`` is compared against only for the ``comment`` — LangSmith
    stores the numeric ``report.score`` itself.
    """
    cfg = config or EvalConfig()
    get_actual = extract_actual or _outputs
    get_expected = extract_expected or _outputs

    def evaluator(run: Any, example: Any) -> dict:
        from structured_eval.integrations._adapter import verdict

        report = evaluate(get_actual(run), get_expected(example), cfg)
        score, _success, reason = verdict(report, threshold)
        return {"key": key, "score": score, "comment": reason}

    evaluator.__name__ = key
    return evaluator
