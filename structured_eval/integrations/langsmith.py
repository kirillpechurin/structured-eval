"""LangSmith adapter: structured-eval as an evaluator function.

`structured_evaluator` returns a callable following LangSmith's
`(run, example) -> dict` contract, so a field-level evaluation is recorded as
one feedback entry. Requires `structured-eval[langsmith]` only to run the host
side; the evaluator itself is pure structured-eval.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from structured_eval.api import evaluate
from structured_eval.integrations._adapter import verdict
from structured_eval.models.config import EvalConfig
from structured_eval.models.result import EvalReport

Extractor = Callable[[Any], Any]


def _outputs(obj: Any) -> Any:
    """Default extraction: the `outputs` payload of a run/example."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get("outputs", obj)
    return getattr(obj, "outputs", obj)


class StructuredEvaluator:
    """A LangSmith evaluator that scores structured outputs field-by-field.

    Instances are callable with LangSmith's `(run, example) -> dict` contract.
    LangSmith stores the numeric `report.score` itself, so `threshold` shapes
    only the `comment`: it decides which fields are called out as failures.

    Attributes:
        config: The configuration each call evaluates under.
        key: The feedback key the score is recorded under in LangSmith.
        threshold: The score a document must reach to count as a pass.

    Example:
        >>> from langsmith import evaluate as langsmith_evaluate  # doctest: +SKIP
        >>> from structured_eval.integrations.langsmith import StructuredEvaluator
        >>> evaluator = StructuredEvaluator(threshold=0.85)
        >>> feedback = evaluator({"outputs": {"status": "paid"}},
        ...                      {"outputs": {"status": "due"}})
        >>> feedback["key"], feedback["score"]
        ('structured_eval', 0.0)
        >>> langsmith_evaluate(
        ...     target,
        ...     data=dataset,
        ...     evaluators=[evaluator]
        ... )  # doctest: +SKIP
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
        """Configure the evaluator; nothing is evaluated until it is called.

        Args:
            config: Field configuration, metrics and policies. Defaults to
                `EvalConfig()`, which compares every scalar with `ExactMatch`.
            key: The feedback key to record the score under.
            threshold: The score a document must reach to count as a pass.
            extract_actual: How to pull the document out of a run. Defaults to
                its `outputs` payload.
            extract_expected: How to pull the reference out of an example.
                Defaults to its `outputs` payload.
        """
        self.config = config or EvalConfig()
        self.key = key
        self.threshold = threshold
        self._get_actual = extract_actual or _outputs
        self._get_expected = extract_expected or _outputs
        self.__name__ = key

    def __call__(self, run: Any, example: Any) -> dict[str, Any]:
        """Score one run against its example.

        Args:
            run: The run whose output is under evaluation.
            example: The dataset example holding the reference.

        Returns:
            The LangSmith feedback entry: the `key`, the document `score`
            (`None` without ground truth) and a `comment` naming what failed.
        """
        report = evaluate(
            self._get_actual(run), self._get_expected(example), self.config
        )
        assert isinstance(report, EvalReport)  # single-document evaluation
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
    """Build a `StructuredEvaluator`, for callers who prefer a function.

    Args:
        config: Field configuration, metrics and policies. Defaults to
            `EvalConfig()`, which compares every scalar with `ExactMatch`.
        key: The feedback key to record the score under.
        threshold: The score a document must reach to count as a pass.
        extract_actual: How to pull the document out of a run. Defaults to its
            `outputs` payload.
        extract_expected: How to pull the reference out of an example. Defaults
            to its `outputs` payload.

    Returns:
        An evaluator callable with LangSmith's `(run, example) -> dict`
        contract.

    Example:
        >>> from langsmith import evaluate as langsmith_evaluate  # doctest: +SKIP
        >>> from structured_eval.integrations.langsmith import structured_evaluator
        >>> evaluator = structured_evaluator(key="invoice_fields", threshold=0.85)
        >>> evaluator({"outputs": {"status": "paid"}},
        ...           {"outputs": {"status": "paid"}})["score"]
        1.0
        >>> langsmith_evaluate(
        ...     target,
        ...     data=dataset,
        ...     evaluators=[evaluator]
        ... )  # doctest: +SKIP
    """
    return StructuredEvaluator(
        config,
        key=key,
        threshold=threshold,
        extract_actual=extract_actual,
        extract_expected=extract_expected,
    )
