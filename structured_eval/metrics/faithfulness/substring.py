from __future__ import annotations

from typing import Any

from structured_eval.model.config import EvalConfig


class SubstringFaithfulness:
    """L1 faithfulness: is each leaf value a substring of the source text?

    For every leaf value in ``actual``, checks whether its string form appears
    (case-insensitively) in ``source``. Fields whose top-level key has
    ``FieldConfig(derived=True)`` are skipped — derived values are computed from
    other fields and need not appear verbatim.

    Known limitation: only top-level ``derived`` flags are respected; nested
    derived fields require v0.2.
    """

    def compute(
        self, actual: dict[str, Any], source: str, config: EvalConfig
    ) -> tuple[float, list[str]]:
        """Return ``(score, hallucinated_fields)``.

        ``score`` is the fraction of checkable fields found in source (1.0 when
        nothing is checkable); ``hallucinated_fields`` lists the paths not found.
        """
        source_lower = source.lower()
        field_configs = config.fields or {}
        scores: dict[str, float] = {}
        self._collect(actual, source_lower, field_configs, prefix="", out=scores)

        if not scores:
            return 1.0, []
        hallucinated = [k for k, s in scores.items() if s == 0.0]
        return sum(scores.values()) / len(scores), hallucinated

    def _collect(
        self,
        obj: dict[str, Any],
        source_lower: str,
        field_configs: dict[str, Any],
        prefix: str,
        out: dict[str, float],
    ) -> None:
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else key
            # Derived check: only top-level keys are looked up in field_configs.
            top_key = path.split(".")[0].split("[")[0]
            fc = field_configs.get(top_key)
            if fc and getattr(fc, "derived", False):
                continue

            if isinstance(value, dict):
                self._collect(value, source_lower, field_configs, path, out)
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    item_path = f"{path}[{i}]"
                    if isinstance(item, dict):
                        self._collect(item, source_lower, field_configs, item_path, out)
                    elif item is not None:
                        out[item_path] = 1.0 if str(item).lower() in source_lower else 0.0
            elif value is not None:
                out[path] = 1.0 if str(value).lower() in source_lower else 0.0
