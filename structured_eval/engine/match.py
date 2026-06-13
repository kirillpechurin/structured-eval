from __future__ import annotations

from typing import Any

from structured_eval.core.config import (
    ArrayFieldConfig,
    EvalConfig,
    ExtraKeysPolicy,
    FieldConfig,
    ObjectFieldConfig,
)
from structured_eval.core.context import EvalContext
from structured_eval.metrics.field.exact import ExactMatch
from structured_eval.metrics.protocol import FieldMetric, get_metric_class
from structured_eval.nodes.base import MISSING, _navigate
from structured_eval.nodes.object_node import ObjectNode
from structured_eval.nodes.scalar import ScalarNode


def _resolve_metric(spec: Any) -> FieldMetric:
    return get_metric_class(spec)() if isinstance(spec, str) else spec


def build_tree(context: EvalContext) -> tuple[Any, list[str]]:
    """Phase 1: build the EvalNode tree and compute leaf (field) metrics.

    Returns ``(root_node, warnings)``. Arrays are not handled yet (Stage 7):
    a list value is treated as a scalar and a warning is recorded.
    """
    builder = _TreeBuilder(context)
    root = builder.node("$", builder.root_config())
    return root, builder.warnings


class _TreeBuilder:
    def __init__(self, context: EvalContext):
        self.context = context
        self.config: EvalConfig = context.config
        self.warnings: list[str] = []

    # ── config resolution ──────────────────────────────────────────────────

    def root_config(self) -> ObjectFieldConfig | ArrayFieldConfig | None:
        if self.config.root is not None:
            return self.config.root
        if self.config.fields:
            return ObjectFieldConfig(fields=dict(self.config.fields))
        return None

    def _field_metrics(self, cfg: Any) -> list[FieldMetric]:
        if isinstance(cfg, FieldConfig) and cfg.metrics is not None:
            specs = cfg.metrics
        elif self.config.default_metrics is not None:
            specs = self.config.default_metrics
        else:
            specs = [ExactMatch()]
        return [_resolve_metric(s) for s in specs]

    def _key_metric(self, cfg: Any) -> FieldMetric | None:
        if isinstance(cfg, FieldConfig) and cfg.key_metric is not None:
            return _resolve_metric(cfg.key_metric)
        return None

    # ── tree construction ────────────────────────────────────────────────

    def _value(self, doc: Any, path: str) -> Any:
        value = _navigate(doc, path)
        return None if value is MISSING else value

    def node(self, path: str, cfg: Any) -> Any:
        actual = self._value(self.context.actual, path)
        expected = self._value(self.context.expected, path)
        ref = expected if expected is not None else actual

        if isinstance(ref, dict):
            return self._object(path, cfg, actual, expected)
        if isinstance(ref, list):
            self.warnings.append(
                f"[ARRAY_UNSUPPORTED] {path!r}: arrays land in Stage 7, "
                "compared as a scalar for now"
            )
        return self._scalar(path, cfg)

    def _child_path(self, path: str, key: str) -> str:
        return key if path in ("$", "") else f"{path}.{key}"

    def _object(
        self, path: str, cfg: Any, actual: Any, expected: Any
    ) -> ObjectNode:
        a_keys = set(actual) if isinstance(actual, dict) else set()
        e_keys = set(expected) if isinstance(expected, dict) else set()
        both = a_keys & e_keys
        missing = sorted(e_keys - a_keys)  # in expected only → FN
        extra = sorted(a_keys - e_keys)  # in actual only → FP (subject to policy)

        if self.config.extra_keys == ExtraKeysPolicy.PENALIZE:
            spurious = extra
        else:
            spurious = []
            for key in extra:
                self.warnings.append(
                    f"[EXTRA_KEY] {self._child_path(path, key)!r} not in expected "
                    "(ExtraKeysPolicy.IGNORE)"
                )

        fields = cfg.fields if isinstance(cfg, ObjectFieldConfig) else {}
        children: dict[str, Any] = {}
        matched: list[Any] = []
        for key in sorted(a_keys | e_keys):
            child = self.node(self._child_path(path, key), fields.get(key))
            children[key] = child
            if key in both:
                matched.append(child)
        for key in missing:
            self.warnings.append(
                f"[MISSING_FIELD] {self._child_path(path, key)!r} absent in actual"
            )

        return ObjectNode(
            path=path,
            context=self.context,
            matched=matched,
            missing=missing,
            spurious=spurious,
            children=children,
        )

    def _scalar(self, path: str, cfg: Any) -> ScalarNode:
        threshold = (
            cfg.threshold
            if isinstance(cfg, FieldConfig) and cfg.threshold is not None
            else 1.0
        )
        node = ScalarNode(
            path=path,
            context=self.context,
            key_metric=self._key_metric(cfg),
            threshold=threshold,
        )
        for metric in self._field_metrics(cfg):
            result = metric.compute(node)
            if isinstance(result, dict):
                node.metric_results.update(result)
            else:
                node.metric_results[metric.name] = result
        return node
