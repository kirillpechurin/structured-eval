from __future__ import annotations

from typing import Any

from structured_eval.alignment import make_aligner
from structured_eval.metrics.base import FieldMetric, get_metric_class
from structured_eval.metrics.exact import ExactMatch
from structured_eval.model.config import (
    ArrayFieldConfig,
    ArrayStrategy,
    EvalConfig,
    ExtraKeysPolicy,
    FieldConfig,
    ObjectFieldConfig,
)
from structured_eval.model.context import EvalContext
from structured_eval.model.nodes.array_node import ArrayNode
from structured_eval.model.nodes.base import MISSING, EvalNode, _navigate
from structured_eval.model.nodes.object_node import ObjectNode
from structured_eval.model.nodes.scalar import ScalarNode


def _resolve_metric(spec: Any) -> FieldMetric:
    if isinstance(spec, str):
        instance = get_metric_class(spec)()
        assert isinstance(instance, FieldMetric)
        return instance
    assert isinstance(spec, FieldMetric)
    return spec


def _weight(cfg: Any) -> float:
    return getattr(cfg, "weight", 1.0) if cfg is not None else 1.0


class TreeBuilder:
    """Phase 1: build the EvalNode tree and compute leaf (field) metrics.

    ``build`` returns ``(root_node, warnings)``. Each node carries an
    ``actual``-side ``path`` and, when arrays reorder elements, a diverging
    ``expected_path`` so each side navigates its own index.
    """

    def __init__(self, context: EvalContext):
        self.context = context
        self.config: EvalConfig = context.config
        self.warnings: list[str] = []

    def build(self) -> tuple[EvalNode, list[str]]:
        root = self.node("$", "$", self.root_config())
        return root, self.warnings

    # ── config resolution ──────────────────────────────────────────────────

    def root_config(self) -> Any:
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

    def _child(self, path: str, key: str) -> str:
        return key if path in ("$", "") else f"{path}.{key}"

    def node(self, apath: str, epath: str, cfg: Any) -> Any:
        actual = self._value(self.context.actual, apath)
        expected = self._value(self.context.expected, epath)
        ref = expected if expected is not None else actual

        if isinstance(ref, dict):
            return self._object(apath, epath, cfg, actual, expected)
        if isinstance(ref, list):
            return self._array(apath, epath, cfg, actual, expected)
        return self._scalar(apath, epath, cfg)

    def _object(self, apath: str, epath: str, cfg: Any, actual: Any, expected: Any) -> ObjectNode:
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
                    f"[EXTRA_KEY] {self._child(apath, key)!r} not in expected "
                    "(ExtraKeysPolicy.IGNORE)"
                )

        fields = cfg.fields if isinstance(cfg, ObjectFieldConfig) else {}
        children: dict[str, Any] = {}
        matched: list[Any] = []
        for key in sorted(a_keys | e_keys):
            child = self.node(self._child(apath, key), self._child(epath, key), fields.get(key))
            children[key] = child
            if key in both:
                matched.append(child)
        for key in missing:
            self.warnings.append(f"[MISSING_FIELD] {self._child(apath, key)!r} absent in actual")

        return ObjectNode(
            path=apath,
            context=self.context,
            expected_path=epath if epath != apath else None,
            weight=_weight(cfg),
            matched=matched,
            missing=missing,
            spurious=spurious,
            children=children,
        )

    def _array(self, apath: str, epath: str, cfg: Any, actual: Any, expected: Any) -> ArrayNode:
        a_list = actual if isinstance(actual, list) else []
        e_list = expected if isinstance(expected, list) else []
        is_cfg = isinstance(cfg, ArrayFieldConfig)
        aligner = make_aligner(
            strategy=cfg.strategy if is_cfg else ArrayStrategy.BY_INDEX,
            params=cfg.params if is_cfg else None,
        )
        result = aligner.align(e_list, a_list)
        item_cfg = cfg.item if is_cfg else None
        items = [
            self.node(f"{apath}[{aidx}]", f"{epath}[{eidx}]", item_cfg)
            for eidx, aidx in result.matched
        ]
        return ArrayNode(
            path=apath,
            context=self.context,
            expected_path=epath if epath != apath else None,
            weight=_weight(cfg),
            match_result=result,
            items=items,
        )

    def _scalar(self, apath: str, epath: str, cfg: Any) -> ScalarNode:
        threshold = (
            cfg.threshold if isinstance(cfg, FieldConfig) and cfg.threshold is not None else 1.0
        )
        node = ScalarNode(
            path=apath,
            context=self.context,
            expected_path=epath if epath != apath else None,
            weight=_weight(cfg),
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
