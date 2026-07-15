from __future__ import annotations

from typing import TYPE_CHECKING, Any

from structured_eval.alignment import make_aligner
from structured_eval.metrics.array_accuracy import ArrayAccuracy
from structured_eval.metrics.base import (
    AnyNodeMetric,
    ArrayMetric,
    BaseMetric,
    FieldMetric,
    GenericMetric,
    Metric,
    ObjectMetric,
    RootMetric,
    resolve_metric,
)
from structured_eval.metrics.exact import ExactMatch
from structured_eval.metrics.invoker import GENERIC_NODE_METHOD
from structured_eval.metrics.mean_score import MeanScore
from structured_eval.metrics.object_accuracy import ObjectAccuracy
from structured_eval.models.config import (
    AnyFieldConfig,
    ArrayFieldConfig,
    ArrayStrategy,
    EvalConfig,
    ExtraKeysPolicy,
    ObjectFieldConfig,
    weight_of,
)
from structured_eval.models.nodes.array_node import ArrayNode
from structured_eval.models.nodes.base import MISSING, EvalNode, navigate
from structured_eval.models.nodes.object_node import ObjectNode
from structured_eval.models.nodes.scalar import ScalarNode
from structured_eval.models.result import EvalWarning, WarningType

if TYPE_CHECKING:
    from structured_eval.models.context import EvalContext

# Metric a node falls back to when the user configured none of its type, so every
# node always carries at least one metric for its key_metric (MeanScore) to mean.
DEFAULT_SCALAR_METRIC = ExactMatch
DEFAULT_OBJECT_METRIC = ObjectAccuracy
DEFAULT_ARRAY_METRIC = ArrayAccuracy


class TreeBuilder:
    """Phase 1: build the EvalNode tree and resolve each node's metric list.

    ``build`` returns ``(root_node, warnings)``. This phase is purely
    structural: it shapes the tree, resolves which metrics apply to each node
    (cascading the config's global metrics by type and adding any per-node
    ``cfg.metrics``), and attaches them to ``node.metrics``. Computation happens
    later, uniformly, in ``MetricRunner``. Each node carries an ``actual``-side
    ``path`` and, when arrays reorder elements, a diverging ``expected_path`` so
    each side navigates its own index.
    """

    def __init__(self, context: EvalContext):
        self.context = context
        self.config: EvalConfig = context.config
        self.warnings: list[EvalWarning] = []
        self._globals = self._resolve_globals()

    def build(self) -> tuple[EvalNode, list[EvalWarning]]:
        root = self.node("$", "$", self.root_config())
        return root, self.warnings

    # ── config resolution ──────────────────────────────────────────────────

    def root_config(self) -> ObjectFieldConfig | ArrayFieldConfig | None:
        if self.config.root is not None:
            return self.config.root
        if self.config.fields:
            return ObjectFieldConfig(fields=dict(self.config.fields))
        return None

    @staticmethod
    def _applies_to(metric: BaseMetric, node_cls: type, is_root: bool) -> bool:
        """Whether ``metric`` should be resolved onto a node of ``node_cls``.

        Typed metrics match their node type (a ``RootMetric`` only at the root);
        an ``AnyNodeMetric`` matches every node; a ``GenericMetric`` matches iff
        it defines the node's ``compute_<kind>``.
        """
        if isinstance(metric, RootMetric):
            return is_root
        if isinstance(metric, AnyNodeMetric):
            return True
        if isinstance(metric, FieldMetric):
            return issubclass(node_cls, ScalarNode)
        if isinstance(metric, ObjectMetric):
            return issubclass(node_cls, ObjectNode)
        if isinstance(metric, ArrayMetric):
            return issubclass(node_cls, ArrayNode)
        if isinstance(metric, GenericMetric):
            method = GENERIC_NODE_METHOD.get(node_cls)
            return method is not None and hasattr(metric, method)
        return False

    def _resolve_globals(self) -> list[BaseMetric]:
        """The cascade set: ``config.metrics``, deduped by identity.

        ``key_metric`` is *not* cascaded here — it is each node's representative
        metric, resolved per node by ``_key_metric`` (and computed last).
        """
        out: list[BaseMetric] = []
        for spec in self.config.metrics:
            metric = resolve_metric(spec)
            if not any(metric is seen for seen in out):
                out.append(metric)
        return out

    def _resolve_metrics(
        self, path: str, node_cls: type, cfg: AnyFieldConfig | None, is_root: bool
    ) -> list[BaseMetric]:
        """Metrics for one node: applicable globals + this node's own (additive).

        Globals cascade by type (a ``RootMetric`` only at the root) and are
        silently filtered where they do not apply — cascading-by-type is
        intentional. Per-node ``cfg.metrics`` are *added* (not a replacement),
        deduped by identity, but an explicit assignment that cannot score this
        node's type is a configuration mistake and **raises** here (build time,
        before any metric runs) rather than being dropped.

        ``out`` only ever holds metrics applicable to this node; if it ends up
        empty the node gets the default for its type so every node always
        carries at least one metric for its ``key_metric`` to summarise (a
        different default is set by putting a metric of that type in
        ``config.metrics``, which cascades).
        """
        out: list[BaseMetric] = []

        def add(metric: BaseMetric) -> None:
            if self._applies_to(metric, node_cls, is_root) and not any(
                metric is s for s in out
            ):
                out.append(metric)

        for metric in self._globals:
            add(metric)
        for spec in getattr(cfg, "metrics", None) or []:
            metric = resolve_metric(spec)
            if not self._applies_to(metric, node_cls, is_root):
                raise ValueError(
                    f"Metric {metric.name!r} cannot score the "
                    f"{node_cls.__name__} field {path!r}."
                )
            add(metric)

        if not out:
            if issubclass(node_cls, ScalarNode):
                add(DEFAULT_SCALAR_METRIC())
            elif issubclass(node_cls, ObjectNode):
                add(DEFAULT_OBJECT_METRIC())
            elif issubclass(node_cls, ArrayNode):
                add(DEFAULT_ARRAY_METRIC())
        return out

    def _key_metric(
        self,
        path: str,
        node_cls: type,
        cfg: AnyFieldConfig | None,
        is_root: bool,
        metrics: list[BaseMetric],
    ) -> Metric[Any]:
        """The node's representative metric (computed last).

        Prefers an explicit ``cfg.key_metric``, then a distributable
        ``config.key_metric`` (each applied only where its type fits), else the
        default ``MeanScore`` (the mean of the node's own metrics).

        A *name string* is resolved against the node's already-resolved
        ``metrics`` first: an equally-named metric is **reused as-is** (same
        instance, same params, no duplicate computation). It is instantiated
        fresh only when the name is not already on the node.

        An explicit per-node ``cfg.key_metric`` that cannot score this node's
        type is a configuration mistake and **raises**; the global
        ``config.key_metric`` is distributable and stays silently filtered.
        """
        for spec, explicit in (
            (getattr(cfg, "key_metric", None), True),
            (self.config.key_metric, False),
        ):
            if spec is None:
                continue
            # Reuse an equally-named metric already on the node; else resolve fresh.
            metric = next(
                (m for m in metrics if isinstance(spec, str) and m.name == spec),
                None,
            ) or resolve_metric(spec)
            if self._applies_to(metric, node_cls, is_root):
                assert isinstance(metric, Metric)  # a key metric has compute()/score()
                return metric
            if explicit:
                raise ValueError(
                    f"Key metric {metric.name!r} cannot score the "
                    f"{node_cls.__name__} field {path!r}."
                )
        return MeanScore()

    # ── tree construction ────────────────────────────────────────────────

    def _value(self, doc: Any, path: str) -> Any:
        value = navigate(doc, path)
        return None if value is MISSING else value

    def _child(self, path: str, key: str) -> str:
        return key if path in ("$", "") else f"{path}.{key}"

    def node(self, apath: str, epath: str, cfg: AnyFieldConfig | None) -> EvalNode:
        actual = self._value(self.context.actual, apath)
        expected = self._value(self.context.expected, epath)
        ref = expected if expected is not None else actual

        if isinstance(ref, dict):
            return self._object(apath, epath, cfg, actual, expected)
        if isinstance(ref, list):
            return self._array(apath, epath, cfg, actual, expected)
        return self._scalar(apath, epath, cfg)

    def _object(
        self,
        apath: str,
        epath: str,
        cfg: AnyFieldConfig | None,
        actual: Any,
        expected: Any,
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
                path = self._child(apath, key)
                self.warnings.append(
                    EvalWarning(
                        type=WarningType.EXTRA_KEY,
                        path=path,
                        message=f"{path!r} not in expected (ExtraKeysPolicy.IGNORE)",
                    )
                )

        fields = cfg.fields if isinstance(cfg, ObjectFieldConfig) else {}
        children: dict[str, Any] = {}
        matched: list[Any] = []
        for key in sorted(a_keys | e_keys):
            child = self.node(
                self._child(apath, key), self._child(epath, key), fields.get(key)
            )
            children[key] = child
            if key in both:
                matched.append(child)
        for key in missing:
            path = self._child(apath, key)
            self.warnings.append(
                EvalWarning(
                    type=WarningType.MISSING_FIELD,
                    path=path,
                    message=f"{path!r} absent in actual",
                )
            )

        is_root = apath == "$"
        metrics = self._resolve_metrics(apath, ObjectNode, cfg, is_root)
        return ObjectNode(
            path=apath,
            context=self.context,
            expected_path=epath if epath != apath else None,
            weight=weight_of(cfg),
            metrics=metrics,
            key_metric=self._key_metric(apath, ObjectNode, cfg, is_root, metrics),
            threshold=self._threshold(cfg),
            matched=matched,
            missing=missing,
            spurious=spurious,
            children=children,
        )

    def _array(
        self,
        apath: str,
        epath: str,
        cfg: AnyFieldConfig | None,
        actual: Any,
        expected: Any,
    ) -> ArrayNode:
        a_list: list[Any] = actual if isinstance(actual, list) else []
        e_list: list[Any] = expected if isinstance(expected, list) else []
        if isinstance(cfg, ArrayFieldConfig):
            aligner = make_aligner(strategy=cfg.strategy, params=cfg.params)
            item_cfg = cfg.item
        else:
            aligner = make_aligner(strategy=ArrayStrategy.BY_INDEX, params=None)
            item_cfg = None
        result = aligner.align(e_list, a_list)
        # TODO: with no expected list (faithfulness / schema-only mode) there are
        # no matched pairs, so array elements get no nodes — value-on-actual
        # metrics (FieldFaithfulness) can't reach them. Materialize actual
        # elements directly in that mode. Roadmap.
        items = [
            self.node(f"{apath}[{aidx}]", f"{epath}[{eidx}]", item_cfg)
            for eidx, aidx in result.matched
        ]
        is_root = apath == "$"
        metrics = self._resolve_metrics(apath, ArrayNode, cfg, is_root)
        return ArrayNode(
            path=apath,
            context=self.context,
            expected_path=epath if epath != apath else None,
            weight=weight_of(cfg),
            metrics=metrics,
            key_metric=self._key_metric(apath, ArrayNode, cfg, is_root, metrics),
            threshold=self._threshold(cfg),
            match_result=result,
            items=items,
        )

    def _scalar(self, apath: str, epath: str, cfg: AnyFieldConfig | None) -> ScalarNode:
        is_root = apath == "$"
        metrics = self._resolve_metrics(apath, ScalarNode, cfg, is_root)
        return ScalarNode(
            path=apath,
            context=self.context,
            expected_path=epath if epath != apath else None,
            weight=weight_of(cfg),
            metrics=metrics,
            key_metric=self._key_metric(apath, ScalarNode, cfg, is_root, metrics),
            threshold=self._threshold(cfg),
        )

    @staticmethod
    def _threshold(cfg: AnyFieldConfig | None) -> float:
        threshold = getattr(cfg, "threshold", None)
        return float(threshold) if threshold is not None else 1.0
