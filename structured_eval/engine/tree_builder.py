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
# Overridable per node type via EvalConfig.default_<type>_metrics.
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
        self._defaults = self._resolve_defaults()

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

    @staticmethod
    def _resolve_metrics(specs: list[Any]) -> list[BaseMetric]:
        """Resolve a list of metric specs to instances."""
        return [resolve_metric(spec) for spec in specs]

    def _resolve_globals(self) -> list[BaseMetric]:
        """The cascade set: ``config.metrics``.

        ``key_metric`` is *not* cascaded here — it is each node's representative
        metric, resolved per node by ``_key_metric`` (and computed last).
        """
        return self._resolve_metrics(self.config.metrics)

    def _resolve_defaults(self) -> dict[type, list[BaseMetric]]:
        """The per-type fallback sets, resolved once and shared across nodes.

        ``config.default_<type>_metrics`` replaces the hard-coded constant for
        that node type; ``None`` keeps the constant. Unlike ``_globals`` these do
        not cascade — ``_node_metrics`` reaches for them only when a node would
        otherwise carry no metric at all, and type-checks them there like any
        other explicit assignment.
        """
        configured: dict[type, tuple[str, type[BaseMetric]]] = {
            ScalarNode: ("default_scalar_metrics", DEFAULT_SCALAR_METRIC),
            ObjectNode: ("default_object_metrics", DEFAULT_OBJECT_METRIC),
            ArrayNode: ("default_array_metrics", DEFAULT_ARRAY_METRIC),
        }
        out: dict[type, list[BaseMetric]] = {}
        for node_cls, (config_attr, fallback) in configured.items():
            specs = getattr(self.config, config_attr)
            out[node_cls] = self._resolve_metrics(specs) if specs else [fallback()]
        return out

    def _node_metrics(
        self, path: str, node_cls: type, cfg: AnyFieldConfig | None, is_root: bool
    ) -> list[BaseMetric]:
        """Metrics for one node: applicable globals + this node's own (additive).

        Globals cascade by type (a ``RootMetric`` only at the root) and are
        silently filtered where they do not apply — cascading-by-type is
        intentional. Per-node ``cfg.metrics`` are *added*, but an explicit
        assignment that cannot score this node's type is a configuration mistake
        and **raises** here (build time, before any metric runs) rather than
        being dropped.

        ``out`` only ever holds metrics applicable to this node; if it ends up
        empty the node falls back to the default set for its type, so every node
        always carries at least one metric for its ``key_metric`` to summarise.
        That set is ``config.default_<type>_metrics`` when configured, else the
        hard-coded constant, and is type-checked here the same way.

        A name is the key a metric's result lands under, so a node never carries
        two metrics sharing one — the second would silently overwrite the first.
        Across *layers* that is an override: this node's own metric displaces the
        equally-named global, which is how one field runs a stricter
        configuration (global ``Numeric(0.1)``, this field ``Numeric(0.001)``).
        Within *one* list it is ambiguous and **raises**; two configurations of
        one metric there need distinct names (``Numeric(0.01, name="strict")``).
        """
        out: list[BaseMetric] = []
        seen: set[str] = set()  # names already taken on this node

        duplicated_metric_error_msg = (
            "Metric {metric_name!r} is assigned twice to the "
            f"{node_cls.__name__} field {path!r}; a metric name must be "
            f"unique per node. Pass name= to tell two configurations of "
            f"one metric apart."
        )

        # The node's own metrics go first: they are the most specific layer, so
        # they claim their names before the cascade gets to add anything.
        for spec in getattr(cfg, "metrics", None) or []:
            metric = resolve_metric(spec)
            if not self._applies_to(metric, node_cls, is_root):
                raise ValueError(
                    f"Metric {metric.name!r} cannot score the "
                    f"{node_cls.__name__} field {path!r}."
                )
            if metric.name in seen:
                raise ValueError(
                    duplicated_metric_error_msg.format(metric_name=metric.name)
                )
            seen.add(metric.name)
            out.append(metric)

        own = set(seen)  # frozen before the cascade, to tell the two cases apart
        for metric in self._globals:
            if not self._applies_to(metric, node_cls, is_root):
                continue
            if metric.name in own:
                # This node redefines the metric — the specific layer wins over
                # the cascade, so the global is dropped rather than rejected.
                continue
            if metric.name in seen:
                # Not an override, then: two globals named alike, which is
                # ambiguous wherever they land together.
                raise ValueError(
                    duplicated_metric_error_msg.format(metric_name=metric.name)
                )
            seen.add(metric.name)
            out.append(metric)

        # Nothing applied — fall back so the node's key_metric has something to
        # summarise. `seen` is necessarily empty here (any own metric would have
        # filled `out`), so a clash can only be inside the default list itself.
        if not out:
            for metric in self._defaults[node_cls]:
                if not self._applies_to(metric, node_cls, is_root):
                    raise ValueError(
                        f"Metric {metric.name!r} cannot score the "
                        f"{node_cls.__name__} field {path!r}."
                    )
                if metric.name in seen:
                    raise ValueError(
                        duplicated_metric_error_msg.format(metric_name=metric.name)
                    )
                seen.add(metric.name)
                out.append(metric)

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
        metrics = self._node_metrics(apath, ObjectNode, cfg, is_root)
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
        metrics = self._node_metrics(apath, ArrayNode, cfg, is_root)
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
        metrics = self._node_metrics(apath, ScalarNode, cfg, is_root)
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
