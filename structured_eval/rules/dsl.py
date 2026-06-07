from __future__ import annotations

import ast
import operator
import re
from typing import Any, Callable

from structured_eval.core.result import RuleResult

# Matches a bare JSONPath like "$.field" or "$.nested.child"
_PLAIN_PATH_RE = re.compile(r'^\$(?:\.[a-zA-Z_][a-zA-Z0-9_]*)+$')
# Matches any JSONPath fragment inside a larger expression
_PATH_IN_EXPR_RE = re.compile(r'\$(?:\.[a-zA-Z_][a-zA-Z0-9_]*)+')

_ARITH_OPS: dict[type[ast.operator], Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}


# ── jsonpath-ng lazy import ────────────────────────────────────────────────────


def _ensure_jsonpath() -> None:
    try:
        import jsonpath_ng  # noqa: F401
    except ImportError:
        raise ImportError(
            "jsonpath-ng is required for the Rule DSL. "
            "Install it with: pip install 'structured-eval[rules]'"
        )


def _resolve_path(path: str, document: dict[str, Any]) -> Any:
    _ensure_jsonpath()
    from jsonpath_ng import parse  # type: ignore[import-untyped]

    matches = parse(path).find(document)
    if not matches:
        raise KeyError(f"Path {path!r} not found in document")
    return matches[0].value


# ── Arithmetic expression evaluator ───────────────────────────────────────────


def _eval_arithmetic(expr: str, document: dict[str, Any]) -> Any:
    """Resolve JSONPath fragments inside expr, then evaluate safe arithmetic."""

    def _replace(m: re.Match[str]) -> str:
        return repr(_resolve_path(m.group(), document))

    resolved = _PATH_IN_EXPR_RE.sub(_replace, expr)
    tree = ast.parse(resolved, mode="eval")
    return _eval_node(tree.body)


def _eval_node(node: ast.expr) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval_node(node.operand)  # type: ignore[operator]
    if isinstance(node, ast.BinOp):
        op_fn = _ARITH_OPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        return op_fn(_eval_node(node.left), _eval_node(node.right))
    raise ValueError(f"Unsupported expression node: {type(node).__name__}")


# ── CustomRule ────────────────────────────────────────────────────────────────


class _CustomRule:
    """Returned by Rule.custom(). Evaluates an arbitrary function over a document."""

    def __init__(self, fn: Callable[[dict[str, Any]], bool], *, name: str = "") -> None:
        self._fn = fn
        self._name = name

    @property
    def name(self) -> str:
        return self._name or "custom"

    def evaluate(self, document: dict[str, Any]) -> RuleResult:
        try:
            passed = bool(self._fn(document))
        except Exception as exc:
            return RuleResult(name=self.name, passed=False, message=str(exc))
        return RuleResult(
            name=self.name,
            passed=passed,
            message="" if passed else "custom rule failed",
        )


# ── Rule ──────────────────────────────────────────────────────────────────────


class Rule:
    """JSONPath-based document constraint.

    Chain a comparison method to create a bound rule, then pass it to
    EvalConfig.rules or call evaluate() directly.

    Examples::

        Rule("$.status").eq("paid")
        Rule("$.total").gte(0)
        Rule("$.total").eq("$.subtotal + $.tax")
        Rule("$.currency").in_(["USD", "EUR"])
        Rule.custom(lambda doc: doc["amount"] > 0, name="positive_amount")
    """

    def __init__(self, path: str, *, name: str = "") -> None:
        self._path = path
        self._name = name
        self._op: str | None = None
        self._rhs: Any = None

    # ── Builder ───────────────────────────────────────────────────────────────

    def _bind(self, op: str, rhs: Any) -> Rule:
        r = Rule(self._path, name=self._name)
        r._op = op
        r._rhs = rhs
        return r

    def eq(self, rhs: Any) -> Rule:
        return self._bind("eq", rhs)

    def lt(self, rhs: Any) -> Rule:
        return self._bind("lt", rhs)

    def gt(self, rhs: Any) -> Rule:
        return self._bind("gt", rhs)

    def lte(self, rhs: Any) -> Rule:
        return self._bind("lte", rhs)

    def gte(self, rhs: Any) -> Rule:
        return self._bind("gte", rhs)

    def in_(self, collection: Any) -> Rule:
        return self._bind("in", collection)

    @classmethod
    def custom(cls, fn: Callable[[dict[str, Any]], bool], *, name: str = "") -> _CustomRule:
        """Wrap an arbitrary function as a rule.

        Args:
            fn: Callable(document) -> bool. Receives the full document dict.
            name: Human-readable name shown in reports.
        """
        return _CustomRule(fn=fn, name=name)

    # ── Evaluation ────────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        if self._name:
            return self._name
        if self._op is not None:
            rhs_str = self._rhs if isinstance(self._rhs, str) else repr(self._rhs)
            return f"{self._path} {self._op} {rhs_str}"
        return self._path

    def evaluate(self, document: dict[str, Any]) -> RuleResult:
        if self._op is None:
            raise ValueError(
                f"Rule {self._path!r} has no comparison — call .eq(), .lt(), etc."
            )

        try:
            lhs = _resolve_path(self._path, document)
        except (KeyError, ImportError) as exc:
            return RuleResult(name=self.name, passed=False, message=str(exc))

        try:
            rhs = self._resolve_rhs(self._rhs, document)
        except Exception as exc:
            return RuleResult(name=self.name, passed=False, message=str(exc))

        try:
            passed = self._compare(lhs, rhs)
        except TypeError as exc:
            return RuleResult(name=self.name, passed=False, message=str(exc))

        msg = "" if passed else f"{self._path!r} ({lhs!r}) does not satisfy {self._op}({rhs!r})"
        return RuleResult(name=self.name, passed=passed, message=msg)

    def _resolve_rhs(self, rhs: Any, document: dict[str, Any]) -> Any:
        if isinstance(rhs, str) and "$" in rhs:
            stripped = rhs.strip()
            if _PLAIN_PATH_RE.match(stripped):
                return _resolve_path(stripped, document)
            return _eval_arithmetic(stripped, document)
        return rhs

    def _compare(self, lhs: Any, rhs: Any) -> bool:
        if self._op == "eq":
            return bool(lhs == rhs)
        if self._op == "lt":
            return bool(lhs < rhs)
        if self._op == "gt":
            return bool(lhs > rhs)
        if self._op == "lte":
            return bool(lhs <= rhs)
        if self._op == "gte":
            return bool(lhs >= rhs)
        if self._op == "in":
            return lhs in rhs
        raise ValueError(f"Unknown operator: {self._op!r}")
