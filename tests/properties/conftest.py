"""Lightweight, dependency-free property-based generators.

The project pins no third-party PBT library (no hypothesis), so we roll a tiny
seeded generator instead. Tests parametrize over ``SEEDS`` and build a fresh
``random.Random(seed)`` per case: every failure is reproducible from its seed
(printed in the test id), and the corpus is deterministic across runs/CI.

These produce the messy inputs an LLM actually emits — mixed types, ``None``,
nested dicts/lists, currency-formatted numbers — which is exactly where a
scoring framework must stay total (never raise) and bounded (stay in [0, 1]).
"""

import random
import string
from typing import Any

# A fixed corpus size keeps the suite fast while still exercising hundreds of
# distinct shapes across all property tests. Bump for a deeper local sweep.
SEEDS = list(range(60))


def _word(rng: random.Random) -> str:
    n = rng.randint(1, 8)
    return "".join(rng.choice(string.ascii_letters + " ") for _ in range(n))


def random_str(rng: random.Random) -> str:
    """A short string, occasionally with leading/trailing/inner whitespace."""
    base = _word(rng)
    if rng.random() < 0.3:
        base = f"  {base}\t{_word(rng)}  "
    return base


def random_number(rng: random.Random) -> Any:
    """An int or float, sometimes rendered as a currency/accounting string."""
    val = rng.choice(
        [
            rng.randint(-1000, 1000),
            round(rng.uniform(-1000, 1000), 2),
        ]
    )
    if rng.random() < 0.25:  # lenient-parser fodder: "$1,234.50", "(123)"
        if val < 0:
            return f"({abs(val):,})"
        return f"${val:,}"
    return val


def random_scalar(rng: random.Random) -> Any:
    """One JSON scalar: str / int / float / bool / None, biased toward values."""
    pick = rng.random()
    if pick < 0.35:
        return random_str(rng)
    if pick < 0.7:
        return random_number(rng)
    if pick < 0.85:
        return rng.choice([True, False])
    return None


def random_json(rng: random.Random, depth: int = 3) -> Any:
    """A random JSON document: nested dicts/lists down to ``depth``."""
    if depth <= 0 or rng.random() < 0.4:
        return random_scalar(rng)
    if rng.random() < 0.5:
        n = rng.randint(0, 4)
        keys = rng.sample(string.ascii_lowercase, k=min(n, 26))
        return {k: random_json(rng, depth - 1) for k in keys}
    n = rng.randint(0, 4)
    return [random_json(rng, depth - 1) for _ in range(n)]


def random_document(rng: random.Random, depth: int = 3) -> dict[str, Any]:
    """A random JSON *document* — always an object root (what ``evaluate`` accepts).

    Scalar/array roots are a separate, partly-unsupported path; document-shaped
    inputs keep these tests focused on the report machinery rather than root-kind
    edge cases.
    """
    n = rng.randint(1, 5)
    keys = rng.sample(string.ascii_lowercase, k=min(n, 26))
    return {k: random_json(rng, depth - 1) for k in keys}
