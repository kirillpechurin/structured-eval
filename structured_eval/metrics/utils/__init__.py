"""Utilities shared across metric implementations (metric-layer only).

Clearly-scoped modules:

* ``calculate`` — the precision / recall / F1 arithmetic (and the ``MatchMode``
  hard/soft enum) used by every P/R/F1 metric, object and array alike.
* ``object_utils`` — turning an object's matched fields into the
  ``(score, threshold)`` pairs that ``calculate.prf_counts`` consumes.
* ``array`` — the same for an array's aligned items, plus missing/spurious counts.
"""
