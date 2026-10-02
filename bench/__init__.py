"""The benchmark package: roster, runner, rating fit, and report.

Not installed with the library -- it lives at the repo root, not under ``src``
-- because it is a *user* of ``chessrl``, not part of it. The point of the
separation is that nothing in ``src/chessrl`` may import from here: the levels
must not know they are being benchmarked, or the benchmark stops being an
observation and becomes part of the thing observed.
"""
