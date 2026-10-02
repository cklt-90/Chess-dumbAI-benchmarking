"""Turning a BenchResult into something a human reads.

Design constraints, in priority order
-------------------------------------
1. **The raw matrix is always printed.** The fitted rating is a *model* of the
   matrix and can be wrong (see ``rating.py`` on non-transitivity). The matrix
   is the data and is never summarised away -- a reader who distrusts the fit
   must always be able to check it against the counts.
2. **Evidence quality travels with every number.** A rating fitted from two
   games and one fitted from two hundred look identical as bare floats. Every
   table row therefore carries its game count, its decisive-game count and its
   standard error, and rows with too few games to be meaningful are flagged
   rather than quietly ranked.
3. **Unfinished games are visible and excluded.** They are not losses, not
   draws, and not evidence, so they get their own column so a reader can see
   where the measurement thinned out.
4. **Plain text, fixed width.** This is a benchmark harness, not a dashboard. The
   report has to survive being pasted into a commit message or a terminal at 80
   columns. No colour, no box drawing beyond ASCII.
"""

from __future__ import annotations

from bench.rating import FitResult, Rating, fit_bradley_terry
from bench.runner import BenchResult

# The name of the all-seams-off L5 variant, matching ``levels.ablation_roster``.
# Held as a constant so the report does not have to pattern-match on substrings
# to find the reference row.
_ABLATION_BASELINE = "L5-ab-blind"


def _truncate(text: str, width: int) -> str:
    return text if len(text) <= width else text[: width - 3] + "..."


def format_matrix(bench: BenchResult, width: int = 12) -> str:
    """The raw win/loss grid, both triangles filled in."""
    grid = bench.matrix()
    name_w = max(len(n) for n in bench.names) if bench.names else 4
    lines = []
    header = " " * (name_w + 2) + "".join(
        f"{_truncate(n, width):>{width}s}" for n in bench.names
    )
    lines.append(header)
    lines.append("-" * len(header))
    for i, n in enumerate(bench.names):
        row = f"{n:>{name_w}s}  " + "".join(
            f"{_truncate(grid[i][j], width):>{width}s}" for j in range(len(bench.names))
        )
        lines.append(row)
    lines.append("")
    lines.append("cells are +wins-losses=draws from the row entrant's point of "
                 "view; (uN) = N unfinished, excluded from all ratings")
    return "\n".join(lines)


def format_ratings(fit: FitResult, min_games: int = 4) -> str:
    """The fitted table, in descending rating order.

    ``min_games`` is the threshold below which a rating is marked provisional.
    It is not a filter: a provisional rating still appears, because dropping it
    would hide evidence, but it is flagged so nobody quotes it as a result.
    """
    if not fit.ratings:
        return "(no decided games were played)"
    lines = [
        f"{'#':>2s}  {'entrant':<14s} {'elo':>8s} {'+/-(se)':>8s} {'games':>6s} "
        f"{'W-D-L':>10s} {'score%':>7s} {'unf':>4s}  note",
        "-" * 78,
    ]
    for rank, r in enumerate(fit.ratings, 1):
        note = ""
        if r.games < min_games:
            note = f"provisional (<{min_games} decided games)"
        lines.append(
            f"{rank:>2d}  {_truncate(r.name, 14):<14s} {r.elo:>8.1f} {r.stderr:>8.1f} "
            f"{r.games:>6d} {r.wins:>4d}-{r.draws}-{r.losses:<4d} "
            f"{100 * r.points_pct:>6.1f}% {r.unfinished:>4d}  {note}"
        )
    lines.append("")
    lines.append(f"anchored so '{fit.anchor}' = 0; ratings are Elo-scale "
                 f"differences fitted from the whole matrix at once")
    lines.append(f"converged after {fit.iterations} iterations "
                 f"(log-likelihood {fit.log_likelihood:.2f})")
    return "\n".join(lines)


def format_residuals(fit: FitResult, top: int = 6) -> str:
    """Where the Bradley-Terry ordering fails to explain the results.

    Large residuals are the honest signal that the matrix is not a ladder: if
    A beats B, B beats C and C beats A, no set of ratings fits all three, and
    the residuals of those pairs will be the largest in the table. Printing them
    is what turns a misleading "everything ordered nicely" into a visible
    caveat.
    """
    if not fit.residuals:
        return "(no decided games)"
    lines = [
        f"{'match':<28s} {'observed':>9s} {'expected':>9s} {'gap':>7s} {'games':>6s}",
        "-" * 64,
    ]
    for a, b, obs, exp, games in fit.residuals[:top]:
        lines.append(
            f"{_truncate(f'{a} vs {b}', 28):<28s} {obs:>9.3f} {exp:>9.3f} "
            f"{abs(obs - exp):>7.3f} {games:>6d}"
        )
    lines.append("")
    lines.append("observations are scores in [0,1]; a large gap means the "
                 "ratings cannot explain that pair")
    return "\n".join(lines)


def format_skipped(bench: BenchResult) -> str:
    if not bench.skipped:
        return ""
    lines = ["skipped entrants:"]
    for name, why in bench.skipped.items():
        lines.append(f"  {name:<16s} {why}")
    return "\n".join(lines)


def format_summary(bench: BenchResult, fit: FitResult) -> str:
    # Count from the records, not from ``len(m.games)``. The ``GameResult`` list
    # is a replay convenience that a JSON round-trip drops, so deriving the game
    # total from it makes a re-rendered report claim zero games while the ratings
    # are correct -- which is worse than being wrong, because it looks fine.
    # ``played + unfinished`` is the durable count and is always present.
    decided = sum(m.record.played for m in bench.matches)
    unfinished = sum(m.record.unfinished for m in bench.matches)
    total_games = decided + unfinished
    blocks = [
        "=" * 78,
        "CHESS-RL BENCH  round-robin scoreboard",
        "=" * 78,
        "",
        f"entrants      : {len(bench.names)}",
        f"matches       : {len(bench.matches)}",
        f"games         : {total_games} ({decided} decided, {unfinished} unfinished)",
        f"wall clock    : {bench.seconds:.1f}s",
        "",
        "-- RATING " + "-" * 68,
        format_ratings(fit),
        "",
        "-- RAW MATRIX " + "-" * 64,
        format_matrix(bench),
        "",
        "-- RESIDUALS " + "-" * 65,
        format_residuals(fit),
    ]
    skipped = format_skipped(bench)
    if skipped:
        blocks += ["", "-- SKIPPED " + "-" * 67, skipped]
    return "\n".join(blocks)


def build_report(bench: BenchResult, *, prior: float = 1.0,
                 anchor: str = "random", min_games: int = 4) -> str:
    """Fit and format in one call, for the CLI and the tests."""
    fit = fit_bradley_terry(bench.records, bench.names, prior=prior, anchor=anchor)
    return format_summary(bench, fit)


def format_ablation(bench: BenchResult, fit: FitResult) -> str:
    """The L5 ablation table: what each of the two model seams is worth.

    Both seams change the *search*, not the model, so the interesting quantity
    is the rating difference from the all-off baseline, with its sign. Reading
    it: a prior that helps only by pruning is worth ~0 rating points, whereas a
    leaf evaluator that helps changes every leaf value and moves the rating
    directly. If ``+eval`` is worth more than ``+prior``, the model's value head
    is doing more work than its policy head -- which is the fact the next round
    of training should be built on.
    """
    by = {r.name: r for r in fit.ratings}
    lines = [
        f"{'variant':<28s} {'elo':>8s} {'delta':>10s} {'se':>7s} {'games':>6s}",
        "-" * 68,
    ]
    # Identify the baseline by name, explicitly. An earlier version looped and
    # kept the last name containing "blind" but not "eval", which happens to be
    # right for the four current variants and would silently pick the wrong row
    # if a fifth were added in a different order. Naming it is clearer than
    # implying it.
    base_name = _ABLATION_BASELINE if _ABLATION_BASELINE in by else None
    if base_name is None:
        # Fall back to the lowest-rated variant rather than guessing a name, and
        # say which one was chosen.
        base_name = min(by, key=lambda n: by[n].elo) if by else None
    base = by[base_name].elo if base_name else 0.0
    for name, r in sorted(by.items(), key=lambda kv: -kv[1].elo):
        delta = r.elo - base
        lines.append(
            f"{_truncate(name, 28):<28s} {r.elo:>8.1f} {delta:>+10.1f} "
            f"{r.stderr:>8.1f} {r.games:>6d}"
        )
    lines.append("")
    note = " (both seams off)" if base_name == _ABLATION_BASELINE else ""
    lines.append(f"baseline is '{base_name}'{note}; "
                 f"delta within one stderr is not a result")
    return "\n".join(lines)
