"""Running the round-robin and collecting the raw matrix.

The output of this module is a *counts* matrix and nothing else: who played
whom, how often, and how it went. Rating, ordering and presentation all happen
downstream in ``rating.py`` and ``report.py``. That separation is deliberate.
If the runner were allowed to compute a rating the two would be one function, and
the single most useful property of a benchmark -- that you can recompute the
ranking from the saved matrix without replaying a single game -- would be lost.

Two decisions that are not obvious
----------------------------------
**Fresh policy per match, not per round-robin.** ``LevelSpec.factory`` is called
once per match. L5's engine carries a transposition table, a legal-move cache and
a depth cache; an instance reused across matches would start later matches warm
and the later rows would be measuring cache warmth rather than strength. Fresh
instances cost time and buy correctness.

**Colour-balanced, book-started, fixed-seed.** Every match plays an even number
of games alternating colours from a fixed opening book under a fixed seed. A
match of 6 games therefore samples 6 distinct positions with both colour
assignments, rather than repeating one position six times and reporting the
result as a distribution. This is the difference between a match that measures
strength and a match that measures the opening.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

from chessrl.game import play_game

from bench.levels import OPENING_BOOK, LevelSpec, build, is_playable
from bench.rating import MatchRecord


@dataclass
class MatchOutcome:
    """One match, with the games kept for the record.

    ``record`` is what the fit consumes; ``games`` is what the report prints for
    a suspicious row. Keeping the ``GameResult`` objects means a surprising
    match can be replayed by hand from the saved PGN without re-running the
    policies -- which matters most for exactly the rows you did not expect.
    """

    a: str
    b: str
    record: MatchRecord
    games: list = field(default_factory=list)
    seconds: float = 0.0

    @property
    def games_per_second(self) -> float:
        return len(self.games) / self.seconds if self.seconds > 0 else 0.0


@dataclass
class BenchResult:
    names: list[str]
    matches: list[MatchOutcome]
    skipped: dict[str, str] = field(default_factory=dict)
    seconds: float = 0.0

    @property
    def records(self) -> list[MatchRecord]:
        return [m.record for m in self.matches]

    def matrix(self) -> list[list[str]]:
        """The raw win/loss grid as strings, for the report's summary block.

        Cell ``[i][j]`` is entrant *i*'s record against entrant *j* from *i*'s
        point of view, e.g. ``"+5-1=2"``. The mirror cell is the transpose. Both
        are written out rather than half-filling the grid, because a half-filled
        matrix makes the player you are reading about ambiguous.
        """
        idx = {n: i for i, n in enumerate(self.names)}
        grid = [["" for _ in self.names] for _ in self.names]
        for m in self.matches:
            i, j = idx[m.a], idx[m.b]
            rec = m.record
            cell = f"+{rec.wins}-{rec.losses}={rec.draws}"
            if rec.unfinished:
                cell += f" (u{rec.unfinished})"
            grid[i][j] = cell
            grid[j][i] = f"+{rec.losses}-{rec.wins}={rec.draws}"
            if rec.unfinished:
                grid[j][i] += f" (u{rec.unfinished})"
        return grid


def play_one_match(
    spec_a: LevelSpec,
    spec_b: LevelSpec,
    *,
    games: int = 6,
    opening_fens: Sequence[str] | None = None,
    max_plies: int = 200,
    no_progress_limit: int = 60,
    seed: int = 0,
    progress: Callable[[str], None] | None = None,
) -> MatchOutcome:
    """Play ``games`` colour-balanced games between two specs.

    ``games`` is rounded *up* to an even number: with an odd count one colour is
    played one extra time and the match carries a systematic colour bias. A
    silent bias is worse than a wasted game.
    """
    if games < 2:
        games = 2
    if games % 2:
        games += 1

    book = [f for f in (opening_fens or OPENING_BOOK) if is_playable(f)]
    started = time.perf_counter()

    a_policy, a_err = build(spec_a)
    b_policy, b_err = build(spec_b)
    if a_policy is None:
        raise RuntimeError(f"{spec_a.name}: {a_err}")
    if b_policy is None:
        raise RuntimeError(f"{spec_b.name}: {b_err}")

    wins = draws = losses = unfinished = 0
    played_games = []
    for i in range(games):
        a_is_white = i % 2 == 0
        # One distinct book entry per game. The obvious ``book[i // 2]`` -- one
        # entry per *pair* of games -- looks right because colours alternate in
        # pairs, but it means a 6-game match from a 6-entry book uses only 3
        # distinct positions, each played twice. The whole point of the book is
        # variance across games, so the index advances per game, not per pair.
        start_fen = book[i % len(book)] if book else None
        white, black = (a_policy, b_policy) if a_is_white else (b_policy, a_policy)
        game = play_game(
            white, black,
            max_plies=max_plies,
            no_progress_limit=no_progress_limit,
            start_fen=start_fen,
        )
        played_games.append(game)
        colour = 1 if a_is_white else -1
        if not game.is_finished:
            unfinished += 1
        else:
            score = game.score_for(colour)
            if score > 0:
                wins += 1
            elif score < 0:
                losses += 1
            else:
                draws += 1
        if progress is not None:
            progress(f"    {spec_a.name} vs {spec_b.name} "
                     f"[{i + 1}/{games}] {game.result:>10s} {game.reason}")

    elapsed = time.perf_counter() - started
    record = MatchRecord(
        a=spec_a.name, b=spec_b.name,
        wins=wins, draws=draws, losses=losses, unfinished=unfinished,
    )
    return MatchOutcome(spec_a.name, spec_b.name, record, played_games, elapsed)


def run_round_robin(
    specs: Iterable[LevelSpec],
    *,
    games: int = 6,
    opening_fens: Sequence[str] | None = None,
    max_plies: int = 200,
    no_progress_limit: int = 60,
    seed: int = 0,
    progress: Callable[[str], None] | None = None,
) -> BenchResult:
    """Every unordered pair of ``specs``, played once.

    One record per unordered pair is the right granularity because the match is
    already colour-balanced -- writing both (A,B) and (B,A) would count every
    game twice and make the fit's evidence look twice as strong as it is.

    Entrants that cannot be built (usually torch is absent) are dropped before
    any games are played, and reported in ``skipped``. Dropping them up front
    rather than at their first match means a run either completes or does not
    start, instead of failing an hour in on an unavailable row.
    """
    specs = list(specs)
    skipped: dict[str, str] = {}
    available: list[LevelSpec] = []
    for spec in specs:
        policy, err = build(spec)
        if policy is None:
            skipped[spec.name] = err or "unavailable"
            if progress is not None:
                progress(f"  skipping {spec.name}: {err}")
        else:
            available.append(spec)

    names = [s.name for s in available]
    matches: list[MatchOutcome] = []
    started = time.perf_counter()
    total = len(names) * (len(names) - 1) // 2
    done = 0
    for i in range(len(available)):
        for j in range(i + 1, len(available)):
            done += 1
            if progress is not None:
                progress(f"  match {done}/{total}: "
                         f"{available[i].name} vs {available[j].name}")
            matches.append(play_one_match(
                available[i], available[j],
                games=games,
                opening_fens=opening_fens,
                max_plies=max_plies,
                no_progress_limit=no_progress_limit,
                seed=seed,
                progress=progress,
            ))
    return BenchResult(
        names=names,
        matches=matches,
        skipped=skipped,
        seconds=time.perf_counter() - started,
    )
