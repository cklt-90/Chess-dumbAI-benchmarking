"""L0: self-play game loop and result bookkeeping.

Every level plays through this loop, so it is the one place where a game's
lifecycle is defined. Keeping it level-agnostic is what makes the L6
tournament a fair comparison: levels differ only in how they choose a move,
never in how a game is run, adjudicated or recorded.

Two policy interfaces are supported, because the benchmark needs both:

* :class:`MovePolicy` -- ``select(board) -> chess.Move``. The natural shape for
  anything that does search (L2, L5) and does not want to expose a
  distribution.
* :class:`DistributionPolicy` -- additionally ``logits(board)`` and
  :meth:`DistributionPolicy.sample`. The natural shape for the learners (L1,
  L1.5, L3, L4), which need the mask, the distribution and the update hook.

:func:`play_game` accepts either; it duck-types on ``sample``.

Adjudication
------------
A naive policy will happily shuffle pieces forever, so games need termination
rules beyond the draw conditions python-chess already detects (threefold
repetition, fifty-move, insufficient material). Two are applied:

* a hard ply cap, ``max_plies``;
* a no-progress cap, counting plies since the last capture or pawn move.

Both are recorded as ``"unfinished"`` rather than silently scored as a draw, so
training can decide whether to treat them as noise. Scoring an unfinished game
as a draw is the single easiest way to accidentally teach a model that
shuffling is as good as winning.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Callable, Protocol, Sequence, runtime_checkable

import chess
import chess.pgn
import numpy as np

from . import value as V
from .cache import MidstateStore


# --------------------------------------------------------------------------
# policy protocols
# --------------------------------------------------------------------------

@runtime_checkable
class MovePolicy(Protocol):
    """Chooses a legal move from a position."""

    name: str

    def select(self, board: chess.Board) -> chess.Move:
        """Return a legal move for ``board``. Must not mutate ``board``."""
        ...


@runtime_checkable
class DistributionPolicy(MovePolicy, Protocol):
    """A :class:`MovePolicy` that also exposes its move distribution."""

    def move_distribution(self, board: chess.Board) -> np.ndarray:
        """Return a normalised distribution over the flat action space.

        Length ``chessrl.masks.ACTION_SPACE``. Zero mass on illegal moves.
        Required for the L1.5 counter logic and for L6 ensemble weighting.
        """
        ...


# --------------------------------------------------------------------------
# result record
# --------------------------------------------------------------------------

@dataclass(slots=True)
class GameResult:
    """Everything a training run or a tournament needs to know about a game."""

    result: str                     # "1-0", "0-1", "1/2-1/2", "unfinished"
    reason: str                     # human-readable termination cause
    plies: int
    moves: list[str] = field(default_factory=list)
    fens: list[str] = field(default_factory=list)
    duration_s: float = 0.0

    @property
    def is_decisive(self) -> bool:
        return self.result in ("1-0", "0-1")

    @property
    def is_finished(self) -> bool:
        return self.result != "unfinished"

    def score_for(self, colour: int) -> float:
        """``+1`` win, ``0`` draw, ``-1`` loss, ``0`` for an unfinished game.

        Unfinished games score zero by design: they carry no information about
        which side was better, and treating them as draws is what teaches a
        naive policy to avoid losing by never committing.
        """
        if not self.is_finished:
            return 0.0
        return V.outcome_score(self.result, colour)

    def to_pgn(self, white: str = "?", black: str = "?") -> str:
        """Render as a PGN string, for eyeballing a game after the fact."""
        game = chess.pgn.Game()
        game.headers["White"] = white
        game.headers["Black"] = black
        game.headers["Result"] = self.result if self.is_finished else "*"
        node = game
        for san in self.moves:
            node = node.add_variation(node.board().parse_san(san))
        return str(game)


# --------------------------------------------------------------------------
# termination helpers
# --------------------------------------------------------------------------

def _termination_reason(board: chess.Board) -> str | None:
    """Cheap terminal conditions, checked every ply.

    Deliberately excludes ``board.is_game_over(claim_draw=True)``. That call
    runs threefold-repetition detection, which scans the entire move stack, so
    calling it every ply makes the game loop quadratic: profiling showed it
    costing 39% of total training time on its own.

    It is also redundant here. Shuffling and repetition are exactly what the
    ply cap and the no-progress cap already catch, at O(1) instead of O(n), so
    dropping the claimable-draw path changes which positions terminate but not
    whether they do. The conditions kept below are all cheap:

    * ``is_checkmate`` / ``is_stalemate`` -- "are there legal moves"
    * ``is_insufficient_material`` -- counts a few piece sets
    * ``is_seventyfive_moves`` -- reads the halfmove clock
    * ``is_fivefold_repetition`` -- only runs after 8 plies of no progress

    The last one is guarded by the halfmove clock for the same reason: it is
    the expensive scan, and there is no point running it when the position has
    obviously been progressing.
    """
    if board.is_checkmate():
        return "checkmate"
    if board.is_stalemate():
        return "stalemate"
    if board.is_insufficient_material():
        return "insufficient_material"
    if board.is_seventyfive_moves():
        return "seventyfive_moves"
    # Only attempt the repetition scan once there is a plausible repetition to
    # find: it needs at least 8 plies of reversible moves.
    if board.halfmove_clock >= 8 and board.is_fivefold_repetition():
        return "fivefold_repetition"
    return None


def _count_men(board_fen: str) -> int:
    """Number of pieces on the board, from the placement field of a FEN."""
    return sum(1 for ch in board_fen if ch.isalpha())


def _progressed(prev_fen: str, cur_fen: str) -> bool:
    """Did this ply involve a capture or a pawn move?"""
    prev_board, cur_board = prev_fen.split()[0], cur_fen.split()[0]
    # A capture removes a man.
    if _count_men(prev_board) != _count_men(cur_board):
        return True
    # A pawn move changes the pawn count on the files it touched. Comparing the
    # set of occupied files per row is enough to detect a push or a capture by
    # a pawn, without needing the move itself.
    for prev_row, cur_row in zip(prev_board.split("/"), cur_board.split("/")):
        if prev_row == cur_row:
            continue
        if (
            prev_row.count("P") != cur_row.count("P")
            or prev_row.count("p") != cur_row.count("p")
        ):
            return True
    return False


def _no_progress_plies(fens: Sequence[str]) -> int:
    """Plies since the last capture or pawn move, derived from a FEN list.

    Kept for replaying a *recorded* game, where no live counter is available.
    The game loop itself does not use this: it maintains the same count
    incrementally, because rescanning the whole list every ply is quadratic and
    showed up as a real cost in profiling.

    The halfmove clock in the FEN would be cheaper but is not reliable for this
    purpose -- it is only guaranteed to reset on a capture or a pawn move by
    the side that just moved, which is not quite the same question.
    """
    count = 0
    for i in range(len(fens) - 1, 0, -1):
        if _progressed(fens[i - 1], fens[i]):
            break
        count += 1
    return count


# --------------------------------------------------------------------------
# the game loop
# --------------------------------------------------------------------------

def play_game(
    white: MovePolicy,
    black: MovePolicy,
    *,
    max_plies: int = 200,
    no_progress_limit: int = 60,
    rng: random.Random | None = None,
    start_fen: str | None = None,
    record_fens: bool = True,
    midstate: MidstateStore | None = None,
    game_id: int | None = None,
    on_ply: Callable[[chess.Board, chess.Move], None] | None = None,
) -> GameResult:
    """Play one game between two policies and return the recorded result.

    Parameters
    ----------
    white, black
        Anything with ``select(board) -> Move``. May be the same object, which
        is how self-play is expressed.
    max_plies
        Hard cap. Reaching it terminates with ``"unfinished"``.
    no_progress_limit
        Terminate after this many plies with no capture and no pawn move. This
        is what stops two blind policies from shuffling a rook back and forth
        until ``max_plies``, which wastes most of a training run's budget.
    rng
        Reserved for policies that want reproducible tie-breaking; the loop
        itself is deterministic given deterministic policies.
    start_fen
        Begin from a midgame position rather than the opening. This is the
        entry point the midstate replay uses to resample earlier states.
    midstate
        If provided, every position of the game is recorded for later replay.
    on_ply
        Hook called after each move with the position *before* the move and the
        move played. The learners use this to collect update targets without
        re-deriving the game afterwards.
    """
    board = chess.Board(start_fen) if start_fen else chess.Board()
    moves: list[str] = []
    fens: list[str] = [board.fen()] if record_fens else []
    started = time.perf_counter()

    if midstate is not None and game_id is not None:
        midstate.record_board(board, game_id, 0)

    # Track no-progress incrementally. Deriving it from the FEN list each ply
    # rescans the whole game every move, which is quadratic in plies and was
    # measurably the second-largest cost in the training loop. Here it is O(1)
    # per ply: reset on a capture or a pawn move, increment otherwise.
    no_progress = 0
    prev_board_fen = board.board_fen()

    reason: str | None = None
    while True:
        reason = _termination_reason(board)
        if reason is not None:
            break
        if len(moves) >= max_plies:
            reason = "max_plies"
            break
        if no_progress >= no_progress_limit:
            reason = "no_progress"
            break

        policy = white if board.turn == chess.WHITE else black
        move = policy.select(board)
        # Guard the contract rather than trusting every policy implementation:
        # an illegal move here would corrupt the whole training run silently.
        if move not in board.legal_moves:
            raise ValueError(
                f"{policy.name} returned illegal move {move} in position "
                f"{board.fen()}"
            )

        if on_ply is not None:
            on_ply(board.copy(stack=False), move)

        is_progress = board.is_capture(move) or board.piece_type_at(
            move.from_square
        ) == chess.PAWN

        san = board.san(move)
        board.push(move)
        moves.append(san)
        no_progress = 0 if is_progress else no_progress + 1
        if record_fens:
            fens.append(board.fen())
            if midstate is not None and game_id is not None:
                midstate.record_board(board, game_id, len(moves))

    if reason in ("checkmate",):
        result = "1-0" if board.turn == chess.BLACK else "0-1"
    elif reason in ("max_plies", "no_progress"):
        result = "unfinished"
    else:
        result = "1/2-1/2"

    return GameResult(
        result=result,
        reason=reason,
        plies=len(moves),
        moves=moves,
        fens=fens,
        duration_s=time.perf_counter() - started,
    )


def play_match(
    a: MovePolicy,
    b: MovePolicy,
    *,
    games: int = 2,
    alternate_colours: bool = True,
    opening_fens: Sequence[str] | None = None,
    max_plies: int = 200,
    no_progress_limit: int = 60,
    seed: int = 0,
) -> dict:
    """Play a short match between two policies and tally the score.

    Alternating colours matters: a policy that is much stronger as White would
    otherwise look better or worse purely by assignment. When ``opening_fens``
    is supplied the match is played from a fixed set of openings, which is how
    you get a low-variance comparison from few games -- a real consideration
    given how expensive L5 games are.
    """
    rng = random.Random(seed)
    score_a = 0.0
    wins = draws = losses = unfinished = 0
    results: list[GameResult] = []

    for i in range(games):
        a_is_white = (i % 2 == 0) if alternate_colours else True
        start_fen = None
        if opening_fens:
            start_fen = opening_fens[i % len(opening_fens)]

        white, black = (a, b) if a_is_white else (b, a)
        game = play_game(
            white, black,
            max_plies=max_plies,
            no_progress_limit=no_progress_limit,
            rng=rng,
            start_fen=start_fen,
        )
        results.append(game)

        a_colour = chess.WHITE if a_is_white else chess.BLACK
        s = game.score_for(a_colour)
        score_a += s
        if not game.is_finished:
            unfinished += 1
        elif s > 0:
            wins += 1
        elif s < 0:
            losses += 1
        else:
            draws += 1

    played = games - unfinished
    return {
        "a": a.name,
        "b": b.name,
        "games": games,
        "played": played,
        "unfinished": unfinished,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "score_a": score_a,
        "points_pct": (score_a / played) if played else 0.0,
        "results": results,
    }


def generate_games(
    policy: MovePolicy,
    n_games: int,
    *,
    midstate: MidstateStore | None = None,
    seed: int = 0,
    max_plies: int = 200,
    no_progress_limit: int = 60,
    progress: Callable[[int, GameResult], None] | None = None,
) -> list[GameResult]:
    """Self-play ``n_games`` games, optionally recording into a midstate store.

    Self-play here means the same policy on both sides. That is the cheapest
    corpus generator available and is what L1 uses to bootstrap; later levels
    mix in games against earlier levels to avoid a self-confirming loop.
    """
    out: list[GameResult] = []
    for i in range(n_games):
        gid = midstate.new_game_id() if midstate is not None else None
        game = play_game(
            policy, policy,
            max_plies=max_plies,
            no_progress_limit=no_progress_limit,
            rng=random.Random(seed + i),
            midstate=midstate,
            game_id=gid,
        )
        out.append(game)
        if progress is not None:
            progress(i + 1, game)
    return out


def summarise_games(results: Sequence[GameResult]) -> dict:
    """Aggregate a batch of games into headline numbers for a log line."""
    n = len(results)
    if n == 0:
        return {"games": 0}
    decisive = sum(1 for g in results if g.is_decisive)
    unfinished = sum(1 for g in results if not g.is_finished)
    return {
        "games": n,
        "decisive": decisive,
        "draws": sum(1 for g in results if g.result == "1/2-1/2"),
        "unfinished": unfinished,
        "draw_rate": round(
            sum(1 for g in results if g.result == "1/2-1/2") / n, 3
        ),
        "unfinished_rate": round(unfinished / n, 3),
        "avg_plies": round(sum(g.plies for g in results) / n, 1),
        "max_plies": max(g.plies for g in results),
        "avg_seconds": round(sum(g.duration_s for g in results) / n, 3),
    }


# --------------------------------------------------------------------------
# a couple of trivial reference policies
# --------------------------------------------------------------------------

class RandomPolicy:
    """Uniform random legal mover. The floor of the benchmark."""

    name = "random"

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def select(self, board: chess.Board) -> chess.Move:
        return self.rng.choice(list(board.legal_moves))


class MaterialPolicy:
    """One-ply greedy on static evaluation, with random tie-breaking.

    Not a benchmark level -- it is a sanity reference. If a learned policy
    cannot beat one-ply greedy within a few hundred games, something is wrong
    with the learning code rather than with the idea.
    """

    name = "greedy-material"

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def select(self, board: chess.Board) -> chess.Move:
        best: list[chess.Move] = []
        best_score = -10**9
        for move in board.legal_moves:
            probe = board.copy(stack=False)
            probe.push(move)
            # Negate: after pushing, evaluate() is from the opponent's view.
            score = -V.evaluate(probe)
            if score > best_score:
                best_score = score
                best = [move]
            elif score == best_score:
                best.append(move)
        return self.rng.choice(best)
