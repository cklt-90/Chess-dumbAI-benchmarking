"""Master sources for H25: supervise L3 on a *true* master, not a weak search.

The existing training hook, :meth:`L3Trainer.search_feedback`, produces a target
of the form ``list[(chess.Move, float)]`` with weights in ``(0, 1]`` and the
best move at ``1.0``. Everything downstream of that — the credit update, the
checkpointing, the held-out metrics — is source-agnostic. This module supplies
*alternative sources* of the same contract:

* :class:`EngineMaster` — a UCI engine (e.g. Stockfish), MultiPV top-k with
  centipawn scores squashed to weights exactly as ``search_feedback`` does.
* :class:`PgnMaster` — a master-games corpus, weighted by how often each move was
  actually played from the position.

Neither the engine binary nor a PGN corpus is vendored; supplying one is the
user's step (see ``bench/DESIGN-h25-true-master.md``). Until then this module is
inert — it imports cleanly and its pure conversion helpers are unit-tested
without any engine.

Design constraint: L0-L3 must run without the optional torch extra. This module
uses only ``chess`` (including ``chess.engine``, which ships with python-chess)
and the standard library, so it stays inside that constraint.
"""
from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Iterable, Protocol, Sequence

import chess
import chess.pgn


def master_position_key(board: chess.Board) -> tuple:
    """A position identity for corpus lookups, ignoring move counters.

    Master-game indexing keys on the position itself, so the halfmove/fullmove
    counters must be excluded or the same position would never be found across
    games that reached it at different times.
    """
    return (
        board.board_fen(),
        board.turn,
        int(board.castling_rights),
        board.ep_square,
    )


def scores_to_weights(
    scored: Iterable[tuple[chess.Move, float]],
    top_k: int = 5,
    margin: float = 300.0,
) -> list[tuple[chess.Move, float]]:
    """Convert ``(move, score)`` pairs to ``(move, weight)`` with best == 1.0.

    The squash is ``exp((score - best) / margin)`` — the same "300cp is clearly
    better" convention as :meth:`L3Trainer.search_feedback`, so an engine-sourced
    arm differs from the weak-search arm in the *labels*, not in the squash.
    Scores must already be from the side-to-move's point of view.
    """
    if top_k < 1:
        raise ValueError("top_k must be positive")
    ranked = sorted(scored, key=lambda kv: kv[1], reverse=True)[:top_k]
    if not ranked:
        return []
    best = ranked[0][1]
    return [(move, math.exp((score - best) / margin)) for move, score in ranked]


def counts_to_weights(
    counts: Counter, top_k: int = 5
) -> list[tuple[chess.Move, float]]:
    """Convert move-frequency counts to ``(move, weight)`` with best == 1.0.

    Weight is ``count / max(count)`` so the most-played master move gets 1.0.
    Ties are broken by move ordering for determinism (``Counter.most_common`` is
    insertion-ordered, which is not stable across corpora), so the caller gets a
    reproducible top-k.
    """
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if not counts:
        return []
    best = max(counts.values())
    if best <= 0:
        return []
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0].uci()))
    return [(move, count / best) for move, count in ranked[:top_k]]


class MasterSource(Protocol):
    """Anything that can label a position with ``(move, weight)`` targets."""

    def targets(self, board: chess.Board) -> list[tuple[chess.Move, float]]:
        ...


class EngineMaster:
    """A UCI engine as a master source (e.g. Stockfish).

    Wraps ``chess.engine``. The engine path is supplied by the caller; this class
    never downloads or installs one. Use as a context manager so the engine
    process is always closed::

        with EngineMaster("/path/to/stockfish", top_k=5, depth=12) as master:
            targets = master.targets(board)
    """

    def __init__(self, engine_path: str | Path, top_k: int = 5,
                 depth: int | None = 12, time_limit: float | None = None,
                 margin: float = 300.0):
        import chess.engine  # local import: keeps module import cheap and clear

        self.top_k = top_k
        self.margin = margin
        if depth is not None:
            self.limit = chess.engine.Limit(depth=depth)
        elif time_limit is not None:
            self.limit = chess.engine.Limit(time=time_limit)
        else:
            raise ValueError("provide either depth or time_limit")
        self.engine = chess.engine.SimpleEngine.popen_uci(str(engine_path))

    def targets(self, board: chess.Board) -> list[tuple[chess.Move, float]]:
        import chess.engine

        infos = self.engine.analyse(board, self.limit, multipv=self.top_k)
        scored: list[tuple[chess.Move, float]] = []
        for info in infos:
            pv = info.get("pv")
            score = info.get("score")
            if not pv or score is None:
                continue
            pov = score.pov(board.turn)
            cp = pov.score(mate_score=100000)
            if cp is None:  # mate distance present but no cp; use the mate score
                cp = pov.score()
            if cp is None:
                continue
            scored.append((pv[0], float(cp)))
        return scores_to_weights(scored, top_k=self.top_k, margin=self.margin)

    def close(self) -> None:
        self.engine.quit()

    def __enter__(self) -> "EngineMaster":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class PgnMaster:
    """A master-games corpus as a master source, weighted by move frequency.

    The corpus is indexed once at construction: every position that occurs in
    the games maps to a :class:`Counter` of the moves played from it. Positions
    absent from the corpus return no targets, so a training loop must skip them
    (a master-games corpus covers elite lines, not the whole position space).
    """

    def __init__(self, pgn_path: str | Path, top_k: int = 5,
                 max_games: int | None = None):
        self.top_k = top_k
        self._index: dict[tuple, Counter] = {}
        self.games_indexed = 0
        with open(pgn_path, encoding="utf-8") as handle:
            while True:
                if max_games is not None and self.games_indexed >= max_games:
                    break
                game = chess.pgn.read_game(handle)
                if game is None:
                    break
                self._index_game(game)
                self.games_indexed += 1

    def _index_game(self, game: chess.pgn.Game) -> None:
        board = game.board()
        for move in game.mainline_moves():
            key = master_position_key(board)
            self._index.setdefault(key, Counter())[move] += 1
            board.push(move)

    @property
    def positions_indexed(self) -> int:
        return len(self._index)

    def targets(self, board: chess.Board) -> list[tuple[chess.Move, float]]:
        counts = self._index.get(master_position_key(board))
        if not counts:
            return []
        return counts_to_weights(counts, top_k=self.top_k)


def load_pgn_master(pgn_path: str | Path, top_k: int = 5,
                    max_games: int | None = None) -> PgnMaster:
    """Convenience constructor mirroring the module's ``PgnMaster``."""
    return PgnMaster(pgn_path, top_k=top_k, max_games=max_games)
