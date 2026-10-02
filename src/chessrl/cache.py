"""L0: position cache keyed by zobrist hash, with midstate replay.

Two distinct needs are served here, and it is worth keeping them separate
because they have opposite performance profiles.

**Transposition cache** (:class:`PositionCache`). A chess position reached by
different move orders is the same position, which is exactly what a zobrist
hash detects. Alpha-beta and minimax at L2/L5 re-derive the same subtrees
constantly, so a small LRU keyed on zobrist turns exponential re-derivation
into a dict lookup. Keys are 64-bit ints from :func:`chess.polyglot.zobrist_hash`,
verified stable across push/pop.

**Midstate replay** (:class:`MidstateStore`). The spec asks to "cache the board
to let it resample previous states, playing from midgame". A random self-play
game at L1 is overwhelmingly decided by move 40, but the interesting training
signal is early. So we record every position of every game and resample them
later -- starting training runs from arbitrary midgame positions without
replaying from the opening.

The subtlety is that a midgame resample needs the *rules state*, not just the
piece placement: castling rights, en-passant target and the halfmove clock are
not recoverable from a plane stack. So the store keeps real
:class:`chess.Board` objects (copied, since ``push`` mutates in place) rather
than planes, and planes are derived lazily on demand.

Memory is the constraint. A board copy plus its planes is a few hundred bytes;
a million positions is a few hundred MB. :class:`MidstateStore` therefore caps
itself and evicts, and can optionally spill to disk as a compact JSON-lines
corpus so a training run can be resumed in a later process.
"""

from __future__ import annotations

import json
import os
import random
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import chess
import chess.polyglot
import numpy as np


# --------------------------------------------------------------------------
# transposition cache
# --------------------------------------------------------------------------

class PositionCache:
    """LRU cache mapping zobrist hash -> anything, with hit/miss statistics.

    Deliberately not ``functools.lru_cache``: we need mutable statistics, the
    ability to inspect occupancy, and to pre-warm from a corpus. ``OrderedDict``
    with move-to-end gives O(1) LRU in pure Python and is fast enough here --
    the alpha-beta hot path is dominated by board copying, not by this.

    ``capacity`` should be sized to the search tree, not to memory anxiety.
    A depth-5 alpha-beta from the opening touches on the order of 10^4-10^5
    distinct positions; order 10^5 slots is comfortable.
    """

    __slots__ = ("_store", "capacity", "hits", "misses", "evictions")

    def __init__(self, capacity: int = 1 << 18):
        self.capacity = capacity
        self._store: OrderedDict[int, object] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    @staticmethod
    def key(board: chess.Board) -> int:
        """Zobrist key for a position. Includes side to move and rights."""
        return chess.polyglot.zobrist_hash(board)

    def get(self, board: chess.Board, default=None):
        """Look up by board. Returns ``default`` on miss."""
        k = self.key(board)
        if k in self._store:
            self._store.move_to_end(k)
            self.hits += 1
            return self._store[k]
        self.misses += 1
        return default

    def put(self, board: chess.Board, value) -> None:
        """Insert or update, evicting the least recently used entry if full."""
        k = self.key(board)
        if k in self._store:
            self._store.move_to_end(k)
            self._store[k] = value
            return
        self._store[k] = value
        if len(self._store) > self.capacity:
            self._store.popitem(last=False)
            self.evictions += 1

    def get_or_compute(self, board: chess.Board, compute):
        """Cache-aside: return the cached value or compute, store and return.

        ``compute`` is called with no arguments, so callers close over whatever
        they need. This is the only entry point the search layers use.
        """
        k = self.key(board)
        if k in self._store:
            self._store.move_to_end(k)
            self.hits += 1
            return self._store[k]
        self.misses += 1
        value = compute()
        self._store[k] = value
        if len(self._store) > self.capacity:
            self._store.popitem(last=False)
            self.evictions += 1
        return value

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def stats(self) -> dict:
        return {
            "size": len(self._store),
            "capacity": self.capacity,
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate": round(self.hit_rate, 4),
        }

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, board: chess.Board) -> bool:
        return self.key(board) in self._store


# --------------------------------------------------------------------------
# midstate store
# --------------------------------------------------------------------------

@dataclass(slots=True)
class Midstate:
    """One recorded position plus the metadata training wants."""

    fen: str
    ply: int
    game_id: int
    movecount: int
    material_ratio: float
    zobrist: int

    def board(self) -> chess.Board:
        """Rebuild a playable board. Fresh object each call -- safe to mutate."""
        return chess.Board(self.fen)


@dataclass
class MidstateStore:
    """A capped, resamplable corpus of positions drawn from self-play games.

    Positions are stored as FEN strings rather than objects: a FEN is ~60 bytes
    against a few hundred for a live board, it round-trips exactly, and it
    carries the castling/en-passant/clock state that planes cannot. Slicing a
    FEN back into a ``chess.Board`` is cheap.

    Bucketing by game phase (see :meth:`sample`) is what lets training runs
    request "give me 40% early, 40% middlegame, 20% endgame" rather than
    inheriting whatever phase distribution self-play happened to produce.
    """

    capacity: int = 200_000
    _entries: list[Midstate] = field(default_factory=list)
    _cursor: int = 0
    _next_game_id: int = 0
    total_seen: int = 0
    truncated: bool = False

    # ---- ingestion -------------------------------------------------------

    def new_game_id(self) -> int:
        """Allocate a fresh game identifier."""
        gid = self._next_game_id
        self._next_game_id += 1
        return gid

    def record_board(self, board: chess.Board, game_id: int, ply: int) -> Midstate:
        """Record a single live board. The position is snapshotted, not aliased."""
        from .value import material_ratio

        entry = Midstate(
            fen=board.fen(),
            ply=ply,
            game_id=game_id,
            movecount=board.fullmove_number,
            material_ratio=material_ratio(board),
            zobrist=chess.polyglot.zobrist_hash(board),
        )
        self._append(entry)
        return entry

    def record_game(self, fens: list[str], game_id: int) -> None:
        """Record a whole game from a FEN list (cheap bulk path)."""
        for ply, fen in enumerate(fens):
            board = chess.Board(fen)
            from .value import material_ratio
            self._append(Midstate(
                fen=fen,
                ply=ply,
                game_id=game_id,
                movecount=board.fullmove_number,
                material_ratio=material_ratio(board),
                zobrist=chess.polyglot.zobrist_hash(board),
            ))

    def _append(self, entry: Midstate) -> None:
        self.total_seen += 1
        if len(self._entries) < self.capacity:
            self._entries.append(entry)
            return
        # Full: overwrite in a ring. Keeps a sliding window of recent play,
        # which is preferable to dropping everything or to evicting randomly.
        self.truncated = True
        self._entries[self._cursor] = entry
        self._cursor = (self._cursor + 1) % self.capacity

    # ---- sampling --------------------------------------------------------

    def by_phase(self, phase: str) -> list[Midstate]:
        """All entries in one phase: ``"early"``, ``"mid"`` or ``"late"``.

        Boundaries are by material ratio, not move number, because a game with
        a queen traded on move 8 is already an endgame in the sense that
        matters. Thresholds match :func:`chessrl.value.game_phase`.
        """
        from .value import PHASE_EARLY_MIN, PHASE_MID_MIN

        if phase == "early":
            return [e for e in self._entries if e.material_ratio >= PHASE_EARLY_MIN]
        if phase == "mid":
            return [
                e for e in self._entries
                if PHASE_MID_MIN <= e.material_ratio < PHASE_EARLY_MIN
            ]
        if phase == "late":
            return [e for e in self._entries if e.material_ratio < PHASE_MID_MIN]
        raise ValueError(f"unknown phase {phase!r}")

    def sample(
        self,
        n: int,
        rng: random.Random | None = None,
        mix: tuple[float, float, float] = (0.3, 0.5, 0.2),
    ) -> list[Midstate]:
        """Draw up to ``n`` *distinct* positions, mixed across phases by ``mix``.

        Default mix is 30% early / 50% middlegame / 20% endgame, which roughly
        inverts the natural distribution of self-play (where most sampled plies
        land in a long, low-information middlegame).

        Phase quotas are targets, not guarantees. When a bucket is short the
        shortfall is backfilled from the remaining unpicked corpus, so the
        result is as close to ``n`` as the corpus allows. Sampling is without
        replacement: if the corpus holds fewer than ``n`` positions you get all
        of them exactly once, never duplicates. Duplicates would silently
        reweight training towards whatever phase happens to be over-represented,
        which is the opposite of what the phase mix is for.
        """
        rng = rng or random.Random()
        if not self._entries or n <= 0:
            return []

        target = min(n, len(self._entries))
        weights = {"early": mix[0], "mid": mix[1], "late": mix[2]}
        buckets = {p: self.by_phase(p) for p in weights}

        chosen: list[Midstate] = []
        used: set[int] = set()

        for phase, weight in weights.items():
            quota = int(round(target * weight))
            if quota <= 0:
                continue
            pool = [e for e in buckets[phase] if id(e) not in used]
            if not pool:
                continue
            take = min(quota, len(pool))
            picked = rng.sample(pool, take)
            chosen.extend(picked)
            used.update(id(e) for e in picked)

        # Backfill from anything not already picked, so the phase mix bends
        # rather than the sample size shrinking to zero.
        if len(chosen) < target:
            remaining = [e for e in self._entries if id(e) not in used]
            shortfall = target - len(chosen)
            if remaining:
                picked = rng.sample(remaining, min(shortfall, len(remaining)))
                chosen.extend(picked)

        rng.shuffle(chosen)
        return chosen

    def sample_boards(self, n: int, **kwargs) -> list[chess.Board]:
        """Convenience: ``sample`` but returning fresh playable boards."""
        return [e.board() for e in self.sample(n, **kwargs)]

    # ---- persistence -----------------------------------------------------

    def save_corpus(self, path: str | os.PathLike) -> int:
        """Write the corpus as JSON-lines. Returns the number of rows written.

        Deliberately a plain text format: a training corpus you cannot eyeball
        or grep is a liability. One JSON object per line, phase fields inline.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for e in self._entries:
                fh.write(json.dumps({
                    "fen": e.fen,
                    "ply": e.ply,
                    "game": e.game_id,
                    "movecount": e.movecount,
                    # Full precision, not rounded: this field drives the phase
                    # bucketing, and rounding it to 4 places shifts entries
                    # across the early/mid boundary on reload.
                    "material_ratio": e.material_ratio,
                }) + "\n")
        return len(self._entries)

    def load_corpus(self, path: str | os.PathLike) -> int:
        """Load a corpus file, appending to whatever is already held."""
        loaded = 0
        with Path(path).open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                board = chess.Board(row["fen"])
                self._append(Midstate(
                    fen=row["fen"],
                    ply=row["ply"],
                    game_id=row["game"],
                    movecount=row["movecount"],
                    material_ratio=row["material_ratio"],
                    zobrist=chess.polyglot.zobrist_hash(board),
                ))
                loaded += 1
        return loaded

    # ---- introspection ---------------------------------------------------

    def stats(self) -> dict:
        return {
            "size": len(self._entries),
            "capacity": self.capacity,
            "total_seen": self.total_seen,
            "games": self._next_game_id,
            "truncated": self.truncated,
            "early": len(self.by_phase("early")),
            "mid": len(self.by_phase("mid")),
            "late": len(self.by_phase("late")),
        }

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[Midstate]:
        return iter(self._entries)

    def __getitem__(self, index):
        """Index into the recorded positions, for inspection and tests."""
        return self._entries[index]
