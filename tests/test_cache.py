"""L0 tests: caching and midstate replay."""

from __future__ import annotations

import random

import chess
import chess.polyglot
import numpy as np
import pytest

from chessrl.cache import MidstateStore, PositionCache
from conftest import INTERESTING_FENS, random_game_fens


# --------------------------------------------------------------------------
# zobrist keys
# --------------------------------------------------------------------------

def test_zobrist_is_stable_across_push_pop():
    board = chess.Board()
    original = PositionCache.key(board)
    board.push_san("e4")
    board.pop()
    assert PositionCache.key(board) == original


def test_zobrist_ignores_move_order():
    """Transpositions must collide: that is the entire point of the cache."""
    a = chess.Board()
    a.push_san("e4")
    a.push_san("e5")
    a.push_san("Nf3")

    b = chess.Board()
    b.push_san("Nf3")
    b.push_san("e5")
    b.push_san("e4")

    assert PositionCache.key(a) == PositionCache.key(b)


def test_zobrist_differs_for_different_positions():
    a = chess.Board()
    b = chess.Board()
    b.push_san("e4")
    assert PositionCache.key(a) != PositionCache.key(b)


def test_zobrist_distinguishes_side_to_move():
    a = chess.Board()
    b = chess.Board()
    b.turn = chess.BLACK
    assert PositionCache.key(a) != PositionCache.key(b)


# --------------------------------------------------------------------------
# transposition cache
# --------------------------------------------------------------------------

def test_cache_hit_and_miss_accounting():
    cache = PositionCache(capacity=16)
    board = chess.Board()
    assert cache.get(board, default=None) is None
    assert cache.misses == 1
    cache.put(board, "value")
    assert cache.get(board) == "value"
    assert cache.hits == 1
    assert cache.hit_rate == 0.5


def test_get_or_compute_only_computes_once():
    cache = PositionCache(capacity=16)
    board = chess.Board()
    calls = []

    def compute():
        calls.append(1)
        return 42

    assert cache.get_or_compute(board, compute) == 42
    assert cache.get_or_compute(board, compute) == 42
    assert len(calls) == 1


def test_cache_evicts_least_recently_used():
    cache = PositionCache(capacity=2)
    boards = []
    # Three distinct positions reached by three legal White opening moves.
    for san in ("e4", "d4", "Nf3"):
        b = chess.Board()
        b.push_san(san)
        boards.append(b)

    cache.put(boards[0], "a")
    cache.put(boards[1], "b")
    cache.get(boards[0])          # refresh 0 so 1 becomes the LRU
    cache.put(boards[2], "c")     # evicts 1

    assert cache.get(boards[0]) == "a"
    assert cache.get(boards[2]) == "c"
    assert cache.get(boards[1], default=None) is None
    assert cache.evictions == 1


def test_cache_capacity_is_respected():
    cache = PositionCache(capacity=4)
    for fen in random_game_fens(seed=1, plies=30):
        cache.put(chess.Board(fen), fen)
    assert len(cache) <= 4


def test_cache_contains_and_len():
    cache = PositionCache(capacity=8)
    board = chess.Board()
    assert board not in cache
    cache.put(board, 1)
    assert board in cache
    assert len(cache) == 1


def test_cache_stats_shape():
    cache = PositionCache(capacity=8)
    cache.put(chess.Board(), 1)
    stats = cache.stats()
    assert set(stats) == {
        "size", "capacity", "hits", "misses", "evictions", "hit_rate",
    }


def test_cache_makes_transposition_search_cheap():
    """A repeated lookup over a real move sequence should hit every time."""
    cache = PositionCache(capacity=1024)
    board = chess.Board()
    fens = []
    for san in ("e4", "e5", "Nf3", "Nc6", "Bb5"):
        board.push_san(san)
        fens.append(board.fen())
        cache.put(board, board.fen())

    for fen in fens:
        assert cache.get(chess.Board(fen)) == fen
    assert cache.misses == 0


# --------------------------------------------------------------------------
# midstate store
# --------------------------------------------------------------------------

def test_midstate_records_a_game():
    store = MidstateStore(capacity=100)
    gid = store.new_game_id()
    board = chess.Board()
    store.record_board(board, gid, ply=0)
    assert len(store) == 1
    assert store.total_seen == 1


def test_midstate_board_rebuilds_exactly():
    store = MidstateStore(capacity=100)
    board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3")
    store.record_board(board, store.new_game_id(), ply=6)
    rebuilt = store[0].board()
    assert rebuilt.fen() == board.fen()
    # And it must be independently mutable.
    rebuilt.push_san("Nf6")
    assert rebuilt.fen() != board.fen()
    assert store[0].board().fen() == board.fen()


def test_midstate_capacity_ring_wraps():
    store = MidstateStore(capacity=5)
    gid = store.new_game_id()
    board = chess.Board()
    for _ in range(12):
        store.record_board(board, gid, ply=0)
    assert len(store) == 5
    assert store.truncated
    assert store.total_seen == 12


def test_midstate_phase_bucketing():
    store = MidstateStore(capacity=100)
    gid = store.new_game_id()
    store.record_board(chess.Board(), gid, 0)                                   # early
    store.record_board(
        chess.Board("8/4k3/8/8/8/8/4K3/7Q w - - 0 1"), gid, 60
    )                                                                           # late
    assert len(store.by_phase("early")) == 1
    assert len(store.by_phase("late")) == 1


def test_midstate_phase_rejects_unknown_label():
    store = MidstateStore(capacity=4)
    with pytest.raises(ValueError):
        store.by_phase("stoneage")


def test_midstate_sample_respects_mix():
    store = MidstateStore(capacity=500)
    gid = store.new_game_id()
    # 60 opening positions, 60 endgame positions.
    for _ in range(60):
        store.record_board(chess.Board(), gid, 0)
        store.record_board(
            chess.Board("8/4k3/8/8/8/8/4K3/7Q w - - 0 1"), gid, 60
        )
    rng = random.Random(0)
    sample = store.sample(50, rng=rng, mix=(0.5, 0.0, 0.5))
    assert len(sample) == 50
    phases = [e.material_ratio >= 0.75 for e in sample]
    assert 0 < sum(phases) < 50


def test_midstate_sample_more_than_available():
    store = MidstateStore(capacity=100)
    gid = store.new_game_id()
    for _ in range(3):
        store.record_board(chess.Board(), gid, 0)
    assert len(store.sample(10)) == 3


def test_midstate_sample_empty_store():
    assert MidstateStore(capacity=4).sample(5) == []


def test_midstate_sample_boards_are_playable():
    store = MidstateStore(capacity=100)
    gid = store.new_game_id()
    for fen in random_game_fens(seed=2, plies=20):
        store.record_board(chess.Board(fen), gid, 0)
    for board in store.sample_boards(5, rng=random.Random(1)):
        assert isinstance(board, chess.Board)
        board.push(next(iter(board.legal_moves)))


def test_midstate_corpus_roundtrip(tmp_path):
    store = MidstateStore(capacity=100)
    gid = store.new_game_id()
    fens = random_game_fens(seed=4, plies=10)
    for ply, fen in enumerate(fens):
        store.record_board(chess.Board(fen), gid, ply)

    path = tmp_path / "corpus.jsonl"
    assert store.save_corpus(path) == len(fens)

    restored = MidstateStore(capacity=100)
    assert restored.load_corpus(path) == len(fens)
    assert [e.fen for e in restored] == [e.fen for e in store]


def test_midstate_corpus_roundtrip_preserves_ratio(tmp_path):
    store = MidstateStore(capacity=100)
    gid = store.new_game_id()
    board = chess.Board("8/4k3/8/8/8/8/4K3/7Q w - - 0 1")
    store.record_board(board, gid, 60)
    path = tmp_path / "c.jsonl"
    store.save_corpus(path)

    restored = MidstateStore(capacity=100)
    restored.load_corpus(path)
    assert restored[0].material_ratio == pytest.approx(store[0].material_ratio)


def test_midstate_load_skips_blank_lines(tmp_path):
    path = tmp_path / "c.jsonl"
    store = MidstateStore(capacity=10)
    store.record_board(chess.Board(), store.new_game_id(), 0)
    store.save_corpus(path)
    with path.open("a", encoding="utf-8") as fh:
        fh.write("\n\n")

    restored = MidstateStore(capacity=10)
    assert restored.load_corpus(path) == 1


def test_midstate_stats_shape():
    store = MidstateStore(capacity=10)
    store.record_board(chess.Board(), store.new_game_id(), 0)
    stats = store.stats()
    assert set(stats) == {
        "size", "capacity", "total_seen", "games", "truncated",
        "early", "mid", "late",
    }


def test_whole_game_records_every_ply():
    store = MidstateStore(capacity=200)
    gid = store.new_game_id()
    board = chess.Board()
    ply = 0
    while not board.is_game_over() and ply < 30:
        store.record_board(board, gid, ply)
        board.push(random.Random(ply).choice(list(board.legal_moves)))
        ply += 1
    assert len(store) == ply
