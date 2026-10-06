import math

import chess
import numpy as np
import pytest

from chessrl.master import (
    EngineMaster,
    PgnMaster,
    counts_to_weights,
    master_position_key,
    scores_to_weights,
)
from chessrl.perceptron import L3Config, L3Trainer

_WEIGHT_NAMES = ("W_from", "b_from", "W_ctx", "W_hid", "W_to_dst", "W_to_src", "W_geom", "promo")


# --------------------------------------------------------------------------
# pure conversion helpers
# --------------------------------------------------------------------------

def test_scores_to_weights_puts_best_at_one_and_squashes_by_margin():
    moves = [chess.Move.from_uci("e2e4"), chess.Move.from_uci("d2d4"),
             chess.Move.from_uci("g1f3")]
    scored = [(moves[0], 300.0), (moves[1], 0.0), (moves[2], -300.0)]
    weights = scores_to_weights(scored, top_k=5, margin=300.0)
    assert weights[0] == (moves[0], pytest.approx(1.0))
    assert weights[1][1] == pytest.approx(math.exp(-1.0))
    assert weights[2][1] == pytest.approx(math.exp(-2.0))


def test_scores_to_weights_respects_top_k_and_sorts_descending():
    moves = [chess.Move.from_uci(u) for u in ("a2a3", "b2b3", "c2c3", "d2d4")]
    scored = [(moves[0], 0.0), (moves[1], 100.0), (moves[2], 50.0), (moves[3], 200.0)]
    weights = scores_to_weights(scored, top_k=2, margin=300.0)
    assert [m for m, _ in weights] == [moves[3], moves[1]]
    assert weights[0][1] == pytest.approx(1.0)


def test_scores_to_weights_empty_and_bad_topk():
    assert scores_to_weights([], top_k=5) == []
    with pytest.raises(ValueError):
        scores_to_weights([], top_k=0)


def test_counts_to_weights_scales_by_max_count():
    e4, d4 = chess.Move.from_uci("e2e4"), chess.Move.from_uci("d2d4")
    counts = {e4: 2, d4: 1}
    weights = counts_to_weights(counts, top_k=5)
    assert weights[0] == (e4, pytest.approx(1.0))
    assert weights[1] == (d4, pytest.approx(0.5))


def test_counts_to_weights_tie_break_is_deterministic_by_uci():
    # Equal counts must not depend on dict insertion order.
    a = chess.Move.from_uci("g1f3")
    b = chess.Move.from_uci("b1c3")
    forward = counts_to_weights({a: 1, b: 1}, top_k=5)
    backward = counts_to_weights({b: 1, a: 1}, top_k=5)
    assert forward == backward
    assert [m.uci() for m, _ in forward] == ["b1c3", "g1f3"]


def test_counts_to_weights_empty_and_zero():
    assert counts_to_weights({}, top_k=5) == []
    assert counts_to_weights({chess.Move.from_uci("e2e4"): 0}, top_k=5) == []


def test_master_position_key_ignores_move_counters():
    board = chess.Board()
    board.push_san("e4")
    key_before = master_position_key(board)
    board.halfmove_clock = 9
    board.fullmove_number = 42
    assert master_position_key(board) == key_before


# --------------------------------------------------------------------------
# PGN master
# --------------------------------------------------------------------------

_PGN = """
[Event "t1"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 1-0

[Event "t2"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 1-0

[Event "t3"]
[Result "1-0"]

1. d4 d5 2. Nf3 Nf6 1-0
"""


def _write_pgn(tmp_path):
    path = tmp_path / "masters.pgn"
    path.write_text(_PGN, encoding="utf-8")
    return path


def test_pgn_master_indexes_games_and_returns_frequency_weights(tmp_path):
    master = PgnMaster(_write_pgn(tmp_path), top_k=5)
    assert master.games_indexed == 3
    # Start position: e4 played twice, d4 once -> 1.0 and 0.5.
    targets = master.targets(chess.Board())
    weights = {m.uci(): w for m, w in targets}
    assert weights["e2e4"] == pytest.approx(1.0)
    assert weights["d2d4"] == pytest.approx(0.5)
    assert len(targets) == 2


def test_pgn_master_returns_nothing_for_unseen_position(tmp_path):
    master = PgnMaster(_write_pgn(tmp_path), top_k=5)
    # A position the corpus never reached.
    board = chess.Board()
    for san in ("e4", "e5", "Nf3", "Nc6", "Bb5"):  # corpus stops at Nc6
        board.push_san(san)
    assert master.targets(board) == []


def test_pgn_master_max_games_limits_index(tmp_path):
    master = PgnMaster(_write_pgn(tmp_path), top_k=5, max_games=1)
    assert master.games_indexed == 1
    # Only t1 indexed, so d4 is absent.
    weights = {m.uci() for m, _ in master.targets(chess.Board())}
    assert weights == {"e2e4"}


# --------------------------------------------------------------------------
# engine master (no binary required for these checks)
# --------------------------------------------------------------------------

def test_engine_master_requires_a_limit_before_opening_engine():
    # No depth and no time_limit must fail fast, before any process is spawned.
    with pytest.raises(ValueError):
        EngineMaster("/nonexistent/stockfish", depth=None, time_limit=None)


# --------------------------------------------------------------------------
# trainer hook + refactor parity
# --------------------------------------------------------------------------

class _StubMaster:
    def __init__(self, targets):
        self._targets = targets

    def targets(self, board):
        return list(self._targets)


def test_train_on_master_applies_stub_targets():
    board = chess.Board()
    master = _StubMaster([(chess.Move.from_uci("e2e4"), 1.0),
                          (chess.Move.from_uci("d2d4"), 0.5)])
    trainer = L3Trainer(config=L3Config(seed=1, search_depth=1, top_k=5))
    result = trainer.train_on_master(board, master)
    assert result == {"applied": True, "k": 2}
    assert trainer.search_updates == 2


def test_train_on_master_reports_not_applied_for_empty_targets():
    board = chess.Board()
    trainer = L3Trainer(config=L3Config(seed=1, search_depth=1, top_k=5))
    assert trainer.train_on_master(board, _StubMaster([])) == {"applied": False, "k": 0}
    assert trainer.search_updates == 0


def test_train_on_search_feedback_is_identical_to_train_on_targets():
    # The refactor must not change the update: feeding the same targets through
    # train_on_targets has to produce bit-identical weights.
    board = chess.Board()
    via_feedback = L3Trainer(config=L3Config(seed=7, search_depth=2, top_k=5))
    via_targets = L3Trainer(config=L3Config(seed=7, search_depth=2, top_k=5))

    via_feedback.train_on_search_feedback(board, depth=2)
    targets = via_targets.search_feedback(board, depth=2)
    via_targets.train_on_targets(board, targets)

    for name in _WEIGHT_NAMES:
        left = getattr(via_feedback.policy.perceptron, name)
        right = getattr(via_targets.policy.perceptron, name)
        assert np.array_equal(left, right), f"refactor changed {name}"
    assert via_feedback.search_updates == via_targets.search_updates
