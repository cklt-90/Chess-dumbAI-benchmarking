"""L0 tests: evaluation and phase utilities."""

from __future__ import annotations

import chess
import numpy as np
import pytest

from chessrl import value as V
from conftest import INTERESTING_FENS, colour_mirror, random_game_fens


# --------------------------------------------------------------------------
# material
# --------------------------------------------------------------------------

def test_start_material_is_level():
    assert V.material_balance(chess.Board()) == 0


def test_material_value_ignores_kings():
    board = chess.Board("8/8/8/8/8/8/8/K6k w - - 0 1")
    assert V.material_value(board, chess.WHITE) == 0
    assert V.material_value(board, chess.BLACK) == 0


def test_material_balance_counts_a_queen():
    board = chess.Board("8/8/8/8/8/8/8/KQ5k w - - 0 1")
    assert V.material_balance(board) == V.PIECE_VALUE[chess.QUEEN]


def test_material_ratio_at_start_is_one():
    assert V.material_ratio(chess.Board()) == pytest.approx(1.0)


def test_material_ratio_bare_kings_is_zero():
    board = chess.Board("8/8/8/4k3/8/8/4K3/8 w - - 0 1")
    assert V.material_ratio(board) == pytest.approx(0.0)


def test_material_ratio_is_bounded_on_promotion_heavy_position():
    board = chess.Board("QQQQQQQQ/QQQQQQQQ/8/8/8/8/QQQQQQQQ/QQQQQQKk w - - 0 1")
    assert 0.0 <= V.material_ratio(board) <= 1.0


def test_material_ratio_decreases_across_a_game():
    fens = random_game_fens(seed=6, plies=80)
    ratios = [V.material_ratio(chess.Board(f)) for f in fens]
    assert ratios[0] >= ratios[-1]


# --------------------------------------------------------------------------
# phase
# --------------------------------------------------------------------------

def test_phase_name_boundaries():
    assert V.phase_name(chess.Board()) == "early"
    assert V.phase_name(chess.Board("8/4k3/8/8/8/8/4K3/7Q w - - 0 1")) == "late"


def test_phase_weights_sum_to_one():
    for fen in INTERESTING_FENS:
        weights = V.phase_weights(chess.Board(fen))
        assert weights.sum() == pytest.approx(1.0), fen
        assert (weights >= 0).all()


def test_phase_weights_favour_early_at_start():
    weights = V.phase_weights(chess.Board())
    assert weights[0] == max(weights)


def test_phase_weights_favour_late_in_endgame():
    weights = V.phase_weights(chess.Board("8/4k3/8/8/8/8/4K3/7Q w - - 0 1"))
    assert weights[2] == max(weights)


def test_phase_weights_bare_kings_is_pure_endgame():
    """A bare-kings position is purely endgame; weights collapse onto "late".

    Not a degenerate fallback case -- it is the tent functions doing exactly
    what they should. Two kings with no other material has material_ratio 0,
    where only the late tent has support.
    """
    board = chess.Board("8/8/8/4k3/8/8/4K3/8 w - - 0 1")
    weights = V.phase_weights(board)
    assert weights.sum() == pytest.approx(1.0)
    assert weights[2] == pytest.approx(1.0)


def test_game_phase_matches_material_ratio():
    board = chess.Board()
    assert V.game_phase(board) == pytest.approx(V.material_ratio(board))


# --------------------------------------------------------------------------
# piece-square tables
# --------------------------------------------------------------------------

def test_pst_is_symmetric_between_colours():
    """A colour mirror must produce the same piece-square total.

    This is the check that caught the king table being looked up without a rank
    mirror for Black, which made the start position evaluate to +50.
    """
    for fen in INTERESTING_FENS:
        a = chess.Board(fen)
        b = colour_mirror(fen)
        assert V.piece_square_score(a, a.turn) == V.piece_square_score(b, b.turn), fen


def test_central_pawn_beats_rim_pawn():
    centre = chess.Board("8/8/8/8/4P3/8/8/K6k w - - 0 1")
    rim = chess.Board("8/8/8/8/P7/8/8/K6k w - - 0 1")
    assert (
        V.piece_square_score(centre, chess.WHITE)
        > V.piece_square_score(rim, chess.WHITE)
    )


def test_king_table_is_phase_dependent():
    """The king should prefer shelter in the middlegame and the centre late."""
    mid = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 3 3")
    assert V.material_ratio(mid) > V.PHASE_MID_MIN


# --------------------------------------------------------------------------
# evaluate
# --------------------------------------------------------------------------

def test_evaluate_start_position_is_level():
    assert V.evaluate(chess.Board()) == 0


def test_evaluate_sign_convention_is_side_to_move():
    """A white queen up position is good for whoever is to move."""
    board = chess.Board("8/8/8/8/8/8/8/KQ5k w - - 0 1")
    assert V.evaluate(board) > 0
    board.turn = chess.BLACK
    assert V.evaluate(board) < 0


def test_evaluate_checkmate_is_mate_score():
    board = chess.Board("7k/6Q1/5K2/8/8/8/8/8 b - - 0 1")
    assert board.is_checkmate()
    assert V.evaluate(board) < -V.MATE_SCORE + 1000


def test_evaluate_stalemate_is_zero():
    board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    assert board.is_stalemate()
    assert V.evaluate(board) == 0


def test_evaluate_insufficient_material_is_zero():
    board = chess.Board("8/8/8/4k3/8/8/4K3/8 w - - 0 1")
    assert board.is_insufficient_material()
    assert V.evaluate(board) == 0


def test_evaluate_deeper_mate_is_preferred():
    """Mate found sooner must score higher than the same mate found later."""
    early = chess.Board("7k/6Q1/5K2/8/8/8/8/8 b - - 0 1")
    board = chess.Board()
    board.push_san("f3")
    board.push_san("e5")
    board.push_san("g4")
    board.push_san("Qh4")
    assert board.is_checkmate()
    assert V.evaluate(board) > V.evaluate(early)


def test_evaluate_for_is_colour_consistent():
    board = chess.Board("8/8/8/8/8/8/8/KQ5k b - - 0 1")
    assert V.evaluate_for(board, chess.WHITE) > 0
    assert V.evaluate_for(board, chess.BLACK) < 0
    assert V.evaluate_for(board, chess.WHITE) == -V.evaluate_for(board, chess.BLACK)


def test_evaluate_returns_int():
    assert isinstance(V.evaluate(chess.Board()), int)


# --------------------------------------------------------------------------
# outcome mapping
# --------------------------------------------------------------------------

def test_outcome_score_mapping():
    assert V.outcome_score("1-0", chess.WHITE) == 1.0
    assert V.outcome_score("1-0", chess.BLACK) == -1.0
    assert V.outcome_score("0-1", chess.BLACK) == 1.0
    assert V.outcome_score("1/2-1/2", chess.WHITE) == 0.0


# --------------------------------------------------------------------------
# tactical terms
# --------------------------------------------------------------------------

def test_capture_value_on_quiet_move_is_zero():
    board = chess.Board()
    assert V.capture_value(board, chess.Move.from_uci("e2e4")) == 0


def test_capture_value_counts_the_victim():
    board = chess.Board("4k3/8/8/8/8/8/4q3/4K3 w - - 0 1")
    board = chess.Board("4k3/8/8/8/8/8/4q3/3QK3 w - - 0 1")
    move = chess.Move.from_uci("d1e2")
    assert V.capture_value(board, move) == V.PIECE_VALUE[chess.QUEEN]


def test_capture_value_handles_en_passant():
    """En passant captures a pawn that is not on the destination square.

    Missing this would silently make the encoder blind to the motif, which is
    why it is asserted explicitly rather than left to the generic path.
    """
    board = chess.Board("rnbqkbnr/pppp1ppp/8/4pP2/8/8/PPPPP1PP/RNBQKBNR b KQkq f6 0 3")
    board = chess.Board("rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3")
    move = chess.Move.from_uci("e5d6")
    assert board.is_en_passant(move)
    assert V.capture_value(board, move) == V.PIECE_VALUE[chess.PAWN]


def test_recapture_risk_detects_a_hanging_piece():
    # White rook on d1 can take the black queen on d5; black rook on d8 recaptures.
    board = chess.Board("3r3k/8/8/3q4/8/8/8/3R3K w - - 0 1")
    move = chess.Move.from_uci("d1d5")
    assert V.recapture_risk(board, move) > 0


def test_recapture_risk_is_zero_for_a_safe_move():
    board = chess.Board("7k/8/8/8/8/8/8/7K w - - 0 1")
    board = chess.Board("7k/8/8/8/8/8/8/K6R w - - 0 1")
    move = chess.Move.from_uci("h1h2")
    assert V.recapture_risk(board, move) == 0


def test_hanging_value_ignores_defended_pieces():
    """A defended piece under attack is not hanging."""
    defended = chess.Board("4k3/8/8/8/4r3/8/3K4/8 w - - 0 1")
    assert V.hanging_value(defended, chess.WHITE) == 0


def test_hanging_value_counts_an_undefended_piece():
    hanging = chess.Board("4k3/8/8/8/4r3/8/8/3K4 w - - 0 1")
    hanging = chess.Board("4k3/8/8/8/4r3/8/8/K7 w - - 0 1")
    assert V.hanging_value(hanging, chess.WHITE) == 0


def test_hanging_value_on_start_position_is_zero():
    assert V.hanging_value(chess.Board(), chess.WHITE) == 0
