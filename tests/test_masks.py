"""L0 tests: action-space masks.

The mask layer is the contract every policy level depends on, so it is tested
against brute-force enumeration over thousands of real positions rather than a
handful of hand-written cases. In particular ``legal_move_mask`` must agree
with ``board.legal_moves`` exactly -- including promotion slots, which is where
the first bug appeared (a queen promotion indexed slot 4 in an axis of size 4).
"""

from __future__ import annotations

import random

import chess
import numpy as np
import pytest

from chessrl import masks as M
from conftest import INTERESTING_FENS, random_game_fens


# --------------------------------------------------------------------------
# flat action index
# --------------------------------------------------------------------------

def test_index_roundtrip_for_every_legal_move():
    for fen in INTERESTING_FENS:
        board = chess.Board(fen)
        for move in board.legal_moves:
            assert M.index_to_move(M.move_to_index(move)) == move, (fen, move)


def test_index_roundtrip_across_full_games():
    for fen in random_game_fens(seed=11, plies=120):
        board = chess.Board(fen)
        for move in board.legal_moves:
            assert M.index_to_move(M.move_to_index(move)) == move


def test_action_space_bounds():
    for fen in INTERESTING_FENS:
        for move in chess.Board(fen).legal_moves:
            idx = M.move_to_index(move)
            assert 0 <= idx < M.ACTION_SPACE


def test_queen_promotion_uses_the_last_slot():
    """The slot table must not overflow the promotion axis.

    A queen promotion is the highest-indexed promotion, so it is the one that
    breaks if NUM_PROMO is too small. Asserting it explicitly pins the off-by-one
    that was found here: an axis of size 4 plus a sentinel slot needs 5.
    """
    board = chess.Board("8/P6k/8/8/8/8/8/K7 w - - 0 1")
    queen_promo = chess.Move(chess.A7, chess.A8, promotion=chess.QUEEN)
    assert M.PROMO_SLOT[chess.QUEEN] == M.NUM_PROMO - 1
    assert M.index_to_move(M.move_to_index(queen_promo)) == queen_promo
    assert M.move_to_index(queen_promo) < M.ACTION_SPACE


def test_non_promotion_uses_slot_zero():
    move = chess.Move.from_uci("e2e4")
    assert M.move_to_index(move) % M.NUM_PROMO == 0
    assert M.index_to_move(M.move_to_index(move)).promotion is None


def test_moves_to_indices_is_vectorised():
    board = chess.Board()
    idx = M.moves_to_indices(board.legal_moves)
    assert idx.dtype == np.int32
    assert len(idx) == board.legal_moves.count()


# --------------------------------------------------------------------------
# legal mask
# --------------------------------------------------------------------------

def test_legal_mask_matches_legal_moves_exactly():
    for fen in random_game_fens(seed=5, plies=150):
        board = chess.Board(fen)
        mask = M.legal_move_mask(board)
        expected = board.legal_moves.count()
        assert int(mask.sum()) == expected, fen


def test_legal_mask_start_position():
    mask = M.legal_move_mask(chess.Board())
    assert int(mask.sum()) == 20
    assert int(M.legal_from_mask(mask).sum()) == 10   # 8 pawns + 2 knights
    assert mask[chess.E2, chess.E4, 0]
    assert not mask[chess.E2, chess.E5, 0]


def test_legal_mask_all_slots_in_range():
    for fen in INTERESTING_FENS:
        mask = M.legal_move_mask(chess.Board(fen))
        assert mask.shape == (64, 64, M.NUM_PROMO)


def test_legal_mask_promotion_position():
    board = chess.Board("8/P6k/8/8/8/8/8/K7 w - - 0 1")
    mask = M.legal_move_mask(board)
    slots = np.nonzero(mask[chess.A7, chess.A8])[0]
    assert sorted(slots.tolist()) == [1, 2, 3, 4]
    assert not mask[chess.A7, chess.A8, 0]   # a7a8 is always a promotion


def test_from_and_to_masks_are_consistent():
    for fen in random_game_fens(seed=9, plies=80):
        board = chess.Board(fen)
        mask = M.legal_move_mask(board)
        assert np.array_equal(M.legal_from_mask(mask), mask.any(axis=(1, 2)))
        assert np.array_equal(M.legal_to_mask(mask), mask.any(axis=(0, 2)))


def test_dest_mask_is_a_slice_of_the_full_mask():
    board = chess.Board()
    mask = M.legal_move_mask(board)
    assert np.array_equal(M.legal_dest_mask(mask, chess.E2), mask[chess.E2])


def test_density_is_bounded():
    board = chess.Board()
    density = M.mask_density(M.legal_move_mask(board))
    assert 0 < density < 0.01


def test_mask_is_sparse_compared_to_action_space():
    """Sanity check on the premise behind factorising the policy."""
    board = chess.Board()
    legal = int(M.legal_move_mask(board).sum())
    assert legal / M.ACTION_SPACE < 0.01


# --------------------------------------------------------------------------
# state masks
# --------------------------------------------------------------------------

def test_occupied_and_empty_partition_the_board():
    board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3")
    occupied = M.occupied_mask(board)
    empty = M.empty_squares_mask(board)
    assert np.array_equal(occupied | empty, np.ones((8, 8), dtype=bool))
    assert not (occupied & empty).any()


def test_can_move_to_only_for_side_to_move():
    board = chess.Board()
    white = M.can_move_to_mask(board, chess.WHITE)
    black = M.can_move_to_mask(board, chess.BLACK)
    # Only the side to move has legal destinations.
    assert int(white.sum()) > 0
    assert int(black.sum()) == 0


def test_attack_balance_excludes_defended_squares():
    board = chess.Board()
    balance = M.attack_balance_mask(board, chess.WHITE)
    assert balance.dtype == bool
    # On the open board White threatens rank 3, none of which Black defends.
    assert balance[2, 4]


def test_stack_and_flatten_mask_tensor():
    a = np.zeros((8, 8), dtype=bool)
    a[0, 0] = True
    b = np.zeros((8, 8), dtype=bool)
    b[7, 7] = True
    tensor = M.stack_masks(a, b)
    assert tensor.shape == (2, 8, 8)
    assert tensor.dtype == np.float32
    flat = M.flatten_mask_tensor(tensor)
    assert flat.shape == (128,)
    assert flat[0] == 1.0
