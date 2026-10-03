"""L0 tests: the encoded board state.

Beyond the usual shape and range checks, this module pins the two properties
that the rest of the benchmark silently depends on:

1. **Channel layout stability.** L3's weight matrix and L4's checkpoints are
   indexed by channel number, so a reordering produces a model that loads
   cleanly and plays nonsense. ``test_channel_layout_is_pinned`` is the guard.
2. **Colour-mirror equivalence with values enabled.** The value planes are the
   most expensive channels and the easiest to get wrong, because a
   mis-canonicalised value plane still looks plausible. See the module
   docstring in ``encode.py`` for the bug this caught.
"""

from __future__ import annotations

import chess
import numpy as np
import pytest

from chessrl import encode as E
from chessrl import value as V
from conftest import INTERESTING_FENS, colour_mirror, random_game_fens


# --------------------------------------------------------------------------
# layout stability
# --------------------------------------------------------------------------

def test_channel_layout_is_pinned():
    """Freeze the channel order. Changing this invalidates saved weights."""
    assert E.CHANNELS[:12] == (
        "pawn_w", "knight_w", "bishop_w", "rook_w", "queen_w", "king_w",
        "pawn_b", "knight_b", "bishop_b", "rook_b", "queen_b", "king_b",
    )
    assert E.BINARY_CHANNELS == 22
    assert E.VALUE_CHANNELS_N == 4
    assert E.TOTAL_CHANNELS == 26
    assert E.VALUE_CHANNELS == (
        "capture_value", "danger_value", "attack_balance", "own_danger",
    )


def test_colour_paired_channels_point_at_real_colour_channels():
    for white_ch, black_ch in E.COLOUR_PAIRED_CHANNELS:
        assert E.CHANNELS[white_ch].endswith("_w")
        assert E.CHANNELS[black_ch].endswith("_b")
        assert white_ch < E.BINARY_CHANNELS
        assert black_ch < E.BINARY_CHANNELS


# --------------------------------------------------------------------------
# shape and range
# --------------------------------------------------------------------------

def test_encode_shape_and_dtype():
    tensor = E.encode(chess.Board())
    assert tensor.shape == (E.TOTAL_CHANNELS, 8, 8)
    assert tensor.dtype == np.float32


def test_encode_values_are_in_range():
    for fen in INTERESTING_FENS:
        tensor = E.encode(chess.Board(fen), with_values=True)
        assert tensor.min() >= 0.0, fen
        assert tensor.max() <= 1.0, fen


def test_encode_into_preallocated_buffer():
    buf = np.zeros((E.TOTAL_CHANNELS, 8, 8), dtype=np.float32)
    out = E.encode(chess.Board(), out=buf)
    assert out is buf
    assert buf.max() > 0


def test_preallocated_buffer_is_fully_overwritten():
    """A stale buffer must not leak values from the previous position."""
    buf = np.full((E.TOTAL_CHANNELS, 8, 8), 9.0, dtype=np.float32)
    E.encode(chess.Board(), out=buf)
    assert buf.max() <= 1.0


def test_encode_does_not_mutate_the_board():
    board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3")
    fen = board.fen()
    E.encode(board, with_values=True)
    assert board.fen() == fen


# --------------------------------------------------------------------------
# channel semantics
# --------------------------------------------------------------------------

def test_piece_planes_match_the_board():
    tensor = E.encode(chess.Board(), canonicalise=False)
    assert int(tensor[:12].sum()) == 32
    assert int(tensor[12].sum()) == 32          # occupied


def test_promotion_rank_channels():
    tensor = E.encode(chess.Board(), canonicalise=False)
    assert int(tensor[19].sum()) == 8           # white promotes on rank 8
    assert int(tensor[20].sum()) == 8           # black promotes on rank 1
    # Array row 7 is rank 8, row 0 is rank 1.
    assert tensor[19, 7, :].all()
    assert tensor[20, 0, :].all()


def test_en_passant_channel_is_empty_without_en_passant():
    tensor = E.encode(chess.Board(), canonicalise=False)
    assert int(tensor[21].sum()) == 0


def test_en_passant_channel_marks_the_target():
    board = chess.Board("rnbqkbnr/pppp1ppp/8/4pP2/8/8/PPPPP1PP/RNBQKBNR b KQkq f6 0 3")
    tensor = E.encode(board, canonicalise=False)
    assert int(tensor[21].sum()) == 1
    assert tensor[21, 5, 5]                     # f6 = rank 6, file f


def test_contested_channel_is_an_intersection():
    board = chess.Board(
        "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"
    )
    tensor = E.encode(board, canonicalise=False)
    expected = E.attack_map(board, chess.WHITE) & E.attack_map(board, chess.BLACK)
    assert np.array_equal(tensor[17] > 0, expected)


def test_capture_target_channel_tracks_legal_captures():
    board = chess.Board("4k3/8/8/8/8/8/4q3/3QK3 w - - 0 1")
    tensor = E.encode(board, with_values=True, canonicalise=False)
    assert tensor[18, 1, 4] > 0                 # e2 is capturable
    assert tensor[E.BINARY_CHANNELS, 1, 4] > 0  # and the capture is worth 900


def test_capture_value_plane_is_zero_when_nothing_to_take():
    tensor = E.encode(chess.Board(), with_values=True, canonicalise=False)
    assert tensor[E.BINARY_CHANNELS].sum() == 0


def test_own_danger_plane_flags_a_hanging_piece():
    board = chess.Board("4k3/8/8/8/4r3/8/8/K7 w - - 0 1")
    plane = E.own_danger_plane(board)
    # The white king on a1 is attacked by the rook on e4 along rank 4? No --
    # the rook on e4 attacks the a4-h4 rank and the e-file, so a1 is safe.
    assert plane.sum() == 0


def test_attack_balance_plane_is_binary():
    tensor = E.encode(chess.Board(), with_values=True, canonicalise=False)
    balance = tensor[E.BINARY_CHANNELS + 2]
    assert set(np.unique(balance).tolist()) <= {0.0, 1.0}


# --------------------------------------------------------------------------
# canonicalisation with values
# --------------------------------------------------------------------------

def test_white_to_move_values_are_not_reflected():
    """The white-to-move path must be a no-op on the value planes.

    This is the specific bug that the unconditional-reflection guard caused:
    white-to-move positions had their value planes flipped, which looked fine
    until compared against the mirror.
    """
    board = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")
    encoded = E.encode(board, with_values=True)[E.BINARY_CHANNELS + 1]
    raw = E.danger_value_plane(board) / E.GROUP_VALUES
    assert np.array_equal(encoded, raw)


@pytest.mark.parametrize("fen", INTERESTING_FENS)
def test_values_survive_colour_mirror(fen):
    src = chess.Board(fen)
    mirror = colour_mirror(fen)
    assert np.array_equal(
        E.encode(src, with_values=True),
        E.encode(mirror, with_values=True),
    ), fen


def test_values_survive_colour_mirror_across_a_game():
    for fen in random_game_fens(seed=8, plies=50):
        src = chess.Board(fen)
        mirror = colour_mirror(fen)
        assert np.array_equal(
            E.encode(src, with_values=True),
            E.encode(mirror, with_values=True),
        ), fen


def test_canonical_false_keeps_absolute_coordinates():
    board = chess.Board("8/8/8/8/8/8/8/KQ5k w - - 0 1")
    tensor = E.encode(board, canonicalise=False)
    assert tensor[4, 0, 1]          # white queen on b1
    assert tensor[11, 0, 7]         # black king on h1


# --------------------------------------------------------------------------
# move and context features
# --------------------------------------------------------------------------

def test_move_feature_length_and_finiteness():
    board = chess.Board()
    for move in board.legal_moves:
        feats = E.encode_move_feature(board, move)
        assert feats.shape == (len(E.MOVE_FEATURE_NAMES),)
        assert np.isfinite(feats).all()


def test_move_feature_marks_captures():
    board = chess.Board("4k3/8/8/8/8/8/4q3/3QK3 w - - 0 1")
    move = chess.Move.from_uci("d1e2")
    feats = E.encode_move_feature(board, move)
    assert feats[E.MOVE_FEATURE_NAMES.index("is_capture")] == 1.0
    assert feats[E.MOVE_FEATURE_NAMES.index("gain")] > 0


def test_move_feature_marks_castling():
    board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    castle = chess.Move.from_uci("e1g1")
    feats = E.encode_move_feature(board, castle)
    assert feats[E.MOVE_FEATURE_NAMES.index("is_castling")] == 1.0


def test_context_features_length_and_range():
    for fen in INTERESTING_FENS:
        feats = E.board_context_features(chess.Board(fen))
        assert feats.shape == (len(E.CONTEXT_FEATURE_NAMES),)
        assert np.isfinite(feats).all()


def test_context_features_detect_check():
    board = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
    feats = E.board_context_features(board)
    assert feats[E.CONTEXT_FEATURE_NAMES.index("in_check")] == 1.0


def test_summarise_reports_every_channel():
    summary = E.summarise(E.encode(chess.Board(), with_values=True))
    assert len(summary) == E.TOTAL_CHANNELS


# --------------------------------------------------------------------------
# encode_canonical_for_move_space (G1 audit: previously untested public API)
# --------------------------------------------------------------------------

def test_encode_canonical_for_move_space_is_the_uncanonicalised_encode():
    """The helper is a named alias for ``encode(board, canonicalise=False)``,
    so the "this is move-space input" choice is explicit at the call site."""
    for fen in INTERESTING_FENS:
        board = chess.Board(fen)
        assert np.array_equal(
            E.encode_canonical_for_move_space(board),
            E.encode(board, canonicalise=False),
        ), fen


def test_encode_canonical_for_move_space_differs_for_black_to_move():
    """For a Black-to-move position the canonical form reflects + colour-swaps,
    so the move-space (uncanonicalised) encode must differ. For a White-to-move
    position canonical is the identity, so they coincide -- the difference must
    therefore be pinned on a black-to-move FEN, not a white-to-move one."""
    board = chess.Board("rnbqkbnr/ppp1pppp/8/3p4/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1")
    move_space = E.encode_canonical_for_move_space(board)
    canonical = E.encode(board)
    assert not np.array_equal(move_space, canonical)


# --------------------------------------------------------------------------
# direct per-plane assertions (G7 audit: previously exercised only indirectly)
# --------------------------------------------------------------------------

def _squares(mask):
    from chessrl import bitboard as B
    return {chess.square_name(s) for s in B.mask_to_squares(mask.astype(bool))}


def test_promotion_rank_mask_is_the_back_rank_for_each_colour():
    white = E.promotion_rank_mask(chess.WHITE)
    black = E.promotion_rank_mask(chess.BLACK)
    # White promotes on rank 8 -> row 7; Black on rank 1 -> row 0. All files, no others.
    assert white[7].all() and not white[:7].any()
    assert black[0].all() and not black[1:].any()


def test_en_passant_mask_marks_only_the_ep_square():
    board = chess.Board()
    board.push_san("e4")  # Black to move, en-passant target e3
    assert _squares(E.en_passant_mask(board)) == {"e3"}
    # A board with no en-passant target yields an empty plane.
    assert not E.en_passant_mask(chess.Board()).any()


def test_capture_target_mask_marks_the_capturable_enemy():
    # White pawn e4 can capture the black pawn on d5.
    board = chess.Board("8/8/8/3p4/4P3/8/8/8 w - - 0 1")
    assert _squares(E.capture_target_mask(board)) == {"d5"}


def test_attack_masks_split_side_to_move_from_opponent():
    # A lone white knight on d5: White attacks its eight squares, Black none.
    board = chess.Board("8/8/8/3N4/8/8/8/8 w - - 0 1")
    knight = {"c3", "e3", "b4", "f4", "b6", "f6", "c7", "e7"}
    assert _squares(E.to_move_attack_mask(board)) == knight
    assert _squares(E.opponent_attack_mask(board)) == set()


def test_contested_mask_marks_squares_both_sides_attack():
    # White knight d5 and black knight a2: both can reach b4 and c3, and the
    # contested plane is exactly the intersection of the two attack maps.
    board = chess.Board("8/8/8/3N4/8/8/n7/8 w - - 0 1")
    white_attacks = _squares(E.to_move_attack_mask(board))
    black_attacks = _squares(E.opponent_attack_mask(board))
    contested = _squares(E.contested_mask(board))
    assert contested == (white_attacks & black_attacks)
    assert contested == {"b4", "c3"}


def test_capture_value_plane_reports_the_best_capture():
    # White pawn e4 can take a black rook on d5: the plane must report the
    # rook's centipawn value at d5 and that it is the maximum on the board.
    board = chess.Board("8/8/8/3r4/4P3/8/8/8 w - - 0 1")
    plane = E.capture_value_plane(board)
    assert plane[chess.D5 >> 3, chess.D5 & 7] == pytest.approx(500.0)
    assert plane.max() == pytest.approx(500.0)
