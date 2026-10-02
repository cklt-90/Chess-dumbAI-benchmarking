"""L0 tests: bitboard planes, canonicalisation and packing.

The symmetry tests here are the most important tests in the repository. A
broken canonicalisation does not crash and does not obviously degrade play --
it just quietly teaches every model that the queenside and kingside are
interchangeable, or that the back ranks are the same colour. Two such bugs were
found while writing this module, both caught by the mirror helper in conftest.
"""

from __future__ import annotations

import chess
import numpy as np
import pytest

from chessrl import PLANE_ORDER
from chessrl import bitboard as B
from conftest import INTERESTING_FENS, colour_mirror, random_game_fens


# --------------------------------------------------------------------------
# board <-> planes
# --------------------------------------------------------------------------

def test_plane_count_matches_order():
    assert len(PLANE_ORDER) == 12
    assert B.empty_planes().shape == (12, 8, 8)


def test_start_position_placement():
    planes = B.board_to_planes(chess.Board())
    assert int(planes.sum()) == 32
    # White pawns on rank 2, black on rank 7, array rows 1 and 6.
    assert planes[0, 1, :].all()
    assert planes[6, 6, :].all()
    assert B.square_names(planes[4]) == ["d1"]   # white queen
    assert B.square_names(planes[5]) == ["e1"]   # white king


def test_planes_to_board_roundtrip():
    for fen in INTERESTING_FENS:
        board = chess.Board(fen)
        rebuilt = B.planes_to_board(B.board_to_planes(board))
        assert rebuilt.board_fen() == board.board_fen(), fen


def test_occupancy_by_colour():
    planes = B.board_to_planes(chess.Board())
    assert int(B.occupancy(planes, chess.WHITE).sum()) == 16
    assert int(B.occupancy(planes, chess.BLACK).sum()) == 16
    assert int(B.occupancy(planes).sum()) == 32


# --------------------------------------------------------------------------
# packing
# --------------------------------------------------------------------------

def test_pack_roundtrip_and_size():
    planes = B.board_to_planes(chess.Board())
    blob = B.pack_planes(planes)
    # 12 planes x 64 squares = 768 bits = 96 bytes. This matters: the cache key
    # hash cost is proportional to it, so a regression here is a performance bug.
    assert len(blob) == 96
    assert np.array_equal(B.unpack_planes(blob), planes)


def test_int_roundtrip():
    planes = B.board_to_planes(chess.Board())
    assert np.array_equal(B.int_to_planes(B.planes_to_int(planes)), planes)


def test_pack_distinguishes_positions():
    a = B.pack_planes(B.board_to_planes(chess.Board("8/8/8/8/8/8/8/K6k w - - 0 1")))
    b = B.pack_planes(B.board_to_planes(chess.Board("8/8/8/8/8/8/8/K5k1 w - - 0 1")))
    assert a != b


# --------------------------------------------------------------------------
# canonicalisation -- the critical correctness properties
# --------------------------------------------------------------------------

def test_canonical_is_identity_for_white_to_move():
    planes = B.board_to_planes(chess.Board())
    assert np.array_equal(B.canonical(planes, turn_white=True), planes)


def test_canonical_maps_pieces_to_mirrored_squares():
    """Black's queen on d8 must land on d1, king e8 on e1, pawns on rank 2.

    This is the specific assertion that fails if the transform is a 180
    rotation instead of a rank reflection: d8 would land on e1, not d1.
    """
    planes = B.board_to_planes(chess.Board())
    canon = B.canonical(planes, turn_white=False)
    assert B.square_names(canon[4]) == ["d1"]
    assert B.square_names(canon[5]) == ["e1"]
    assert B.square_names(canon[1]) == ["b1", "g1"]   # knights keep their files
    assert canon[0, 1, :].all()                        # black pawns on rank 2


def test_canonical_is_an_involution():
    for fen in INTERESTING_FENS:
        planes = B.board_to_planes(chess.Board(fen))
        once = B.canonical(planes, turn_white=False)
        twice = B.canonical(once, turn_white=False)
        assert np.array_equal(twice, planes), fen


def test_flip_ranks_preserves_files():
    planes = B.board_to_planes(chess.Board())
    flipped = B.flip_ranks(planes)
    # A file-reflecting transform would move the a1 rook to h-file. Assert it
    # does not.
    assert B.square_names(flipped[3]) == ["a8", "h8"]


def test_apply_canonical_is_self_inverse_and_file_preserving():
    plane = B.board_to_planes(chess.Board())[4]  # white queen plane
    sent = B.apply_canonical(plane, turn_white=False)
    assert B.square_names(sent) == ["d8"]
    back = B.apply_canonical(sent, turn_white=False)
    assert np.array_equal(back, plane)


def test_apply_canonical_accepts_3d_arrays():
    stack = B.board_to_planes(chess.Board())[:3]
    out = B.apply_canonical(stack, turn_white=False)
    assert out.shape == stack.shape


# --------------------------------------------------------------------------
# attack maps
# --------------------------------------------------------------------------

def test_attack_map_start_position():
    board = chess.Board()
    white = B.attack_map(board, chess.WHITE)
    # 16 pawn moves + 6 knight moves = 22 attacked squares.
    assert int(white.sum()) == 22


def test_defended_map_start_position():
    board = chess.Board()
    defended = B.defended_map(board, chess.WHITE)
    # Every piece on rank 1 bar the rooks is defended, plus all 8 pawns.
    assert int(defended.sum()) == 14
    assert not defended[0, 0]   # a1 rook is not defended


def test_attack_map_on_empty_board():
    board = chess.Board("8/8/8/8/4k3/8/8/8 w - - 0 1")
    assert int(B.attack_map(board, chess.WHITE).sum()) == 0


# --------------------------------------------------------------------------
# end-to-end symmetry
# --------------------------------------------------------------------------

@pytest.mark.parametrize("fen", INTERESTING_FENS)
def test_canonical_survives_colour_mirror(fen):
    """Encoding a position and its colour mirror must agree exactly."""
    from chessrl import encode as E

    src = chess.Board(fen)
    mirror = colour_mirror(fen)
    assert np.array_equal(E.encode(src), E.encode(mirror)), fen


def test_symmetry_holds_across_a_whole_random_game():
    from chessrl import encode as E

    for fen in random_game_fens(seed=3, plies=60):
        src = chess.Board(fen)
        mirror = colour_mirror(fen)
        assert np.array_equal(E.encode(src), E.encode(mirror)), fen
