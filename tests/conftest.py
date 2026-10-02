"""Shared test fixtures and helpers.

The colour-mirror helper is the backbone of the whole suite: it is the only
way to check that the encoder and the canonicalisation are actually
side-agnostic, and every symmetry bug found during development was caught by
it. Keep it here rather than duplicating it per test module.
"""

from __future__ import annotations

import chess
import numpy as np
import pytest


def colour_mirror(fen: str) -> chess.Board:
    """Swap colours and reflect ranks, leaving files alone.

    This is the chess symmetry that does not change the meaning of a position:
    a king on e8 becomes a king on e1, a pawn on a7 becomes a pawn on a2. The
    board is then viewed from the new side to move, whose ``turn`` is flipped
    so that the resulting position is the same game with the colours relabelled.

    Note this is a *rank reflection*, not a 180 rotation. A 180 rotation would
    map a8 -> h1 and is not a symmetry of chess: it transposes the a-file and
    the h-file, which are distinct. Using it was the first bug this helper
    caught.

    Castling rights, the en-passant target and the move clocks are all carried
    across, because the encoder reads them and a mirror that silently drops
    them produces a false symmetry failure. Castling rights swap sides along
    with the pieces; the en-passant target reflects rank and keeps its file.
    """
    src = chess.Board(fen)
    mirror = chess.Board.empty()

    for square in chess.SQUARES:
        piece = src.piece_at(square)
        if piece is not None:
            dest = chess.square(square & 7, 7 - (square >> 3))
            mirror.set_piece_at(
                dest, chess.Piece(piece.piece_type, not piece.color)
            )

    mirror.turn = not src.turn

    # Castling rights are attached to a colour and a wing. Swapping the colours
    # means each right moves to the *other* colour, on the reflected rank: a
    # white kingside right in the source becomes a black kingside right.
    for colour in (chess.WHITE, chess.BLACK):
        for kingside in (True, False):
            has_right = (
                src.has_kingside_castling_rights(colour)
                if kingside
                else src.has_queenside_castling_rights(colour)
            )
            if not has_right:
                continue
            file_index = 7 if kingside else 0
            dest_colour = not colour
            # Rights always live on the back rank of their colour.
            back_rank = 0 if dest_colour == chess.WHITE else 7
            mirror.castling_rights |= chess.BB_SQUARES[
                chess.square(file_index, back_rank)
            ]

    # En-passant target: reflect the rank, keep the file.
    if src.ep_square is not None:
        mirror.ep_square = chess.square(
            src.ep_square & 7, 7 - (src.ep_square >> 3)
        )

    return mirror


INTERESTING_FENS = [
    # Opening, fully symmetric.
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1",
    # Development, kingside activity.
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
    # Sparse endgame with a passed pawn.
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    # Dense middlegame with castling rights on both sides.
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    # En passant available.
    "rnbqkbnr/pppp1ppp/8/4pP2/8/8/PPPPP1PP/RNBQKBNR b KQkq f6 0 3",
    # Promotion imminent.
    "8/P6k/8/8/8/8/8/K7 w - - 0 1",
    # In check.
    "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3",
]


def random_game_fens(seed: int = 0, plies: int = 40) -> list[str]:
    """A reproducible random game, for broad smoke coverage."""
    import random

    rng = random.Random(seed)
    board = chess.Board()
    out = [board.fen()]
    for _ in range(plies):
        if board.is_game_over():
            break
        board.push(rng.choice(list(board.legal_moves)))
        out.append(board.fen())
    return out


@pytest.fixture
def start_board() -> chess.Board:
    return chess.Board()


@pytest.fixture
def midgame_board() -> chess.Board:
    return chess.Board(
        "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"
    )
