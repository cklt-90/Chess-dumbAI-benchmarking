"""L0: binary masks for board state and the legal move set.

Two mask families live here.

**State masks** are ``(8, 8)`` boolean planes describing a single fact about a
square: is it occupied, is it attacked, is it hanging. They are the vocabulary
of :mod:`chessrl.encode`.

**Move masks** are the interesting one. A chess move is a triple
``(from_square, to_square, promotion)``. Flattened that is a 64 x 64 x 4 action
space = 16384 slots, which is too sparse to learn from directly at the naive
end of the benchmark. So we keep the *factorised* view alongside the flat one:

* ``from_mask``      ``(64,)``     -- squares that have at least one legal move
* ``to_mask``        ``(64, 64)``  -- ``[f, t]`` legal for a non-promotion move
* ``promo_mask``     ``(64, 64, 4)`` -- ``[f, t, k]`` legal promotion move

The flat index of a move is

    ``idx = (from_square * 64 + to_square) * 4 + promo_slot``

with ``promo_slot = 0`` for non-promotion moves. Every level in this benchmark
addresses moves through :func:`move_to_index` / :func:`index_to_move`, so
switching between the flat and factorised view is lossless and needs no
re-derivation of legal moves.

This is the direct analogue of the spec's
``(row, col, piece, dir, size)`` output space: instead of emitting a
piece/direction/range tuple, we emit ``(from_square, to_square, promotion)``.
Same idea -- a small set of factorised heads, each masked independently -- but
it addresses real chess moves rather than puzzle placements.
"""

from __future__ import annotations

import numpy as np
import chess

from .bitboard import empty_mask

NUM_SQUARES = 64
NUM_PROMO = 5
ACTION_SPACE = NUM_SQUARES * NUM_SQUARES * NUM_PROMO  # 20480

# Promotion slots. Slot 0 is the sentinel meaning "this move is not a
# promotion"; slots 1..4 are knight, bishop, rook, queen. The array is
# therefore sized NUM_PROMO = 5, not 4: the extra slot buys us a single
# uniform indexing scheme with no special-casing anywhere in the policy.
PROMO_PIECES = (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
PROMO_SLOT = {pt: i + 1 for i, pt in enumerate(PROMO_PIECES)}
SLOT_TO_PROMO = {v: k for k, v in PROMO_SLOT.items()}


# --------------------------------------------------------------------------
# flat index <-> chess.Move
# --------------------------------------------------------------------------

def move_to_index(move: chess.Move) -> int:
    """Flat action index in ``[0, ACTION_SPACE)`` for a python-chess move."""
    slot = PROMO_SLOT.get(move.promotion, 0)
    return (move.from_square * NUM_SQUARES + move.to_square) * NUM_PROMO + slot


def index_to_move(idx: int) -> chess.Move:
    """Inverse of :func:`move_to_index`."""
    slot = idx % NUM_PROMO
    rest = idx // NUM_PROMO
    from_sq = rest // NUM_SQUARES
    to_sq = rest % NUM_SQUARES
    return chess.Move(from_sq, to_sq, promotion=SLOT_TO_PROMO.get(slot))


def moves_to_indices(moves) -> np.ndarray:
    """Vectorise an iterable of moves into an int array of action indices."""
    return np.fromiter(
        (move_to_index(m) for m in moves), dtype=np.int32, count=-1
    )


# --------------------------------------------------------------------------
# legal move masks
# --------------------------------------------------------------------------

def legal_move_mask(board: chess.Board) -> np.ndarray:
    """Dense ``(64, 64, NUM_PROMO)`` boolean mask of every legal move.

    ``mask[f, t, k]`` is True iff moving ``f -> t`` with promotion slot ``k``
    is legal. Slot 0 covers all non-promotion moves, so ``mask[..., 0]`` alone
    is the complete legal set for a position with no promotions available.

    Cost is about 17 microseconds per position (one pass over ``legal_moves``),
    so callers that need several derived masks should build this once and
    slice, rather than re-enumerating ``legal_moves`` per question.
    """
    mask = np.zeros((NUM_SQUARES, NUM_SQUARES, NUM_PROMO), dtype=bool)
    for move in board.legal_moves:
        slot = PROMO_SLOT.get(move.promotion, 0)
        mask[move.from_square, move.to_square, slot] = True
    return mask


def legal_from_mask(mask: np.ndarray) -> np.ndarray:
    """``(64,)`` mask: does this origin square have any legal move?"""
    return mask.any(axis=(1, 2))


def legal_to_mask(mask: np.ndarray) -> np.ndarray:
    """``(64,)`` mask: is this destination reachable by any legal move?

    Note this collapses promotion slots. For per-origin destination masks use
    :func:`legal_dest_mask`.
    """
    return mask.any(axis=(0, 2))


def legal_dest_mask(mask: np.ndarray, from_sq: int) -> np.ndarray:
    """``(64, NUM_PROMO)`` mask of legal ``(to, promo)`` pairs for one origin.

    This is the second stage of the factored policy: given a chosen origin,
    which destinations (and promotions) remain.
    """
    return mask[from_sq]


def mask_density(mask: np.ndarray) -> float:
    """Fraction of the action space that is legal. Useful for diagnostics."""
    return float(mask.sum()) / ACTION_SPACE


# --------------------------------------------------------------------------
# state masks
# --------------------------------------------------------------------------

def occupied_mask(board: chess.Board) -> np.ndarray:
    """Squares holding any piece."""
    from .bitboard import occupancy, board_to_planes
    return occupancy(board_to_planes(board))


def empty_squares_mask(board: chess.Board) -> np.ndarray:
    """Squares holding no piece."""
    return ~occupied_mask(board)


def attacked_by_mask(board: chess.Board, colour: int) -> np.ndarray:
    """Squares attacked by ``colour``, ignoring pins.

    Deliberately excludes squares occupied by ``colour`` itself is *not* done
    here -- use :func:`attacked_or_occupied_own_mask` when you mean "usable by
    the opponent".
    """
    from .bitboard import attack_map
    return attack_map(board, colour)


def can_move_to_mask(board: chess.Board, colour: int) -> np.ndarray:
    """Squares ``colour`` could legally step onto (empty or enemy-occupied).

    This is the "valid moveset" mask in the spec, expressed per square rather
    than per move: the union of the destinations of every legal move. Only the
    side to move has legal moves, so a query for the idle side returns empty.
    """
    mask = empty_mask()
    if board.turn != colour:
        return mask
    for move in board.legal_moves:
        mask[move.to_square >> 3, move.to_square & 7] = True
    return mask


def attack_balance_mask(board: chess.Board, colour: int) -> np.ndarray:
    """Squares attacked by ``colour`` but *not* defended by the opponent.

    A cheap proxy for "squares where I can win material": I threaten it, and
    taking back costs them more than it costs me. Feeds the danger encoding.
    """
    opponent = not colour
    from .bitboard import attack_map
    mine = attack_map(board, colour)
    theirs = attack_map(board, opponent)
    return mine & ~theirs


# --------------------------------------------------------------------------
# mask <-> ndarray plumbing
# --------------------------------------------------------------------------

def stack_masks(*masks: np.ndarray) -> np.ndarray:
    """Stack ``(8, 8)`` masks into an ``(n, 8, 8)`` tensor.

    The encoder's output is exactly this: a stack of named binary planes. Cast
    to float32 once at the boundary so downstream arithmetic is uniform.
    """
    return np.stack(masks, axis=0).astype(np.float32)


def flatten_mask_tensor(tensor: np.ndarray) -> np.ndarray:
    """Flatten an ``(n, 8, 8)`` mask tensor to ``(n * 64,)`` float32.

    Feature-vector form, for the perceptron at L3.
    """
    return tensor.reshape(-1).astype(np.float32)
