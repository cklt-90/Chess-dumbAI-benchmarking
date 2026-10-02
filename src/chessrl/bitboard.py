"""L0: bitboard and plane encoding.

The whole benchmark is built on one representation: a ``(12, 8, 8)`` boolean
array of piece planes, plus a handful of derived binary masks. Everything is
numpy; nothing here allocates per-square Python objects on the hot path.

Conventions
-----------
* Square index follows python-chess: ``0 = a1``, ``7 = h1``, ``56 = a8``,
  ``63 = h8``. So ``rank = sq >> 3``, ``file = sq & 7``.
* Plane index follows :data:`chessrl.PLANE_ORDER`. White pieces are planes
  0-5, black 6-11. ``plane ^ 6`` flips colour; ``plane % 6`` gives piece type.
* Board orientation in the array mirrors python-chess exactly: row 0 of the
  array is rank 1, not rank 8. This is deliberate -- it means ``ndarray``
  coordinates are pure ``(rank, file)`` and no flipping happens anywhere,
  which removes a whole class of off-by-one bugs. Rendering flips for display
  only.

Why arrays and not 64-bit ints
------------------------------
python-chess already exposes an integer bitboard per piece type via
``Board.pieces_mask``. We keep that as the *authority* and mirror into arrays,
rather than reimplementing attack tables. The array form is what lets us do
whole-board arithmetic (danger maps, weighted evaluations) in one vectorised
op. :func:`pack_planes` / :func:`unpack_planes` convert between the two worlds
and are used by the cache.
"""

from __future__ import annotations

import numpy as np
import chess

from . import NUM_PLANES, PLANE_ORDER

# (type, colour) -> plane index, precomputed once.
_PLANE_OF = {sym: i for i, sym in enumerate(PLANE_ORDER)}
_PIECE_TYPE_OF_PLANE = ("p", "n", "b", "r", "q", "k") * 2


def empty_planes() -> np.ndarray:
    """A fresh all-False ``(12, 8, 8)`` stack of piece planes."""
    return np.zeros((NUM_PLANES, 8, 8), dtype=bool)


def empty_mask() -> np.ndarray:
    """A fresh all-False ``(8, 8)`` square mask."""
    return np.zeros((8, 8), dtype=bool)


# --------------------------------------------------------------------------
# square <-> array coordinate helpers
# --------------------------------------------------------------------------

def square_to_rc(sq: int) -> tuple[int, int]:
    """``a1=0`` -> ``(row, col)`` in array coords. Row is rank, col is file."""
    return sq >> 3, sq & 7


def rc_to_square(row: int, col: int) -> int:
    """``(row, col)`` array coords -> python-chess square index."""
    return (row << 3) + col


def squares_to_mask(squares) -> np.ndarray:
    """Turn any iterable of square indices into an ``(8, 8)`` boolean mask."""
    mask = empty_mask()
    for sq in squares:
        mask[sq >> 3, sq & 7] = True
    return mask


def mask_to_squares(mask: np.ndarray):
    """Yield the square indices of every True cell, ascending. Vectorised."""
    if not mask.any():
        return
    rows = mask.any(axis=1).nonzero()[0]
    for row in rows:
        for col in mask[row].nonzero()[0]:
            yield (row << 3) + int(col)


def square_names(mask: np.ndarray) -> list[str]:
    """Debug helper: the algebraic names of every True cell."""
    return [chess.square_name(sq) for sq in mask_to_squares(mask)]


# --------------------------------------------------------------------------
# board -> planes
# --------------------------------------------------------------------------

def board_to_planes(board: chess.Board, out: np.ndarray | None = None) -> np.ndarray:
    """Mirror a python-chess board into the ``(12, 8, 8)`` plane stack.

    Pass ``out`` to fill a preallocated buffer in place; otherwise a new array
    is returned. This is the single hot spot for state construction, so the
    inner loop uses ``scan_forward`` over the integer bitboards rather than
    iterating ``board.piece_map()``.
    """
    planes = empty_planes() if out is None else out
    if out is not None:
        planes[:] = False

    for plane_idx, symbol in enumerate(PLANE_ORDER):
        colour = chess.WHITE if plane_idx < 6 else chess.BLACK
        piece_type = (
            chess.PAWN, chess.KNIGHT, chess.BISHOP,
            chess.ROOK, chess.QUEEN, chess.KING,
        )[plane_idx % 6]
        bb = board.pieces_mask(piece_type, colour)
        for sq in chess.scan_forward(bb):
            planes[plane_idx, sq >> 3, sq & 7] = True
    return planes


def planes_to_board(planes: np.ndarray) -> chess.Board:
    """Inverse of :func:`board_to_planes`. Used by the cache to rebuild state.

    Note this reconstructs *placement* only -- castling rights, en-passant
    target and the halfmove clock are not recoverable from planes alone. The
    cache therefore stores the full ``Board`` and uses planes only for keys.
    """
    board = chess.Board.empty()
    for plane_idx in range(NUM_PLANES):
        colour = chess.WHITE if plane_idx < 6 else chess.BLACK
        piece_type = (
            chess.PAWN, chess.KNIGHT, chess.BISHOP,
            chess.ROOK, chess.QUEEN, chess.KING,
        )[plane_idx % 6]
        for sq in mask_to_squares(planes[plane_idx]):
            board.set_piece_at(sq, chess.Piece(piece_type, colour))
    return board


# --------------------------------------------------------------------------
# colour / orientation operations -- all view-level, never mutate callers
# --------------------------------------------------------------------------

def flip_colour(planes: np.ndarray, block: int = 6) -> np.ndarray:
    """Swap the two colour blocks of the stack: white planes <-> black.

    ``block`` is the number of planes per colour (6 in the raw piece stack).
    Arrays that stack *more* channels than the 12 piece planes must not be
    passed through here directly -- use :func:`canonical` with an explicit
    ``n_colour_planes`` instead, since only the leading ``2*block`` planes are
    colour-segregated. Everything after that is already side-to-move-relative
    and only needs rotating, not swapping.
    """
    flipped = planes.copy()
    flipped[:block] = planes[block:2 * block]
    flipped[block:2 * block] = planes[:block]
    return flipped


def rotate_180(planes: np.ndarray) -> np.ndarray:
    """Geometric 180-degree rotation of every plane, colours unchanged.

    WARNING: this is almost never what you want for colour mirroring. It maps
    a8 -> h1, i.e. it turns the board round *and* flips left-right, which is not
    a chess symmetry -- the a-file and the h-file are not interchangeable
    (only queenside/kingside are, and only under the full 180 that includes the
    colour swap). Kept for completeness; :func:`flip_ranks` is the mirror.
    """
    return planes[:, ::-1, ::-1].copy()


def flip_ranks(planes: np.ndarray) -> np.ndarray:
    """Reflect ranks only: a8 -> a1, d8 -> d1, e1 -> e8.

    This IS the correct colour mirror up to the colour swap. Files are
    preserved, which is what makes it a genuine reflection: a black king on e8
    maps to e1, a black queen on d8 maps to d1, and a black pawn on a7 maps to
    a2. Applying this to the piece planes *and* swapping colours yields the
    position with the colours exchanged, still viewed from White's side of the
    board.
    """
    return planes[:, ::-1, :].copy()


def canonical(planes: np.ndarray, turn_white: bool, block: int = 6) -> np.ndarray:
    """Position as seen by the side to move, always from "their" a1 side.

    This is the input form for every learner in the benchmark. It gives white
    and black identical statistics, so a model trained on white-to-move
    generalises to black-to-move with no extra parameters.

    For black to move the transform is: reflect the ranks, and swap the two
    colour blocks. Files are preserved -- the mirror is a rank reflection, not
    a 180 rotation. Using ``rotate_180`` here is a subtle and silent bug: it
    passes on a symmetric position like the opening array, then transposes
    queenside and kingside everywhere else, so the model learns nothing about
    file geometry.

    Works on any channel count. ``block`` is how many planes make up one
    colour's piece set; only the leading ``2 * block`` planes are colour
    swapped, and everything beyond is rank-reflected only.
    """
    if turn_white:
        return planes.copy()

    out = np.empty_like(planes)
    # Colour blocks: swap AND reflect ranks, in a single copy each.
    out[:block] = planes[block:2 * block, ::-1, :]
    out[block:2 * block] = planes[:block, ::-1, :]
    # Everything after the colour blocks is side-relative already: reflect only.
    out[2 * block:] = planes[2 * block:, ::-1, :]
    return out


def apply_canonical(plane_stack: np.ndarray, turn_white: bool) -> np.ndarray:
    """Map a quantity between canonical and real-board coordinates.

    :func:`canonical` is a rank reflection plus a colour-block swap. Any
    *move-space* quantity -- a logit vector over squares, a move mask, a policy
    over source squares -- lives only in the spatial half of that transform, so
    converting it is a pure rank reflection. The colour swap is deliberately NOT
    applied here: a move emitted in canonical coordinates refers to a square,
    and squares are not coloured.

    This function is its own inverse, so the same call converts in both
    directions. Format-agnostic: it accepts ``(8, 8)``, ``(n, 8, 8)``, or any
    array whose last two axes are (rank, file); it reflects only those axes.
    """
    if turn_white:
        return plane_stack
    return plane_stack[..., ::-1, :].copy()


# --------------------------------------------------------------------------
# packing -- for hashing and for the "all objects as numpy arrays" requirement
# --------------------------------------------------------------------------

def pack_planes(planes: np.ndarray) -> bytes:
    """Pack the plane stack into a compact, hashable byte string.

    ``np.packbits`` collapses 8 squares per byte, so a full position becomes
    12 bytes. That is the key type for the position cache -- cheaper to hash
    and compare than the 768-byte bool array.
    """
    return np.packbits(planes, axis=None).tobytes()


def unpack_planes(blob: bytes) -> np.ndarray:
    """Inverse of :func:`pack_planes`."""
    bits = np.unpackbits(np.frombuffer(blob, dtype=np.uint8))
    return bits[: NUM_PLANES * 64].reshape(NUM_PLANES, 8, 8).astype(bool)


def planes_to_int(planes: np.ndarray) -> int:
    """Fold the whole placement into one Python int (768 bits).

    Slower than :func:`pack_planes` but convenient for set membership and for
    writing positions out to a plain-text corpus.
    """
    packed = np.packbits(planes, axis=None)
    return int.from_bytes(packed.tobytes(), "big")


def int_to_planes(value: int) -> np.ndarray:
    """Inverse of :func:`planes_to_int`."""
    raw = value.to_bytes(NUM_PLANES * 8, "big")
    bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8))
    return bits[: NUM_PLANES * 64].reshape(NUM_PLANES, 8, 8).astype(bool)


# --------------------------------------------------------------------------
# occupancy and attack surface
# --------------------------------------------------------------------------

def occupancy(planes: np.ndarray, colour: int | None = None) -> np.ndarray:
    """Mask of occupied squares, optionally restricted to one colour."""
    if colour is None:
        return planes.any(axis=0)
    lo, hi = (0, 6) if colour == chess.WHITE else (6, 12)
    return planes[lo:hi].any(axis=0)


def attack_map(board: chess.Board, colour: int) -> np.ndarray:
    """Mask of every square ``colour`` attacks, ignoring pins.

    Used by the danger-value encoding: "am I hanging here" is a question about
    the opponent's attack set, not about legality. ``board.attacks`` per piece
    is far cheaper than enumerating legal moves and is the right coarse signal.
    """
    mask = empty_mask()
    for sq in chess.scan_forward(board.occupied_co[colour]):
        # board.attacks() returns a SquareSet; .mask gives the int bitboard
        # that scan_forward needs.
        for target in chess.scan_forward(board.attacks(sq).mask):
            mask[target >> 3, target & 7] = True
    return mask


def defended_map(board: chess.Board, colour: int) -> np.ndarray:
    """Mask of own squares that are attacked by another own piece."""
    own = board.occupied_co[colour]
    mask = empty_mask()
    for sq in chess.scan_forward(own):
        for target in chess.scan_forward(board.attacks(sq).mask):
            if chess.BB_SQUARES[target] & own:
                mask[target >> 3, target & 7] = True
    return mask
