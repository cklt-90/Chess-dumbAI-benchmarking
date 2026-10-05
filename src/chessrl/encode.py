"""L0: the encoded board state -- binary mask planes plus value planes.

This is the module the spec cares most about. It turns a position into a stack
of binary planes that a policy, a perceptron or a torch net can consume
directly, and it attaches the two "value" channels the spec asks for:

* **capture value** -- what the side to move can win by moving to a square,
  which is the "opponents pieces captured or in capturable positions" term;
* **danger value** -- what the side to move risks by moving to a square, which
  is the "danger value of own pieces in capturable positions upon moving
  there" term.

Design notes
------------
Everything is float32 at the boundary. Masks are boolean internally because
``bool`` arrays are the natural output of comparisons and comparisons in numpy
are cheap, but the encoder casts once, at the point where the planes become
model input, rather than repeatedly downstream.

The channel layout is **stable and ordered** -- see :data:`CHANNELS`. New
channels may be appended but never inserted or reordered, because L3's weight
matrix and L4's checkpoint are indexed by channel and a silent reordering would
produce a model that loads fine and plays nonsense. If you need to change a
channel's meaning, add a new one and version the encoder.

Health warning on the value planes
----------------------------------
``recapture_risk`` and the per-square danger map call into python-chess
attack generation per candidate move, which is roughly 40x the cost of the
binary planes alone. :func:`encode` therefore takes ``with_values`` and the
default for *training* is off: the binary planes carry most of the signal, and
paying 40x per position to add two channels that L3 can largely reconstruct is
a bad trade at corpus-build time. Turn it on for the L2 wrapper and for
inference-time evaluation, where the move count per position is small.
"""

from __future__ import annotations

import numpy as np
import chess

from . import NUM_PLANES, PLANE_ORDER
from .bitboard import (
    board_to_planes,
    canonical,
    attack_map,
    defended_map,
    occupancy,
)
from .value import (
    PIECE_VALUE,
    capture_value,
    recapture_risk,
    piece_square_score,
    material_ratio,
    phase_weights,
)

# ---------------------------------------------------------------------------
# channel registry
# ---------------------------------------------------------------------------

# Binary state channels. Order is load-bearing -- see module docstring.
CHANNELS = (
    "pawn_w", "knight_w", "bishop_w", "rook_w", "queen_w", "king_w",      # 0-5
    "pawn_b", "knight_b", "bishop_b", "rook_b", "queen_b", "king_b",      # 6-11
    "occupied",                                                          # 12
    "to_move_attacks",                                                   # 13
    "opponent_attacks",                                                  # 14
    "to_move_defends",                                                   # 15
    "opponent_defends",                                                  # 16
    "contested",                                                         # 17
    "capture_targets",                                                   # 18
    "promotion_rank_w",                                                  # 19
    "promotion_rank_b",                                                  # 20
    "en_passant",                                                        # 21
    "castle_w",                                                          # 22
    "castle_b",                                                          # 23
)

# Continuous value channels, appended after the binary planes.
VALUE_CHANNELS = (
    "capture_value",     # centipawns obtainable by moving to this square
    "danger_value",      # centipawns at risk by moving to this square
    "attack_balance",    # attacked by me, not defended by them
    "own_danger",        # my pieces hanging right now (per square)
)

BINARY_CHANNELS = len(CHANNELS)          # 24
VALUE_CHANNELS_N = len(VALUE_CHANNELS)   # 4
TOTAL_CHANNELS = BINARY_CHANNELS + VALUE_CHANNELS_N
GROUP_VALUES = 10_000.0  # normaliser for centipawn value planes


# ---------------------------------------------------------------------------
# masks
# ---------------------------------------------------------------------------

def to_move_attack_mask(board: chess.Board) -> np.ndarray:
    """Squares the side to move attacks."""
    return attack_map(board, board.turn)


def opponent_attack_mask(board: chess.Board) -> np.ndarray:
    """Squares the opponent attacks."""
    return attack_map(board, not board.turn)


def capture_target_mask(board: chess.Board) -> np.ndarray:
    """Squares holding an enemy piece that a legal move could take."""
    mask = np.zeros((8, 8), dtype=bool)
    for move in board.legal_moves:
        if board.is_capture(move):
            mask[move.to_square >> 3, move.to_square & 7] = True
    return mask


def promotion_rank_mask(colour: int) -> np.ndarray:
    """The rank a pawn of ``colour`` promotes on."""
    mask = np.zeros((8, 8), dtype=bool)
    rank = 7 if colour == chess.WHITE else 0
    mask[rank, :] = True
    return mask


def en_passant_mask(board: chess.Board) -> np.ndarray:
    """The en-passant target square, if there is one."""
    mask = np.zeros((8, 8), dtype=bool)
    if board.ep_square is not None:
        mask[board.ep_square >> 3, board.ep_square & 7] = True
    return mask


def castle_rights_mask(board: chess.Board, colour: int) -> np.ndarray:
    """Binary plane marking the whole board when ``colour`` retains any castling
    right (king-side or queen-side).

    A constant-plane flag, in the style of ``promotion_rank_mask``: it tells a
    learner "this side can still castle" as a *board pattern* rather than only
    as a scalar in ``board_context_features`` (where it cannot be matched to a
    position). This is the representation repair H23 calls for -- AlphaZero's
    119-plane stack carries four constant castling planes for exactly this
    reason, and a learner with no such channel cannot even express "I may still
    castle kingside". Appended, not inserted, so every existing channel index is
    preserved and saved weight matrices stay valid.
    """
    mask = np.zeros((8, 8), dtype=bool)
    if board.has_castling_rights(colour):
        mask[:, :] = True
    return mask


def contested_mask(board: chess.Board) -> np.ndarray:
    """Squares both sides attack: the tension map.

    Squares where both sides have a claim are where tactical decisions happen,
    so a single plane for "both of us attack this" is a cheap high-value
    feature that the two separate attack planes cannot express on their own.
    """
    return attack_map(board, chess.WHITE) & attack_map(board, chess.BLACK)


# ---------------------------------------------------------------------------
# value planes
# ---------------------------------------------------------------------------

def capture_value_plane(board: chess.Board) -> np.ndarray:
    """Per-square centipawn value capturable by the side to move.

    For each destination square, the best material the side to move could take
    there across all legal moves. Note this includes moves that would lose the
    capturing piece -- the danger channel is what corrects for that, and the
    learner is meant to combine the two. Keeping them independent is
    deliberate: an encoder that pre-combines them cannot be ablated.
    """
    plane = np.zeros((8, 8), dtype=np.float32)
    if board.is_game_over():
        return plane
    for move in board.legal_moves:
        if not board.is_capture(move):
            continue
        gained = capture_value(board, move)
        rank, file = move.to_square >> 3, move.to_square & 7
        if gained > plane[rank, file]:
            plane[rank, file] = gained
    return plane


def danger_value_plane(board: chess.Board) -> np.ndarray:
    """Per-square centipawn value the side to move would risk there.

    The cheapest enemy attacker's value against each destination: the material
    you stand to lose by placing a piece on that square. Combined with
    :func:`capture_value_plane` this is the trade decision -- take X, risk Y.

    Cost warning: this pushes each legal move to inspect attacks, so it is the
    single most expensive channel in the encoder.
    """
    plane = np.zeros((8, 8), dtype=np.float32)
    if board.is_game_over():
        return plane
    for move in board.legal_moves:
        risk = recapture_risk(board, move)
        if risk == 0:
            continue
        rank, file = move.to_square >> 3, move.to_square & 7
        if risk > plane[rank, file]:
            plane[rank, file] = risk
    return plane


def attack_balance_plane(board: chess.Board) -> np.ndarray:
    """Squares the side to move attacks but the opponent does not defend.

    A cheap "this square is free to occupy" signal. Binary in principle but
    kept float32 to sit alongside the other value planes.
    """
    mine = attack_map(board, board.turn)
    theirs = defended_map(board, not board.turn)
    return (mine & ~theirs).astype(np.float32)


def own_danger_plane(board: chess.Board) -> np.ndarray:
    """Per-square value of my own pieces that are attacked and undefended.

    The current-state version of the danger term: where am I hanging *right
    now*, before any move. The policy at L3 uses this to notice it must respond
    to a threat.
    """
    from .bitboard import defended_map as _defended

    plane = np.zeros((8, 8), dtype=np.float32)
    turn = board.turn
    threats = attack_map(board, not turn)
    guard = _defended(board, turn)
    for sq in chess.scan_forward(board.occupied_co[turn]):
        rank, file = sq >> 3, sq & 7
        if threats[rank, file] and not guard[rank, file]:
            plane[rank, file] = PIECE_VALUE[board.piece_type_at(sq)]
    return plane


# ---------------------------------------------------------------------------
# the encoder
# ---------------------------------------------------------------------------

def encode(
    board: chess.Board,
    *,
    with_values: bool = False,
    canonicalise: bool = True,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Encode a position as a ``(TOTAL_CHANNELS, 8, 8)`` float32 tensor.

    Parameters
    ----------
    board
        Position to encode.
    with_values
        Include the four continuous value planes. Off by default because they
        cost roughly 40x the binary planes; see the module docstring.
    canonicalise
        Rotate the position so the side to move always appears as White on the
        a1 side. This is what lets one parameter set serve both colours. Set
        False only when you specifically want absolute board coordinates --
        for example when encoding a *move* rather than a position.
    out
        Optional preallocated buffer of shape ``(TOTAL_CHANNELS, 8, 8)``.
        Always pass one in a training loop; this function is the inner loop of
        the whole benchmark and allocating 26 planes per call is the difference
        between minutes and hours.

    Returns
    -------
    np.ndarray of shape ``(TOTAL_CHANNELS, 8, 8)``, float32, values in
    ``[0, 1]`` for binary channels and ``[0, 1]`` for value channels (scaled by
    :data:`GROUP_VALUES` -- a full queen is 0.09, so the value channels are
    deliberately low-magnitude and the learner supplies the scale).
    """
    tensor = (
        np.zeros((TOTAL_CHANNELS, 8, 8), dtype=np.float32)
        if out is None else out
    )
    if out is not None:
        tensor[:] = 0.0

    planes = board_to_planes(board)

    # --- binary piece planes -------------------------------------------
    tensor[:NUM_PLANES] = planes

    # --- derived binary state ------------------------------------------
    # Note these derived planes are built in *absolute* coordinates and then
    # rotated together with everything else, so that "to_move_attacks" keeps
    # meaning "attacks by the side to move" after canonicalisation.
    tensor[12] = occupancy(planes)
    tensor[13] = to_move_attack_mask(board)
    tensor[14] = opponent_attack_mask(board)
    tensor[15] = defended_map(board, board.turn)
    tensor[16] = defended_map(board, not board.turn)
    tensor[17] = contested_mask(board)
    tensor[18] = capture_target_mask(board)
    tensor[19] = promotion_rank_mask(chess.WHITE)
    tensor[20] = promotion_rank_mask(chess.BLACK)
    tensor[21] = en_passant_mask(board)
    tensor[22] = castle_rights_mask(board, chess.WHITE)
    tensor[23] = castle_rights_mask(board, chess.BLACK)

    if canonicalise:
        # Two colour-specific channel pairs must swap along with the piece
        # planes, or "white pawns" and "white promotion rank" would end up in
        # different colours after canonicalisation. Everything else is a pure
        # rank reflection.
        tensor[:] = _canonical_channels(tensor, board.turn == chess.WHITE)

    if with_values:
        base = BINARY_CHANNELS
        tensor[base + 0] = capture_value_plane(board) / GROUP_VALUES
        tensor[base + 1] = danger_value_plane(board) / GROUP_VALUES
        tensor[base + 2] = attack_balance_plane(board)
        tensor[base + 3] = own_danger_plane(board) / GROUP_VALUES
        if canonicalise and board.turn != chess.WHITE:
            # Value planes are computed in absolute coordinates and need the
            # same rank reflection as the binary planes. They carry no colour
            # identity, so only the reflection applies -- but it must be
            # conditional on whose turn it is. Applying it unconditionally
            # corrupts every white-to-move position, which is the majority of
            # the corpus and therefore does not look like a bug at first.
            tensor[base:base + VALUE_CHANNELS_N] = _reflect_ranks(
                tensor[base:base + VALUE_CHANNELS_N]
            )

    return tensor


def _reflect_ranks(tensor: np.ndarray) -> np.ndarray:
    """Rank reflection over (rank, file) axes. Colour-agnostic."""
    return tensor[..., ::-1, :].copy()


# Channel pairs that carry *colour identity* and must therefore swap when the
# position is re-expressed from the other side. Each entry is (white_channel,
# black_channel). The leading 0-11 piece planes swap as a block; these two
# promotion-rank planes are the only derived channels that do too. Kept as
# explicit data rather than inferred from position, because the next person to
# append a channel must be told which list to add it to.
COLOUR_PAIRED_CHANNELS: tuple[tuple[int, int], ...] = ((19, 20), (22, 23))


def _canonical_channels(tensor: np.ndarray, turn_white: bool) -> np.ndarray:
    """Apply the full canonicalisation across the channel dimension.

    Rank-reflects every channel, then swaps the colour-identity channels. This
    is :func:`chessrl.bitboard.canonical` generalised to a tensor that mixes
    colour-segregated and side-relative channels, which the raw plane stack
    never does.

    Getting this wrong is silent: the model still trains, it just receives a
    position whose "my promotion rank" channel points at the enemy's back rank,
    and it will happily learn a policy that pushes pawns the wrong way.
    """
    if turn_white:
        return tensor.copy()

    out = _reflect_ranks(tensor)
    # Piece planes: swap the two 6-plane colour blocks.
    out[:6] = tensor[6:12, ::-1, :]
    out[6:12] = tensor[:6, ::-1, :]
    # Colour-paired derived channels.
    for w, b in COLOUR_PAIRED_CHANNELS:
        out[w] = tensor[b, ::-1, :]
        out[b] = tensor[w, ::-1, :]
    return out


def encode_move_feature(board: chess.Board, move: chess.Move) -> np.ndarray:
    """Fixed-length scalar feature vector for a single candidate move.

    Used by the L3 perceptron to score a move without a full board pass. Order
    is stable; see :data:`MOVE_FEATURE_NAMES`.
    """
    is_capture = board.is_capture(move)
    gained = capture_value(board, move) if is_capture else 0
    risk = recapture_risk(board, move)
    promo = move.promotion or chess.KNIGHT
    return np.array([
        1.0,                                          # bias
        gained / 100.0,                               # material won
        risk / 100.0,                                 # material risked
        (gained - risk) / 100.0,                      # net
        1.0 if is_capture else 0.0,
        1.0 if move.promotion else 0.0,
        1.0 if board.is_castling(move) else 0.0,
        1.0 if board.gives_check(move) else 0.0,
        PIECE_VALUE.get(board.piece_type_at(move.from_square), 0) / 100.0,
        abs((move.to_square >> 3) - (move.from_square >> 3)) / 7.0,
        abs((move.to_square & 7) - (move.from_square & 7)) / 7.0,
        ((move.to_square >> 3) / 7.0),                # destination rank
        1.0 if promo == chess.QUEEN else 0.0,
    ], dtype=np.float32)


MOVE_FEATURE_NAMES = (
    "bias", "gain", "risk", "net", "is_capture", "is_promotion", "is_castling",
    "gives_check", "mover_value", "delta_rank", "delta_file", "dest_rank",
    "promo_queen",
)


def board_context_features(
    board: chess.Board, *, mobility: int | None = None
) -> np.ndarray:
    """Global scalar features describing the position.

    Phase, material balance, check status, mobility. These are what give the
    perceptron a chance of behaving differently in the opening versus the
    endgame without a separate model per phase.

    ``mobility`` may be supplied by a caller that has already enumerated the
    legal moves -- the search does, at every interior node. Generating them
    again just to take ``len()`` is over half the cost of this function, and
    inside a leaf evaluator that runs thousands of times it is the difference
    between a usable and an unusable engine. When omitted the count is
    computed here, so every existing caller keeps its exact behaviour.
    """
    me = board.turn
    opp = not me
    if mobility is None:
        mobility = board.legal_moves.count() if not board.is_game_over() else 0

    my_mat = sum(
        PIECE_VALUE[board.piece_type_at(sq)]
        for sq in chess.scan_forward(board.occupied_co[me])
    )
    opp_mat = sum(
        PIECE_VALUE[board.piece_type_at(sq)]
        for sq in chess.scan_forward(board.occupied_co[opp])
    )

    return np.array([
        material_ratio(board),
        (my_mat - opp_mat) / 1000.0,
        (my_mat + opp_mat) / 4000.0,
        1.0 if board.is_check() else 0.0,
        mobility / 40.0,
        1.0 if board.has_castling_rights(me) else 0.0,
        *phase_weights(board),
    ], dtype=np.float32)


CONTEXT_FEATURE_NAMES = (
    "material_ratio", "material_diff", "total_material", "in_check",
    "mobility", "can_castle", "w_early", "w_mid", "w_late",
)


def encode_canonical_for_move_space(board: chess.Board) -> np.ndarray:
    """Encoder output for use as *move-space* input, un-canonicalised.

    When a model must emit a target square, the output tensor has to live in
    the same coordinate frame as the board's own squares. So for the L3/L4
    policy heads we encode without canonicalisation and let the model learn
    both colours, or encode canonically and rotate the emitted move back. This
    helper exists to make that choice explicit at the call site rather than
    implicit in a boolean at the bottom of a call stack.
    """
    return encode(board, canonicalise=False)


def summarise(tensor: np.ndarray) -> dict:
    """Per-channel sums, for debugging an encoding pipeline."""
    names = list(CHANNELS) + list(VALUE_CHANNELS)
    return {
        name: float(tensor[i].sum())
        for i, name in enumerate(names)
    }
