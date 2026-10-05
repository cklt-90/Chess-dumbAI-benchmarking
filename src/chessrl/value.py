"""L0: evaluation and game-phase utilities.

Deliberately small and dependency-free. Every level of the benchmark that
needs to know "who is winning here" calls into this module, so a change in
evaluation is a one-file change and all levels move together -- which is
required for the L6 tournament to be a fair comparison.

Two things live here:

**Material and piece-square evaluation.** Used as the leaf function of
alpha-beta (L2/L5) and as the fallback when a learned model is weak (L3). It
is intentionally classical and boring: a benchmark of learning methods should
not be confounded by a clever hand-written eval. Values are centipawns.

**Game phase.** The spec's "early, mid and late game, mixed by ratio which
depends on movecount and remaining piece count". Phase here is a continuous
quantity in ``[0, 1]`` derived from remaining material, plus two thresholds
that discretise it. Material-driven rather than move-number-driven because
after a queen trade on move 10 the position is functionally an endgame, and a
movecount-based phase would mislabel it.
"""

from __future__ import annotations

import chess
import numpy as np

# Centipawn material values. These are the standard beginner numbers, kept
# because they are the least contentious baseline available.
PIECE_VALUE = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}

# Total non-pawn, non-king material at the start of a game: 2*(2*320+2*330+900)
# = 2*(640+660+900) = 2*2200 = 4400. Used to normalise the material ratio.
STARTING_MINOR_MAJOR = 4400

# Phase thresholds on ``material_ratio``. Chosen so that:
#   ratio >= 0.75  -> early   (roughly, no minor or major traded yet)
#   0.35 <= r < 0.75 -> middlegame
#   r < 0.35       -> endgame
PHASE_EARLY_MIN = 0.75
PHASE_MID_MIN = 0.35


def material_value(board: chess.Board, colour: int) -> int:
    """Total centipawn value of ``colour``'s material, king excluded."""
    total = 0
    for piece_type, value in PIECE_VALUE.items():
        if value == 0:
            continue
        total += value * chess.popcount(board.pieces_mask(piece_type, colour))
    return total


def material_balance(board: chess.Board) -> int:
    """White-minus-black material in centipawns, from White's point of view."""
    return (
        material_value(board, chess.WHITE) - material_value(board, chess.BLACK)
    )


def material_ratio(board: chess.Board) -> float:
    """Remaining minor+major material as a fraction of the opening total.

    ``1.0`` at the start position, ``0.0`` once every minor and major piece is
    off the board. In ``[0, 1]``, monotonically decreasing over a game except
    for promotions, which push it back up -- correctly, since promoting adds a
    real attacker.
    """
    remaining = 0
    for piece_type in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
        remaining += PIECE_VALUE[piece_type] * (
            chess.popcount(board.pieces_mask(piece_type, chess.WHITE))
            + chess.popcount(board.pieces_mask(piece_type, chess.BLACK))
        )
    return min(1.0, remaining / STARTING_MINOR_MAJOR)


def game_phase(board: chess.Board) -> float:
    """Continuous phase in ``[0, 1]``: ``1`` = opening, ``0`` = bare endgame.

    For L1.5 the spec wants "separate probability for early, mid and late game,
    mixed by ratio". The mixing weights come from a soft partition of this
    value, so a position at the boundary contributes to both phases rather than
    flipping abruptly -- which matters early in training when counts per bucket
    are small and a hard boundary would starve the buckets.
    """
    return material_ratio(board)


def phase_name(board: chess.Board) -> str:
    """Discretised phase label: ``"early"``, ``"mid"`` or ``"late"``."""
    r = material_ratio(board)
    if r >= PHASE_EARLY_MIN:
        return "early"
    if r >= PHASE_MID_MIN:
        return "mid"
    return "late"


def phase_weights(board: chess.Board) -> np.ndarray:
    """Soft phase mix as ``(early, mid, late)`` weights summing to 1.

    Tent functions over ``material_ratio`` with peaks at 1.0, 0.55 and 0.15.
    This is the "mix by ratio" the spec asks for: a position with ratio 0.55 is
    weighted 50/50 between early and mid rather than being assigned to one.
    """
    r = material_ratio(board)

    def tent(peak: float, width: float) -> float:
        return max(0.0, 1.0 - abs(r - peak) / width)

    w = np.array([tent(1.0, 0.45), tent(0.55, 0.40), tent(0.15, 0.30)],
                 dtype=np.float32)
    total = w.sum()
    if total <= 0:
        # Degenerate positions (e.g. promotion storms) fall back to uniform.
        return np.full(3, 1 / 3, dtype=np.float32)
    return w / total


# --------------------------------------------------------------------------
# piece-square tables
# --------------------------------------------------------------------------

# Tables are written from White's point of view with rank 8 first (the way
# chess literature prints them). Squares are indexed a8..h1 reading left to
# right, top to bottom, then flipped at load time to match our a1=0 convention.
_PST_SRC = {
    chess.PAWN: (
        0,  0,  0,  0,  0,  0,  0,  0,
        50, 50, 50, 50, 50, 50, 50, 50,
        10, 10, 20, 30, 30, 20, 10, 10,
        5,  5, 10, 25, 25, 10,  5,  5,
        0,  0,  0, 20, 20,  0,  0,  0,
        5, -5,-10,  0,  0,-10, -5,  5,
        5, 10, 10,-20,-20, 10, 10,  5,
        0,  0,  0,  0,  0,  0,  0,  0,
    ),
    chess.KNIGHT: (
        -50,-40,-30,-30,-30,-30,-40,-50,
        -40,-20,  0,  0,  0,  0,-20,-40,
        -30,  0, 10, 15, 15, 10,  0,-30,
        -30,  5, 15, 20, 20, 15,  5,-30,
        -30,  0, 15, 20, 20, 15,  0,-30,
        -30,  5, 10, 15, 15, 10,  5,-30,
        -40,-20,  0,  5,  5,  0,-20,-40,
        -50,-40,-30,-30,-30,-30,-40,-50,
    ),
    chess.BISHOP: (
        -20,-10,-10,-10,-10,-10,-10,-20,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -10,  0,  5, 10, 10,  5,  0,-10,
        -10,  5,  5, 10, 10,  5,  5,-10,
        -10,  0, 10, 10, 10, 10,  0,-10,
        -10, 10, 10, 10, 10, 10, 10,-10,
        -10,  5,  0,  0,  0,  0,  5,-10,
        -20,-10,-10,-10,-10,-10,-10,-20,
    ),
    chess.ROOK: (
        0,  0,  0,  0,  0,  0,  0,  0,
        5, 10, 10, 10, 10, 10, 10,  5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        0,  0,  0,  5,  5,  0,  0,  0,
    ),
    chess.QUEEN: (
        -20,-10,-10, -5, -5,-10,-10,-20,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -10,  0,  5,  5,  5,  5,  0,-10,
        -5,  0,  5,  5,  5,  5,  0, -5,
        0,  0,  5,  5,  5,  5,  0, -5,
        -10,  5,  5,  5,  5,  5,  0,-10,
        -10,  0,  5,  0,  0,  0,  0,-10,
        -20,-10,-10, -5, -5,-10,-10,-20,
    ),
    chess.KING: (
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -20,-30,-30,-40,-40,-30,-30,-20,
        -10,-20,-20,-20,-20,-20,-20,-10,
        20, 20,  0,  0,  0,  0, 20, 20,
        20, 30, 10,  0,  0, 10, 30, 20,
    ),
}

# King table for the endgame: centralise rather than hide. Blended in by phase.
_KING_ENDGAME_SRC = (
    -50,-40,-30,-20,-20,-30,-40,-50,
    -30,-20,-10,  0,  0,-10,-20,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-30,  0,  0,  0,  0,-30,-30,
    -50,-30,-30,-30,-30,-30,-30,-50,
)


def _load_pst(src: tuple) -> np.ndarray:
    """Load an a8-first table and index it a1-first as ``[rank, file]``.

    The source tuple reads rank 8 down to rank 1. Our arrays are rank 1
    upwards, so we reverse the rank order and reshape to ``(8, 8)``.
    """
    arr = np.array(src, dtype=np.int32).reshape(8, 8)  # [row0=rank8]
    return arr[::-1].copy()  # [row0=rank1]


PST = {pt: _load_pst(src) for pt, src in _PST_SRC.items()}
PST[chess.KING] = _load_pst(_KING_ENDGAME_SRC)  # replaced below by phase blend
KING_PST_MID = _load_pst(_PST_SRC[chess.KING])
KING_PST_END = _load_pst(_KING_ENDGAME_SRC)


def king_shelter_score(board: chess.Board, colour: int) -> int:
    """Pawn-shield and wing-safety bonus for ``colour``'s king.

    Gives every level that consults ``evaluate`` -- L2's search and the
    greedy-material policy -- a *reason* to castle, which is the missing
    incentive H23 diagnosed: castling is legal and in the mask, but with a
    material+PST-only eval nothing rewards it, so only the searching levels
    (which see king safety through search) ever do.

    The signal has three parts, all White-perspective and rank-mirrored for
    Black exactly like the piece-square tables:

    * pawns directly in front of and beside the king reward a intact shield;
    * an open king file (<=1 friendly pawn on it) is penalised as exposure;
    * the king off the exposed central files (d/e) is rewarded, so a castled
      king on the wing scores higher than an uncastled king stuck on e1 -- this
      is what turns "castle" from a neutral move into a slightly favourable one.

    The whole term is scaled by game phase (full in the opening, half in the
    bare endgame) so a castled shield does not wrongly dominate late play. It is
    deliberately small -- at most about a third of a pawn -- so it nudges
    without overriding material. Because it is a pure function of the board
    state (potential-based), it rewards castling through consequences, not via a
    per-move bonus, which is the shaping regime the literature endorses.
    """
    king_sq = board.king(colour)
    if king_sq is None:
        return 0
    kf = chess.square_file(king_sq)
    kr = chess.square_rank(king_sq)
    # Rank "ahead" of the king, toward the enemy: up for White, down for Black.
    ahead = (kr + 1) if colour == chess.WHITE else (kr - 1)
    r = material_ratio(board)
    phase_scale = 0.5 + 0.5 * r  # 1.0 opening, 0.5 bare endgame
    score = 0
    if 0 <= ahead <= 7:
        for df in (-1, 0, 1):
            f = kf + df
            if 0 <= f <= 7:
                sq = chess.square(f, ahead)
                if board.piece_at(sq) == chess.Piece(chess.PAWN, colour):
                    score += 6
                else:
                    score -= 4
    # Open-file exposure: fewer than two friendly pawns on the king's file.
    file_pawns = chess.popcount(
        board.pieces_mask(chess.PAWN, colour) & chess.BB_FILES[kf]
    )
    if file_pawns <= 1:
        score -= 6
    # Castled kings sit on the wing files, away from the exposed centre.
    score += -3 if kf in (3, 4) else 3
    return int(score * phase_scale)


def piece_square_score(board: chess.Board, colour: int) -> int:
    """Total piece-square bonus for ``colour``, with a phase-blended king table.

    A king in the middlegame wants shelter; in the endgame it wants the centre.
    Rather than pick one table, we lerp between them on the material ratio,
    which makes the eval continuous and avoids a discontinuity at the phase
    boundary that would show up as evaluation noise during search.

    The tables are written from White's point of view, so Black's lookup
    mirrors the rank. Every piece type must do this -- including the king,
    which is the easy one to forget because the king's table is handled in a
    separate branch. Forgetting it was a real bug: the start position evaluated
    to +50 instead of 0, silently biasing every search towards White.
    """
    r = material_ratio(board)
    score = 0
    for sq in chess.scan_forward(board.occupied_co[colour]):
        piece_type = board.piece_type_at(sq)
        rank = sq >> 3
        file = sq & 7
        # Mirror ranks for Black so the table is symmetric.
        rr = rank if colour == chess.WHITE else 7 - rank
        if piece_type == chess.KING:
            mid = int(KING_PST_MID[rr, file])
            end = int(KING_PST_END[rr, file])
            score += int(r * mid + (1.0 - r) * end)
        else:
            score += int(PST[piece_type][rr, file])
    return score


# --------------------------------------------------------------------------
# the evaluation function
# --------------------------------------------------------------------------

# Endgame detection for mate-score scaling: below this ratio a king+pawn vs
# king position is a genuine draw, so we are more willing to call it level.
MATE_SCORE = 100_000


def evaluate(board: chess.Board) -> int:
    """Static evaluation in centipawns, from the side-to-move's point of view.

    Sign convention: **positive is good for the side to move.** This is the
    convention alpha-beta wants (negamax), and sticking to it everywhere avoids
    the sign-flipping bugs that plague hand-rolled search.

    Terminal positions return ``+/-MATE_SCORE`` adjusted by ply so that a mate
    found sooner is preferred over one found later, which stops the search from
    dithering once it sees a forced win.
    """
    if board.is_checkmate():
        return -MATE_SCORE + board.ply()
    if (
        board.is_stalemate()
        or board.is_insufficient_material()
        or board.is_seventyfive_moves()
        or board.is_fivefold_repetition()
    ):
        return 0

    white = (
        material_value(board, chess.WHITE)
        + piece_square_score(board, chess.WHITE)
        + king_shelter_score(board, chess.WHITE)
    )
    black = (
        material_value(board, chess.BLACK)
        + piece_square_score(board, chess.BLACK)
        + king_shelter_score(board, chess.BLACK)
    )
    score = white - black  # White's point of view
    return score if board.turn == chess.WHITE else -score


def evaluate_for(board: chess.Board, colour: int) -> int:
    """Evaluation from a fixed colour's point of view."""
    raw = evaluate(board)
    return raw if board.turn == colour else -raw


def outcome_score(result: str, colour: int) -> float:
    """Map a game result string to a target in ``{-1, 0, +1}`` for ``colour``.

    ``result`` is python-chess's ``"1-0"`` / ``"0-1"`` / ``"1/2-1/2"``. Used as
    the terminal reward by every learner, so the mapping lives in one place.
    """
    if result == "1-0":
        return 1.0 if colour == chess.WHITE else -1.0
    if result == "0-1":
        return 1.0 if colour == chess.BLACK else -1.0
    return 0.0


# --------------------------------------------------------------------------
# tactical "value of a square" terms used by the encoding
# --------------------------------------------------------------------------

def capture_value(board: chess.Board, move: chess.Move) -> int:
    """Centipawn value of what ``move`` captures, 0 for a quiet move.

    En passant is handled explicitly because python-chess returns None for
    ``piece_at`` on the destination in that case -- the captured pawn sits
    elsewhere, and missing it would silently make the encoder blind to one of
    the sharper tactical motifs.
    """
    if board.is_en_passant(move):
        return PIECE_VALUE[chess.PAWN]
    victim = board.piece_at(move.to_square)
    return PIECE_VALUE[victim.piece_type] if victim else 0


def recapture_risk(board: chess.Board, move: chess.Move) -> int:
    """Cheapest enemy attacker's value against the destination, or 0 if safe.

    Approximates "if I move here, what can hit me next ply". This is the
    "in capturable position upon moving there" term in the spec, and it is the
    cheapest available signal that separates a good move from a blunder.
    """
    probe = board.copy(stack=False)
    probe.push(move)
    opp = probe.turn
    attackers = probe.attackers(opp, move.to_square)
    if not attackers:
        return 0
    cheapest = min(
        PIECE_VALUE[probe.piece_type_at(sq)] for sq in attackers
    )
    return cheapest


def hanging_value(board: chess.Board, colour: int) -> int:
    """Total value of ``colour``'s pieces currently under attack and undefended.

    The "danger value of own pieces" term. Counts a piece as hanging only if no
    friendly piece defends the square, which is a strictly better predictor of
    material loss than "attacked" alone -- a defended knight under attack is
    not actually in danger.
    """
    from .bitboard import defended_map, attack_map

    enemy = not colour
    threats = attack_map(board, enemy)
    guard = defended_map(board, colour)
    own = board.occupied_co[colour]

    total = 0
    for sq in chess.scan_forward(own):
        rank, file = sq >> 3, sq & 7
        if threats[rank, file] and not guard[rank, file]:
            total += PIECE_VALUE[board.piece_type_at(sq)]
    return total
