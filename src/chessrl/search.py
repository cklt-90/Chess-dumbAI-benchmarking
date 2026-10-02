"""L2: alpha-beta minimax, and the non-learning wrapper that feeds it L1.

Two things live here, and the difference between them is the point of the
level.

**Alpha-beta minimax.** A classical search over :func:`chessrl.value.evaluate`,
with move ordering, a transposition table, iterative deepening and quiescence.
It is the first level that plays recognisable chess, and it is deliberately
hand-written rather than learned so it can act as a *fixed* reference against
which the learners are measured.

**The non-learning wrapper.** The spec asks for "half-informing L1's
probabilities via minimax of board_state in 5 moves" and is explicit that this
level "must have no trainable parameters". So the wrapper takes L1's blind
distribution and uses it purely as a *move-ordering heuristic* for the search.

A crucial corollary: because the wrapper has no parameters, it cannot improve
with training. Running it for a million games produces exactly the same policy
as running it for zero. It is in the benchmark to establish how much of L2's
strength comes from *search* alone, with no learning anywhere in the system.
Any learner that cannot beat it has not earned the right to call itself
informed.

What ordering can and cannot change
-----------------------------------
This is worth being precise about, because the obvious statement is false.
Ordering cannot change the **value** the search returns: alpha-beta with a
correct window computes the same minimax value whatever order it examines
children in. But ordering *does* decide which move is returned when several
moves share that value, because the first move to reach the alpha bound is the
one kept.

That is not a defect, it is the entire mechanism by which a prior contributes.
Measured example: in one position Nf6 and Nc6 both evaluate to exactly 90 at
depth 3; the plain engine returns Nf6 by board order, while the wrapper returns
Nc6 because L1's blind table happens to weight the b8 origin slightly higher.
The prior resolved a genuine tie, on no chess grounds whatsoever -- which is
exactly what a blind prior *should* be capable of, and no more.

So the honest claim is: the wrapper cannot change the search's verdict, and it
cannot change a move that is uniquely best. It only breaks ties, and it holds
no parameters of its own.

The 5-move link, and why it is not a search
-------------------------------------------
The spec says the wrapper should "use the most probable move from L1 to inform
min/max heuristics after 5 moves". Read literally that is a shallow search
guided by a blind prior. It is implemented here as: at each node the moves are
sorted by L1's probability before being searched, so the alpha-beta cutoffs
happen on L1's favourite moves first -- which costs nothing at runtime beyond a
sort and is the only channel through which a prior can act without changing the
minimax value.
"""

from __future__ import annotations

import chess
import chess.polyglot
import numpy as np

from . import masks as M
from . import value as V
from .policy import FactoredSoftmaxPolicy

# --------------------------------------------------------------------------
# constants
# --------------------------------------------------------------------------

# Mate scores are kept well below the material scale so that no amount of
# material can ever outrank a forced mate. value.MATE_SCORE is 100_000; the
# search uses the same value so the two agree when the search falls back to a
# static evaluation of a terminal node.
INF = 10**9

# Quiescence depth cap. Unbounded quiescence on a position with many captures
# explodes; 4 plies of forcing moves is enough to resolve the horizon effect
# that motivates it without becoming a second full search.
QUIESCE_DEPTH = 4

# A transposition entry is only safe to use when the stored depth is at least
# what is being asked for. Storing the depth is what makes the table correct
# rather than merely fast.
TT_EXACT, TT_LOWER, TT_UPPER = 0, 1, 2


# --------------------------------------------------------------------------
# move ordering
# --------------------------------------------------------------------------

def score_move(board: chess.Board, move: chess.Move) -> int:
    """Static ordering score for a move: MVV-LVA plus promotions.

    Most-valuable-victim / least-valuable-attacker is the workhorse heuristic
    and costs almost nothing: search a queen capture first and the cutoff
    happens immediately, search it last and the whole tree is wasted.

    This is *not* part of the evaluation. It only decides the order in which
    children are examined, which alpha-beta is allowed to do freely -- the
    final value and the chosen move are unchanged if the ordering is good or
    bad, only the time taken differs.
    """
    victim = board.piece_at(move.to_square)
    score = 0
    if victim is not None:
        attacker = board.piece_type_at(move.from_square)
        # Multiply by 16 so the value dominates and the attacker's cheapness
        # breaks ties: PxQ beats QxP.
        score += 16 * V.PIECE_VALUE[victim.piece_type]
        score -= V.PIECE_VALUE[attacker]
    elif board.is_en_passant(move):
        score += 16 * V.PIECE_VALUE[chess.PAWN]
        score -= V.PIECE_VALUE[chess.PAWN]

    if move.promotion is not None:
        # Promotion is nearly always worth examining early.
        score += 16 * V.PIECE_VALUE[move.promotion]

    return score


def order_moves(
    board: chess.Board,
    moves: list[chess.Move],
    *,
    tt_move: chess.Move | None = None,
    prior: dict[chess.Move, float] | None = None,
    prior_weight: float = 0.0,
) -> list[chess.Move]:
    """Sort moves best-first, optionally blending in a policy prior.

    Three signals, in decreasing priority:

    1. The transposition table's remembered best move. If this position has
       been searched before, that move is the single best guess available and
       is tried first.
    2. ``prior`` -- L1's probability for each move, scaled by ``prior_weight``.
       This is the wrapper's entire contribution. It is additive to the static
       MVV-LVA score, so a strongly-preferred quiet move can be examined before
       a mildly-attractive capture, but a queen capture is still a queen
       capture.
    3. Static MVV-LVA.

    A stable sort is used so moves that tie keep their board order, which makes
    the whole search deterministic -- required for reproducibility and made
    explicit because Python's sort is stable but relying on that implicitly is
    the kind of thing that breaks silently if it ever changes.
    """
    def key(move: chess.Move) -> float:
        if tt_move is not None and move == tt_move:
            return float(INF)
        s = float(score_move(board, move))
        if prior is not None and prior_weight:
            s += prior_weight * prior.get(move, 0.0)
        return s

    return sorted(moves, key=key, reverse=True)


# --------------------------------------------------------------------------
# the search
# --------------------------------------------------------------------------

class MinimaxEngine:
    """Alpha-beta negamax over the static evaluation.

    Written as a class rather than a bare function because the transposition
    table and the node counter are genuine state that must persist across the
    iterative-deepening passes within one move.

    ``nodes`` is exposed because it is the number that makes search
    improvements visible: a change that halves the node count at equal depth is
    a real win, and without the counter it is invisible.
    """

    def __init__(
        self,
        *,
        depth: int = 5,
        use_quiescence: bool = True,
        use_tt: bool = True,
        use_iterative_deepening: bool = True,
        prior_weight: float = 0.0,
        engine_rng=None,
    ):
        self.depth = depth
        self.use_quiescence = use_quiescence
        self.use_tt = use_tt
        self.use_iterative_deepening = use_iterative_deepening
        # Weight applied to a move prior during ordering. Zero means "ignore
        # the prior entirely", which is how the plain L2 engine runs.
        self.prior_weight = prior_weight

        self.tt: dict[int, tuple[int, int, int, chess.Move | None]] = {}
        self.nodes = 0
        self.cutoffs = 0
        self.tt_hits = 0
        self.name = "L2-minimax"

    # ---- public entry point --------------------------------------------

    def select(self, board: chess.Board, prior=None) -> chess.Move:
        """Search ``board`` and return the best move found.

        ``prior`` is an optional ``move -> probability`` mapping used only for
        ordering. Passing it cannot change the search's verdict at full depth,
        which is what makes the L1 wrapper non-learning -- see the module
        docstring.
        """
        moves = list(board.legal_moves)
        if not moves:
            raise ValueError(f"no legal moves in {board.fen()}")
        if len(moves) == 1:
            return moves[0]

        self.nodes = 0
        self.cutoffs = 0
        self.tt_hits = 0

        # Iterative deepening: search depth 1, then 2, ... up to self.depth.
        # The value is that each pass fills the transposition table and gives
        # the next pass a best-move guess, which makes the deeper search much
        # cheaper -- and if the clock ran out, the shallow result is already in
        # hand rather than half a deep search.
        start = 1 if self.use_iterative_deepening else self.depth
        best = moves[0]
        for d in range(start, self.depth + 1):
            best = self._root(board, moves, d, prior)
        return best

    # ---- root -----------------------------------------------------------

    def _prior_map(
        self, board: chess.Board, moves: list[chess.Move], prior
    ) -> dict[chess.Move, float] | None:
        """Normalise the caller's ``prior`` into a ``move -> score`` mapping.

        ``prior`` may arrive as a dict keyed by move (what
        :class:`InformedMinimaxPolicy` builds) or as ``None``. Returning
        ``None`` rather than an empty dict matters: :func:`order_moves` uses
        ``None`` to mean "skip the prior entirely", which is also what
        ``prior_weight == 0`` would achieve, but keeping the two conditions
        separate makes it obvious that a plain L2 search never even looks at a
        prior rather than merely not being affected by one.
        """
        if prior is None or not self.prior_weight:
            return None
        return prior

    def _root(
        self, board: chess.Board, moves: list[chess.Move], depth: int, prior
    ) -> chess.Move:
        """One full-width pass at the root, returning the best move.

        The root is handled separately from interior nodes because there is no
        alpha-beta window to narrow here -- we need the true value of every
        root move, not just a bound, so the window starts fully open and only
        alpha moves.
        """
        move, _ = self._root_with_value(board, moves, depth, prior)
        return move

    def _root_with_value(
        self, board: chess.Board, moves: list[chess.Move], depth: int, prior
    ) -> tuple[chess.Move, int]:
        """As :meth:`_root`, but also returns the value it assigned.

        Split out because the *value* is what several invariants are about, and
        the only other way to get it is to re-derive the root loop in the test.
        In L2 the two levels of the ladder that share a search (the plain
        engine and the informed wrapper) must agree on the value while being
        allowed to disagree on the move, so the value has to be observable.

        The value returned is in the side-to-move frame, like every other
        value this search produces.
        """
        tt_entry = self.tt.get(chess.polyglot.zobrist_hash(board))
        tt_move = tt_entry[3] if tt_entry else None

        prior_scores = self._prior_map(board, moves, prior)
        ordered = order_moves(
            board, moves, tt_move=tt_move, prior=prior_scores,
            prior_weight=self.prior_weight,
        )

        alpha = -INF
        best_move = ordered[0]
        best_value = -INF

        for move in ordered:
            board.push(move)
            value = -self._negamax(board, depth - 1, -INF, -alpha)
            board.pop()

            if value > best_value:
                best_value = value
                best_move = move
            if value > alpha:
                alpha = value

        self.tt[chess.polyglot.zobrist_hash(board)] = (
            depth, best_value, TT_EXACT, best_move
        )
        return best_move, best_value

    def _search_value(self, board: chess.Board, depth: int) -> int:
        """The value this engine assigns to ``board`` at ``depth``.

        Deliberately does *not* clear the transposition table or the caches: it
        exists to be called twice in a row on two engines that differ only in
        one setting, and a passing comparison should be about the algorithm
        rather than about which engine happened to warm its table first.
        """
        moves = list(board.legal_moves)
        if not moves:
            return 0
        _, value = self._root_with_value(board, moves, depth, prior=None)
        return value

    # ---- interior -------------------------------------------------------

    def _negamax(
        self, board: chess.Board, depth: int, alpha: int, beta: int
    ) -> int:
        """Negamax with alpha-beta pruning.

        The sign convention is the standard one: the returned value is always
        from the point of view of the side to move, and the caller negates.
        This is what lets one function serve both min and max -- the "min" side
        is just the max side after a negation.
        """
        self.nodes += 1

        # --- terminal and horizon handling, in order of cheapness ---------
        if board.is_checkmate():
            # Prefer a mate that is closer. The ply term makes a mate in 1
            # score higher than a mate in 3, so the search does not shuffle
            # once it sees a forced win.
            return -V.MATE_SCORE + board.ply()
        if (
            board.is_stalemate()
            or board.is_insufficient_material()
            or board.is_seventyfive_moves()
            or board.is_fivefold_repetition()
        ):
            return 0

        if depth <= 0:
            if self.use_quiescence:
                return self._quiescence(board, alpha, beta)
            return V.evaluate(board)

        # --- transposition lookup ----------------------------------------
        key = chess.polyglot.zobrist_hash(board)
        tt_move = None
        if self.use_tt:
            entry = self.tt.get(key)
            if entry is not None:
                e_depth, e_value, e_flag, e_move = entry
                tt_move = e_move
                # Only trust an entry that was searched at least as deeply.
                # Using a shallower result as if it were exact is the classic
                # transposition-table bug: it is fast and gives wrong answers.
                if e_depth >= depth:
                    if e_flag == TT_EXACT:
                        self.tt_hits += 1
                        return e_value
                    if e_flag == TT_LOWER and e_value > alpha:
                        alpha = e_value
                    elif e_flag == TT_UPPER and e_value < beta:
                        beta = e_value
                    if alpha >= beta:
                        self.tt_hits += 1
                        return e_value

        legal = list(board.legal_moves)
        if not legal:
            # Not checkmate and not stalemate but no legal moves cannot happen
            # under python-chess's rules; kept as a guard rather than an
            # assertion because a crash mid-game is worse than a wrong number.
            return 0

        ordered = order_moves(board, legal, tt_move=tt_move)
        original_alpha = alpha
        best_value = -INF
        best_move = ordered[0]

        for move in ordered:
            board.push(move)
            value = -self._negamax(board, depth - 1, -beta, -alpha)
            board.pop()

            if value > best_value:
                best_value = value
                best_move = move
            if value > alpha:
                alpha = value
            if alpha >= beta:
                # Beta cutoff: the opponent would never allow this line, so
                # the remaining moves cannot affect the answer.
                self.cutoffs += 1
                break

        if self.use_tt:
            if best_value <= original_alpha:
                flag = TT_UPPER
            elif best_value >= beta:
                flag = TT_LOWER
            else:
                flag = TT_EXACT
            self.tt[key] = (depth, best_value, flag, best_move)

        return best_value

    # ---- quiescence -----------------------------------------------------

    def _quiescence(
        self, board: chess.Board, alpha: int, beta: int, qdepth: int = 0
    ) -> int:
        """Extend the search over captures only, to blunt the horizon effect.

        The problem this solves: at the depth limit the search stops mid-
        exchange and evaluates a position where a piece is about to be
        captured, so it happily walks into losing a queen "because" the
        recapture is beyond its horizon. Searching captures until the position
        is quiet removes most of that.

        Stand-pat first: the side to move is never *forced* to capture, so the
        static evaluation is a lower bound on what it can achieve. If that
        already beats beta, cut. This is the single biggest saving in the
        function and it is also correct, not merely a heuristic.

        ``qdepth`` is carried explicitly. An earlier version tried to derive it
        from ``board.ply()``, which does not work: ``board.ply()`` counts plies
        from the start of the game, not from the start of the extension, so the
        derived value grew without bound and silently disabled the depth cap.
        """
        if board.is_checkmate():
            return -V.MATE_SCORE + board.ply()

        stand_pat = V.evaluate(board)
        if stand_pat >= beta:
            return stand_pat
        if stand_pat > alpha:
            alpha = stand_pat

        if qdepth >= QUIESCE_DEPTH:
            return stand_pat

        captures = [m for m in board.legal_moves if board.is_capture(m)]
        if not captures:
            return stand_pat

        for move in order_moves(board, captures):
            board.push(move)
            value = -self._quiescence(board, -beta, -alpha, qdepth + 1)
            board.pop()
            if value >= beta:
                return value
            if value > alpha:
                alpha = value
        return alpha


# --------------------------------------------------------------------------
# L2 as a policy
# --------------------------------------------------------------------------

class MinimaxPolicy:
    """The plain L2 player: alpha-beta with no external guidance.

    Has no trainable parameters and no randomness, so two instances with the
    same depth play identical games. That determinism is a feature for a
    reference level -- it makes any difference between runs attributable to the
    component under test rather than to the opponent.
    """

    def __init__(
        self,
        depth: int = 4,
        *,
        name: str | None = None,
        use_quiescence: bool = True,
        use_tt: bool = True,
        seed: int | None = None,
    ):
        self.engine = MinimaxEngine(
            depth=depth,
            use_quiescence=use_quiescence,
            use_tt=use_tt,
        )
        self.depth = depth
        self.name = name or f"L2-minimax-d{depth}"
        # Accepted and ignored: the protocol allows a seed and some harnesses
        # pass one uniformly. Silently swallowing it is better than a TypeError
        # from a caller that has no reason to know this policy is deterministic.
        self._seed = seed

    def select(self, board: chess.Board) -> chess.Move:
        return self.engine.select(board)

    @property
    def nodes(self) -> int:
        return self.engine.nodes


class InformedMinimaxPolicy:
    """L2 wrapped around L1: the spec's "non-learning wrapper".

    The blind L1 distribution is used **only to order the search**. It cannot
    change the search's value, cannot be updated, and holds no parameters of
    its own. It *can* decide which move is returned when several moves share
    the best value -- see the module docstring, which is the honest statement
    of the property and the mechanism by which a prior earns its place.

    The design intent is to isolate one question: does a hand-written search
    plus a blind prior beat the same search alone? If yes, the prior is
    carrying real ordering information. If no, the prior is noise, and any
    presumably-stronger learner that fails to beat *this* has a problem with
    its own learning rather than with its features.

    Because it does not learn, running it for more games changes nothing. That
    is asserted in the tests, and it is the property the spec demanded when it
    said this level "must have no trainable parameters".
    """

    def __init__(
        self,
        prior_policy: FactoredSoftmaxPolicy | None = None,
        *,
        depth: int = 4,
        prior_weight: float = 200.0,
        name: str | None = None,
    ):
        self.prior_policy = prior_policy or FactoredSoftmaxPolicy(seed=0)
        self.prior_weight = prior_weight
        self.depth = depth
        self.engine = MinimaxEngine(
            depth=depth,
            use_quiescence=True,
            use_tt=True,
            prior_weight=prior_weight,
        )
        self.name = name or f"L2-informed-d{depth}"

    def _prior(self, board: chess.Board) -> dict[chess.Move, float]:
        """L1's probability for each legal move, scaled for the ordering scale.

        ``prior_weight`` matters more than it looks. The static MVV-LVA score
        is O(1600) for a queen capture, while a probability is O(0.05). Without
        scaling, the prior would be invisible; with the wrong scale it would
        swamp MVV-LVA and start *hurting* the ordering. 200 puts a typical move
        (p=0.05) at ~10 points, enough to break ties among quiet moves without
        displacing a real capture.
        """
        dist = self.prior_policy.move_distribution(board)
        out: dict[chess.Move, float] = {}
        for move in board.legal_moves:
            out[move] = float(dist[M.move_to_index(move)])
        return out

    def select(self, board: chess.Board) -> chess.Move:
        return self.engine.select(board, prior=self._prior(board))

    @property
    def nodes(self) -> int:
        return self.engine.nodes
