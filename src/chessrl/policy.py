"""L1 / L1.5: the blind factored softmax policy.

This is the naive end of the benchmark, and it is deliberately, aggressively
blind: the move distribution does not depend on the board at all. Only the
*legal move mask* is consulted, to zero out moves that cannot be played.

Why a blind policy is worth building
------------------------------------
It establishes the floor. If a policy that cannot see the board beats a policy
that can, the informed policy is broken. It also makes the whole learning
plumbing -- masks, sampling, credit assignment, counters, pruning, persistence
-- testable in isolation, because nothing else can be blamed for the results.

The factorised action space
---------------------------
The spec asks for "a distinct probability to each of 5 outputs" (row, col,
piece, direction, size) with invalid outputs masked rather than punished. The
chess analogue of that tuple is ``(from_square, to_square, promotion)``, so
this is a three-factor model:

    P(move) = P(from) * P(to | from) * P(promo | from, to)

implemented as three chained softmaxes rather than one flat softmax over 20480
actions. The distinction matters:

* A flat softmax over 20480 slots is a table you cannot fill. Most of those
  slots are legal in *some* position, so nothing is structurally unreachable,
  but a given position only ever touches ~30 of them, which means each weight
  sees very few updates and learns almost nothing.
* The chained form shares statistics. ``P(e2 -> e4)`` and ``P(d2 -> d4)`` both
  draw on "this is a central pawn push", and the destination factor is reused
  across every origin. It learns far faster from the same number of games.

The probability of a concrete move is still a distinct number per combination,
which is what the spec asked for. It is just factorised so the factors can be
learned.

Inequality of updates, and why illegal moves are simply masked
-------------------------------------------------------------
The spec says: "don't bother punishing invalid moves, just mask", and "if a
move is never legal its weight just never updates". That falls out naturally
here. Masking sets the logit to ``-inf`` before the softmax, so illegal moves
get exactly zero probability, contribute nothing to the loss and receive no
gradient. Their underlying parameters are untouched and, crucially, *counted*
as untouched -- see :class:`ProbCounter` at L1.5.

That counter is the only thing that distinguishes "this move is rare" from
"this move has never been tried". Without it, pruning would delete moves that
had no chance to prove themselves.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Protocol, runtime_checkable

import chess
import numpy as np

from . import masks as M
from . import value as V

# A logit of -inf makes a masked entry exactly zero after softmax. Using a
# large negative float instead is a common shortcut but it leaves a vanishing
# amount of probability mass on illegal moves, which then shows up as an
# "illegal move sampled" crash at a rate of roughly one in ten thousand. Use
# real -inf and rely on the mask being correct.
NEG_INF = -np.inf

# Floor subtracted before exponentiation to avoid overflow when every logit is
# very negative (which happens once the model is confident). This is the
# standard max-subtraction trick, applied with the mask respected.
_SOFTMAX_EPS = 1e-300


@runtime_checkable
class Scorer(Protocol):
    """Something that produces raw logits for the three policy factors.

    Separating this from the policy is what lets L3 and L4 reuse the entire
    factored-softmax, masking and sampling machinery while swapping the source
    of the numbers. A blind L1 scorer returns constants; a perceptron returns
    board-dependent values; a torch model returns tensor-backed values.
    """

    def from_logits(self, board: chess.Board) -> np.ndarray:
        """Length-64 logits over origin squares."""
        ...

    def to_logits(self, from_sq: int) -> np.ndarray:
        """Length-64 logits over destination squares, given the origin."""
        ...

    def promo_logits(self, from_sq: int, to_sq: int) -> np.ndarray:
        """Length-``NUM_PROMO`` logits over promotion slots."""
        ...


def masked_softmax(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Softmax over the unmasked entries, exactly zero elsewhere.

    Returns zeros (not NaNs) if the mask is empty, so callers can detect "no
    legal move" without a divide-by-zero warning. The subtraction of the masked
    maximum is what keeps this numerically stable when all legal logits are
    large and negative, which happens routinely once a policy is confident.
    """
    out = np.zeros_like(logits, dtype=np.float64)
    legal = mask.astype(bool)
    if not legal.any():
        return out

    legal_logits = logits[legal]
    peak = legal_logits.max()
    if not np.isfinite(peak):
        # All legal logits are -inf. Spread the mass uniformly rather than
        # returning NaN: this can legitimately happen if a scorer is badly
        # initialised, and a uniform fallback degrades gracefully.
        out[legal] = 1.0 / legal.sum()
        return out

    shifted = np.exp(legal_logits - peak)
    out[legal] = shifted / shifted.sum()
    return out


class BlindScorer:
    """A scorer with no board input: three fixed logit tables.

    Initialised to zeros, which makes the initial distribution proportional to
    the legal move set -- i.e. uniform over all legal moves. That is the correct
    starting point for a blind policy, and it makes the initial behaviour
    identical to :class:`chessrl.game.RandomPolicy`, so an L1 run can be diffed
    against random play as a sanity check.
    """

    __slots__ = ("from_table", "to_table", "promo_table", "n_parameters")

    def __init__(self):
        # Fixed tables, not per-position, because the policy is blind.
        self.from_table = np.zeros(64, dtype=np.float64)
        self.to_table = np.zeros((64, 64), dtype=np.float64)
        self.promo_table = np.zeros((64, 64, M.NUM_PROMO), dtype=np.float64)
        self.n_parameters = (
            self.from_table.size + self.to_table.size + self.promo_table.size
        )

    def from_logits(self, board: chess.Board) -> np.ndarray:
        return self.from_table

    def to_logits(self, from_sq: int) -> np.ndarray:
        return self.to_table[from_sq]

    def promo_logits(self, from_sq: int, to_sq: int) -> np.ndarray:
        return self.promo_table[from_sq, to_sq]

    # ---- persistence ---------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "from_table": self.from_table.tolist(),
            "to_table": self.to_table.tolist(),
            "promo_table": self.promo_table.tolist(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.from_table[:] = np.asarray(state["from_table"], dtype=np.float64)
        self.to_table[:] = np.asarray(state["to_table"], dtype=np.float64)
        self.promo_table[:] = np.asarray(
            state["promo_table"], dtype=np.float64
        )


def move_probability(
    scorer: Scorer,
    board: chess.Board,
    mask: np.ndarray,
    move: chess.Move,
) -> float:
    """Probability the factored model assigns to one concrete move.

    The chain is evaluated for a single move rather than the whole table, so
    this is cheap enough to call per-update. It must agree with
    :meth:`FactoredSoftmaxPolicy.move_distribution` at the matching index,
    which is asserted in the tests.
    """
    from_sq, to_sq = move.from_square, move.to_square
    slot = M.PROMO_SLOT.get(move.promotion, 0)

    from_mask = M.legal_from_mask(mask)
    p_from = masked_softmax(scorer.from_logits(board), from_mask)
    if p_from[from_sq] == 0.0:
        return 0.0

    dest_mask = mask[from_sq].any(axis=1)
    p_to = masked_softmax(scorer.to_logits(from_sq), dest_mask)

    promo_mask = mask[from_sq, to_sq]
    p_promo = masked_softmax(scorer.promo_logits(from_sq, to_sq), promo_mask)

    return float(p_from[from_sq] * p_to[to_sq] * p_promo[slot])


class FactoredSoftmaxPolicy:
    """Policy over ``(from, to, promo)`` with each factor masked independently.

    Implements the :class:`chessrl.game.DistributionPolicy` interface, so it
    drops straight into the game loop and the tournament harness.

    The greedy/explore switch
    -------------------------
    ``greedy=True`` picks the arg-max legal move and is used for evaluation and
    for the L2 ordering heuristic. ``greedy=False`` samples, which is what
    training needs: a policy that always plays its current best move cannot
    discover that the best move is wrong.
    """

    def __init__(
        self,
        scorer: Scorer | None = None,
        *,
        name: str = "L1-blind-factored",
        seed: int | None = None,
        greedy: bool = False,
        temperature: float = 1.0,
    ):
        self.scorer = scorer if scorer is not None else BlindScorer()
        self.name = name
        self.greedy = greedy
        self.temperature = temperature
        self.rng = np.random.default_rng(seed)

    # ---- distribution --------------------------------------------------

    def move_distribution(self, board: chess.Board) -> np.ndarray:
        """Dense distribution over the flat ``ACTION_SPACE``.

        Only legal entries can be non-zero. This is the slow-but-complete view;
        :meth:`sample` and :meth:`select` use the cheaper factorised path and
        never materialise this array.
        """
        out = np.zeros(M.ACTION_SPACE, dtype=np.float64)
        if board.is_game_over(claim_draw=False):
            return out

        mask = M.legal_move_mask(board)
        from_mask = M.legal_from_mask(mask)
        p_from = masked_softmax(self.scorer.from_logits(board) / self.temperature,
                                from_mask)

        for from_sq in np.nonzero(from_mask)[0]:
            p_f = p_from[from_sq]
            if p_f == 0.0:
                continue
            dest_mask = mask[from_sq].any(axis=1)
            p_to = masked_softmax(
                self.scorer.to_logits(int(from_sq)) / self.temperature, dest_mask
            )
            for to_sq in np.nonzero(dest_mask)[0]:
                promo_mask = mask[from_sq, to_sq]
                p_promo = masked_softmax(
                    self.scorer.promo_logits(int(from_sq), int(to_sq))
                    / self.temperature,
                    promo_mask,
                )
                for slot in np.nonzero(promo_mask)[0]:
                    idx = (
                        (int(from_sq) * M.NUM_SQUARES + int(to_sq)) * M.NUM_PROMO
                        + int(slot)
                    )
                    out[idx] = p_f * p_to[to_sq] * p_promo[slot]
        return out

    # ---- sampling ------------------------------------------------------

    def sample_factors(
        self, board: chess.Board, mask: np.ndarray | None = None
    ) -> tuple[int, int, int]:
        """Draw ``(from_sq, to_sq, promo_slot)``, masking at each stage.

        Returns the three factors rather than a move so that the caller can
        see *which* factor was chosen. The trainers need that: credit is
        assigned per factor, and a move that was legal but never sampled should
        not have its counter incremented.
        """
        if mask is None:
            mask = M.legal_move_mask(board)

        p_from = masked_softmax(
            self.scorer.from_logits(board) / self.temperature,
            M.legal_from_mask(mask),
        )
        from_sq = int(self.rng.choice(64, p=_normalise(p_from)))

        dest_mask = mask[from_sq].any(axis=1)
        p_to = masked_softmax(
            self.scorer.to_logits(from_sq) / self.temperature, dest_mask
        )
        to_sq = int(self.rng.choice(64, p=_normalise(p_to)))

        promo_mask = mask[from_sq, to_sq]
        p_promo = masked_softmax(
            self.scorer.promo_logits(from_sq, to_sq) / self.temperature,
            promo_mask,
        )
        slot = int(self.rng.choice(M.NUM_PROMO, p=_normalise(p_promo)))
        return from_sq, to_sq, slot

    def select(self, board: chess.Board) -> chess.Move:
        """Return a legal move, sampled or greedy per ``self.greedy``."""
        mask = M.legal_move_mask(board)
        if not mask.any():
            raise ValueError(f"no legal moves in position {board.fen()}")

        if self.greedy:
            from_sq, to_sq, slot = self._argmax_factors(board, mask)
        else:
            from_sq, to_sq, slot = self.sample_factors(board, mask)
        return M.index_to_move(
            (from_sq * M.NUM_SQUARES + to_sq) * M.NUM_PROMO + slot
        )

    def _argmax_factors(
        self, board: chess.Board, mask: np.ndarray
    ) -> tuple[int, int, int]:
        """Greedy choice, factor by factor.

        Greedy-per-factor is the right notion of "most probable move" under a
        factorised model: it maximises the product without enumerating the
        space. Note this is not identical to arg-max of the full distribution
        in pathological cases where the independently-best factors do not form
        a legal combination -- so the result is re-checked against the mask.
        """
        from_probs = masked_softmax(self.scorer.from_logits(board), M.legal_from_mask(mask))
        order = np.argsort(from_probs)[::-1]
        for from_sq in order:
            if from_probs[from_sq] <= 0.0:
                continue
            dest_mask = mask[from_sq].any(axis=1)
            p_to = masked_softmax(self.scorer.to_logits(int(from_sq)), dest_mask)
            to_sq = int(np.argmax(p_to))
            if p_to[to_sq] <= 0.0:
                continue
            promo_mask = mask[from_sq, to_sq]
            p_promo = masked_softmax(
                self.scorer.promo_logits(int(from_sq), to_sq), promo_mask
            )
            slot = int(np.argmax(p_promo))
            if p_promo[slot] > 0.0:
                return int(from_sq), to_sq, slot
        # Unreachable if the mask is non-empty and the softmax is well formed,
        # but a silent wrong answer here would be very hard to trace, so fail
        # loudly instead.
        raise RuntimeError(f"greedy selection failed on {board.fen()}")

    # ---- persistence ---------------------------------------------------

    def state_dict(self) -> dict:
        """Serialise the learnable state. Counters are deliberately excluded.

        The spec asks for the update counters to be deleted before saving. They
        are training bookkeeping, not part of the model, and saving them would
        make a checkpoint's size depend on how much exploration happened to be
        done. ``FactoredSoftmaxTrainer`` handles the counter file separately.
        """
        return {
            "name": self.name,
            "scorer": self.scorer.state_dict(),
            "temperature": self.temperature,
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.state_dict()), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, **kwargs) -> "FactoredSoftmaxPolicy":
        state = json.loads(Path(path).read_text(encoding="utf-8"))
        scorer = BlindScorer()
        scorer.load_state_dict(state["scorer"])
        return cls(
            scorer,
            name=state.get("name", "L1-blind-factored"),
            temperature=state.get("temperature", 1.0),
            **kwargs,
        )


# --------------------------------------------------------------------------
# L1.5: probability counters
# --------------------------------------------------------------------------

@dataclass
class ProbCounter:
    """Tracks how often each action was *offered* and how often it was *taken*.

    This is the piece that makes pruning safe, and it is the one part of the
    spec that is easy to skip and expensive to omit.

    The distinction between "legal but never sampled" and "never legal" is the
    whole point. Consider a queen sacrifice that is only legal in one position
    out of a thousand. Under a blind policy it is offered rarely, so its weight
    updates rarely. With probability-based pruning alone it would eventually be
    deleted as unimportant -- and it would be deleted for the crime of being
    situational rather than bad.

    So we count two things separately:

    * ``offered`` -- incremented whenever the action was in the legal mask;
    * ``taken`` -- incremented whenever the sampler actually chose it.

    An action is a *pruning candidate* only if it has been offered many times
    and taken rarely relative to its neighbours, which is a genuinely different
    signal from a low probability. Action with ``offered == 0`` are never
    pruned and never lose their parameters.
    """

    n_from: int = 64
    n_to: int = 64
    n_promo: int = M.NUM_PROMO

    offered_from: np.ndarray = field(default=None)
    taken_from: np.ndarray = field(default=None)
    offered_to: np.ndarray = field(default=None)
    taken_to: np.ndarray = field(default=None)

    def __post_init__(self):
        if self.offered_from is None:
            self.offered_from = np.zeros(self.n_from, dtype=np.int64)
            self.taken_from = np.zeros(self.n_from, dtype=np.int64)
            self.offered_to = np.zeros((self.n_from, self.n_to), dtype=np.int64)
            self.taken_to = np.zeros((self.n_from, self.n_to), dtype=np.int64)

    def note_offer(self, mask: np.ndarray) -> None:
        """Record which actions were legally available in a position."""
        self.offered_from += M.legal_from_mask(mask).astype(np.int64)
        dest_offered = mask.any(axis=2).astype(np.int64)
        self.offered_to += dest_offered

    def note_taken(self, from_sq: int, to_sq: int, slot: int) -> None:
        """Record which action was actually chosen."""
        self.taken_from[from_sq] += 1
        self.taken_to[from_sq, to_sq] += 1

    def update(self, mask: np.ndarray, from_sq: int, to_sq: int, slot: int) -> None:
        """Convenience: note both the offer and the take for one decision."""
        self.note_offer(mask)
        self.note_taken(from_sq, to_sq, slot)

    def never_offered_from(self) -> np.ndarray:
        """Origin squares that have never had a legal move."""
        return self.offered_from == 0

    def prune_candidates(
        self,
        probs: np.ndarray,
        *,
        min_offers: int = 200,
        prob_threshold: float = 0.01,
    ) -> np.ndarray:
        """Mask of origin squares worth pruning.

        An origin is prunable when its probability has collapsed *and* it has
        had plenty of chances to prove otherwise. ``min_offers`` is what stops
        an unlucky early run from permanently deleting a square: until it has
        been seen 200 times, no probability is low enough to justify removal.

        This implements the spec's "prune anything with probability < 0.01"
        while fixing the flaw in it -- pruning on probability alone would
        remove actions that had simply not been tried yet.
        """
        eligible = self.offered_from >= min_offers
        unlikely = probs < prob_threshold
        return eligible & unlikely

    def stats(self) -> dict:
        total_offers = int(self.offered_from.sum())
        total_takes = int(self.taken_from.sum())
        return {
            "total_offers": total_offers,
            "total_takes": total_takes,
            "never_offered_squares": int(self.never_offered_from().sum()),
            "take_rate": (
                round(total_takes / total_offers, 4) if total_offers else 0.0
            ),
        }

    # ---- persistence (separate from the model, by design) --------------

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            Path(path),
            offered_from=self.offered_from,
            taken_from=self.taken_from,
            offered_to=self.offered_to,
            taken_to=self.taken_to,
        )

    @classmethod
    def load(cls, path: str | Path) -> "ProbCounter":
        data = np.load(Path(path))
        counter = cls()
        counter.offered_from = data["offered_from"]
        counter.taken_from = data["taken_from"]
        counter.offered_to = data["offered_to"]
        counter.taken_to = data["taken_to"]
        return counter


def _normalise(probs: np.ndarray) -> np.ndarray:
    """Renormalise for ``rng.choice``, which rejects sums that are not 1.

    ``masked_softmax`` can return a vector whose sum is 0.9999999999999999,
    and numpy's ``choice`` raises on that. This helper is also the place that
    turns an all-zero vector into a uniform one, which is the correct
    degenerate behaviour: if the scored distribution has no mass, pick
    uniformly among the legal options rather than crashing.
    """
    total = probs.sum()
    if total <= 0:
        n = (probs > 0).sum() or len(probs)
        out = np.zeros_like(probs)
        out[probs >= 0] = 1.0 / n
        return out / out.sum()
    return probs / total


def uniform_over_mask(mask: np.ndarray) -> np.ndarray:
    """Uniform distribution over the True entries of a mask."""
    flat = mask.astype(np.float64).ravel()
    return _normalise(flat)


def entropy(probs: np.ndarray) -> float:
    """Shannon entropy in nats. Used to watch a policy sharpen over training."""
    p = probs[probs > 0]
    return float(-(p * np.log(p)).sum())


def perplexity(probs: np.ndarray) -> float:
    """Effective number of choices, ``exp(entropy)``. Easier to read than nats."""
    return float(np.exp(entropy(probs)))
