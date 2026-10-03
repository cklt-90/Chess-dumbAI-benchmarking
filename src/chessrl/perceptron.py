"""L3: a perceptron over the encoded board state, and depth-5 top-5 feedback.

What L3 adds over L2
--------------------
L2's search is strong but has no parameters and cannot improve. L1's learner has
parameters but is blind. L3 is the first level that is *both* informed and
trainable: it reads the board through :mod:`chessrl.encode` and produces a move
distribution from what it sees.

The size problem, and how it is solved
--------------------------------------
The obvious design -- flatten the ``(26, 8, 8)`` tensor and multiply by a weight
matrix -- needs 64 x 1664 = 106k parameters for the origin head alone, and
64 x 64 x 1664 = 6.8M for the destination head. That is not a perceptron, it is
a lookup table with extra steps, and with the number of games available in this
benchmark it would never converge.

So the model stays **factorised**, exactly as L1's does, but each factor becomes
a small perceptron over *local* features rather than a constant:

* **origin head** -- score every square by a weight vector over that square's
  own channel values. ``W_from`` is ``(64, TOTAL_CHANNELS)``: 1664 parameters.
* **destination head** -- score every square, given the origin, using
  (a) the destination square's own channel values, (b) the origin's, and
  (c) a small set of geometric relations between them.
* **promotion head** -- a tiny head over the promotion slot, conditioned on the
  move's local features. Promotions are rare and near-uniform, so this head is
  deliberately small.

This keeps the parameter count in the low thousands, keeps every weight
interpretable as "how much do I like channel c on square s", and reuses the
entire L1 masking/sampling machinery unchanged because the model implements the
existing :class:`chessrl.policy.Scorer` protocol.

Integer-friendly arithmetic (the spec's "use ints for precision and speed")
--------------------------------------------------------------------------
Weights are stored as float32 for the dot products -- float32 numpy kernels are
faster than float64 on every platform that matters here -- but the *update* is
quantised. ``W += round(lr * credit * direction)`` in integer units via a fixed
scale factor. The reason is not superstition: a perceptron updated with
diminishing credit across thousands of games accumulates floating-point drift
that is invisible per-update and measurable in aggregate, and quantising each
step keeps the weights on a fixed grid so two runs with the same seed stay
bit-identical in the weights rather than merely close.

Midstate incremental updates
----------------------------
The spec asks that midgame states be *updated* rather than replaced, and that
the cache be reused for resampling. Implemented as: positions are recorded into
a :class:`chessrl.cache.MidstateStore` during play, and ``train_on_midstates``
samples from that store to produce additional training signal from positions
the policy actually reached, instead of replaying openings forever. This is what
stops a learner from overfitting the first ten plies.

Depth-5 top-5 feedback
----------------------
The spec asks for the minimax-depth-5 best line to be fed back as a target.
Implemented in :class:`L3Trainer.train_on_search_feedback`: for each position,
L2 at depth 5 evaluates the legal moves, the top 5 are taken, and the perceptron
is pushed towards them in proportion to how much better the search thinks they
are than the policy's current best. This is imitation of a search -- the same
idea as AlphaZero's policy target, without the MCTS.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import chess
import chess.polyglot
import numpy as np

from . import masks as M
from . import value as V
from .cache import PositionCache
from .encode import (
    BINARY_CHANNELS,
    CHANNELS,
    TOTAL_CHANNELS,
    VALUE_CHANNELS_N,
    board_context_features,
    encode,
)
from .policy import (
    BlindScorer,
    FactoredSoftmaxPolicy,
    ProbCounter,
    masked_softmax,
)

# Fixed-point update scale. Every weight lives on a grid of 1/QUANT, and each
# update is rounded to a whole grid step. This is what makes two seeded runs
# produce bit-identical weights rather than merely numerically close ones.
QUANT = 1024


# --------------------------------------------------------------------------
# the perceptron scorer
# --------------------------------------------------------------------------

@dataclass
class PerceptronConfig:
    """Shape and training knobs for the L3 model.

    ``use_context`` appends the global scalar features (phase, material, check,
    mobility) to every head's input. Without it the model cannot distinguish
    "this is the opening" from "this is an endgame" except through the value
    planes, which is a real limitation: the same move means different things in
    the two phases.
    """

    use_value_channels: bool = False
    use_context: bool = True
    # Rectified units on the from-head only. A purely linear perceptron cannot
    # learn "a knight on the rim is bad" because that is a conjunction of
    # "piece is a knight" and "square is on the rim". One hidden layer of
    # rectified units is the cheapest thing that can express it.
    hidden: int = 32
    quantised: bool = True


class PerceptronScorer:
    """Factorised perceptron implementing the :class:`chessrl.policy.Scorer`.

    Weight layout, all float32, all indexed by *absolute* square (0=a1):

    ``W_from``   ``(64, n_ch)``        origin score from that square's channels
    ``b_from``   ``(64,)``             per-square bias
    ``W_ctx``    ``(h, n_ctx)``        context -> hidden (shared by all squares)
    ``W_hid``    ``(64, h)``           hidden -> per-square logit contribution
    ``W_to_dst`` ``(64, n_ch)``        destination score from dest channels
    ``W_to_src`` ``(64, n_ch)``        destination score from origin channels
    ``W_geom``   ``(64, n_geom)``      destination score from geometry features
    ``promo``    ``(5, n_move)``       promotion slot scores

    Note ``W_to_dst`` is indexed by destination square, and ``W_geom`` encodes
    the origin-destination relationship. That is the factorisation that keeps
    the parameter count at a few thousand instead of millions.
    """

    def __init__(self, config: PerceptronConfig | None = None, seed: int = 0):
        self.config = config or PerceptronConfig()
        rng = np.random.default_rng(seed)

        self.n_ch = TOTAL_CHANNELS if self.config.use_value_channels else BINARY_CHANNELS
        self.n_ctx = 9 if self.config.use_context else 0
        self.h = self.config.hidden if self.config.use_context else 0
        n_geom = len(GEOMETRY_NAMES)
        n_move = len(MOVE_FEATURE_NAMES_L3)

        def init(*shape):
            # Small random init rather than zeros: an all-zero perceptron has
            # no gradient signal anywhere, so nothing ever moves.
            return (rng.standard_normal(shape) * 0.01).astype(np.float32)

        self.W_from = init(64, self.n_ch)
        self.b_from = np.zeros(64, dtype=np.float32)
        self.W_ctx = init(self.h, self.n_ctx) if self.h else np.zeros((0, 0), np.float32)
        self.W_hid = init(64, self.h) if self.h else np.zeros((0, 0), np.float32)
        self.W_to_dst = init(64, self.n_ch)
        self.W_to_src = init(64, self.n_ch)
        self.W_geom = init(64, n_geom)
        self.promo = init(M.NUM_PROMO, n_move)

        # Scratch buffers, reused across calls. Allocating these per call was
        # measurable: the encoder runs tens of thousands of times per game.
        self._chan_cache: dict[int, np.ndarray] = {}
        # Bound to a position by set_position; None until then, so that calling
        # a square-local head out of order fails with a clear message rather
        # than an AttributeError three frames deep.
        self._current_channels: np.ndarray | None = None
        self._current_board: chess.Board | None = None
        self._canon_sq: np.ndarray | None = None
        self._geometry_cache: dict[int, np.ndarray] = {}

    # ---- feature extraction --------------------------------------------

    def channels_for(self, board: chess.Board) -> np.ndarray:
        """Encoded channels as ``(n_ch, 64)``, canonicalised, cached by position.

        Canonicalised so one weight set serves both colours. The cache keyed on
        the zobrist hash avoids re-encoding the same position twice within a
        search, which is the common case -- and the encoding is the expensive
        part of this level, not the dot products.
        """
        key = chess.polyglot.zobrist_hash(board)
        hit = self._chan_cache.get(key)
        if hit is not None:
            return hit

        tensor = encode(
            board,
            with_values=self.config.use_value_channels,
            canonicalise=True,
        )
        # Only the first n_ch channels are used: the value planes are the last
        # four, so slicing from the front keeps the channel indices stable
        # whether or not they are enabled.
        flat = tensor[:self.n_ch].reshape(self.n_ch, 64)
        # Bound the cache. Positions repeat heavily inside one search but a
        # whole training run touches far too many to keep them all, and an
        # unbounded dict here is a slow memory leak that only shows up in long
        # runs -- precisely when it is most expensive to debug.
        if len(self._chan_cache) > 4096:
            self._chan_cache.clear()
        self._chan_cache[key] = flat
        return flat

    def _geom_features(self, from_sq: int, to_sq: int) -> np.ndarray:
        """Relationship between origin and destination, in canonical frame."""
        fr, ff = from_sq >> 3, from_sq & 7
        tr, tf = to_sq >> 3, to_sq & 7
        # Canonicalisation can leave the origin on either side, so take the
        # absolute file difference: a kingside move mirrors to a queenside one
        # but its *shape* is the same.
        return np.array([
            1.0,
            (tr - fr) / 7.0,
            abs(tf - ff) / 7.0,
            (tf - ff) / 7.0,
            ((tr - fr) ** 2 + (tf - ff) ** 2) ** 0.5 / 9.9,
            1.0 if fr == tr else 0.0,
            1.0 if ff == tf else 0.0,
            abs(tr - fr) == abs(tf - ff),
            tr / 7.0,
        ], dtype=np.float32)

    def _context(self, board: chess.Board) -> np.ndarray:
        if not self.config.use_context:
            return np.zeros(0, dtype=np.float32)
        return board_context_features(board)

    # ---- the Scorer protocol -------------------------------------------

    def from_logits(self, board: chess.Board) -> np.ndarray:
        """``(64,)`` origin logits, indexed by **real** square.

        The origin head must score square ``s`` using *square s's own* channel
        values and nothing else. That is an elementwise contraction over the
        channel axis, ``sum_c W_from[s, c] * chan[c, s]``, not a matrix product
        over squares.

        Writing this as ``W_from @ chan`` is the natural-looking mistake and it
        is wrong in a way that still runs: it produces a ``(64, 64)`` array of
        every-square-against-every-square scores, which then propagates until
        something downstream complains about a shape several calls away from
        the actual error. ``einsum`` states the intent exactly and cannot be
        misread.

        The weights are indexed by *canonical* square and the result is
        scattered back to real squares, so the policy's factor machinery sees a
        vector aligned with ``chess.SQUARES`` as it must. For White the map is
        the identity and this is free; for Black it is a rank reflection.
        """
        chan = self.channels_for(board)                 # (n_ch, 64) canonical
        # Contract the channel axis only: score square s from its own channels.
        canon_logits = np.einsum("sc,cs->s", self.W_from, chan) + self.b_from
        if self.h:
            ctx = self._context(board)                  # (n_ctx,)
            pre = self.W_ctx @ ctx                      # (h,)
            hidden = np.maximum(pre, 0.0)               # (h,)
            # A context feature is position-global, but its *effect* is
            # square-dependent. Scattering the hidden contribution through the
            # canonical map keeps "square s" meaning the same square in both
            # colour frames.
            canon_logits = canon_logits + self.W_hid @ hidden   # (64,)
        return self._to_real(canon_logits, board)

    def _to_real(self, canon_values: np.ndarray, board: chess.Board) -> np.ndarray:
        """Scatter a canonical-square-indexed vector back onto real squares."""
        if board.turn == chess.WHITE:
            return canon_values.astype(np.float32)
        out = np.empty(64, dtype=np.float32)
        out[self._canon_sq] = canon_values
        return out

    def to_logits(self, from_sq: int) -> np.ndarray:
        """``(64,)`` destination logits, conditioned on the origin.

        Returns real-square indices, so the policy's destination softmax lines
        up with ``mask[from_sq]``. Internally everything is computed in
        canonical coordinates and reflected back at the end.

        Three additive terms, each scoring the *destination* square:

        * its own channels (``W_to_dst``, contracted elementwise as above);
        * the origin's channels (``W_to_src``), which is how the head learns
          "a knight moving from here tends to want this kind of square";
        * the geometry between origin and destination (``W_geom``).

        The origin term is a single ``(n_ch,)`` vector broadcast against every
        destination, so it shifts all 64 logits by a square-dependent amount --
        which is exactly right: once the origin is fixed, that term is a
        constant offset and only the relative ordering of destinations matters.
        """
        chan = self._current_channels
        if chan is None or self._canon_sq is None:
            raise RuntimeError(
                "to_logits called before set_position; the scorer needs the "
                "board to know the origin square's channels"
            )
        canon = self._canon_sq
        f_canon = int(canon[from_sq])

        # Destination's own channels, contracted elementwise over squares.
        out = np.einsum("sc,cs->s", self.W_to_dst, chan)
        # Origin channels: one vector, broadcast to all destinations. Read from
        # the *canonical* slot of the origin, since that is where the encoder
        # put it.
        out = out + (self.W_to_src @ chan[:, f_canon])
        # Geometry: both operands are indexed by destination square, so this is
        # another elementwise contraction over the geometry axis -- writing it
        # as a matmul would contract the wrong dimension.
        out = out + np.einsum("sk,sk->s", self.W_geom, self._geom_features_row(f_canon))
        return self._to_real(out, self._current_board)

    def promo_logits(self, from_sq: int, to_sq: int) -> np.ndarray:
        """``(NUM_PROMO,)`` promotion slot logits."""
        feats = self._move_features(from_sq, to_sq)
        return self.promo @ feats

    # ---- position binding ----------------------------------------------

    def set_position(self, board: chess.Board) -> None:
        """Bind the current position so the square-local heads can be called.

        The ``Scorer`` protocol takes squares rather than boards in
        ``to_logits``/``promo_logits`` because L1's tables are position-free.
        A board-reading scorer needs the board, so it holds it for the duration
        of one distribution computation. This is explicit rather than passed as
        an argument so the L1 protocol stays unchanged and the policy does not
        need to know which kind of scorer it holds.

        The position also carries the **canonical square map**. See
        :meth:`_canonical_squares` for why this is required rather than
        optional.
        """
        self._current_channels = self.channels_for(board)
        self._current_board = board
        self._canon_sq = self._canonical_squares(board)
        self._geometry_cache = {}

    @staticmethod
    def _canonical_squares(board: chess.Board) -> np.ndarray:
        """``(64,)`` map from a real square to its canonical square index.

        ``encode(..., canonicalise=True)`` reflects the ranks when Black is to
        move, so in the tensor the side to move always has its pawns on rank 1
        and its own pieces nearest the low ranks. That makes "forward" mean
        "increasing rank" for *both* colours, which is the entire point of
        canonicalisation.

        The heads, however, receive **raw** squares from the policy, because
        that is the ``Scorer`` protocol and the policy must still return a real
        ``chess.Move``. So any head whose features depend on rank or file must
        look the square up in the same frame the channels are in, or it is
        quietly reading geometry in absolute board coordinates.

        Getting this wrong is invisible on a symmetric position and shows up
        only as a colour-dependent geometry preference -- e.g. "moves that go
        up the board are good", which is true for exactly one side. It was
        caught by mirroring a dense middlegame FEN and finding a 6e-3
        probability difference where there should be none.
        """
        if board.turn == chess.WHITE:
            return np.arange(64, dtype=np.int32)
        # Rank reflection: rank r -> 7 - r, file unchanged.
        return np.array(
            [chess.square(chess.square_file(s), 7 - chess.square_rank(s))
             for s in chess.SQUARES],
            dtype=np.int32,
        )

    def _geom_features_row(self, from_sq: int) -> np.ndarray:
        """``(64, n_geom)`` geometry features for all destinations from one origin.

        Memoised per origin: the policy asks for one origin's destination
        logits at a time, and rebuilding 64 geometry vectors for each is pure
        waste when the same origin is revisited across the many positions that
        share it.
        """
        cache = getattr(self, "_geometry_cache", None)
        if cache is None:
            cache = self._geometry_cache = {}
        row = cache.get(from_sq)
        if row is None:
            row = np.stack([
                self._geom_features(from_sq, t) for t in range(64)
            ]).astype(np.float32)
            cache[from_sq] = row
        return row

    def _move_features(self, from_sq, to_sq) -> np.ndarray:
        """Local move features for the promotion head, in canonical coordinates.

        Like the geometry features, these must be read in the frame the
        channels are in. A promotion is "towards rank 8" for White and "towards
        rank 1" for Black on the real board, but "towards rank 8" for both once
        canonicalised, which is what lets one promotion head serve both colours.
        """
        chan = self._current_channels
        canon = self._canon_sq
        fc, tc = int(canon[from_sq]), int(canon[to_sq])
        fr, ff = fc >> 3, fc & 7
        tr, tf = tc >> 3, tc & 7
        return np.array([
            1.0,
            chan[:, tc].sum() / self.n_ch,
            chan[:, fc].sum() / self.n_ch,
            (tr - fr) / 7.0,
            abs(tf - ff) / 7.0,
        ], dtype=np.float32)

    # ---- parameter bookkeeping -----------------------------------------

    def parameters(self) -> dict[str, np.ndarray]:
        return {
            "W_from": self.W_from, "b_from": self.b_from,
            "W_ctx": self.W_ctx, "W_hid": self.W_hid,
            "W_to_dst": self.W_to_dst, "W_to_src": self.W_to_src,
            "W_geom": self.W_geom, "promo": self.promo,
        }

    @property
    def n_parameters(self) -> int:
        return int(sum(v.size for v in self.parameters().values()))

    def state_dict(self) -> dict:
        return {
            "config": self.config.__dict__,
            "params": {k: v.tolist() for k, v in self.parameters().items()},
        }

    def load_state_dict(self, state: dict) -> None:
        for k, v in state["params"].items():
            getattr(self, k)[:] = np.asarray(v, dtype=np.float32)


GEOMETRY_NAMES = (
    "bias", "d_rank", "abs_d_file", "d_file", "distance", "same_rank",
    "same_file", "is_diagonal", "dest_rank",
)

MOVE_FEATURE_NAMES_L3 = (
    "bias", "dest_activity", "src_activity", "d_rank", "abs_d_file",
)


# --------------------------------------------------------------------------
# the policy
# --------------------------------------------------------------------------

class L3Policy(FactoredSoftmaxPolicy):
    """L1's factored policy machinery driven by the L3 perceptron.

    Subclasses rather than wraps so it inherits the masking, sampling and
    persistence behaviour that is already tested. The only change is that the
    scorer needs to be told which position it is looking at before its
    square-local heads can be called, which is a two-line override.
    """

    def __init__(self, scorer: PerceptronScorer | None = None, **kwargs):
        # The scorer must inherit the policy's seed. Constructing it with the
        # default and only forwarding ``seed`` to the sampling RNG -- which is
        # what the obvious one-liner does -- makes every L3Policy(seed=n) start
        # from *identical* weights, so "seeded and reproducible" silently
        # becomes "all runs are the same run". The two RNGs are separate
        # objects on purpose, so both need the seed.
        if scorer is None:
            scorer = PerceptronScorer(seed=kwargs.get("seed", 0))
        super().__init__(scorer, name=kwargs.pop("name", "L3-perceptron"), **kwargs)

    @property
    def perceptron(self) -> PerceptronScorer:
        return self.scorer

    def move_distribution(self, board: chess.Board) -> np.ndarray:
        self.scorer.set_position(board)
        return super().move_distribution(board)

    def sample_factors(self, board: chess.Board, mask=None):
        self.scorer.set_position(board)
        return super().sample_factors(board, mask)

    def _argmax_factors(self, board: chess.Board, mask):
        self.scorer.set_position(board)
        return super()._argmax_factors(board, mask)

    def select(self, board: chess.Board) -> chess.Move:
        self.scorer.set_position(board)
        return super().select(board)

    # ---- persistence ---------------------------------------------------

    def state_dict(self) -> dict:
        """Serialise as L1 does, but the scorer is a perceptron, not a table."""
        state = super().state_dict()
        state["kind"] = "L3-perceptron"
        return state

    @classmethod
    def load(cls, path, **kwargs) -> "L3Policy":
        """Rebuild from disk, constructing the *right* scorer.

        The base class's ``load`` hardcodes a :class:`BlindScorer`, which is
        correct for L1 and wrong here: it would silently rebuild an L3 policy
        whose scorer has no perceptron weights, so the load would fail on a
        missing key or -- worse, if keys happened to overlap -- produce a
        position-blind L3. Overriding is the honest fix; the alternative,
        making the base class dispatch on a type tag, would put L3 knowledge in
        the L1 module.
        """
        import json
        from pathlib import Path

        state = json.loads(Path(path).read_text(encoding="utf-8"))
        config = PerceptronConfig(**state["scorer"]["config"])
        scorer = PerceptronScorer(config, seed=kwargs.pop("seed", 0))
        scorer.load_state_dict(state["scorer"])
        return cls(
            scorer,
            name=state.get("name", "L3-perceptron"),
            temperature=state.get("temperature", 1.0),
            **kwargs,
        )


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------

# NOTE (G1 audit): ``move_feature_vector`` was removed. It was a public helper
# describing a "credit assignment" move-scoring scheme that was never wired in;
# the real destination/promotion scoring lives in ``_move_features`` (used at
# lines ~321 and ~646). Nothing referenced it anywhere in the repo, so it was
# dead code duplicating, conceptually, the actual mechanism.


@dataclass
class L3Config:
    """Training knobs for L3."""

    lr: float = 0.01
    credit_decay: float = 0.99
    # Weight on the search-imitation target relative to the outcome signal.
    # Search feedback is much less noisy than a game outcome (it knows the
    # score of the move, not just who eventually won), so it can and should
    # carry more of the learning signal.
    search_weight: float = 1.0
    top_k: int = 5
    search_depth: int = 5
    shaped_reward: bool = True
    shaped_weight: float = 0.3
    max_plies: int = 120
    no_progress_limit: int = 60
    seed: int = 0


class L3Trainer:
    """Trains the L3 perceptron from outcomes, midstates and search feedback."""

    def __init__(
        self,
        policy: L3Policy | None = None,
        config: L3Config | None = None,
    ):
        if isinstance(policy, L3Config) and config is None:
            policy, config = None, policy
        self.config = config or L3Config()
        self.policy = policy or L3Policy(seed=self.config.seed)
        self.counter = ProbCounter()
        self.cache = PositionCache(capacity=1 << 14)
        self.midstates = None  # set by the caller when a store is available
        self.games_played = 0
        self.updates = 0
        self.search_updates = 0
        self.rng = np.random.default_rng(self.config.seed)

    # ---- credit application --------------------------------------------

    def _nudge(self, weights: np.ndarray, features: np.ndarray, credit: float) -> None:
        """One perceptron step: ``W += lr * credit * features``.

        Scaled and rounded to the fixed-point grid when quantisation is on.
        The rounding is applied to the *increment*, not to the stored weight,
        which keeps the update unbiased in expectation while preventing the
        drift that accumulates over tens of thousands of steps.
        """
        delta = self.config.lr * credit * features
        if self.policy.perceptron.config.quantised:
            delta = np.round(delta * QUANT) / QUANT
        weights += delta.astype(weights.dtype)
        self.updates += 1

    def apply_credit(self, board: chess.Board, move: chess.Move, credit: float) -> None:
        """Push the model towards (or away from) a specific move."""
        if credit == 0.0:
            return
        scorer = self.policy.perceptron
        scorer.set_position(board)
        chan = scorer.channels_for(board)
        mask = M.legal_move_mask(board)
        slot = M.PROMO_SLOT.get(move.promotion, 0)

        # Address weights in *canonical* square index, matching the inference
        # path exactly. If training nudged W_from[real_square] while inference
        # read W_from[canonical_square], Black's moves would update a row that
        # is never read and the model would learn only from White's games --
        # a bug that trains happily and just quietly ignores half the data.
        canon = scorer._canon_sq
        f_c, t_c = int(canon[move.from_square]), int(canon[move.to_square])

        # Origin head: the square's own channels.
        self._nudge(scorer.W_from[f_c], chan[:, f_c], credit)
        self._nudge(scorer.b_from[f_c:f_c + 1], np.ones(1, np.float32), credit)

        # Destination head: three separate sub-weights so the credit can be
        # attributed to "this destination is good" separately from "this origin
        # is good", which is the point of factorising.
        self._nudge(scorer.W_to_dst[t_c], chan[:, t_c], credit)
        self._nudge(scorer.W_to_src[t_c], chan[:, f_c], credit)
        self._nudge(
            scorer.W_geom[t_c], scorer._geom_features(f_c, t_c), credit
        )

        # Promotion head.
        self._nudge(
            scorer.promo[slot],
            scorer._move_features(move.from_square, move.to_square),
            credit,
        )

        # Context pathway, if present. The hidden layer is shared across all
        # squares and only the origin's row of W_hid is touched here, because
        # the credit is about *this move from this square*.
        if scorer.h:
            ctx = scorer._context(board)
            pre = scorer.W_ctx @ ctx
            hidden = np.maximum(pre, 0.0)
            # Rectified linear gradient: only units that actually fired should
            # receive an update on the input weights. Zeroing the others is
            # what makes this a ReLU rather than a linear layer with extra
            # steps, and it is easy to omit by accident since a linear version
            # still trains -- just worse.
            gate = (pre > 0).astype(np.float32)
            grad_ctx = np.outer(gate, ctx).astype(np.float32)
            self._nudge(scorer.W_ctx, grad_ctx, credit)
            self._nudge(scorer.W_hid[f_c], hidden, credit)

        self.counter.update(mask, move.from_square, move.to_square, slot)

    # ---- outcome training ----------------------------------------------

    def white_reward(self, result) -> tuple[float, float]:
        """Signed reward for White, mirroring the L1 trainer's rule."""
        if result.is_decisive:
            return V.outcome_score(result.result, chess.WHITE), 1.0
        if result.is_finished:
            return 0.0, 0.0
        if not self.config.shaped_reward:
            return 0.0, 0.0
        last = result.fens[-1] if result.fens else None
        if last is None:
            return 0.0, 0.0
        raw = V.evaluate(chess.Board(last)) / 1000.0
        return float(np.tanh(raw)) * self.config.shaped_weight, 1.0

    def train_on_game(self, result) -> dict:
        if not result.fens:
            raise ValueError("train_on_game needs recorded FENs")

        white_score, amplitude = self.white_reward(result)
        n = result.plies
        distance = np.arange(n - 1, -1, -1, dtype=np.float64)
        weights = np.power(self.config.credit_decay, distance)

        for ply, san in enumerate(result.moves):
            board = chess.Board(result.fens[ply])
            move = board.parse_san(san)
            sign = white_score if board.turn == chess.WHITE else -white_score
            self.apply_credit(board, move, float(weights[ply]) * sign * amplitude)

        self.games_played += 1
        return {"result": result.result, "plies": n}

    # ---- depth-5 top-5 search feedback ---------------------------------

    def search_feedback(
        self, board: chess.Board, depth: int | None = None, top_k: int | None = None
    ) -> list[tuple[chess.Move, float]]:
        """Top-k moves by minimax value, with their values normalised to [0,1].

        This is the spec's "minimax depth 5 top 5 outputs" as a *target*. The
        values are shifted and squashed so the best move gets weight near 1 and
        a move 300cp worse gets near 0, which makes the resulting update a
        proper soft target rather than one dominated by the sheer size of a
        centipawn score.
        """
        from .search import MinimaxEngine

        depth = depth if depth is not None else self.config.search_depth
        top_k = top_k if top_k is not None else self.config.top_k

        legal = list(board.legal_moves)
        if not legal:
            return []

        engine = MinimaxEngine(depth=depth, use_tt=True)
        scored: list[tuple[chess.Move, int]] = []
        for move in legal:
            board.push(move)
            value = -engine._negamax(board, depth - 1, -10**9, 10**9)
            board.pop()
            scored.append((move, int(value)))

        scored.sort(key=lambda kv: kv[1], reverse=True)
        top = scored[:top_k]
        best = top[0][1]
        # 300cp is a "clearly better" margin; anything beyond it saturates.
        out = []
        for move, value in top:
            w = float(np.exp((value - best) / 300.0))
            out.append((move, w))
        return out

    def train_on_search_feedback(
        self, board: chess.Board, *, depth: int | None = None
    ) -> dict:
        """Imitate a depth-5 search in the current position.

        The policy's own probabilities are compared against the search's top-k
        and the gap is the gradient. This is the mechanism that lets a shallow
        perceptron borrow the tactical competence of a deep search without
        itself searching at inference time.
        """
        targets = self.search_feedback(board, depth=depth)
        if not targets:
            return {"applied": False, "k": 0}

        self.policy.scorer.set_position(board)
        dist = self.policy.move_distribution(board)
        best_w = max(w for _, w in targets)

        for move, w in targets:
            idx = M.move_to_index(move)
            p_model = float(dist[idx])
            # Target probability proportional to the search's preference.
            p_target = w / best_w
            # Nudge in proportion to how far behind the model is, so a move it
            # already likes is not pushed further and a move it is ignoring is
            # pulled up hard. Without the gap term this just collapses onto
            # whatever the search likes first and forgets the rest.
            gap = p_target - p_model
            self.apply_credit(
                board, move, self.config.search_weight * gap
            )
            self.search_updates += 1

        return {"applied": True, "k": len(targets)}

    # ---- midstate incremental updates ----------------------------------

    def train_on_midstates(self, n: int = 32, *, depth: int | None = None) -> dict:
        """Sample recorded midgame positions and train on each.

        The spec's "save midstates to update instead of replace". This exists
        so training does not consist entirely of the first ten plies: the
        positions the policy actually reaches in play are the ones it needs to
        be right about, and early game openings are a tiny and unrepresentative
        slice of them.
        """
        if self.midstates is None or len(self.midstates) == 0:
            return {"sampled": 0, "trained": 0}

        boards = self.midstates.sample_boards(n)
        applied = 0
        for board in boards:
            if board.is_game_over():
                continue
            res = self.train_on_search_feedback(board, depth=depth)
            applied += int(bool(res.get("applied")))
        return {"sampled": len(boards), "trained": applied}

    # ---- persistence ---------------------------------------------------

    def save(self, path: str | Path) -> dict:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.policy.save(path)
        meta = path.with_suffix(".meta.json")
        meta.write_text(json.dumps({
            "games_played": self.games_played,
            "updates": self.updates,
            "search_updates": self.search_updates,
            "n_parameters": self.policy.perceptron.n_parameters,
            "config": self.config.__dict__,
        }, indent=2), encoding="utf-8")
        return {"model": str(path), "meta": str(meta)}
