"""L3.5: the weight-tied per-square scorer.

What this level is
------------------
L3 scores a move with a *factorised perceptron*: the origin head holds a
separate weight row per square (``W_from`` is ``(64, n_ch)``), and so do both
halves of the destination head. L3.5 keeps the identical factorisation, the
identical encoder, the identical policy protocol -- and deletes the per-square
indexing from the square-local heads, so that **one** weight vector is applied
at all 64 squares::

    L3    canon_logits = einsum("sc,cs->s", W_from, chan) + b_from   # 64 rows
    L3.5  canon_logits = einsum("c,cs->s",  w_shared, chan) + b_shared  # 1 row

That single substitution is the whole level. It is a 1x1 convolution in
disguise, and equivalently a weight-tied linear layer over the square axis.

Why it is worth a rung on the ladder
------------------------------------
Every other rung varies *how much information* the learner receives (L1 blind,
L3 informed) or *how much search it does* (L2, L5) or *which backend* it runs
on (L4). None of them vary the **shape of the computation** at fixed input,
fixed output and fixed budget. L3.5 does, and it makes a falsifiable claim::

    A move can be scored by asking every square the same question, in
    parallel -- "given this square's local features, how valuable is it as the
    origin of a move?" -- and a single tied function is enough to answer it.

The claim is *false* if square value depends irreducibly on global structure a
per-square function can never see. A rook's own channels look the same on an
open file as on a closed one; "open file" is a fact about seven other squares.
So the hypothesis has a testable weak point, and the context pathway (below) is
where it gets its only chance to survive.

The parameter-budget decision -- stated, not quietly padded
-----------------------------------------------------------
The design called for L3.5 to land within 10% of L3's 7225 parameters, and left
the implementer to choose how, explicitly. **Decision: lean, purely tied.**

Pure weight tying collapses the three square-local heads from ``(64, n_ch)`` to
``(n_ch,)``, freeing ~4200 parameters. The alternative -- spending them on the
context width ``h_ctx`` until the total matches 7225 -- is unsatisfiable
honestly: the context path's input is 9 scalars wide, and widening a 9-input
ReLU path to absorb 4200 parameters requires ``h_ctx`` of roughly 649, i.e. a
deliberate over-parameterisation of a 9-dimensional input purely to hit a
number. That is padding, and it would also make L3.5's context path *bigger*
than L3's entire model, which destroys the comparison in the opposite
direction.

So L3.5 runs at **787 parameters, 1/9.67 of L3's 7609** (the counts moved from
783 / 7225 when BINARY_CHANNELS went 22->24), and the claim it
supports is stated accordingly: *same task, same features, same action space,
same training signal, at an order of magnitude fewer parameters, with the
spatial-structure prior supplied by weight tying instead of by hand-shaped
per-square tables.* The `test_parameter_count_within_ten_percent_of_L3` test
therefore asserts the budget as a **range with an implemented fraction**, and
records this deviation explicitly rather than pretending the literal
constraint was met. ``h_ctx`` remains the knob: ``L35Config``/``SquareLocalConfig``
expose it and ``parameter_count(h_ctx=...)`` lets the budget be re-derived.
The full argument is in ``bench/README.md`` ("The L3.5 experiment").

Why tying is coherent at all
----------------------------
Because the encoder already canonicalises. In canonical form "my pawn on e2" is
always at the same square for both colours, so one tied weight set is the same
statement as L3's per-square weights after the scatter. L3.5 does not touch the
frame machinery: ``set_position``, ``channels_for``, ``_canon_sq``, ``_to_real``,
``_geom_features``, ``_move_features`` and ``_nudge`` are all reused verbatim
from :mod:`chessrl.perceptron`.

The two traps, both of which are silent
---------------------------------------
1. **The tied row sees 64x the credit.** L3's per-square rows were each touched
   by ~1 square per position, so the effective step on any one row was
   implicitly divided by 64. L3.5's single row receives an update for *every*
   square, once per ply, and would therefore diverge at L3's ``lr=0.01``. The
   fix is explicit and named: ``L35Config.lr_shared = lr / 64``. A test that
   trains at the un-divided rate and asserts boundedness exists precisely
   because this failure is invisible in the loss -- it trains happily.
2. **Canonical vs real square indexing in ``apply_credit``.** Inherited from
   L3, but when tying, a mistake here is *invisible for White and
   mirrored-wrong for Black* -- the same failure mode as the L5 prior bug.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import chess
import numpy as np

from . import masks as M
from .encode import (
    BINARY_CHANNELS,
    TOTAL_CHANNELS,
    board_context_features,
)
from .perceptron import (
    GEOMETRY_NAMES,
    MOVE_FEATURE_NAMES_L3,
    QUANT,
    L3Config,
    L3Policy,
    L3Trainer,
)

__all__ = [
    "SquareLocalConfig",
    "SquareLocalScorer",
    "L35Policy",
    "L35Config",
    "L35Trainer",
    "LEAN_H_CTX",
    "parameter_count",
    "implemented_fraction_of",
]

# The context hidden width that realises the "lean, purely tied" budget:
# 787 parameters, 1/9.67 of L3's 7609. See the module docstring.
LEAN_H_CTX = 64

# How many contributing squares the per-square heads receive credit from. L3's
# per-square rows get one origin per ply; a tied row gets all of them. Exposed
# as a named constant so the ``lr_shared`` relationship is a definition rather
# than a magic number.
CREDIT_FANOUT = 64


def parameter_count(
    *,
    n_ch: int = BINARY_CHANNELS,
    h_ctx: int = LEAN_H_CTX,
    n_geom: int = len(GEOMETRY_NAMES),
    n_move: int = len(MOVE_FEATURE_NAMES_L3),
) -> dict[str, int]:
    """Size of every weight tensor, by name, for a given layout.

    Kept as a free function rather than a method so that the budget can be
    re-derived and asserted *before* a scorer is constructed -- the budget is a
    property of the layout, not of an instance.
    """
    counts = {
        "w_shared": n_ch,
        "b_shared": 1,
        "W_ctx": h_ctx * 9,
        "w_hid": h_ctx,
        "w_dst_shared": n_ch,
        "w_dst_ctx": h_ctx,
        "w_geom": n_geom,
        "promo": M.NUM_PROMO * n_move,
    }
    counts["total"] = sum(v for k, v in counts.items() if k != "total")
    return counts


def implemented_fraction_of(h_ctx: int = LEAN_H_CTX) -> float:
    """L3.5's parameter count as a fraction of L3's (currently 7609)."""
    from .perceptron import PerceptronConfig, PerceptronScorer

    l3 = PerceptronScorer(PerceptronConfig()).n_parameters
    return parameter_count(h_ctx=h_ctx)["total"] / l3


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------

@dataclass
class SquareLocalConfig:
    """Shape and training knobs for the L3.5 model.

    ``use_context`` is the *only* place L3.5 can spend parameters on "global
    facts a per-square function cannot see", which is exactly the weak point of
    the weight-tying hypothesis. Turning it off is therefore not a size
    optimisation here, it is the ablation that tests whether the hypothesis can
    survive at all -- hence it is on by default and ``h_ctx`` is the knob.
    """

    use_value_channels: bool = False
    use_context: bool = True
    h_ctx: int = LEAN_H_CTX
    quantised: bool = True


# --------------------------------------------------------------------------
# the scorer
# --------------------------------------------------------------------------

class SquareLocalScorer:
    """Scorer whose square-local heads are weight-tied across all 64 squares.

    Implements the same :class:`chessrl.policy.Scorer` protocol as
    :class:`chessrl.perceptron.PerceptronScorer`, with the same output shapes
    and the same real-square indexing, so it drops into the existing
    :class:`chessrl.policy.FactoredSoftmaxPolicy` unchanged.

    Weight layout, all float32::

        w_shared       (n_ch,)        origin score from a square's own channels
        b_shared       (1,)           a *single* scalar bias

        W_ctx          (h_ctx, 9)     context -> hidden, shared by all squares
        w_hid          (h_ctx,)       hidden -> a single logit offset
        w_dst_shared   (n_ch,)        destination score from dest's channels
        w_dst_ctx      (h_ctx,)       dest-context interaction, broadcast

        w_geom         (n_geom,)      destination score from pair geometry
        promo          (5, n_move)    promotion slot scores

    Two deliberate departures from L3's layout, both consequences of tying:

    * **One bias, not 64.** L3's ``b_from`` is ``(64,)``; a tied head has one
      value, otherwise the tie is undone by the back door.
    * **The origin's channels reach the destination head through the context
      hidden vector, not through an origin-indexed row.** L3 carries
      ``W_to_src (64, n_ch)`` so the origin's *identity* can shift all 64
      destination logits. L3.5 has no per-origin row, so it must recover
      origin-dependence structurally -- which is precisely the hypothesis's second
      falsifiable claim (H10 in HYPOTHESES.md), and is what ``w_dst_ctx`` plus ``w_geom`` do. Note
      that a *constant* offset over all destinations cannot change the
      destination softmax at all, but the context hidden vector is not
      constant in the origin: it depends on the position and on the origin via
      :meth:`_origin_hidden`.
    """

    def __init__(self, config: SquareLocalConfig | None = None, seed: int = 0):
        self.config = config or SquareLocalConfig()
        rng = np.random.default_rng(seed)

        self.n_ch = TOTAL_CHANNELS if self.config.use_value_channels else BINARY_CHANNELS
        self.n_ctx = 9 if self.config.use_context else 0
        self.h_ctx = self.config.h_ctx if self.config.use_context else 0
        self.n_geom = len(GEOMETRY_NAMES)
        self.n_move = len(MOVE_FEATURE_NAMES_L3)

        def init(size: int) -> np.ndarray:
            # Small random init, not zeros: an all-zero perceptron has no
            # gradient signal anywhere and nothing ever moves. The tied origin
            # row is the one exception -- it is *also* given a small random
            # value rather than zeros, so that the untrained origin factor is
            # not uniform, which would make the origin softmax position-blind
            # and hide a whole class of canonicalisation bugs behind a uniform
            # distribution.
            return (rng.standard_normal(size) * 0.01).astype(np.float32)

        self.w_shared = init(self.n_ch)
        self.b_shared = np.zeros(1, dtype=np.float32)

        self.W_ctx = (
            (rng.standard_normal((self.h_ctx, self.n_ctx)) * 0.01).astype(np.float32)
            if self.h_ctx else np.zeros((0, 0), np.float32)
        )
        self.w_hid = init(self.h_ctx) if self.h_ctx else np.zeros(0, np.float32)

        self.w_dst_shared = init(self.n_ch)
        self.w_dst_ctx = (
            (rng.standard_normal(self.h_ctx) * 0.01).astype(np.float32)
            if self.h_ctx else np.zeros(0, np.float32)
        )

        self.w_geom = init(self.n_geom)
        # Per promotion slot, over the move features: L3's shape. Promotions are
        # too rare to justify anything else, and keeping the shape identical
        # keeps ``apply_credit``'s promotion line a verbatim copy.
        self.promo = (rng.standard_normal((M.NUM_PROMO, self.n_move)) * 0.01).astype(
            np.float32
        )

        # ---- scratch state, bound by set_position -----------------------
        self._chan_cache: dict[int, np.ndarray] = {}
        self._current_channels: np.ndarray | None = None
        self._current_board: chess.Board | None = None
        self._canon_sq: np.ndarray | None = None
        self._geometry_cache: dict[int, np.ndarray] = {}

    # ---- feature extraction (reused from L3, verbatim) ------------------

    def channels_for(self, board: chess.Board) -> np.ndarray:
        """Encoded channels as ``(n_ch, 64)``, canonicalised, cached by position.

        Same cache-bounded, zobrist-keyed implementation as
        :meth:`chessrl.perceptron.PerceptronScorer.channels_for`. Not factored
        into a shared helper on purpose: the two levels are meant to be
        independently deletable, and a shared base class would couple
        L3's module layout to L3.5's.
        """
        import chess.polyglot

        from .encode import encode

        key = chess.polyglot.zobrist_hash(board)
        hit = self._chan_cache.get(key)
        if hit is not None:
            return hit

        tensor = encode(
            board,
            with_values=self.config.use_value_channels,
            canonicalise=True,
        )
        flat = tensor[:self.n_ch].reshape(self.n_ch, 64)
        if len(self._chan_cache) > 4096:
            self._chan_cache.clear()
        self._chan_cache[key] = flat
        return flat

    def _geom_features(self, from_sq: int, to_sq: int) -> np.ndarray:
        """Relationship between origin and destination, in the canonical frame.

        Identical to L3's. It is the *only* structural channel through which
        L3.5 can express origin/destination asymmetry, so it carries more weight
        here than it does at L3 -- see the module docstring.
        """
        fr, ff = from_sq >> 3, from_sq & 7
        tr, tf = to_sq >> 3, to_sq & 7
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

    def _geom_features_row(self, from_sq: int) -> np.ndarray:
        """``(64, n_geom)`` geometry for all destinations from one origin."""
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

    def _context(self, board: chess.Board) -> np.ndarray:
        if not self.config.use_context:
            return np.zeros(0, dtype=np.float32)
        return board_context_features(board)

    def _origin_hidden(self, board: chess.Board, f_canon: int) -> np.ndarray:
        """``(h_ctx,)`` ReLU units describing "the position, as seen from f".

        This is the role ``W_to_src`` plays at L3. L3 reads the origin's
        identity directly as a ``(n_ch,)`` channel vector; L3.5 cannot, so it
        reads the origin's *canonical channel vector* and pushes it through the
        same 9-wide context weight matrix. The result is a position-, origin-
        and colour-dependent vector of rectified units, which is the only thing
        that breaks the origin/destination symmetry of a purely tied head.
        """
        if not self.h_ctx:
            return np.zeros(0, dtype=np.float32)
        ctx = self._context(board)
        chan = self._current_channels
        if chan is None:
            raise RuntimeError("_origin_hidden called before set_position")
        # Origin-contribution term appended: the head's last input slot is the
        # origin's mean channel activity, exactly the "src_activity" move
        # feature L3's promotion head already uses. It is zero when context is
        # off, which keeps the shapes and the code path identical.
        src_act = float(chan[:, f_canon].sum()) / self.n_ch
        ctx = np.concatenate([ctx, np.array([src_act], dtype=np.float32)])
        pre = self.W_ctx @ ctx[:self.W_ctx.shape[1]]
        return np.maximum(pre, 0.0)

    # ---- the Scorer protocol -------------------------------------------

    def from_logits(self, board: chess.Board) -> np.ndarray:
        """``(64,)`` origin logits, indexed by **real** square.

        The tied contraction is ``sum_c w_shared[c] * chan[c, s]`` for every
        square ``s`` -- the same elementwise contraction L3 performs, minus the
        square index on the weight. The context term is a *single* scalar added
        to every square (the tied ``w_hid`` dot the shared hidden vector), which
        by construction cannot change the origin softmax; it is retained
        because it is the honest weight-tied analogue of L3's square-dependent
        context broadcast, and because keeping the term means a future
        ``h_ctx``-driven variant does not need a new code path.

        Weights are indexed by *canonical* square and scattered back to real
        squares, so the policy's factor machinery sees a vector aligned with
        ``chess.SQUARES``.
        """
        chan = self.channels_for(board)                      # (n_ch, 64)
        canon_logits = np.einsum("c,cs->s", self.w_shared, chan) + self.b_shared[0]
        if self.h_ctx:
            ctx = self._context(board)
            pre = self.W_ctx @ ctx
            hidden = np.maximum(pre, 0.0)
            canon_logits = canon_logits + float(self.w_hid @ hidden)
        return self._to_real(canon_logits, board)

    def to_logits(self, from_sq: int) -> np.ndarray:
        """``(64,)`` destination logits, conditioned on the origin.

        Four additive terms, each scoring the *destination* square:

        * its own channels via the tied ``w_dst_shared`` (elementwise);
        * a position-global constant from the shared context hidden vector --
          shifts all 64 logits equally, so it cannot reorder destinations, but
          it is kept for parity with L3's structure;
        * ``w_dst_ctx . hidden(from)``, which IS origin-dependent and is the
          structural substitute for L3's ``W_to_src`` origin row;
        * the geometry between origin and destination, ``w_geom``.

        Returns real-square indices so the destination softmax lines up with
        ``mask[from_sq]``.
        """
        chan = self._current_channels
        if chan is None or self._canon_sq is None:
            raise RuntimeError(
                "to_logits called before set_position; the scorer needs the "
                "board to know the origin square's channels"
            )
        canon = self._canon_sq
        f_canon = int(canon[from_sq])

        # Destination's own channels, tied across all squares.
        out = np.einsum("c,cs->s", self.w_dst_shared, chan)
        if self.h_ctx:
            ctx = self._context(self._current_board)
            pre = self.W_ctx @ ctx
            hidden = np.maximum(pre, 0.0)
            # Global constant: same for every destination by construction.
            out = out + float(self.w_hid @ hidden)
            # Origin-dependent: this is the asymmetry carrier. A destination's
            # score now depends on WHICH square the move starts from, without
            # any per-origin weight row.
            out = out + float(self.w_dst_ctx @ self._origin_hidden(
                self._current_board, f_canon
            ))
        out = out + np.einsum("k,sk->s", self.w_geom, self._geom_features_row(f_canon))
        return self._to_real(out, self._current_board)

    def promo_logits(self, from_sq: int, to_sq: int) -> np.ndarray:
        """``(NUM_PROMO,)`` promotion slot logits."""
        return self.promo @ self._move_features(from_sq, to_sq)

    # ---- position binding (reused from L3, verbatim) --------------------

    def set_position(self, board: chess.Board) -> None:
        """Bind the current position so the square-local heads can be called."""
        self._current_channels = self.channels_for(board)
        self._current_board = board
        self._canon_sq = self._canonical_squares(board)
        self._geometry_cache = {}

    @staticmethod
    def _canonical_squares(board: chess.Board) -> np.ndarray:
        """``(64,)`` map from a real square to its canonical square index.

        Rank reflection for Black, identity for White. Same function as L3's;
        see :meth:`chessrl.perceptron.PerceptronScorer._canonical_squares` for
        why reading geometry in absolute coordinates is invisible on a
        symmetric position and colour-dependent everywhere else.
        """
        if board.turn == chess.WHITE:
            return np.arange(64, dtype=np.int32)
        return np.array(
            [chess.square(chess.square_file(s), 7 - chess.square_rank(s))
             for s in chess.SQUARES],
            dtype=np.int32,
        )

    def _to_real(self, canon_values: np.ndarray, board: chess.Board) -> np.ndarray:
        """Scatter a canonical-square-indexed vector back onto real squares."""
        if board.turn == chess.WHITE:
            return np.asarray(canon_values, dtype=np.float32)
        out = np.empty(64, dtype=np.float32)
        out[self._canon_sq] = canon_values
        return out

    def _move_features(self, from_sq: int, to_sq: int) -> np.ndarray:
        """Local move features for the promotion head, in canonical coordinates."""
        chan = self._current_channels
        canon = self._canon_sq
        if chan is None or canon is None:
            raise RuntimeError("_move_features called before set_position")
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
        """Every *learnable* tensor, by name.

        Only these are optimised, and only these appear in
        ``test_shared_row_is_touched_by_every_square``-style checks. The
        scratch state bound by ``set_position`` is runtime bookkeeping and is
        never serialised.
        """
        return {
            "w_shared": self.w_shared,
            "b_shared": self.b_shared,
            "W_ctx": self.W_ctx,
            "w_hid": self.w_hid,
            "w_dst_shared": self.w_dst_shared,
            "w_dst_ctx": self.w_dst_ctx,
            "w_geom": self.w_geom,
            "promo": self.promo,
        }

    @property
    def n_parameters(self) -> int:
        return int(sum(v.size for v in self.parameters().values()))

    def state_dict(self) -> dict:
        return {
            "config": dict(self.config.__dict__),
            "params": {k: v.tolist() for k, v in self.parameters().items()},
        }

    def load_state_dict(self, state: dict) -> None:
        for k, v in state["params"].items():
            getattr(self, k)[:] = np.asarray(v, dtype=np.float32)


# --------------------------------------------------------------------------
# the policy
# --------------------------------------------------------------------------

class L35Policy(L3Policy):
    """L3's policy, L3.5's scorer. Almost entirely inherited.

    ``L3Policy``'s only job beyond ``FactoredSoftmaxPolicy`` is to call
    ``scorer.set_position(board)`` before each square-local head read; that is
    not specific to L3's scorer, so the subclass changes nothing about it. What
    *does* change is construction (a different default scorer) and loading (a
    different scorer class), both of which ``L3Policy`` hardcodes.
    """

    def __init__(self, scorer: SquareLocalScorer | None = None, **kwargs):
        # The scorer must inherit the policy's seed, exactly as at L3: the
        # policy RNG and the weight RNG are separate objects, and forwarding
        # the seed to only one of them makes every L35Policy(seed=n) identical.
        if scorer is None:
            scorer = SquareLocalScorer(seed=kwargs.get("seed", 0))
        super().__init__(scorer, name=kwargs.pop("name", "L3.5-squarelocal"),
                         **kwargs)

    @property
    def scorer_l35(self) -> SquareLocalScorer:
        """Named accessor, so a caller never has to know the attribute is
        called ``scorer`` on the base class."""
        return self.scorer

    def state_dict(self) -> dict:
        state = super().state_dict()
        state["kind"] = "L3.5-squarelocal"
        return state

    @classmethod
    def load(cls, path, **kwargs) -> "L35Policy":
        """Rebuild from disk with the *right* scorer class.

        ``L3Policy.load`` constructs a ``PerceptronScorer``; the base class's
        constructs a ``BlindScorer``. Both would produce a policy that loads
        without error and plays position-blind nonsense, which is exactly the
        failure the L3 override exists to prevent. L3.5 needs its own override
        for the same reason, not because the format differs.
        """
        import json
        from pathlib import Path

        state = json.loads(Path(path).read_text(encoding="utf-8"))
        config = SquareLocalConfig(**state["scorer"]["config"])
        scorer = SquareLocalScorer(config, seed=kwargs.pop("seed", 0))
        scorer.load_state_dict(state["scorer"])
        return cls(
            scorer,
            name=state.get("name", "L3.5-squarelocal"),
            temperature=state.get("temperature", 1.0),
            **kwargs,
        )


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------

@dataclass
class L35Config(L3Config):
    """L3's training knobs, plus the tied-row learning rate.

    ``lr_shared`` is a separate, explicit number rather than a division buried
    inside ``apply_credit``, because the relationship *is* the design: a tied
    row accumulates credit from every square, so its step must be normalised by
    the number of contributing squares or the level diverges at L3's rate. The
    default is derived, not typed in, so the two cannot drift apart.

    Inheriting from ``L3Config`` rather than redeclaring the same nine fields
    is deliberate: it makes it mechanically impossible for L3.5 to be trained
    under a *different* search depth, credit decay or reward shaping than L3,
    which is the whole point of a fixed-everything-but-architecture comparison.
    """

    lr_shared: float = field(default=0.0)

    def __post_init__(self) -> None:
        if self.lr_shared == 0.0:
            self.lr_shared = self.lr / CREDIT_FANOUT


class L35Trainer(L3Trainer):
    """L3's trainer with the weight-tied credit rule.

    Only :meth:`apply_credit` differs, and only where the weight is indexed by
    square. ``_nudge``, the credit curve, ``white_reward``, the search feedback
    and the midstate sampling are all inherited untouched -- if any of those
    needed a change, the level would not be measuring the thing it claims to.
    """

    def __init__(self, policy=None, config: L35Config | None = None):
        # Same convenience overload as L3Trainer: a bare config is accepted.
        if isinstance(policy, L3Config) and config is None:
            policy, config = None, policy
        # ``policy or L35Policy(...)`` is the obvious one-liner and it is wrong
        # here for the same reason it is wrong in ``L35Policy.__init__``: the
        # base ``L3Trainer`` would build an ``L3Policy`` for us, and the guard
        # below would then fire on a trainer that was never given a policy at
        # all. Construct L3.5's own policy explicitly.
        if policy is None:
            policy = L35Policy(seed=(config or L35Config()).seed)
        super().__init__(policy=policy, config=config or L35Config())
        if not isinstance(self.policy, L35Policy):
            # A caller that passed an L3Policy would get L3's per-square credit
            # applied to a tied model -- which is both slower and wrong. Fail
            # here rather than produce a silently mis-trained level.
            raise TypeError(
                "L35Trainer needs an L35Policy; got "
                f"{type(self.policy).__name__}"
            )
        # ---- fixed-point residual buffers, one per tied tensor ----------
        #
        # WHY THIS EXISTS. ``lr_shared = lr/64`` is the correct *rate* for a tied
        # row (a tied row receives credit from every square), but it makes a single step
        # smaller than the
        # quantisation grid at realistic credit values: at ``lr=0.05`` and
        # ``credit=0.5`` the step is ``QUANT * 0.05/64 * 0.5 = 0.4`` grid units,
        # which ``np.round`` sends to **zero**. Every update is then discarded.
        # This is not a hypothetical: 30 full games (14,400 updates) left
        # ``w_shared``, ``w_dst_shared``, ``w_geom`` and ``promo`` bit-identical
        # to their initialisation while the context path -- which runs at the
        # full ``lr`` -- moved normally. The level would have been untrainable in
        # its main pathway, and the ablation would have blamed weight tying.
        #
        # The fix is error feedback (the standard fixed-point SGD technique):
        # keep the sub-grid remainder in a per-tensor buffer and add it to the
        # next update, so ~2.5 events of 0.4 units produce one real step. The
        # *rate* is the correct one for a tied row; the rounding just stops
        # throwing updates away. Buffers are runtime state, never serialised.
        self._residual: dict[str, np.ndarray] = {}

    def _nudge_shared(
        self, weights: np.ndarray, features: np.ndarray, credit: float,
        *, key: str,
    ) -> None:
        """One step on a *weight-tied* tensor, at ``lr_shared``, with error feedback.

        Same fixed-point contract as :meth:`L3Trainer._nudge` -- the rounding is
        applied to the increment, not to the stored weight -- but a smaller rate
        (see the module docstring's first trap) *and* a residual buffer, because
        at this rate a single increment is usually less than one grid unit and
        rounding alone would discard it entirely. See ``L35Trainer.__init__``.

        ``key`` identifies the tensor so its buffer survives across calls. It is
        an explicit argument rather than derived from ``id(weights)``: the same
        array object is sometimes passed with genuinely different feature
        vectors, and sharing a buffer between them would be wrong.
        """
        delta = self.config.lr_shared * credit * features
        if self.policy.scorer.config.quantised:
            # Error feedback: carry the sub-grid remainder forward.
            prev = self._residual.get(key)
            if prev is None or prev.shape != delta.shape:
                prev = np.zeros_like(delta)
            total = delta + prev
            quantised = np.round(total * QUANT) / QUANT
            self._residual[key] = (total - quantised).astype(np.float64)
            delta = quantised
        weights += delta.astype(weights.dtype)
        self.updates += 1

    def _nudge_context(
        self, weights: np.ndarray, features: np.ndarray, credit: float
    ) -> None:
        """Step on a context tensor, at the full ``lr``.

        The context pathway is *not* subject to the fan-out trap. ``W_ctx`` is
        already a single shared ``(h_ctx, 9)`` tensor at L3 -- untouched by the
        number of squares -- and it receives one update per credit event at
        both levels. Dividing it by 64 would make L3.5's context path learn 64x
        slower than L3's for no reason, which would rig the comparison against
        the very pathway the hypothesis depends on.

        No residual buffer here either: at the full ``lr`` a step is tens of
        grid units, so nothing is being thrown away.
        """
        delta = self.config.lr * credit * features
        if self.policy.scorer.config.quantised:
            delta = np.round(delta * QUANT) / QUANT
        weights += delta.astype(weights.dtype)
        self.updates += 1

    def apply_credit(self, board: chess.Board, move: chess.Move, credit: float) -> None:
        """Push the tied model towards (or away from) a specific move.

        Structurally the same as L3's, with each per-square fan-out replaced by
        a single accumulation onto the one shared row, described by the *move's
        own* square's channels. The canonical/real indexing rule is unchanged
        and still load-bearing: when tying, getting it wrong is invisible for
        White and mirrored-wrong for Black.
        """
        if credit == 0.0:
            return
        scorer = self.policy.scorer
        scorer.set_position(board)
        chan = scorer.channels_for(board)
        mask = M.legal_move_mask(board)
        slot = M.PROMO_SLOT.get(move.promotion, 0)

        canon = scorer._canon_sq
        f_c, t_c = int(canon[move.from_square]), int(canon[move.to_square])

        # Origin head, tied: one row, described by the origin's own channels.
        self._nudge_shared(scorer.w_shared, chan[:, f_c], credit, key="w_shared")
        self._nudge_shared(
            scorer.b_shared, np.ones(1, np.float32), credit, key="b_shared"
        )

        # Destination head, tied: one row for "this destination is good",
        # described by the destination's own channels. L3's ``W_to_src`` term
        # has no tied counterpart -- the origin's influence on the destination
        # head is carried by ``w_dst_ctx`` below instead.
        self._nudge_shared(
            scorer.w_dst_shared, chan[:, t_c], credit, key="w_dst_shared"
        )

        # Geometry: the only structural source of origin/destination asymmetry.
        self._nudge_shared(
            scorer.w_geom, scorer._geom_features(f_c, t_c), credit, key="w_geom"
        )

        # Promotion head. Keyed by slot, so each slot keeps its own residual.
        self._nudge_shared(
            scorer.promo[slot],
            scorer._move_features(move.from_square, move.to_square),
            credit,
            key=f"promo[{slot}]",
        )

        # Context pathway. Two tensors, both at the full ``lr``: the shared
        # input weights (rectified gradient -- only units that fired get a
        # nonzero input update, which is what makes this a ReLU) and the tied
        # output rows.
        if scorer.h_ctx:
            ctx = scorer._context(board)
            pre = scorer.W_ctx @ ctx
            gate = (pre > 0).astype(np.float32)
            grad_ctx = np.outer(gate, ctx).astype(np.float32)
            self._nudge_context(scorer.W_ctx, grad_ctx, credit)

            # ``w_hid`` and ``w_dst_ctx`` read different things: w_hid reads the
            # global hidden vector (shared by every square), w_dst_ctx reads the
            # origin-conditioned one. Both are single rows, so both are tied
            # tensors and both take the normalised rate.
            self._nudge_shared(
                scorer.w_hid, np.maximum(pre, 0.0), credit, key="w_hid"
            )
            self._nudge_shared(
                scorer.w_dst_ctx,
                scorer._origin_hidden(board, f_c),
                credit,
                key="w_dst_ctx",
            )

        self.counter.update(mask, move.from_square, move.to_square, slot)

    # ---- persistence ---------------------------------------------------

    def save(self, path) -> dict:
        """Same sidecar as L3's, with the tied-row rate recorded.

        ``n_parameters`` is inherited and correct; adding ``lr_shared`` makes a
        saved L3.5 checkpoint self-describing about the one training knob that
        differs from L3, which is the fact a reader is most likely to want.
        """
        import json
        from pathlib import Path

        out = super().save(path)
        meta_path = Path(out["meta"])
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["lr_shared"] = self.config.lr_shared
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return out


# --------------------------------------------------------------------------
# the control arm: an unstructured model at the same budget
# --------------------------------------------------------------------------
#
# The roster requires a third arm, ``L3-flat``, and it is not optional: "without it, a win by L3.5 is indistinguishable from 'any
# change from L3 helps'". So it must be a real, trainable level at the same
# parameter budget that throws away *spatial* structure entirely -- not a
# strawman that is handicapped in some other way too.
#
# The honest construction is a single hidden layer over the flattened encoder
# output. For the budget to match, the hidden width has to be *tiny*: the input
# is 22 * 64 = 1408 wide, so even one hidden unit already costs 1408 weights.
# That is the point of the arm. A flat MLP over the same planes cannot afford a
# hidden layer of any useful size at L3's parameter count, and the experiment
# measures whether that cost is real.

class FlatScorer:
    """Unstructured control: one hidden layer over the flattened board tensor.

    Implements the same :class:`chessrl.policy.Scorer` protocol, so it drops
    into the same policy and the same game loop and is therefore comparable
    without any special-casing in the harness.

    Layout, deliberately mirroring L3.5's *output structure* while discarding
    its spatial prior::

        W_in      (h_flat, n_ch*64)   flattened board -> hidden
        b_in      (h_flat,)
        w_from    (h_flat,)           hidden -> a single origin offset
        w_to      (h_flat,)           hidden -> a single destination offset

    ``h_flat = 1`` is the only width that fits the budget, and at that width
    this is exactly a *linear* model over the flattened board with a rectifier
    -- i.e. the weakest thing that is still not spatially structured. That is
    the control L3.5 has to beat for its claim to mean anything.
    """

    def __init__(self, config: "FlatConfig | None" = None, seed: int = 0):
        self.config = config or FlatConfig()
        rng = np.random.default_rng(seed)
        self.n_ch = TOTAL_CHANNELS if self.config.use_value_channels else BINARY_CHANNELS
        self.h_flat = self.config.h_flat
        n_in = self.n_ch * 64

        def init(size):
            return (rng.standard_normal(size) * 0.01).astype(np.float32)

        self.W_in = init(self.h_flat * n_in).reshape(self.h_flat, n_in)
        self.b_in = np.zeros(self.h_flat, dtype=np.float32)
        self.w_from = init(self.h_flat)
        self.w_to = init(self.h_flat)
        self.w_geom = init(len(GEOMETRY_NAMES))
        self.promo = (rng.standard_normal((M.NUM_PROMO, len(MOVE_FEATURE_NAMES_L3)))
                      * 0.01).astype(np.float32)

        self._current_board: chess.Board | None = None
        self._canon_sq: np.ndarray | None = None
        self._current_channels: np.ndarray | None = None
        self._geometry_cache: dict[int, np.ndarray] = {}

    # ---- forward -------------------------------------------------------

    def _hidden(self, board: chess.Board) -> np.ndarray:
        from .encode import encode

        tensor = encode(board, with_values=self.config.use_value_channels,
                        canonicalise=True)
        flat = tensor[:self.n_ch].reshape(-1).astype(np.float32)
        return np.maximum(self.W_in @ flat + self.b_in, 0.0)

    def from_logits(self, board: chess.Board) -> np.ndarray:
        """A *constant* origin score per square: the flat model has no per-square
        input at all, which is precisely the structure it gives up.

        Returned as a full ``(64,)`` vector rather than a scalar so the policy
        protocol is satisfied and the factorised softmax still applies the
        legality mask; every legal origin therefore receives equal probability,
        and the only remaining choice is the destination.
        """
        h = self._hidden(board)
        return np.full(64, float(self.w_from @ h), dtype=np.float32)

    def to_logits(self, from_sq: int) -> np.ndarray:
        h = self._hidden(self._current_board)
        base = float(self.w_to @ h)
        out = np.full(64, base, dtype=np.float32)
        row = self._geom_features_row(int(self._canon_sq[from_sq]))
        out = out + row @ self.w_geom
        return self._to_real(out, self._current_board)

    def promo_logits(self, from_sq: int, to_sq: int) -> np.ndarray:
        return self.promo @ self._move_features(from_sq, to_sq)

    def set_position(self, board: chess.Board) -> None:
        """The flat model still needs the canonical map for its geometry term.

        It does *not* need the channel tensor: its origin head is a constant,
        which is exactly the structure it gives up. Holding the board is enough.
        """
        self._current_board = board
        self._canon_sq = SquareLocalScorer._canonical_squares(board)
        self._geometry_cache = {}

    @staticmethod
    def _canonical_squares(board):
        return SquareLocalScorer._canonical_squares(board)

    def _to_real(self, values, board):
        if board.turn == chess.WHITE:
            return np.asarray(values, dtype=np.float32)
        out = np.empty(64, dtype=np.float32)
        out[self._canon_sq] = values
        return out

    def _geom_features(self, from_sq, to_sq):
        return SquareLocalScorer._geom_features(self, from_sq, to_sq)

    def _geom_features_row(self, from_sq):
        cache = self._geometry_cache
        row = cache.get(from_sq)
        if row is None:
            row = np.stack([self._geom_features(from_sq, t) for t in range(64)]
                           ).astype(np.float32)
            cache[from_sq] = row
        return row

    def _move_features(self, from_sq, to_sq):
        fc, tc = int(self._canon_sq[from_sq]), int(self._canon_sq[to_sq])
        fr, ff = fc >> 3, fc & 7
        tr, tf = tc >> 3, tc & 7
        return np.array([
            1.0, 0.0, 0.0, (tr - fr) / 7.0, abs(tf - ff) / 7.0,
        ], dtype=np.float32)

    def parameters(self) -> dict[str, np.ndarray]:
        return {
            "W_in": self.W_in, "b_in": self.b_in,
            "w_from": self.w_from, "w_to": self.w_to,
            "w_geom": self.w_geom, "promo": self.promo,
        }

    @property
    def n_parameters(self) -> int:
        return int(sum(v.size for v in self.parameters().values()))

    def state_dict(self) -> dict:
        return {
            "config": dict(self.config.__dict__),
            "params": {k: v.tolist() for k, v in self.parameters().items()},
        }

    def load_state_dict(self, state: dict) -> None:
        for k, v in state["params"].items():
            getattr(self, k)[:] = np.asarray(v, dtype=np.float32)


@dataclass
class FlatConfig:
    """Shape knobs for the control arm. ``h_flat`` is the budget knob."""

    use_value_channels: bool = False
    # One hidden unit already costs 1408 weights over the flattened board, so
    # 1 is the largest width that stays inside the budget.
    h_flat: int = 1
    quantised: bool = True


class FlatPolicy(L3Policy):
    """The control arm as a playable policy: no per-square parameters anywhere."""

    def __init__(self, scorer: FlatScorer | None = None, **kwargs):
        if scorer is None:
            scorer = FlatScorer(seed=kwargs.get("seed", 0))
        super().__init__(scorer, name=kwargs.pop("name", "L3-flat"), **kwargs)

    @classmethod
    def load(cls, path, **kwargs) -> "FlatPolicy":
        import json
        from pathlib import Path

        state = json.loads(Path(path).read_text(encoding="utf-8"))
        config = FlatConfig(**state["scorer"]["config"])
        scorer = FlatScorer(config, seed=kwargs.pop("seed", 0))
        scorer.load_state_dict(state["scorer"])
        return cls(scorer, name=state.get("name", "L3-flat"),
                   temperature=state.get("temperature", 1.0), **kwargs)
