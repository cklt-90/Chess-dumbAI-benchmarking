"""L5: alpha-beta whose ordering and leaf evaluation come from L4.

What this level adds
--------------------
L2 searches well with no parameters. L4 has parameters but no search. L5 is the
first level with both, and the interesting part is *how* they combine.

The spec asks for a decision model and an evaluation model that "only deviate by
the last layer". Read literally as two separate networks trained from scratch,
that instruction is just two models. Read as an architecture it is much
better: **one shared trunk with two heads**. The trunk learns a representation
of the position; the *decision* head turns it into a move distribution and the
*evaluation* head turns it into a centipawn value. They differ only in their
final layer because they are the same features read two ways -- which is also
what makes them cheaper to train than the sum of their parts.

Where the two heads plug into L2
--------------------------------
:class:`~chessrl.search.MinimaxEngine` has exactly two places a learned model can
act, and L5 uses both:

* **ordering** -- the decision head's move probabilities seed the move ordering.
  This is the same seam :class:`~chessrl.search.InformedMinimaxPolicy` uses, but
  the prior is now learned rather than L1's blind table.
* **leaf evaluation** -- the evaluation head replaces :func:`chessrl.value.evaluate`
  at the horizon.

The important structural claim, and the one the tests pin down: **ordering can
only change which move is returned among equals, never the search's value**,
whereas **leaf evaluation changes the value itself**. So a bad ordering costs
time and a bad evaluator costs correctness. That asymmetry is why the eval head
is trained against game outcomes while the decision head is trained against
search targets, and why L5 is measured on both axes separately.

The horizon is where a learned evaluator earns its keep
-------------------------------------------------------
At depth 5 the engine sees five plies and then asks a static function whether
the position is good. `value.evaluate` answers with material and piece-square
tables: it cannot see that a piece is hanging, because being en prise is not a
property of a square. The eval head can, in principle, learn it. That is the
honest version of the claim L5 makes -- it does not make the search deeper, it
makes the *leaf* less wrong.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path

import chess
import numpy as np

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "chessrl.guided is L5 and needs PyTorch, an optional extra. Install:\n"
        "    ./.venv/Scripts/python.exe -m pip install torch "
        "--index-url https://download.pytorch.org/whl/cpu"
    ) from exc

from . import masks as M
from .encode import (
    BINARY_CHANNELS,
    TOTAL_CHANNELS,
    board_context_features,
    encode,
)
from .perceptron import GEOMETRY_NAMES, MOVE_FEATURE_NAMES_L3, PerceptronScorer
from .search import QUIESCE_DEPTH, INF, MinimaxEngine, order_moves, score_move
from .torch_model import TorchConfig, TorchScorer, torch_masked_softmax


@dataclass
class GuidedConfig:
    """Shape of the shared trunk and the two heads."""

    use_value_channels: bool = False
    # Trunk width. The shared representation, before the two heads read it.
    trunk: int = 64
    # Depth of the trunk's hidden stack. One layer is enough to mix the
    # per-square channels into a position-level summary; more overfits a corpus
    # this size.
    trunk_layers: int = 1
    value_scale: float = 400.0
    seed: int = 0


class GuidedModel(nn.Module):
    """A shared trunk with a decision head and an evaluation head.

    Architecture, and why each piece is shaped as it is:

    **Trunk** -- the encoded planes ``(n_ch, 64)`` are flattened per square and
    projected to ``trunk`` units, with the global context vector concatenated.
    This is a position *summary*: one vector describing the whole board. Both
    heads read this same vector, which is the "deviate only by the last layer"
    instruction taken seriously.

    **Decision head** -- a *factorised* head, not a flat one. A flat head over
    20480 actions would need 1.3M parameters for a 64-wide trunk, and it would
    also have to learn from scratch that a move is a (from, to, promo) triple.
    Instead the head reuses L4's trick: origin logits from the trunk's
    per-square rows, destination logits from the trunk plus geometry. So the
    "last layer" of the decision head is the same three small projections L3 and
    L4 use, and the trunk is what is shared.

    **Evaluation head** -- one scalar per position, in centipawns after scaling.
    Deliberately a single linear read of the trunk: values are a much lower
    bandwidth signal than a move distribution and a deep value head just
    overfits.

    The heads genuinely differ only in the final projection. Everything before
    that point is literally the same tensor.
    """

    def __init__(self, config: GuidedConfig | None = None):
        super().__init__()
        self.config = config or GuidedConfig()
        gen = torch.Generator().manual_seed(self.config.seed)
        self.n_ch = (
            TOTAL_CHANNELS if self.config.use_value_channels else BINARY_CHANNELS
        )
        self.n_ctx = 9

        def init(*shape) -> nn.Parameter:
            return nn.Parameter(
                torch.randn(*shape, generator=gen, dtype=torch.float32) * 0.02
            )

        # ---- shared trunk ------------------------------------------------
        # Per-square projection: each square's channel vector -> trunk units.
        self.trunk_sq = init(64, self.n_ch, self.config.trunk)
        # Context projection, added to every square's trunk row.
        self.trunk_ctx = init(self.n_ctx, self.config.trunk)
        self.trunk_bias = nn.Parameter(torch.zeros(64, self.config.trunk))

        # ---- decision head: factorised, mirroring L4 ---------------------
        self.dec_from = init(self.config.trunk, 1)
        self.dec_promo = init(len(MOVE_FEATURE_NAMES_L3), M.NUM_PROMO)
        # Destination scoring reads the trunk rows and the geometry.
        self.dec_dst = init(self.config.trunk, 1)
        self.dec_src = init(self.config.trunk, 1)
        self.dec_geom = init(len(GEOMETRY_NAMES), 1)

        # ---- evaluation head: one linear read of the trunk ---------------
        self.val_head = init(self.config.trunk, 1)
        self.val_ctx = init(self.n_ctx, 1)

        # Geometry features, precomputed and shared with L3/L4 so the three
        # levels cannot drift on what "diagonal" means.
        scorer = PerceptronScorer.__new__(PerceptronScorer)
        table = np.stack([
            np.stack([PerceptronScorer._geom_features(scorer, f, t)
                      for t in range(64)])
            for f in range(64)
        ]).astype(np.float32)
        self.register_buffer(
            "geom_table", torch.as_tensor(table), persistent=False
        )

    # ---- the trunk -----------------------------------------------------

    def trunk_features(
        self, chan: torch.Tensor, ctx: torch.Tensor
    ) -> torch.Tensor:
        """``(B, 64, trunk)`` -- the shared representation both heads read.

        ``chan`` is ``(B, n_ch, 64)``. The per-square projection is a batched
        einsum, the context term is broadcast to all 64 squares, and the bias is
        per-square. This is the *only* place features are computed; the two
        heads cannot diverge before their final projections because there is
        nothing to diverge into.

        The projection is written as broadcast-multiply-then-sum, and the two
        tempting alternatives were both measured and both rejected:

        * ``bmm`` on a ``(64, B, n_ch) @ (64, n_ch, trunk)`` view is the fastest
          form when the batch is large -- 53us/board at B=64 against 262 for
          this spelling -- but at B=1, which is the *search's* regime, it costs
          1001us against 252. The two callers want opposite things, so the form
          that is merely decent at both was kept rather than one that is
          excellent for training and disastrous for play.
        * ``einsum('bcs,sck->bsk')`` is the natural spelling and is the slowest
          of the three at every size measured, because its contraction-path
          machinery dwarfs the 90k-element product.

        The real search-side win was not this line at all: it was the batch
        size (see :meth:`single_threaded`) and deferring ``encode`` behind the
        value cache. A previous "optimisation" that swapped this for ``bmm``
        made the depth-3 middlegame 40% *slower*, which is why the profile
        numbers are recorded here rather than the intent.
        """
        per_square = (
            chan.permute(0, 2, 1).unsqueeze(-1) * self.trunk_sq.unsqueeze(0)
        ).sum(2)
        context = (ctx @ self.trunk_ctx).unsqueeze(1)      # (B, 1, trunk)
        return per_square + context + self.trunk_bias

    # ---- the two heads -------------------------------------------------

    def decision_logits(
        self, trunk: torch.Tensor, chan: torch.Tensor
    ) -> torch.Tensor:
        """``(B, 64)`` origin logits, the decision head's only output.

        One weight vector applied at every square, so the contraction is over
        the trunk axis with ``dec_from`` shaped ``(trunk, 1)`` and squeezed to
        ``(trunk,)``. Writing this as ``dec_from.unsqueeze(0)`` instead is a
        latent trap: the unsqueezed singleton lands on the *trunk* axis and the
        trailing 1 on the square axis, which broadcasts only when the trunk
        width happens to equal 64 -- the square count. It did not crash at the
        default width, it just contracted the wrong axis and returned different
        numbers from the equivalent explicit form. With ``trunk=32`` it raises.
        """
        return (trunk * self.dec_from.squeeze(-1)).sum(-1)

    def destination_logits(
        self, trunk: torch.Tensor, chan: torch.Tensor, from_sq: torch.Tensor
    ) -> torch.Tensor:
        """``(B, 64)`` destination logits from the same trunk rows.

        Three additive terms, matching L3/L4's destination head exactly in
        structure: the destination's own trunk row, the origin's trunk row
        broadcast to all destinations, and the geometry between them.
        """
        if trunk.shape[0] == 1 and from_sq.shape[0] != 1:
            trunk = trunk.expand(from_sq.shape[0], -1, -1)
        out = (trunk * self.dec_dst.squeeze(-1)).sum(-1)  # (B, 64) over squares
        # Gather each batch element's origin row: the result must be
        # (B, trunk) -- the *origin's* trunk vector, one per row of the batch.
        #
        # ``trunk`` is (B, 64 squares, trunk) and ``gather`` indexes axis 1, so
        # the index tensor has to be (B, 1, trunk) and filled with the origin
        # square. Written as ``(B, 64, 1)`` instead -- which is what an
        # unsqueeze in the wrong place produces -- the gather returns
        # (B, 64 squares, trunk) and squeezing the last axis leaves the *square*
        # axis, turning this term into a sum over all 64 destinations' trunk
        # rows. That is not the quantity the head is supposed to add, and it
        # only stayed shape-legal while the trunk width happened to equal 64.
        idx = from_sq.reshape(-1, 1, 1).expand(-1, 1, trunk.shape[2])
        src = trunk.gather(1, idx).squeeze(1)              # (B, trunk)
        # One scalar per batch element, reshaped to (B, 1) so it broadcasts as a
        # per-origin offset rather than pairing up with the square axis.
        out = out + (src * self.dec_src.squeeze(-1)).sum(-1).reshape(-1, 1)
        geom = self.geom_table[from_sq]                    # (B, 64, n_geom)
        out = out + (geom * self.dec_geom.squeeze(-1)).sum(-1)
        return out

    def promo_logits(
        self, chan: torch.Tensor, from_sq: torch.Tensor, to_sq: torch.Tensor
    ) -> torch.Tensor:
        feats = TorchScorer.move_features(chan, from_sq, to_sq)
        return feats @ self.dec_promo

    def evaluate(self, trunk: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        """``(B,)`` centipawn value from the point of view of the side to move."""
        raw = (trunk.mean(dim=1) * self.val_head.squeeze(-1)).sum(-1)
        raw = raw + (ctx @ self.val_ctx).squeeze(-1)
        return torch.tanh(raw) * self.config.value_scale

    def n_parameters(self) -> int:
        return int(sum(p.numel() for p in self.parameters() if p.requires_grad))


def _trunk_inputs(board: chess.Board, n_ch: int, *, mobility: int | None = None):
    """Encode one board into the ``(1, n_ch, 64)`` / ``(1, n_ctx)`` pair.

    ``mobility`` is threaded through to :func:`board_context_features` so a
    caller that already knows the legal-move count -- the search, always --
    does not pay to regenerate moves inside the leaf evaluator.
    """
    tensor = encode(board, with_values=(n_ch > BINARY_CHANNELS), canonicalise=True)
    chan = torch.as_tensor(
        tensor[:n_ch].reshape(n_ch, 64), dtype=torch.float32
    ).unsqueeze(0)
    ctx = torch.as_tensor(
        board_context_features(board, mobility=mobility), dtype=torch.float32
    ).unsqueeze(0)
    return chan, ctx


class GuidedEngine(MinimaxEngine):
    """L2's search with a learned ordering prior and a learned leaf evaluator.

    Subclasses :class:`MinimaxEngine` and overrides exactly two things, because
    those are the only two seams that exist. Everything else -- alpha-beta, the
    transposition table, iterative deepening, quiescence -- is L2's code
    unchanged, which keeps the comparison between the levels honest: if L5 is
    stronger, it is because of the model, not because the search was rewritten.

    ``use_model_eval`` defaults to True and ``use_model_prior`` to True; setting
    either to False gives the ablation, which is how the two contributions can
    be measured separately rather than claimed together.
    """

    def __init__(
        self,
        model: GuidedModel,
        *,
        depth: int = 5,
        use_model_eval: bool = True,
        use_model_prior: bool = True,
        prior_weight: float = 50.0,
        sync_every: int = 256,
        **kwargs,
    ):
        super().__init__(depth=depth, **kwargs)
        self.model = model
        self.use_model_eval = use_model_eval
        self.use_model_prior = use_model_prior
        self.prior_weight = prior_weight if use_model_prior else 0.0
        self.name = "L5-guided"
        # The engine runs entirely on numpy boards, so the torch forward pass is
        # the expensive part. The value cache is the single biggest win: interior
        # nodes repeat heavily and each repeated eval is a full tensor pass.
        self._value_cache: dict[int, int] = {}
        self._prior_cache: dict[int, dict[chess.Move, float]] = {}
        self.sync_every = sync_every
        self.leaf_evals = 0
        self.prior_lookups = 0

    # ---- leaf evaluation -----------------------------------------------

    def model_evaluate(self, board: chess.Board, *, mobility: int | None = None) -> int:
        """Centipawns for the side to move, from the evaluation head.

        Cached on the zobrist hash: at depth 5 the same leaf position is reached
        by many move orders, and a torch forward pass per visit dominated the
        whole search before this cache existed.

        ``mobility`` is optional and only saves work: when the caller already
        knows how many legal moves exist, the context encoder can skip
        regenerating them.
        """
        key = chess.polyglot.zobrist_hash(board)
        hit = self._value_cache.get(key)
        if hit is not None:
            return hit

        chan, ctx = _trunk_inputs(board, self.model.n_ch, mobility=mobility)
        with torch.no_grad():
            trunk = self.model.trunk_features(chan, ctx)
            raw = float(self.model.evaluate(trunk, ctx)[0])
        # Clamp away from the mate band so a learned value can never masquerade
        # as a forced win and make the search prefer it over a real mate.
        value = int(np.clip(raw, -self._mate_floor, self._mate_floor))
        if len(self._value_cache) > 200_000:
            self._value_cache.clear()
        self._value_cache[key] = value
        self.leaf_evals += 1
        return value

    _mate_floor = 9000

    def _negamax(self, board: chess.Board, depth: int, alpha: int, beta: int) -> int:
        """L2's negamax with the leaf swap.

        The override is deliberately narrow: terminal detection, the TT, move
        ordering and the alpha-beta loop are all still L2's. Only the two
        ``V.evaluate`` call sites change, which is what makes the level claim
        "the search is L2's, the knowledge is learned" checkable.
        """
        if not self.use_model_eval:
            return super()._negamax(board, depth, alpha, beta)

        self.nodes += 1
        if board.is_checkmate():
            return -100_000 + board.ply()
        if (board.is_stalemate() or board.is_insufficient_material()
                or board.is_seventyfive_moves()
                or board.is_fivefold_repetition()):
            return 0

        if depth <= 0:
            if self.use_quiescence:
                return self._quiescence(board, alpha, beta)
            return self.model_evaluate(board)

        key = chess.polyglot.zobrist_hash(board)
        tt_move = None
        if self.use_tt:
            entry = self.tt.get(key)
            if entry is not None:
                e_depth, e_value, e_flag, e_move = entry
                tt_move = e_move
                if e_depth >= depth:
                    if e_flag == 0:
                        self.tt_hits += 1
                        return e_value
                    if e_flag == 1 and e_value > alpha:
                        alpha = e_value
                    elif e_flag == 2 and e_value < beta:
                        beta = e_value
                    if alpha >= beta:
                        self.tt_hits += 1
                        return e_value

        legal = list(board.legal_moves)
        if not legal:
            return 0

        prior = self._model_prior(board) if self.use_model_prior else None
        ordered = order_moves(
            board, legal, tt_move=tt_move, prior=prior,
            prior_weight=self.prior_weight,
        )
        original_alpha = alpha
        best_value = -INF
        best_move = ordered[0]

        for move in ordered:
            board.push(move)
            value = -self._negamax(board, depth - 1, -beta, -alpha)
            board.pop()
            if value > best_value:
                best_value, best_move = value, move
            if value > alpha:
                alpha = value
            if alpha >= beta:
                self.cutoffs += 1
                break

        if self.use_tt:
            if best_value <= original_alpha:
                flag = 2
            elif best_value >= beta:
                flag = 1
            else:
                flag = 0
            self.tt[key] = (depth, best_value, flag, best_move)
        return best_value

    def _quiescence(self, board, alpha, beta, qdepth=0):
        """L2's quiescence with the model as the stand-pat value.

        Overridden rather than inherited because ``super()._quiescence`` calls
        ``V.evaluate`` for stand-pat; leaving that in place would mean captures
        are judged by the static evaluator while quiet positions are judged by
        the model, and the two would disagree at exactly the moment the search
        is trying to be precise.
        """
        if not self.use_model_eval:
            return super()._quiescence(board, alpha, beta, qdepth)

        self.nodes += 1
        stand_pat = self.model_evaluate(board)
        if stand_pat >= beta:
            return beta
        if alpha < stand_pat:
            alpha = stand_pat
        if qdepth >= QUIESCE_DEPTH:
            return alpha

        captures = [m for m in board.legal_moves if board.is_capture(m)]
        ordered = sorted(captures, key=lambda m: score_move(board, m), reverse=True)
        for move in ordered:
            board.push(move)
            value = -self._quiescence(board, -beta, -alpha, qdepth + 1)
            board.pop()
            if value >= beta:
                return beta
            if value > alpha:
                alpha = value
        return alpha

    # ---- ordering prior -------------------------------------------------

    def _model_prior(self, board: chess.Board) -> dict[chess.Move, float]:
        """Decision-head probabilities for this position's legal moves.

        Cached by position: the same interior node is reordered on every
        iterative-deepening pass and every transposition, and the forward pass
        is far more expensive than the ordering itself.

        All origins are scored in **one batched call**. Looping a forward pass
        per move is the obvious way to write this and it cost ~3s per root
        position against ~0.2s batched, because per-call Python and dispatch
        overhead dwarfed the arithmetic at this size.
        """
        key = chess.polyglot.zobrist_hash(board)
        hit = self._prior_cache.get(key)
        if hit is not None:
            return hit

        # ``board.legal_moves`` is regenerated on every iteration because it is a
        # lazy generator, and this function asks for it three times. Materialise
        # once: the move list is needed anyway, and its length is exactly the
        # mobility the context encoder wants.
        legal = list(board.legal_moves)
        chan, ctx = _trunk_inputs(board, self.model.n_ch, mobility=len(legal))
        mask = M.legal_move_mask(board)
        canon = PerceptronScorer._canonical_squares(board)
        origins = sorted({move.from_square for move in legal})
        canon_origins = [int(canon[f]) for f in origins]

        # The trunk is canonical, so *everything* the heads produce lives on
        # canonical squares -- and the masks must therefore be built there too.
        # Reading a real-coordinate mask against canonical logits is the bug
        # this position went through: for White the two frames coincide and it
        # is invisible; for Black the mask admits real squares while the logits
        # are nonzero at reflected ones, so the prior came out all zeros.
        #
        # The move mask is permuted into canonical coordinates by reindexing
        # both square axes through the same map. Promotions are unaffected --
        # `canon` acts on squares, not on the promotion slot.
        canon_index = np.asarray(canon, dtype=np.int64)
        canon_mask = mask[canon_index][:, canon_index]

        with torch.no_grad():
            trunk = self.model.trunk_features(chan, ctx)
            p_from = torch_masked_softmax(
                self.model.decision_logits(trunk, chan)[0],
                M.legal_from_mask(canon_mask),
            )
            # Destinations are read at canonical origins and canonical
            # destinations, so this stays in one frame throughout.
            canon_origin_t = torch.tensor(canon_origins, dtype=torch.long)
            to_logits = self.model.destination_logits(trunk, chan, canon_origin_t)
            per_origin = {
                f_c: torch_masked_softmax(
                    to_logits[i], canon_mask[f_c].any(axis=1)
                )
                for i, f_c in enumerate(canon_origins)
            }
            out = {
                move: float(
                    p_from[int(canon[move.from_square])]
                    * per_origin[int(canon[move.from_square])][
                        int(canon[move.to_square])
                    ]
                )
                for move in legal
            }

        if len(self._prior_cache) > 50_000:
            self._prior_cache.clear()
        self._prior_cache[key] = out
        self.prior_lookups += 1
        return out

    def clear_caches(self) -> None:
        """Drop the value and prior caches and the transposition table."""
        self._value_cache.clear()
        self._prior_cache.clear()
        self.tt.clear()

    def _search_value(self, board: chess.Board, depth: int) -> int:
        """The value the search would assign to ``board`` at ``depth``.

        A test seam, and the only reason it exists is that the level's central
        claim cannot be checked through ``select``: two searches can return the
        same *move* while disagreeing about the *value*, and two searches can
        return different moves while agreeing about the value. Asserting the
        value invariant needs the value, so it is exposed rather than fished
        out of a returned tuple.

        Unlike :meth:`GuidedPolicy.select` this does not clear the caches or
        deepen iteratively, and it does not pin the thread count -- it is meant
        to be called in pairs where both sides should share whatever state has
        accumulated, so that a passing test is about the algorithm and not
        about cache warmth.
        """
        alpha = -INF
        best = -INF
        for move in order_moves(
            board, list(board.legal_moves),
            prior=self._model_prior(board) if self.use_model_prior else None,
            prior_weight=self.prior_weight,
        ):
            board.push(move)
            value = -self._negamax(board, depth - 1, -INF, -alpha)
            board.pop()
            if value > best:
                best = value
            if value > alpha:
                alpha = value
        return best

    @contextlib.contextmanager
    def single_threaded(self):
        """Run the enclosed block with one torch thread, then restore.

        PyTorch defaults to one OpenMP thread per core. For the tensors this
        engine produces -- a single position against a ``(64, n_ch, trunk)``
        weight, so a few tens of thousands of multiply-adds -- the fork/join
        cost of a parallel region exceeds the arithmetic it parallelises.
        Measured on the 8-core box this was written on, one leaf's trunk
        projection takes 785us at the default 8 threads and 471us at 1; across
        the ~900 leaves of a depth-3 search that is seconds, not milliseconds.

        Applied around the search rather than in ``__init__`` because the
        thread count is process-global: setting it in a constructor would
        silently slow down a trainer built in the same process, which does
        benefit from the parallelism.
        """
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            yield
        finally:
            torch.set_num_threads(previous)


class GuidedPolicy:
    """A ``MovePolicy``: search-based, so it exposes no distribution.

    Deliberately *not* a ``DistributionPolicy``. L5 makes decisions by searching;
    asking it for a probability over all 20480 actions would mean either
    enumerating a search per move -- which is not what the level is -- or
    inventing a distribution it does not actually use. L1/L3/L4 have
    distributions because that is their decision rule. Keeping the two families
    separate is what lets the L6 ensemble know which levels can be blended and
    which can only be voted.
    """

    def __init__(self, model: GuidedModel | None = None, *, depth: int = 5, **kwargs):
        self.model = model or GuidedModel()
        self.engine = GuidedEngine(self.model, depth=depth, **kwargs)
        self.name = "L5-guided-policy"

    @property
    def depth(self) -> int:
        return self.engine.depth

    def select(self, board: chess.Board) -> chess.Move:
        self.engine.clear_caches()
        moves = list(board.legal_moves)
        if not moves:
            raise ValueError(f"no legal moves in {board.fen()}")
        if len(moves) == 1:
            return moves[0]

        # Iterative deepening, exactly as L2 does it, so the TT is warm for the
        # final pass. Reimplemented rather than inherited because
        # ``MinimaxEngine.select`` calls its own ``_root``, which uses
        # ``V.evaluate``-based ordering for the first pass.
        best = moves[0]
        with self.engine.single_threaded():
            for d in range(1, self.engine.depth + 1):
                best = self._root(board, moves, d)
        return best

    def _root(self, board, moves, depth):
        prior = self.engine._model_prior(board) if self.engine.use_model_prior else None
        tt_entry = self.engine.tt.get(chess.polyglot.zobrist_hash(board))
        tt_move = tt_entry[3] if tt_entry else None
        ordered = order_moves(
            board, moves, tt_move=tt_move, prior=prior,
            prior_weight=self.engine.prior_weight,
        )
        alpha = -INF
        best_move, best_value = ordered[0], -INF
        for move in ordered:
            board.push(move)
            value = -self.engine._negamax(board, depth - 1, -INF, -alpha)
            board.pop()
            if value > best_value:
                best_value, best_move = value, move
            if value > alpha:
                alpha = value
        self.engine.tt[chess.polyglot.zobrist_hash(board)] = (
            depth, best_value, 0, best_move
        )
        return best_move


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------

@dataclass
class GuidedTrainConfig:
    """Training knobs for the two heads.

    The heads are trained on different signals on purpose, and that is the
    point rather than an implementation detail: the decision head imitates a
    search (dense, informative, directly about moves), while the evaluation head
    is fitted to game outcomes (sparse, noisy, but the only source of ground
    truth about who actually won). Training them from one blended loss would
    blur two very different noise levels.
    """

    decision_lr: float = 0.02
    value_lr: float = 0.01
    top_k: int = 5
    search_depth: int = 3
    credit_decay: float = 0.99
    value_weight: float = 1.0


class GuidedTrainer:
    """Trains the trunk through both heads."""

    def __init__(self, model: GuidedModel | None = None,
                 config: GuidedTrainConfig | None = None):
        if isinstance(model, GuidedTrainConfig) and config is None:
            model, config = None, model
        self.config = config or GuidedTrainConfig()
        self.model = model or GuidedModel()
        params = list(self.model.parameters())
        self.decision_opt = torch.optim.Adam(params, lr=self.config.decision_lr)
        self.value_opt = torch.optim.Adam(params, lr=self.config.value_lr)
        self.decision_steps = 0
        self.value_steps = 0

    def train_on_search_feedback(self, board: chess.Board, *,
                                 depth: int | None = None) -> dict:
        """Imitate a search with the decision head.

        Same objective as L4's: the search scores every legal move, the top-k
        become a soft target, and the head is pulled towards them by cross
        entropy. The difference is that the trunk is shared, so this also moves
        the features the *evaluation* head reads.
        """
        depth = depth if depth is not None else self.config.search_depth
        legal = list(board.legal_moves)
        if not legal:
            return {"applied": False, "k": 0}

        engine = MinimaxEngine(depth=depth, use_tt=True)
        scored = []
        for move in legal:
            board.push(move)
            value = -engine._negamax(board, depth - 1, -10**9, 10**9)
            board.pop()
            scored.append((move, int(value)))
        scored.sort(key=lambda kv: kv[1], reverse=True)
        top = scored[: self.config.top_k]
        best = top[0][1]

        legal = list(board.legal_moves)
        chan, ctx = _trunk_inputs(board, self.model.n_ch, mobility=len(legal))
        canon = PerceptronScorer._canonical_squares(board)
        # Same canonical-frame discipline as ``GuidedEngine._model_prior``: the
        # trunk is canonical, so the masks have to be too, and the probabilities
        # are read at canonical squares. Mixing the frames loses the gradient on
        # every Black-to-move position -- the loss still falls, because the
        # clamp floors it, but it stops being a function of the head's output.
        canon_index = np.asarray(canon, dtype=np.int64)
        canon_mask = M.legal_move_mask(board)[canon_index][:, canon_index]

        trunk = self.model.trunk_features(chan, ctx)
        p_from = torch_masked_softmax(
            self.model.decision_logits(trunk, chan)[0],
            M.legal_from_mask(canon_mask),
        )

        weights, losses = [], []
        for move, value in top:
            f = int(canon[move.from_square])
            t = int(canon[move.to_square])
            idx = torch.tensor([f], dtype=torch.long)
            p_to = torch_masked_softmax(
                self.model.destination_logits(trunk, chan, idx)[0],
                canon_mask[f].any(axis=1),
            )
            weight = float(np.exp((value - best) / 300.0))
            p_move = (p_from[f] * p_to[t]).clamp(1e-9, 1.0)
            weights.append(weight)
            losses.append(-weight * torch.log(p_move))

        loss = torch.stack(losses).sum() / (sum(weights) or 1.0)
        self.decision_opt.zero_grad()
        loss.backward()
        self.decision_opt.step()
        self.decision_steps += 1
        return {"applied": True, "k": len(top), "loss": float(loss.item())}

    def train_on_game(self, result, *, eval_batch: int = 64) -> dict:
        """Fit the evaluation head to the game's outcome.

        Every ply is a training example whose target is (a decayed form of) the
        final result from the side to move's point of view. The decay matters
        for the same reason it does in L1/L3: the position on move 3 did not
        cause a loss on move 60, and treating it as equally responsible is how a
        value head learns to fear the opening.

        Unfinished games are skipped rather than scored as draws -- the same
        convention as everywhere else in this repo, and for the same reason.
        """
        if not result.fens or not result.is_finished:
            return {"trained": 0, "skipped": True}
        if not result.is_decisive:
            # A draw is genuine information, but a *small* amount of it; the
            # sign is ambiguous and the magnitude near zero.
            white_score = 0.0
        else:
            white_score = 1.0 if result.result == "1-0" else -1.0

        plies = len(result.moves)
        if plies == 0:
            return {"trained": 0, "skipped": True}

        # Newest positions get the most credit; the opening barely any.
        decay = self.config.credit_decay
        indices = list(range(plies))
        weights = np.power(decay, np.arange(plies - 1, -1, -1)).astype(np.float32)
        # Subsample to keep one training step bounded: a 200-ply game is 200
        # forward passes otherwise, and the late positions carry the signal.
        if len(indices) > eval_batch:
            keep = np.argsort(weights)[::-1][:eval_batch]
            indices = sorted(keep.tolist())
        indices = [i for i in indices if i < len(result.fens)]

        chans, ctxs, targets, wts = [], [], [], []
        for i in indices:
            board = chess.Board(result.fens[i])
            if board.is_game_over():
                continue
            chan, ctx = _trunk_inputs(board, self.model.n_ch)
            sign = white_score if board.turn == chess.WHITE else -white_score
            chans.append(chan[0])
            ctxs.append(ctx[0])
            targets.append(sign)
            wts.append(float(weights[i]))
        if not chans:
            return {"trained": 0, "skipped": True}

        chan_b = torch.stack(chans)
        ctx_b = torch.stack(ctxs)
        target_b = torch.as_tensor(targets, dtype=torch.float32)
        weight_b = torch.as_tensor(wts, dtype=torch.float32)
        weight_b = weight_b / (weight_b.sum() + 1e-9)

        trunk = self.model.trunk_features(chan_b, ctx_b)
        raw = self.model.evaluate(trunk, ctx_b)
        # Targets are +/-1; the head outputs tanh * value_scale, so rescale the
        # target into the head's own units rather than comparing centipawns to
        # a unit interval.
        target_scaled = target_b * self.model.config.value_scale
        loss = (weight_b * (raw - target_scaled) ** 2).sum() / self.config.value_scale**2

        self.value_opt.zero_grad()
        loss.backward()
        self.value_opt.step()
        self.value_steps += 1
        return {"trained": len(chans), "loss": float(loss.item())}

    def save(self, path: str | Path) -> dict:
        import json

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"config": self.model.config.__dict__,
             "state": self.model.state_dict()},
            path,
        )
        meta = path.with_suffix(".meta.json")
        meta.write_text(json.dumps({
            "n_parameters": self.model.n_parameters(),
            "decision_steps": self.decision_steps,
            "value_steps": self.value_steps,
            "config": self.config.__dict__,
        }, indent=2), encoding="utf-8")
        return {"model": str(path), "meta": str(meta)}

    @staticmethod
    def load(path: str | Path, config: GuidedTrainConfig | None = None) -> "GuidedTrainer":
        blob = torch.load(path, weights_only=False)
        model = GuidedModel(GuidedConfig(**blob["config"]))
        model.load_state_dict(blob["state"])
        return GuidedTrainer(model, config)
