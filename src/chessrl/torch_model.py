"""L4: the L3 model rewritten in torch, behind the same protocol.

The spec says L4 "overwrites everything else", which is the right instinct about
*intent* -- torch is the general tool and the hand-rolled numpy perceptron is a
detour -- but the wrong instruction about *this repo*. The benchmark only means
anything if every level is interchangeable, so L4 is the same factorised model
with the same feature extraction, expressed as an ``nn.Module``. That buys three
things the numpy version cannot have:

* autograd, so the credit-assignment hand-derivation in ``L3Trainer`` can be
  checked against a real gradient instead of trusted;
* an optimiser, so training is not a fixed learning rate;
* a place to grow, which is what L5 and L6 need.

Everything shared with L3 -- the encoder, the canonical frame, the geometry
features, the factored policy and its masks -- is imported, never forked.
Forking it would make the parity test vacuous: two implementations that agree
only because they are the same code prove nothing.

The parity test is the point of this level
------------------------------------------
``tests/test_torch_model.py::test_logits_match_l3_exactly`` loads L3's weights
into L4 and requires every logit to agree to float32 tolerance. This is a much
stronger claim than "the model trains and looks reasonable": it pins the
architecture, the feature order, the canonical convention and the weight layout
all at once, and it fails loudly on any of them. Getting it to pass is what
proves the numpy perceptron was correct rather than merely self-consistent.

torch is an optional extra
--------------------------
Importing this module without torch installed raises a clear ``ImportError``.
L0-L3 must never import it; ``tests/test_perceptron.py`` asserts that the numpy
perceptron does not.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import chess
import numpy as np

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
except ImportError as exc:  # pragma: no cover - covered by a dedicated test
    raise ImportError(
        "chessrl.torch_model needs PyTorch, which is an optional extra used "
        "only by L4 and L5. Install it with:\n"
        "    ./.venv/Scripts/python.exe -m pip install torch "
        "--index-url https://download.pytorch.org/whl/cpu"
    ) from exc

from . import masks as M
from .encode import BINARY_CHANNELS, TOTAL_CHANNELS, board_context_features, encode
from .perceptron import (
    GEOMETRY_NAMES,
    MOVE_FEATURE_NAMES_L3,
    PerceptronConfig,
    PerceptronScorer,
)
from .policy import FactoredSoftmaxPolicy


def torch_masked_softmax(logits: torch.Tensor, mask: np.ndarray) -> torch.Tensor:
    """Differentiable ``masked_softmax``, mirroring :func:`chessrl.policy.masked_softmax`.

    A torch-side twin is required rather than adapting the numpy one, because
    the numpy version detaches: converting logits to numpy and back throws away
    the graph, so ``loss.backward()`` fails with "does not require grad" several
    frames away from the actual mistake. Keeping the whole forward pass in torch
    is what makes autograd work at all.

    Numerically the construction is the same: subtract the masked maximum before
    exponentiating, and return exact zeros outside the mask.
    """
    legal = torch.as_tensor(mask.astype(bool), dtype=torch.bool,
                            device=logits.device)
    if not bool(legal.any()):
        return torch.zeros_like(logits)
    neg_inf = torch.finfo(logits.dtype).min
    masked = torch.where(legal, logits, torch.full_like(logits, neg_inf))
    return torch.softmax(masked, dim=-1)


@dataclass
class TorchConfig:
    """Model shape. Mirrors :class:`PerceptronConfig` one for one.

    Kept as a separate dataclass rather than reusing the numpy one so the two
    levels can diverge later without one dragging the other -- but every field
    here has an exact counterpart, because that is what the parity test needs.
    """

    use_value_channels: bool = False
    use_context: bool = True
    hidden: int = 32

    def to_perceptron_config(self) -> PerceptronConfig:
        return PerceptronConfig(
            use_value_channels=self.use_value_channels,
            use_context=self.use_context,
            hidden=self.hidden,
            quantised=False,
        )

    @classmethod
    def from_perceptron_config(cls, config: PerceptronConfig) -> "TorchConfig":
        return cls(
            use_value_channels=config.use_value_channels,
            use_context=config.use_context,
            hidden=config.hidden,
        )


def canonical_squares(board: chess.Board) -> np.ndarray:
    """Canonical square map for a position, delegated to L3's implementation.

    Delegated rather than reimplemented: this map is what makes one weight set
    serve both colours, and two copies of it that drift apart is precisely the
    failure mode the parity test is meant to catch.
    """
    return PerceptronScorer._canonical_squares(board)


def geometry_table() -> np.ndarray:
    """``(64, 64, n_geom)`` geometry features, built by L3's own function.

    A reimplementation here is exactly the fork that would make the parity test
    meaningless.
    """
    scorer = PerceptronScorer(PerceptronConfig(use_context=False))
    rows = [
        np.stack([scorer._geom_features(from_sq, t) for t in range(64)])
        for from_sq in range(64)
    ]
    return np.stack(rows).astype(np.float32)


class TorchScorer(nn.Module):
    """L3's factorised perceptron as a torch module.

    Parameter names deliberately match :class:`PerceptronScorer`'s attributes so
    that transfer in both directions is a straight copy with no name mapping to
    get wrong:

    ``W_from (64, n_ch)``, ``b_from (64,)``, ``W_ctx (h, n_ctx)``,
    ``W_hid (64, h)``, ``W_to_dst (64, n_ch)``, ``W_to_src (64, n_ch)``,
    ``W_geom (64, n_geom)``, ``promo (NUM_PROMO, n_move)``.

    The heads take tensors of *square indices* so a whole batch of positions can
    be scored in one matmul. That batching is the practical reason L4 exists
    beyond the exercise -- the numpy heads are inherently one position at a time.
    """

    PARAM_NAMES = (
        "W_from", "b_from", "W_ctx", "W_hid",
        "W_to_dst", "W_to_src", "W_geom", "promo",
    )

    def __init__(self, config: TorchConfig | None = None, seed: int = 0):
        super().__init__()
        self.config = config or TorchConfig()
        self.n_ch = (
            TOTAL_CHANNELS if self.config.use_value_channels else BINARY_CHANNELS
        )
        self.n_ctx = 9 if self.config.use_context else 0
        self.h = self.config.hidden if self.config.use_context else 0

        gen = torch.Generator().manual_seed(seed)

        def init(*shape) -> nn.Parameter:
            return nn.Parameter(
                torch.randn(*shape, generator=gen, dtype=torch.float32) * 0.01
            )

        self.W_from = init(64, self.n_ch)
        self.b_from = nn.Parameter(torch.zeros(64, dtype=torch.float32))
        if self.h:
            self.W_ctx = init(self.h, self.n_ctx)
            self.W_hid = init(64, self.h)
        else:
            # Registering as None (rather than omitting) keeps the attribute
            # present so transfer code can iterate PARAM_NAMES unconditionally.
            self.register_parameter("W_ctx", None)
            self.register_parameter("W_hid", None)
        self.W_to_dst = init(64, self.n_ch)
        self.W_to_src = init(64, self.n_ch)
        self.W_geom = init(64, len(GEOMETRY_NAMES))
        self.promo = init(M.NUM_PROMO, len(MOVE_FEATURE_NAMES_L3))

        # Buffers, not parameters: they must travel with the module but must
        # never receive a gradient.
        self.register_buffer(
            "geom_table",
            torch.as_tensor(geometry_table(), dtype=torch.float32),
            persistent=False,
        )

    # ---- the three heads -----------------------------------------------

    def from_logits(
        self, chan: torch.Tensor, ctx: torch.Tensor | None = None
    ) -> torch.Tensor:
        """``(B, 64)`` origin logits from a ``(B, n_ch, 64)`` batch.

        ``chan.permute(0, 2, 1) * W_from`` then summing the last axis contracts
        the *channel* axis per square, which is the operation L3 spells as
        ``einsum("sc,cs->s")``. Writing it as a matmul over the batch is the same
        mistake as in numpy: it silently produces every-square-against-every-
        square scores.

        ``ctx=None`` with the context pathway enabled means "use the context
        already bound to the module", not "skip the pathway". Skipping it would
        silently change the model's output -- a difference of the same order as
        the weights themselves -- and the caller would have no way to notice.
        Passing ``None`` while a context exists is only legitimate from the
        convenience paths, so the fallback is what makes them safe.
        """
        logits = (chan.permute(0, 2, 1) * self.W_from.unsqueeze(0)).sum(-1)
        logits = logits + self.b_from
        if self.h:
            if ctx is None:
                ctx = getattr(self, "_bound_ctx", None)
            if ctx is None:
                raise RuntimeError(
                    "the context pathway is enabled but no context was "
                    "supplied; call set_position or pass ctx explicitly"
                )
            hidden = F.relu(ctx @ self.W_ctx.t())       # (B, h)
            logits = logits + hidden @ self.W_hid.t()   # (B, 64)
        return logits

    def to_logits(self, chan: torch.Tensor, from_sq: torch.Tensor) -> torch.Tensor:
        """``(B, 64)`` destination logits conditioned on a batch of origins.

        ``chan`` may have ``B == 1`` while ``from_sq`` has many entries: scoring
        every origin of one position in a single call is the common case and is
        the whole reason to batch. Broadcasting is explicit rather than relying
        on torch's implicit rules so that a genuine shape mismatch still fails
        loudly instead of quietly scoring the wrong plane.
        """
        if chan.shape[0] == 1 and from_sq.shape[0] != 1:
            chan = chan.expand(from_sq.shape[0], -1, -1)
        if chan.shape[0] != from_sq.shape[0]:
            raise ValueError(
                f"batch mismatch: {chan.shape[0]} positions vs "
                f"{from_sq.shape[0]} origins"
            )
        out = (chan.permute(0, 2, 1) * self.W_to_dst.unsqueeze(0)).sum(-1)
        src = self._at_square(chan, from_sq)            # (B, n_ch)
        out = out + src @ self.W_to_src.t()
        geom = self.geom_table[from_sq]                 # (B, 64, n_geom)
        out = out + (geom * self.W_geom.unsqueeze(0)).sum(-1)
        return out

    def promo_logits(
        self, chan: torch.Tensor, from_sq: torch.Tensor, to_sq: torch.Tensor
    ) -> torch.Tensor:
        """``(B, NUM_PROMO)`` promotion-slot logits."""
        return self.move_features(chan, from_sq, to_sq) @ self.promo.t()

    @staticmethod
    def _at_square(chan: torch.Tensor, sq: torch.Tensor) -> torch.Tensor:
        """Gather ``chan[b, :, sq[b]]`` for a batch, as ``(B, n_ch)``."""
        idx = sq.reshape(-1, 1, 1).expand(-1, chan.shape[1], 1)
        return chan.gather(2, idx).squeeze(-1)

    @classmethod
    def move_features(
        cls, chan: torch.Tensor, from_sq: torch.Tensor, to_sq: torch.Tensor
    ) -> torch.Tensor:
        """``(B, n_move)`` promotion-head features for a batch of moves."""
        n_ch = chan.shape[1]
        fr = (from_sq >> 3).float()
        ff = (from_sq & 7).float()
        tr = (to_sq >> 3).float()
        tf = (to_sq & 7).float()
        src = cls._at_square(chan, from_sq).sum(-1) / n_ch
        dst = cls._at_square(chan, to_sq).sum(-1) / n_ch
        return torch.stack(
            [torch.ones_like(fr), dst, src, (tr - fr) / 7.0, (tf - ff).abs() / 7.0],
            dim=-1,
        )

    # ---- transfer to and from L3 ---------------------------------------

    def load_from_perceptron(self, scorer: PerceptronScorer) -> None:
        """Copy L3 weights in. The parity test lives or dies on this."""
        with torch.no_grad():
            for name in ("W_from", "b_from", "W_to_dst", "W_to_src",
                         "W_geom", "promo"):
                getattr(self, name).copy_(
                    torch.as_tensor(getattr(scorer, name), dtype=torch.float32)
                )
            if self.h:
                self.W_ctx.copy_(torch.as_tensor(scorer.W_ctx, dtype=torch.float32))
                self.W_hid.copy_(torch.as_tensor(scorer.W_hid, dtype=torch.float32))

    def to_perceptron(
        self, scorer: PerceptronScorer | None = None
    ) -> PerceptronScorer:
        """Copy the weights back out into an L3 scorer."""
        out = scorer or PerceptronScorer(self.config.to_perceptron_config())
        with torch.no_grad():
            for name in ("W_from", "b_from", "W_to_dst", "W_to_src",
                         "W_geom", "promo"):
                getattr(out, name)[:] = (
                    getattr(self, name).detach().cpu().numpy().astype(np.float32)
                )
            if self.h:
                out.W_ctx[:] = self.W_ctx.detach().cpu().numpy()
                out.W_hid[:] = self.W_hid.detach().cpu().numpy()
        return out

    @property
    def n_parameters(self) -> int:
        return int(sum(p.numel() for p in self.parameters() if p.requires_grad))

    def parameter_dict(self) -> dict[str, np.ndarray]:
        """Named parameters as numpy, mirroring ``PerceptronScorer.parameters``."""
        out = {}
        for name, param in self.named_parameters():
            out[name] = param.detach().cpu().numpy()
        return out

    def state_dict_plain(self) -> dict:
        """A JSON-friendly snapshot, matching L3's shape for comparison."""
        return {
            "config": self.config.__dict__,
            "params": {k: v.tolist() for k, v in self.parameter_dict().items()},
        }

    def load_state_dict_plain(self, state: dict) -> None:
        """Inverse of :meth:`state_dict_plain`.

        Named separately from ``nn.Module.load_state_dict`` on purpose: the
        inherited one expects torch's own nested format, and calling it with a
        JSON blob fails in a confusing way.
        """
        named = dict(self.named_parameters())
        with torch.no_grad():
            for name, values in state["params"].items():
                if name in named:
                    named[name].copy_(
                        torch.as_tensor(values, dtype=torch.float32)
                    )


# --------------------------------------------------------------------------
# the position-bound scorer the policy protocol needs
# --------------------------------------------------------------------------

class TorchScorerAdapter:
    """Wraps :class:`TorchScorer` to satisfy the square-indexed ``Scorer``.

    This exists because of a naming collision that is easy to walk into: the
    ``Scorer`` protocol's ``from_logits`` takes a *board* and returns a
    raw-square-indexed vector, while the ``nn.Module``'s ``from_logits`` takes
    *encoded planes* and returns a canonical-square-indexed vector. The policy
    calls ``self.scorer.from_logits``, so if ``self.scorer`` were the module
    itself the board would land in a tensor argument.

    The adapter owns the current position's planes, context and canonical map,
    and translates in both directions. It is deliberately a plain object rather
    than an ``nn.Module``: it holds no parameters, so keeping it outside the
    parameter tree means ``optimiser.parameters()`` picks up exactly the model.
    """

    def __init__(self, model: TorchScorer):
        self.model = model
        self._chan: torch.Tensor | None = None
        self._ctx: torch.Tensor | None = None
        self._canon_sq: np.ndarray | None = None
        self._turn_white = True

    @property
    def config(self) -> TorchConfig:
        return self.model.config

    def set_position(self, board: chess.Board) -> None:
        tensor = encode(
            board,
            with_values=self.model.config.use_value_channels,
            canonicalise=True,
        )
        n_ch = self.model.n_ch
        flat = tensor[:n_ch].reshape(n_ch, 64)
        self._chan = torch.as_tensor(flat, dtype=torch.float32).unsqueeze(0)
        if self.model.h:
            ctx = board_context_features(board)
            self._ctx = torch.as_tensor(ctx, dtype=torch.float32).unsqueeze(0)
        else:
            self._ctx = None
        self._turn_white = board.turn == chess.WHITE
        self._canon_sq = canonical_squares(board)
        # Mirror the context onto the module so that a convenience call which
        # omits it does not silently drop the pathway. See
        # ``TorchScorer.from_logits``.
        self.model._bound_ctx = self._ctx

    def _scatter(self, canon_values: torch.Tensor) -> np.ndarray:
        """Map a ``(64,)`` canonical-square vector onto raw squares."""
        values = canon_values.detach().cpu().numpy().astype(np.float32)
        if self._turn_white:
            return values
        out = np.empty(64, dtype=np.float32)
        out[self._canon_sq] = values
        return out

    def _require_position(self) -> None:
        if self._chan is None:
            raise RuntimeError(
                "set_position must be called before using the torch heads"
            )

    # ---- Scorer protocol ------------------------------------------------

    def from_logits(self, board: chess.Board) -> np.ndarray:
        # Bind here rather than trusting the caller: the base policy calls this
        # directly from ``_argmax_factors`` and ``move_distribution``, so a stale
        # binding would score the previous position without complaining.
        self.set_position(board)
        with torch.no_grad():
            canon = self.model.from_logits(self._chan, self._ctx)[0]
        return self._scatter(canon)

    def to_logits(self, from_sq: int) -> np.ndarray:
        self._require_position()
        idx = torch.tensor([int(self._canon_sq[from_sq])], dtype=torch.long)
        with torch.no_grad():
            canon = self.model.to_logits(self._chan, idx)[0]
        return self._scatter(canon)

    def promo_logits(self, from_sq: int, to_sq: int) -> np.ndarray:
        self._require_position()
        idx = torch.tensor([int(self._canon_sq[from_sq])], dtype=torch.long)
        jdx = torch.tensor([int(self._canon_sq[to_sq])], dtype=torch.long)
        with torch.no_grad():
            out = self.model.promo_logits(self._chan, idx, jdx)[0]
        return out.detach().cpu().numpy().astype(np.float32)

    # ---- persistence, delegate to the module ----------------------------

    def state_dict(self) -> dict:
        return {
            "config": self.model.config.__dict__,
            "params": {k: v.tolist() for k, v in self.model.parameter_dict().items()},
        }

    def load_state_dict(self, state: dict) -> None:
        self.model.load_state_dict_plain(state)


class TorchPolicy(FactoredSoftmaxPolicy):
    """The factored policy driven by a torch model.

    ``self.scorer`` is a :class:`TorchScorerAdapter`, so the masking, sampling
    and temperature machinery is literally L1's, unchanged. That is the whole
    value of implementing the same protocol: L4 differs from L3 only in where
    the numbers come from.
    """

    def __init__(self, model: TorchScorer | None = None, **kwargs):
        if model is None:
            model = TorchScorer(seed=kwargs.get("seed", 0))
        self.model = model
        super().__init__(
            TorchScorerAdapter(model),
            name=kwargs.pop("name", "L4-torch"),
            **kwargs,
        )

    @property
    def adapter(self) -> TorchScorerAdapter:
        return self.scorer

    # ---- policy entry points: bind, then delegate -----------------------

    def move_distribution(self, board: chess.Board) -> np.ndarray:
        self.adapter.set_position(board)
        return super().move_distribution(board)

    def sample_factors(self, board: chess.Board, mask=None):
        self.adapter.set_position(board)
        return super().sample_factors(board, mask)

    def _argmax_factors(self, board: chess.Board, mask):
        self.adapter.set_position(board)
        return super()._argmax_factors(board, mask)

    def select(self, board: chess.Board) -> chess.Move:
        self.adapter.set_position(board)
        return super().select(board)

    # ---- persistence ---------------------------------------------------

    def state_dict(self) -> dict:
        state = super().state_dict()
        state["kind"] = "L4-torch"
        return state

    @classmethod
    def load(cls, path, **kwargs) -> "TorchPolicy":
        """Rebuild from disk with a :class:`TorchScorer`, not a blind one."""
        state = json.loads(Path(path).read_text(encoding="utf-8"))
        config = TorchConfig(**state["scorer"]["config"])
        model = TorchScorer(config, seed=kwargs.pop("seed", 0))
        model.load_state_dict_plain(state["scorer"])
        return cls(
            model,
            name=state.get("name", "L4-torch"),
            temperature=state.get("temperature", 1.0),
            **kwargs,
        )


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------

@dataclass
class TorchTrainConfig:
    """Training knobs. Deliberately tiny -- see the module docstring on scope."""

    lr: float = 0.02
    weight_decay: float = 0.0
    credit_decay: float = 0.99
    search_weight: float = 1.0
    top_k: int = 5
    search_depth: int = 5
    shaped_reward: bool = True
    shaped_weight: float = 0.3
    seed: int = 0


class TorchTrainer:
    """Trains a :class:`TorchPolicy` with autograd.

    Structure follows ``L3Trainer`` on purpose, so the two can be compared
    directly, but the update is a genuine backward pass rather than a
    hand-derived nudge. Where L3 uses a fixed-point-scaled step, this uses an
    optimiser -- which is the whole reason to rewrite the level.
    """

    def __init__(self, policy: TorchPolicy | None = None,
                 config: TorchTrainConfig | None = None):
        if isinstance(policy, TorchTrainConfig) and config is None:
            policy, config = None, policy
        self.config = config or TorchTrainConfig()
        self.policy = policy or TorchPolicy(seed=self.config.seed)
        self.optimiser = torch.optim.Adam(
            self.policy.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        self.games_played = 0
        self.updates = 0
        self.search_updates = 0

    def _dist_tensor(self, board: chess.Board) -> torch.Tensor:
        """Model probabilities for every legal move, differentiable end to end.

        Everything stays in torch. Routing logits through
        :func:`chessrl.policy.masked_softmax` and back would detach the graph,
        and the failure surfaces as a bare "does not require grad" from
        ``backward()`` -- several frames away from the conversion that caused it.
        """
        self.policy.adapter.set_position(board)
        mask = M.legal_move_mask(board)
        chan = self.policy.adapter._chan
        ctx = self.policy.adapter._ctx
        p_from = torch_masked_softmax(
            self.policy.model.from_logits(chan, ctx)[0], M.legal_from_mask(mask)
        )
        out = torch.zeros(M.ACTION_SPACE, dtype=torch.float32)
        for move in board.legal_moves:
            dest_mask = mask[move.from_square].any(axis=1)
            idx = torch.tensor(
                [int(self.policy.adapter._canon_sq[move.from_square])],
                dtype=torch.long,
            )
            p_to = torch_masked_softmax(
                self.policy.model.to_logits(chan, idx)[0], dest_mask
            )
            out[M.move_to_index(move)] = (
                p_from[move.from_square] * p_to[move.to_square]
            )
        return out

    def train_on_search_feedback(self, board: chess.Board, *, depth: int | None = None
                                 ) -> dict:
        """Pull the model towards a depth-N search's top-k moves.

        A cross-entropy-style loss against the search's soft target: the
        standard policy-imitation objective, and the thing autograd makes easy.
        L3 has to approximate the same update with a hand-derived per-move
        gradient, which is exactly the derivation the parity test validates.
        """
        from .search import MinimaxEngine

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

        self.policy.adapter.set_position(board)
        mask = M.legal_move_mask(board)
        chan = self.policy.adapter._chan
        ctx = self.policy.adapter._ctx
        p_from = torch_masked_softmax(
            self.policy.model.from_logits(chan, ctx)[0], M.legal_from_mask(mask)
        )

        weights = []
        losses = []
        for move, value in top:
            f = int(self.policy.adapter._canon_sq[move.from_square])
            t = int(self.policy.adapter._canon_sq[move.to_square])
            idx = torch.tensor([f], dtype=torch.long)
            p_to = torch_masked_softmax(
                self.policy.model.to_logits(chan, idx)[0],
                mask[move.from_square].any(axis=1),
            )
            # The search's preference, as a positive weight. Normalised against
            # the best move so the strongest one contributes 1.0 and a move
            # 300cp worse contributes ~0.37.
            weight = float(np.exp((value - best) / 300.0))
            p_move = (p_from[move.from_square] * p_to[t]).clamp(1e-9, 1.0)
            weights.append(weight)
            losses.append(-weight * torch.log(p_move))

        # Normalise by the total weight. Without this the loss scales with how
        # many moves the search likes, so a busy position gets a bigger step
        # than a quiet one for no principled reason.
        total = sum(weights) or 1.0
        loss = torch.stack(losses).sum() / total

        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()
        self.search_updates += len(top)
        return {"applied": True, "k": len(top), "loss": float(loss.item())}

    def save(self, path: str | Path) -> dict:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.policy.save(path)
        meta = path.with_suffix(".meta.json")
        meta.write_text(json.dumps({
            "games_played": self.games_played,
            "updates": self.updates,
            "search_updates": self.search_updates,
            "n_parameters": self.policy.model.n_parameters,
            "config": self.config.__dict__,
        }, indent=2), encoding="utf-8")
        return {"model": str(path), "meta": str(meta)}
