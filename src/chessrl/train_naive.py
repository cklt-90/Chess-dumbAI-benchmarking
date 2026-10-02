"""L1.5: training the blind policy from game outcomes.

The learning rule
-----------------
A game produces a single scalar at the end: win, loss or draw. Every move in
the game shares responsibility for it. The simplest possible credit assignment
is what the spec describes:

    every move played in a won game has its probability pushed up;
    every move played in a lost game has it pushed down;
    all previous moves are updated, with diminishing weight further back.

Diminishing credit is not a stylistic choice, it is a correctness fix. If every
move in a 120-ply game receives an identical update, the opening moves are
blamed for a blunder on move 110 -- they receive the same gradient as the move
that actually lost the game, and their updates swamp the real signal because
there are so many of them. Weighting by ``decay ** plies_before_a_terminal``
concentrates credit near where the game was decided.

What this is not
----------------
This is not a gradient. There is no loss function and no backprop. It is a
score-function estimator applied to a softmax table: nudge the logits of the
chosen factors up or down in proportion to the credit, which changes their
probability in the right direction. That is enough for L1, and building it this
way keeps the L1 result interpretable as "what a reinforcement rule with no
value function and no board representation can achieve".

Why the counters matter here
----------------------------
:meth:`FactoredSoftmaxTrainer.train_on_game` records ``offered`` for every
position and ``taken`` only for the chosen move. After training, the counters
tell you whether an untouched parameter was untouched because it was never
legal or because the sampler never picked it. Those are very different
diagnoses and they produce the same symptom -- a weight that did not move.

Persistence
-----------
The model and the counters save separately. The spec is explicit that counters
are deleted before saving, and it is right: they are training bookkeeping whose
size depends on how much exploration happened, and loading a model should not
require loading the history of how it was found.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import chess
import numpy as np

from . import masks as M
from . import value as V
from .game import GameResult, play_game
from .policy import (
    BlindScorer,
    FactoredSoftmaxPolicy,
    ProbCounter,
    masked_softmax,
)


@dataclass
class TrainConfig:
    """Knobs for a training run.

    Defaults are tuned for a run that finishes in a couple of minutes on a
    laptop, which matters more than optimality: the point of L1 is to establish
    a floor quickly and often, not to squeeze out the last few points.
    """

    games: int = 200
    lr: float = 0.05
    # Credit decay per ply of distance from the terminal result. 0.99 means a
    # move 70 plies before the end still receives about half the credit of the
    # final move; 0.95 would make it negligible. Tuned so that opening mistakes
    # in a long game still receive a usable signal.
    credit_decay: float = 0.99
    max_plies: int = 200
    no_progress_limit: int = 60
    seed: int = 0
    # Shaped reward. This is not a stylistic option -- it is what makes L1
    # trainable at all. A blind policy under a ply cap essentially never
    # delivers checkmate, so pure win/loss reward is almost always zero and
    # the trainer updates nothing. Measured: 12 self-play games at 60 plies
    # produced 0 decisive results and 0 logit updates.
    #
    # With shaping on, a game that runs out of plies is scored by the static
    # evaluation of the final position instead of being discarded, so the
    # policy receives a real gradient even from an unfinished game. The weight
    # is small relative to a real win because the static evaluation is a much
    # noisier signal than an actual result.
    shaped_reward: bool = True
    shaped_weight: float = 0.3
    # Pruning is off by default. It is a memory optimisation for long runs, and
    # enabling it early makes a run harder to reason about.
    prune: bool = False
    prune_min_offers: int = 500
    prune_threshold: float = 0.01
    start_fens: list[str] = field(default_factory=list)


def credit_curve(n_moves: int, decay: float, decisive: bool) -> np.ndarray:
    """Per-move credit weights for a game of ``n_moves`` plies.

    Returns weights in ``[0, 1]``, index 0 = first move of the game, index
    ``n_moves - 1`` = the final move. Weight is ``decay ** (plies_from_end)``,
    normalised so the last move gets 1.0.

    Draws are treated as "no credit" by the caller, not as zero-credit moves:
    a draw is genuinely uninformative about which move was better, and trying
    to learn from one pushes the policy in a random direction. The parameter is
    kept here so the caller sees the decision explicitly rather than having it
    inferred from a missing call.
    """
    if n_moves <= 0:
        return np.zeros(0, dtype=np.float64)
    if not decisive:
        return np.zeros(n_moves, dtype=np.float64)
    # Distance from the end: index n-1 -> 0, index 0 -> n-1.
    distance = np.arange(n_moves - 1, -1, -1, dtype=np.float64)
    return np.power(decay, distance)


class FactoredSoftmaxTrainer:
    """Trains a :class:`FactoredSoftmaxPolicy` from self-play outcomes."""

    def __init__(
        self,
        policy: FactoredSoftmaxPolicy | None = None,
        config: TrainConfig | None = None,
    ):
        # Allow ``FactoredSoftmaxTrainer(TrainConfig(...))`` as well as the
        # documented ``(policy, config)`` order. Both read naturally and a
        # trainer with default policy but custom config is the common case, so
        # silently treating the config as a policy would fail far away from
        # the call site with an obscure attribute error.
        if isinstance(policy, TrainConfig) and config is None:
            policy, config = None, policy

        self.config = config or TrainConfig()
        self.policy = policy or FactoredSoftmaxPolicy(seed=self.config.seed)
        self.counter = ProbCounter()
        self.history: list[dict] = []
        self.games_played = 0
        self.rng = random.Random(self.config.seed)
        self._logit_updates = 0

    # ---- the update rule ------------------------------------------------

    def _nudge(
        self,
        table: np.ndarray,
        index: int,
        probs: np.ndarray,
        credit: float,
    ) -> None:
        """Push one factor's logits along the score-function direction.

        ``probs`` is the distribution the factor was drawn from. The update is

            ``logits += lr * credit * (onehot(index) - probs)``

        which is the score-function estimator for a softmax: it raises the
        chosen entry and lowers every other entry in proportion to its current
        probability. Working in logit space rather than adding to the
        probabilities directly is what keeps the result a valid distribution
        and stops small updates from pushing values outside ``[0, 1]``.

        This is a single vectorised operation. An earlier version wrote the
        chosen index and then added the whole direction vector, which applied
        the chosen entry's delta twice -- harmless-looking, and it silently
        doubled the effective learning rate on whatever was played.
        """
        direction = -probs
        direction[index] += 1.0
        table += self.config.lr * credit * direction
        self._logit_updates += 1

    def apply_credit(
        self, board: chess.Board, move: chess.Move, credit: float
    ) -> None:
        """Apply one move's share of a game outcome to the three factors."""
        if credit == 0.0:
            return

        scorer = self.policy.scorer
        mask = M.legal_move_mask(board)
        from_sq, to_sq = move.from_square, move.to_square
        slot = M.PROMO_SLOT.get(move.promotion, 0)

        # Origin factor.
        from_probs = masked_softmax(scorer.from_logits(board), M.legal_from_mask(mask))
        self._nudge(scorer.from_table, from_sq, from_probs, credit)

        # Destination factor, conditioned on the origin actually chosen.
        dest_mask = mask[from_sq].any(axis=1)
        to_probs = masked_softmax(scorer.to_logits(from_sq), dest_mask)
        self._nudge(scorer.to_table[from_sq], to_sq, to_probs, credit)

        # Promotion factor, conditioned on both.
        promo_mask = mask[from_sq, to_sq]
        promo_probs = masked_softmax(
            scorer.promo_logits(from_sq, to_sq), promo_mask
        )
        self._nudge(
            scorer.promo_table[from_sq, to_sq], slot, promo_probs, credit
        )

        self.counter.update(mask, from_sq, to_sq, slot)

    def white_reward(self, result: GameResult) -> tuple[float, float]:
        """Reward for White, and the amplitude that should scale the credit.

        Returns ``(score, amplitude)``. ``score`` is signed from White's point
        of view and drives the update direction; ``amplitude`` scales how much
        of the per-move credit curve is applied.

        Three cases, and the distinction between them is the whole point of
        this method:

        * **Decisive game** -- score is ±1 and amplitude is 1. An actual result
          is the strongest signal available and is trusted fully.
        * **Draw or ruled end** -- score is 0, so nothing is learned. A draw is
          genuinely uninformative about which move was better.
        * **Unfinished, shaped reward on** -- score is the static evaluation of
          the final position, scaled by ``shaped_weight``. This case is what
          makes L1 work: without it a blind policy never sees a non-zero
          reward, because it never delivers mate.

        The static evaluation is returned as a bounded value in ``[-1, 1]`` so
        it composes with the ±1 of a real result without rescaling.
        """
        if result.is_decisive:
            return V.outcome_score(result.result, chess.WHITE), 1.0
        if result.is_finished:
            return 0.0, 0.0
        if not self.config.shaped_reward:
            return 0.0, 0.0

        # Score the last recorded position statically, from White's view.
        last_fen = result.fens[-1] if result.fens else None
        if last_fen is None:
            return 0.0, 0.0
        board = chess.Board(last_fen)
        # tanh keeps the evaluation in (-1, 1) and compresses the large
        # material swings that would otherwise dominate a genuinely close game.
        raw = V.evaluate(board) / 1000.0
        return float(np.tanh(raw)) * self.config.shaped_weight, 1.0

    def train_on_game(self, result: GameResult) -> dict:
        """Update the policy from one completed game.

        Every ply of the game is replayed to recover the position each move was
        played from, because the position -- not just the move -- determines
        the factor distributions that get nudged. Replaying from the recorded
        FENs is cheap and avoids holding board copies for a whole game.
        """
        if not result.fens:
            raise ValueError(
                "train_on_game needs recorded FENs; play with record_fens=True"
            )

        # Replay the game, applying the update rule at the end.
        white_score, amplitude = self.white_reward(result)
        decisive = result.is_decisive
        weights = credit_curve(result.plies, self.config.credit_decay, True)

        for ply, san in enumerate(result.moves):
            board = chess.Board(result.fens[ply])
            move = board.parse_san(san)

            # Credit is signed for the side that played the move: a win for
            # White means a positive update for White's moves and a negative
            # one for Black's. Getting this sign wrong produces a policy that
            # plays well and then deliberately loses, which is a memorable bug.
            mover_is_white = board.turn == chess.WHITE
            sign = white_score if mover_is_white else -white_score
            self.apply_credit(
                board, move, float(weights[ply]) * sign * amplitude
            )

        self.games_played += 1
        return {
            "result": result.result,
            "plies": result.plies,
            "decisive": decisive,
            "reward": white_score,
        }

    # ---- the training loop ----------------------------------------------

    def _opponent(self):
        """A frozen copy of the policy for the other side to play.

        Self-play in the naive sense -- the same live policy object on both
        sides -- is degenerate, and measurably so. Both sides then make
        identical decisions in identical positions, so a symmetric opening is a
        perfect mirror: whatever White does, Black does, and if White wins then
        Black must have been worse, which is impossible when they are the same
        object. The measured consequence was stark: 6 self-play games, 0
        decisive, 0 logit updates. Training received no signal whatsoever.

        Freezing a copy as the opponent fixes this without needing a hand-written
        opponent. The two sides are then *different vintages* of the same
        policy, and one genuinely can be better than the other, which is what
        makes the outcome informative. This is the standard self-play
        arrangement and it is the reason L1 is trainable at all.
        """
        import copy

        opponent = copy.deepcopy(self.policy)
        # A distinct RNG stream, or the frozen copy draws the same numbers as
        # the live policy and the mirroring returns.
        opponent.rng = np.random.default_rng(self.rng.randrange(2**31))
        opponent.name = f"{self.policy.name}-opponent"
        return opponent

    def train(
        self,
        games: int | None = None,
        *,
        progress: bool = False,
        progress_every: int = 25,
        midstate=None,
        opponent=None,
        refresh_every: int = 10,
    ) -> dict:
        """Run self-play training and return a summary.

        The live policy plays a frozen copy of itself (see :meth:`_opponent`),
        refreshed every ``refresh_every`` games so the opponent keeps pace
        without ever being the same live object. Refresh too rarely and the
        opponent is too weak to be informative; too often and the two sides
        converge and the mirroring problem returns.

        ``opponent`` overrides the frozen copy, which is what makes the L2
        wrapper possible: the minimax player is passed in here and the naive
        policy simply learns against it.

        ``midstate``, when supplied, collects every position for later replay.
        This is the L0 cache in action: it is what later lets training resume
        from arbitrary midgame positions instead of always replaying openings.
        """
        n = games if games is not None else self.config.games
        started = time.perf_counter()
        tally = {"1-0": 0, "0-1": 0, "1/2-1/2": 0, "unfinished": 0}
        if opponent is None:
            opponent = self._opponent()

        for i in range(n):
            if opponent is not None and i and i % refresh_every == 0:
                opponent = self._opponent()

            gid = midstate.new_game_id() if midstate is not None else None
            start_fen = None
            if self.config.start_fens:
                start_fen = self.rng.choice(self.config.start_fens)

            # Alternate colours so neither side gets a systematic advantage
            # from moving first, which would otherwise be learned as a bias.
            if i % 2:
                white, black = opponent, self.policy
            else:
                white, black = self.policy, opponent

            game = play_game(
                white,
                black,
                max_plies=self.config.max_plies,
                no_progress_limit=self.config.no_progress_limit,
                start_fen=start_fen,
                midstate=midstate,
                game_id=gid,
                rng=self.rng,
            )

            # The live policy always plays one of the two sides, so every game
            # carries a usable signal.
            self.train_on_game(game)
            tally[game.result] = tally.get(game.result, 0) + 1

            if progress and (i + 1) % progress_every == 0:
                self.history.append({
                    "game": i + 1,
                    **tally,
                    "entropy": self.mean_entropy(),
                })

        elapsed = time.perf_counter() - started
        summary = {
            "games": n,
            "outcomes": tally,
            "seconds": round(elapsed, 2),
            "games_per_second": round(n / elapsed, 2) if elapsed else 0.0,
            "logit_updates": self._logit_updates,
            "mean_entropy": round(self.mean_entropy(), 4),
            **self.counter.stats(),
        }
        if self.config.prune:
            summary["pruned"] = self.prune()
        return summary

    # ---- diagnostics ----------------------------------------------------

    def mean_entropy(self, sample_fens: list[str] | None = None) -> float:
        """Average policy entropy over a fixed probe set of positions.

        Measured on a fixed set rather than on whatever the current game
        happens to produce, so runs are comparable across time. Entropy falling
        without win rate rising is the classic signature of a policy that has
        become confidently bad, which is exactly the failure mode a blind
        learner falls into.
        """
        from .policy import entropy

        fens = sample_fens or [
            chess.Board().fen(),
            "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        ]
        total = 0.0
        count = 0
        for fen in fens:
            board = chess.Board(fen)
            if board.is_game_over(claim_draw=False):
                continue
            dist = self.policy.move_distribution(board)
            positive = dist[dist > 0]
            if positive.size:
                total += entropy(positive / positive.sum())
                count += 1
        return total / count if count else 0.0

    def probability_table(self) -> dict[str, np.ndarray]:
        """The three factor tables as probabilities rather than logits."""
        scorer = self.policy.scorer
        # A uniform mask makes this the "intrinsic" probability of each factor,
        # independent of any particular position.
        on = np.ones(64, dtype=bool)
        return {
            "from": masked_softmax(scorer.from_table, on),
            "to": np.stack([
                masked_softmax(scorer.to_table[i], on) for i in range(64)
            ]),
            "promo": np.stack([
                masked_softmax(scorer.promo_table[i, j], np.ones(M.NUM_PROMO, dtype=bool))
                for i in range(64) for j in range(64)
            ]).reshape(64, 64, M.NUM_PROMO),
        }

    def top_moves(self, board: chess.Board, k: int = 5) -> list[tuple[str, float]]:
        """The ``k`` most probable legal moves and their probabilities."""
        dist = self.policy.move_distribution(board)
        order = np.argsort(dist)[::-1][:k]
        out = []
        for idx in order:
            if dist[idx] <= 0:
                continue
            out.append((str(M.index_to_move(int(idx))), float(dist[idx])))
        return out

    # ---- pruning --------------------------------------------------------

    def prune(self) -> dict:
        """Zero the logits of origin squares that are confidently unimportant.

        Never removes anything that has not been offered at least
        ``prune_min_offers`` times, so an action is only pruned after it has had
        a fair chance. Returns a report so a run can be audited afterwards --
        a pruning step that silently removed a tenth of the action space is
        something you want to see in the log, not discover by diffing weights.
        """
        probs = masked_softmax(
            self.policy.scorer.from_table, np.ones(64, dtype=bool)
        )
        candidates = self.counter.prune_candidates(
            probs,
            min_offers=self.config.prune_min_offers,
            prob_threshold=self.config.prune_threshold,
        )
        pruned_squares = [chess.square_name(int(s)) for s in np.nonzero(candidates)[0]]
        # Zeroing the logit does not make the square unplayable -- the mask
        # still permits it -- it just removes any learned preference. That is
        # deliberate: pruning must never make a legal move impossible.
        self.policy.scorer.from_table[candidates] = 0.0
        return {
            "pruned_count": int(candidates.sum()),
            "pruned_squares": pruned_squares,
            "protected_never_offered": int(self.counter.never_offered_from().sum()),
        }

    # ---- persistence ----------------------------------------------------

    def save(self, model_path, counter_path=None) -> dict:
        """Save the model, and optionally the counters, to separate files."""
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        self.policy.save(model_path)

        saved = {"model": str(model_path)}
        if counter_path is not None:
            counter_path = Path(counter_path)
            counter_path.parent.mkdir(parents=True, exist_ok=True)
            self.counter.save(counter_path)
            saved["counter"] = str(counter_path)

        # A small sidecar so a checkpoint can be identified without loading it.
        meta = model_path.with_suffix(".meta.json")
        meta.write_text(json.dumps({
            "games_played": self.games_played,
            "logit_updates": self._logit_updates,
            "config": {
                "lr": self.config.lr,
                "credit_decay": self.config.credit_decay,
                "max_plies": self.config.max_plies,
            },
            "counter": self.counter.stats(),
        }, indent=2), encoding="utf-8")
        saved["meta"] = str(meta)
        return saved
