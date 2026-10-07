"""H23 behavioural half -- does a level castle when castling is available?

Registered protocol (HYPOTHESES.md, H23):

    Command: filtered position set (rights intact, both wings open, >=4 plies
    from any capture), then compare each level's castle rate against L2-d3 as
    oracle.

Why the self-play census (``h23_census.py``) cannot answer this
--------------------------------------------------------------
That census counts special moves inside self-play games. Those games are
dominated by fivefold repetition (see CLAIMS-CHANGELOG), so the count is a
statement about how the games end, not about whether the policy *would* castle
given the choice. The registered command replaces it with a static question
asked on a deliberately constructed position set: given a position where
castling is available -- and where a depth-3 search says it is best -- what does
each level play?

The corpus confound (measured 2026-10-07)
-----------------------------------------
The pilot's random-play corpus is castling-sparse. Of 500 training positions
only 15 have a legal castling move and only 3 have a castling move in the
teacher's top-5; the held-out set has 1 legal and 0 in targets. So a trained L3
that never castles would be uninterpretable -- the signal never presented the
choice. This script therefore builds a *castling-rich* labelled corpus and
measures the trained arm on a held-out curated set, with a castling-poor control
trained on the same volume.

Filter semantics
----------------
- "rights intact": the side to move still has >=1 castling right.
- "both wings open": both castling moves are legal (strict) -- also reported
  with >=1 wing legal (relaxed), because strict positions are rare.
- ">=4 plies from any capture": tracked exactly during replay (the halfmove
  clock resets on pawn moves too, so it is only a conservative proxy).

Modes
-----
``--probe``   report curated-set yield only (fast, no training).
``--static``  run the registered census on the curated set for every level.
``--train``   train L3 on a castling-rich corpus, then re-run the census.

Example smoke:
    ./.venv/Scripts/python.exe bench/scripts/h23_behavioural.py --probe --games 40
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[2]
for _path in (_REPO, _REPO / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import chess
import numpy as np

from chessrl.game import MaterialPolicy, RandomPolicy, play_game
from chessrl.masks import move_to_index
from chessrl.perceptron import L3Config, L3Policy, L3Trainer
from chessrl.policy import FactoredSoftmaxPolicy
from chessrl.search import MinimaxPolicy

# Reuse the pilot's position identity so dedup semantics match every other
# experiment in the repo. Never re-implement position sampling inline.
from bench.scripts.l3_heldout_diagnostic import _position_key


# --------------------------------------------------------------------------
# curated-set construction
# --------------------------------------------------------------------------

def legal_castling_moves(board: chess.Board) -> list[chess.Move]:
    """The castling moves legal in ``board`` (0, 1 or 2 of them)."""
    return [move for move in board.legal_moves if board.is_castling(move)]


class QuietPolicy:
    """Uniform over non-capturing legal moves, to generate opening-like positions.

    Random play reaches tactically noisy positions where a quiet developing move
    (castling) is almost never the single best move -- measured 2026-10-07:
    castling is the depth-3 best in 8/583 and the Stockfish best in 7/583
    castling-legal random positions. So the registered "castling is clearly
    best" set is nearly empty under random play. Quiet play keeps material
    balanced and the position opening-like, which is where castling is a normal
    move. Falls back to any legal move when no quiet move exists.
    """

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def select(self, board: chess.Board) -> chess.Move:
        quiet = [move for move in board.legal_moves if not board.is_capture(move)]
        return self.rng.choice(quiet or list(board.legal_moves))


def scan_castling_positions(
    n_games: int,
    seed: int,
    *,
    min_since_capture: int = 4,
    max_plies: int = 120,
    source: str = "random",
) -> list[dict]:
    """Scan games for positions where castling is legal.

    One position per *ply* is considered (not one per game), because the pilot's
    one-per-game sampler discards the early plies where rights are most often
    intact. Positions are deduped on the pilot's ``_position_key``.
    """
    rng = random.Random(seed)
    seen: set[tuple] = set()
    out: list[dict] = []
    for game_index in range(n_games):
        if source == "quiet":
            game = play_game(
                QuietPolicy(rng.randrange(1 << 31)),
                QuietPolicy(rng.randrange(1 << 31)),
                max_plies=min(max_plies, 40), no_progress_limit=100,
                rng=rng, record_fens=True,
            )
        else:
            game = play_game(
                RandomPolicy(rng.randrange(1 << 31)),
                RandomPolicy(rng.randrange(1 << 31)),
                max_plies=max_plies, no_progress_limit=60,
                rng=rng, record_fens=True,
            )
        board = chess.Board()
        since_capture = 0
        for san in game.moves:
            if since_capture >= min_since_capture and board.has_castling_rights(board.turn):
                castles = legal_castling_moves(board)
                if castles:
                    key = _position_key(board)
                    if key not in seen:
                        seen.add(key)
                        out.append({
                            "fen": board.fen(),
                            "n_castles": len(castles),
                            "plies_since_capture": since_capture,
                            "source_game": game_index,
                        })
            move = board.parse_san(san)
            since_capture = 0 if board.is_capture(move) else since_capture + 1
            board.push(move)
    return out


def d3_label_fn(depth: int, top_k: int):
    """Label source: the built-in depth-``depth`` top-``top_k`` search (L2-d3)."""
    labeler = L3Trainer(L3Config(seed=0, search_depth=depth, top_k=top_k))
    return lambda board: labeler.search_feedback(board, depth=depth, top_k=top_k)


def stockfish_label_fn(engine_path: str, depth: int, top_k: int):
    """Label source: a UCI engine (Stockfish), MultiPV top-``top_k``.

    Needed because the depth-3 oracle ranks castling *best* in only ~1% of
    castling-legal positions (measured 2026-10-07): the evaluator's castling
    signal is too weak for the registered "castling is clearly best" filter to
    yield a usable set. A true master supplies the castling preference the
    local teacher lacks.
    """
    from chessrl.master import EngineMaster

    master = EngineMaster(engine_path, top_k=top_k, depth=depth)

    def label(board):
        return master.targets(board)

    label.close = master.close  # so the caller can shut the engine process down
    return label


def oracle_label(records: list[dict], label_fn) -> list[dict]:
    """Attach ``(move, weight)`` labels from ``label_fn`` to each position."""
    labeled = []
    for record in records:
        board = chess.Board(record["fen"])
        targets = label_fn(board)
        if not targets:
            raise AssertionError(f"nonterminal position produced no targets: {record['fen']}")
        rows = [{"uci": move.uci(), "weight": float(weight)} for move, weight in targets]
        best_move = chess.Move.from_uci(rows[0]["uci"])
        has_castle = any(board.is_castling(chess.Move.from_uci(row["uci"])) for row in rows)
        labeled.append({
            **record,
            "targets": rows,
            "oracle_best_is_castle": bool(board.is_castling(best_move)),
            "oracle_topk_has_castle": bool(has_castle),
        })
    return labeled


# --------------------------------------------------------------------------
# census
# --------------------------------------------------------------------------

def _castle_mass(policy, board: chess.Board) -> float:
    """Total probability the policy puts on legal castling moves."""
    distribution = policy.move_distribution(board)
    return float(sum(distribution[move_to_index(move)] for move in legal_castling_moves(board)))


def _select_is_castle(policy, board: chess.Board) -> bool:
    move = policy.select(board)
    return bool(board.is_castling(move))


def census(policy_factory, records: list[dict]) -> dict:
    """Castle rate and (for distribution policies) castling mass on ``records``."""
    policy = policy_factory()
    select_castle = []
    mass = []
    mass_supported = True
    for record in records:
        board = chess.Board(record["fen"])
        select_castle.append(float(_select_is_castle(policy, board)))
        if mass_supported:
            try:
                mass.append(_castle_mass(policy, board))
            except (AttributeError, NotImplementedError):
                mass_supported = False
    result = {
        "positions": len(records),
        "select_castle_rate": float(np.mean(select_castle)) if select_castle else float("nan"),
        "select_castle_count": int(sum(select_castle)),
    }
    if mass_supported and mass:
        result["castling_probability_mass"] = float(np.mean(mass))
    else:
        result["castling_probability_mass"] = None
    return result


LEVELS = {
    "random": lambda: RandomPolicy(seed=1),
    "L1-blind": lambda: FactoredSoftmaxPolicy(seed=1),
    "material": lambda: MaterialPolicy(seed=1),
    "L2-d3": lambda: MinimaxPolicy(depth=3),
    "L3-untrained": lambda: L3Policy(seed=1),
}


def uniform_castling_baseline(records: list[dict]) -> float:
    """Castling mass a policy with *no preference* (uniform over legal moves) gets.

    This is the reference that makes the census interpretable: with ~35 legal
    moves and 1-2 castling moves, ~3-6% castling mass is the "does not care"
    level, not evidence of anything.
    """
    masses = []
    for record in records:
        board = chess.Board(record["fen"])
        legal = list(board.legal_moves)
        if legal:
            masses.append(len(legal_castling_moves(board)) / len(legal))
    return float(np.mean(masses)) if masses else float("nan")


def run_static(records: list[dict], levels: dict | None = None) -> dict:
    """The registered census: each level's castle rate vs the oracle.

    Returns, per subset, ``{"positions": n, "levels": {name: row}}`` -- the
    per-level rows are nested so a subset's size is unambiguous.
    """
    levels = levels or LEVELS
    subsets = {
        "all": records,
        "oracle_best_is_castle": [r for r in records if r["oracle_best_is_castle"]],
        "oracle_topk_has_castle": [r for r in records if r["oracle_topk_has_castle"]],
    }
    report = {}
    for subset_name, subset in subsets.items():
        report[subset_name] = {
            "positions": len(subset),
            "uniform_baseline_castling_mass": (uniform_castling_baseline(subset) if subset
                                               else float("nan")),
            "levels": ({name: census(factory, subset) for name, factory in levels.items()}
                       if subset else {}),
        }
    return report


# --------------------------------------------------------------------------
# trained arm: does a trained L3 castle where the oracle says it should?
# --------------------------------------------------------------------------

def evaluate_castling(policy, records: list[dict]) -> dict:
    """Castle rate and castling mass for a *distribution* policy."""
    select_castle = []
    mass = []
    for record in records:
        board = chess.Board(record["fen"])
        select_castle.append(float(_select_is_castle(policy, board)))
        mass.append(_castle_mass(policy, board))
    return {
        "positions": len(records),
        "select_castle_rate": float(np.mean(select_castle)) if select_castle else float("nan"),
        "castling_probability_mass": float(np.mean(mass)) if mass else float("nan"),
    }


def split_records(records: list[dict], heldout_fraction: float, seed: int):
    """Deterministic disjoint train / held-out split of a labelled position set."""
    if not records:
        return [], []
    ordered = sorted(records, key=lambda r: r["fen"])
    rng = random.Random(seed)
    rng.shuffle(ordered)
    n_held = max(1, int(round(len(ordered) * heldout_fraction)))
    n_held = min(n_held, len(ordered) - 1) if len(ordered) > 1 else len(ordered)
    return ordered[n_held:], ordered[:n_held]


def build_corpora(rich: list[dict], poor: list[dict], heldout_fraction: float,
                  split_seed: int) -> dict:
    """RICH / POOR / MIXED training corpora plus both held-out sets.

    The MIXED arm exists to break a confound in the plain RICH arm. Every
    position in this curated set has castling *legal* (that is how the set was
    built), so a RICH-only corpus perfectly confounds "castling is legal" with
    "castling is targeted": there is no negative example anywhere in training,
    and the learner can only ever learn "castle whenever legal". MIXED puts both
    classes in front of the same learner, which is what makes a *conditional*
    decision learnable at all.

    Two MIXED sizes, because each controls a different nuisance:
      - ``mixed_half``: same total volume as RICH (half rich + half poor), so
        updates are matched, at the cost of half the RICH dose.
      - ``mixed_full``: full RICH dose plus an equal POOR dose, so the RICH dose
        is matched, at the cost of twice the volume.
    Reporting both means the conclusion does not ride on which nuisance was
    chosen.
    """
    rich_train, rich_held = split_records(rich, heldout_fraction, split_seed)
    poor_train, poor_held = split_records(poor, heldout_fraction, split_seed)
    # Match the control's volume to the RICH corpus: the contrast is "was
    # castling ever a target", not "which arm saw more positions".
    n = min(len(rich_train), len(poor_train))
    rich_train, poor_train = rich_train[:n], poor_train[:n]

    half = n // 2
    mixed_half = _shuffle(rich_train[:half] + poor_train[:half], split_seed + 1)
    mixed_full = _shuffle(rich_train + poor_train, split_seed + 2)
    return {
        "rich_train": rich_train,
        "poor_train": poor_train,
        "mixed_half_train": mixed_half,
        "mixed_full_train": mixed_full,
        "rich_held": rich_held,
        "poor_held": poor_held,
        "n_matched": n,
    }


def _shuffle(records: list[dict], seed: int) -> list[dict]:
    """Deterministic shuffle, so class is not correlated with training order."""
    out = list(records)
    random.Random(seed).shuffle(out)
    return out


def apply_negative_castling(trainer, record: dict) -> int:
    """Give castling a negative gradient where the teacher rejected it.

    The production update path (``_apply_cached_targets``, mirrored from
    ``train_on_search_feedback``) applies credit **only to moves in the target
    list**. A move the teacher did *not* choose never receives a gradient, so
    "do not castle here" is unlearnable in principle -- it can only be expressed
    indirectly, by raising the alternatives and letting softmax renormalisation
    do the work. This supplies the missing half: the symmetric gap for a target
    probability of zero.

    Returns the number of castling moves credited (0 when the teacher did want
    castling, so the arm stays a pure addition to the standard update).
    """
    if record.get("oracle_topk_has_castle"):
        return 0
    board = chess.Board(record["fen"])
    castles = legal_castling_moves(board)
    if not castles:
        return 0
    trainer.policy.scorer.set_position(board)
    distribution = trainer.policy.move_distribution(board)
    for move in castles:
        gap = 0.0 - float(distribution[move_to_index(move)])
        trainer.apply_credit(board, move, trainer.config.search_weight * gap)
    return len(castles)


def train_and_evaluate(
    train_records: list[dict],
    eval_sets: dict[str, list[dict]],
    seeds: list[int],
    *,
    depth: int,
    top_k: int,
    search_weight: float,
    epochs: int = 1,
    neg_castling: bool = False,
) -> dict:
    """Train one L3 per seed on ``train_records``; score on every eval set.

    Uses the pilot's ``_apply_cached_targets`` so the update path is identical
    to every other supervised arm in the repo. ``epochs`` repeats the corpus so
    a small castling-rich set still receives enough updates.

    ``neg_castling`` additionally credits castling moves negatively wherever the
    teacher did not target them (see ``apply_negative_castling``). It is off by
    default because it is *not* the production update path -- it exists to
    isolate whether the missing negative gradient, rather than model capacity,
    is what blocks conditional castling.

    ``eval_sets`` is a mapping of name -> positions. Scoring an arm on a set the
    *other* arm was trained on (and which this arm never saw) is what separates
    a position-conditional castling decision from a blanket "always castle"
    bias -- a distinction the RICH arm's headline number cannot make alone.
    """
    from bench.scripts.l3_heldout_diagnostic import _apply_cached_targets

    per_seed = []
    for seed in seeds:
        trainer = L3Trainer(config=L3Config(
            seed=seed, search_depth=depth, top_k=top_k, search_weight=search_weight,
        ))
        untrained = {name: evaluate_castling(trainer.policy, recs)
                     for name, recs in eval_sets.items()}
        negative_updates = 0
        for _ in range(epochs):
            for record in train_records:
                _apply_cached_targets(trainer, record)
                if neg_castling:
                    negative_updates += apply_negative_castling(trainer, record)
        trained = {name: evaluate_castling(trainer.policy, recs)
                   for name, recs in eval_sets.items()}
        per_seed.append({
            "seed": seed,
            "updates": trainer.search_updates,
            "negative_updates": negative_updates,
            "untrained": untrained,
            "trained": trained,
        })
    summary = {name: _seed_summary(per_seed, name) for name in eval_sets}
    if "rich_held" in eval_sets and "poor_held" in eval_sets:
        summary["discrimination"] = _discrimination_summary(per_seed)
    return {"per_seed": per_seed, "summary": summary}


def _discrimination_summary(per_seed: list[dict]) -> dict:
    """Castling mass on ``rich_held`` minus on ``poor_held``, per seed.

    This is the quantity the headline numbers cannot express: a large positive
    value means the learner castles *where the teacher wants it* and not where
    it does not; a value near zero means the two are indistinguishable, i.e. a
    blanket habit. Reported as trained-minus-untrained so it is a learning
    effect rather than a property of the untrained prior.
    """
    from bench.scripts.l3_heldout_diagnostic import _t_critical_95

    metric = "castling_probability_mass"
    trained, untrained = [], []
    for row in per_seed:
        t = row["trained"]["rich_held"][metric] - row["trained"]["poor_held"][metric]
        u = row["untrained"]["rich_held"][metric] - row["untrained"]["poor_held"][metric]
        trained.append(t)
        untrained.append(u)
    paired = [t - u for t, u in zip(trained, untrained)]
    mean_paired = float(np.mean(paired))
    sd = float(np.std(paired, ddof=1)) if len(paired) > 1 else 0.0
    half = (_t_critical_95(len(paired)) * sd / math.sqrt(len(paired))
            if len(paired) > 1 else 0.0)
    return {
        "metric": "castling_mass(rich_held) - castling_mass(poor_held)",
        "trained_mean": float(np.mean(trained)),
        "untrained_mean": float(np.mean(untrained)),
        "paired_gain_mean": mean_paired,
        "paired_gain_95pct_t_interval": [mean_paired - half, mean_paired + half],
        "seed_level_paired_differences": paired,
        "interval_scope": "conditional on this fixed corpus/split; initialization-seed variability only",
    }


def _seed_summary(per_seed: list[dict], eval_name: str) -> dict:
    """Mean and seed-conditional 95% t interval of the trained-minus-untrained gap."""
    from bench.scripts.l3_heldout_diagnostic import _t_critical_95

    out = {}
    for metric in ("select_castle_rate", "castling_probability_mass"):
        trained = [row["trained"][eval_name][metric] for row in per_seed]
        untrained = [row["untrained"][eval_name][metric] for row in per_seed]
        paired = [t - u for t, u in zip(trained, untrained)]
        mean = float(np.mean(trained))
        mean_paired = float(np.mean(paired))
        sd = float(np.std(paired, ddof=1)) if len(paired) > 1 else 0.0
        half = (_t_critical_95(len(paired)) * sd / math.sqrt(len(paired))
                if len(paired) > 1 else 0.0)
        out[metric] = {
            "trained_mean": mean,
            "untrained_mean": float(np.mean(untrained)),
            "paired_gain_mean": mean_paired,
            "paired_gain_95pct_t_interval": [mean_paired - half, mean_paired + half],
            "seed_level_paired_differences": paired,
        }
    out["interval_scope"] = "conditional on this fixed corpus/split; initialization-seed variability only"
    return out


# --------------------------------------------------------------------------
# feature probe: CAN the castling logit separate RICH from POOR at all?
# --------------------------------------------------------------------------
#
# The trained arms answer "did the learner condition castling on the position".
# They cannot distinguish "would not" from "could not". This probe addresses
# the second question directly, using only the features the castling logit is
# actually a function of:
#
#   score(from, to) = from_logit(from) + to_logit(to | from)
#   from_logit(s)   = W_from[s].chan[:, s] + b_from[s] + (W_hid @ relu(W_ctx @ ctx))[s]
#   to_logit(t | f) = W_to_dst[t].chan[:, t] + W_to_src[t].chan[:, f] + W_geom[t].geom(f, t)
#
# So apart from the geometry (constant for a given castling move) the logit
# depends on: the channel vectors at the king square and the destination
# square, and the nine position-global context scalars. Nothing else on the
# board reaches it -- a tactic on the other side of the board is invisible to
# this score. If RICH and POOR are not separable from those inputs alone, no
# amount of training or corpus balancing can produce conditional castling.

def probe_feature_rows(records: list[dict], *, variant: str = "context_local"):
    """Feature matrix + labels for one variant of the separability probe.

    ``context``            the nine global scalars only.
    ``context_local``      those plus the encoder channels at the castling
                           origin and destination squares -- everything the
                           castling logit is a function of.
    ``full_board_pooled``  channel means over the whole board. Not visible to
                           the castling logit; included as the control that
                           separates "the information is not there" from "the
                           information is there but unreachable".
    """
    from chessrl.encode import board_context_features

    if variant not in ("context", "context_local", "full_board_pooled"):
        raise ValueError(f"unknown probe variant: {variant}")
    scorer = L3Policy(seed=0).scorer
    rows, labels = [], []
    for record in records:
        board = chess.Board(record["fen"])
        scorer.set_position(board)
        chan = scorer.channels_for(board)
        if variant == "full_board_pooled":
            features = list(chan.mean(axis=1))
        else:
            features = list(board_context_features(board))
            if variant == "context_local":
                castles = legal_castling_moves(board)
                if not castles:
                    continue
                canon = scorer._canon_sq
                origins = sorted({int(canon[m.from_square]) for m in castles})
                dests = sorted({int(canon[m.to_square]) for m in castles})
                features += list(np.mean(chan[:, origins], axis=1))
                features += list(np.mean(chan[:, dests], axis=1))
        rows.append([float(v) for v in features])
        labels.append(1.0 if record["oracle_topk_has_castle"] else 0.0)
    return np.asarray(rows, dtype=np.float64), np.asarray(labels, dtype=np.float64)


def fit_linear_probe(x: np.ndarray, y: np.ndarray, *, steps: int = 400,
                     lr: float = 0.5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardised logistic regression by plain gradient descent.

    numpy-only on purpose: the .venv carries numpy/chess/pytest and nothing
    else, and a probe does not need a solver library. Returns
    ``(weights, mean, sd)``; the standardisation must be carried into
    evaluation, or the held-out rows would be standardised by their own
    statistics and the reported accuracy would be a leak.
    """
    mean = x.mean(axis=0)
    sd = x.std(axis=0)
    sd[sd == 0.0] = 1.0
    z = np.hstack([np.ones((x.shape[0], 1)), (x - mean) / sd])
    w = np.zeros(z.shape[1], dtype=np.float64)
    for _ in range(steps):
        p = 1.0 / (1.0 + np.exp(-(z @ w)))
        w -= lr * (z.T @ (p - y)) / len(y)
    return w, mean, sd


def probe_scores(x: np.ndarray, w: np.ndarray, mean: np.ndarray,
                 sd: np.ndarray) -> np.ndarray:
    """Fitted-probe probability for each row."""
    z = np.hstack([np.ones((x.shape[0], 1)), (x - mean) / sd])
    return 1.0 / (1.0 + np.exp(-(z @ w)))


def probe_accuracy(x: np.ndarray, y: np.ndarray, w: np.ndarray,
                   mean: np.ndarray, sd: np.ndarray) -> float:
    """Accuracy of a fitted probe on a (held-out) feature matrix."""
    p = probe_scores(x, w, mean, sd)
    return float(np.mean((p >= 0.5).astype(float) == y))


def roc_auc(scores: np.ndarray, y: np.ndarray) -> float:
    """Rank-based AUC (Mann-Whitney U), tie-aware.

    Accuracy is the wrong readout at 84/16 class balance: predicting the
    majority class every time scores ~0.83 while carrying no information at
    all. AUC is threshold-free, so it separates "no signal" (0.5) from
    "signal the decision threshold failed to use".
    """
    pos = scores[y == 1.0]
    neg = scores[y == 0.0]
    if not len(pos) or not len(neg):
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranked = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranked[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    rank_sum = float(np.sum(ranked[y == 1.0]))
    return (rank_sum - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))


def bootstrap_auc_ci(scores: np.ndarray, y: np.ndarray, *, resamples: int = 2000,
                     seed: int = 0) -> list[float]:
    """Stratified bootstrap 95% interval for an AUC.

    AUC point estimates on ~34 positives are nowhere near tight enough to read
    on their own, and the whole point of this probe is a comparison between two
    feature sets -- which needs an interval, not a difference of two numbers.
    Stratified (resample each class separately) so every replicate keeps both
    classes present; an unstratified draw can land on a single class and return
    NaN.
    """
    n = len(y)
    pos = [i for i in range(n) if y[i] == 1.0]
    neg = [i for i in range(n) if y[i] == 0.0]
    if not pos or not neg:
        return [float("nan"), float("nan")]
    rng = random.Random(seed)
    draws = []
    for _ in range(resamples):
        idx = [rng.choice(pos) for _ in pos] + [rng.choice(neg) for _ in neg]
        idx_arr = np.asarray(idx)
        draws.append(roc_auc(scores[idx_arr], y[idx_arr]))
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def run_feature_probe(records: list[dict], *, split_seed: int = 9001,
                      heldout_fraction: float = 0.34, steps: int = 400) -> dict:
    """Can a linear map from the castling logit's own inputs predict RICH/POOR?

    Reports accuracy against the majority-class baseline. A probe at the
    baseline means the two classes are not separable from the information the
    castling decision can see -- i.e. conditional castling is not merely
    unlearned but unavailable to this architecture.
    """
    out = {}
    for name in ("context", "context_local", "full_board_pooled"):
        x, y = probe_feature_rows(records, variant=name)
        order = list(range(len(y)))
        random.Random(split_seed).shuffle(order)
        n_held = max(1, int(round(len(order) * heldout_fraction)))
        held, train = order[:n_held], order[n_held:]
        w, mean, sd = fit_linear_probe(x[train], y[train], steps=steps)
        baseline = float(max(np.mean(y[held]), 1.0 - np.mean(y[held])))
        held_p = probe_scores(x[held], w, mean, sd)
        out[name] = {
            "features": int(x.shape[1]),
            "n": int(len(y)),
            "train_accuracy": probe_accuracy(x[train], y[train], w, mean, sd),
            "heldout_accuracy": probe_accuracy(x[held], y[held], w, mean, sd),
            "majority_baseline": baseline,
            # The threshold-free readouts: at this class balance a probe that
            # simply predicts the majority class scores 0.83 accuracy and 0.5
            # AUC, so the pair distinguishes "no signal" from "unused signal".
            "heldout_auc": roc_auc(held_p, y[held]),
            "heldout_auc_95pct_bootstrap_interval": bootstrap_auc_ci(held_p, y[held]),
            "heldout_predicted_positive_rate": float(np.mean(held_p >= 0.5)),
            "heldout_positive_base_rate": float(np.mean(y[held])),
            "heldout_n_positive": int(np.sum(y[held] == 1.0)),
        }
    return out


# --------------------------------------------------------------------------
# conditioning test: does castling mass TRACK the teacher's castling rank?
# --------------------------------------------------------------------------

_BUCKET_ORDER = ("1", "2-3", "4-5", "6-10", "11-20", ">20")


def rank_bucket(rank: int) -> str:
    """Bucket a teacher castling rank for reporting."""
    if rank <= 1:
        return "1"
    if rank <= 3:
        return "2-3"
    if rank <= 5:
        return "4-5"
    if rank <= 10:
        return "6-10"
    if rank <= 20:
        return "11-20"
    return ">20"


def spearman(xs: list[float], ys: list[float]) -> float:
    """Spearman rank correlation, without a scipy dependency."""
    n = len(xs)
    if n < 2:
        return float("nan")

    def ranks(values):
        order = sorted(range(n), key=lambda i: values[i])
        out = [0.0] * n
        for position, index in enumerate(order, start=1):
            out[index] = float(position)
        return out

    rx, ry = ranks(xs), ranks(ys)
    mx, my = float(np.mean(rx)), float(np.mean(ry))
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else float("nan")


def castling_rank(board: chess.Board, engine, depth: int, multipv: int) -> dict:
    """The engine's rank (1-based) and score gap for the best castling move.

    ``rank`` is the castling move's position in the engine's MultiPV ordering;
    ``gap_cp`` is best-minus-castling in centipawns. A rank beyond ``multipv``
    is reported as ``multipv + 1`` (i.e. "worse than everything measured").
    """
    import chess.engine

    castles = set(legal_castling_moves(board))
    infos = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=multipv)
    scored = []
    for info in infos:
        pv = info.get("pv")
        score = info.get("score")
        if pv and score is not None:
            scored.append((pv[0], score.pov(board.turn).score(mate_score=100000)))
    if not scored:
        return {"rank": multipv + 1, "gap_cp": None}
    best = scored[0][1]
    for index, (move, cp) in enumerate(scored, start=1):
        if move in castles:
            return {"rank": index, "gap_cp": best - cp}
    return {"rank": len(scored) + 1, "gap_cp": best - scored[-1][1]}


def compute_castling_ranks(records: list[dict], engine_path: str,
                           depth: int, multipv: int) -> list[dict]:
    """Attach the engine's castling rank/gap to each position (one engine pass)."""
    import chess.engine

    engine = chess.engine.SimpleEngine.popen_uci(str(engine_path))
    out = []
    try:
        for record in records:
            info = castling_rank(chess.Board(record["fen"]), engine, depth, multipv)
            out.append({**record, **info})
    finally:
        engine.quit()
    return out


def train_policies(train_records: list[dict], seeds: list[int], *,
                   depth: int, top_k: int, search_weight: float, epochs: int):
    """One trained L3 policy per seed, via the pilot's cached-label update path."""
    from bench.scripts.l3_heldout_diagnostic import _apply_cached_targets

    policies = []
    for seed in seeds:
        trainer = L3Trainer(config=L3Config(
            seed=seed, search_depth=depth, top_k=top_k, search_weight=search_weight,
        ))
        for _ in range(epochs):
            for record in train_records:
                _apply_cached_targets(trainer, record)
        policies.append(trainer.policy)
    return policies


def conditioning_summary(policies, ranked_records: list[dict]) -> dict:
    """Mean castling mass per teacher-castling-rank bucket, plus the correlation.

    A model that learned the teacher's *judgement* shows castling mass falling as
    the teacher's castling rank worsens. A blanket "always castle" habit shows a
    flat profile and a Spearman rho near zero.
    """
    rows = []
    for record in ranked_records:
        board = chess.Board(record["fen"])
        masses = [_castle_mass(policy, board) for policy in policies]
        rows.append({
            "fen": record["fen"], "rank": int(record["rank"]),
            "gap_cp": record["gap_cp"], "castling_mass": float(np.mean(masses)),
        })
    buckets: dict[str, list[float]] = {}
    for row in rows:
        buckets.setdefault(rank_bucket(row["rank"]), []).append(row["castling_mass"])
    rho = spearman([float(r["rank"]) for r in rows], [r["castling_mass"] for r in rows])
    return {
        "positions": len(rows),
        "spearman_teacher_rank_vs_castling_mass": rho if math.isfinite(rho) else None,
        "buckets": {
            name: {"n": len(buckets[name]), "castling_mass": float(np.mean(buckets[name]))}
            for name in _BUCKET_ORDER if name in buckets
        },
        "rows": rows,
    }


# --------------------------------------------------------------------------
# two-class conditioning: is the flatness castling-specific or architectural?
# --------------------------------------------------------------------------

_MOVE_CLASSES = ("castling", "capture")


def move_in_class(board: chess.Board, move: chess.Move, kind: str) -> bool:
    """Is ``move`` a member of ``kind``?

    'castling' is a *board-wide* decision -- whether the king is safe after
    castling depends on the pawn structure and open files across a whole flank,
    none of which reach the two-square receptive field of ``score(from, to)``.
    'capture' is *locally* decidable: the destination square's own channel
    vector already carries ``capture_targets``, ``opponent_defends`` and
    ``contested``, so the scorer can in principle tell a winning capture from a
    losing one.

    Running the identical conditioning test on both separates "the learner
    cannot condition on board-wide context" (the tier-2 claim) from "the
    learner cannot condition at all" (a training-signal or corpus problem,
    which would mean the tier-2 entry is aimed at the wrong defect).
    """
    if kind == "castling":
        return bool(board.is_castling(move))
    if kind == "capture":
        return bool(board.is_capture(move))
    raise ValueError(f"unknown move class: {kind!r}")


def class_moves(board: chess.Board, kind: str) -> list[chess.Move]:
    """Legal moves belonging to ``kind``."""
    return [m for m in board.legal_moves if move_in_class(board, m, kind)]


def class_mass(policy, board: chess.Board, kind: str) -> float:
    """Total probability the policy puts on moves of ``kind``."""
    distribution = policy.move_distribution(board)
    return float(sum(distribution[move_to_index(m)] for m in class_moves(board, kind)))


def analyse_scored(board: chess.Board, engine, depth: int,
                   multipv: int) -> list[tuple[chess.Move, int]]:
    """The engine's MultiPV ordering as ``(move, cp_from_side_to_move)``.

    Split out so one engine call serves every move class. Analysing twice per
    position would double the dominant cost of the probe for no information.
    """
    import chess.engine

    infos = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=multipv)
    scored = []
    for info in infos:
        pv = info.get("pv")
        score = info.get("score")
        if pv and score is not None:
            scored.append((pv[0], score.pov(board.turn).score(mate_score=100000)))
    return scored


def rank_within(scored: list[tuple[chess.Move, int]], members: set,
                multipv: int) -> dict:
    """Rank (1-based) of the best member of ``members`` inside a scored list."""
    if not scored:
        return {"rank": multipv + 1, "gap_cp": None}
    best = scored[0][1]
    for index, (move, cp) in enumerate(scored, start=1):
        if move in members:
            return {"rank": index, "gap_cp": best - cp}
    return {"rank": len(scored) + 1, "gap_cp": best - scored[-1][1]}


def class_rank(board: chess.Board, engine, depth: int, multipv: int, kind: str) -> dict:
    """The engine's rank (1-based) of the best move belonging to ``kind``.

    Mirrors :func:`castling_rank` but for an arbitrary move class. A rank beyond
    ``multipv`` is reported as ``multipv + 1`` ("worse than everything
    measured"). ``class_legal`` records how many members the class had, which
    is what makes the rank interpretable: a rank of 5 means something very
    different when there were 3 captures legal than when there were 40.
    """
    members = set(class_moves(board, kind))
    if not members:
        return {"rank": multipv + 1, "gap_cp": None, "class_legal": 0}
    info = rank_within(analyse_scored(board, engine, depth, multipv), members, multipv)
    return {**info, "class_legal": len(members)}


def compute_class_ranks(records: list[dict], engine_path: str, depth: int,
                        multipv: int, kinds: tuple[str, ...]) -> list[dict]:
    """Attach every class's rank/gap to each position, one engine call each."""
    import chess.engine

    engine = chess.engine.SimpleEngine.popen_uci(str(engine_path))
    out = []
    try:
        for record in records:
            board = chess.Board(record["fen"])
            scored = analyse_scored(board, engine, depth, multipv)
            row = dict(record)
            for kind in kinds:
                members = set(class_moves(board, kind))
                info = ({"rank": multipv + 1, "gap_cp": None} if not members
                        else rank_within(scored, members, multipv))
                row[f"{kind}_rank"] = info["rank"]
                row[f"{kind}_gap_cp"] = info["gap_cp"]
                row[f"{kind}_legal"] = len(members)
            out.append(row)
    finally:
        engine.quit()
    return out


def class_in_targets(board: chess.Board, record: dict, kind: str) -> bool:
    """Does the teacher's listed top-k contain a move of ``kind``?"""
    return any(move_in_class(board, chess.Move.from_uci(row["uci"]), kind)
               for row in record["targets"])


def class_mass_over_uniform(policy, board: chess.Board, kind: str) -> float:
    """Class probability mass divided by the no-preference share.

    Raw class mass is not comparable across classes: there are ~35 legal moves
    but only 1-2 castling moves, so castling mass and capture mass live on
    different scales. Dividing by ``n_class / n_legal`` puts both on a common
    axis where 1.0 means "indifferent" and values above 1 mean a preference.
    Without this, a cross-class comparison would mostly measure how many moves
    each class contains.
    """
    legal = list(board.legal_moves)
    members = [m for m in legal if move_in_class(board, m, kind)]
    if not members or not legal:
        return float("nan")
    distribution = policy.move_distribution(board)
    mass = float(sum(distribution[move_to_index(m)] for m in members))
    return mass / (len(members) / len(legal))


def bootstrap_spearman_ci(ranks: list[float], values: list[float], *,
                          resamples: int = 4000, seed: int = 7001) -> list[float]:
    """95% bootstrap interval for a Spearman rho.

    Reported because a bare rho invites reading a sign as a finding. At n=194
    the interval is roughly +/-0.14 wide, which is the difference between "this
    class conditions" and "this class is flat"; the interval is the only thing
    that separates those readings.
    """
    rng = np.random.default_rng(seed)
    n = len(ranks)
    if n < 3:
        return [float("nan"), float("nan")]
    r = np.asarray(ranks, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    out = []
    for _ in range(resamples):
        idx = rng.integers(0, n, n)
        value = spearman(list(r[idx]), list(v[idx]))
        if math.isfinite(value):
            out.append(value)
    if not out:
        return [float("nan"), float("nan")]
    return [float(v) for v in np.percentile(out, [2.5, 97.5])]


def paired_spearman_difference(rows_a: list[dict], rows_b: list[dict], *,
                               resamples: int = 4000, seed: int = 7002) -> dict:
    """Bootstrap interval for ``rho(a) - rho(b)`` on shared positions.

    The primary endpoint of the two-class test. Both arms are scored on the same
    held-out FENs, so the difference is paired and resampling positions is the
    right unit -- resampling seeds would answer a different question (does the
    effect survive re-initialisation, which the 8-seed mean already covers).
    """
    by_a = {r["fen"]: r for r in rows_a}
    by_b = {r["fen"]: r for r in rows_b}
    common = sorted(set(by_a) & set(by_b))
    if len(common) < 3:
        return {"n_common_positions": len(common), "mean": None,
                "ci95": [None, None], "excludes_zero": False}
    rng = np.random.default_rng(seed)
    ra = np.asarray([by_a[f]["rank"] for f in common], dtype=np.float64)
    va = np.asarray([by_a[f]["class_mass_over_uniform"] for f in common], dtype=np.float64)
    rb = np.asarray([by_b[f]["rank"] for f in common], dtype=np.float64)
    vb = np.asarray([by_b[f]["class_mass_over_uniform"] for f in common], dtype=np.float64)
    diffs = []
    for _ in range(resamples):
        idx = rng.integers(0, len(common), len(common))
        s_a = spearman(list(ra[idx]), list(va[idx]))
        s_b = spearman(list(rb[idx]), list(vb[idx]))
        if math.isfinite(s_a) and math.isfinite(s_b):
            diffs.append(s_a - s_b)
    if not diffs:
        return {"n_common_positions": len(common), "mean": None,
                "ci95": [None, None], "excludes_zero": False}
    lo, hi = (float(v) for v in np.percentile(diffs, [2.5, 97.5]))
    return {
        "n_common_positions": len(common),
        "mean": float(np.mean(diffs)),
        "ci95": [lo, hi],
        "excludes_zero": bool(lo > 0 or hi < 0),
        "orientation": "rho(first_class) - rho(second_class)",
    }


def class_conditioning_summary(policies, ranked_records: list[dict], kind: str) -> dict:
    """Normalised class preference by the teacher's class rank.

    A model that learned the teacher's *judgement* shows preference falling as
    the teacher's rank for that class worsens. A blanket habit shows a flat
    profile and a Spearman rho near zero -- but only if the class actually has
    rank spread, which :func:`class_rank_variance_report` checks separately.
    """
    rows = []
    for record in ranked_records:
        board = chess.Board(record["fen"])
        values = [class_mass_over_uniform(policy, board, kind) for policy in policies]
        rows.append({
            "fen": record["fen"],
            "rank": int(record[f"{kind}_rank"]),
            "gap_cp": record[f"{kind}_gap_cp"],
            "class_mass_over_uniform": float(np.mean(values)),
        })
    buckets: dict[str, list[float]] = {}
    for row in rows:
        buckets.setdefault(rank_bucket(row["rank"]), []).append(row["class_mass_over_uniform"])
    finite = [r for r in rows if math.isfinite(r["class_mass_over_uniform"])]
    rho = (spearman([float(r["rank"]) for r in finite],
                    [r["class_mass_over_uniform"] for r in finite])
           if len(finite) >= 2 else float("nan"))
    ci = (bootstrap_spearman_ci([float(r["rank"]) for r in finite],
                                [r["class_mass_over_uniform"] for r in finite])
          if len(finite) >= 3 else [float("nan"), float("nan")])
    return {
        "class": kind,
        "positions": len(rows),
        "spearman_teacher_rank_vs_preference": rho if math.isfinite(rho) else None,
        "spearman_95pct_bootstrap_interval": ci,
        "spearman_interval_excludes_zero": bool(
            math.isfinite(ci[0]) and math.isfinite(ci[1])
            and (ci[0] > 0 or ci[1] < 0)),
        "buckets": {
            name: {"n": len(buckets[name]),
                   "class_mass_over_uniform": float(np.mean(buckets[name]))}
            for name in _BUCKET_ORDER if name in buckets
        },
        "rows": rows,
    }


def match_corpus_sizes(per_class: dict, cap: int, seed: int) -> dict:
    """Trim every class's RICH/POOR lists to a common size.

    Without this the two classes would train on different corpora (91 castling-
    rich positions exist, ~496 capture-rich ones), and any difference in
    conditioning would be confounded with training-set size. Selection is
    deterministic given ``seed`` so the arm is reproducible.
    """
    rng = np.random.default_rng(seed)
    out = {}
    for kind, (rich, poor) in per_class.items():
        chosen_rich = list(rich)
        chosen_poor = list(poor)
        if len(chosen_rich) > cap:
            idx = rng.choice(len(chosen_rich), size=cap, replace=False)
            chosen_rich = [chosen_rich[i] for i in sorted(idx)]
        if len(chosen_poor) > cap:
            idx = rng.choice(len(chosen_poor), size=cap, replace=False)
            chosen_poor = [chosen_poor[i] for i in sorted(idx)]
        out[kind] = (chosen_rich, chosen_poor)
    return out


def class_rank_variance_report(ranked: list[dict], kinds: tuple[str, ...]) -> dict:
    """Is there enough teacher-rank variance for the conditioning test to mean anything?

    A class whose best member is ranked 1 in essentially every position has no
    rank spread, so a Spearman correlation against it is undefined rather than
    zero. That distinction decides whether a flat rho is evidence of "does not
    condition" or evidence of "the instrument cannot see anything here", so it
    is reported explicitly rather than left implicit in a near-zero number.
    """
    report = {}
    for kind in kinds:
        ranks = [r[f"{kind}_rank"] for r in ranked if r.get(f"{kind}_legal")]
        gaps = [r[f"{kind}_gap_cp"] for r in ranked
                if r.get(f"{kind}_legal") and r.get(f"{kind}_gap_cp") is not None]
        if not ranks:
            report[kind] = {"positions_class_legal": 0}
            continue
        counts: dict[int, int] = {}
        for rank in ranks:
            counts[int(rank)] = counts.get(int(rank), 0) + 1
        report[kind] = {
            "positions_class_legal": len(ranks),
            "distinct_ranks": len(counts),
            "rank_1_share": counts.get(1, 0) / len(ranks),
            "rank_histogram": {str(k): counts[k] for k in sorted(counts)},
            "gap_cp_distinct_values": len(set(gaps)),
            "gap_cp_zero_share": (sum(1 for g in gaps if g == 0) / len(gaps)) if gaps else None,
            "usable_rank_variance": len(counts) > 1 and counts.get(1, 0) / len(ranks) < 0.95,
        }
    return report


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",
                        choices=("probe", "static", "train", "conditioning", "probe_features",
                                 "probe_classes", "conditioning2"),
                        default="probe")
    parser.add_argument("--games", type=int, default=200,
                        help="random games to scan for castling-legal positions")
    parser.add_argument("--min-since-capture", type=int, default=4)
    parser.add_argument("--source", choices=("random", "quiet"), default="random",
                        help="position generator: random play, or quiet (opening-like) play")
    parser.add_argument("--scan-seed", type=int, default=271828)
    parser.add_argument("--depth", type=int, default=3, help="oracle search depth (d3)")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--oracle", choices=("d3", "stockfish"), default="d3",
                        help="label source for the census")
    parser.add_argument("--engine", default=None, help="UCI engine path (stockfish oracle)")
    parser.add_argument("--oracle-depth", type=int, default=12, help="engine search depth")
    parser.add_argument("--levels", default="random,L1-blind,material,L3-untrained",
                        help="levels for the static census (add L2-d3 to include the slow search)")
    parser.add_argument("--heldout-positions", type=int, default=128)
    parser.add_argument("--train-positions", type=int, default=250)
    parser.add_argument("--seeds", default="101,102,103,104,105,106,107,108")
    parser.add_argument("--labels", default="bench/h23_behavioural_static.json",
                        help="cached labelled positions (train mode)")
    parser.add_argument("--heldout-fraction", type=float, default=0.34)
    parser.add_argument("--split-seed", type=int, default=9001)
    parser.add_argument("--search-weights", default="1.0,0.125",
                        help="comma-separated search_weight values for the trained arm")
    parser.add_argument("--epochs", type=int, default=4,
                        help="passes over the (small) castling-rich corpus")
    parser.add_argument("--multipv", type=int, default=20,
                        help="engine MultiPV width for the conditioning rank measurement")
    parser.add_argument("--move-classes", default="castling,capture",
                        help="comma-separated move classes for probe_classes/conditioning2")
    parser.add_argument("--max-per-class", type=int, default=10_000,
                        help="cap on RICH/POOR corpus size per class, so the classes are matched")
    parser.add_argument("--class-ranks", default="bench/h23_class_rank_variance.json",
                        help="cached per-class teacher ranks from --mode probe_classes")
    parser.add_argument("--out", default=None)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    started = time.perf_counter()

    if args.mode in ("train", "conditioning", "probe_features", "probe_classes",
                     "conditioning2"):
        labels_path = Path(args.labels)
        if not labels_path.exists():
            print(f"{args.mode} mode needs cached labels at {labels_path}; run --mode static first")
            return 1
        labeled = json.loads(labels_path.read_text(encoding="utf-8"))["positions"]
        candidates, both, one = [], [], []
        print(f"loaded {len(labeled)} labelled positions from {labels_path}", flush=True)
    else:
        print(f"scanning {args.games} {args.source}-play games for castling-legal positions ...",
              flush=True)
        candidates = scan_castling_positions(
            args.games, args.scan_seed, min_since_capture=args.min_since_capture,
            source=args.source,
        )
        both = [c for c in candidates if c["n_castles"] == 2]
        one = [c for c in candidates if c["n_castles"] == 1]
        print(f"  castling-legal positions: {len(candidates)} "
              f"(both wings {len(both)}, one wing {len(one)})", flush=True)
        if args.mode == "probe":
            payload = {
                "mode": "probe",
                "games_scanned": args.games,
                "candidates": len(candidates),
                "both_wings": len(both),
                "one_wing": len(one),
                "min_since_capture": args.min_since_capture,
                "runtime_seconds": round(time.perf_counter() - started, 3),
            }
            print(json.dumps(payload, indent=2))
            return 0
        if not candidates:
            print("no castling-legal positions found; increase --games")
            return 1
        if args.oracle == "stockfish":
            if not args.engine:
                print("--oracle stockfish needs --engine <path>")
                return 1
            label_fn = stockfish_label_fn(args.engine, args.oracle_depth, args.top_k)
        else:
            label_fn = d3_label_fn(args.depth, args.top_k)
        print(f"labelling with {args.oracle} (depth "
              f"{args.oracle_depth if args.oracle == 'stockfish' else args.depth}) ...", flush=True)
        labeled = oracle_label(candidates, label_fn)
        getattr(label_fn, "close", lambda: None)()

    n_best = sum(r["oracle_best_is_castle"] for r in labeled)
    n_topk = sum(r["oracle_topk_has_castle"] for r in labeled)
    print(f"  oracle best is castling: {n_best}/{len(labeled)}; "
          f"castling in top-{args.top_k}: {n_topk}/{len(labeled)}", flush=True)

    # Cache the labels immediately: labelling is the expensive step, and a later
    # failure must not discard it. (This happened once -- ~11 minutes of depth-3
    # search lost to a printer bug, because output was only written at the end.)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"oracle": args.oracle, "positions": labeled}, indent=2),
                       encoding="utf-8")
        print(f"  cached labels -> {out}", flush=True)

    payload = {
        "mode": args.mode,
        "oracle": args.oracle,
        "source": args.source,
        "games_scanned": args.games,
        "candidates": len(candidates),
        "both_wings": len(both),
        "one_wing": len(one),
        "oracle_best_is_castle": n_best,
        "oracle_topk_has_castle": n_topk,
        "depth": args.depth,
        "oracle_depth": args.oracle_depth if args.oracle == "stockfish" else args.depth,
        "top_k": args.top_k,
        "min_since_capture": args.min_since_capture,
    }

    if args.mode == "probe_features":
        probe = run_feature_probe(labeled, split_seed=args.split_seed,
                                  heldout_fraction=args.heldout_fraction)
        print("\n=== feature probe: can the castling logit's inputs separate "
              "RICH from POOR? ===")
        print(f"  {'features':<20} {'dims':>6} {'held-acc':>9} {'majority':>9} "
              f"{'held-AUC':>17} {'n+':>4} {'pred+':>7}")
        for name, row in probe.items():
            lo, hi = row["heldout_auc_95pct_bootstrap_interval"]
            auc_s = f"{row['heldout_auc']:.3f} [{lo:.2f},{hi:.2f}]"
            print(f"  {name:<20} {row['features']:>6} {row['heldout_accuracy']:>9.3f} "
                  f"{row['majority_baseline']:>9.3f} {auc_s:>17} "
                  f"{row['heldout_n_positive']:>4} "
                  f"{row['heldout_predicted_positive_rate']:>7.3f}")
        payload["feature_probe"] = probe
    elif args.mode == "static":
        levels = {name: LEVELS[name] for name in args.levels.split(",") if name.strip()}
        unknown = set(levels) - set(LEVELS)
        if unknown:
            print(f"unknown level(s): {sorted(unknown)}; choose from {sorted(LEVELS)}")
            return 1
        report = run_static(labeled, levels)
        print("\n=== static census (registered command) ===")
        for subset_name, block in report.items():
            if block["positions"] == 0:
                print(f"\n[{subset_name}] no positions")
                continue
            print(f"\n[{subset_name}] n={block['positions']}  "
                  f"(uniform no-preference castling mass "
                  f"{block['uniform_baseline_castling_mass']:.4f})")
            print(f"  {'level':<14} {'select-castle':>14} {'castle-mass':>12}")
            for name, row in block["levels"].items():
                mass = row.get("castling_probability_mass")
                mass_s = "n/a" if mass is None else f"{mass:.4f}"
                print(f"  {name:<14} {row['select_castle_rate']:>14.4f} {mass_s:>12}")
        payload["static_census"] = report
        payload["positions"] = labeled
    elif args.mode == "train":
        seeds = [int(v) for v in args.seeds.split(",") if v.strip()]
        weights = [float(v) for v in args.search_weights.split(",") if v.strip()]
        # RICH = the teacher is *willing to castle* here, i.e. castling is among
        # the top-k targets the learner is actually shown. "Castling is the
        # single best move" is too rare to build a corpus from (7/583 under
        # Stockfish, 8/583 under L2-d3) -- castling is usually a good move, not
        # the best one. POOR = castling legal but never targeted.
        rich = [r for r in labeled if r["oracle_topk_has_castle"]]
        poor = [r for r in labeled if not r["oracle_topk_has_castle"]]
        corpora = build_corpora(rich, poor, args.heldout_fraction, args.split_seed)
        # Evaluate on BOTH held-out sets. `rich_held` = the teacher wants to
        # castle; `poor_held` = castling legal but the teacher does not. An arm
        # that castles on both has learned a blanket bias; one that castles only
        # on `rich_held` has learned a position-conditional decision.
        eval_sets = {"rich_held": corpora["rich_held"], "poor_held": corpora["poor_held"]}
        print(f"  RICH (castling in top-{args.top_k}): {len(rich)} -> "
              f"train {len(corpora['rich_train'])} / held {len(corpora['rich_held'])}")
        print(f"  POOR (castling legal, never targeted): {len(poor)} -> "
              f"train {len(corpora['poor_train'])} / held {len(corpora['poor_held'])}")
        print(f"  MIXED_half (volume-matched to RICH): {len(corpora['mixed_half_train'])}")
        print(f"  MIXED_full (RICH dose matched): {len(corpora['mixed_full_train'])}")
        arms = {}
        for weight in weights:
            for arm_name, corpus_key, neg in (
                ("rich", "rich_train", False),
                ("poor", "poor_train", False),
                ("mixed_half", "mixed_half_train", False),
                ("mixed_full", "mixed_full_train", False),
                # Same corpus as mixed_half, plus an explicit negative gradient
                # for castling where the teacher rejected it. Any difference
                # against mixed_half is attributable to the negative class.
                ("mixed_half+neg", "mixed_half_train", True),
                ("mixed_full+neg", "mixed_full_train", True),
            ):
                arms[f"{arm_name}@sw{weight:g}"] = train_and_evaluate(
                    corpora[corpus_key], eval_sets, seeds, depth=args.depth,
                    top_k=args.top_k, search_weight=weight, epochs=args.epochs,
                    neg_castling=neg)
        print("\n=== trained arm (castling mass; untrained baseline in brackets) ===")
        print(f"  {'arm':<18} {'rich_held':>20} {'poor_held':>20} {'discrimination':>22}")
        for name, arm in arms.items():
            cells = []
            for eval_name in ("rich_held", "poor_held"):
                s = arm["summary"][eval_name]["castling_probability_mass"]
                cells.append(f"{s['trained_mean']:.4f} ({s['untrained_mean']:.4f})")
            d = arm["summary"]["discrimination"]
            disc = (f"{d['trained_mean']:+.4f} "
                    f"[{d['paired_gain_95pct_t_interval'][0]:+.3f},"
                    f"{d['paired_gain_95pct_t_interval'][1]:+.3f}]")
            print(f"  {name:<18} {cells[0]:>20} {cells[1]:>20} {disc:>22}")
        payload["train_arms"] = arms
        payload["corpora"] = {
            "rich_total": len(rich), "rich_train": len(corpora["rich_train"]),
            "rich_held": len(corpora["rich_held"]),
            "poor_total": len(poor), "poor_train": len(corpora["poor_train"]),
            "poor_held": len(corpora["poor_held"]),
            "mixed_half_train": len(corpora["mixed_half_train"]),
            "mixed_full_train": len(corpora["mixed_full_train"]),
            "seeds": seeds, "search_weights": weights, "epochs": args.epochs,
            "heldout_fraction": args.heldout_fraction, "split_seed": args.split_seed,
        }
    elif args.mode == "conditioning2":
        # The two-class test. Same positions, same teacher, same seeds, same
        # update path; the only thing that varies is which move class the
        # corpus targets. The three-way read is pre-registered:
        #   captures condition + castling does not -> the defect is board-wide
        #       context, and the tier-2 entry is aimed correctly;
        #   neither conditions -> the defect is in the training signal, and the
        #       tier-2 entry is aimed at the wrong thing;
        #   both condition -> castling's flatness was a corpus artefact and the
        #       tier-2 gate should not have been opened on it.
        ranks_path = Path(args.class_ranks)
        if not ranks_path.exists():
            print(f"conditioning2 needs cached class ranks at {ranks_path}; "
                  f"run --mode probe_classes first")
            return 1
        ranked_all = json.loads(ranks_path.read_text(encoding="utf-8"))["positions"]
        by_fen = {r["fen"]: r for r in ranked_all}
        seeds = [int(v) for v in args.seeds.split(",") if v.strip()]
        weights = [float(v) for v in args.search_weights.split(",") if v.strip()]
        kinds = tuple(v for v in args.move_classes.split(",") if v.strip())

        # ONE shared held-out set for every class. Splitting separately per class
        # (as the castling-only path does) leaves the classes evaluated on almost
        # disjoint positions -- measured overlap was 4 of 48 -- which makes the
        # paired comparison of their conditioning meaningless. Training corpora
        # still differ by class, because that is the variable under test.
        common_legal = []
        for record in labeled:
            board = chess.Board(record["fen"])
            if all(class_moves(board, kind) for kind in kinds):
                common_legal.append(record)
        pool_train, pool_held = split_records(common_legal, args.heldout_fraction,
                                              args.split_seed)
        ranked = [by_fen[r["fen"]] for r in pool_held if r["fen"] in by_fen]
        print(f"  common-legal {len(common_legal)} -> shared train "
              f"{len(pool_train)} / held {len(pool_held)} (ranked {len(ranked)})",
              flush=True)

        per_class = {}
        for kind in kinds:
            rich, poor = [], []
            for record in pool_train:
                board = chess.Board(record["fen"])
                (rich if class_in_targets(board, record, kind) else poor).append(record)
            per_class[kind] = (rich, poor)
        available = {k: (len(r), len(p)) for k, (r, p) in per_class.items()}
        cap = min([min(r, p) for r, p in available.values()] + [args.max_per_class])
        print(f"  available RICH/POOR per class: "
              f"{ {k: v for k, v in available.items()} }; matched cap {cap}", flush=True)
        per_class = match_corpus_sizes(per_class, cap, args.split_seed)

        payload_arms = {}
        for kind in kinds:
            rich, poor = per_class[kind]
            print(f"  [{kind}] rich train {len(rich)} / poor train {len(poor)}",
                  flush=True)
            for weight in weights:
                policies = train_policies(
                    rich, seeds, depth=args.depth, top_k=args.top_k,
                    search_weight=weight, epochs=args.epochs)
                summary = class_conditioning_summary(policies, ranked, kind)
                payload_arms[f"{kind}@sw{weight:g}"] = summary
                print(f"    {kind}@sw{weight:g}: rho="
                      f"{summary['spearman_teacher_rank_vs_preference']:+.3f}", flush=True)

        print("\n=== two-class conditioning: preference (mass / uniform) by teacher rank ===")
        print(f"  {'arm':<18} " + " ".join(f"{b:>7}" for b in _BUCKET_ORDER)
              + f" {'rho':>7}")
        for name, arm in payload_arms.items():
            cells = []
            for bucket in _BUCKET_ORDER:
                entry = arm["buckets"].get(bucket)
                cells.append(f"{entry['class_mass_over_uniform']:.2f}" if entry else "-")
            rho = arm["spearman_teacher_rank_vs_preference"]
            lo, hi = arm["spearman_95pct_bootstrap_interval"]
            flag = "" if arm["spearman_interval_excludes_zero"] else "  (includes 0)"
            print(f"  {name:<18} " + " ".join(f"{c:>7}" for c in cells)
                  + f" {(rho if rho is not None else float('nan')):+7.3f}"
                  + f"  [{lo:+.3f},{hi:+.3f}]{flag}")
        # Primary endpoint: the paired difference in rho between the classes.
        # A per-class rho alone cannot say whether castling is *worse at
        # conditioning* than a locally-decidable class -- that needs the
        # difference and its interval.
        paired = {}
        if len(kinds) == 2:
            first, second = kinds
            for weight in weights:
                key = f"sw{weight:g}"
                paired[key] = paired_spearman_difference(
                    payload_arms[f"{first}@{key}"]["rows"],
                    payload_arms[f"{second}@{key}"]["rows"],
                )
                row = paired[key]
                print(f"  paired rho({first}) - rho({second}) @ {key}: "
                      f"{row['mean']:+.3f} [{row['ci95'][0]:+.3f},{row['ci95'][1]:+.3f}]"
                      f"  n={row['n_common_positions']}"
                      f"  {'excludes 0' if row['excludes_zero'] else 'includes 0'}",
                      flush=True)
        payload["two_class_conditioning"] = payload_arms
        payload["two_class_paired_difference"] = paired
        payload["two_class_meta"] = {
            "kinds": list(kinds), "matched_cap": cap,
            "available_rich_poor": available,
            "shared_heldout_positions": len(ranked),
            "heldout_is_shared_across_classes": True,
            "seeds": seeds, "search_weights": weights, "epochs": args.epochs,
            "multipv": args.multipv, "engine_depth": args.oracle_depth,
            "bucket_order": list(_BUCKET_ORDER),
            "metric": "class mass divided by the no-preference share (n_class/n_legal)",
            "predeclared_read": {
                "captures_condition_castling_does_not": "defect is board-wide context",
                "neither_conditions": "defect is the training signal, not the receptive field",
                "both_condition": "castling flatness was a corpus artefact",
            },
        }
    elif args.mode == "probe_classes":
        # Gate the two-class comparison on instrument validity BEFORE running it.
        # If a class's best member is ranked 1 in nearly every position there is
        # no rank spread to correlate against, and a flat rho would be an
        # artefact of the instrument rather than a finding about the learner.
        if not args.engine:
            print("probe_classes mode needs --engine <path>")
            return 1
        kinds = tuple(v for v in args.move_classes.split(",") if v.strip())
        print(f"  measuring class ranks (multipv {args.multipv}, depth "
              f"{args.oracle_depth}) for {len(labeled)} positions ...", flush=True)
        ranked = compute_class_ranks(labeled, args.engine, args.oracle_depth,
                                     args.multipv, kinds)
        report = class_rank_variance_report(ranked, kinds)
        print("\n=== is there usable teacher-rank variance per move class? ===")
        for kind, row in report.items():
            if not row.get("positions_class_legal"):
                print(f"  {kind:<10} class legal in 0 positions")
                continue
            print(f"  {kind:<10} legal {row['positions_class_legal']:>4}  "
                  f"distinct ranks {row['distinct_ranks']:>3}  "
                  f"rank-1 share {row['rank_1_share']:.3f}  "
                  f"gap-cp distinct {row['gap_cp_distinct_values']:>4}  "
                  f"usable {row['usable_rank_variance']}")
            print(f"             rank histogram: {row['rank_histogram']}")
        payload["class_rank_variance"] = report
        # Cache the ranked records, not just the summary: this engine pass is
        # the expensive step (~6 min for 583 positions at depth 12 / MultiPV 20)
        # and conditioning2 consumes the per-position ranks. Writing only the
        # summary threw that work away and forced a re-measurement.
        payload["positions"] = ranked
    else:  # conditioning
        if not args.engine:
            print("conditioning mode needs --engine <path>")
            return 1
        seeds = [int(v) for v in args.seeds.split(",") if v.strip()]
        weights = [float(v) for v in args.search_weights.split(",") if v.strip()]
        rich = [r for r in labeled if r["oracle_topk_has_castle"]]
        poor = [r for r in labeled if not r["oracle_topk_has_castle"]]
        corpora = build_corpora(rich, poor, args.heldout_fraction, args.split_seed)
        held = corpora["rich_held"] + corpora["poor_held"]
        # Cache the ranks: this is the expensive step (one engine pass per
        # position at depth 12 x multipv 20), and it is arm-independent, so it
        # must be computed once and reused across every arm.
        ranks_path = Path(args.out).with_suffix("") if args.out else None
        ranks_path = Path(str(ranks_path) + ".ranks.json") if ranks_path else None
        ranked = None
        if ranks_path and ranks_path.exists():
            cached = json.loads(ranks_path.read_text(encoding="utf-8"))
            if cached.get("meta", {}).get("multipv") == args.multipv and \
               cached.get("meta", {}).get("depth") == args.oracle_depth and \
               len(cached.get("rows", [])) == len(held):
                ranked = cached["rows"]
                print(f"  reused cached ranks -> {ranks_path}", flush=True)
        if ranked is None:
            print(f"  measuring teacher castling rank (multipv {args.multipv}, "
                  f"depth {args.oracle_depth}) for {len(held)} positions ...", flush=True)
            ranked = compute_castling_ranks(held, args.engine, args.oracle_depth, args.multipv)
            if ranks_path:
                ranks_path.write_text(json.dumps({
                    "meta": {"multipv": args.multipv, "depth": args.oracle_depth,
                             "engine": args.engine},
                    "rows": ranked,
                }, indent=2), encoding="utf-8")
                print(f"  cached ranks -> {ranks_path}", flush=True)
        arms = {}
        for weight in weights:
            for arm_name, corpus_key in (
                ("rich", "rich_train"),
                ("mixed_half", "mixed_half_train"),
                ("mixed_full", "mixed_full_train"),
            ):
                print(f"  training {arm_name}@sw{weight:g} "
                      f"({len(corpora[corpus_key])} positions)", flush=True)
                policies = train_policies(
                    corpora[corpus_key], seeds, depth=args.depth, top_k=args.top_k,
                    search_weight=weight, epochs=args.epochs)
                arms[f"{arm_name}@sw{weight:g}"] = conditioning_summary(policies, ranked)
        print("\n=== conditioning: castling mass by teacher castling rank ===")
        print(f"  {'arm':<18} " + " ".join(f"{b:>7}" for b in _BUCKET_ORDER) + f" {'rho':>7}")
        for name, arm in arms.items():
            cells = []
            for bucket in _BUCKET_ORDER:
                entry = arm["buckets"].get(bucket)
                cells.append(f"{entry['castling_mass']:.3f}" if entry else "-")
            print(f"  {name:<18} " + " ".join(f"{c:>7}" for c in cells)
                  + f" {arm['spearman_teacher_rank_vs_castling_mass']:+7.3f}")
        payload["conditioning"] = arms
        payload["conditioning_meta"] = {
            "rich_train": len(corpora["rich_train"]),
            "mixed_half_train": len(corpora["mixed_half_train"]),
            "mixed_full_train": len(corpora["mixed_full_train"]),
            "held": len(held), "rich_held": len(corpora["rich_held"]),
            "poor_held": len(corpora["poor_held"]),
            "seeds": seeds, "search_weights": weights, "epochs": args.epochs,
            "multipv": args.multipv, "engine_depth": args.oracle_depth,
            "bucket_order": list(_BUCKET_ORDER),
        }

    payload["provenance"] = {
        "perceptron_py_sha256": hashlib.sha256(
            (_REPO / "src/chessrl/perceptron.py").read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(_HERE.read_bytes()).hexdigest(),
        "runtime_seconds": round(time.perf_counter() - started, 3),
    }
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
