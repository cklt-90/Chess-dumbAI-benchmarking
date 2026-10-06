"""Matched held-out pilot for L3 supervised imitation learning.

The script generates one fixed training corpus and one disjoint held-out set,
labels each exactly once with the same depth-limited search, and reuses those
labels across independent model-initialisation seeds. Checkpoints are indexed
by training positions seen. This is an exploratory/power-planning pilot, not a
strength test or a confirmatory result.

Example (small wiring smoke):
    ./.venv/Scripts/python.exe bench/scripts/l3_heldout_diagnostic.py \
        --train-positions 4 --heldout-positions 4 --seeds 1,2 \
        --checkpoints 2,4 --depth 2 --out bench/l3_heldout_smoke.json

Example (variance pilot):
    ./.venv/Scripts/python.exe bench/scripts/l3_heldout_diagnostic.py \
        --train-positions 250 --heldout-positions 128 --seeds 101,102,103,104,105,106,107,108 \
        --checkpoints 125,250 --depth 3 --mde 0.05 \
        --out bench/l3_heldout_gradient_pilot.json
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

from chessrl.game import RandomPolicy, play_game
from chessrl.masks import move_to_index
from chessrl.perceptron import L3Config, L3Policy, L3Trainer


_WEIGHT_NAMES = ("W_from", "b_from", "W_ctx", "W_hid", "W_to_dst", "W_to_src", "W_geom", "promo")


def _position_key(board: chess.Board) -> tuple:
    """Identify equivalent search states while ignoring the fullmove number."""
    return (
        board.board_fen(),
        board.turn,
        int(board.castling_rights),
        board.ep_square,
        board.halfmove_clock,
    )


def collect_positions(count: int, seed: int, excluded: set[tuple]) -> list[dict]:
    """Collect one nonterminal, unique position from each random-play game."""
    rng = random.Random(seed)
    seen = set(excluded)
    records: list[dict] = []
    game_id = 0
    max_games = max(100, count * 5)
    while len(records) < count and game_id < max_games:
        game = play_game(
            RandomPolicy(rng.randrange(1 << 31)),
            RandomPolicy(rng.randrange(1 << 31)),
            max_plies=120,
            no_progress_limit=60,
            rng=rng,
            record_fens=True,
        )
        candidates = []
        for fen in game.fens:
            board = chess.Board(fen)
            if board.is_game_over():
                continue
            key = _position_key(board)
            if key not in seen:
                candidates.append((fen, key))
        if candidates:
            fen, key = rng.choice(candidates)
            seen.add(key)
            records.append({"fen": fen, "source_game": game_id})
        game_id += 1
    if len(records) != count:
        raise RuntimeError(f"collected {len(records)} of {count} unique positions")
    return records


def label_positions(records: list[dict], depth: int, top_k: int) -> list[dict]:
    """Cache the exact teacher targets once, shared by all training seeds."""
    labeler = L3Trainer(L3Config(seed=0, search_depth=depth, top_k=top_k))
    labeled = []
    for record in records:
        board = chess.Board(record["fen"])
        targets = labeler.search_feedback(board, depth=depth, top_k=top_k)
        if not targets:
            raise AssertionError("nonterminal position produced no teacher targets")
        labeled.append({
            **record,
            "targets": [{"uci": move.uci(), "weight": float(weight)}
                        for move, weight in targets],
        })
    return labeled


def _apply_cached_targets(trainer: L3Trainer, record: dict) -> None:
    """Apply cached targets with train_on_search_feedback's current semantics."""
    board = chess.Board(record["fen"])
    targets = [(chess.Move.from_uci(item["uci"]), float(item["weight"]))
               for item in record["targets"]]
    trainer.policy.scorer.set_position(board)
    distribution = trainer.policy.move_distribution(board)
    best_weight = max(weight for _, weight in targets)
    for move, weight in targets:
        model_probability = float(distribution[move_to_index(move)])
        target_probability = weight / best_weight
        gap = target_probability - model_probability
        trainer.apply_credit(
            board, move, trainer.config.search_weight * gap
        )
        trainer.search_updates += 1


def assert_cached_update_matches_native(
    record: dict, depth: int, top_k: int, seed: int
) -> None:
    """Check cached-label replay against the production update path."""
    config = L3Config(seed=seed, search_depth=depth, top_k=top_k)
    native = L3Trainer(config=config)
    cached = L3Trainer(config=L3Config(seed=seed, search_depth=depth, top_k=top_k))
    board = chess.Board(record["fen"])
    native_targets = native.search_feedback(board, depth=depth, top_k=top_k)
    cached_targets = [
        (chess.Move.from_uci(item["uci"]), float(item["weight"]))
        for item in record["targets"]
    ]
    if len(native_targets) != len(cached_targets) or any(
        native_move != cached_move
        or not math.isclose(native_weight, cached_weight, rel_tol=1e-12, abs_tol=1e-12)
        for (native_move, native_weight), (cached_move, cached_weight)
        in zip(native_targets, cached_targets)
    ):
        raise AssertionError("cached teacher targets differ from fresh search labels")
    native.train_on_search_feedback(board, depth=depth)
    _apply_cached_targets(cached, record)
    for name in _WEIGHT_NAMES:
        left = getattr(native.policy.perceptron, name)
        right = getattr(cached.policy.perceptron, name)
        if not np.array_equal(left, right):
            raise AssertionError(f"cached update differs from native on {name}")
    if native.search_updates != cached.search_updates:
        raise AssertionError("cached update changed the search-update count")


def evaluate(policy: L3Policy, heldout: list[dict]) -> dict:
    topk_mass = []
    weighted_probability = []
    weighted_cross_entropy = []
    topk_hit = []
    teacher_best_hit = []
    entropy = []
    for record in heldout:
        board = chess.Board(record["fen"])
        distribution = policy.move_distribution(board)
        target_rows = record["targets"]
        targets = [chess.Move.from_uci(item["uci"]) for item in target_rows]
        target_indices = {move_to_index(move) for move in targets}
        target_weights = np.asarray([item["weight"] for item in target_rows], dtype=np.float64)
        normalized_weights = target_weights / target_weights.sum()
        best_weight = float(target_weights.max())
        best_target_indices = {
            move_to_index(move)
            for move, weight in zip(targets, target_weights)
            if math.isclose(float(weight), best_weight, rel_tol=1e-12, abs_tol=1e-12)
        }
        legal = list(board.legal_moves)
        legal_probabilities = np.asarray(
            [distribution[move_to_index(move)] for move in legal], dtype=np.float64
        )
        total = float(legal_probabilities.sum())
        if not math.isfinite(total) or not math.isclose(total, 1.0, abs_tol=1e-6):
            raise AssertionError(f"legal move probabilities sum to {total}")
        predicted_best = float(legal_probabilities.max())
        predicted_best_indices = {
            move_to_index(move)
            for move, probability in zip(legal, legal_probabilities)
            if math.isclose(float(probability), predicted_best, rel_tol=1e-12, abs_tol=1e-12)
        }
        positive = legal_probabilities[legal_probabilities > 0]
        entropy.append(float(-np.sum(positive * np.log(positive))))
        target_probabilities = np.asarray(
            [distribution[move_to_index(move)] for move in targets], dtype=np.float64
        )
        topk_mass.append(float(sum(distribution[index] for index in target_indices)))
        weighted_probability.append(float(np.dot(normalized_weights, target_probabilities)))
        weighted_cross_entropy.append(float(-np.dot(
            normalized_weights, np.log(np.maximum(target_probabilities, 1e-12))
        )))
        topk_hit.append(float(bool(predicted_best_indices & target_indices)))
        teacher_best_hit.append(float(bool(predicted_best_indices & best_target_indices)))
    return {
        "teacher_topk_probability_mass": float(np.mean(topk_mass)),
        "teacher_weighted_probability_score": float(np.mean(weighted_probability)),
        "teacher_weighted_cross_entropy_nats": float(np.mean(weighted_cross_entropy)),
        "policy_top1_in_teacher_topk_rate_tie_aware": float(np.mean(topk_hit)),
        "policy_top1_in_teacher_best_set_rate_tie_aware": float(np.mean(teacher_best_hit)),
        "mean_legal_move_entropy_nats": float(np.mean(entropy)),
        "positions": len(heldout),
    }


_T_CRIT_95_DF_1_TO_30 = (
    12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
    2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
    2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042,
)


def _t_critical_95(seed_count: int) -> float:
    degrees_freedom = seed_count - 1
    if degrees_freedom < 1:
        raise ValueError("at least two seeds are needed for an interval")
    if degrees_freedom <= len(_T_CRIT_95_DF_1_TO_30):
        return _T_CRIT_95_DF_1_TO_30[degrees_freedom - 1]
    return 2.05  # conservative approximation to the 95% two-sided t critical value


def _mean_sd(values: list[float]) -> tuple[float, float]:
    if not values:
        raise ValueError("cannot summarize an empty seed sample")
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return mean, sd


def _approximate_seed_count(sd: float, mde: float) -> int | None:
    """Planning approximation: t critical value plus normal 80% power quantile."""
    if sd == 0.0:
        # A zero pilot SD is not evidence of zero population variability and
        # cannot support a data-driven sample-size estimate.
        return None
    for seed_count in range(2, 10001):
        detectable = math.sqrt(seed_count) * mde / sd
        if detectable >= _t_critical_95(seed_count) + 0.84:
            return seed_count
    raise RuntimeError("sample-size approximation exceeded 10,000 seeds")


def _summarise(seed_results: list[dict], final_checkpoint: int, mde: float) -> dict:
    baseline = [row["checkpoints"][0]["metrics"]["teacher_topk_probability_mass"]
                for row in seed_results]
    final = [row["checkpoints"][-1]["metrics"]["teacher_topk_probability_mass"]
             for row in seed_results]
    paired = [after - before for before, after in zip(baseline, final)]
    mean_gain, sd_gain = _mean_sd(paired)
    seed_count = len(paired)
    standard_error = sd_gain / math.sqrt(seed_count)
    t_critical = _t_critical_95(seed_count)
    ci_half_width = t_critical * standard_error
    # This is a planning approximation (t critical plus normal 80% power
    # quantile), not an exact noncentral-t power calculation.
    required_seeds = _approximate_seed_count(sd_gain, mde)
    return {
        "primary_endpoint": "mean held-out probability mass on the teacher's listed top-k moves at final checkpoint",
        "final_positions_seen": final_checkpoint,
        "training_seed_replications": seed_count,
        "mean_seed_paired_gain": mean_gain,
        "sd_seed_paired_gain": sd_gain,
        "standard_error_across_seeds": standard_error,
        "approx_95pct_t_interval_for_mean_gain": [
            mean_gain - ci_half_width, mean_gain + ci_half_width
        ],
        "minimum_meaningful_effect": mde,
        "approx_seed_count_for_80pct_power_5pct_two_sided": required_seeds,
        "power_planning_method": "t critical value plus normal 80% power quantile; approximate, not exact noncentral-t",
        "interval_scope": (
            "conditional on this fixed training corpus and held-out set; estimates "
            "initialization-seed variability only"
        ),
        "power_planning_scope": (
            "conditional on the fixed corpora; excludes training-corpus and held-out-position "
            "sampling uncertainty; refine with an exact noncentral-t method before confirmation"
        ),
        "seed_level_paired_differences": paired,
        "interpretation": "exploratory variance pilot only; no confirmatory claim",
    }


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-positions", type=int, default=250)
    parser.add_argument("--heldout-positions", type=int, default=128)
    parser.add_argument("--seeds", default="101,102,103,104,105,106,107,108")
    parser.add_argument("--checkpoints", default="125,250")
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--data-seed", type=int, default=314159)
    parser.add_argument("--mde", type=float, default=0.05,
                        help="predeclared absolute gain in teacher-top-k mass")
    parser.add_argument("--out", default="bench/l3_heldout_gradient_pilot.json")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    checkpoints = sorted(set(int(value) for value in args.checkpoints.split(",") if value.strip()))
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("provide at least two unique training seeds for variance estimation")
    if args.train_positions <= 0 or args.heldout_positions <= 0:
        raise ValueError("position counts must be positive")
    if args.depth < 1 or args.top_k < 1:
        raise ValueError("search depth and top-k must be positive")
    if not checkpoints or checkpoints[-1] != args.train_positions:
        raise ValueError("checkpoints must end at --train-positions")
    if checkpoints[0] <= 0 or any(value < 1 for value in checkpoints):
        raise ValueError("checkpoints must be positive positions-seen counts")
    if not (0 < args.mde <= 1):
        raise ValueError("--mde must be in (0, 1]")

    started = time.perf_counter()
    train_records = collect_positions(args.train_positions, args.data_seed, set())
    train_keys = {_position_key(chess.Board(record["fen"])) for record in train_records}
    heldout_records = collect_positions(
        args.heldout_positions, args.data_seed + 1, train_keys
    )
    training = label_positions(train_records, args.depth, args.top_k)
    heldout = label_positions(heldout_records, args.depth, args.top_k)

    # This executable parity guard makes sure the cached-label path has not
    # accidentally changed the learner's established target/update semantics.
    assert_cached_update_matches_native(training[0], args.depth, args.top_k, seeds[0])

    seed_results = []
    for seed in seeds:
        trainer = L3Trainer(config=L3Config(seed=seed, search_depth=args.depth, top_k=args.top_k))
        checkpoints_for_seed = [{"positions_seen": 0, "metrics": evaluate(trainer.policy, heldout)}]
        for position_count, record in enumerate(training, start=1):
            _apply_cached_targets(trainer, record)
            if position_count in checkpoints:
                checkpoints_for_seed.append({
                    "positions_seen": position_count,
                    "metrics": evaluate(trainer.policy, heldout),
                })
        if [item["positions_seen"] for item in checkpoints_for_seed] != [0, *checkpoints]:
            raise AssertionError("one or more position-seen checkpoints were missed")
        seed_results.append({"training_seed": seed, "checkpoints": checkpoints_for_seed})

    summary = _summarise(seed_results, checkpoints[-1], args.mde)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    source_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    model_sha = hashlib.sha256((_REPO / "src/chessrl/perceptron.py").read_bytes()).hexdigest()
    payload = {
        "design": {
            "purpose": "matched held-out supervised-learning variance pilot",
            "labeler": "L3Trainer.search_feedback",
            "teacher_topk_cutoff_ties": "uses listed top-k; ties inherit legal-move iteration order",
            "depth": args.depth,
            "top_k": args.top_k,
            "train_positions": args.train_positions,
            "heldout_positions": args.heldout_positions,
            "training_seeds": seeds,
            "checkpoints_positions_seen": [0, *checkpoints],
            "data_seed": args.data_seed,
            "same_cached_train_and_heldout_labels_across_seeds": True,
            "mde_used_for_planning": args.mde,
            "position_key_overlap": len(train_keys & {
                _position_key(chess.Board(record["fen"])) for record in heldout_records
            }),
            "primary_endpoint": summary["primary_endpoint"],
            "source_game_sampling": "one random nonterminal position per random-play game",
            "training_target_normalization": "unchanged: weight / max(target weights)",
        },
        "summary": summary,
        "seed_results": seed_results,
        "cached_training_labels": training,
        "cached_heldout_labels": heldout,
        "provenance": {
            "perceptron_py_sha256": model_sha,
            "diagnostic_script_sha256": source_sha,
            "runtime_seconds": round(time.perf_counter() - started, 3),
        },
    }
    if payload["design"]["position_key_overlap"] != 0:
        raise AssertionError("training and held-out position sets overlap")
    out.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"summary": summary, "runtime_seconds": payload["provenance"]["runtime_seconds"], "out": str(out)}, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
