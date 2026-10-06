"""Per-position diagnosis of the held-out disagreement in the L3 pilot.

The aggregate pilot (``l3_heldout_diagnostic.py``) showed, across training:

    teacher top-5 probability mass  0.19 -> 0.25   (up)
    teacher weighted cross-entropy  3.67 -> 4.49   (worse)
    mean legal-move entropy         3.13 -> 2.32   (sharper)

Aggregates cannot say whether the cross-entropy deterioration is broad or
driven by a subset of positions where the policy becomes over-confident in the
*wrong* move. This script answers that directly.

It does **not** recompute teacher labels. It reads the frozen corpus that the
pilot already cached (``cached_training_labels`` / ``cached_heldout_labels``),
replays the identical per-seed training trajectory, and at every checkpoint
records a per-held-out-position view:

  * chance on the teacher's listed top-k (mass on the listed set);
  * probability on the teacher's best move, tie-aware (the *set* of best moves,
    so score-tied teacher outputs are not spuriously punished);
  * weighted cross-entropy and policy entropy for that position;
  * the per-position change from baseline, so positions can be split into
    improved / unchanged / regressed and the worst regressions listed.

Scope: this is an exploratory readout of one frozen corpus and held-out set.
Seed-level intervals from the pilot are conditional on that corpus; nothing
here establishes that any effect generalises to freshly sampled positions, and
it must not be used to unlock a scale run. Interpretation only.

Example (reuse the pilot's cached corpus, full replay):
    ./.venv/Scripts/python.exe bench/scripts/l3_heldout_per_position.py \\
        --pilot bench/l3_heldout_gradient_pilot.json \\
        --out bench/l3_heldout_per_position.json

Example (smoke):
    ./.venv/Scripts/python.exe bench/scripts/l3_heldout_per_position.py \\
        --pilot bench/l3_heldout_gradient_pilot.json --max-positions 16 \\
        --out bench/l3_heldout_per_position_smoke.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
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

from chessrl.masks import move_to_index
from chessrl.perceptron import L3Config, L3Policy, L3Trainer

# Reuse the pilot's cached-label replay so the per-position view is guaranteed
# to sit on the same training trajectory, not a re-derived approximation.
from bench.scripts.l3_heldout_diagnostic import _apply_cached_targets, _position_key


def _position_metrics(policy: L3Policy, record: dict) -> dict:
    """Metrics for a single held-out position.

    Mirrors ``evaluate`` in the aggregate pilot field-by-field, but returns a
    dict per position instead of means. The teacher-best probability is
    tie-aware: if several moves share the teacher's top weight, the chance on
    the best move is the *sum* over that tied set, so a correct model is not
    penalised for choosing a co-optimal move.
    """
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
    target_probabilities = np.asarray(
        [distribution[move_to_index(move)] for move in targets], dtype=np.float64
    )

    legal = list(board.legal_moves)
    legal_probabilities = np.asarray(
        [distribution[move_to_index(move)] for move in legal], dtype=np.float64
    )
    total = float(legal_probabilities.sum())
    if not math.isfinite(total) or not math.isclose(total, 1.0, abs_tol=1e-6):
        raise AssertionError(f"legal move probabilities sum to {total}")

    positive = legal_probabilities[legal_probabilities > 0]
    entropy = float(-np.sum(positive * np.log(positive)))

    return {
        "fen": record["fen"],
        "teacher_topk_probability_mass": float(sum(distribution[i] for i in target_indices)),
        "teacher_best_probability_mass_tie_aware": float(
            sum(distribution[i] for i in best_target_indices)
        ),
        "teacher_weighted_probability_score": float(
            np.dot(normalized_weights, target_probabilities)
        ),
        "teacher_weighted_cross_entropy_nats": float(
            -np.dot(normalized_weights, np.log(np.maximum(target_probabilities, 1e-12)))
        ),
        "mean_legal_move_entropy_nats": entropy,
        "n_teacher_targets": len(target_rows),
        "n_teacher_best_moves": len(best_target_indices),
    }


# Metric names whose per-position delta is a "higher is better" quantity when
# positive and a regression when negative (same direction as the teacher).
_HIGHER_BETTER = (
    "teacher_topk_probability_mass",
    "teacher_best_probability_mass_tie_aware",
    "teacher_weighted_probability_score",
)
# Metrics where higher is *worse*.
_LOWER_BETTER = ("teacher_weighted_cross_entropy_nats",)


def summarise_positions(baseline: list[dict], final: list[dict]) -> dict:
    """Agregate per-position deltas into counts and worst-regression lists.

    ``baseline`` and ``final`` are aligned per-position metric dicts (same order,
    same held-out set). Counts treat a change as a move in the teacher's
    direction only when it exceeds a tiny numeric tolerance; ties are counted
    separately. All deltas are ``final - baseline``.
    """
    if len(baseline) != len(final):
        raise ValueError("baseline and final must cover the same positions")

    def _delta_rows(name: str) -> list[float]:
        return [f[name] - b[name] for b, f in zip(baseline, final)]

    tol = 1e-9
    out: dict = {"positions": len(baseline)}
    for name in _HIGHER_BETTER:
        deltas = _delta_rows(name)
        improved = sum(1 for d in deltas if d > tol)
        regressed = sum(1 for d in deltas if d < -tol)
        out[name] = {
            "mean_delta": float(np.mean(deltas)),
            "improved_positions": improved,
            "regressed_positions": regressed,
            "unchanged_positions": len(deltas) - improved - regressed,
            "mean_delta_over_improved": _mean_or_none([d for d in deltas if d > tol]),
            "mean_delta_over_regressed": _mean_or_none([d for d in deltas if d < -tol]),
        }
    for name in _LOWER_BETTER:
        deltas = _delta_rows(name)
        # For a lower-is-better metric, a *negative* delta is an improvement.
        improved = sum(1 for d in deltas if d < -tol)
        regressed = sum(1 for d in deltas if d > tol)
        out[name] = {
            "mean_delta": float(np.mean(deltas)),
            "improved_positions": improved,
            "regressed_positions": regressed,
            "unchanged_positions": len(deltas) - improved - regressed,
            "mean_delta_over_improved": _mean_or_none([d for d in deltas if d < -tol]),
            "mean_delta_over_regressed": _mean_or_none([d for d in deltas if d > tol]),
        }

    ent_deltas = _delta_rows("mean_legal_move_entropy_nats")
    out["mean_legal_move_entropy_nats"] = {
        "mean_delta": float(np.mean(ent_deltas)),
        "reduced_entropy_positions": sum(1 for d in ent_deltas if d < -tol),
        "increased_entropy_positions": sum(1 for d in ent_deltas if d > tol),
        "unchanged_positions": sum(1 for d in ent_deltas if abs(d) <= tol),
    }
    return out


def _mean_or_none(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _cross_entropy_concentration(baseline: list[dict], final: list[dict]) -> dict:
    """How much of the total cross-entropy increase sits in the worst positions.

    Answers "is the deterioration broad or concentrated?" directly: the gross
    positive increase is the sum of per-position CE rises; the top decile share
    tells us whether a handful of positions dominate it. Also correlates the CE
    rise with the entropy fall, which is the "confident in the wrong move"
    signature. Returns zeros when no position regressed.
    """
    metric = "teacher_weighted_cross_entropy_nats"
    deltas = np.asarray([f[metric] - b[metric] for b, f in zip(baseline, final)], dtype=np.float64)
    entropy_deltas = np.asarray(
        [f["mean_legal_move_entropy_nats"] - b["mean_legal_move_entropy_nats"]
         for b, f in zip(baseline, final)],
        dtype=np.float64,
    )
    gross_positive = float(deltas[deltas > 0].sum())
    if gross_positive <= 0.0:
        return {"gross_positive_cross_entropy_increase": 0.0,
                "top_decile_share": 0.0,
                "n_positions_worst_decile": 0,
                "correlation_ce_rise_with_entropy_fall": None}
    n_top = max(1, math.ceil(len(deltas) * 0.1))
    top_share = float(np.sort(deltas)[-n_top:].sum() / gross_positive)
    correlation = None
    if np.std(deltas) > 0 and np.std(entropy_deltas) > 0:
        correlation = float(np.corrcoef(deltas, entropy_deltas)[0, 1])
    return {
        "gross_positive_cross_entropy_increase": gross_positive,
        "n_positions_worst_decile": n_top,
        "top_decile_share": top_share,
        "correlation_ce_rise_with_entropy_fall": correlation,
    }


def worst_regressions(baseline: list[dict], final: list[dict], count: int) -> list[dict]:
    """The ``count`` positions with the largest cross-entropy increase."""
    rows = []
    for b, f in zip(baseline, final):
        delta = f["teacher_weighted_cross_entropy_nats"] - b["teacher_weighted_cross_entropy_nats"]
        rows.append({
            "fen": f["fen"],
            "cross_entropy_delta": float(delta),
            "baseline_cross_entropy": b["teacher_weighted_cross_entropy_nats"],
            "final_cross_entropy": f["teacher_weighted_cross_entropy_nats"],
            "baseline_topk_mass": b["teacher_topk_probability_mass"],
            "final_topk_mass": f["teacher_topk_probability_mass"],
            "baseline_best_mass": b["teacher_best_probability_mass_tie_aware"],
            "final_best_mass": f["teacher_best_probability_mass_tie_aware"],
            "baseline_entropy": b["mean_legal_move_entropy_nats"],
            "final_entropy": f["mean_legal_move_entropy_nats"],
            "n_teacher_targets": f["n_teacher_targets"],
        })
    rows.sort(key=lambda row: row["cross_entropy_delta"], reverse=True)
    return rows[:count]


def _seed_mean(per_seed: list[list[dict]]) -> list[dict]:
    """Average per-position metrics across seeds (position order is shared)."""
    n = len(per_seed[0])
    for rows in per_seed:
        if len(rows) != n:
            raise ValueError("seeds cover different numbers of held-out positions")
    keys = ("teacher_topk_probability_mass", "teacher_best_probability_mass_tie_aware",
            "teacher_weighted_probability_score", "teacher_weighted_cross_entropy_nats",
            "mean_legal_move_entropy_nats")
    out = []
    for i in range(n):
        row = {"fen": per_seed[0][i]["fen"], "n_teacher_targets": per_seed[0][i]["n_teacher_targets"]}
        for key in keys:
            row[key] = float(np.mean([rows[i][key] for rows in per_seed]))
        out.append(row)
    return out


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", default="bench/l3_heldout_gradient_pilot.json",
                        help="pilot JSON holding cached labels and the design")
    parser.add_argument("--out", default="bench/l3_heldout_per_position.json")
    parser.add_argument("--max-positions", type=int, default=0,
                        help="cap training positions for a smoke run (0 = use all)")
    parser.add_argument("--worst", type=int, default=15,
                        help="how many worst-regression positions to list")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    started = time.perf_counter()
    pilot = json.loads(Path(args.pilot).read_text(encoding="utf-8"))
    design = pilot["design"]
    depth = design["depth"]
    top_k = design["top_k"]
    seeds = design["training_seeds"]
    checkpoints = design["checkpoints_positions_seen"]
    training = pilot["cached_training_labels"]
    heldout = pilot["cached_heldout_labels"]
    if args.max_positions:
        training = training[: args.max_positions]
        checkpoints = [c for c in checkpoints if c <= args.max_positions]
        if not checkpoints or checkpoints[-1] != args.max_positions:
            checkpoints = [*checkpoints, args.max_positions]
            checkpoints = sorted(set(checkpoints))

    # Provenance guard: the cached trajectory is only the pilot's trajectory if
    # the perceptron code that produced it is still the code we replay.
    current_model_sha = hashlib.sha256((_REPO / "src/chessrl/perceptron.py").read_bytes()).hexdigest()
    model_sha_matches = current_model_sha == pilot["provenance"]["perceptron_py_sha256"]

    per_seed_results = []
    for seed in seeds:
        trainer = L3Trainer(config=L3Config(seed=seed, search_depth=depth, top_k=top_k))
        baseline = [_position_metrics(trainer.policy, record) for record in heldout]
        checkpoints_for_seed = {}
        for position_count, record in enumerate(training, start=1):
            _apply_cached_targets(trainer, record)
            if position_count in checkpoints:
                checkpoints_for_seed[position_count] = [
                    _position_metrics(trainer.policy, item) for item in heldout
                ]
        per_seed_results.append({"training_seed": seed, "baseline": baseline,
                                 "checkpoints": checkpoints_for_seed})

    baseline_mean = _seed_mean([row["baseline"] for row in per_seed_results])
    checkpoint_views = []
    for checkpoint in checkpoints:
        if checkpoint not in per_seed_results[0]["checkpoints"]:
            continue
        final_mean = _seed_mean([row["checkpoints"][checkpoint] for row in per_seed_results])
        summary = summarise_positions(baseline_mean, final_mean)
        checkpoint_views.append({
            "positions_seen": checkpoint,
            "seed_mean_position_summary": summary,
            "worst_regression_positions": worst_regressions(baseline_mean, final_mean, args.worst),
            "concentration": _cross_entropy_concentration(baseline_mean, final_mean),
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "design": {
            "purpose": "exploratory per-position diagnosis of held-out metric disagreement",
            "source_pilot": str(args.pilot),
            "reuses_cached_labels": True,
            "depth": depth,
            "top_k": top_k,
            "training_seeds": seeds,
            "checkpoints_positions_seen": checkpoints,
            "train_positions": len(training),
            "heldout_positions": len(heldout),
            "training_target_normalization": "unchanged: weight / max(target weights)",
            "position_delta_reference": "shared baseline at 0 positions seen, averaged across seeds",
            "per_position_metrics_averaged_across_seeds": True,
            "provenance_guard": {
                "current_perceptron_sha256": current_model_sha,
                "pilot_perceptron_sha256": pilot["provenance"]["perceptron_py_sha256"],
                "matches": model_sha_matches,
            },
        },
        "scope": (
            "exploratory; conditional on this one frozen training corpus and held-out set; "
            "does not establish generality to freshly sampled positions; not a scale signal"
        ),
        "checkpoint_views": checkpoint_views,
        "runtime_seconds": round(time.perf_counter() - started, 3),
    }
    out.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({
        "out": str(out),
        "model_sha_matches_pilot": model_sha_matches,
        "runtime_seconds": payload["runtime_seconds"],
        "checkpoint_views": [
            {"positions_seen": v["positions_seen"],
             "topk_mean_delta": v["seed_mean_position_summary"]["teacher_topk_probability_mass"]["mean_delta"],
             "ce_mean_delta": v["seed_mean_position_summary"]["teacher_weighted_cross_entropy_nats"]["mean_delta"],
             "entropy_mean_delta": v["seed_mean_position_summary"]["mean_legal_move_entropy_nats"]["mean_delta"],
             "regressed_ce_positions": v["seed_mean_position_summary"]["teacher_weighted_cross_entropy_nats"]["regressed_positions"],
             "ce_top_decile_share": v["concentration"]["top_decile_share"],
             "corr_ce_rise_entropy_fall": v["concentration"]["correlation_ce_rise_with_entropy_fall"]}
            for v in checkpoint_views
        ],
    }, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
