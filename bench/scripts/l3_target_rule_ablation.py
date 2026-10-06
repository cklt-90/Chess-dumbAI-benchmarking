"""Matched local ablation of the L3 search-feedback target rule.

The per-position diagnosis of the pilot showed broad, misdirected
over-sharpening: at 500 positions seen the policy's mass on the teacher's
listed top-k rose, but its mass on the teacher's *best* move improved on only
33 of 128 held-out positions, weighted cross-entropy worsened on 107 of 128,
and entropy fell on 127 of 128. That points at the update *target*, not at a
per-position outlier subset.

This script tests the target rule with three matched arms. All arms replay the
**same frozen cached corpus and held-out set** and the **same training seeds**;
they differ only in how the teacher's listed weights are turned into the update
target ``p_target`` and, for the third arm, the overall step scale:

  * ``max``  (current / control):   p_target = weight / max(weights)
        The best move is assigned target 1.0, so any residual model
        probability below 1.0 is pushed up on every single position.

  * ``sum``  (alternative):         p_target = weight / sum(weights)
        The listed weights become a proper probability distribution over the
        top-k, so no single move is assigned target 1.0 and the total target
        mass is bounded by 1.

  * ``max_scaled`` (magnitude-matched control): same ``max`` target rule, but
        ``search_weight`` multiplied by ``--mag-scale``, so the aggregate
        update magnitude matches the ``sum`` arm. This exists to rule out the
        confound that any smaller step looks better on one frozen trajectory.

Everything except the target mapping and the arm's declared ``search_weight``
multiplier is held constant: the 300cp squash in ``search_feedback``, the
credit/update path, the checkpoint schedule, and the metrics. This is
deliberately *not* a bug fix and not folded into the pilot result; it is a
separate, matched, exploratory comparison conditional on one frozen corpus and
one held-out set. It does not establish generality to freshly sampled positions
and must not be read as a scale signal.

The ``max`` and ``sum`` rules preserve the weight vector's *shape* (identical
ratios) but differ in *scale*: ``max`` pins the best move at target 1.0 and can
assign up to k=5 units of target mass across the listed set, while ``sum``
bounds total target mass to 1. That scale difference also makes the ``sum``
arm take smaller aggregate update steps, which is the confound the third arm
addresses. A third arm, ``max_scaled``, keeps the ``max`` rule but multiplies
``search_weight`` by ``--mag-scale`` (default 0.3742, the corpus-average ratio
of sum-arm to max-arm total target mass) so its update magnitude matches the
``sum`` arm. If ``max_scaled`` still degrades cross-entropy, the target *rule*
is implicated; if it does not, the effect was step size.

Example:
    ./.venv/Scripts/python.exe bench/scripts/l3_target_rule_ablation.py \\
        --pilot bench/l3_heldout_gradient_pilot.json \\
        --out bench/l3_target_rule_ablation.json

Smoke:
    ./.venv/Scripts/python.exe bench/scripts/l3_target_rule_ablation.py \\
        --pilot bench/l3_heldout_gradient_pilot.json --max-positions 16 \\
        --out bench/l3_target_rule_ablation_smoke.json
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

from bench.scripts.l3_heldout_diagnostic import _position_key
from bench.scripts.l3_heldout_per_position import _position_metrics, _seed_mean, summarise_positions

# Target rules: how teacher weights map to ``p_target``. The 'max' and 'sum'
# arms differ ONLY here; the 'max_scaled' arm additionally matches the raw
# update magnitude so the effect of the target *scale* can be told apart from
# the effect of a smaller *step*.
_BUILD_TARGET = {
    "max": lambda weights: weights / float(np.max(weights)),
    "sum": lambda weights: weights / float(np.sum(weights)),
}

# Arm definitions: name -> (target rule, search_weight multiplier). The
# multiplier normalises the aggregate update magnitude; for 'max_scaled' it is
# set from ``--mag-scale`` so the max rule takes steps the same size as the sum
# rule. A step-size sweep appends extra ``max@<mult>`` arms with the same rule.
_NAMED_ARMS = {
    "max": ("max", 1.0),
    "sum": ("sum", 1.0),
    "max_scaled": ("max", None),  # None -> use --mag-scale
}


def build_arms(mag_scale: float, sweep: list[float] | None = None) -> dict:
    """Resolve the arm registry, expanding a step-size sweep if requested.

    Sweep arms keep the ``max`` target rule fixed and vary only
    ``search_weight``, so the resulting curve isolates the step-size effect
    from the target-rule effect.
    """
    arms = dict(_NAMED_ARMS)
    if sweep:
        # Sweep mode is defined solely by the requested multipliers; the max
        # control is whichever sweep arm uses multiplier 1.0. De-duplicate so a
        # sweep containing 1.0 does not silently emit the same arm twice.
        unique = sorted(set(float(m) for m in sweep))
        arms = {f"max@{mult:g}": ("max", mult) for mult in unique}
    resolved = {}
    for name, (rule, scale) in arms.items():
        resolved[name] = (rule, mag_scale if scale is None else scale)
    return resolved


def _apply_target_rule(trainer: L3Trainer, record: dict, rule: str,
                       weight_scale: float = 1.0) -> None:
    """Apply cached targets with the given target rule and weight scaling.

    Mirrors ``train_on_search_feedback`` / ``_apply_cached_targets`` exactly
    except for the ``p_target`` mapping (``rule``) and an overall
    ``search_weight`` multiplier (``weight_scale``), so an arm's trajectory
    differs from the control only through those two knobs.
    """
    build = _BUILD_TARGET[rule]
    board = chess.Board(record["fen"])
    targets = [(chess.Move.from_uci(item["uci"]), float(item["weight"]))
               for item in record["targets"]]
    trainer.policy.scorer.set_position(board)
    distribution = trainer.policy.move_distribution(board)
    weights = np.asarray([weight for _, weight in targets], dtype=np.float64)
    target_probabilities = build(weights)
    for (move, _), target_probability in zip(targets, target_probabilities):
        model_probability = float(distribution[move_to_index(move)])
        gap = float(target_probability) - model_probability
        trainer.apply_credit(
            board, move, weight_scale * trainer.config.search_weight * gap
        )
        trainer.search_updates += 1


def _run_arm(pilot: dict, arm: str, rule: str, weight_scale: float,
             max_positions: int) -> dict:
    design = pilot["design"]
    depth, top_k = design["depth"], design["top_k"]
    seeds = design["training_seeds"]
    checkpoints = design["checkpoints_positions_seen"]
    training = pilot["cached_training_labels"]
    heldout = pilot["cached_heldout_labels"]
    if max_positions:
        training = training[:max_positions]
        checkpoints = sorted(set([c for c in checkpoints if c <= max_positions] + [max_positions]))

    per_seed = []
    for seed in seeds:
        trainer = L3Trainer(config=L3Config(seed=seed, search_depth=depth, top_k=top_k))
        baseline = [_position_metrics(trainer.policy, record) for record in heldout]
        views = {}
        for position_count, record in enumerate(training, start=1):
            _apply_target_rule(trainer, record, rule, weight_scale)
            if position_count in checkpoints:
                views[position_count] = [_position_metrics(trainer.policy, item) for item in heldout]
        per_seed.append({"training_seed": seed, "baseline": baseline, "checkpoints": views})

    baseline_mean = _seed_mean([row["baseline"] for row in per_seed])
    checkpoint_views = []
    for checkpoint in checkpoints:
        if checkpoint not in per_seed[0]["checkpoints"]:
            continue
        final_mean = _seed_mean([row["checkpoints"][checkpoint] for row in per_seed])
        checkpoint_views.append({
            "positions_seen": checkpoint,
            "seed_mean_position_summary": summarise_positions(baseline_mean, final_mean),
            "final_seed_mean_metrics": _final_means(final_mean),
        })
    return {"arm": arm, "rule": rule, "weight_scale": weight_scale,
            "baseline_seed_mean_metrics": _final_means(baseline_mean),
            "checkpoint_views": checkpoint_views, "per_seed": per_seed}


def _final_means(rows: list[dict]) -> dict:
    keys = ("teacher_topk_probability_mass", "teacher_best_probability_mass_tie_aware",
            "teacher_weighted_probability_score", "teacher_weighted_cross_entropy_nats",
            "mean_legal_move_entropy_nats")
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def _paired_deltas(arm: dict, metric: str) -> dict:
    """Per-seed paired change from baseline, matched across seeds by index."""
    out = {}
    for view in arm["checkpoint_views"]:
        checkpoint = view["positions_seen"]
        for seed_result in arm["per_seed"]:
            final = seed_result["checkpoints"].get(checkpoint)
            if final is None:
                continue
            base = seed_result["baseline"]
            deltas = [f[metric] - b[metric] for b, f in zip(base, final)]
            out.setdefault(checkpoint, []).append(float(np.mean(deltas)))
    return out


def _seed_ci(per_seed_values: list[float]) -> dict:
    """Seed-conditional mean and 95% t interval for a paired mean delta."""
    arr = np.asarray(per_seed_values, dtype=np.float64)
    n = len(arr)
    mean = float(arr.mean())
    if n < 2:
        return {"mean": mean, "sd": 0.0, "n_seeds": n, "ci95": [mean, mean]}
    sd = float(arr.std(ddof=1))
    from bench.scripts.l3_heldout_diagnostic import _t_critical_95
    half = _t_critical_95(n) * sd / math.sqrt(n)
    return {"mean": mean, "sd": sd, "n_seeds": n, "ci95": [mean - half, mean + half]}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", default="bench/l3_heldout_gradient_pilot.json")
    parser.add_argument("--out", default="bench/l3_target_rule_ablation.json")
    parser.add_argument("--max-positions", type=int, default=0)
    parser.add_argument("--mag-scale", type=float, default=0.3742,
                        help="search_weight multiplier for the magnitude-matched max arm")
    parser.add_argument("--sweep", default="",
                        help="comma-separated search_weight multipliers; keeps the max rule "
                             "fixed and replaces the named arms with a step-size sweep")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    started = time.perf_counter()
    pilot = json.loads(Path(args.pilot).read_text(encoding="utf-8"))

    current_model_sha = hashlib.sha256((_REPO / "src/chessrl/perceptron.py").read_bytes()).hexdigest()
    model_sha_matches = current_model_sha == pilot["provenance"]["perceptron_py_sha256"]

    sweep = [float(v) for v in args.sweep.split(",") if v.strip()]
    arms = build_arms(args.mag_scale, sweep or None)
    arm_order = list(arms)
    results = {name: _run_arm(pilot, name, *arms[name], args.max_positions)
               for name in arm_order}

    comparison = []
    for i, checkpoint in enumerate(v["positions_seen"]
                                   for v in results[arm_order[0]]["checkpoint_views"]):
        row = {"positions_seen": checkpoint}
        for arm in arm_order:
            view = results[arm]["checkpoint_views"][i]
            s = view["seed_mean_position_summary"]
            baseline = results[arm]["baseline_seed_mean_metrics"]
            final = view["final_seed_mean_metrics"]
            row[arm] = {
                "weighted_cross_entropy_delta": s["teacher_weighted_cross_entropy_nats"]["mean_delta"],
                "best_move_mass_delta": s["teacher_best_probability_mass_tie_aware"]["mean_delta"],
                "topk_mass_delta": s["teacher_topk_probability_mass"]["mean_delta"],
                "entropy_delta": s["mean_legal_move_entropy_nats"]["mean_delta"],
                "ce_regressed_positions": s["teacher_weighted_cross_entropy_nats"]["regressed_positions"],
                # Absolute levels, so "did the set-mass gain land on the best
                # move?" can be read off directly rather than inferred.
                "baseline_topk_mass": baseline["teacher_topk_probability_mass"],
                "final_topk_mass": final["teacher_topk_probability_mass"],
                "baseline_best_move_mass": baseline["teacher_best_probability_mass_tie_aware"],
                "final_best_move_mass": final["teacher_best_probability_mass_tie_aware"],
                "final_best_over_topk_ratio": (
                    final["teacher_best_probability_mass_tie_aware"]
                    / final["teacher_topk_probability_mass"]
                    if final["teacher_topk_probability_mass"] else None),
            }
        comparison.append(row)

    # Seed-conditional intervals for the CE delta, the metric that carried the
    # pilot's disagreement, for every arm at every checkpoint.
    ce_intervals = {}
    for arm in arm_order:
        per_ckpt = _paired_deltas(results[arm], "teacher_weighted_cross_entropy_nats")
        ce_intervals[arm] = {str(cp): _seed_ci(vals) for cp, vals in sorted(per_ckpt.items())}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if sweep:
        arm_descriptions = {
            name: f"target rule '{rule}', search_weight multiplier {scale:g}"
            for name, (rule, scale) in arms.items()
        }
        purpose = "matched local step-size sweep with the max target rule held fixed"
    else:
        arm_descriptions = {
            "max": "control: p_target = weight / max(weights) (current production rule)",
            "sum": "alternative: p_target = weight / sum(weights)",
            "max_scaled": (
                f"magnitude-matched control: same 'max' target rule, search_weight * "
                f"{args.mag_scale} so aggregate update magnitude matches the sum arm"
            ),
        }
        purpose = "matched local ablation of the L3 search-feedback target rule"
    payload = {
        "design": {
            "purpose": purpose,
            "source_pilot": str(args.pilot),
            "reuses_cached_labels": True,
            "arms": arm_descriptions,
            "magnitude_scale": args.mag_scale,
            "sweep_multipliers": sweep,
            "held_constant": ["target rule (sweep mode)", "300cp squash margin",
                              "credit/update path", "checkpoint schedule", "metrics",
                              "corpus", "seeds"],
            "training_seeds": pilot["design"]["training_seeds"],
            "checkpoints_positions_seen": [v["positions_seen"]
                                           for v in results[arm_order[0]]["checkpoint_views"]],
            "provenance_guard": {
                "current_perceptron_sha256": current_model_sha,
                "pilot_perceptron_sha256": pilot["provenance"]["perceptron_py_sha256"],
                "matches": model_sha_matches,
            },
        },
        "scope": (
            "separate matched exploratory ablation; conditional on this one frozen corpus "
            "and held-out set; not a bug fix, not folded into the pilot result, and not a "
            "scale signal; does not establish generality to freshly sampled positions"
        ),
        "comparison": comparison,
        "seed_conditional_ce_intervals": ce_intervals,
        "runtime_seconds": round(time.perf_counter() - started, 3),
    }
    out.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"out": str(out), "model_sha_matches_pilot": model_sha_matches,
                      "runtime_seconds": payload["runtime_seconds"],
                      "comparison": comparison}, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
