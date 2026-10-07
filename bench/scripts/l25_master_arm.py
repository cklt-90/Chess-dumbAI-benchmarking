"""H25 master arm: does a *true* master supervise better than a weak one?

This is the runnable core of H25 ("Master-graded beats self-taught at equal
positions seen"). The full H25 needs a self-play arm, which is blocked (the
self-play signal is retired — 40/40 draws). The piece that *is* runnable now is
the comparison the user's "master = Stockfish" clarification points at:

    does imitating a **true** master (a UCI engine) produce a better student
    than imitating the **weak** master (the built-in depth-k search),
    at equal positions seen, same corpus, same seeds?

Design (matched, one free variable = the label source):

* one frozen training corpus and held-out set (reused from the pilot protocol);
* the same initialization seeds and checkpoints;
* **arm `weak`** trains on depth-k `search_feedback` labels;
* **arm `true`** trains on `EngineMaster` (Stockfish) labels;
* the *update path is identical* — both go through ``L3Trainer.train_on_targets``
  — so a difference is attributable to the labels, not the learner.

Yardstick. Each arm is evaluated against **both** teachers' held-out labels, so
the "home advantage" of the true arm (it is scored against its own teacher) is
visible rather than hidden. The primary endpoint is held-out **best-move mass
against Stockfish** at the final checkpoint, with a seed-conditional 95%
t interval and the paired weak-vs-true difference.

Scope. Exploratory; conditional on one frozen corpus and held-out set; the
intervals cover initialization seeds only. Not a strength claim — this measures
imitation of a teacher, not chess ability. No scale run is authorised by it.

Example:
    ./.venv/Scripts/python.exe bench/scripts/l25_master_arm.py \\
        --engine "C:/path/to/stockfish.exe" \\
        --out bench/l25_master_arm.json

Smoke:
    ... --train-positions 16 --heldout-positions 16 --seeds 1,2 \\
        --checkpoints 8,16 --weak-depth 2 --engine-depth 4 \\
        --out bench/l25_master_arm_smoke.json
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
for _p in (_REPO, _REPO / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import chess
import numpy as np

from chessrl.master import EngineMaster
from chessrl.perceptron import L3Config, L3Trainer

from bench.scripts.l3_heldout_diagnostic import _position_key, _t_critical_95, collect_positions
from bench.scripts.l3_heldout_per_position import _position_metrics

_METRIC_KEYS = (
    "teacher_topk_probability_mass",
    "teacher_best_probability_mass_tie_aware",
    "teacher_weighted_probability_score",
    "teacher_weighted_cross_entropy_nats",
    "mean_legal_move_entropy_nats",
)


def _to_target_rows(pairs) -> list[dict]:
    return [{"uci": move.uci(), "weight": float(weight)} for move, weight in pairs]


def label_records(records: list[dict], source, label: str) -> tuple[list[dict], int]:
    """Attach ``targets`` to each record from ``source(board) -> [(Move, w)]``.

    Returns the labelled records and the number of positions the source could
    not label (dropped). A non-zero drop count is reported, not silently hidden.
    """
    labelled: list[dict] = []
    dropped = 0
    for record in records:
        board = chess.Board(record["fen"])
        pairs = source(board)
        if not pairs:
            dropped += 1
            continue
        labelled.append({**record, "targets": _to_target_rows(pairs)})
    return labelled, dropped


def label_from_cache(records: list[dict], by_fen: dict) -> tuple[list[dict], int]:
    """Attach cached ``targets`` (already in ``{uci, weight}`` form) by FEN.

    Used to skip re-labelling with the weak master when a prior run's corpus is
    known to be identical (same data seed, depth and top-k). A non-zero drop
    count means the cached corpus did not cover these positions and the caller
    must not treat the run as matched.
    """
    labelled: list[dict] = []
    dropped = 0
    for record in records:
        rows = by_fen.get(record["fen"])
        if not rows:
            dropped += 1
            continue
        labelled.append({**record, "targets": rows})
    return labelled, dropped


def evaluate(policy, records: list[dict]) -> dict:
    rows = [_position_metrics(policy, record) for record in records]
    return {key: float(np.mean([row[key] for row in rows])) for key in _METRIC_KEYS}


def _weights_of(records: list[dict]) -> list[list[float]]:
    return [[t["weight"] for t in record["targets"]] for record in records]


def _target_entropy(weights: list[float]) -> float:
    arr = np.asarray(weights, dtype=np.float64)
    p = arr / arr.sum()
    return float(-(p * np.log(p)).sum())


def label_diagnostics(weak_records: list[dict], true_records: list[dict]) -> dict:
    """How different are the two teachers' labels on the same positions?"""
    weak_w, true_w = _weights_of(weak_records), _weights_of(true_records)
    n = min(len(weak_w), len(true_w))
    agree = sum(
        1 for a, b in zip(weak_records, true_records)
        if a["targets"][0]["uci"] == b["targets"][0]["uci"]
    )
    in_top5 = sum(
        1 for a, b in zip(weak_records, true_records)
        if a["targets"][0]["uci"] in {t["uci"] for t in b["targets"]}
    )
    return {
        "positions": n,
        "weak_mean_target_entropy_nats": float(np.mean([_target_entropy(w) for w in weak_w])),
        "true_mean_target_entropy_nats": float(np.mean([_target_entropy(w) for w in true_w])),
        "weak_mean_weight_spread": float(np.mean([w[0] - w[-1] for w in weak_w])),
        "true_mean_weight_spread": float(np.mean([w[0] - w[-1] for w in true_w])),
        "top1_agreement_rate": agree / n if n else None,
        "weak_top1_in_true_top5_rate": in_top5 / n if n else None,
    }


def run_arm(train_records: list[dict], yardsticks: dict[str, list[dict]],
            seeds: list[int], checkpoints: list[int], depth: int, top_k: int,
            search_weight: float) -> dict:
    per_seed = []
    for seed in seeds:
        trainer = L3Trainer(config=L3Config(
            seed=seed, search_depth=depth, top_k=top_k, search_weight=search_weight))
        views = {"0": {name: evaluate(trainer.policy, recs)
                       for name, recs in yardsticks.items()}}
        for count, record in enumerate(train_records, start=1):
            board = chess.Board(record["fen"])
            targets = [(chess.Move.from_uci(t["uci"]), t["weight"]) for t in record["targets"]]
            trainer.train_on_targets(board, targets)
            if count in checkpoints:
                views[str(count)] = {name: evaluate(trainer.policy, recs)
                                     for name, recs in yardsticks.items()}
        per_seed.append({"training_seed": seed, "views": views})
    return {"per_seed": per_seed}


def _mean_over_seeds(arm: dict, checkpoint: str, yardstick: str, key: str) -> float:
    return float(np.mean([s["views"][checkpoint][yardstick][key] for s in arm["per_seed"]]))


def _paired_ci(arm: dict, checkpoint: str, yardstick: str, key: str) -> dict:
    """Seed-conditional change from baseline and its 95% t interval."""
    deltas = [s["views"][checkpoint][yardstick][key] - s["views"]["0"][yardstick][key]
              for s in arm["per_seed"]]
    arr = np.asarray(deltas, dtype=np.float64)
    n = len(arr)
    mean = float(arr.mean())
    if n < 2:
        return {"mean_delta": mean, "sd": float(arr.std()), "n_seeds": n, "ci95": [mean, mean]}
    sd = float(arr.std(ddof=1))
    half = _t_critical_95(n) * sd / math.sqrt(n)
    return {"mean_delta": mean, "sd": sd, "n_seeds": n, "ci95": [mean - half, mean + half]}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True, help="path to a UCI engine (Stockfish)")
    parser.add_argument("--engine-depth", type=int, default=8)
    parser.add_argument("--weak-depth", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--search-weight", type=float, default=1.0,
                        help="step size; held equal across arms so the labels are the free variable")
    parser.add_argument("--train-positions", type=int, default=500)
    parser.add_argument("--heldout-positions", type=int, default=128)
    parser.add_argument("--seeds", default="101,102,103,104,105,106,107,108")
    parser.add_argument("--checkpoints", default="125,250,500")
    parser.add_argument("--data-seed", type=int, default=314159)
    parser.add_argument("--weak-labels-cache", default="",
                        help="prior run JSON holding cached weak-master labels; skips "
                             "re-labelling when the corpus is identical (same seed/depth/top-k)")
    parser.add_argument("--out", default="bench/l25_master_arm.json")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    started = time.perf_counter()
    seeds = [int(v) for v in args.seeds.split(",") if v.strip()]
    checkpoints = sorted({int(v) for v in args.checkpoints.split(",") if v.strip()})
    if len(seeds) < 2:
        raise ValueError("need at least two seeds for an interval")

    train_raw = collect_positions(args.train_positions, args.data_seed, set())
    train_keys = {_position_key(chess.Board(r["fen"])) for r in train_raw}
    heldout_raw = collect_positions(args.heldout_positions, args.data_seed + 1, train_keys)

    weak_labeler = L3Trainer(L3Config(seed=0, search_depth=args.weak_depth, top_k=args.top_k))
    weak_source = lambda b: weak_labeler.search_feedback(b, depth=args.weak_depth, top_k=args.top_k)

    cache_used = bool(args.weak_labels_cache)
    if cache_used:
        cached = json.loads(Path(args.weak_labels_cache).read_text(encoding="utf-8"))
        weak_by_fen = {r["fen"]: r["targets"]
                       for r in cached["cached_training_labels"] + cached["cached_heldout_labels"]}

    with EngineMaster(args.engine, top_k=args.top_k, depth=args.engine_depth) as master:
        true_source = master.targets

        if cache_used:
            weak_train, weak_drop = label_from_cache(train_raw, weak_by_fen)
            weak_held, weak_hdrop = label_from_cache(heldout_raw, weak_by_fen)
        else:
            weak_train, weak_drop = label_records(train_raw, weak_source, "weak")
            weak_held, weak_hdrop = label_records(heldout_raw, weak_source, "weak")
        true_train, true_drop = label_records(train_raw, true_source, "true")
        true_held, true_hdrop = label_records(heldout_raw, true_source, "true")
    if cache_used and (weak_drop or weak_hdrop):
        raise AssertionError("weak-label cache did not cover the corpus; run is not matched")

    # Align the two label sets to the positions both could label, so the arms
    # train on exactly the same corpus (otherwise the comparison is confounded).
    common = [r["fen"] for r in weak_train]
    true_by_fen = {r["fen"]: r for r in true_train}
    train_common = [true_by_fen[f] for f in common if f in true_by_fen]
    weak_by_fen = {r["fen"]: r for r in weak_train}
    weak_train_aligned = [weak_by_fen[r["fen"]] for r in train_common]

    yardsticks = {"stockfish": true_held, "weak": weak_held}
    arms = {
        "weak": run_arm(weak_train_aligned, yardsticks, seeds, checkpoints,
                        args.weak_depth, args.top_k, args.search_weight),
        "true": run_arm(train_common, yardsticks, seeds, checkpoints,
                        args.weak_depth, args.top_k, args.search_weight),
    }

    primary_key = "teacher_best_probability_mass_tie_aware"
    primary_yardstick = "stockfish"
    final = str(checkpoints[-1])
    comparison = {
        "primary_endpoint": "held-out best-move mass vs Stockfish labels at final checkpoint",
        "final_checkpoint": checkpoints[-1],
        "weak": _paired_ci(arms["weak"], final, primary_yardstick, primary_key),
        "true": _paired_ci(arms["true"], final, primary_yardstick, primary_key),
    }
    comparison["true_minus_weak_mean_delta"] = (
        comparison["true"]["mean_delta"] - comparison["weak"]["mean_delta"])

    # Intervals for *every* metric on *both* yardsticks, because the metrics
    # disagree (best-move mass vs cross-entropy) and a single-endpoint readout
    # would hide that.
    metric_intervals = {
        arm: {ys: {k: _paired_ci(a, final, ys, k) for k in _METRIC_KEYS}
              for ys in yardsticks}
        for arm, a in arms.items()
    }

    # Full per-checkpoint, per-yardstick table for both arms.
    table = {}
    for checkpoint in ["0", *[str(c) for c in checkpoints]]:
        table[checkpoint] = {
            arm: {ys: {k: _mean_over_seeds(a, checkpoint, ys, k) for k in _METRIC_KEYS}
                  for ys in yardsticks}
            for arm, a in arms.items()
        }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "design": {
            "purpose": "H25 runnable core: true-master vs weak-master supervision at matched budget",
            "engine": str(args.engine),
            "engine_depth": args.engine_depth,
            "weak_depth": args.weak_depth,
            "top_k": args.top_k,
            "search_weight": args.search_weight,
            "training_seeds": seeds,
            "checkpoints_positions_seen": [0, *checkpoints],
            "train_positions": len(train_common),
            "heldout_positions": len(true_held),
            "label_drop_counts": {"weak_train": weak_drop, "true_train": true_drop,
                                  "weak_heldout": weak_hdrop, "true_heldout": true_hdrop},
            "update_path": "L3Trainer.train_on_targets (identical for both arms)",
            "yardsticks": "both arms scored against both teachers' held-out labels",
            "weak_labels_cache_used": cache_used,
            "perceptron_py_sha256": hashlib.sha256(
                (_REPO / "src/chessrl/perceptron.py").read_bytes()).hexdigest(),
            "master_py_sha256": hashlib.sha256(
                (_REPO / "src/chessrl/master.py").read_bytes()).hexdigest(),
        },
        "scope": ("exploratory; conditional on one frozen corpus and held-out set; "
                  "intervals cover init seeds only; measures teacher imitation, not strength"),
        "label_diagnostics": label_diagnostics(weak_train_aligned, train_common),
        "comparison": comparison,
        "metric_intervals": metric_intervals,
        "table": table,
        "per_seed": {arm: a["per_seed"] for arm, a in arms.items()},
        "runtime_seconds": round(time.perf_counter() - started, 3),
    }
    out.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"out": str(out), "runtime_seconds": payload["runtime_seconds"],
                      "label_diagnostics": payload["label_diagnostics"],
                      "comparison": comparison}, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
