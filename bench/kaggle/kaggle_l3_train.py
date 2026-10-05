"""Kaggle-ready training + evaluation driver for the L3 perceptron.

This is the local analogue of the training-signal axis (HYPOTHESES.md Group G/H).
It reuses the existing, tested trainers in ``chessrl`` -- there is no new
learning code here, only orchestration behind a portable CLI -- so a run on
Kaggle is *exactly* a run on this machine, only with more games / positions.

Arms
----
  selfplay     L3 trains by self-play outcome RL. The incumbent signal
               (H24 / H26 baseline). Plays the live policy against a frozen
               copy of itself (refreshed every ``refresh_every`` games) and
               feeds each game to ``L3Trainer.train_on_game``.
  supervised   L3 imitates a depth-D search on a corpus of positions
               (H25 / H11). The cheapest *trained* arm and the one the
               literature review flagged as the first to run: a built-in
               weak master (depth-5 search) supplies the label. Uses
               ``L3Trainer.train_on_search_feedback``.
  randomwalk   L3 trains on the *outcomes* of random-vs-random games
               (H26 control). Uses ``L3Trainer.train_on_game`` over a
               generated random corpus. Must NOT beat untrained L3 -- it is
               the load-bearing control for every other arm.
  both         selfplay then supervised, in that order (the recommended
               Kaggle job: learn from play, then sharpen with search).

Every arm is numpy-only (no torch). That is deliberate: L0-L3 must run without
the optional torch extra, and the supervised/self-play/random-walk arms are the
regime-white-space questions -- they do not need a GPU.

Outputs (written to --out-dir)
-------------------------------
  l3_<mode>.json        the trained model (``L3Policy.save``)
  l3_<mode>.meta.json   training summary (games, updates, seconds, ...)
  results.json          the head-to-head evaluation tally + training summary
  summary.md            a short human-readable report

Usage
-----
  # Local smoke test (seconds-to-minutes, not hours)
  python bench/kaggle/kaggle_l3_train.py --mode both --games 20 --positions 300 \
      --search-depth 4 --eval-against untrained --eval-games 20 --seed 1 \
      --out-dir bench/kaggle/out_smoke

  # Kaggle scale (hours on CPU)
  python bench/kaggle/kaggle_l3_train.py --mode both --games 4000 --positions 60000 \
      --search-depth 5 --eval-against material --eval-games 40 --seed 7 \
      --out-dir /kaggle/working/out

Kaggle setup
------------
  * The repo root must be importable. This script bootstraps ``src/`` and
    ``bench/`` onto ``sys.path`` from its own location, so just drop the repo
    somewhere on the Kaggle notebook's filesystem (e.g. ``/kaggle/working/``).
  * ``pip install python-chess numpy`` if they are not already present.
  * Do NOT install torch for this script -- it is not imported.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

# --- path bootstrap: make the script run from anywhere -------------------
# Running a script file directly puts the script's *directory* (bench/kaggle)
# on sys.path, not the repo root. `python -m bench` does the opposite. So we
# add both the repo root (for the `bench` package) and repo/src (for
# `chessrl`) explicitly.
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
for _p in (_REPO, _REPO / "src"):
    _s = str(_p)
    if _p.is_dir() and _s not in sys.path:
        sys.path.insert(0, _s)

import chess  # noqa: E402  (import after path bootstrap)
import numpy as np  # noqa: E402

from chessrl.perceptron import L3Config, L3Policy, L3Trainer  # noqa: E402
from chessrl.game import (  # noqa: E402
    MaterialPolicy,
    RandomPolicy,
    generate_games,
    play_game,
    play_match,
)
from chessrl.search import MinimaxPolicy  # noqa: E402

# Reuse the bench's fixed, White-to-move opening book so the evaluation match
# is low-variance and comparable to the in-repo bench results.
from bench.levels import OPENING_BOOK  # noqa: E402


# --------------------------------------------------------------------------
# reference opponent factory
# --------------------------------------------------------------------------

def build_reference(name: str, seed: int = 0):
    """Construct a fixed opponent for the evaluation match.

    ``untrained`` is the decisive control: a trained L3 must beat an untrained
    L3 for the training signal to mean anything (H5's logic, applied to L3).
    """
    if name == "untrained":
        return L3Policy(seed=seed)
    if name == "material":
        return MaterialPolicy(seed=0)
    if name == "random":
        return RandomPolicy(seed=0)
    if name == "L2-d1":
        opp = MinimaxPolicy(depth=1)
        opp.name = "L2-d1"
        return opp
    raise SystemExit(f"unknown --eval-against: {name} (want untrained|material|random|L2-d1)")


# --------------------------------------------------------------------------
# corpus generation + training arms
# --------------------------------------------------------------------------

def collect_positions(n: int, seed: int, max_plies: int, no_progress: int) -> list[str]:
    """Gather ``n`` diverse midgame FENs from random self-play.

    Random play produces a far wider position spread than opening-only replay,
    which is exactly what an imitation target needs to generalise.
    """
    rng = random.Random(seed)
    fens: list[str] = []
    while len(fens) < n:
        g = play_game(
            RandomPolicy(rng.randrange(1 << 31)),
            RandomPolicy(rng.randrange(1 << 31)),
            max_plies=max_plies,
            no_progress_limit=no_progress,
            rng=rng,
            record_fens=True,
        )
        fens.extend(g.fens)
    rng.shuffle(fens)
    return fens[:n]


def run_selfplay(trainer: L3Trainer, games: int, max_plies: int,
                 no_progress: int, seed: int, refresh_every: int = 10) -> dict:
    """Self-play outcome RL: the live policy plays a frozen copy of itself.

    This mirrors ``FactoredSoftmaxTrainer.train`` (the L1 version). The frozen
    copy is refreshed every ``refresh_every`` games so the opponent keeps pace
    without ever being the same live object -- refreshing too rarely leaves the
    opponent too weak to be informative, too often and the two sides converge
    and the mirroring problem returns. ``L3Trainer`` exposes ``train_on_game``
    but not the self-play loop, so the loop lives here and reuses it.
    """
    rng = random.Random(seed)
    tally = {"1-0": 0, "0-1": 0, "1/2-1/2": 0, "unfinished": 0}
    opponent = copy.deepcopy(trainer.policy)
    opponent.rng = np.random.default_rng(rng.randrange(1 << 31))
    for i in range(games):
        if i and i % refresh_every == 0:
            opponent = copy.deepcopy(trainer.policy)
            opponent.rng = np.random.default_rng(rng.randrange(1 << 31))
        if i % 2:
            white, black = opponent, trainer.policy
        else:
            white, black = trainer.policy, opponent
        game = play_game(
            white, black,
            max_plies=max_plies,
            no_progress_limit=no_progress,
            rng=rng,
            record_fens=True,
        )
        trainer.train_on_game(game)
        tally[game.result] = tally.get(game.result, 0) + 1
    return {
        "games": games,
        "outcomes": tally,
        "updates": trainer.updates,
        "search_updates": trainer.search_updates,
    }


def run_supervised(trainer: L3Trainer, positions: list[str], search_depth: int) -> int:
    applied = 0
    for fen in positions:
        board = chess.Board(fen)
        if board.is_game_over():
            continue
        res = trainer.train_on_search_feedback(board, depth=search_depth)
        applied += int(bool(res.get("applied")))
    return applied


def run_randomwalk(trainer: L3Trainer, n_games: int, max_plies: int,
                   no_progress: int, seed: int) -> int:
    rng = random.Random(seed)
    for _ in range(n_games):
        g = play_game(
            RandomPolicy(rng.randrange(1 << 31)),
            RandomPolicy(rng.randrange(1 << 31)),
            max_plies=max_plies,
            no_progress_limit=no_progress,
            rng=rng,
            record_fens=True,
        )
        trainer.train_on_game(g)
    return n_games


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def render_md(summary: dict) -> str:
    lines = ["# L3 training run", ""]
    lines.append(f"**mode:** `{summary['mode']}`  **seed:** {summary['seed']}")
    lines.append("")
    lines.append("## Training")
    for arm, data in summary["arms"].items():
        lines.append(f"### {arm}  ({data.get('seconds', '?')}s)")
        for k, v in data.items():
            if k == "seconds":
                continue
            lines.append(f"- {k}: {v}")
        lines.append("")
    ev = summary["eval"]
    lines.append("## Evaluation")
    lines.append(f"{ev['a']}  vs  {ev['b']}  ({ev['games']} games)")
    lines.append("")
    lines.append(f"- wins: {ev['wins']}  draws: {ev['draws']}  "
                 f"losses: {ev['losses']}  unfinished: {ev['unfinished']}")
    lines.append(f"- points_pct (wins / played): {ev['points_pct']:.3f}")
    lines.append("")
    lines.append("> `points_pct` counts only decisive wins; the bench's own "
                 "Bradley-Terry fit (which scores draws as 0.5) is the "
                 "authoritative metric. Re-run the trained model through "
                 "`python -m bench` for that.")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train + evaluate the L3 perceptron.")
    p.add_argument("--mode", choices=["selfplay", "supervised", "randomwalk", "both"],
                   default="both")
    p.add_argument("--games", type=int, default=200,
                   help="self-play / random-walk training games")
    p.add_argument("--positions", type=int, default=20000,
                   help="supervised: number of search-feedback positions")
    p.add_argument("--search-depth", type=int, default=5,
                   help="supervised: depth of the labelling search")
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--credit-decay", type=float, default=0.99)
    p.add_argument("--search-weight", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--shaped-weight", type=float, default=0.3)
    p.add_argument("--shaped-reward", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max-plies", type=int, default=120)
    p.add_argument("--no-progress", type=int, default=60)
    p.add_argument("--eval-against", default="untrained",
                   choices=["untrained", "material", "random", "L2-d1"])
    p.add_argument("--eval-games", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="./kaggle_out")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cfg = L3Config(
        lr=args.lr,
        credit_decay=args.credit_decay,
        search_weight=args.search_weight,
        top_k=args.top_k,
        search_depth=args.search_depth,
        shaped_reward=args.shaped_reward,
        shaped_weight=args.shaped_weight,
        max_plies=args.max_plies,
        no_progress_limit=args.no_progress,
        seed=args.seed,
    )
    trainer = L3Trainer(config=cfg)
    summary: dict = {"mode": args.mode, "seed": args.seed, "arms": {}}

    if args.mode in ("selfplay", "both"):
        t0 = time.perf_counter()
        s = run_selfplay(
            trainer, args.games, args.max_plies, args.no_progress, args.seed
        )
        summary["arms"]["selfplay"] = {**s, "seconds": round(time.perf_counter() - t0, 2)}

    if args.mode in ("supervised", "both"):
        t0 = time.perf_counter()
        positions = collect_positions(
            args.positions, args.seed + 1, args.max_plies, args.no_progress
        )
        applied = run_supervised(trainer, positions, args.search_depth)
        summary["arms"]["supervised"] = {
            "positions_requested": args.positions,
            "positions_applied": applied,
            "search_depth": args.search_depth,
            "seconds": round(time.perf_counter() - t0, 2),
        }

    if args.mode == "randomwalk":
        t0 = time.perf_counter()
        n = run_randomwalk(trainer, args.games, args.max_plies, args.no_progress, args.seed)
        summary["arms"]["randomwalk"] = {
            "games": n, "seconds": round(time.perf_counter() - t0, 2)
        }

    saved = trainer.save(out / f"l3_{args.mode}.json")

    # Evaluate: reload from disk so the persistence path is itself tested.
    trained = L3Policy.load(saved["model"])
    trained.name = f"L3-{args.mode}-trained"
    opponent = build_reference(args.eval_against, seed=args.seed)
    match = play_match(
        trained, opponent,
        games=args.eval_games,
        opening_fens=OPENING_BOOK,
        seed=args.seed,
        max_plies=args.max_plies,
        no_progress_limit=args.no_progress,
    )
    summary["eval"] = {
        k: match[k] for k in (
            "a", "b", "games", "played", "unfinished",
            "wins", "draws", "losses", "score_a", "points_pct",
        )
    }

    (out / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "summary.md").write_text(render_md(summary), encoding="utf-8")

    print(json.dumps(summary["eval"], indent=2))
    print(f"\nWrote outputs to {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
