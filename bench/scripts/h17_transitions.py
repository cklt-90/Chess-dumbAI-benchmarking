"""H17 -- does transition-focused midstate sampling change what the model learns?

H17's hypothesis: positions near a *transition* (the move played was a capture or
a pawn move) carry more learning signal than quiet positions, so a store that
flags and prioritises them should learn faster per gradient update.

This script exercises the code added to ``chessrl.cache``:
* builds a MidstateStore from a few random self-play games (so the transition
  tag is populated from real move sequences),
* shows that ``sample_near_transition`` returns a far higher fraction of
  near-transition positions than the uniform ``sample``,
* runs a *light, indicative* matched-update comparison: two identical L3 trainers
  (same seed) each get the same number of search-feedback updates, one fed
  uniform samples and one transition-focused samples, and we compare how much
  each improves on a fixed probe (greedy move == depth-2 search).

The decisive version of H17 -- games-to-fixed-accuracy over a large corpus -- is
the natural next step; this is the cheap, reusable smoke test of the mechanism.
"""
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import random

import chess
import numpy as np

from chessrl.cache import MidstateStore
from chessrl.diagnostics import PROBE_FENS
from chessrl.perceptron import L3Config, L3Policy, L3Trainer
from chessrl.search import MinimaxEngine


def play_random_game(rng: random.Random, max_plies: int = 40) -> list[str]:
    b = chess.Board()
    fens = [b.fen()]
    while len(fens) < max_plies and not b.is_game_over(claim_draw=False):
        b.push(rng.choice(list(b.legal_moves)))
        fens.append(b.fen())
    return fens


def build_store(n_games: int = 8, seed: int = 0) -> MidstateStore:
    rng = random.Random(seed)
    store = MidstateStore()
    for _ in range(n_games):
        store.record_game(play_random_game(rng), store.new_game_id())
    return store


def probe_agreement(trainer: L3Trainer, depth: int = 2) -> float:
    """Fraction of PROBE_FENS where the policy's greedy move matches search."""
    eng = MinimaxEngine(depth=depth)
    total = 0
    ok = 0
    for fen in PROBE_FENS:
        board = chess.Board(fen)
        if board.is_game_over(claim_draw=False):
            continue
        total += 1
        best = eng.select(board)
        trainer.policy.greedy = True
        if trainer.policy.select(board) == best:
            ok += 1
    return ok / total if total else 0.0


def main() -> None:
    store = build_store(n_games=8, seed=0)
    flagged = sum(e.near_transition for e in store)
    print(f"corpus: {len(store)} positions, {flagged} near-transition "
          f"({100 * flagged / len(store):.0f}%)")

    uni = store.sample(200, rng=random.Random(1))
    trans = store.sample_near_transition(200, rng=random.Random(1))
    fu = sum(e.near_transition for e in uni) / len(uni)
    ft = sum(e.near_transition for e in trans) / len(trans)
    print(f"uniform sample:    {100 * fu:.0f}% near-transition")
    print(f"transition sample: {100 * ft:.0f}% near-transition")

    # Matched-update training comparison (indicative; small corpus).
    n_iter, batch = 40, 8
    a = L3Trainer(L3Config(seed=7))
    b = L3Trainer(L3Config(seed=7))
    base_a = probe_agreement(a)
    base_b = probe_agreement(b)
    for it in range(n_iter):
        for entry in store.sample(batch, rng=random.Random(it)):
            a.train_on_search_feedback(entry.board(), depth=2)
        for entry in store.sample_near_transition(batch, rng=random.Random(1000 + it)):
            b.train_on_search_feedback(entry.board(), depth=2)
    after_a = probe_agreement(a)
    after_b = probe_agreement(b)
    print("\nProbe agreement (greedy == depth-2 search):")
    print(f"  uniform    sampling: {base_a:.2f} -> {after_a:.2f}  (+{after_a - base_a:+.2f})")
    print(f"  transition sampling: {base_b:.2f} -> {after_b:.2f}  (+{after_b - base_b:+.2f})")
    print(f"  improvement delta (transition - uniform): "
          f"{(after_b - base_b) - (after_a - base_a):+.2f} "
          "(indicative; decisive H17 needs a larger corpus)")


if __name__ == "__main__":
    main()
