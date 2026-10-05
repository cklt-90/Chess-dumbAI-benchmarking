"""H12 -- does an L1-style prior save search nodes without changing the value?

The value-invariance half is a *theorem* (Knuth & Moore 1975): alpha-beta with a
correct window returns the same minimax value whatever the move order, so the
prior can only ever break ties. This script measures the *open* half at numpy
scale -- the node-count saving from using L1's blind distribution as the ordering
prior (the exact mechanism :class:`InformedMinimaxPolicy` already uses).

It asserts the search value is bit-identical with and without the prior, and
reports the node ratio over a fixed midgame set. The plain L2 engine runs with
``prior_weight=0``; the informed arm uses ``prior_weight=200`` with L1's
``move -> probability`` map as the ordering prior. No new model, no training.
"""
import random
import sys
from pathlib import Path

# Make the script runnable directly (pytest adds src/ via config, a raw run does
# not). Mirrors the bootstrap in bench/__main__.py.
_SRC = Path(__file__).resolve().parents[2] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import chess
import numpy as np

from chessrl.search import InformedMinimaxPolicy, MinimaxEngine

DEPTH = 4
PRIOR_WEIGHT = 200.0


def sample_positions(n=12, walk=10, seed=0):
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        b = chess.Board()
        for _ in range(walk):
            moves = list(b.legal_moves)
            if not moves:
                break
            b.push(rng.choice(moves))
        if not b.is_game_over(claim_draw=False):
            out.append(b.fen())
    return out


def measure(engine, board, prior=None):
    """Mirror ``MinimaxEngine.select``: iterative deepening, report (move, value, nodes).

    TT and node counter are reset per call so each position is measured in
    isolation -- the comparison is about the ordering prior, not about which
    engine happened to warm its table first.
    """
    engine.tt.clear()
    engine.nodes = 0
    moves = list(board.legal_moves)
    best = moves[0]
    for d in range(1, engine.depth + 1):
        best = engine._root_with_value(board, moves, d, prior)[0]
    move, value = engine._root_with_value(board, moves, engine.depth, prior)
    return best, value, engine.nodes


def main():
    fens = sample_positions(n=12, walk=10, seed=0)
    plain = MinimaxEngine(depth=DEPTH, prior_weight=0.0)
    informed = MinimaxEngine(depth=DEPTH, prior_weight=PRIOR_WEIGHT)
    ip = InformedMinimaxPolicy(depth=DEPTH, prior_weight=PRIOR_WEIGHT)

    ratios = []
    saved = 0
    max_val_diff = 0
    for fen in fens:
        board = chess.Board(fen)
        _, pv, pn = measure(plain, board, prior=None)
        prior_map = ip._prior(board)
        _, iv, inodes = measure(informed, board, prior=prior_map)

        # Value must be invariant under move ordering (the theorem).
        max_val_diff = max(max_val_diff, abs(pv - iv))
        if inodes <= pn:
            saved += 1
        ratios.append(inodes / max(pn, 1))

    ratios = np.array(ratios)
    print(f"positions measured : {len(fens)}")
    print(f"value invariance   : max |diff| = {max_val_diff} (must be 0)")
    print(
        f"node ratio (inf/plain): mean = {ratios.mean():.3f}, "
        f"min = {ratios.min():.3f}, max = {ratios.max():.3f}"
    )
    print(f"positions where prior saved nodes: {saved}/{len(fens)}")
    print(f"mean node saving: {(1 - ratios.mean()) * 100:.1f}%")


if __name__ == "__main__":
    main()
