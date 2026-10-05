"""H21 — does an untrained factorised policy carry a usable spatial prior?

Runs the probe-set correlation the register asks for: for many seeds, correlate an
untrained L3Policy's per-move log-probability against centrality (distance to
board centre) and forwardness (rank advance for the side to move), tested against
a permutation null over move labels. Decisive either way; near-zero cost, no games.
"""
import numpy as np
import chess

from chessrl.perceptron import L3Policy
from chessrl.diagnostics import PROBE_FENS
from chessrl import masks as M


def centrality(sq):
    f, r = chess.square_file(sq), chess.square_rank(sq)
    return -((f - 3.5) ** 2 + (r - 3.5) ** 2) ** 0.5


def forwardness(sq, colour):
    r = chess.square_rank(sq)
    return r if colour == chess.WHITE else 7 - r


def correlate(seed):
    """Correlate an untrained L3's per-move log-prob against spatial features.

    Deterministic: the permutation null is drawn from a Generator seeded by
    ``seed``, so the same seed always yields the same null and the regression
    test in tests/test_cheap_harvest.py reproduces exactly. The qualitative
    result is unchanged -- only the randomness is pinned.
    """
    pol = L3Policy(seed=seed)
    gen = np.random.default_rng(seed)
    cents, forws, logps = [], [], []
    for fen in PROBE_FENS:
        b = chess.Board(fen)
        if b.is_game_over(claim_draw=False):
            continue
        dist = pol.move_distribution(b)
        for mv in b.legal_moves:
            idx = M.move_to_index(mv)
            cents.append(centrality(mv.to_square))
            forws.append(forwardness(mv.to_square, b.turn))
            logps.append(float(np.log(dist[idx] + 1e-12)))
    arr = np.array(logps)
    c_r = float(np.corrcoef(arr, np.array(cents))[0, 1])
    f_r = float(np.corrcoef(arr, np.array(forws))[0, 1])
    # Permutation null: shuffle log-probs across moves 200 times.
    null_c, null_f = [], []
    for _ in range(200):
        perm = gen.permutation(arr)
        null_c.append(float(np.corrcoef(perm, np.array(cents))[0, 1]))
        null_f.append(float(np.corrcoef(perm, np.array(forws))[0, 1]))
    return c_r, f_r, float(np.mean(null_c)), float(np.mean(null_f))


def main():
    seeds = range(20)
    res = [correlate(s) for s in seeds]
    c = np.array([r[0] for r in res])
    f = np.array([r[1] for r in res])
    nc = np.array([r[2] for r in res])
    nf = np.array([r[3] for r in res])
    print(f"centrality  r = {c.mean():+.4f}  (perm-null {nc.mean():+.4f})")
    print(f"forwardness r = {f.mean():+.4f}  (perm-null {nf.mean():+.4f})")
    print(f"separartion from null: cent {abs(c.mean()-nc.mean()):.4f}, fwd {abs(f.mean()-nf.mean()):.4f}")


if __name__ == "__main__":
    main()
