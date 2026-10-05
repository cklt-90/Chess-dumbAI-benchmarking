"""H18 — at depth 2-3, are L2's mistakes tactical or positional?

Samples random midgame positions (random walks from the start, skipping terminal
states) and compares the move chosen by shallow search (d2/d3) against a deep
reference (d5). Each disagreement is classified by whether a capture is legal
within 2 plies of the divergence point. The literature settles the principle
(quiescence exists to blunt the horizon effect); this measures the *magnitude at
your depths*. No new model, no training.
"""
import chess
import random

from chessrl.search import MinimaxEngine


def move_at(board, depth):
    return MinimaxEngine(depth=depth).select(board)


def capture_near(board, plies=2):
    def rec(b, d):
        if d == 0:
            return False
        for m in b.legal_moves:
            if b.is_capture(m):
                return True
            b.push(m)
            r = rec(b, d - 1)
            b.pop()
            if r:
                return True
        return False

    return rec(board.copy(), plies)


def sample_positions(n=100, walk=10, seed=0):
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


def main():
    fens = sample_positions(n=100, walk=10, seed=0)
    for shallow in (2, 3):
        tactical = 0
        total = 0
        for fen in fens:
            b = chess.Board(fen)
            ds = move_at(b, shallow)
            d5 = move_at(b, 5)
            if ds != d5:
                total += 1
                if capture_near(b):
                    tactical += 1
        frac = (tactical / total) if total else float("nan")
        print(
            f"d{shallow}-vs-d5: {total}/{len(fens)} positions disagree; "
            f"tactical fraction = {tactical}/{total} = {frac:.2f}"
        )


if __name__ == "__main__":
    main()
