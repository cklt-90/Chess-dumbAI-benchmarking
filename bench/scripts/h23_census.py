"""H23-fix: re-run the special-move census after the encode/value edits.

The baseline row lives in HYPOTHESES.md (measured 2026-10-02, ply cap 120-160):

    | level          | castling | en passant | promotion |
    | random         | 1/6      | 2          | 0         |
    | L1 blind       | 1/6      | 0          | 0         |
    | material       | 3/6      | 0          | 0         |
    | L2-d2 minimax  | 6 (3/3)  | 0          | 0         |
    | L3 untrained   | 1/6      | 0          | 0         |

H23's claim: castling is legal but under-played because nothing rewards it -- no
castling-rights channel in ``encode`` and no king-shelter term in
``value.evaluate``. This script reproduces that census after the fix (two appended
castling-rights planes + a king-shelter eval term) so the two can be compared.

The interesting readout: blind (L1) and untrained (L3) should stay at ~1/6
(their move choice does not consult the new channel or the new eval term), while
material and L2 -- which DO consult ``evaluate`` -- may now castle more, because
the shelter term gives them a reason. The encode channel only changes behaviour
once a learner is *trained* on it, which is out of scope for this no-training
harvest; it is locked by tests/test_encode.py instead.

Run from the repo root with the managed interpreter:

    ./.venv/Scripts/python.exe bench/scripts/h23_census.py
"""
import random
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import chess

from chessrl.game import play_game, RandomPolicy, MaterialPolicy
from chessrl.search import MinimaxPolicy
from chessrl.perceptron import L3Policy
from chessrl.policy import FactoredSoftmaxPolicy


def count_special(moves):
    """Replay a SAN move list and tally castling / en-passant / promotion moves."""
    board = chess.Board()
    castle = ep = promo = 0
    for san in moves:
        move = board.parse_san(san)
        if board.is_castling(move):
            castle += 1
        if board.is_en_passant(move):
            ep += 1
        if move.promotion:
            promo += 1
        board.push(move)
    return castle, ep, promo


def census(make_policy, games: int = 6, max_plies: int = 120, seed: int = 0):
    tot = [0, 0, 0]
    for g in range(games):
        result = play_game(
            make_policy(), make_policy(),
            max_plies=max_plies, rng=random.Random(seed + g),
        )
        c, e, p = count_special(result.moves)
        tot[0] += c
        tot[1] += e
        tot[2] += p
    return tuple(tot)


LEVELS = {
    "random": lambda: RandomPolicy(),
    "L1-blind": lambda: FactoredSoftmaxPolicy(seed=1),
    "material": lambda: MaterialPolicy(),
    "L2-d2": lambda: MinimaxPolicy(depth=2),
    "L3-untrained": lambda: L3Policy(seed=1),
}


def main() -> None:
    print(f"{'level':<14} {'castling':>10} {'en_passant':>12} {'promotion':>11}")
    print("-" * 50)
    for name, make in LEVELS.items():
        c, e, p = census(make)
        print(f"{name:<14} {c:>10} {e:>12} {p:>11}")
    print("\n(Compare against the baseline row in HYPOTHESES.md / CHEAP-HARVEST.md.)")


if __name__ == "__main__":
    main()
