"""chess-rl-bench: a harness for benchmarking naive RL approaches to chess.

Layers
------
L0  foundation   bitboard, masks, cache, encode, value
L1  policy       blind factored softmax, most-probable-move sampling
L1.5 training    diminishing-credit updates, prob counters, pruning
L2  search       alpha-beta minimax, non-learning wrapper
L3  perceptron   int-friendly linear model, midstate incremental updates
L4  torch        same model, torch backend
L5  guided       alpha-beta ordered and evaluated by the L4 models
L6  ensemble     all of the above, weighted by predictive accuracy

Every level implements the :class:`chessrl.policy.Policy` protocol, so levels
can be swapped, stacked and played against each other in one tournament.
"""

__version__ = "0.1.0"

WHITE = 1
BLACK = 0

# Piece-plane ordering used throughout. Index = plane number in the (12, 8, 8)
# encoded state. The layout is chosen so that `plane ^ 6` flips colour and
# `plane % 6` yields the piece type, which both packing and evaluation rely on.
PLANE_ORDER = (
    "P", "N", "B", "R", "Q", "K",   # 0-5   white
    "p", "n", "b", "r", "q", "k",   # 6-11  black
)
NUM_PLANES = len(PLANE_ORDER)

__all__ = ["__version__", "WHITE", "BLACK", "PLANE_ORDER", "NUM_PLANES"]
