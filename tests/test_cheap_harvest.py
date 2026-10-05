"""Cheap-harvest regression tests (H21, H18).

These wrap the two no-training experiments in ``bench/scripts/`` as reusable
regression checks. The canonical *measurement* -- the full 100-position sweep
and the exact numbers reported in HYPOTHESES.md -- lives in those scripts and is
run on demand. Here we assert only the qualitative conclusions, so a later
refactor that quietly changes the untrained prior or the search horizon fails
loudly instead of silently.

Conventions follow tests/test_perceptron.py: plain functions, pytest asserts,
no training, no new models. Neither test imports torch.
"""
import sys
from pathlib import Path

import chess
import numpy as np
import pytest
import random

# The experiment code lives in bench/scripts, outside the configured pythonpath.
# Import it directly so this file stays the single source of truth and the
# scripts are not duplicated here.
SCRIPTS = Path(__file__).resolve().parents[1] / "bench" / "scripts"
sys.path.insert(0, str(SCRIPTS))
import h18_horizon  # noqa: E402
import h21_prior  # noqa: E402

# H13/H3 re-fit against the stored matrix; bench/ is outside the configured
# pythonpath, so add the repo root for ``bench.rating``.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from bench.rating import MatchRecord, fit_bradley_terry  # noqa: E402


# --------------------------------------------------------------------------
# H21 -- an untrained factorised policy carries a centre bias, not a forward bias
# --------------------------------------------------------------------------

def test_untrained_l3_has_central_prior_not_forward_prior():
    """H21: the geometry features give a central lean, not a forward lean.

    An untrained L3 has board *inputs* but random *weights*, so its move
    log-probabilities are shaped only by the geometry features. Those include a
    centrality term (distance to board centre) but no rank-advance term, so we
    expect a positive correlation with centrality and *no* correlation with
    forwardness. Assert both at once, with margins wide enough to tolerate the
    permutation null's sampling noise.
    """
    central_gaps, forward_gaps = [], []
    for seed in range(20):
        c_r, f_r, c_null, f_null = h21_prior.correlate(seed)
        central_gaps.append(c_r - c_null)
        forward_gaps.append(f_r - f_null)

    central_gap = float(np.mean(central_gaps))
    forward_gap = float(np.mean(forward_gaps))

    # A real, non-trivial central bias, clearly separated from the null.
    assert central_gap > 0.02, f"central bias collapsed: {central_gap:.4f}"
    # No forward bias: the gap must be indistinguishable from the null.
    assert abs(forward_gap) < 0.02, f"unexpected forward bias: {forward_gap:.4f}"
    # And the two must be different in *kind*, not both small effects.
    assert central_gap > 3.0 * abs(forward_gap) + 1e-6


# --------------------------------------------------------------------------
# H18 -- shallow search's disagreements with deep search are tactical
# --------------------------------------------------------------------------

def test_shallow_search_mistakes_are_tactical():
    """H18: at depth 2-3, L2's errors are tactical, not positional.

    For each shallow depth we compare the chosen move against a deep reference
    over sampled midgame positions. Every disagreement is classified by whether
    a capture is available within two plies. The literature settles the
    principle (quiescence exists to blunt the horizon effect); this guards the
    *magnitude at our depths*: disagreements must be overwhelmingly tactical and
    the comparison must not be vacuous.

    The canonical sweep uses a depth-5 reference over 100 positions (see
    bench/scripts/h18_horizon.py); the test uses depth 4 and a smaller sample
    for speed while asserting the same qualitative conclusion.
    """
    fens = h18_horizon.sample_positions(n=18, walk=8, seed=0)
    assert len(fens) == 18

    for shallow in (2, 3):
        tactical = 0
        total = 0
        for fen in fens:
            board = chess.Board(fen)
            chosen = h18_horizon.move_at(board, shallow)
            deep = h18_horizon.move_at(board, 4)
            if chosen != deep:
                total += 1
                if h18_horizon.capture_near(board):
                    tactical += 1
        # Not vacuous: the shallow engine must actually differ sometimes.
        assert total > 0, (
            f"d{shallow} never disagreed with the deep reference -- "
            "the engine or sample changed; check bench/scripts/h18_horizon.py"
        )
        frac = tactical / total
        assert frac >= 0.8, (
            f"d{shallow}: only {frac:.2f} of {total} disagreements were tactical "
            f"({tactical} tactical) -- shallow errors are no longer horizon errors"
        )


# --------------------------------------------------------------------------
# H13/H3 -- does counting unfinished games as draws move the rating?
# --------------------------------------------------------------------------

def test_unfinished_excluded_by_default_is_unchanged():
    """The repo-wide convention: ply-capped (unfinished) games are not evidence.

    A pile of unfinished games must leave the fitted rating exactly where it was
    with no unfinished games -- this is the load-bearing invariant ``bench/rating.py``
    exists to protect, and it must hold whether or not the experiment flag exists.
    """
    clean = fit_bradley_terry(
        [MatchRecord("X", "Y", wins=6)], ["X", "Y"], anchor="Y"
    )
    padded = fit_bradley_terry(
        [MatchRecord("X", "Y", wins=6, unfinished=500)], ["X", "Y"], anchor="Y"
    )
    assert padded.ratings[0].elo == pytest.approx(clean.ratings[0].elo, abs=1e-9)


def test_counting_unfinished_as_draws_moves_the_rating():
    """H3: scoring a ply-capped game as a draw is not neutral -- it drags ratings
    toward zero and fakes precision. When the experiment flag is on, 200 unfinished
    games (counted as 100 draws) must pull a 6-win entrant's rating down hard,
    and must not equal the default fit.
    """
    recs = [MatchRecord("X", "Y", wins=6, unfinished=200)]
    base = fit_bradley_terry(recs, ["X", "Y"], anchor="Y")
    flag = fit_bradley_terry(
        recs, ["X", "Y"], anchor="Y", include_unfinished_as_draws=True
    )
    x_base = next(r.elo for r in base.ratings if r.name == "X")
    x_flag = next(r.elo for r in flag.ratings if r.name == "X")

    # Default: 6 decided wins, unfinished ignored -> X clearly ahead of Y.
    assert x_base > 200.0
    # Flag: the result is dominated by the faked draws -> X collapses toward zero.
    assert x_flag < x_base - 50.0


# --------------------------------------------------------------------------
# H17 -- does transition-tagging change what the store samples?
# --------------------------------------------------------------------------

def test_record_game_tags_near_transition_moves():
    """H17: a capture or pawn move played from a position must flag it.

    ``record_game`` recovers the move between consecutive FENs and tags the
    position as near a transition when that move was a capture or a pawn push.
    The flank moves (knight/bishop) in this sequence must not be flagged.
    """
    from chessrl.cache import MidstateStore

    store = MidstateStore()
    board = chess.Board()
    fens = [board.fen()]
    for uci in ("e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6"):
        board.push(chess.Move.from_uci(uci))
        fens.append(board.fen())
    store.record_game(fens, store.new_game_id())

    assert any(e.near_transition for e in store)
    flagged = [e for e in store if e.near_transition]
    # e2e4, e7e5 and a7a6 are pawn pushes; the two knight/bishop moves are not.
    assert 2 <= len(flagged) <= 4


def test_sample_near_transition_prefers_transition_positions():
    """H17: the transition sampler must prioritise flagged positions over quiet ones."""
    from chessrl.cache import Midstate, MidstateStore

    store = MidstateStore()
    # 6 near-transition (all early phase) + 6 quiet (all early phase).
    for i in range(6):
        store._append(Midstate(
            fen=chess.Board().fen(), ply=i, game_id=0, movecount=1,
            material_ratio=1.0, zobrist=i, near_transition=True,
        ))
    for i in range(6):
        store._append(Midstate(
            fen=chess.Board().fen(), ply=10 + i, game_id=1, movecount=40,
            material_ratio=1.0, zobrist=100 + i, near_transition=False,
        ))

    trans = store.sample_near_transition(6, rng=random.Random(0))
    uni = store.sample(6, rng=random.Random(0))
    frac_trans = sum(e.near_transition for e in trans) / len(trans)
    frac_uni = sum(e.near_transition for e in uni) / len(uni)

    assert len(trans) == 6
    # The whole transition sample is flagged; uniform sampling over a 50/50 pool is not.
    assert frac_trans == 1.0
    assert frac_trans > frac_uni
