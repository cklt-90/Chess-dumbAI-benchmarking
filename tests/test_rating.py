"""Rating-fit tests: does the Bradley-Terry fit return the *right* numbers?

Why these tests assert analytic values rather than "it converged"
-----------------------------------------------------------------
Every bug this fit has ever had was a *scale* bug, and every one of them
produced a result that looked fine -- the optimiser converged, the ratings were
monotone in the wins, the table printed. What was wrong was the magnitude: a
10-0 sweep reporting 0.0 Elo, or 3127, or 825 when the true answer is 370.

"Converged" is therefore worthless as an assertion here. The tests below
compare against closed-form maxima for the two-entrant case, which is the only
situation where the answer is known independently:

    two entrants, ``n`` identical results, prior precision ``p``
    maximise ``n * log sigmoid(theta) - p * theta**2 / 4``

which has a unique root in ``theta`` (the prior makes it strictly concave and
bounded) that can be found by bisection to machine precision, independently of
the fit's own solver. If the fit and the bisection disagree, one of them has a
scale error -- and it will not be the bisection, because it has no scales in it.

The second class of test is structural: a cyclic set of results must *not* be
given a rating order, a ladder must, and the anchor must not change any
difference.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from bench.rating import (
    _ELO_SCALE,
    _LOGISTIC_SCALE,
    MatchRecord,
    Rating,
    expected_score,
    fit_bradley_terry,
)


# --------------------------------------------------------------------------
# an independent oracle
# --------------------------------------------------------------------------

def _sweep_elo_bisect(n: int, prior: float = 1.0) -> float:
    """Elo gap of a clean ``n``-game two-entrant sweep, by bisection.

    Solves ``f(theta) = n * (1 - sigmoid(theta)) - prior * theta / 2 = 0`` where
    ``theta`` is the gap in logistic units (each entrant sits at ``±theta/2``,
    so the prior gradient on the gap is ``prior * theta / 2``).

    Deliberately does not touch anything in ``rating.py`` except the constant
    that defines the logistic scale: this is the check on the fit, not a
    restatement of it.
    """
    lo, hi = 0.0, 1e4
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        f = n * (1.0 - 1.0 / (1.0 + math.exp(-mid))) - prior * mid / 2.0
        if f > 0:
            lo = mid
        else:
            hi = mid
    return (0.5 * (lo + hi)) / _LOGISTIC_SCALE


def _fit(records, names, **kw):
    return fit_bradley_terry(records, names, **kw)


# --------------------------------------------------------------------------
# the calibration: the numbers that every scale bug got wrong
# --------------------------------------------------------------------------

@pytest.mark.parametrize("n", [3, 6, 10, 20, 100, 500])
def test_clean_sweep_matches_the_analytic_maximum(n):
    """A sweep must fit at the closed-form optimum, not merely near it."""
    res = _fit([MatchRecord("X", "Y", wins=n)], ["X", "Y"], anchor="Y")
    want = _sweep_elo_bisect(n)
    got = res.ratings[0].elo
    assert res.converged
    assert got == pytest.approx(want, abs=0.05), (
        f"{n}-game sweep fitted at {got:.2f}, analytic optimum {want:.2f}"
    )


def test_sweep_rating_grows_with_the_length_of_the_sweep():
    """More evidence of a lopsided result means a larger rating, not the same.

    This is the property a mis-scaled prior destroys in the most confusing way:
    with the prior too strong every sweep fits at ~0 regardless of length, and
    the table looks like a tuning quirk rather than an error.
    """
    elos = [
        _fit([MatchRecord("X", "Y", wins=n)], ["X", "Y"], anchor="Y").ratings[0].elo
        for n in (3, 10, 100, 1000)
    ]
    assert elos == sorted(elos)
    assert elos[0] < elos[-1]
    # The growth is logarithmic in the game count, so it must be *far*
    # sub-linear: 333x the games (3 -> 1000) is worth well under an order of
    # magnitude of rating, not 333x. A prior that scaled linearly in games
    # would fail this; so would a prior that has no effect at all (flat).
    ratio = elos[-1] / elos[0]
    assert 2.0 < ratio < 10.0, f"3 games -> {elos[0]:.1f}, 1000 games -> {elos[-1]:.1f}"


def test_even_match_fits_at_zero():
    res = _fit([MatchRecord("X", "Y", wins=5, losses=5)], ["X", "Y"], anchor="Y")
    assert res.ratings[0].elo == pytest.approx(0.0, abs=1e-6)
    assert res.ratings[1].elo == pytest.approx(0.0, abs=1e-6)


def test_all_draws_fits_at_zero():
    res = _fit([MatchRecord("X", "Y", draws=10)], ["X", "Y"], anchor="Y")
    assert res.ratings[0].elo == pytest.approx(0.0, abs=1e-6)


def test_draws_are_between_a_win_and_a_loss():
    """Half a win to each side must land strictly between the two extremes."""
    sweep = _fit([MatchRecord("X", "Y", wins=10)], ["X", "Y"], anchor="Y")
    half = _fit([MatchRecord("X", "Y", wins=5, draws=5)], ["X", "Y"], anchor="Y")
    even = _fit([MatchRecord("X", "Y", wins=5, losses=5)], ["X", "Y"], anchor="Y")
    assert even.ratings[0].elo < half.ratings[0].elo < sweep.ratings[0].elo


def test_no_nan_or_inf_for_any_reasonable_input():
    """The whole point of the prior. An undefeated entrant has an infinite MLE
    and a naive fit returns ``inf`` or, worse, ``nan``."""
    cases = [
        [MatchRecord("X", "Y", wins=10_000)],
        [MatchRecord("X", "Y", wins=1)],
        [MatchRecord("X", "Y", losses=999)],
        [MatchRecord("X", "Y", wins=1, draws=98, losses=1)],
    ]
    for recs in cases:
        res = _fit(recs, ["X", "Y"], anchor="Y")
        for r in res.ratings:
            assert math.isfinite(r.elo), (recs, r)
            assert math.isfinite(r.stderr), (recs, r)


def test_the_prior_bounds_an_undefeated_entrant():
    for n in (10, 100, 1000, 10_000):
        res = _fit([MatchRecord("X", "Y", wins=n)], ["X", "Y"], anchor="Y")
        assert res.ratings[0].elo < 2000.0, n


# --------------------------------------------------------------------------
# structure: order, cycles, anchoring, resiliance to representation
# --------------------------------------------------------------------------

def test_ladder_is_ordered_correctly():
    recs = [
        MatchRecord("A", "B", wins=8, losses=2),
        MatchRecord("B", "C", wins=8, losses=2),
        MatchRecord("A", "C", wins=9, losses=1),
    ]
    res = _fit(recs, ["A", "B", "C"], anchor="C")
    by = {r.name: r.elo for r in res.ratings}
    assert by["A"] > by["B"] > by["C"] == 0.0


def test_a_cycle_is_not_given_an_order():
    """Three-way rock-paper-scissors: every entrant wins 8 and loses 8 against
    one opponent from the other, so the data contains no ranking and the fit
    must say so (equal ratings) rather than inventing one from the order the
    records happen to be listed in."""
    recs = [
        MatchRecord("A", "B", wins=8, losses=2),
        MatchRecord("B", "C", wins=8, losses=2),
        MatchRecord("C", "A", wins=8, losses=2),
    ]
    res = _fit(recs, ["A", "B", "C"])
    elos = [r.elo for r in res.ratings]
    assert max(elos) - min(elos) == pytest.approx(0.0, abs=1e-6)


def test_anchor_only_shifts_the_level_not_the_differences():
    recs = [
        MatchRecord("A", "B", wins=8, losses=2),
        MatchRecord("B", "C", wins=8, losses=2),
        MatchRecord("A", "C", wins=9, losses=1),
    ]
    a = {r.name: r.elo for r in _fit(recs, ["A", "B", "C"], anchor="C").ratings}
    b = {r.name: r.elo for r in _fit(recs, ["A", "B", "C"], anchor="A").ratings}
    assert a["A"] - a["B"] == pytest.approx(b["A"] - b["B"], abs=1e-6)
    assert b["A"] == pytest.approx(0.0, abs=1e-6)
    assert a["C"] == pytest.approx(0.0, abs=1e-6)


def test_order_of_records_does_not_change_the_fit():
    recs = [
        MatchRecord("A", "B", wins=8, losses=2),
        MatchRecord("B", "C", wins=3, losses=7),
        MatchRecord("A", "C", wins=5, losses=5),
    ]
    forward = {r.name: r.elo for r in _fit(recs, ["A", "B", "C"], anchor="C").ratings}
    backward = {
        r.name: r.elo
        for r in _fit(list(reversed(recs)), ["C", "B", "A"], anchor="C").ratings
    }
    for name in ("A", "B", "C"):
        assert forward[name] == pytest.approx(backward[name], abs=1e-6)


def test_unfinished_games_are_not_evidence():
    """A pile of unfinished games must not move the fitted rating at all."""
    clean = _fit([MatchRecord("X", "Y", wins=6)], ["X", "Y"], anchor="Y")
    padded = _fit(
        [MatchRecord("X", "Y", wins=6, unfinished=500)], ["X", "Y"], anchor="Y"
    )
    assert padded.ratings[0].elo == pytest.approx(clean.ratings[0].elo, abs=1e-9)


def test_entrant_with_no_decided_games_still_appears():
    recs = [MatchRecord("A", "B", wins=6), MatchRecord("A", "C", unfinished=40)]
    res = _fit(recs, ["A", "B", "C"], anchor="C")
    by = {r.name: r for r in res.ratings}
    assert set(by) == {"A", "B", "C"}
    assert by["C"].games == 0
    assert by["C"].unfinished == 40


def test_more_games_shrink_the_standard_error():
    few = _fit([MatchRecord("X", "Y", wins=5, losses=5)], ["X", "Y"])
    many = _fit([MatchRecord("X", "Y", wins=50, losses=50)], ["X", "Y"])
    se_few = next(r.stderr for r in few.ratings if r.name == "X")
    se_many = next(r.stderr for r in many.ratings if r.name == "X")
    assert se_many < se_few


def test_standard_error_has_the_elo_scale_not_the_logistic_scale():
    """The information is the second derivative of the *logistic* likelihood, so
    the standard error must be converted back to Elo units. Forgetting the
    conversion understates every error bar by ~174x, which would make a six-game
    match look like a precise measurement.

    Two entrants, one game each way, edge-to-edge: the exact standard error is
    ``1 / (c * sqrt(2 * p * (1-p)))`` with ``p = 0.5``, i.e. ``1/c`` scaled by
    ``2``.
    """
    res = _fit([MatchRecord("X", "Y", wins=1, losses=1)], ["X", "Y"], prior=1.0)
    want = 1.0 / (_LOGISTIC_SCALE * math.sqrt(2 * 0.25 + 1.0))
    got = next(r.stderr for r in res.ratings if r.name == "X")
    assert got == pytest.approx(want, rel=1e-9)
    # and the scale is the big one: an Elo-scale error bar, not a logistic one
    assert got > 100.0


# --------------------------------------------------------------------------
# bookkeeping
# --------------------------------------------------------------------------

def test_report_counts_are_aggregated_from_both_sides():
    recs = [
        MatchRecord("A", "B", wins=3, draws=2, losses=1, unfinished=4),
        MatchRecord("A", "C", wins=1, draws=0, losses=4),
    ]
    res = _fit(recs, ["A", "B", "C"], anchor="C")
    by = {r.name: r for r in res.ratings}
    assert (by["A"].wins, by["A"].draws, by["A"].losses) == (4, 2, 5)
    assert (by["B"].wins, by["B"].losses) == (1, 3)     # A's losses are B's wins
    assert by["A"].unfinished == 4
    assert by["A"].games == by["A"].wins + by["A"].draws + by["A"].losses


def test_empty_input_is_handled():
    assert fit_bradley_terry([], []).ratings == []
    assert fit_bradley_terry([], ["A", "B"]).ratings == []
    assert fit_bradley_terry([MatchRecord("A", "B", unfinished=3)], ["A", "B"]).ratings == []


def test_records_for_unknown_entrants_are_ignored():
    recs = [MatchRecord("A", "B", wins=4), MatchRecord("A", "Z", wins=100)]
    res = _fit(recs, ["A", "B"], anchor="B")
    assert {r.name for r in res.ratings} == {"A", "B"}


def test_ratings_are_sorted_descending():
    recs = [
        MatchRecord("A", "B", wins=9, losses=1),
        MatchRecord("B", "C", wins=9, losses=1),
        MatchRecord("A", "C", wins=10, losses=0),
    ]
    res = _fit(recs, ["C", "A", "B"], anchor="C")
    elos = [r.elo for r in res.ratings]
    assert elos == sorted(elos, reverse=True)


def test_residuals_are_sorted_by_how_badly_the_model_misses():
    recs = [
        MatchRecord("A", "B", wins=5, losses=5),     # well explained
        MatchRecord("B", "C", wins=9, losses=1),     # not
    ]
    res = _fit(recs, ["A", "B", "C"])
    gaps = [abs(obs - exp) for _, _, obs, exp, _ in res.residuals]
    assert gaps == sorted(gaps, reverse=True)


# --------------------------------------------------------------------------
# expected_score: the public conversion, on the Elo scale
# --------------------------------------------------------------------------

def test_expected_score_is_the_elo_curve():
    assert expected_score(0, 0) == pytest.approx(0.5)
    assert expected_score(400, 0) == pytest.approx(10 / 11)   # 10:1 by definition
    assert expected_score(0, 400) == pytest.approx(1 / 11)
    assert expected_score(200, 0) + expected_score(0, 200) == pytest.approx(1.0)


def test_expected_score_is_monotone_and_bounded():
    xs = np.linspace(-3000, 3000, 601)
    ps = np.array([expected_score(x, 0.0) for x in xs])
    assert np.all(np.diff(ps) > 0)
    assert ps.min() > 0.0 and ps.max() < 1.0
    assert np.all(np.isfinite(ps))


def test_the_elo_scale_is_the_conventional_400():
    assert _ELO_SCALE == 400.0
    assert _LOGISTIC_SCALE == pytest.approx(math.log(10) / 400)
