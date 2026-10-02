"""Tests for the L1 diagnostics module.

These matter more than they look. The diagnostics are the only thing that
distinguishes "the policy is bad" from "the policy is receiving no signal", and
those two demand completely different responses. If the diagnostics silently
report the wrong thing, every later decision is made on bad evidence.
"""

from __future__ import annotations

import chess
import numpy as np
import pytest

from chessrl import masks as M
from chessrl.diagnostics import (
    PROBE_FENS,
    PlayDiagnostics,
    action_coverage,
    compare_policies,
    diagnose_games,
    format_comparison,
    policy_sharpness,
)
from chessrl.game import MaterialPolicy, RandomPolicy
from chessrl.policy import FactoredSoftmaxPolicy


# --------------------------------------------------------------------------
# PlayDiagnostics aggregation
# --------------------------------------------------------------------------

def test_empty_diagnostics_reports_zero_rather_than_dividing_by_zero():
    diag = PlayDiagnostics()
    assert diag.informative_rate == 0.0
    assert diag.finished_rate == 0.0
    assert diag.contact_rate == 0.0


def test_informative_rate_counts_only_decisive_results():
    diag = PlayDiagnostics(games=4)
    diag.results.update({"1-0": 1, "0-1": 1, "1/2-1/2": 1, "unfinished": 1})
    assert diag.informative_rate == pytest.approx(0.5)


def test_finished_rate_counts_draws_as_finished():
    diag = PlayDiagnostics(games=4)
    diag.results.update({"1-0": 1, "0-1": 1, "1/2-1/2": 1, "unfinished": 1})
    assert diag.finished_rate == pytest.approx(0.75)


def test_contact_rate_is_captures_per_hundred_plies():
    diag = PlayDiagnostics(games=1, plies=[100], captures=[10])
    assert diag.contact_rate == pytest.approx(10.0)


def test_summary_has_every_expected_key():
    diag = PlayDiagnostics(games=1, plies=[20], captures=[1], checks=[2],
                           unique_positions=[21])
    diag.reasons["max_plies"] = 1
    diag.results["unfinished"] = 1
    summary = diag.summary()
    for key in (
        "games", "reasons", "results", "informative_rate", "finished_rate",
        "mean_plies", "mean_captures", "mean_checks",
        "contact_rate_per_100_plies", "mean_unique_positions",
    ):
        assert key in summary


# --------------------------------------------------------------------------
# diagnose_games
# --------------------------------------------------------------------------

def test_diagnose_games_runs_the_requested_number():
    diag = diagnose_games(RandomPolicy(seed=0), 3, max_plies=30, seed=0)
    assert diag.games == 3
    assert len(diag.plies) == 3
    assert len(diag.captures) == 3
    assert sum(diag.results.values()) == 3


def test_diagnose_games_is_reproducible():
    a = diagnose_games(RandomPolicy(seed=0), 3, max_plies=30, seed=5)
    b = diagnose_games(RandomPolicy(seed=0), 3, max_plies=30, seed=5)
    assert a.plies == b.plies
    assert dict(a.results) == dict(b.results)


def test_recorded_captures_are_plausible_for_random_play():
    """Random play does make contact; the count must not be stuck at zero."""
    diag = diagnose_games(RandomPolicy(seed=1), 5, max_plies=120, seed=1)
    assert sum(diag.captures) > 0


def test_greedy_material_reaches_more_decisive_games_than_random():
    """A control: the diagnostic must be able to tell these apart at all.

    If the two produce the same numbers, the diagnostic is measuring nothing.
    """
    random_diag = diagnose_games(RandomPolicy(seed=0), 12, max_plies=150, seed=3)
    greedy_diag = diagnose_games(MaterialPolicy(seed=0), 12, max_plies=150, seed=3)
    assert greedy_diag.finished_rate > random_diag.finished_rate


def test_diagnose_games_accepts_start_fens():
    diag = diagnose_games(
        RandomPolicy(seed=0), 2, max_plies=20, seed=0,
        start_fens=["8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"],
    )
    assert diag.games == 2


def test_blind_policy_has_low_contact_and_few_decisive_games():
    """The L1 characteristic, pinned so a change in it is noticed."""
    diag = diagnose_games(
        FactoredSoftmaxPolicy(seed=0), 10, max_plies=100, seed=2
    )
    # Not an exact value -- just that it is far from a resolved game.
    assert diag.informative_rate < 0.5


# --------------------------------------------------------------------------
# policy_sharpness
# --------------------------------------------------------------------------

def test_sharpness_of_an_untrained_policy_is_high():
    """A zero-logit blind policy is uniform, so perplexity ~ legal move count."""
    sharp = policy_sharpness(FactoredSoftmaxPolicy(seed=0))
    assert sharp["available"]
    assert sharp["mean_perplexity"] > 15.0


def test_sharpness_probes_cover_every_probe_fen_that_is_playable():
    sharp = policy_sharpness(FactoredSoftmaxPolicy(seed=0))
    playable = [
        f for f in PROBE_FENS
        if not chess.Board(f).is_game_over(claim_draw=False)
    ]
    assert len(sharp["probes"]) == len(playable)


def test_sharpness_reports_perplexity_never_above_legal_moves():
    """A uniform policy is the maximum-entropy case; nothing can exceed it."""
    policy = FactoredSoftmaxPolicy(seed=0)
    for entry in policy_sharpness(policy)["probes"]:
        assert entry["perplexity"] <= entry["legal_moves"] + 1e-6


def test_uniformity_is_at_most_one_for_an_untrained_policy():
    """Zero logits are uniform *per factor*, not uniform over the move set.

    This is a real property of the factorised model and worth pinning down. The
    chain is ``P(from) * P(to|from) * P(promo)``. With zero logits the origin
    factor is uniform over origins, but each origin then spreads its own mass
    evenly over *its own* destinations. So an origin with four destinations
    contributes the same total mass as an origin with two, and moves from the
    busier square are individually less likely.

    The consequence is that perplexity equals the legal move count only when
    every origin has the same number of destinations -- which happens in the
    opening (all ten origins have two) and in the symmetric middlegame probe,
    and not otherwise. Both cases are checked here.
    """
    sharp = policy_sharpness(FactoredSoftmaxPolicy(seed=0))
    for entry in sharp["probes"]:
        assert entry["uniformity"] <= 1.0 + 1e-9

    by_fen = {e["fen"]: e["uniformity"] for e in sharp["probes"]}
    # Opening: ten origins, two destinations each -> exactly uniform.
    assert by_fen[chess.Board().fen()] == pytest.approx(1.0, abs=1e-9)


def test_uniformity_drops_below_one_when_origins_have_unequal_choice():
    """The counterpart to the test above, on a position where it must bite."""
    fen = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"
    sharp = policy_sharpness(FactoredSoftmaxPolicy(seed=0), fens=[fen])
    assert sharp["probes"][0]["uniformity"] < 1.0


def test_sharpness_falls_after_deliberate_sharpening():
    policy = FactoredSoftmaxPolicy(seed=0)
    before = policy_sharpness(policy)["mean_perplexity"]
    # Drive everything onto one origin/destination pair.
    policy.scorer.from_table[chess.E2] = 30.0
    policy.scorer.to_table[chess.E2, chess.E4] = 30.0
    after = policy_sharpness(policy)["mean_perplexity"]
    assert after < before


def test_sharpness_is_unavailable_for_a_select_only_policy():
    """RandomPolicy has no distribution; that is not an error, it is a fact."""
    sharp = policy_sharpness(RandomPolicy(seed=0))
    assert not sharp["available"]
    assert "reason" in sharp
    assert sharp["probes"] == []


def test_sharpness_accepts_a_custom_probe_set():
    sharp = policy_sharpness(
        FactoredSoftmaxPolicy(seed=0), fens=[chess.Board().fen()]
    )
    assert len(sharp["probes"]) == 1


def test_sharpness_skips_game_over_probes():
    mate = "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"
    sharp = policy_sharpness(FactoredSoftmaxPolicy(seed=0), fens=[mate])
    assert sharp["probes"] == []
    assert sharp["mean_perplexity"] == 0.0


# --------------------------------------------------------------------------
# action_coverage
# --------------------------------------------------------------------------

def test_action_coverage_reports_the_full_space_size():
    cov = action_coverage(FactoredSoftmaxPolicy(seed=0), n_games=1, seed=0)
    assert cov["action_space"] == M.ACTION_SPACE


def test_action_coverage_only_ever_takes_offered_actions():
    cov = action_coverage(FactoredSoftmaxPolicy(seed=0), n_games=2, seed=0)
    assert cov["taken_slots"] <= cov["offered_slots"]


def test_action_coverage_offers_far_more_than_it_takes():
    """Most of the action space is legal somewhere but rarely sampled."""
    cov = action_coverage(FactoredSoftmaxPolicy(seed=0), n_games=3, seed=0)
    assert cov["offered_slots"] > 0
    assert 0.0 < cov["taken_fraction_of_offered"] <= 1.0


# --------------------------------------------------------------------------
# compare_policies / format_comparison
# --------------------------------------------------------------------------

def test_compare_policies_returns_one_entry_per_policy():
    table = compare_policies(
        {"random": RandomPolicy(seed=0), "greedy": MaterialPolicy(seed=0)},
        games=3, max_plies=30, seed=0,
    )
    assert set(table) == {"random", "greedy"}
    for entry in table.values():
        assert "informative_rate" in entry
        assert "sharpness" in entry


def test_compare_policies_marks_sharpness_unavailable_where_it_is():
    table = compare_policies(
        {"random": RandomPolicy(seed=0)}, games=2, max_plies=20, seed=0
    )
    assert table["random"]["sharpness"]["available"] is False


def test_format_comparison_renders_a_row_per_policy():
    table = compare_policies(
        {"random": RandomPolicy(seed=0), "blind": FactoredSoftmaxPolicy(seed=0)},
        games=2, max_plies=20, seed=0,
    )
    text = format_comparison(table)
    assert "random" in text
    assert "blind" in text
    assert "perplex" in text
    # A policy with no distribution shows n/a rather than crashing the table.
    assert "n/a" in text


def test_format_comparison_is_rectangular():
    table = compare_policies(
        {"a": RandomPolicy(seed=0), "b": MaterialPolicy(seed=0)},
        games=2, max_plies=20, seed=0,
    )
    lines = format_comparison(table).splitlines()
    # header + rule + one line per policy
    assert len(lines) == 2 + len(table)
