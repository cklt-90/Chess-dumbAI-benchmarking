import numpy as np
import pytest

from bench.scripts.l3_target_rule_ablation import (
    _BUILD_TARGET,
    _final_means,
    _seed_ci,
    build_arms,
)


def test_max_rule_assigns_best_move_target_one():
    weights = np.asarray([1.0, 0.5, 0.25], dtype=np.float64)
    targets = _BUILD_TARGET["max"](weights)
    assert targets[0] == pytest.approx(1.0)
    assert np.all(targets <= 1.0)


def test_sum_rule_is_a_proper_distribution():
    weights = np.asarray([1.0, 0.5, 0.25], dtype=np.float64)
    targets = _BUILD_TARGET["sum"](weights)
    assert targets.sum() == pytest.approx(1.0)
    # No single move is assigned target 1.0, so the best move is not pushed to
    # certainty -- this is the whole point of the alternative rule.
    assert targets[0] < 1.0


def test_sum_rule_preserves_shape_but_bounds_total_target_mass_to_one():
    # Both rules scale the same weight vector, so they preserve its *shape*
    # (the ratios between moves). What differs is the total mass: the max rule
    # can assign up to k units of target probability across the listed moves,
    # while the sum rule never exceeds 1. That over-large target is the
    # proposed cause of the persistent upward push on every listed move.
    weights = np.asarray([1.0, 0.5100057221966218, 0.4360492863215356,
                          0.3114032239145977, 0.12286529741718302], dtype=np.float64)
    max_targets = _BUILD_TARGET["max"](weights)
    sum_targets = _BUILD_TARGET["sum"](weights)
    # Identical shape: the ratio between the best and worst target is the same.
    assert (sum_targets[0] / sum_targets[-1]) == pytest.approx(
        max_targets[0] / max_targets[-1], rel=1e-9)
    # Different scale: max reaches 1.0 and sums above 1; sum sums to exactly 1.
    assert max_targets[0] == pytest.approx(1.0)
    assert max_targets.sum() > 1.0
    assert sum_targets.sum() == pytest.approx(1.0)
    assert sum_targets[0] < 1.0


def test_full_k_sum_rule_matches_max_rule_ratio_direction():
    # Sanity: on a flat weight vector both rules agree, because max == sum/k.
    weights = np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    assert _BUILD_TARGET["max"](weights) == pytest.approx([1.0, 1.0, 1.0, 1.0])
    assert _BUILD_TARGET["sum"](weights) == pytest.approx([0.25, 0.25, 0.25, 0.25])


def test_final_means_averages_each_metric_across_positions():
    rows = [
        {"teacher_topk_probability_mass": 0.2, "teacher_best_probability_mass_tie_aware": 0.1,
         "teacher_weighted_probability_score": 0.0, "teacher_weighted_cross_entropy_nats": 1.0,
         "mean_legal_move_entropy_nats": 2.0},
        {"teacher_topk_probability_mass": 0.4, "teacher_best_probability_mass_tie_aware": 0.3,
         "teacher_weighted_probability_score": 0.0, "teacher_weighted_cross_entropy_nats": 3.0,
         "mean_legal_move_entropy_nats": 4.0},
    ]
    means = _final_means(rows)
    assert means["teacher_topk_probability_mass"] == pytest.approx(0.3)
    assert means["teacher_best_probability_mass_tie_aware"] == pytest.approx(0.2)
    assert means["teacher_weighted_cross_entropy_nats"] == pytest.approx(2.0)
    assert means["mean_legal_move_entropy_nats"] == pytest.approx(3.0)


def test_three_arms_are_defined_with_expected_rules_and_scaling():
    # The magnitude-matched arm must reuse the max rule but request a scale
    # factor, so that the step-size confound is separated from the target rule.
    arms = build_arms(mag_scale=0.3742)
    assert arms["max"] == ("max", 1.0)
    assert arms["sum"] == ("sum", 1.0)
    assert arms["max_scaled"] == ("max", 0.3742)


def test_sweep_mode_keeps_the_max_rule_and_varies_only_the_step():
    sweep = build_arms(mag_scale=0.3742, sweep=[1.0, 0.5, 0.25, 0.125])
    # Sweep mode is defined solely by the multipliers; every arm uses the max
    # rule, and a sweep containing 1.0 does not duplicate an arm.
    assert set(sweep) == {"max@1", "max@0.5", "max@0.25", "max@0.125"}
    assert all(rule == "max" for rule, _ in sweep.values())
    # Only the search_weight multiplier changes between sweep arms.
    multipliers = sorted(scale for _, scale in sweep.values())
    assert multipliers == [0.125, 0.25, 0.5, 1.0]


def test_sweep_mode_deduplicates_repeated_multipliers():
    sweep = build_arms(mag_scale=0.3742, sweep=[0.5, 1.0, 0.5])
    assert set(sweep) == {"max@1", "max@0.5"}


def test_seed_ci_is_zero_width_for_a_single_seed():
    result = _seed_ci([0.5])
    assert result["mean"] == pytest.approx(0.5)
    assert result["sd"] == pytest.approx(0.0)
    assert result["n_seeds"] == 1
    assert result["ci95"] == [pytest.approx(0.5), pytest.approx(0.5)]


def test_seed_ci_uses_t_interval_for_multiple_seeds():
    # Three seeds with a known spread; df=2 -> t=4.303.
    values = [0.0, 0.1, 0.2]
    result = _seed_ci(values)
    assert result["n_seeds"] == 3
    assert result["mean"] == pytest.approx(0.1)
    half = 4.303 * result["sd"] / np.sqrt(3)
    assert result["ci95"][0] == pytest.approx(0.1 - half)
    assert result["ci95"][1] == pytest.approx(0.1 + half)
