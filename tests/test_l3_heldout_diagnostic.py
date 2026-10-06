import math

import pytest

from bench.scripts.l3_heldout_diagnostic import (
    _approximate_seed_count,
    _summarise,
    _t_critical_95,
)


def test_t_critical_values_cover_table_and_large_df_fallback():
    assert _t_critical_95(2) == pytest.approx(12.706)
    assert _t_critical_95(31) == pytest.approx(2.042)
    assert _t_critical_95(32) == pytest.approx(2.05)
    with pytest.raises(ValueError):
        _t_critical_95(1)


def test_approximate_seed_count_uses_t_based_threshold():
    sd, mde = 0.2, 0.05
    n = _approximate_seed_count(sd, mde)
    assert n is not None and n >= 2
    assert math.sqrt(n) * mde / sd >= _t_critical_95(n) + 0.84
    assert math.sqrt(n - 1) * mde / sd < _t_critical_95(n - 1) + 0.84


def test_zero_pilot_variance_does_not_claim_a_sample_size():
    assert _approximate_seed_count(0.0, 0.05) is None


def test_summary_uses_seed_paired_differences_and_reports_scope():
    seed_results = [
        {"checkpoints": [
            {"positions_seen": 0, "metrics": {"teacher_topk_probability_mass": before}},
            {"positions_seen": 10, "metrics": {"teacher_topk_probability_mass": after}},
        ]}
        for before, after in ((0.2, 0.3), (0.2, 0.4), (0.2, 0.5))
    ]
    result = _summarise(seed_results, final_checkpoint=10, mde=0.05)
    assert result["training_seed_replications"] == 3
    assert result["mean_seed_paired_gain"] == pytest.approx(0.2)
    assert result["sd_seed_paired_gain"] == pytest.approx(0.1)
    assert result["standard_error_across_seeds"] == pytest.approx(0.1 / math.sqrt(3))
    assert result["approx_95pct_t_interval_for_mean_gain"] == pytest.approx(
        [0.2 - 4.303 * 0.1 / math.sqrt(3), 0.2 + 4.303 * 0.1 / math.sqrt(3)]
    )
    assert "fixed training corpus and held-out set" in result["interval_scope"]
