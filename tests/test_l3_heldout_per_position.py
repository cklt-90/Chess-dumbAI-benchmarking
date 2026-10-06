import pytest

from bench.scripts.l3_heldout_per_position import (
    _cross_entropy_concentration,
    _mean_or_none,
    _seed_mean,
    summarise_positions,
    worst_regressions,
)


def _row(fen="x", topk=0.0, best=0.0, wp=0.0, ce=0.0, entropy=0.0, n=5):
    return {
        "fen": fen,
        "teacher_topk_probability_mass": topk,
        "teacher_best_probability_mass_tie_aware": best,
        "teacher_weighted_probability_score": wp,
        "teacher_weighted_cross_entropy_nats": ce,
        "mean_legal_move_entropy_nats": entropy,
        "n_teacher_targets": n,
        "n_teacher_best_moves": 1,
    }


def test_summarise_counts_improve_and_regress_in_teacher_direction():
    baseline = [_row(topk=0.2, ce=2.0, entropy=3.0), _row(topk=0.3, ce=2.0, entropy=3.0),
                _row(topk=0.4, ce=2.0, entropy=3.0)]
    final = [_row(topk=0.5, ce=1.5, entropy=2.5), _row(topk=0.3, ce=2.0, entropy=3.0),
             _row(topk=0.2, ce=2.5, entropy=2.8)]
    result = summarise_positions(baseline, final)

    topk = result["teacher_topk_probability_mass"]
    assert topk["improved_positions"] == 1
    assert topk["unchanged_positions"] == 1
    assert topk["regressed_positions"] == 1
    assert topk["mean_delta"] == pytest.approx((0.3 + 0.0 - 0.2) / 3)
    assert topk["mean_delta_over_improved"] == pytest.approx(0.3)
    assert topk["mean_delta_over_regressed"] == pytest.approx(-0.2)

    # Cross-entropy is lower-is-better: a negative delta is an improvement.
    ce = result["teacher_weighted_cross_entropy_nats"]
    assert ce["improved_positions"] == 1
    assert ce["unchanged_positions"] == 1
    assert ce["regressed_positions"] == 1
    assert ce["mean_delta_over_improved"] == pytest.approx(-0.5)
    assert ce["mean_delta_over_regressed"] == pytest.approx(0.5)

    entropy = result["mean_legal_move_entropy_nats"]
    assert entropy["reduced_entropy_positions"] == 2
    assert entropy["unchanged_positions"] == 1
    assert entropy["increased_entropy_positions"] == 0


def test_summarise_rejects_misaligned_position_sets():
    with pytest.raises(ValueError):
        summarise_positions([_row()], [_row(), _row()])


def test_worst_regressions_orders_by_cross_entropy_increase_and_reports_context():
    baseline = [_row(fen="a", ce=2.0), _row(fen="b", ce=2.0), _row(fen="c", ce=2.0)]
    final = [_row(fen="a", ce=2.1, topk=0.30, best=0.10, entropy=2.5),
             _row(fen="b", ce=3.0, topk=0.05, best=0.01, entropy=1.0),
             _row(fen="c", ce=1.9)]
    worst = worst_regressions(baseline, final, count=2)
    assert [row["fen"] for row in worst] == ["b", "a"]
    top = worst[0]
    assert top["cross_entropy_delta"] == pytest.approx(1.0)
    assert top["baseline_cross_entropy"] == pytest.approx(2.0)
    assert top["final_cross_entropy"] == pytest.approx(3.0)
    assert top["final_topk_mass"] == pytest.approx(0.05)
    assert top["final_best_mass"] == pytest.approx(0.01)
    assert top["final_entropy"] == pytest.approx(1.0)


def test_mean_or_none_is_none_on_empty_selection():
    assert _mean_or_none([]) is None
    assert _mean_or_none([1.0, 3.0]) == pytest.approx(2.0)


def test_seed_mean_averages_each_position_across_seeds():
    seed_a = [_row(fen="p", topk=0.2), _row(fen="q", topk=0.6)]
    seed_b = [_row(fen="p", topk=0.4), _row(fen="q", topk=0.8)]
    averaged = _seed_mean([seed_a, seed_b])
    assert [row["fen"] for row in averaged] == ["p", "q"]
    assert averaged[0]["teacher_topk_probability_mass"] == pytest.approx(0.3)
    assert averaged[1]["teacher_topk_probability_mass"] == pytest.approx(0.7)


def test_seed_mean_rejects_seeds_with_different_heldout_sizes():
    with pytest.raises(ValueError):
        _seed_mean([[_row()], [_row(), _row()]])


def test_concentration_is_zero_when_nothing_regressed():
    baseline = [_row(ce=2.0, entropy=3.0), _row(ce=2.0, entropy=3.0)]
    final = [_row(ce=1.5, entropy=2.5), _row(ce=1.8, entropy=2.7)]
    result = _cross_entropy_concentration(baseline, final)
    assert result["gross_positive_cross_entropy_increase"] == pytest.approx(0.0)
    assert result["top_decile_share"] == pytest.approx(0.0)
    assert result["correlation_ce_rise_with_entropy_fall"] is None


def test_concentration_sums_gross_positive_increase():
    # 10 positions so the worst decile is exactly one position.
    baseline = [_row(ce=2.0, entropy=3.0) for _ in range(10)]
    final = [_row(ce=2.0, entropy=3.0) for _ in range(10)]
    final[0] = _row(ce=5.0, entropy=1.0)   # +3, the whole gross increase
    final[1] = _row(ce=2.5, entropy=2.5)   # +0.5
    final[2] = _row(ce=1.0, entropy=4.0)   # -1 improvement
    result = _cross_entropy_concentration(baseline, final)
    assert result["gross_positive_cross_entropy_increase"] == pytest.approx(3.5)
    assert result["n_positions_worst_decile"] == 1
    assert result["top_decile_share"] == pytest.approx(3.0 / 3.5)
    # Larger CE rises here go with larger entropy falls, so the correlation is
    # strongly negative.
    assert result["correlation_ce_rise_with_entropy_fall"] < -0.9
