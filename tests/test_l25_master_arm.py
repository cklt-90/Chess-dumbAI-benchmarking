import math

import pytest

from bench.scripts.l25_master_arm import (
    _target_entropy,
    _to_target_rows,
    label_diagnostics,
    label_from_cache,
)


def test_target_entropy_uniform_and_point_mass():
    assert _target_entropy([1.0, 1.0, 1.0, 1.0]) == pytest.approx(math.log(4))
    # A single-move target has zero entropy.
    assert _target_entropy([1.0]) == pytest.approx(0.0)


def test_target_entropy_is_scale_invariant():
    # Entropy depends on the shape, not the absolute scale of the weights.
    assert _target_entropy([2.0, 2.0]) == pytest.approx(_target_entropy([1.0, 1.0]))


def _rec(fen, *pairs):
    return {"fen": fen, "targets": [{"uci": u, "weight": w} for u, w in pairs]}


def test_label_diagnostics_counts_top1_agreement_and_top5_membership():
    weak = [_rec("a", ("e2e4", 1.0), ("d2d4", 0.5)),
            _rec("b", ("g1f3", 1.0), ("b1c3", 0.4))]
    true = [_rec("a", ("e2e4", 1.0), ("g1f3", 0.3)),   # agree on top-1
            _rec("b", ("e2e4", 1.0), ("d2d4", 0.2))]   # disagree; g1f3 not in top-5
    diag = label_diagnostics(weak, true)
    assert diag["positions"] == 2
    assert diag["top1_agreement_rate"] == pytest.approx(0.5)
    # weak top-1 ('g1f3') is absent from true's top-5 on position b.
    assert diag["weak_top1_in_true_top5_rate"] == pytest.approx(0.5)


def test_label_from_cache_uses_fen_and_reports_drops():
    records = [{"fen": "a"}, {"fen": "b"}, {"fen": "c"}]
    by_fen = {"a": [{"uci": "e2e4", "weight": 1.0}],
              "b": [{"uci": "d2d4", "weight": 1.0}]}
    labelled, dropped = label_from_cache(records, by_fen)
    assert [r["fen"] for r in labelled] == ["a", "b"]
    assert dropped == 1
    assert labelled[0]["targets"][0]["uci"] == "e2e4"


def test_to_target_rows_formats_moves_and_floats():
    import chess

    rows = _to_target_rows([(chess.Move.from_uci("e2e4"), 1.0),
                            (chess.Move.from_uci("d2d4"), 0.5)])
    assert rows == [{"uci": "e2e4", "weight": 1.0}, {"uci": "d2d4", "weight": 0.5}]
