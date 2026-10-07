"""Focused tests for the H23 behavioural harness (bench/scripts/h23_behavioural.py).

These cover the *mechanics* of the registered census -- which moves count as
castling, that the curated set really is filtered and deduped, that the oracle
flags are consistent with the search labels, and that the census reports a rate
in [0, 1] and a mass only when the policy can produce a distribution. They do
not assert any chess conclusion; that is the experiment's job, not a test's.
"""
from __future__ import annotations

import chess
import numpy as np
import pytest

from bench.scripts.h23_behavioural import (
    _castle_mass,
    _discrimination_summary,
    apply_negative_castling,
    build_corpora,
    census,
    class_in_targets,
    class_mass_over_uniform,
    class_moves,
    class_rank_variance_report,
    conditioning_summary,
    d3_label_fn,
    fit_linear_probe,
    legal_castling_moves,
    match_corpus_sizes,
    move_in_class,
    oracle_label,
    probe_feature_rows,
    probe_scores,
    rank_bucket,
    rank_within,
    roc_auc,
    scan_castling_positions,
    spearman,
    split_records,
    train_and_evaluate,
)


def _board(pieces, castling=0, turn=chess.WHITE) -> chess.Board:
    board = chess.Board.empty()
    for square, piece in pieces:
        board.set_piece_at(square, piece)
    board.castling_rights = castling
    board.turn = turn
    return board


def _kings_and_rooks(white_rooks, black_king=True):
    pieces = [(chess.E1, chess.Piece(chess.KING, chess.WHITE))]
    for sq in white_rooks:
        pieces.append((sq, chess.Piece(chess.ROOK, chess.WHITE)))
    if black_king:
        pieces.append((chess.E8, chess.Piece(chess.KING, chess.BLACK)))
    return pieces


def test_legal_castling_moves_both_wings():
    board = _board(_kings_and_rooks([chess.A1, chess.H1]),
                   castling=chess.BB_A1 | chess.BB_H1)
    moves = legal_castling_moves(board)
    assert len(moves) == 2
    assert all(board.is_castling(move) for move in moves)


def test_legal_castling_moves_one_wing():
    # Only the a1 rook and the queenside right -> exactly one castling move.
    board = _board(_kings_and_rooks([chess.A1]), castling=chess.BB_A1)
    moves = legal_castling_moves(board)
    assert len(moves) == 1
    assert board.is_castling(moves[0])


def test_start_position_has_no_castling_move():
    assert legal_castling_moves(chess.Board()) == []


def test_scan_returns_filtered_and_deduped_positions():
    records = scan_castling_positions(20, seed=7, min_since_capture=4)
    assert records, "expected at least one castling-legal position in 20 games"
    keys = set()
    for record in records:
        board = chess.Board(record["fen"])
        # rights intact + at least one legal castling move
        assert board.has_castling_rights(board.turn)
        assert legal_castling_moves(board)
        assert record["n_castles"] == len(legal_castling_moves(board))
        # >=4 plies since the last capture, as tracked during replay
        assert record["plies_since_capture"] >= 4
        from bench.scripts.l3_heldout_diagnostic import _position_key
        keys.add(_position_key(board))
    assert len(keys) == len(records), "scan produced duplicate positions"


def test_oracle_flags_match_search_labels():
    records = scan_castling_positions(20, seed=11, min_since_capture=4)[:6]
    labeled = oracle_label(records, d3_label_fn(2, 5))
    for record in labeled:
        board = chess.Board(record["fen"])
        top = chess.Move.from_uci(record["targets"][0]["uci"])
        assert record["oracle_best_is_castle"] == board.is_castling(top)
        expected_topk = any(
            board.is_castling(chess.Move.from_uci(row["uci"]))
            for row in record["targets"]
        )
        assert record["oracle_topk_has_castle"] == expected_topk


def test_census_reports_rate_in_unit_interval_and_mass_for_distribution_policy():
    from chessrl.perceptron import L3Policy

    records = scan_castling_positions(20, seed=13, min_since_capture=4)[:5]
    labeled = oracle_label(records, d3_label_fn(2, 5))
    table = census(lambda: L3Policy(seed=1), labeled)
    assert table["positions"] == len(labeled)
    assert 0.0 <= table["select_castle_rate"] <= 1.0
    mass = table["castling_probability_mass"]
    assert mass is not None and 0.0 <= mass <= 1.0


def test_census_reports_no_mass_for_pure_search_policy():
    from chessrl.search import MinimaxPolicy

    records = scan_castling_positions(20, seed=17, min_since_capture=4)[:3]
    labeled = oracle_label(records, d3_label_fn(2, 5))
    table = census(lambda: MinimaxPolicy(depth=1), labeled)
    # A search policy has no move_distribution, so mass is reported as None.
    assert table["castling_probability_mass"] is None
    assert 0.0 <= table["select_castle_rate"] <= 1.0


def test_split_records_is_disjoint_and_covers_all():
    records = [{"fen": f"f{i}"} for i in range(10)]
    train, held = split_records(records, heldout_fraction=0.34, seed=1)
    assert len(train) + len(held) == 10
    assert len(held) == 3
    assert {r["fen"] for r in train}.isdisjoint({r["fen"] for r in held})


def test_train_and_evaluate_wiring_produces_bounded_metrics():
    records = scan_castling_positions(20, seed=23, min_since_capture=4)[:6]
    labeled = oracle_label(records, d3_label_fn(2, 5))
    result = train_and_evaluate(
        labeled, {"rich_held": labeled}, [1, 2], depth=2, top_k=5, search_weight=0.125)
    assert len(result["per_seed"]) == 2
    summary = result["summary"]["rich_held"]["castling_probability_mass"]
    assert 0.0 <= summary["trained_mean"] <= 1.0
    assert len(summary["seed_level_paired_differences"]) == 2
    assert "interval_scope" in result["summary"]["rich_held"]


def test_rank_bucket_boundaries():
    assert rank_bucket(1) == "1"
    assert rank_bucket(2) == "2-3"
    assert rank_bucket(3) == "2-3"
    assert rank_bucket(4) == "4-5"
    assert rank_bucket(5) == "4-5"
    assert rank_bucket(6) == "6-10"
    assert rank_bucket(10) == "6-10"
    assert rank_bucket(11) == "11-20"
    assert rank_bucket(20) == "11-20"
    assert rank_bucket(21) == ">20"


def test_spearman_perfect_and_inverse():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)


def _synthetic(n, tag):
    return [{"fen": f"fen-{tag}-{i}", "oracle_topk_has_castle": True} for i in range(n)]


def test_roc_auc_perfect_separation_and_chance():
    y = np.array([0.0, 0.0, 1.0, 1.0])
    assert roc_auc(np.array([0.1, 0.2, 0.8, 0.9]), y) == pytest.approx(1.0)
    assert roc_auc(np.array([0.9, 0.8, 0.2, 0.1]), y) == pytest.approx(0.0)
    # Ties: every score identical => no ranking information => chance.
    assert roc_auc(np.array([0.5, 0.5, 0.5, 0.5]), y) == pytest.approx(0.5)


def test_roc_auc_is_tie_aware():
    y = np.array([0.0, 0.0, 1.0, 1.0])
    # Two ties, each straddling a class boundary: 0.5 credit per pair.
    assert roc_auc(np.array([0.5, 0.5, 0.5, 0.5]), y) == pytest.approx(0.5)
    assert roc_auc(np.array([0.1, 0.1, 0.1, 0.9]), y) == pytest.approx(0.75)


def test_probe_recovers_a_planted_signal():
    """Positive control: the probe must find a signal that is actually there.

    Without this, a low AUC on the real label is ambiguous -- it could mean
    "no signal" or "the probe is broken". Here the label is a threshold on one
    of the context features themselves, so a working probe must separate it.
    """
    records = oracle_label(scan_castling_positions(30, seed=41, min_since_capture=4)[:60],
                           d3_label_fn(1, 5))
    x, _ = probe_feature_rows(records, variant="context")
    # Index 4 is mobility/40 -- literally one of the features, so this is a
    # lower bound on the probe's competence, not a chess claim.
    planted = (x[:, 4] > np.median(x[:, 4])).astype(float)
    w, mean, sd = fit_linear_probe(x, planted, steps=1500)
    assert roc_auc(probe_scores(x, w, mean, sd), planted) > 0.95


def test_probe_variants_have_expected_widths():
    records = oracle_label(scan_castling_positions(20, seed=43, min_since_capture=4)[:20],
                           d3_label_fn(1, 5))
    x_context, _ = probe_feature_rows(records, variant="context")
    x_local, _ = probe_feature_rows(records, variant="context_local")
    x_board, _ = probe_feature_rows(records, variant="full_board_pooled")
    assert x_context.shape[1] == 9          # the nine global scalars
    assert x_board.shape[1] == 24           # one mean per channel
    # Local adds the origin and destination channel vectors (24 each).
    assert x_local.shape[1] == 9 + 48
    assert len(x_board) == len(x_context)


def test_probe_variant_rejects_unknown_name():
    records = oracle_label(scan_castling_positions(10, seed=47, min_since_capture=4)[:5],
                           d3_label_fn(1, 5))
    with pytest.raises(ValueError):
        probe_feature_rows(records, variant="nonsense")


def test_apply_negative_castling_is_inert_where_teacher_wants_castling():
    from chessrl.perceptron import L3Config, L3Trainer

    trainer = L3Trainer(config=L3Config(seed=1))
    fen = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"
    board = chess.Board(fen)
    record = {"fen": fen, "oracle_topk_has_castle": True,
              "targets": [{"uci": "e1g1", "weight": 1.0}]}
    before = _castle_mass(trainer.policy, board)
    assert apply_negative_castling(trainer, record) == 0
    assert _castle_mass(trainer.policy, board) == pytest.approx(before)


def test_apply_negative_castling_reduces_castling_mass_where_teacher_rejected():
    from chessrl.perceptron import L3Config, L3Trainer

    trainer = L3Trainer(config=L3Config(seed=1))
    fen = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"
    board = chess.Board(fen)
    # Teacher targets a quiet king move, never castling -> POOR class.
    record = {"fen": fen, "oracle_topk_has_castle": False,
              "targets": [{"uci": "e1e2", "weight": 1.0}]}
    before = _castle_mass(trainer.policy, board)
    credited = apply_negative_castling(trainer, record)
    assert credited == 2  # both wings are legal here
    after = _castle_mass(trainer.policy, board)
    assert after < before


def test_build_corpora_keeps_train_and_held_disjoint():
    rich = _synthetic(40, "r")
    poor = _synthetic(60, "p")
    c = build_corpora(rich, poor, 0.25, 9001)
    for train_key in ("rich_train", "poor_train", "mixed_half_train", "mixed_full_train"):
        train_fens = {r["fen"] for r in c[train_key]}
        for held_key in ("rich_held", "poor_held"):
            assert not (train_fens & {r["fen"] for r in c[held_key]}), (train_key, held_key)


def test_build_corpora_mixed_sizes_and_class_balance():
    rich = _synthetic(40, "r")
    poor = _synthetic(60, "p")
    c = build_corpora(rich, poor, 0.25, 9001)
    n = c["n_matched"]
    # mixed_half is volume-matched to RICH; mixed_full is RICH-dose-matched.
    assert len(c["mixed_half_train"]) == 2 * (n // 2)
    assert len(c["mixed_full_train"]) == 2 * n
    # Both mixed corpora contain BOTH classes -- that is the whole point: the
    # RICH arm's training data has no negative example at all.
    for key in ("mixed_half_train", "mixed_full_train"):
        fens = {r["fen"] for r in c[key]}
        assert sum(1 for f in fens if f.startswith("fen-r-")) > 0
        assert sum(1 for f in fens if f.startswith("fen-p-")) > 0
    # The pure RICH corpus contains only RICH: the confound being controlled.
    assert all(r["fen"].startswith("fen-r-") for r in c["rich_train"])
    assert all(r["fen"].startswith("fen-p-") for r in c["poor_train"])


def test_build_corpora_is_deterministic():
    rich = _synthetic(30, "r")
    poor = _synthetic(30, "p")
    a = build_corpora(rich, poor, 0.25, 1234)
    b = build_corpora(rich, poor, 0.25, 1234)
    for key in ("rich_train", "poor_train", "mixed_half_train", "mixed_full_train",
                "rich_held", "poor_held"):
        assert [r["fen"] for r in a[key]] == [r["fen"] for r in b[key]]


def test_discrimination_summary_is_zero_when_both_sets_agree():
    per_seed = [
        {"trained": {"rich_held": {"castling_probability_mass": 0.5},
                     "poor_held": {"castling_probability_mass": 0.5}},
         "untrained": {"rich_held": {"castling_probability_mass": 0.02},
                       "poor_held": {"castling_probability_mass": 0.02}}},
        {"trained": {"rich_held": {"castling_probability_mass": 0.6},
                     "poor_held": {"castling_probability_mass": 0.6}},
         "untrained": {"rich_held": {"castling_probability_mass": 0.03},
                       "poor_held": {"castling_probability_mass": 0.03}}},
    ]
    summary = _discrimination_summary(per_seed)
    # Both seeds: identical mass on both sets => no discrimination, even though
    # the castling *level* rose a lot. This is exactly the blanket-bias case.
    assert summary["trained_mean"] == pytest.approx(0.0)
    assert summary["untrained_mean"] == pytest.approx(0.0)
    assert summary["paired_gain_mean"] == pytest.approx(0.0)


def test_discrimination_summary_is_positive_when_only_rich_rises():
    per_seed = [
        {"trained": {"rich_held": {"castling_probability_mass": 0.50},
                     "poor_held": {"castling_probability_mass": 0.03}},
         "untrained": {"rich_held": {"castling_probability_mass": 0.02},
                       "poor_held": {"castling_probability_mass": 0.02}}},
        {"trained": {"rich_held": {"castling_probability_mass": 0.60},
                     "poor_held": {"castling_probability_mass": 0.04}},
         "untrained": {"rich_held": {"castling_probability_mass": 0.03},
                       "poor_held": {"castling_probability_mass": 0.03}}},
    ]
    summary = _discrimination_summary(per_seed)
    assert summary["trained_mean"] == pytest.approx(0.515)
    assert summary["paired_gain_mean"] > 0.4
    assert len(summary["seed_level_paired_differences"]) == 2


def test_conditioning_summary_buckets_and_bounds():
    from chessrl.perceptron import L3Policy

    records = scan_castling_positions(20, seed=29, min_since_capture=4)[:6]
    ranked = [{**r, "rank": i + 1, "gap_cp": 10 * (i + 1)} for i, r in enumerate(records)]
    summary = conditioning_summary([L3Policy(seed=1)], ranked)
    assert summary["positions"] == 6
    assert sum(block["n"] for block in summary["buckets"].values()) == 6
    rho = summary["spearman_teacher_rank_vs_castling_mass"]
    assert rho is None or -1.0 <= rho <= 1.0


def test_move_in_class_separates_castling_from_capture():
    # The two-class conditioning test is only meaningful if the classes are
    # actually disjoint and each recognises its own moves.
    board = _board([
        (chess.E1, chess.Piece(chess.KING, chess.WHITE)),
        (chess.H1, chess.Piece(chess.ROOK, chess.WHITE)),
        (chess.C3, chess.Piece(chess.PAWN, chess.WHITE)),
        (chess.D4, chess.Piece(chess.PAWN, chess.BLACK)),
        (chess.E8, chess.Piece(chess.KING, chess.BLACK)),
    ], castling=chess.BB_H1, turn=chess.WHITE)
    castle = chess.Move.from_uci("e1g1")
    capture = chess.Move.from_uci("c3d4")
    assert move_in_class(board, castle, "castling")
    assert not move_in_class(board, castle, "capture")
    assert move_in_class(board, capture, "capture")
    assert not move_in_class(board, capture, "castling")
    assert class_moves(board, "castling") == [castle]
    assert capture in class_moves(board, "capture")
    with pytest.raises(ValueError):
        move_in_class(board, castle, "nonsense")


def test_rank_within_finds_the_best_member_or_reports_beyond_multipv():
    a, b, c = (chess.Move.from_uci(uci) for uci in ("a2a3", "b2b3", "c2c3"))
    scored = [(a, 50), (b, 40), (c, 30)]
    assert rank_within(scored, {b}, 20) == {"rank": 2, "gap_cp": 10}
    # A member the engine did not list is ranked just past the end, with the gap
    # measured against the worst line rather than silently reported as rank 1.
    missing = rank_within(scored, {chess.Move.from_uci("d2d4")}, 20)
    assert missing == {"rank": 4, "gap_cp": 20}
    # No engine output at all must give the sentinel, not a crash.
    assert rank_within([], {a}, 20) == {"rank": 21, "gap_cp": None}


def test_class_rank_variance_report_flags_a_saturated_class():
    # A class whose best member ranks 1 everywhere has no spread, so a flat
    # correlation against it would be an instrument artefact. The report has to
    # say that explicitly instead of leaving it to be inferred from a near-zero
    # rho -- this is the gate that decides whether the two-class comparison can
    # be run at all.
    rows = [
        {"castling_legal": 2, "castling_rank": 1, "castling_gap_cp": 0},
        {"castling_legal": 2, "castling_rank": 1, "castling_gap_cp": 0},
        {"capture_legal": 5, "capture_rank": 1, "capture_gap_cp": 0},
        {"capture_legal": 5, "capture_rank": 7, "capture_gap_cp": 120},
    ]
    report = class_rank_variance_report(rows, ("castling", "capture"))
    assert report["castling"]["usable_rank_variance"] is False
    assert report["castling"]["rank_1_share"] == pytest.approx(1.0)
    assert report["castling"]["distinct_ranks"] == 1
    assert report["capture"]["usable_rank_variance"] is True
    assert report["capture"]["distinct_ranks"] == 2
    assert report["capture"]["gap_cp_zero_share"] == pytest.approx(0.5)


class _UniformPolicy:
    """A policy with no preference at all -- uniform over legal moves.

    Used to pin the normalisation: by construction, an indifferent policy must
    score exactly 1.0 on `class_mass_over_uniform` for every move class.
    """

    def move_distribution(self, board):
        from chessrl.masks import ACTION_SPACE, move_to_index

        legal = list(board.legal_moves)
        distribution = np.zeros(ACTION_SPACE, dtype=np.float64)
        for move in legal:
            distribution[move_to_index(move)] = 1.0 / len(legal)
        return distribution


def test_class_mass_over_uniform_is_one_for_an_indifferent_policy():
    # This is what makes castling and captures comparable at all: raw mass is
    # dominated by how many moves the class contains (1-2 castles vs many
    # captures), so the normalised value is the only cross-class-readable one.
    board = _board([
        (chess.E1, chess.Piece(chess.KING, chess.WHITE)),
        (chess.H1, chess.Piece(chess.ROOK, chess.WHITE)),
        (chess.C3, chess.Piece(chess.PAWN, chess.WHITE)),
        (chess.D4, chess.Piece(chess.PAWN, chess.BLACK)),
        (chess.E8, chess.Piece(chess.KING, chess.BLACK)),
    ], castling=chess.BB_H1, turn=chess.WHITE)
    policy = _UniformPolicy()
    assert class_mass_over_uniform(policy, board, "castling") == pytest.approx(1.0)
    assert class_mass_over_uniform(policy, board, "capture") == pytest.approx(1.0)


def test_class_mass_over_uniform_exceeds_one_when_the_class_is_favoured():
    class _Favouring:
        def move_distribution(self, board):
            from chessrl.masks import ACTION_SPACE, move_to_index

            distribution = np.zeros(ACTION_SPACE, dtype=np.float64)
            castles = [m for m in board.legal_moves if board.is_castling(m)]
            others = [m for m in board.legal_moves if not board.is_castling(m)]
            for move in castles:
                distribution[move_to_index(move)] = 0.5 / len(castles)
            for move in others:
                distribution[move_to_index(move)] = 0.5 / len(others)
            return distribution

    board = _board([
        (chess.E1, chess.Piece(chess.KING, chess.WHITE)),
        (chess.H1, chess.Piece(chess.ROOK, chess.WHITE)),
        (chess.E8, chess.Piece(chess.KING, chess.BLACK)),
    ], castling=chess.BB_H1, turn=chess.WHITE)
    value = class_mass_over_uniform(_Favouring(), board, "castling")
    assert value > 1.0


def test_class_in_targets_reads_the_teacher_list():
    board = _board([
        (chess.E1, chess.Piece(chess.KING, chess.WHITE)),
        (chess.H1, chess.Piece(chess.ROOK, chess.WHITE)),
        (chess.E8, chess.Piece(chess.KING, chess.BLACK)),
    ], castling=chess.BB_H1, turn=chess.WHITE)
    with_castle = {"targets": [{"uci": "e1g1", "weight": 1.0}]}
    without = {"targets": [{"uci": "h1h2", "weight": 1.0}]}
    assert class_in_targets(board, with_castle, "castling") is True
    assert class_in_targets(board, without, "castling") is False


def test_match_corpus_sizes_caps_both_classes_and_is_deterministic():
    # Unmatched corpora would confound "does this class condition?" with "how
    # much data did this class get", so the sizes must be forced equal.
    per_class = {
        "castling": ([{"id": i} for i in range(9)], [{"id": i} for i in range(5)]),
        "capture": ([{"id": i} for i in range(3)], [{"id": i} for i in range(7)]),
    }
    matched = match_corpus_sizes(per_class, cap=4, seed=11)
    assert [len(r) for r, _ in matched.values()] == [4, 3]
    assert [len(p) for _, p in matched.values()] == [4, 4]
    # Same seed -> same subsample; a different seed is allowed to differ.
    again = match_corpus_sizes(per_class, cap=4, seed=11)
    assert matched == again
