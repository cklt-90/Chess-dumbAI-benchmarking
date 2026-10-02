"""L1 tests: the blind factored softmax policy and its counters."""

from __future__ import annotations

import chess
import numpy as np
import pytest

from chessrl import masks as M
from chessrl.policy import (
    BlindScorer,
    FactoredSoftmaxPolicy,
    ProbCounter,
    entropy,
    masked_softmax,
    move_probability,
    perplexity,
    uniform_over_mask,
)
from conftest import INTERESTING_FENS


# --------------------------------------------------------------------------
# masked_softmax: the numerical core everything else depends on
# --------------------------------------------------------------------------

def test_masked_softmax_is_zero_on_masked_entries():
    logits = np.array([1.0, 2.0, 3.0, 4.0])
    mask = np.array([True, False, True, False])
    probs = masked_softmax(logits, mask)
    assert probs[1] == 0.0 and probs[3] == 0.0
    assert probs[0] > 0.0 and probs[2] > 0.0
    assert probs.sum() == pytest.approx(1.0)


def test_masked_softmax_returns_zeros_for_empty_mask():
    probs = masked_softmax(np.ones(8), np.zeros(8, dtype=bool))
    assert np.all(probs == 0.0)
    # Not NaN -- callers use all-zero as the "no legal move" signal.
    assert not np.isnan(probs).any()


def test_masked_softmax_is_stable_with_large_negative_logits():
    # Once a policy is confident its logits go very negative. Without the
    # max-subtraction this underflows to zero and produces NaN on divide.
    logits = np.array([-800.0, -801.0, -799.0])
    probs = masked_softmax(logits, np.ones(3, dtype=bool))
    assert probs.sum() == pytest.approx(1.0)
    assert probs[2] > probs[0] > probs[1]


def test_masked_softmax_uniform_fallback_when_all_logits_are_neg_inf():
    logits = np.array([-np.inf, -np.inf, -np.inf, -np.inf])
    mask = np.array([True, True, False, False])
    probs = masked_softmax(logits, mask)
    assert probs[0] == pytest.approx(0.5)
    assert probs[1] == pytest.approx(0.5)
    assert probs[2] == 0.0 and probs[3] == 0.0


def test_masked_softmax_matches_naive_softmax_without_masking():
    logits = np.array([0.3, -1.2, 2.0, 0.7])
    mask = np.ones(4, dtype=bool)
    probs = masked_softmax(logits, mask)
    ref = np.exp(logits - logits.max())
    ref /= ref.sum()
    assert np.allclose(probs, ref)


# --------------------------------------------------------------------------
# BlindScorer
# --------------------------------------------------------------------------

def test_blind_scorer_starts_at_zero_logits():
    scorer = BlindScorer()
    assert np.all(scorer.from_table == 0.0)
    assert np.all(scorer.to_table == 0.0)
    assert np.all(scorer.promo_table == 0.0)


def test_blind_scorer_ignores_the_board():
    scorer = BlindScorer()
    a = chess.Board()
    b = chess.Board("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1")
    # The same table object regardless of position: this is what "blind" means.
    assert scorer.from_logits(a) is scorer.from_logits(b)


def test_blind_scorer_roundtrips_through_state_dict():
    scorer = BlindScorer()
    rng = np.random.default_rng(0)
    scorer.from_table[:] = rng.normal(size=64)
    scorer.to_table[:] = rng.normal(size=(64, 64))
    scorer.promo_table[:] = rng.normal(size=(64, 64, M.NUM_PROMO))

    other = BlindScorer()
    other.load_state_dict(scorer.state_dict())
    assert np.allclose(other.from_table, scorer.from_table)
    assert np.allclose(other.to_table, scorer.to_table)
    assert np.allclose(other.promo_table, scorer.promo_table)


# --------------------------------------------------------------------------
# FactoredSoftmaxPolicy
# --------------------------------------------------------------------------

def test_zero_logits_gives_uniform_over_legal_moves():
    """The initial distribution must equal random play.

    This is the sanity check that lets L1 be diffed against RandomPolicy: a
    zero scorer is uniform over the legal mask, so the two agree exactly.
    """
    board = chess.Board()
    policy = FactoredSoftmaxPolicy(seed=0)
    dist = policy.move_distribution(board)

    legal = [M.move_to_index(m) for m in board.legal_moves]
    n = len(legal)
    assert dist.sum() == pytest.approx(1.0)
    for idx in legal:
        assert dist[idx] == pytest.approx(1.0 / n, abs=1e-9)


def test_distribution_is_zero_on_illegal_moves():
    board = chess.Board()
    policy = FactoredSoftmaxPolicy(seed=0)
    dist = policy.move_distribution(board)
    legal = {M.move_to_index(m) for m in board.legal_moves}
    nonzero = set(np.nonzero(dist)[0].tolist())
    assert nonzero <= legal


def test_distribution_over_interesting_positions_always_sums_to_one():
    policy = FactoredSoftmaxPolicy(seed=0)
    for fen in INTERESTING_FENS:
        board = chess.Board(fen)
        if board.is_game_over(claim_draw=False):
            continue
        dist = policy.move_distribution(board)
        assert dist.sum() == pytest.approx(1.0, abs=1e-9), fen


def test_move_probability_agrees_with_move_distribution():
    """The cheap per-move path and the dense table must not disagree.

    They are computed by different code paths, so a disagreement here is a
    silent bug: training would nudge one probability while evaluation reported
    another.
    """
    policy = FactoredSoftmaxPolicy(seed=0)
    rng = np.random.default_rng(0)
    for fen in INTERESTING_FENS:
        board = chess.Board(fen)
        if board.is_game_over(claim_draw=False):
            continue
        dist = policy.move_distribution(board)
        mask = M.legal_move_mask(board)
        for move in list(board.legal_moves)[:8]:
            idx = M.move_to_index(move)
            cheap = move_probability(policy.scorer, board, mask, move)
            assert cheap == pytest.approx(dist[idx], abs=1e-12), (
                fen, move.uci()
            )


def test_select_returns_a_legal_move_from_every_probe_position():
    policy = FactoredSoftmaxPolicy(seed=3)
    for fen in INTERESTING_FENS:
        board = chess.Board(fen)
        if board.is_game_over(claim_draw=False):
            continue
        move = policy.select(board)
        assert move in board.legal_moves, fen


def test_select_greedy_returns_a_legal_move():
    policy = FactoredSoftmaxPolicy(seed=3, greedy=True)
    for fen in INTERESTING_FENS:
        board = chess.Board(fen)
        if board.is_game_over(claim_draw=False):
            continue
        move = policy.select(board)
        assert move in board.legal_moves, fen


def test_greedy_selects_the_argmax_move():
    """With a hand-set table, greedy must pick the highest-logit legal move."""
    policy = FactoredSoftmaxPolicy(greedy=True)
    board = chess.Board()
    # Make e2-e4 by far the best origin/destination pair.
    policy.scorer.from_table[chess.E2] = 10.0
    policy.scorer.to_table[chess.E2, chess.E4] = 10.0
    move = policy.select(board)
    assert move == chess.Move(chess.E2, chess.E4)


def test_greedy_is_deterministic():
    policy = FactoredSoftmaxPolicy(seed=7, greedy=True)
    policy.scorer.from_table[:] = np.arange(64, dtype=np.float64)
    board = chess.Board()
    picks = {policy.select(board) for _ in range(20)}
    assert len(picks) == 1


def test_sampling_is_reproducible_for_a_fixed_seed():
    board = chess.Board()
    a = FactoredSoftmaxPolicy(seed=42)
    b = FactoredSoftmaxPolicy(seed=42)
    assert [a.select(board) for _ in range(20)] == [
        b.select(board) for _ in range(20)
    ]


def test_sampling_actually_varies():
    board = chess.Board()
    policy = FactoredSoftmaxPolicy(seed=0)
    picks = {policy.select(board) for _ in range(60)}
    assert len(picks) > 5


def test_sample_factors_returns_a_legal_combination():
    policy = FactoredSoftmaxPolicy(seed=1)
    for fen in INTERESTING_FENS:
        board = chess.Board(fen)
        if board.is_game_over(claim_draw=False):
            continue
        mask = M.legal_move_mask(board)
        for _ in range(5):
            f, t, slot = policy.sample_factors(board, mask)
            assert mask[f, t, slot], (fen, f, t, slot)


def test_promotion_factor_is_masked_when_not_a_promotion():
    """A non-promotion move must have zero probability on every promo slot > 0."""
    board = chess.Board()
    policy = FactoredSoftmaxPolicy()
    mask = M.legal_move_mask(board)
    # e2-e4 is a plain pawn push; the only legal slot is the sentinel.
    assert mask[chess.E2, chess.E4, 0]
    assert not mask[chess.E2, chess.E4, 1:].any()

    dist = policy.move_distribution(board)
    for slot in range(1, M.NUM_PROMO):
        idx = (chess.E2 * 64 + chess.E4) * M.NUM_PROMO + slot
        assert dist[idx] == 0.0


def test_underpromotion_slots_are_reachable_in_a_promotion_position():
    board = chess.Board("8/P6k/8/8/8/8/8/K7 w - - 0 1")
    dist = FactoredSoftmaxPolicy().move_distribution(board)
    # a7-a8 with each of N/B/R/Q must be possible; the sentinel slot 0 must not.
    for slot in range(1, M.NUM_PROMO):
        idx = (chess.A7 * 64 + chess.A8) * M.NUM_PROMO + slot
        assert dist[idx] > 0.0, slot
    assert dist[(chess.A7 * 64 + chess.A8) * M.NUM_PROMO] == 0.0


def test_temperature_zero_approaches_greedy():
    policy = FactoredSoftmaxPolicy(temperature=0.05)
    board = chess.Board()
    policy.scorer.from_table[chess.E2] = 5.0
    policy.scorer.to_table[chess.E2, chess.E4] = 5.0
    dist = policy.move_distribution(board)
    best = M.move_to_index(chess.Move(chess.E2, chess.E4))
    assert dist[best] > 0.99


def test_distribution_is_empty_on_a_finished_board():
    mate = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    assert mate.is_game_over()
    dist = FactoredSoftmaxPolicy().move_distribution(mate)
    assert dist.sum() == 0.0


def test_save_and_load_roundtrip(tmp_path):
    policy = FactoredSoftmaxPolicy(seed=0, name="roundtrip")
    rng = np.random.default_rng(0)
    policy.scorer.from_table[:] = rng.normal(size=64)

    path = tmp_path / "model.json"
    policy.save(path)

    loaded = FactoredSoftmaxPolicy.load(path)
    assert loaded.name == "roundtrip"
    assert np.allclose(loaded.scorer.from_table, policy.scorer.from_table)

    board = chess.Board()
    assert np.allclose(
        loaded.move_distribution(board), policy.move_distribution(board)
    )


def test_saved_state_excludes_counters(tmp_path):
    """The spec is explicit: counters are deleted before saving."""
    policy = FactoredSoftmaxPolicy(seed=0)
    import json

    state = policy.state_dict()
    assert "counter" not in state
    assert "counters" not in state

    path = tmp_path / "m.json"
    policy.save(path)
    on_disk = json.loads(path.read_text())
    assert "counter" not in on_disk


# --------------------------------------------------------------------------
# ProbCounter
# --------------------------------------------------------------------------

def test_counter_counts_offers_and_takes_separately():
    board = chess.Board()
    mask = M.legal_move_mask(board)
    counter = ProbCounter()
    counter.update(mask, chess.E2, chess.E4, 0)

    assert counter.offered_from[chess.E2] == 1
    assert counter.taken_from[chess.E2] == 1
    assert counter.taken_to[chess.E2, chess.E4] == 1
    # A legal-but-untaken square is offered but not taken. This is the whole
    # distinction the counter exists to make.
    assert counter.offered_from[chess.D2] == 1
    assert counter.taken_from[chess.D2] == 0


def test_counter_reports_never_offered_squares():
    board = chess.Board()
    counter = ProbCounter()
    # Initially nothing has been offered.
    assert counter.never_offered_from().all()

    counter.note_offer(M.legal_move_mask(board))
    # The opening position offers moves from the eight pawns and four minors.
    never = counter.never_offered_from()
    assert not never[chess.E2]
    assert never[chess.A8]  # nothing has been offered from Black's back rank


def test_prune_candidates_ignores_squares_never_offered():
    """Pruning on probability alone would delete untested squares."""
    counter = ProbCounter()
    probs = np.full(64, 1.0 / 64)
    probs[chess.A1] = 0.0  # improbable, but never offered

    candidates = counter.prune_candidates(
        probs, min_offers=10, prob_threshold=0.01
    )
    assert not candidates.any()


def test_prune_candidates_fires_after_enough_offers():
    counter = ProbCounter()
    counter.offered_from[chess.A1] = 500
    probs = np.full(64, 0.02)
    probs[chess.A1] = 0.001

    candidates = counter.prune_candidates(
        probs, min_offers=200, prob_threshold=0.01
    )
    assert candidates[chess.A1]
    assert candidates.sum() == 1


def test_prune_candidates_respects_min_offers_boundary():
    counter = ProbCounter()
    counter.offered_from[chess.A1] = 199
    probs = np.zeros(64)
    probs[chess.A1] = 0.0
    assert not counter.prune_candidates(
        probs, min_offers=200, prob_threshold=0.01
    ).any()

    counter.offered_from[chess.A1] = 200
    assert counter.prune_candidates(
        probs, min_offers=200, prob_threshold=0.01
    )[chess.A1]


def test_counter_stats_shape():
    counter = ProbCounter()
    counter.note_offer(M.legal_move_mask(chess.Board()))
    counter.note_taken(chess.E2, chess.E4, 0)
    stats = counter.stats()
    # The opening offers moves from 10 squares (8 pawns + 2 knights). The
    # counter tracks origins, not moves, so this is 10 rather than 20.
    assert stats["total_offers"] == 10
    assert stats["total_takes"] == 1
    assert 0.0 < stats["take_rate"] <= 1.0


def test_counter_save_load_roundtrip(tmp_path):
    counter = ProbCounter()
    counter.update(M.legal_move_mask(chess.Board()), chess.E2, chess.E4, 0)
    path = tmp_path / "counter.npz"
    counter.save(path)

    loaded = ProbCounter.load(path)
    assert np.array_equal(loaded.offered_from, counter.offered_from)
    assert np.array_equal(loaded.taken_from, counter.taken_from)
    assert np.array_equal(loaded.taken_to, counter.taken_to)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def test_entropy_of_uniform_is_log_n():
    probs = np.full(8, 1.0 / 8)
    assert entropy(probs) == pytest.approx(np.log(8))


def test_entropy_of_a_point_mass_is_zero():
    probs = np.zeros(8)
    probs[3] = 1.0
    assert entropy(probs) == pytest.approx(0.0)


def test_perplexity_matches_exp_entropy():
    probs = np.array([0.25, 0.25, 0.5])
    assert perplexity(probs) == pytest.approx(np.exp(entropy(probs)))


def test_uniform_over_mask_is_uniform_and_normalised():
    mask = np.zeros(64, dtype=bool)
    mask[[0, 5, 9, 63]] = True
    probs = uniform_over_mask(mask)
    assert probs.sum() == pytest.approx(1.0)
    assert probs[0] == pytest.approx(0.25)
    assert probs[1] == 0.0
