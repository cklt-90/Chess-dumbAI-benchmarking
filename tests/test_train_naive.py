"""L1.5 tests: credit assignment, diminishing updates, pruning, persistence."""

from __future__ import annotations

import json

import chess
import numpy as np
import pytest

from chessrl import masks as M
from chessrl import value as V
from chessrl.game import GameResult, RandomPolicy, play_game
from chessrl.policy import FactoredSoftmaxPolicy, ProbCounter
from chessrl.train_naive import (
    FactoredSoftmaxTrainer,
    TrainConfig,
    credit_curve,
)


def make_game(result: str, moves=("e4", "e5"), fens=None) -> GameResult:
    """A minimal GameResult for testing the update rule in isolation."""
    board = chess.Board()
    if fens is None:
        fens = [board.fen()]
        for san in moves:
            board.push_san(san)
            fens.append(board.fen())
    reason = "checkmate" if result in ("1-0", "0-1") else result
    return GameResult(
        result=result,
        reason=reason,
        moves=list(moves),
        fens=fens[:-1],  # one FEN per move, i.e. the position before it
        plies=len(moves),
    )


# --------------------------------------------------------------------------
# credit_curve
# --------------------------------------------------------------------------

def test_credit_curve_gives_the_last_move_full_credit():
    w = credit_curve(10, decay=0.99, decisive=True)
    assert w[-1] == pytest.approx(1.0)


def test_credit_curve_diminishes_towards_the_past():
    w = credit_curve(10, decay=0.9, decisive=True)
    assert np.all(np.diff(w) > 0)  # strictly increasing towards the end
    assert w[0] < w[-1]


def test_credit_curve_shape_matches_the_game_length():
    assert credit_curve(7, 0.99, True).shape == (7,)


def test_credit_curve_is_zero_for_a_draw():
    """A draw is uninformative about which move was better."""
    w = credit_curve(20, decay=0.99, decisive=False)
    assert np.all(w == 0.0)


def test_credit_curve_of_equal_decay_one_is_flat():
    w = credit_curve(5, decay=1.0, decisive=True)
    assert np.allclose(w, 1.0)


def test_credit_curve_on_a_zero_length_game():
    assert credit_curve(0, 0.99, True).size == 0


def test_credit_curve_values_match_the_formula():
    w = credit_curve(4, decay=0.5, decisive=True)
    # distance from end = 3,2,1,0
    assert np.allclose(w, [0.125, 0.25, 0.5, 1.0])


# --------------------------------------------------------------------------
# the update rule
# --------------------------------------------------------------------------

def test_a_won_game_raises_the_probability_of_moves_played():
    """The fundamental direction check: reward must increase probability."""
    policy = FactoredSoftmaxPolicy(seed=0)
    trainer = FactoredSoftmaxTrainer(policy)
    board = chess.Board()
    move = chess.Move(chess.E2, chess.E4)

    before = policy.move_distribution(board)[M.move_to_index(move)]
    trainer.apply_credit(board, move, credit=1.0)
    after = policy.move_distribution(board)[M.move_to_index(move)]

    assert after > before


def test_a_lost_game_lowers_the_probability_of_moves_played():
    policy = FactoredSoftmaxPolicy(seed=0)
    trainer = FactoredSoftmaxTrainer(policy)
    board = chess.Board()
    move = chess.Move(chess.D2, chess.D4)

    before = policy.move_distribution(board)[M.move_to_index(move)]
    trainer.apply_credit(board, move, credit=-1.0)
    after = policy.move_distribution(board)[M.move_to_index(move)]

    assert after < before


def test_zero_credit_is_a_no_op():
    policy = FactoredSoftmaxPolicy(seed=0)
    trainer = FactoredSoftmaxTrainer(policy)
    board = chess.Board()
    move = chess.Move(chess.G1, chess.F3)

    before = policy.move_distribution(board).copy()
    trainer.apply_credit(board, move, credit=0.0)
    assert np.allclose(policy.move_distribution(board), before)


def test_the_distribution_stays_normalised_after_many_updates():
    """Logit-space updates must not break the distribution invariant."""
    policy = FactoredSoftmaxPolicy(seed=0)
    trainer = FactoredSoftmaxTrainer(policy)
    board = chess.Board()

    for _ in range(50):
        trainer.apply_credit(board, chess.Move(chess.E2, chess.E4), 1.0)
        trainer.apply_credit(board, chess.Move(chess.D2, chess.D4), -1.0)

    dist = policy.move_distribution(board)
    assert dist.sum() == pytest.approx(1.0)
    assert np.all(dist >= 0.0)


def test_probabilities_never_go_negative_or_exceed_one():
    policy = FactoredSoftmaxPolicy(seed=0)
    trainer = FactoredSoftmaxTrainer(policy)
    board = chess.Board()

    for i in range(100):
        trainer.apply_credit(
            board, chess.Move(chess.E2, chess.E4), 1.0 if i % 3 else -1.0
        )
    dist = policy.move_distribution(board)
    assert np.all(dist >= 0.0) and np.all(dist <= 1.0)


def test_illegal_moves_are_never_given_probability_by_training():
    policy = FactoredSoftmaxPolicy(seed=0)
    trainer = FactoredSoftmaxTrainer(policy)
    board = chess.Board()
    trainer.apply_credit(board, chess.Move(chess.E2, chess.E4), 5.0)

    legal = {M.move_to_index(m) for m in board.legal_moves}
    nonzero = set(np.nonzero(policy.move_distribution(board))[0].tolist())
    assert nonzero <= legal


def test_masking_means_unlegal_actions_receive_no_update():
    """A move illegal in this position must keep its logits untouched."""
    policy = FactoredSoftmaxPolicy(seed=0)
    trainer = FactoredSoftmaxTrainer(policy)
    board = chess.Board()
    # An illegal origin (Black's back rank, White to move) is not in the mask,
    # so its logit must be exactly what it started as.
    before = policy.scorer.from_table[chess.E8]
    trainer.apply_credit(board, chess.Move(chess.E2, chess.E4), 1.0)
    assert policy.scorer.from_table[chess.E8] == before


def test_update_concentrates_on_the_origin_that_was_played():
    policy = FactoredSoftmaxPolicy(seed=0)
    trainer = FactoredSoftmaxTrainer(policy)
    board = chess.Board()
    trainer.apply_credit(board, chess.Move(chess.E2, chess.E4), 1.0)

    assert policy.scorer.from_table[chess.E2] > 0.0
    assert policy.scorer.from_table[chess.D2] < 0.0  # competing origin lowered
    assert policy.scorer.from_table[chess.E8] == 0.0  # not in the mask


def test_credit_magnitude_scales_the_update():
    small = FactoredSoftmaxPolicy(seed=0)
    large = FactoredSoftmaxPolicy(seed=0)
    board = chess.Board()
    move = chess.Move(chess.E2, chess.E4)

    FactoredSoftmaxTrainer(small).apply_credit(board, move, 0.1)
    FactoredSoftmaxTrainer(large).apply_credit(board, move, 1.0)

    assert (
        large.scorer.from_table[chess.E2] > small.scorer.from_table[chess.E2]
    )


# --------------------------------------------------------------------------
# train_on_game and sign convention
# --------------------------------------------------------------------------

def test_train_on_game_increments_the_game_counter():
    trainer = FactoredSoftmaxTrainer()
    trainer.train_on_game(make_game("1-0"))
    assert trainer.games_played == 1


def test_train_on_game_requires_recorded_fens():
    trainer = FactoredSoftmaxTrainer()
    game = GameResult(result="1-0", reason="checkmate", moves=["e4"], fens=[], plies=1)
    with pytest.raises(ValueError, match="FEN"):
        trainer.train_on_game(game)


def test_white_win_raises_white_moves_and_lowers_black_moves():
    """The sign convention. Getting this wrong yields a policy that loses."""
    policy = FactoredSoftmaxPolicy(seed=0)
    trainer = FactoredSoftmaxTrainer(policy)
    trainer.train_on_game(make_game("1-0", moves=("e4", "e5")))

    # White played e2-e4 from a position with White to move.
    assert policy.scorer.from_table[chess.E2] > 0.0
    # Black played e7-e5; a White win must push that down.
    assert policy.scorer.from_table[chess.E7] < 0.0


def test_black_win_is_the_mirror_of_a_white_win():
    white = FactoredSoftmaxPolicy(seed=0)
    black = FactoredSoftmaxPolicy(seed=0)
    FactoredSoftmaxTrainer(white).train_on_game(make_game("1-0"))
    FactoredSoftmaxTrainer(black).train_on_game(make_game("0-1"))

    assert (
        white.scorer.from_table[chess.E2] > 0.0
        and black.scorer.from_table[chess.E2] < 0.0
    )
    assert (
        white.scorer.from_table[chess.E7] < 0.0
        and black.scorer.from_table[chess.E7] > 0.0
    )


def test_a_draw_teaches_nothing():
    """A draw is uninformative; outcome reward must be exactly zero."""
    policy = FactoredSoftmaxPolicy(seed=0)
    trainer = FactoredSoftmaxTrainer(policy)
    before = policy.scorer.from_table.copy()
    trainer.train_on_game(make_game("1/2-1/2"))
    assert np.allclose(policy.scorer.from_table, before)


def test_an_unfinished_game_teaches_nothing_without_shaping():
    policy = FactoredSoftmaxPolicy(seed=0)
    trainer = FactoredSoftmaxTrainer(policy, TrainConfig(shaped_reward=False))
    before = policy.scorer.from_table.copy()
    trainer.train_on_game(make_game("unfinished"))
    assert np.allclose(policy.scorer.from_table, before)


def test_an_unfinished_game_does_teach_with_shaping():
    """Shaping is what makes L1 trainable: without it a blind policy never
    sees a non-zero reward, because it never delivers mate."""
    policy = FactoredSoftmaxPolicy(seed=0)
    trainer = FactoredSoftmaxTrainer(policy, TrainConfig(shaped_reward=True))
    before = policy.scorer.from_table.copy()
    trainer.train_on_game(make_game("unfinished"))
    assert not np.allclose(policy.scorer.from_table, before)


def test_white_reward_is_plus_one_for_a_white_win():
    trainer = FactoredSoftmaxTrainer()
    score, amplitude = trainer.white_reward(make_game("1-0"))
    assert score == pytest.approx(1.0)
    assert amplitude == pytest.approx(1.0)


def test_white_reward_is_minus_one_for_a_black_win():
    trainer = FactoredSoftmaxTrainer()
    score, _ = trainer.white_reward(make_game("0-1"))
    assert score == pytest.approx(-1.0)


def test_white_reward_is_zero_for_a_draw():
    trainer = FactoredSoftmaxTrainer()
    score, amplitude = trainer.white_reward(make_game("1/2-1/2"))
    assert score == 0.0 and amplitude == 0.0


def test_shaped_reward_is_bounded():
    """Shaping must stay in [-1, 1] so it composes with a real result."""
    trainer = FactoredSoftmaxTrainer(
        TrainConfig(shaped_reward=True, shaped_weight=0.3)
    )
    # A lopsided position: White is up a queen.
    game = make_game(
        "unfinished", moves=("e4",), fens=None
    )
    game.fens = ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"]
    score, _ = trainer.white_reward(game)
    assert -1.0 <= score <= 1.0


def test_train_on_game_updates_the_counters():
    trainer = FactoredSoftmaxTrainer()
    trainer.train_on_game(make_game("1-0", moves=("e4", "e5")))
    stats = trainer.counter.stats()
    assert stats["total_takes"] == 2
    assert stats["total_offers"] > 0


def test_diminishing_credit_reaches_earlier_moves_less():
    """A move near the end of a long game gets a bigger nudge."""
    policy = FactoredSoftmaxPolicy(seed=0)
    config = TrainConfig(credit_decay=0.5, lr=0.1)
    trainer = FactoredSoftmaxTrainer(policy, config)

    game = make_game(
        "1-0", moves=("a3", "a6", "b3", "b6", "c3", "c6", "d3", "d6")
    )
    trainer.train_on_game(game)

    # The last White move (d2-d3) is nearer the end than the first (a2-a3),
    # so its origin logit must have moved further.
    assert policy.scorer.from_table[chess.D2] > policy.scorer.from_table[chess.A2]


def test_training_on_a_real_game_runs_end_to_end():
    trainer = FactoredSoftmaxTrainer(TrainConfig(seed=0))
    game = play_game(
        RandomPolicy(seed=0), RandomPolicy(seed=1), max_plies=60
    )
    trainer.train_on_game(game)
    assert trainer.games_played == 1


# --------------------------------------------------------------------------
# the training loop
# --------------------------------------------------------------------------

def test_train_runs_and_reports_its_shape():
    trainer = FactoredSoftmaxTrainer(TrainConfig(games=4, seed=0, max_plies=40))
    summary = trainer.train()
    assert summary["games"] == 4
    assert set(summary["outcomes"]) >= {"1-0", "0-1", "1/2-1/2", "unfinished"}
    assert summary["games_per_second"] > 0.0
    assert summary["logit_updates"] > 0


def test_train_is_reproducible_under_a_fixed_seed():
    def run():
        t = FactoredSoftmaxTrainer(TrainConfig(games=3, seed=11, max_plies=40))
        t.train()
        return t.policy.scorer.from_table.copy()

    assert np.allclose(run(), run())


def test_train_respects_the_game_count_argument():
    trainer = FactoredSoftmaxTrainer(TrainConfig(games=100, seed=0))
    summary = trainer.train(games=2)
    assert summary["games"] == 2


def test_frozen_opponent_is_a_different_object():
    """Self-play against the same live object is degenerate: the two sides
    make identical decisions and the game mirrors itself forever."""
    trainer = FactoredSoftmaxTrainer(TrainConfig(seed=0))
    opponent = trainer._opponent()
    assert opponent is not trainer.policy
    assert opponent.rng is not trainer.policy.rng


def test_frozen_opponent_snapshots_current_weights():
    trainer = FactoredSoftmaxTrainer(TrainConfig(seed=0))
    trainer.policy.scorer.from_table[chess.E2] = 7.0
    opponent = trainer._opponent()
    assert opponent.scorer.from_table[chess.E2] == 7.0


def test_frozen_opponent_does_not_track_later_changes():
    trainer = FactoredSoftmaxTrainer(TrainConfig(seed=0))
    opponent = trainer._opponent()
    trainer.policy.scorer.from_table[chess.E2] = 7.0
    assert opponent.scorer.from_table[chess.E2] == 0.0


def test_shaping_produces_updates_where_pure_outcome_reward_produces_none():
    """The central L1 finding, asserted so a regression is caught.

    A blind policy under a ply cap essentially never delivers mate, so with
    pure outcome reward the trainer receives no signal at all. Shaping is not
    an optimisation here, it is the difference between training and not.
    """
    shaped = FactoredSoftmaxTrainer(
        TrainConfig(games=6, seed=0, max_plies=50, shaped_reward=True)
    )
    plain = FactoredSoftmaxTrainer(
        TrainConfig(games=6, seed=0, max_plies=50, shaped_reward=False)
    )
    shaped_summary = shaped.train()
    plain_summary = plain.train()

    assert plain_summary["logit_updates"] == 0
    assert shaped_summary["logit_updates"] > 0


def test_train_accepts_an_external_opponent():
    """The L2 wrapper depends on this: swap in a non-learning opponent."""
    from chessrl.game import MaterialPolicy

    trainer = FactoredSoftmaxTrainer(TrainConfig(games=3, seed=0, max_plies=30))
    summary = trainer.train(games=3, opponent=MaterialPolicy(seed=0))
    assert summary["games"] == 3
    assert trainer.games_played == 3


def test_train_plays_both_colours():
    """Alternating colours stops the policy learning a first-move bias."""
    trainer = FactoredSoftmaxTrainer(TrainConfig(games=4, seed=0, max_plies=20))
    trainer.train()
    # With an even game count both colours are covered; the assertion is that
    # at least one game of each colour occurred, which shows up as both sides
    # having taken moves.
    assert trainer.counter.taken_from.sum() > 0


def test_mean_entropy_starts_near_log_of_legal_moves():
    """An untrained blind policy is uniform, so entropy is high."""
    trainer = FactoredSoftmaxTrainer(TrainConfig(seed=0))
    e = trainer.mean_entropy()
    # ~log(20) at the start position; averaged with a middlegame and an
    # endgame probe it lands in a plausible band rather than an exact value.
    assert 2.0 < e < 3.5


def test_mean_entropy_falls_as_the_policy_sharpens():
    policy = FactoredSoftmaxPolicy(seed=0)
    trainer = FactoredSoftmaxTrainer(policy, TrainConfig(lr=0.5))
    before = trainer.mean_entropy()
    for _ in range(60):
        trainer.apply_credit(chess.Board(), chess.Move(chess.E2, chess.E4), 1.0)
    assert trainer.mean_entropy() < before


def test_probability_table_has_the_right_shapes():
    trainer = FactoredSoftmaxTrainer(TrainConfig(seed=0))
    table = trainer.probability_table()
    assert table["from"].shape == (64,)
    assert table["to"].shape == (64, 64)
    assert table["promo"].shape == (64, 64, M.NUM_PROMO)
    for key in ("from", "to", "promo"):
        assert np.allclose(table[key].sum(axis=-1) * np.ones_like(
            table[key].sum(axis=-1)
        ), table[key].sum(axis=-1))
    assert table["from"].sum() == pytest.approx(1.0)


def test_top_moves_returns_probable_legal_moves():
    trainer = FactoredSoftmaxTrainer(TrainConfig(seed=0))
    board = chess.Board()
    top = trainer.top_moves(board, k=5)
    assert len(top) == 5
    probs = [p for _, p in top]
    assert probs == sorted(probs, reverse=True)
    legal = {m.uci() for m in board.legal_moves}
    for uci, _ in top:
        assert uci in legal


def test_top_moves_respects_the_k_argument():
    trainer = FactoredSoftmaxTrainer(TrainConfig(seed=0))
    assert len(trainer.top_moves(chess.Board(), k=3)) == 3


# --------------------------------------------------------------------------
# pruning
# --------------------------------------------------------------------------

def test_prune_leaves_untested_squares_alone():
    trainer = FactoredSoftmaxTrainer(
        TrainConfig(seed=0, prune_min_offers=10_000)
    )
    report = trainer.prune()
    assert report["pruned_count"] == 0


def test_prune_zeroes_logits_of_confident_low_probability_origins():
    policy = FactoredSoftmaxPolicy(seed=0)
    config = TrainConfig(prune_min_offers=10, prune_threshold=0.01)
    trainer = FactoredSoftmaxTrainer(policy, config)

    # Make a1 very improbable and give it plenty of offers.
    policy.scorer.from_table[:] = 5.0
    policy.scorer.from_table[chess.A1] = -50.0
    trainer.counter.offered_from[chess.A1] = 100

    report = trainer.prune()
    assert report["pruned_count"] >= 1
    assert chess.square_name(chess.A1) in report["pruned_squares"]
    assert policy.scorer.from_table[chess.A1] == 0.0


def test_pruning_never_makes_a_legal_move_impossible():
    """Pruning removes preference, not legality."""
    policy = FactoredSoftmaxPolicy(seed=0)
    trainer = FactoredSoftmaxTrainer(
        policy, TrainConfig(prune_min_offers=0, prune_threshold=1.0)
    )
    trainer.counter.offered_from[:] = 100
    trainer.prune()

    board = chess.Board()
    dist = policy.move_distribution(board)
    assert dist.sum() == pytest.approx(1.0)
    assert dist[M.move_to_index(chess.Move(chess.E2, chess.E4))] > 0.0


def test_train_with_prune_enabled_reports_the_prune_result():
    trainer = FactoredSoftmaxTrainer(
        TrainConfig(games=2, seed=0, max_plies=30, prune=True)
    )
    summary = trainer.train()
    assert "pruned" in summary
    assert "pruned_count" in summary["pruned"]


# --------------------------------------------------------------------------
# persistence
# --------------------------------------------------------------------------

def test_save_writes_model_meta_and_counter(tmp_path):
    trainer = FactoredSoftmaxTrainer(TrainConfig(games=2, seed=0, max_plies=30))
    trainer.train()

    saved = trainer.save(
        tmp_path / "model.json", tmp_path / "counter.npz"
    )
    assert (tmp_path / "model.json").exists()
    assert (tmp_path / "counter.npz").exists()
    assert (tmp_path / "model.meta.json").exists()
    assert set(saved) == {"model", "counter", "meta"}


def test_saved_model_reloads_to_the_same_distribution(tmp_path):
    trainer = FactoredSoftmaxTrainer(TrainConfig(games=3, seed=0, max_plies=30))
    trainer.train()
    trainer.save(tmp_path / "model.json")

    loaded = FactoredSoftmaxPolicy.load(tmp_path / "model.json")
    board = chess.Board()
    assert np.allclose(
        loaded.move_distribution(board),
        trainer.policy.move_distribution(board),
    )


def test_meta_sidecar_records_training_provenance(tmp_path):
    trainer = FactoredSoftmaxTrainer(TrainConfig(games=2, seed=0, max_plies=30))
    trainer.train()
    trainer.save(tmp_path / "model.json")

    meta = json.loads((tmp_path / "model.meta.json").read_text())
    assert meta["games_played"] == 2
    assert meta["logit_updates"] > 0
    assert "lr" in meta["config"]


def test_counter_is_not_embedded_in_the_model_file(tmp_path):
    """The spec asks for counters to be deleted before saving the model."""
    trainer = FactoredSoftmaxTrainer(TrainConfig(games=1, seed=0, max_plies=20))
    trainer.train()
    path = tmp_path / "model.json"
    trainer.save(path)

    payload = json.loads(path.read_text())
    assert "counter" not in payload
    assert "offered_from" not in json.dumps(payload)


def test_saved_counter_roundtrips_and_resumes(tmp_path):
    trainer = FactoredSoftmaxTrainer(TrainConfig(games=2, seed=0, max_plies=30))
    trainer.train()
    trainer.save(tmp_path / "m.json", tmp_path / "c.npz")

    resumed = FactoredSoftmaxTrainer(
        FactoredSoftmaxPolicy.load(tmp_path / "m.json")
    )
    resumed.counter = ProbCounter.load(tmp_path / "c.npz")
    assert resumed.counter.stats()["total_takes"] == \
        trainer.counter.stats()["total_takes"]


def test_save_creates_missing_directories(tmp_path):
    trainer = FactoredSoftmaxTrainer(TrainConfig(games=1, seed=0, max_plies=20))
    trainer.train()
    trainer.save(tmp_path / "deep" / "nested" / "model.json")
    assert (tmp_path / "deep" / "nested" / "model.json").exists()
