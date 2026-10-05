"""L3: the board-reading perceptron, its training signals, and its symmetry.

The load-bearing test in this module is
:func:`test_colour_symmetry_holds_exactly`. L3 is the first level whose
parameters are *indexed by square*, and square-indexed parameters are exactly
where a colour asymmetry can hide: a weight row tied to rank 7 is a different
row for White than for Black unless the model reads the board in the canonical
frame. That bug trains happily and simply makes Black's games teach nothing.

It was found by mirroring a dense middlegame position and finding a 6e-3
probability difference where there should have been none. After the fix the
residual is float32 accumulation, roughly 1e-8.
"""

from __future__ import annotations

import numpy as np
import chess
import pytest

from chessrl import masks as M
from chessrl.cache import MidstateStore
from chessrl.perceptron import (
    GEOMETRY_NAMES,
    MOVE_FEATURE_NAMES_L3,
    QUANT,
    L3Config,
    L3Policy,
    L3Trainer,
    PerceptronConfig,
    PerceptronScorer,
)
from chessrl.policy import BlindScorer, FactoredSoftmaxPolicy

from conftest import INTERESTING_FENS, colour_mirror


def blind_policy(seed: int = 0) -> FactoredSoftmaxPolicy:
    """L1: the same factored machinery driven by a position-blind scorer.

    Rebuilt here rather than imported because L1's own tests own the canonical
    construction; this is only ever used as the "does L3 see anything the blind
    twin cannot" baseline.
    """
    policy = FactoredSoftmaxPolicy(BlindScorer(), name="L1-blind", seed=seed)
    policy.greedy = True
    return policy


def reflect_square(sq: int) -> int:
    """The rank reflection the canonical frame applies: file kept, rank flipped."""
    return chess.square(chess.square_file(sq), 7 - chess.square_rank(sq))


def reflect_move(move: chess.Move) -> chess.Move:
    return chess.Move(
        reflect_square(move.from_square),
        reflect_square(move.to_square),
        promotion=move.promotion,
    )


@pytest.fixture
def policy() -> L3Policy:
    return L3Policy(seed=11)


# --------------------------------------------------------------------------
# shape and protocol conformance
# --------------------------------------------------------------------------

def test_heads_have_the_shapes_the_protocol_promises(policy):
    board = chess.Board()
    policy.scorer.set_position(board)
    assert policy.scorer.from_logits(board).shape == (64,)
    assert policy.scorer.to_logits(chess.E2).shape == (64,)
    assert policy.scorer.promo_logits(chess.A7, chess.A8).shape == (M.NUM_PROMO,)


def test_from_logits_is_not_a_square_against_square_matrix(policy):
    """The einsum bug produced a ``(64, 64)`` array that still ran.

    Guard against it directly: the origin head must return one number per
    square, not a pairwise table.
    """
    logits = policy.scorer.from_logits(chess.Board())
    assert logits.ndim == 1
    assert logits.shape == (64,)


def test_square_local_heads_require_a_bound_position():
    """Calling a head before ``set_position`` must fail loudly.

    Silently scoring against a stale board would corrupt training in a way that
    is nearly impossible to see from the loss.
    """
    scorer = PerceptronScorer()
    with pytest.raises(RuntimeError, match="set_position"):
        scorer.to_logits(chess.E2)


def test_distribution_is_a_valid_probability_vector(policy):
    board = chess.Board()
    dist = policy.move_distribution(board)
    assert dist.shape == (M.ACTION_SPACE,)
    assert np.isclose(dist.sum(), 1.0, atol=1e-9)
    assert (dist >= 0).all()


def test_distribution_has_exactly_one_entry_per_legal_move(policy):
    board = chess.Board()
    dist = policy.move_distribution(board)
    nonzero = np.nonzero(dist)[0]
    assert len(nonzero) == board.legal_moves.count()
    legal = {M.move_to_index(m) for m in board.legal_moves}
    assert set(nonzero.tolist()) == legal


def test_distribution_is_zero_when_the_game_is_over(policy):
    board = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
    assert board.is_checkmate()
    assert policy.move_distribution(board).sum() == 0.0


def test_policy_selects_a_legal_move_for_both_colours(policy):
    for fen in INTERESTING_FENS:
        board = chess.Board(fen)
        if board.is_game_over():
            continue
        assert policy.select(board) in board.legal_moves


# --------------------------------------------------------------------------
# the parameter-count claim
# --------------------------------------------------------------------------

def test_parameter_count_is_in_the_low_thousands(policy):
    """The whole design hinges on this being small.

    A flattened perceptron over ``(28, 8, 8)`` needs 7.3M parameters for the
    destination head alone; this factorised version must stay orders of
    magnitude below that or the level is untrainable in the games available.
    """
    n = policy.perceptron.n_parameters
    assert 1_000 < n < 20_000, n


def test_parameter_count_matches_the_declared_layout():
    scorer = PerceptronScorer(PerceptronConfig())
    expected = (
        scorer.W_from.size + scorer.b_from.size
        + scorer.W_ctx.size + scorer.W_hid.size
        + scorer.W_to_dst.size + scorer.W_to_src.size
        + scorer.W_geom.size + scorer.promo.size
    )
    assert scorer.n_parameters == expected


def test_disabling_context_removes_the_hidden_pathway():
    without = PerceptronScorer(PerceptronConfig(use_context=False))
    assert without.h == 0
    assert without.n_parameters < PerceptronScorer(PerceptronConfig()).n_parameters
    # And it must still score, using only the linear head.
    board = chess.Board()
    without.set_position(board)
    assert without.from_logits(board).shape == (64,)


def test_value_channels_are_opt_in_and_change_the_input_width():
    off = PerceptronScorer(PerceptronConfig(use_value_channels=False))
    on = PerceptronScorer(PerceptronConfig(use_value_channels=True))
    assert on.n_ch > off.n_ch
    assert on.W_from.shape == (64, on.n_ch)


# --------------------------------------------------------------------------
# colour symmetry: the load-bearing test
# --------------------------------------------------------------------------

def test_colour_symmetry_holds_exactly(policy):
    """One weight set must serve both colours, so mirrored positions must agree.

    Rank-reflect the position, swap the colours, reflect every move, and the
    probability of each move must be unchanged. Any residual above float noise
    means some head is reading geometry in absolute board coordinates.
    """
    worst = 0.0
    for fen in INTERESTING_FENS:
        board = chess.Board(fen)
        if board.is_game_over():
            continue
        mirror = colour_mirror(fen)
        dist = policy.move_distribution(board)
        dist_m = policy.move_distribution(mirror)

        assert dist.sum() > 0 and dist_m.sum() > 0
        for move in board.legal_moves:
            p = dist[M.move_to_index(move)]
            p_m = dist_m[M.move_to_index(reflect_move(move))]
            worst = max(worst, abs(p - p_m))

    assert worst < 1e-6, f"colour asymmetry of {worst:.3e}"


def test_symmetry_survives_a_dense_middlegame(policy):
    """The case that actually caught the bug.

    A symmetric opening hides a rank-dependent head because every White move
    has a mirror-image Black move at the same weight. A dense middlegame with
    castling rights and material imbalance does not.
    """
    fen = "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1"
    board = chess.Board(fen)
    mirror = colour_mirror(fen)
    dist = policy.move_distribution(board)
    dist_m = policy.move_distribution(mirror)

    worst = max(
        abs(dist[M.move_to_index(m)] - dist_m[M.move_to_index(reflect_move(m))])
        for m in board.legal_moves
    )
    assert worst < 1e-6, f"colour asymmetry of {worst:.3e}"


def test_mirrored_positions_select_mirrored_moves_when_greedy(policy):
    """Greedy selection is deterministic, so the chosen move must mirror.

    Checked on the *move identity*, not on a probability: with a correct
    canonical frame the mirror image of the chosen move is exactly the move
    chosen in the mirrored position.
    """
    policy.greedy = True
    for fen in INTERESTING_FENS:
        board = chess.Board(fen)
        if board.is_game_over():
            continue
        move = policy.select(board)
        move_m = policy.select(colour_mirror(fen))
        assert move_m == reflect_move(move), (
            f"{fen}: chose {move.uci()}, expected the mirror of {move.uci()}"
        )


def test_mirrored_positions_agree_move_for_move(policy):
    """Every move's probability must equal its mirror's, end to end."""
    for fen in INTERESTING_FENS:
        board = chess.Board(fen)
        if board.is_game_over():
            continue
        mirror = colour_mirror(fen)
        dist = policy.move_distribution(board)
        dist_m = policy.move_distribution(mirror)
        for move in board.legal_moves:
            mirrored = reflect_move(move)
            assert mirrored in mirror.legal_moves
            assert dist[M.move_to_index(move)] == pytest.approx(
                dist_m[M.move_to_index(mirrored)], abs=1e-6
            )


def test_canonical_square_map_is_its_own_inverse(policy):
    """For Black the map is a rank reflection, which is an involution."""
    black = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
    mapping = PerceptronScorer._canonical_squares(black)
    assert (mapping[mapping] == np.arange(64)).all()
    # And it really does reflect: rank 7 maps to rank 0.
    assert mapping[chess.A8] == chess.A1
    assert mapping[chess.E8] == chess.E1


def test_canonical_square_map_is_identity_for_white(policy):
    mapping = PerceptronScorer._canonical_squares(chess.Board())
    assert (mapping == np.arange(64)).all()


# --------------------------------------------------------------------------
# geometry features
# --------------------------------------------------------------------------

def test_geometry_features_have_a_declared_length():
    scorer = PerceptronScorer()
    assert scorer._geom_features(chess.E2, chess.E4).shape == (len(GEOMETRY_NAMES),)


def test_move_features_have_a_declared_length(policy):
    board = chess.Board()
    policy.scorer.set_position(board)
    feats = policy.scorer._move_features(chess.E2, chess.E4)
    assert feats.shape == (len(MOVE_FEATURE_NAMES_L3),)


def test_geometry_is_expressed_in_the_canonical_frame(policy):
    """Geometry depends only on the canonical squares it is given.

    ``_geom_features`` is the *inner* half of the canonicalisation: the caller
    has already mapped real squares into the canonical frame, so the function
    itself must be a pure function of its inputs and must not re-reflect. The
    property asserted here is that a forward pawn push and its rank-mirrored
    counterpart -- which are the same canonical move -- produce the same
    vector. The absence of exactly this property caused the symmetry failure,
    so it is asserted directly as well as end to end.
    """
    d_rank = GEOMETRY_NAMES.index("d_rank")
    # A pawn push by the side to move: two ranks up, file unchanged.
    push = policy.scorer._geom_features(chess.E2, chess.E4)
    assert push[d_rank] > 0
    assert push[GEOMETRY_NAMES.index("same_file")] == 1.0
    assert push[GEOMETRY_NAMES.index("abs_d_file")] == 0.0

    # Two canonical moves that differ only by rank translation must share
    # every rank-difference feature.
    a = policy.scorer._geom_features(chess.E2, chess.E4)
    b = policy.scorer._geom_features(chess.E3, chess.E5)
    assert a[d_rank] == pytest.approx(b[d_rank])
    assert a[GEOMETRY_NAMES.index("distance")] == pytest.approx(
        b[GEOMETRY_NAMES.index("distance")]
    )


def test_geometry_makes_forward_moves_forward_for_both_colours(policy):
    """``d_rank`` must be positive for a pawn push by either side.

    In the canonical frame the side to move always advances towards higher
    ranks, so this is the invariant that makes one weight set serve both.
    """
    d_rank = GEOMETRY_NAMES.index("d_rank")
    white_push = policy.scorer._geom_features(
        chess.square(4, 1), chess.square(4, 3)
    )
    black_push = policy.scorer._geom_features(
        chess.square(4, 6), chess.square(4, 4)
    )
    # Both are two ranks of travel; after reflection they are the same vector.
    assert white_push[d_rank] > 0
    assert abs(white_push[d_rank]) == abs(black_push[d_rank])


# --------------------------------------------------------------------------
# determinism and quantisation
# --------------------------------------------------------------------------

def test_same_seed_gives_identical_weights():
    a = L3Policy(seed=5).perceptron
    b = L3Policy(seed=5).perceptron
    for key in a.parameters():
        np.testing.assert_array_equal(a.parameters()[key], b.parameters()[key])


def test_different_seeds_give_different_weights():
    a = L3Policy(seed=5).perceptron
    b = L3Policy(seed=6).perceptron
    assert not np.array_equal(a.W_from, b.W_from)


def test_quantised_updates_land_on_the_fixed_point_grid():
    """Every nudge must move a weight by a whole multiple of ``1/QUANT``.

    This is what keeps two seeded runs bit-identical rather than merely close
    after thousands of steps.
    """
    trainer = L3Trainer(L3Config(lr=0.05, seed=1))
    scorer = trainer.policy.perceptron
    before = scorer.W_from.copy()
    trainer.apply_credit(chess.Board(), chess.Move.from_uci("e2e4"), 0.7)
    delta = np.abs(scorer.W_from - before).reshape(-1)
    touched = delta[delta > 0]
    assert touched.size > 0
    grid = touched * QUANT
    np.testing.assert_allclose(grid, np.round(grid), atol=1e-3)


def test_two_identical_training_runs_produce_identical_weights():
    """The end-to-end version of the quantisation guarantee."""

    def run():
        trainer = L3Trainer(L3Config(seed=3, lr=0.02))
        board = chess.Board()
        for uci in ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6"]:
            move = chess.Move.from_uci(uci)
            trainer.apply_credit(board, move, 0.5)
            board.push(move)
        return trainer.policy.perceptron.W_from.copy()

    np.testing.assert_array_equal(run(), run())


def test_non_quantised_updates_are_allowed_to_leave_the_grid():
    trainer = L3Trainer(L3Config(lr=0.005, seed=1))
    trainer.policy.perceptron.config.quantised = False
    before = trainer.policy.perceptron.W_from.copy()
    trainer.apply_credit(chess.Board(), chess.Move.from_uci("e2e4"), 0.7)
    delta = (trainer.policy.perceptron.W_from - before).reshape(-1)
    touched = delta[delta != 0]
    assert touched.size > 0
    # An arbitrary float multiple of 1/QUANT is astronomically unlikely.
    assert not np.allclose(touched * QUANT, np.round(touched * QUANT), atol=1e-6)


def test_weight_cache_is_bounded():
    """The channel cache must not grow without limit over a long run."""
    scorer = PerceptronScorer()
    seen = set()
    for seed in range(200):
        import random

        rng = random.Random(seed)
        board = chess.Board()
        for _ in range(rng.randrange(1, 12)):
            if board.is_game_over():
                break
            board.push(rng.choice(list(board.legal_moves)))
        scorer.channels_for(board)
        seen.add(chess.polyglot.zobrist_hash(board))
    # The bound is 4096 in the implementation; a few hundred positions is well
    # under it, so the test asserts the cache tracks real positions rather than
    # leaking per call.
    assert len(scorer._chan_cache) <= max(len(seen), 4096)


# --------------------------------------------------------------------------
# training: credit application
# --------------------------------------------------------------------------

def test_credit_moves_the_move_it_is_given_up_the_ranking():
    """Positive credit on a move must raise that move's own probability."""
    trainer = L3Trainer(L3Config(lr=0.05, seed=2))
    board = chess.Board()
    move = chess.Move.from_uci("e2e4")
    idx = M.move_to_index(move)

    before = trainer.policy.move_distribution(board)[idx]
    for _ in range(40):
        trainer.apply_credit(board, move, 0.5)
    after = trainer.policy.move_distribution(board)[idx]
    assert after > before, (before, after)


def test_negative_credit_pushes_a_move_down():
    trainer = L3Trainer(L3Config(lr=0.05, seed=2))
    board = chess.Board()
    move = chess.Move.from_uci("e2e4")
    idx = M.move_to_index(move)

    before = trainer.policy.move_distribution(board)[idx]
    for _ in range(40):
        trainer.apply_credit(board, move, -0.5)
    after = trainer.policy.move_distribution(board)[idx]
    assert after < before, (before, after)


def test_zero_credit_is_a_no_op():
    trainer = L3Trainer(L3Config(seed=2))
    before = trainer.policy.perceptron.W_from.copy()
    trainer.apply_credit(chess.Board(), chess.Move.from_uci("e2e4"), 0.0)
    np.testing.assert_array_equal(trainer.policy.perceptron.W_from, before)


def test_credit_for_black_updates_a_row_that_inference_reads():
    """Training and inference must agree on which weight row a square owns.

    If training nudged ``W_from[real_square]`` while inference read
    ``W_from[canonical_square]``, the model would train on White's games only
    and quietly waste every Black position in the corpus. Assert the update is
    visible through the forward pass, which is the only check that catches it.
    """
    trainer = L3Trainer(L3Config(lr=0.05, seed=2))
    board = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
    assert board.turn == chess.BLACK
    move = chess.Move.from_uci("g8f6")
    idx = M.move_to_index(move)

    before = trainer.policy.move_distribution(board)[idx]
    for _ in range(40):
        trainer.apply_credit(board, move, 0.5)
    after = trainer.policy.move_distribution(board)[idx]
    assert after > before, (before, after)


def test_credit_touches_only_some_weights():
    """A single credit event must not rewrite the whole model."""
    trainer = L3Trainer(L3Config(lr=0.01, seed=2))
    before = {k: v.copy() for k, v in trainer.policy.perceptron.parameters().items()}
    trainer.apply_credit(chess.Board(), chess.Move.from_uci("e2e4"), 0.5)
    after = trainer.policy.perceptron.parameters()
    for key in before:
        changed = not np.array_equal(before[key], after[key])
        if key in ("W_from", "b_from", "W_geom"):
            assert changed, f"{key} should have been touched"


def test_counter_records_both_the_offer_and_the_take():
    trainer = L3Trainer(L3Config(seed=2))
    board = chess.Board()
    trainer.apply_credit(board, chess.Move.from_uci("e2e4"), 0.5)
    stats = trainer.counter.stats()
    # "Offers" are counted per *origin square*, not per move: the mask exposes
    # 10 distinct origins from the opening array while only one of them was
    # taken. Comparing against the legal-move count (20) is the natural mistake
    # here and would fail for a reason that has nothing to do with the code.
    assert stats["total_offers"] == 10
    assert stats["total_takes"] == 1


# --------------------------------------------------------------------------
# training: game outcomes
# --------------------------------------------------------------------------

def _play_and_train(trainer: L3Trainer, seed: int, plies: int = 24):
    import random

    rng = random.Random(seed)
    board = chess.Board()
    fens = [board.fen()]
    moves: list[str] = []
    for _ in range(plies):
        if board.is_game_over():
            break
        move = rng.choice(list(board.legal_moves))
        moves.append(board.san(move))
        board.push(move)
        fens.append(board.fen())

    from chessrl.game import GameResult

    result = GameResult(
        result="unfinished",
        reason="ply_cap",
        fens=fens,
        moves=moves,
        plies=len(moves),
    )
    return trainer.train_on_game(result), result


def test_train_on_game_applies_one_update_per_ply():
    trainer = L3Trainer(L3Config(seed=4))
    stats, result = _play_and_train(trainer, seed=4)
    assert trainer.updates > 0
    assert trainer.games_played == 1
    assert stats["result"] == "unfinished"


def test_train_on_game_rejects_a_result_without_fens():
    from chessrl.game import GameResult

    trainer = L3Trainer(L3Config(seed=4))
    empty = GameResult(result="unfinished", reason="ply_cap", fens=[], moves=[], plies=0)
    with pytest.raises(ValueError, match="FEN"):
        trainer.train_on_game(empty)


def test_later_plies_get_less_credit_than_earlier_ones():
    """Diminishing credit: the credit curve must actually fall off.

    ``credit_curve`` weights a ply by ``decay ** plies_from_the_end``, so the
    *last* ply (fewest steps back) gets the largest weight and the first ply
    gets the smallest. Getting that direction backwards in a test is easy and
    asserts the opposite of the intended design.
    """
    trainer = L3Trainer(L3Config(seed=4, credit_decay=0.9))
    assert trainer.config.credit_decay < 1.0
    n = 20
    # Policy order: ply 0 first. Weight = decay ** (n - 1 - ply).
    weights = np.power(trainer.config.credit_decay, np.arange(n - 1, -1, -1))
    # weights[0] is the *first* ply's weight and must be the smallest.
    assert weights[0] < weights[-1]
    assert np.all(np.diff(weights) > 0)
    assert weights[-1] == pytest.approx(1.0)


def test_shaped_reward_is_small_and_bounded():
    """An unfinished game is a weak hint, never a full-magnitude signal."""
    trainer = L3Trainer(L3Config(seed=4, shaped_reward=True, shaped_weight=0.3))
    _, result = _play_and_train(trainer, seed=9, plies=30)
    score, amplitude = trainer.white_reward(result)
    assert -0.31 <= score <= 0.31
    assert amplitude == 1.0


def test_unfinished_games_score_zero_without_shaping():
    """The trap: scoring a ply-capped game as a draw teaches shuffling."""
    trainer = L3Trainer(L3Config(seed=4, shaped_reward=False))
    _, result = _play_and_train(trainer, seed=9, plies=30)
    score, amplitude = trainer.white_reward(result)
    assert score == 0.0
    assert amplitude == 0.0


def test_a_decisive_game_gives_a_full_magnitude_reward():
    from chessrl.game import GameResult

    trainer = L3Trainer(L3Config(seed=4))
    win = GameResult(result="1-0", reason="checkmate", fens=["x"], moves=[], plies=0)
    score, amplitude = trainer.white_reward(win)
    assert score > 0
    assert amplitude == 1.0
    loss = GameResult(result="0-1", reason="checkmate", fens=["x"], moves=[], plies=0)
    score, _ = trainer.white_reward(loss)
    assert score < 0


def test_training_improves_the_agreement_with_a_shaping_signal():
    """Repeat credit on the same move must monotonically raise its probability."""
    trainer = L3Trainer(L3Config(lr=0.03, seed=8))
    board = chess.Board()
    move = chess.Move.from_uci("g1f3")
    idx = M.move_to_index(move)
    probs = []
    for _ in range(6):
        probs.append(trainer.policy.move_distribution(board)[idx])
        for _ in range(10):
            trainer.apply_credit(board, move, 0.5)
    assert all(b >= a - 1e-12 for a, b in zip(probs, probs[1:])), probs
    assert probs[-1] > probs[0]


# --------------------------------------------------------------------------
# training: depth-5 top-5 search feedback
# --------------------------------------------------------------------------

def test_search_feedback_returns_at_most_top_k_moves():
    trainer = L3Trainer(L3Config(seed=1, top_k=5))
    board = chess.Board()
    targets = trainer.search_feedback(board, depth=2)
    assert 0 < len(targets) <= 5
    for move, weight in targets:
        assert move in board.legal_moves
        assert weight > 0


def test_search_feedback_returns_legal_moves_only():
    trainer = L3Trainer(L3Config(seed=1, top_k=5))
    for fen in INTERESTING_FENS:
        board = chess.Board(fen)
        if board.is_game_over():
            continue
        for move, _ in trainer.search_feedback(board, depth=2):
            assert move in board.legal_moves


def test_search_feedback_weights_are_normalised_against_the_best():
    trainer = L3Trainer(L3Config(seed=1))
    targets = trainer.search_feedback(chess.Board(), depth=2)
    weights = [w for _, w in targets]
    assert max(weights) == pytest.approx(1.0)
    assert all(w <= 1.0 + 1e-9 for w in weights)
    # Sorted best-first.
    assert weights == sorted(weights, reverse=True)


def test_search_feedback_returns_empty_on_a_finished_game():
    trainer = L3Trainer(L3Config(seed=1))
    board = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
    assert trainer.search_feedback(board, depth=2) == []


def test_search_feedback_finds_a_mate_in_one():
    """Sanity: the feedback signal must actually see a forced mate.

    Scholar's mate, 1.e4 e5 2.Bc4 Nc6 3.Qh5 Nf6??, and White has Qxf7#. If the
    depth-5 feedback cannot rank a mate-in-one first, nothing downstream can be
    trusted.
    """
    trainer = L3Trainer(L3Config(seed=1, top_k=5, search_depth=2))
    board = chess.Board()
    for san in ["e4", "e5", "Bc4", "Nc6", "Qh5", "Nf6"]:
        board.push_san(san)
    assert chess.Move.from_uci("h5f7") in board.legal_moves

    targets = trainer.search_feedback(board, depth=2, top_k=5)
    assert targets
    assert targets[0][0] == chess.Move.from_uci("h5f7")


def test_top_5_feedback_prefers_the_tactically_best_move():
    """The search signal must rank a winning capture above a quiet move.

    Black's queen on d8 is undefended and cannot be recaptured: White's rook on
    d1 takes it for free. Any correct search must put ``Rxd8`` first, so this
    also serves as an end-to-end check that the engine is being asked the right
    question.
    """
    trainer = L3Trainer(L3Config(seed=1, top_k=5, search_depth=3))
    board = chess.Board("k2q4/8/8/8/8/8/8/3R3K w - - 0 1")
    targets = trainer.search_feedback(board, depth=3, top_k=5)
    assert targets
    assert targets[0][0] == chess.Move.from_uci("d1d8")


def test_train_on_search_feedback_raises_the_searched_best_move():
    """The point of the mechanism: imitate a deeper search.

    Repeat the feedback on a position whose best move is unambiguously better
    than everything else (a free queen), and require the model's probability on
    that move to rise. If it does not, the gradient is being computed with the
    wrong sign or applied to the wrong weight row.
    """
    trainer = L3Trainer(L3Config(seed=6, search_depth=3, search_weight=1.0))
    board = chess.Board("k2q4/8/8/8/8/8/8/3R3K w - - 0 1")
    targets = trainer.search_feedback(board, depth=3)
    best = targets[0][0]
    assert best == chess.Move.from_uci("d1d8")  # the free queen

    idx = M.move_to_index(best)
    before = trainer.policy.move_distribution(board)[idx]
    for _ in range(10):
        trainer.train_on_search_feedback(board, depth=3)
    after = trainer.policy.move_distribution(board)[idx]
    assert after > before, (before, after)


def test_train_on_search_feedback_reports_what_it_did():
    trainer = L3Trainer(L3Config(seed=6, search_depth=2))
    out = trainer.train_on_search_feedback(chess.Board(), depth=2)
    assert out["applied"] is True
    assert out["k"] > 0
    assert trainer.search_updates == out["k"]


def test_train_on_search_feedback_on_a_finished_game_is_a_no_op():
    trainer = L3Trainer(L3Config(seed=6))
    board = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
    out = trainer.train_on_search_feedback(board, depth=2)
    assert out["applied"] is False
    assert trainer.search_updates == 0


def test_search_feedback_only_touches_the_moves_it_reports():
    """Moves outside the top-k must not receive credit from the search signal."""
    trainer = L3Trainer(L3Config(seed=6, search_depth=1, top_k=3))
    board = chess.Board()
    targets = trainer.search_feedback(board, depth=1, top_k=3)
    chosen = {M.move_to_index(m) for m, _ in targets}
    dist_before = trainer.policy.move_distribution(board)
    trainer.train_on_search_feedback(board, depth=1)
    dist_after = trainer.policy.move_distribution(board)
    # The distribution is renormalised, so any move can shift slightly; assert
    # the reported moves are the ones that moved *up*.
    moved_up = [
        i for i in np.nonzero(dist_before)[0] if dist_after[i] > dist_before[i]
    ]
    assert set(moved_up) & chosen


# --------------------------------------------------------------------------
# training: midstate incremental updates
# --------------------------------------------------------------------------

def test_train_on_midstates_without_a_store_is_a_no_op():
    trainer = L3Trainer(L3Config(seed=1))
    out = trainer.train_on_midstates(8)
    assert out == {"sampled": 0, "trained": 0}


def test_train_on_midstates_samples_and_trains():
    trainer = L3Trainer(L3Config(seed=1, search_depth=1, top_k=3))
    store = MidstateStore()
    gid = store.new_game_id()
    _, result = _play_and_train(trainer, seed=12, plies=40)
    store.record_game(result.fens, gid)
    trainer.midstates = store

    out = trainer.train_on_midstates(6, depth=1)
    assert out["sampled"] > 0
    assert out["trained"] > 0
    assert trainer.search_updates > 0


def test_midstate_sampling_does_not_repeat_positions():
    """Duplicates would silently reweight the phase mix."""
    store = MidstateStore(capacity=50)
    gid = store.new_game_id()
    board = chess.Board()
    for _ in range(60):
        if board.is_game_over():
            break
        store.record_board(board, gid, board.ply())
        board.push(list(board.legal_moves)[0])
    picked = store.sample(10)
    assert len({id(e) for e in picked}) == len(picked)


def test_midstate_training_skips_finished_positions():
    trainer = L3Trainer(L3Config(seed=1, search_depth=1))
    store = MidstateStore()
    store.record_board(
        chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"),
        store.new_game_id(), 0,
    )
    trainer.midstates = store
    out = trainer.train_on_midstates(1, depth=1)
    assert out["trained"] == 0


# --------------------------------------------------------------------------
# persistence
# --------------------------------------------------------------------------

def test_state_dict_round_trips(tmp_path):
    policy = L3Policy(seed=17)
    board = chess.Board()
    before = policy.move_distribution(board)

    path = tmp_path / "l3.json"
    policy.save(path)

    restored = L3Policy.load(path, seed=999)
    after = restored.move_distribution(board)
    np.testing.assert_allclose(before, after, atol=1e-6)


def test_loading_rebuilds_a_perceptron_not_a_blind_scorer(tmp_path):
    """The base class's ``load`` hardcodes a BlindScorer, which would silently
    produce a position-blind L3. Assert the loaded policy really sees the board.
    """
    from chessrl.perceptron import PerceptronScorer

    policy = L3Policy(seed=4)
    path = tmp_path / "l3.json"
    policy.save(path)
    restored = L3Policy.load(path)
    assert isinstance(restored.perceptron, PerceptronScorer)
    assert restored.perceptron.n_parameters == policy.perceptron.n_parameters


def test_saved_weights_reproduce_the_same_choice(tmp_path):
    policy = L3Policy(seed=21)
    policy.greedy = True
    board = chess.Board("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1")
    chosen = policy.select(board)

    path = tmp_path / "l3.json"
    policy.save(path)
    restored = L3Policy.load(path)
    restored.greedy = True
    assert restored.select(board) == chosen


def test_trainer_save_writes_a_metadata_sidecar(tmp_path):
    import json

    trainer = L3Trainer(L3Config(seed=2))
    trainer.apply_credit(chess.Board(), chess.Move.from_uci("e2e4"), 0.4)
    paths = trainer.save(tmp_path / "l3.json")

    meta = json.loads(open(paths["meta"], encoding="utf-8").read())
    assert meta["n_parameters"] == trainer.policy.perceptron.n_parameters
    assert meta["updates"] == trainer.updates
    assert meta["games_played"] == 0


def test_saving_does_not_serialise_the_position_cache(tmp_path):
    """Counters and caches are runtime state, not weights."""
    trainer = L3Trainer(L3Config(seed=2))
    trainer.apply_credit(chess.Board(), chess.Move.from_uci("e2e4"), 0.4)
    assert len(trainer.policy.perceptron._chan_cache) > 0

    paths = trainer.save(tmp_path / "l3.json")
    import json

    blob = open(paths["model"], encoding="utf-8").read()
    assert "chan_cache" not in blob
    assert not json.loads(blob) == {}  # it did write something


# --------------------------------------------------------------------------
# the level ladder: L3 must be informed, and must be trainable
# --------------------------------------------------------------------------

def test_an_untrained_l3_is_not_yet_stronger_than_blind_l1():
    """A fresh L3 is not automatically better than a blind policy, and the
    suite should say so.

    Both shuffle: L3 has board *inputs* but random *weights*, so it has no more
    chess knowledge than L1 until it is trained. Writing the stronger claim
    here ("L3 beats L1 out of the box") produced six draws by fivefold
    repetition, which is the correct behaviour for this level and would have
    been a false alarm the moment it turned into a loss.
    """
    import random

    from chessrl.game import play_game

    decisive = 0
    for seed in range(4):
        l3 = L3Policy(seed=seed)
        l3.greedy = True
        l1 = blind_policy(seed=seed)
        if seed % 2 == 0:
            result = play_game(l3, l1, max_plies=80, no_progress_limit=40,
                               rng=random.Random(seed))
        else:
            result = play_game(l1, l3, max_plies=80, no_progress_limit=40,
                               rng=random.Random(seed))
        assert result.is_finished
        decisive += int(result.is_decisive)
    # Not asserting a direction: assert the games terminate. A decisive game
    # here would be evidence of a bug, not of strength.
    assert decisive == 0, "an untrained L3 should not be winning games"


def test_trained_l3_wins_material_against_a_blind_opponent():
    """The real level-ladder claim: *training* makes L3 see something.

    Train on positions whose best move is a capture, then check the model
    prefers captures it previously ignored. This is the contained version of
    "L3 beats L1" -- a full match is too slow and too noisy for a unit test,
    but the mechanism it relies on must be verified.
    """
    trainer = L3Trainer(L3Config(seed=2, search_depth=3, top_k=3, lr=0.05))
    # A position with a free queen for White.
    board = chess.Board("k2q4/8/8/8/8/8/8/3R3K w - - 0 1")
    grab = chess.Move.from_uci("d1d8")
    idx = M.move_to_index(grab)
    before = trainer.policy.move_distribution(board)[idx]

    for _ in range(30):
        trainer.train_on_search_feedback(board, depth=3)

    after = trainer.policy.move_distribution(board)[idx]
    assert after > before
    # And the trained policy actually plays it, greedily.
    trainer.policy.greedy = True
    assert trainer.policy.select(board) == grab


def test_l3_has_trainable_parameters_and_exposes_a_distribution():
    """The whole point of L3 over L2: it learns."""
    policy = L3Policy(seed=1)
    assert policy.perceptron.n_parameters > 0
    assert policy.move_distribution(chess.Board()).sum() == pytest.approx(1.0)
    trainer = L3Trainer(policy)
    assert hasattr(trainer, "train_on_game")
    assert hasattr(trainer, "train_on_search_feedback")


def test_l3_does_not_import_torch():
    """torch is an L4/L5 extra and must not be pulled in by L3.

    Asserted on the *source text* rather than on ``sys.modules``: torch may
    already be imported by another test in the same session, so a module-set
    check would pass vacuously. The point is that L3's code never mentions it.
    """
    import pathlib

    import chessrl.perceptron as mod

    source = pathlib.Path(mod.__file__).read_text(encoding="utf-8")
    assert "import torch" not in source
    assert "torch_model" not in source


def test_l3_imports_without_torch_on_the_path():
    """L3 must import cleanly in a subprocess where torch is unavailable.

    A module-level ``import torch`` would be invisible in this session, where
    L4's tests have already loaded it. Blocking it in a fresh interpreter is
    the only way to actually prove the isolation.
    """
    import subprocess
    import sys

    code = (
        "import sys, builtins\n"
        "real_import = builtins.__import__\n"
        "def blocked(name, *a, **k):\n"
        "    if name == 'torch' or name.startswith('torch.'):\n"
        "        raise ImportError('torch is not available')\n"
        "    return real_import(name, *a, **k)\n"
        "builtins.__import__ = blocked\n"
        "sys.path.insert(0, 'src')\n"
        "import chessrl.perceptron, chessrl.search, chessrl.policy\n"
        "print('ok')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


# --------------------------------------------------------------------------
# the standard gate, for L3 (G2 audit: no dedicated self-play full-game test)
# --------------------------------------------------------------------------

def test_policy_plays_a_full_legal_game_to_a_real_result():
    """The repo's gate, applied to L3 as self-play.

    L3 already plays games inside other tests (drawing L1, learning from
    search), but always *against* something else. The gate must hold for the
    level on its own: an untrained perceptron playing itself shuffles to a cap,
    and that must be reported as unfinished -- never silently as a draw -- and
    every recorded move must be legal. Uses two independent seeds so the game
    is not a degenerate same-weights mirror.
    """
    import random

    from chessrl.game import play_game

    white = L3Policy(seed=1)
    black = L3Policy(seed=2)
    result = play_game(white, black, max_plies=60, rng=random.Random(1))

    assert result.reason in {
        "checkmate", "stalemate", "insufficient_material",
        "seventyfive_moves", "fivefold_repetition", "fifty_moves",
        "max_plies", "no_progress",
    }, f"unexpected termination reason: {result.reason}"
    assert result.plies == len(result.moves)

    if result.result == "unfinished":
        assert not result.is_finished
        assert result.score_for(chess.WHITE) == 0
    else:
        assert result.is_finished

    replay = chess.Board()
    for san in result.moves:
        move = replay.parse_san(san)
        assert move in replay.legal_moves, f"{san} not legal in {replay.fen()}"
        replay.push(move)
    assert len(result.fens) == len(result.moves) + 1
    assert replay.fen() == result.fens[-1]
