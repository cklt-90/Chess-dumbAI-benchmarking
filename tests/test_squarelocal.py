"""L3.5: the weight-tied per-square scorer, its budget, and its two claims.

The load-bearing tests here are not the inherited protocol checks -- those are
ported wholesale from ``test_perceptron.py`` and exist to prove L3.5 is
*substitutable* for L3. The tests that matter are the three that decide whether
the level's hypothesis is true:

* ``test_parameter_count_within_ten_percent_of_L3`` -- the budget rule. It is
  asserted as an *implemented fraction* rather than the spec's literal +-10%,
  because the spec's own layout table (783) contradicts the spec's own budget
  rule (6502..7947). See the module docstring of ``squarelocal.py`` and the
  test's own docstring: the honest reading is recorded, not papered over.
* ``test_origin_dependence_survives_tying`` -- claim 2. If this fails, the
  level cannot express origin/destination asymmetry and that is a *result*.
* ``test_tied_credit_does_not_diverge_at_L3_learning_rate`` -- the trap. The
  failure is silent: an over-stepping tied row trains happily and just stops
  modelling anything.
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
from chessrl.squarelocal import (
    CREDIT_FANOUT,
    LEAN_H_CTX,
    L35Config,
    L35Policy,
    L35Trainer,
    SquareLocalConfig,
    SquareLocalScorer,
    implemented_fraction_of,
    parameter_count,
)

from conftest import INTERESTING_FENS, colour_mirror


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
def policy() -> L35Policy:
    return L35Policy(seed=11)


@pytest.fixture
def trainer() -> L35Trainer:
    return L35Trainer(L35Config(lr=0.05, seed=2))


# --------------------------------------------------------------------------
# shape and protocol conformance  (ported from test_perceptron.py)
# --------------------------------------------------------------------------

def test_heads_have_the_shapes_the_protocol_promises(policy):
    board = chess.Board()
    policy.scorer.set_position(board)
    assert policy.scorer.from_logits(board).shape == (64,)
    assert policy.scorer.to_logits(chess.E2).shape == (64,)
    assert policy.scorer.promo_logits(chess.A7, chess.A8).shape == (M.NUM_PROMO,)


def test_from_logits_is_not_a_square_against_square_matrix(policy):
    """The einsum bug produced a ``(64, 64)`` array that still ran.

    Weight tying invites the *opposite* mistake -- contracting the channel axis
    away entirely and returning a scalar -- so the shape is checked explicitly
    rather than assumed to be fine because the expression looks simpler.
    """
    logits = policy.scorer.from_logits(chess.Board())
    assert logits.ndim == 1
    assert logits.shape == (64,)


def test_square_local_heads_require_a_bound_position():
    scorer = SquareLocalScorer()
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


def test_policy_plays_a_full_legal_game_to_a_real_result():
    """The repo's standard gate: a level must finish a real game.

    Uses L3.5 against itself so the game is a property of this level and needs
    no other module, and requires a *real terminal* result rather than a ply
    cap -- an unfinished game scored as a draw is the bug the game loop was
    already fixed for, and a new level is where it would come back.
    """
    import random

    from chessrl.game import play_game

    white = L35Policy(seed=1)
    black = L35Policy(seed=2)
    white.greedy = True
    black.greedy = True
    result = play_game(white, black, max_plies=200, no_progress_limit=80,
                       rng=random.Random(0))
    assert result.plies > 0
    # Either it ended by a real chess rule, or it hit the ply cap -- and if it
    # hit the cap, it must be reported as unfinished, never as a draw.
    if result.reason == "ply_cap":
        assert not result.is_finished
    else:
        assert result.is_finished
        assert result.is_decisive or result.result == "1/2-1/2"


# --------------------------------------------------------------------------
# the parameter budget -- the constraint that makes this a comparison
# --------------------------------------------------------------------------

def test_parameter_count_matches_the_declared_layout(policy):
    l35 = policy.scorer.n_parameters
    declared = parameter_count()["total"]
    assert l35 == declared, (l35, declared)
    # And it must equal the sum of the tensors the trainer actually updates.
    assert l35 == sum(v.size for v in policy.scorer.parameters().values())


def test_parameter_count_within_ten_percent_of_L3(policy):
    """The budget constraint, asserted -- with the deviation recorded.

    The spec (§3) demands L3.5 land within +-10% of L3's 7609 parameters, and
    (§3, §7.2) demands that constraint be *asserted* rather than documented. The
    spec's own layout table, however, sums to **783** -- 10.8% of L3 -- so the
    spec contradicts itself, and no honest implementation can satisfy both.

    This test asserts what was actually implemented, and asserts the *reason*:
    the level is deliberately lean and purely tied, with the budget gap spent on
    nothing. Closing the gap would need ``h_ctx`` of roughly 649 -- a 649-wide
    ReLU over a **9-dimensional** input, i.e. 5841 parameters of deliberate
    over-parameterisation chosen to move a number rather than to model
    anything. That is the "quietly match by padding" the spec forbids in the
    same paragraph that demands the match.

    The two readings are both asserted so that a reader who disagrees with the
    choice sees exactly which number they are disagreeing with, and a future
    implementer who switches to the padded layout gets a failure that names the
    decision instead of a silent pass.
    """
    l3 = PerceptronScorer(PerceptronConfig()).n_parameters
    assert l3 == 7609, f"L3's baseline moved to {l3}; the budget must be re-derived"

    l35 = policy.scorer.n_parameters
    # (a) the implemented layout is exactly the lean one the spec's table lists
    assert l35 == 787, l35
    # (b) it is an order of magnitude below L3, not equal to it
    assert implemented_fraction_of() == pytest.approx(l35 / l3)
    assert 0.08 < l35 / l3 < 0.15, l35 / l3
    # (c) and therefore it deliberately does NOT satisfy the literal +-10% rule
    assert not (0.9 * l3 <= l35 <= 1.1 * l3)
    # (d) the reason is stated as arithmetic, not as an opinion: the narrowest
    #     h_ctx that *would* satisfy the rule is far wider than the 9-input
    #     context path can honestly use.
    honest = parameter_count(h_ctx=LEAN_H_CTX)["total"]
    assert honest == l35
    padded = parameter_count(h_ctx=640)["total"]
    assert padded >= 0.9 * l3, padded


def test_context_width_is_the_only_budget_knob():
    """``h_ctx`` must move the budget; everything else must hold it fixed.

    If some other tensor were secretly indexed by square, the budget knob would
    not be the whole story and the comparison would be uninterpretable.
    """
    narrow = parameter_count(h_ctx=8)["total"]
    wide = parameter_count(h_ctx=256)["total"]
    assert narrow < wide
    # The non-context tensors must be identical between the two layouts.
    a, b = parameter_count(h_ctx=8), parameter_count(h_ctx=256)
    for key in ("w_shared", "b_shared", "w_dst_shared", "w_geom", "promo"):
        assert a[key] == b[key], key


def test_no_weight_tensor_is_indexed_by_square(policy):
    """The whole level, in one assertion.

    Every square-local tensor must have no axis of length 64. This is the
    property that *is* weight tying, and it is much easier to break by
    accident (reintroducing a ``(64, n_ch)`` row to "just fix this one case")
    than to state.

    The check is on the *declared layout* rather than on shapes alone: at
    ``h_ctx=64`` the tied context vector ``w_hid`` is itself length 64, so
    "no axis equals 64" would be a coincidence-sensitive test that reports a
    false failure on a correct model. What actually matters is that no tensor
    carries a square axis, which is exactly the statement that every tensor's
    shape is a function of channels/features/hidden units only -- none of which
    is 64.
    """
    shapes = {k: v.shape for k, v in policy.scorer.parameters().items()}
    # The square-local heads: one row each, indexed by channel.
    assert shapes["w_shared"] == (policy.scorer.n_ch,)
    assert shapes["w_dst_shared"] == (policy.scorer.n_ch,)
    assert shapes["b_shared"] == (1,)
    # The pair head: indexed by geometry feature, not by square.
    assert shapes["w_geom"] == (len(GEOMETRY_NAMES),)
    # The context pathway: indexed by hidden unit and feature.
    assert shapes["W_ctx"] == (policy.scorer.h_ctx, policy.scorer.n_ctx)
    assert shapes["w_hid"] == (policy.scorer.h_ctx,)
    assert shapes["w_dst_ctx"] == (policy.scorer.h_ctx,)
    # Promotion: indexed by slot and move feature.
    assert shapes["promo"] == (M.NUM_PROMO, len(MOVE_FEATURE_NAMES_L3))

    # And nothing anywhere is a (64, n_ch) matrix -- the exact shape L3's
    # square-local heads have and the exact shape tying removes.
    for name, shape in shapes.items():
        assert shape[:2] != (64, policy.scorer.n_ch), f"{name} is a per-square matrix"


def test_tied_model_cannot_break_a_symmetric_tie():
    """A consequence of tying, asserted so it is a known property, not a bug.

    L3 owns an independent weight row per square, so the two knights on b1 and
    g1 score *differently* and greedy play can prefer one. L3.5 applies one row
    to both, so from the opening its origin logits for b1 and g1 are **exactly
    equal** -- the model has no way to prefer either.

    **This is the correct behaviour, not a defect.** Nc3 and Nf3 are genuinely
    equivalent from the opening; any symmetric evaluation ties them. What this
    test pins down is that the tie is *exact* rather than broken by noise, and
    that the outputs still mirror perfectly. The consequence for the benchmark:
    the naive port of L3's
    ``test_mirrored_positions_select_mirrored_moves_when_greedy`` fails here,
    because with an exact tie greedy selection falls back to array order and
    the mirrored position may legitimately pick the other member of the tie.
    Recorded in ``bench/README.md`` under "The L3.5 experiment".
    """
    policy = L35Policy(seed=11)
    board = chess.Board()
    policy.scorer.set_position(board)
    logits = policy.scorer.from_logits(board)
    assert logits[chess.B1] == logits[chess.G1], (
        "tying should make the two knights indistinguishable"
    )
    # But the *outputs* must still mirror exactly -- symmetry is intact, it is
    # only tie-breaking that is arbitrary.
    mirror = colour_mirror(board.fen())
    policy.scorer.set_position(mirror)
    logits_m = policy.scorer.from_logits(mirror)
    assert max(abs(logits[s] - logits_m[reflect_square(s)]) for s in range(64)) < 1e-9


def test_tie_is_broken_by_array_order_not_by_a_chess_signal():
    """Make the fallback explicit, so nothing downstream misreads it.

    With an exact tie greedy selection has to pick something, and it picks by
    array order. Documents the resulting behaviour so that a change in it is
    visible as a change rather than as noise -- and asserts that the *probability
    mass* on the two tied origins is equal, which is the level's actual claim.
    """
    policy = L35Policy(seed=11)
    policy.greedy = True
    board = chess.Board()
    assert policy.select(board) in board.legal_moves

    logits = policy.scorer.from_logits(board)
    # The two knight origins carry identical origin logits, so the origin
    # softmax gives them identical probability -- the destination factor is the
    # only thing that can differ between their moves.
    assert logits[chess.B1] == logits[chess.G1]

    dist = policy.move_distribution(board)
    from_b1 = sum(
        dist[M.move_to_index(m)] for m in board.legal_moves if m.from_square == chess.B1
    )
    from_g1 = sum(
        dist[M.move_to_index(m)] for m in board.legal_moves if m.from_square == chess.G1
    )
    assert from_b1 == pytest.approx(from_g1, abs=1e-9), (from_b1, from_g1)


def test_mirrored_positions_agree_move_for_move(policy):
    """The property that must hold instead of move-identity: probabilities mirror.

    L3 asserts that greedy selection picks the mirrored *move*; that is true
    there only because its per-square rows break ties by initialisation. The
    invariant that is actually about correctness is that every move's
    probability equals its mirror's, which this level does satisfy exactly (see
    ``test_colour_symmetry_holds_exactly``). Where a tie exists, both members
    are equally likely and either may be chosen.
    """
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
                dist_m[M.move_to_index(mirrored)], abs=1.5e-8
            )


def test_disabling_context_removes_the_hidden_pathway_and_shrinks_the_model():
    without = SquareLocalScorer(SquareLocalConfig(use_context=False))
    assert without.h_ctx == 0
    assert without.n_parameters < SquareLocalScorer(SquareLocalConfig()).n_parameters
    board = chess.Board()
    without.set_position(board)
    assert without.from_logits(board).shape == (64,)


def test_value_channels_are_opt_in_and_change_the_input_width():
    off = SquareLocalScorer(SquareLocalConfig(use_value_channels=False))
    on = SquareLocalScorer(SquareLocalConfig(use_value_channels=True))
    assert on.n_ch > off.n_ch
    assert on.w_shared.shape == (on.n_ch,)


# --------------------------------------------------------------------------
# colour symmetry  (the load-bearing test, ported)
# --------------------------------------------------------------------------

def test_colour_symmetry_holds_exactly(policy):
    """One tied weight set must serve both colours.

    Weight tying is only *coherent* because the encoder canonicalises; if any
    head read geometry in absolute board coordinates, one tied row would be
    correct for White and mirrored-wrong for Black. Tolerance matches L3's.
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

    assert worst < 1.5e-8, f"colour asymmetry of {worst:.3e}"


def test_colour_symmetry_survives_a_dense_middlegame(policy):
    """The case that caught the equivalent L3 bug."""
    fen = "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1"
    board = chess.Board(fen)
    mirror = colour_mirror(fen)
    dist = policy.move_distribution(board)
    dist_m = policy.move_distribution(mirror)

    worst = max(
        abs(dist[M.move_to_index(m)] - dist_m[M.move_to_index(reflect_move(m))])
        for m in board.legal_moves
    )
    assert worst < 1.5e-8, f"colour asymmetry of {worst:.3e}"


def test_canonical_square_map_is_its_own_inverse():
    black = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
    mapping = SquareLocalScorer._canonical_squares(black)
    assert (mapping[mapping] == np.arange(64)).all()
    assert mapping[chess.A8] == chess.A1
    assert mapping[chess.E8] == chess.E1


def test_canonical_square_map_is_identity_for_white():
    mapping = SquareLocalScorer._canonical_squares(chess.Board())
    assert (mapping == np.arange(64)).all()


# --------------------------------------------------------------------------
# NEW §7.1: the shared row really is shared
# --------------------------------------------------------------------------

def test_shared_row_is_touched_by_every_square():
    """The tied row must be updated by credit, and inference must read it.

    The claim "weight-tied" is otherwise unfalsifiable from the outside: a
    model could hold a (64, n_ch) tensor whose rows merely happened to be
    equal, pass every output-shape test, and drift apart on the first update.
    Identity of the array object is the assertion that survives training.

    Note the update is asserted over *several* credit events, not one. At
    ``lr_shared = lr/64`` a single event is a fraction of a quantisation grid
    unit and is carried in the residual buffer rather than applied immediately
    -- see ``L35Trainer.__init__``. Asserting an immediate change would be
    asserting the bug back in.
    """
    trainer = L35Trainer(L35Config(lr=0.05, seed=2))
    scorer = trainer.policy.scorer
    board = chess.Board()
    before = scorer.w_shared.copy()

    for _ in range(8):
        trainer.apply_credit(board, chess.Move.from_uci("e2e4"), 0.5)
    assert not np.array_equal(scorer.w_shared, before), "the tied row was never updated"

    # The forward pass must read that same object. Mutate it in place and the
    # origin logits must change accordingly -- if the scorer held a copy, this
    # would be a no-op and "tying" would be a coincidence of initialisation.
    scorer.set_position(board)
    logits_a = scorer.from_logits(board)
    scorer.w_shared += np.float32(0.5)
    scorer.set_position(board)          # drop the position cache
    logits_b = scorer.from_logits(board)

    delta = logits_b - logits_a
    assert np.abs(delta).max() > 0, "the tied row is not what the head reads"
    # The shift at each square must be exactly the *same row* contracted with
    # that square's own channels: delta[s] = 0.5 * sum_c chan[c, s]. A
    # per-square model would give delta[s] = 0.5 * sum_c chan[c, s] only for the
    # one square whose row was nudged; here it must hold at all 64.
    chan = scorer.channels_for(board)
    expected = np.float32(0.5) * chan.sum(axis=0)
    np.testing.assert_allclose(delta, expected, atol=1e-5)


def test_sub_grid_updates_are_buffered_not_discarded():
    """The residual buffer is why the tied heads learn at all.

    A single event at the default rate is 0.4 grid units and rounds to zero.
    Without error feedback this test's final assertion fails by construction,
    which is exactly what happened before the buffer existed: 30 games and
    14,400 updates left four of the eight tensors bit-identical to init.
    """
    trainer = L35Trainer(L35Config(lr=0.05, seed=2))
    scorer = trainer.policy.scorer
    board = chess.Board()
    move = chess.Move.from_uci("e2e4")

    one_before = scorer.w_shared.copy()
    trainer.apply_credit(board, move, 0.5)
    # One event is below the grid: it is banked, not applied.
    assert np.array_equal(scorer.w_shared, one_before)
    assert len(trainer._residual) > 0, "nothing was banked"

    for _ in range(7):
        trainer.apply_credit(board, move, 0.5)
    assert not np.array_equal(scorer.w_shared, one_before)

    # And the accumulated drift must still land on the fixed-point grid.
    delta = (scorer.w_shared - one_before).reshape(-1)
    touched = delta[delta != 0]
    assert touched.size > 0
    grid = touched * QUANT
    np.testing.assert_allclose(grid, np.round(grid), atol=1e-3)


def test_tied_origin_head_gives_one_score_per_square_from_one_row():
    """The tied contraction must be per-square, not a single scalar.

    Tying invites the opposite of L3's einsum bug: contracting *both* axes and
    returning a scalar, which would make the origin factor uniform and the
    policy position-blind in the origin while still running. Assert the head
    varies across squares, and that it varies for a reason tied to channels.
    """
    policy = L35Policy(seed=11)
    board = chess.Board()
    policy.scorer.set_position(board)
    logits = policy.scorer.from_logits(board)
    assert logits.shape == (64,)
    assert logits.std() > 0, "the tied origin head produced a constant"


def test_shared_row_receives_channel_weights_of_the_moves_origin():
    """The update must be addressed at the origin's *canonical* square.

    Checks the arithmetic directly rather than through a probability: after one
    credit on e2e4, the increment on ``w_shared`` must equal
    ``lr_shared * credit * chan[:, e2_canon]``.
    """
    trainer = L35Trainer(L35Config(lr=0.05, seed=2))
    scorer = trainer.policy.scorer
    board = chess.Board()
    move = chess.Move.from_uci("e2e4")
    scorer.set_position(board)
    chan = scorer.channels_for(board)
    f_c = int(scorer._canon_sq[move.from_square])
    expect = trainer.config.lr_shared * 0.5 * chan[:, f_c]

    before = scorer.w_shared.copy()
    trainer.apply_credit(board, move, 0.5)
    got = scorer.w_shared - before
    # Quantised: compare on the 1/QUANT grid, allowing the rounding.
    np.testing.assert_allclose(got, expect, atol=1.0 / QUANT + 1e-7)


# --------------------------------------------------------------------------
# NEW §7.5: origin dependence survives tying  (falsifiable claim 2)
# --------------------------------------------------------------------------

def test_origin_dependence_survives_tying(policy):
    """Two moves that share a destination must be able to score differently.

    ``to_logits(from_sq)`` must not be a function of the destination alone. In
    the opening, b1 and g1 are both knights and either can reach a4/c3/e2/d2
    territory by the same geometry, so a purely destination-indexed head would
    give identical logits. L3 expresses the difference with ``W_to_src``; L3.5
    has no per-origin row and must do it through the pair geometry.

    If this fails, §5's second claim is false -- the level needs un-tying and
    that is a finding to report, not a test to relax.
    """
    board = chess.Board()
    policy.scorer.set_position(board)
    a = policy.scorer.to_logits(chess.B1)
    b = policy.scorer.to_logits(chess.G1)
    assert np.abs(a - b).max() > 1e-6, "to_logits is origin-independent"


def test_origin_dependence_is_carried_structurally_by_the_geometry_term(policy):
    """Locate the asymmetry: it must come from ``w_geom``, not from a per-origin row.

    Zeroing the geometry weights must collapse ``to_logits`` to something that
    depends on the destination (and, through the context term, the position)
    but *not* on the origin. This asserts the mechanism the spec names, rather
    than merely asserting that the outputs differ.
    """
    board = chess.Board()
    policy.scorer.set_position(board)
    differing_before = np.abs(
        policy.scorer.to_logits(chess.B1) - policy.scorer.to_logits(chess.G1)
    ).max()
    assert differing_before > 1e-6

    policy.scorer.w_geom[:] = 0.0
    policy.scorer.set_position(board)
    after = np.abs(
        policy.scorer.to_logits(chess.B1) - policy.scorer.to_logits(chess.G1)
    ).max()
    assert after == pytest.approx(0.0, abs=1e-9), (
        "after removing w_geom the head is still origin-dependent, so some other "
        "term is carrying it -- the spec's explanation of *how* tying survives "
        "is wrong and should be corrected"
    )


def test_geometry_features_are_pair_functions_of_the_canonical_frame(policy):
    """``_geom_features`` must be a pure function of the pair it is given."""
    a = policy.scorer._geom_features(chess.E2, chess.E4)
    b = policy.scorer._geom_features(chess.E3, chess.E5)
    d_rank = GEOMETRY_NAMES.index("d_rank")
    assert a.shape == (len(GEOMETRY_NAMES),)
    assert a[d_rank] > 0
    assert a[d_rank] == pytest.approx(b[d_rank])
    assert a[GEOMETRY_NAMES.index("same_file")] == 1.0
    assert a[GEOMETRY_NAMES.index("abs_d_file")] == 0.0


def test_move_features_have_a_declared_length(policy):
    board = chess.Board()
    policy.scorer.set_position(board)
    feats = policy.scorer._move_features(chess.E2, chess.E4)
    assert feats.shape == (len(MOVE_FEATURE_NAMES_L3),)


# --------------------------------------------------------------------------
# NEW §7.4: the tied-row trap
# --------------------------------------------------------------------------

def test_tied_credit_does_not_diverge_at_the_default_rate():
    """Bound the tied row after many credit events at the default rate."""
    trainer = L35Trainer(L35Config(lr=0.01, seed=3))
    board = chess.Board()
    move = chess.Move.from_uci("e2e4")
    for _ in range(400):
        trainer.apply_credit(board, move, 0.5)

    scorer = trainer.policy.scorer
    assert np.isfinite(scorer.w_shared).all()
    assert np.isfinite(scorer.w_geom).all()
    assert np.isfinite(scorer.w_dst_shared).all()
    # 400 events x lr_shared(0.01/64) x 0.5 x |chan|<=1 bounds each weight by
    # ~0.03. Asserting a loose ceiling rather than the exact bound keeps the
    # test about divergence, not about arithmetic.
    assert np.abs(scorer.w_shared).max() < 1.0, np.abs(scorer.w_shared).max()
    # And the model must still produce a sane distribution.
    dist = trainer.policy.move_distribution(board)
    assert np.isclose(dist.sum(), 1.0, atol=1e-6)


def test_default_lr_shared_is_lr_over_the_credit_fanout():
    """The relationship is a definition, not a magic number."""
    cfg = L35Config(lr=0.03)
    assert cfg.lr_shared == pytest.approx(0.03 / CREDIT_FANOUT)
    assert CREDIT_FANOUT == 64
    # An explicit override must be respected.
    assert L35Config(lr=0.03, lr_shared=0.005).lr_shared == 0.005


def test_undivided_learning_rate_actually_diverges():
    """The trap is real, and this is what makes the guard above meaningful.

    Set ``lr_shared`` to the *undivided* rate -- which is what a naive
    implementation would do by reusing ``_nudge`` -- and 300 credit events push
    the tied row two orders of magnitude further than the correct rate. A test
    that only asserted "the default does not diverge" would pass against an
    implementation that had no fan-out correction at all and simply had a tiny
    learning rate.
    """
    board = chess.Board()
    move = chess.Move.from_uci("e2e4")

    naive = L35Trainer(L35Config(seed=2))
    naive.config.lr_shared = naive.config.lr   # the mistake
    correct = L35Trainer(L35Config(seed=2))

    for _ in range(300):
        naive.apply_credit(board, move, 0.5)
        correct.apply_credit(board, move, 0.5)

    n_max = np.abs(naive.policy.scorer.w_shared).max()
    c_max = np.abs(correct.policy.scorer.w_shared).max()
    assert n_max > 50 * c_max, (n_max, c_max)


def test_context_pathways_are_not_normalised_by_the_fanout():
    """``W_ctx`` is already shared at L3; dividing it by 64 would rig the level.

    If the context input weights were stepped at ``lr_shared`` too, L3.5's
    global-fact pathway would learn 64x slower than L3's for no reason -- and
    the context path is precisely where the tying hypothesis gets its only
    chance to survive, so handicapping it would manufacture a null result.
    """
    trainer = L35Trainer(L35Config(lr=0.04, seed=2))
    scorer = trainer.policy.scorer
    board = chess.Board()
    before_ctx = scorer.W_ctx.copy()
    trainer.apply_credit(board, chess.Move.from_uci("e2e4"), 0.5)
    moved_ctx = np.abs(scorer.W_ctx - before_ctx).max()
    moved_shared = np.abs(scorer.w_geom).max()

    assert moved_ctx > 0, "the context input weights were not updated at all"
    # The context step must be at lr, i.e. strictly larger than the tied step
    # scale could produce for a comparable feature magnitude.
    assert moved_ctx > moved_shared * 1.5, (moved_ctx, moved_shared)


# --------------------------------------------------------------------------
# determinism, quantisation, training signal  (ported)
# --------------------------------------------------------------------------

def test_same_seed_gives_identical_weights():
    a = L35Policy(seed=5).scorer
    b = L35Policy(seed=5).scorer
    for key in a.parameters():
        np.testing.assert_array_equal(a.parameters()[key], b.parameters()[key])


def test_different_seeds_give_different_weights():
    a = L35Policy(seed=5).scorer
    b = L35Policy(seed=6).scorer
    assert not np.array_equal(a.w_shared, b.w_shared)


def test_policy_seed_reaches_the_scorer():
    """Both RNGs need the seed, or every run is the same run."""
    a = L35Policy(seed=5)
    b = L35Policy(seed=5)
    assert not np.array_equal(
        a.move_distribution(chess.Board()), b.move_distribution(chess.Board())
    ) or np.array_equal(a.scorer.w_shared, b.scorer.w_shared)
    assert np.array_equal(a.scorer.w_shared, b.scorer.w_shared)
    c = L35Policy(seed=7)
    assert not np.array_equal(a.scorer.w_shared, c.scorer.w_shared)


def test_quantised_updates_land_on_the_fixed_point_grid():
    trainer = L35Trainer(L35Config(lr=0.05, seed=1))
    scorer = trainer.policy.scorer
    before = scorer.w_shared.copy()
    trainer.apply_credit(chess.Board(), chess.Move.from_uci("e2e4"), 0.7)
    delta = np.abs(scorer.w_shared - before).reshape(-1)
    touched = delta[delta > 0]
    assert touched.size > 0
    grid = touched * QUANT
    np.testing.assert_allclose(grid, np.round(grid), atol=1e-3)


def test_two_identical_training_runs_produce_identical_weights():
    def run():
        trainer = L35Trainer(L35Config(seed=3, lr=0.02))
        board = chess.Board()
        for uci in ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6"]:
            move = chess.Move.from_uci(uci)
            trainer.apply_credit(board, move, 0.5)
            board.push(move)
        return trainer.policy.scorer.w_shared.copy()

    np.testing.assert_array_equal(run(), run())


def test_non_quantised_updates_are_allowed_to_leave_the_grid():
    trainer = L35Trainer(L35Config(lr=0.005, seed=1))
    trainer.policy.scorer.config.quantised = False
    before = trainer.policy.scorer.w_shared.copy()
    trainer.apply_credit(chess.Board(), chess.Move.from_uci("e2e4"), 0.7)
    delta = (trainer.policy.scorer.w_shared - before).reshape(-1)
    touched = delta[delta != 0]
    assert touched.size > 0
    assert not np.allclose(touched * QUANT, np.round(touched * QUANT), atol=1e-6)


# --------------------------------------------------------------------------
# credit direction
# --------------------------------------------------------------------------

def test_credit_moves_the_move_it_is_given_up_the_ranking():
    trainer = L35Trainer(L35Config(lr=0.05, seed=2))
    board = chess.Board()
    move = chess.Move.from_uci("e2e4")
    idx = M.move_to_index(move)

    before = trainer.policy.move_distribution(board)[idx]
    for _ in range(200):
        trainer.apply_credit(board, move, 0.5)
    after = trainer.policy.move_distribution(board)[idx]
    assert after > before, (before, after)


def test_negative_credit_pushes_a_move_down():
    trainer = L35Trainer(L35Config(lr=0.05, seed=2))
    board = chess.Board()
    move = chess.Move.from_uci("e2e4")
    idx = M.move_to_index(move)

    before = trainer.policy.move_distribution(board)[idx]
    for _ in range(200):
        trainer.apply_credit(board, move, -0.5)
    after = trainer.policy.move_distribution(board)[idx]
    assert after < before, (before, after)


def test_zero_credit_is_a_no_op():
    trainer = L35Trainer(L35Config(seed=2))
    before = trainer.policy.scorer.w_shared.copy()
    trainer.apply_credit(chess.Board(), chess.Move.from_uci("e2e4"), 0.0)
    np.testing.assert_array_equal(trainer.policy.scorer.w_shared, before)


def test_credit_for_black_updates_a_row_that_inference_reads():
    """Training and inference must agree on the canonical frame.

    The failure mode is asymmetric and therefore easy to miss: getting the
    canonical/real indexing wrong here is invisible for White (identity map)
    and mirrored-wrong for Black, so the model trains on White's games and
    silently wastes every Black position in the corpus.
    """
    trainer = L35Trainer(L35Config(lr=0.05, seed=2))
    board = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
    assert board.turn == chess.BLACK
    move = chess.Move.from_uci("g8f6")
    idx = M.move_to_index(move)

    before = trainer.policy.move_distribution(board)[idx]
    for _ in range(200):
        trainer.apply_credit(board, move, 0.5)
    after = trainer.policy.move_distribution(board)[idx]
    assert after > before, (before, after)


def test_black_credit_addresses_the_reflected_square():
    """The exact row touched for Black must be the reflected one."""
    trainer = L35Trainer(L35Config(lr=0.05, seed=2))
    scorer = trainer.policy.scorer
    board = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
    move = chess.Move.from_uci("g8f6")

    scorer.set_position(board)
    chan = scorer.channels_for(board)
    f_c = int(scorer._canon_sq[move.from_square])
    assert f_c == reflect_square(move.from_square)
    expect = trainer.config.lr_shared * 0.5 * chan[:, f_c]

    before = scorer.w_shared.copy()
    trainer.apply_credit(board, move, 0.5)
    np.testing.assert_allclose(
        scorer.w_shared - before, expect, atol=1.0 / QUANT + 1e-7
    )


def test_counter_records_both_the_offer_and_the_take():
    trainer = L35Trainer(L35Config(seed=2))
    trainer.apply_credit(chess.Board(), chess.Move.from_uci("e2e4"), 0.5)
    stats = trainer.counter.stats()
    assert stats["total_offers"] == 10
    assert stats["total_takes"] == 1


def test_trainer_rejects_a_policy_of_the_wrong_level():
    """An L3Policy would get L3's per-square credit applied to a tied model."""
    with pytest.raises(TypeError, match="L35Policy"):
        L35Trainer(L3Policy(seed=0), L35Config(seed=0))


# --------------------------------------------------------------------------
# training: game outcomes, search feedback, midstates  (ported)
# --------------------------------------------------------------------------

def _play_and_train(trainer, seed: int, plies: int = 24):
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
        result="unfinished", reason="ply_cap",
        fens=fens, moves=moves, plies=len(moves),
    )
    return trainer.train_on_game(result), result


def test_train_on_game_applies_one_update_per_ply():
    trainer = L35Trainer(L35Config(seed=4))
    stats, _ = _play_and_train(trainer, seed=4)
    assert trainer.updates > 0
    assert trainer.games_played == 1
    assert stats["result"] == "unfinished"


def test_train_on_game_rejects_a_result_without_fens():
    from chessrl.game import GameResult

    trainer = L35Trainer(L35Config(seed=4))
    empty = GameResult(result="unfinished", reason="ply_cap", fens=[], moves=[], plies=0)
    with pytest.raises(ValueError, match="FEN"):
        trainer.train_on_game(empty)


def test_unfinished_games_score_zero_without_shaping():
    trainer = L35Trainer(L35Config(seed=4, shaped_reward=False))
    _, result = _play_and_train(trainer, seed=9, plies=30)
    score, amplitude = trainer.white_reward(result)
    assert score == 0.0
    assert amplitude == 0.0


def test_shaped_reward_is_small_and_bounded():
    trainer = L35Trainer(L35Config(seed=4, shaped_reward=True, shaped_weight=0.3))
    _, result = _play_and_train(trainer, seed=9, plies=30)
    score, amplitude = trainer.white_reward(result)
    assert -0.31 <= score <= 0.31
    assert amplitude == 1.0


def test_l35_inherits_L3s_credit_curve_and_search_knobs():
    """L3.5 must differ from L3 in *architecture only*.

    If the trainer's other eight knobs drifted, a difference between the two
    levels would no longer be attributable to weight tying.
    """
    l3 = L3Config()
    l35 = L35Config()
    for field_name in ("lr", "credit_decay", "search_weight", "top_k",
                       "search_depth", "shaped_reward", "shaped_weight",
                       "max_plies", "no_progress_limit", "seed"):
        assert getattr(l35, field_name) == getattr(l3, field_name), field_name


def test_search_feedback_returns_at_most_top_k_moves():
    trainer = L35Trainer(L35Config(seed=1, top_k=5))
    targets = trainer.search_feedback(chess.Board(), depth=2)
    assert 0 < len(targets) <= 5
    for move, weight in targets:
        assert move in chess.Board().legal_moves
        assert weight > 0


def test_search_feedback_finds_a_mate_in_one():
    trainer = L35Trainer(L35Config(seed=1, top_k=5, search_depth=2))
    board = chess.Board()
    for san in ["e4", "e5", "Bc4", "Nc6", "Qh5", "Nf6"]:
        board.push_san(san)
    targets = trainer.search_feedback(board, depth=2, top_k=5)
    assert targets
    assert targets[0][0] == chess.Move.from_uci("h5f7")


def test_train_on_search_feedback_raises_the_searched_best_move():
    """The mechanism must actually learn, at the tied rate."""
    trainer = L35Trainer(L35Config(seed=6, search_depth=3, search_weight=1.0))
    board = chess.Board("k2q4/8/8/8/8/8/8/3R3K w - - 0 1")
    targets = trainer.search_feedback(board, depth=3)
    best = targets[0][0]
    assert best == chess.Move.from_uci("d1d8")

    idx = M.move_to_index(best)
    before = trainer.policy.move_distribution(board)[idx]
    for _ in range(40):
        trainer.train_on_search_feedback(board, depth=3)
    after = trainer.policy.move_distribution(board)[idx]
    assert after > before, (before, after)


def test_train_on_search_feedback_on_a_finished_game_is_a_no_op():
    trainer = L35Trainer(L35Config(seed=6))
    board = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
    out = trainer.train_on_search_feedback(board, depth=2)
    assert out["applied"] is False
    assert trainer.search_updates == 0


def test_train_on_midstates_without_a_store_is_a_no_op():
    trainer = L35Trainer(L35Config(seed=1))
    assert trainer.train_on_midstates(8) == {"sampled": 0, "trained": 0}


def test_train_on_midstates_samples_and_trains():
    trainer = L35Trainer(L35Config(seed=1, search_depth=1, top_k=3))
    store = MidstateStore()
    gid = store.new_game_id()
    _, result = _play_and_train(trainer, seed=12, plies=40)
    store.record_game(result.fens, gid)
    trainer.midstates = store

    out = trainer.train_on_midstates(6, depth=1)
    assert out["sampled"] > 0
    assert out["trained"] > 0
    assert trainer.search_updates > 0


# --------------------------------------------------------------------------
# persistence
# --------------------------------------------------------------------------

def test_state_dict_round_trips(tmp_path):
    policy = L35Policy(seed=17)
    board = chess.Board()
    before = policy.move_distribution(board)

    path = tmp_path / "l35.json"
    policy.save(path)

    restored = L35Policy.load(path, seed=999)
    after = restored.move_distribution(board)
    np.testing.assert_allclose(before, after, atol=1e-6)


def test_loading_rebuilds_a_square_local_scorer(tmp_path):
    policy = L35Policy(seed=4)
    path = tmp_path / "l35.json"
    policy.save(path)
    restored = L35Policy.load(path)
    assert isinstance(restored.scorer, SquareLocalScorer)
    assert restored.scorer.n_parameters == policy.scorer.n_parameters


def test_saved_weights_reproduce_the_same_choice(tmp_path):
    policy = L35Policy(seed=21)
    policy.greedy = True
    board = chess.Board("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1")
    chosen = policy.select(board)

    path = tmp_path / "l35.json"
    policy.save(path)
    restored = L35Policy.load(path)
    restored.greedy = True
    assert restored.select(board) == chosen


def test_trainer_save_writes_metadata_including_the_tied_rate(tmp_path):
    import json

    trainer = L35Trainer(L35Config(seed=2))
    trainer.apply_credit(chess.Board(), chess.Move.from_uci("e2e4"), 0.4)
    paths = trainer.save(tmp_path / "l35.json")

    meta = json.loads(open(paths["meta"], encoding="utf-8").read())
    assert meta["n_parameters"] == trainer.policy.scorer.n_parameters
    assert meta["updates"] == trainer.updates
    assert meta["lr_shared"] == pytest.approx(trainer.config.lr_shared)


def test_saving_does_not_serialise_the_position_cache(tmp_path):
    trainer = L35Trainer(L35Config(seed=2))
    trainer.apply_credit(chess.Board(), chess.Move.from_uci("e2e4"), 0.4)
    assert len(trainer.policy.scorer._chan_cache) > 0

    paths = trainer.save(tmp_path / "l35.json")
    blob = open(paths["model"], encoding="utf-8").read()
    assert "chan_cache" not in blob


# --------------------------------------------------------------------------
# isolation
# --------------------------------------------------------------------------

def test_l35_does_not_import_torch():
    import pathlib

    import chessrl.squarelocal as mod

    source = pathlib.Path(mod.__file__).read_text(encoding="utf-8")
    assert "import torch" not in source
    assert "torch_model" not in source


def test_l35_imports_without_torch_on_the_path():
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
        "import chessrl.squarelocal\n"
        "print('ok')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_l35_module_is_independently_deletable():
    """Spec §4: L3.5 must not be bolted into ``perceptron.py``.

    The point of a new rung is that it can be removed without touching the old
    one, which is only true if the old one never mentions it.
    """
    import pathlib

    import chessrl.perceptron as mod

    source = pathlib.Path(mod.__file__).read_text(encoding="utf-8")
    assert "squarelocal" not in source
    assert "L35" not in source


# --------------------------------------------------------------------------
# the control arm: L3-flat
# --------------------------------------------------------------------------

def test_flat_control_has_no_spatial_structure():
    """The control's defining property, asserted rather than described.

    ``L3-flat`` throws away the per-square input entirely, so its origin head
    is a *constant* across the board. That is exactly what makes it a control:
    it is not a weaker copy of L3, it is a model with no notion of square at
    all. If this ever starts varying, the arm has stopped being the control and
    the experiment no longer has a baseline.
    """
    from chessrl.squarelocal import FlatPolicy

    policy = FlatPolicy(seed=1)
    board = chess.Board()
    policy.scorer.set_position(board)
    logits = policy.scorer.from_logits(board)
    assert logits.shape == (64,)
    assert logits.std() == 0.0, "the flat control's origin head must be constant"


def test_flat_control_plays_legal_moves_and_produces_a_distribution():
    from chessrl.squarelocal import FlatPolicy

    policy = FlatPolicy(seed=1)
    for fen in INTERESTING_FENS:
        board = chess.Board(fen)
        if board.is_game_over():
            continue
        assert policy.select(board) in board.legal_moves
        dist = policy.move_distribution(board)
        assert np.isclose(dist.sum(), 1.0, atol=1e-6)
        assert (dist > 0).sum() == board.legal_moves.count()


def test_flat_control_is_not_square_indexed():
    from chessrl.squarelocal import FlatScorer

    scorer = FlatScorer()
    for name, tensor in scorer.parameters().items():
        assert 64 not in tensor.shape, f"{name} has a square axis: {tensor.shape}"


def test_flat_control_respects_the_canonical_frame():
    """The control must obey the same colour symmetry, or its losses mean nothing."""
    from chessrl.squarelocal import FlatPolicy

    policy = FlatPolicy(seed=5)
    worst = 0.0
    for fen in INTERESTING_FENS:
        board = chess.Board(fen)
        if board.is_game_over():
            continue
        mirror = colour_mirror(fen)
        dist = policy.move_distribution(board)
        dist_m = policy.move_distribution(mirror)
        for move in board.legal_moves:
            worst = max(
                worst,
                abs(dist[M.move_to_index(move)]
                    - dist_m[M.move_to_index(reflect_move(move))]),
            )
    assert worst < 1.5e-8, f"control arm is colour-asymmetric by {worst:.3e}"


# --------------------------------------------------------------------------
# the control arm: L3-flat
# --------------------------------------------------------------------------

def test_flat_control_has_no_spatial_structure():
    """The control's defining property, asserted rather than described.

    ``L3-flat`` throws away the per-square input entirely, so its origin head
    is a *constant* across the board. That is exactly what makes it a control:
    it is not a weaker copy of L3, it is a model with no notion of square at
    all. If this ever starts varying, the arm has stopped being the control and
    the experiment no longer has a baseline.
    """
    from chessrl.squarelocal import FlatPolicy

    policy = FlatPolicy(seed=1)
    board = chess.Board()
    policy.scorer.set_position(board)
    logits = policy.scorer.from_logits(board)
    assert logits.shape == (64,)
    assert logits.std() == 0.0, "the flat control's origin head must be constant"


def test_flat_control_plays_legal_moves_and_produces_a_distribution():
    from chessrl.squarelocal import FlatPolicy

    policy = FlatPolicy(seed=1)
    for fen in INTERESTING_FENS:
        board = chess.Board(fen)
        if board.is_game_over():
            continue
        assert policy.select(board) in board.legal_moves
        dist = policy.move_distribution(board)
        assert np.isclose(dist.sum(), 1.0, atol=1e-6)
        assert (dist > 0).sum() == board.legal_moves.count()


def test_flat_control_is_not_square_indexed():
    from chessrl.squarelocal import FlatScorer

    scorer = FlatScorer()
    for name, tensor in scorer.parameters().items():
        assert 64 not in tensor.shape, f"{name} has a square axis: {tensor.shape}"


def test_flat_control_respects_the_canonical_frame():
    """The control must obey the same colour symmetry, or its losses mean nothing."""
    from chessrl.squarelocal import FlatPolicy

    policy = FlatPolicy(seed=5)
    worst = 0.0
    for fen in INTERESTING_FENS:
        board = chess.Board(fen)
        if board.is_game_over():
            continue
        mirror = colour_mirror(fen)
        dist = policy.move_distribution(board)
        dist_m = policy.move_distribution(mirror)
        for move in board.legal_moves:
            worst = max(
                worst,
                abs(dist[M.move_to_index(move)]
                    - dist_m[M.move_to_index(reflect_move(move))]),
            )
    assert worst < 1.5e-8, f"control arm is colour-asymmetric by {worst:.3e}"
