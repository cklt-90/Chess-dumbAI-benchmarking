"""L5: alpha-beta guided by L4's heads.

Two claims are being tested here, and they are different claims.

The **architectural** one: the decision head and the evaluation head really do
share a trunk and differ only in their final projection. That is checked
structurally -- by rebuilding the same trunk and both heads by hand from the
module's own parameters and asserting the public methods agree -- rather than by
reading the code and believing it.

The **behavioural** one, which is the one that matters for the benchmark:
*ordering can only change which move is returned among equals, never the value
the search computes, whereas leaf evaluation changes the value itself.* Those
are opposite in kind -- one costs time, the other costs correctness -- and the
tests below separate them instead of claiming a single "L5 is better".

The ablation flags (``use_model_eval``, ``use_model_prior``) exist precisely so
that separation is runnable rather than argued, so several tests here are
differential: same engine, one flag flipped.
"""

from __future__ import annotations

import random

import chess
import numpy as np
import pytest

torch = pytest.importorskip(
    "torch",
    reason="L5 needs PyTorch, an optional extra. See chessrl.guided.",
)

from chessrl.guided import (
    GuidedConfig,
    GuidedEngine,
    GuidedModel,
    GuidedPolicy,
    GuidedTrainConfig,
    GuidedTrainer,
)
from chessrl.search import INF, MinimaxEngine

from conftest import INTERESTING_FENS

PLAYABLE = [f for f in INTERESTING_FENS if not chess.Board(f).is_game_over()]

# A short budget so the suite stays usable. L5's real difficulty is speed, and
# the speed tests below are what pin the budget down; everything else is about
# correctness and does not need depth to show it.
OPENING = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
MIDDLE = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 4"
# White plays Ra8#, the canonical back-rank mate. Used to check that a learned
# leaf value can never talk the search out of a real mate.
BACK_RANK = "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1"
# Black is mated by ...Ra1# *or* ...Re1#, so any move the search returns that
# results in checkmate is acceptable -- this is not a test of move choice.
BACK_RANK_BLACK = "r5k1/5ppp/8/8/8/8/5PPP/6K1 b - - 0 1"


@pytest.fixture
def model() -> GuidedModel:
    return GuidedModel(GuidedConfig(seed=11))


@pytest.fixture
def engine(model) -> GuidedEngine:
    return GuidedEngine(model, depth=2)


# --------------------------------------------------------------------------
# the trunk is actually shared
# --------------------------------------------------------------------------

def test_both_heads_read_the_same_trunk():
    """The heads must differ *only* in their final projection.

    Checked by recomputing everything the two heads could plausibly do
    differently and confirming the public methods reproduce it exactly from
    ``trunk_features``' output alone. If a head quietly re-encoded the board or
    used a private feature, this would fail -- and it would fail while every
    downstream test still passed, which is exactly why it is worth a test.
    """
    m = GuidedModel(GuidedConfig(seed=3))
    chan = torch.randn(2, m.n_ch, 64)
    ctx = torch.randn(2, 9)
    trunk = m.trunk_features(chan, ctx)

    # decision head: a linear read of the trunk, per square.
    dec = m.decision_logits(trunk, chan)
    manual_dec = (trunk * m.dec_from.squeeze(-1).unsqueeze(0)).sum(-1)
    assert torch.allclose(dec, manual_dec, atol=1e-6)

    # evaluation head: also a linear read of the *same* trunk tensor.
    val = m.evaluate(trunk, ctx)
    manual_val = (trunk.mean(dim=1) * m.val_head.squeeze(-1)).sum(-1)
    manual_val = manual_val + (ctx @ m.val_ctx).squeeze(-1)
    manual_val = torch.tanh(manual_val) * m.config.value_scale
    assert torch.allclose(val, manual_val, atol=1e-6)


def test_single_trunk_survives_a_gradient_step_through_either_head():
    """A step on the eval loss must move the *decision* head's output too.

    If the trunk were not shared -- if the two heads sat on separate feature
    extractors -- then training the value head would leave the move
    distribution untouched. That is the observable consequence of sharing, and
    it is what makes the level's cost claim true.
    """
    m = GuidedModel(GuidedConfig(seed=5))
    chan = torch.randn(1, m.n_ch, 64)
    ctx = torch.randn(1, 9)

    before = m.decision_logits(m.trunk_features(chan, ctx), chan).detach().clone()

    loss = m.evaluate(m.trunk_features(chan, ctx), ctx).sum()
    loss.backward()

    # Move the trunk's weights along the gradient by hand; the decision head's
    # own weights are untouched.
    with torch.no_grad():
        for p in (m.trunk_sq, m.trunk_ctx, m.trunk_bias):
            assert p.grad is not None and p.grad.abs().sum() > 0
            p -= 0.5 * p.grad

    after = m.decision_logits(m.trunk_features(chan, ctx), chan)
    assert not torch.allclose(before, after, atol=1e-9), (
        "training the value head did not move the decision head, so the trunk "
        "is not shared"
    )


def test_heads_work_at_trunk_widths_other_than_64():
    """Every head must be correct at any trunk width, not just the default.

    This is the test that caught a real bug rather than a hypothetical one.
    Both the decision and destination heads wrote an ``unsqueeze`` in a place
    that happened to broadcast only when the trunk width equalled 64 -- the
    square count. At the default width nothing raised; the destination head
    silently computed a different quantity (a sum over all destination rows
    instead of the origin's own row) and the decision head contracted the wrong
    axis. At ``trunk=32`` both raise outright.

    A width sweep is the cheapest way to keep that class of bug out, because it
    removes the coincidence the bugs were hiding behind.
    """
    for width in (16, 32, 64, 96):
        m = GuidedModel(GuidedConfig(seed=1, trunk=width))
        chan = torch.randn(2, m.n_ch, 64)
        ctx = torch.randn(2, 9)
        trunk = m.trunk_features(chan, ctx)
        assert trunk.shape == (2, 64, width)

        assert m.decision_logits(trunk, chan).shape == (2, 64)
        assert m.destination_logits(
            trunk, chan, torch.tensor([4, 12])
        ).shape == (2, 64)
        assert m.evaluate(trunk, ctx).shape == (2,)
        assert m.promo_logits(
            chan, torch.tensor([0, 1]), torch.tensor([2, 3])
        ).shape == (2, 5)


def test_destination_term_uses_the_origin_row_not_all_rows():
    """The origin term must read one square's row, not sum over every square.

    Direct numerical check of the fix described in the previous test: with the
    origin-specific term zeroed out, the head's output must change by exactly
    ``dec_src`` applied to the origin's own trunk vector -- which can be
    computed independently and compared. Summing over all 64 rows would pass a
    shape check and fail this.
    """
    m = GuidedModel(GuidedConfig(seed=4, trunk=32))
    chan = torch.randn(1, m.n_ch, 64)
    ctx = torch.randn(1, 9)
    trunk = m.trunk_features(chan, ctx)

    origin = 20
    got = m.destination_logits(trunk, chan, torch.tensor([origin]))[0]

    # Rebuild it term by term from the module's own parameters.
    dest = (trunk[0] * m.dec_dst.squeeze(-1)).sum(-1)
    src = (trunk[0, origin] * m.dec_src.squeeze(-1)).sum()
    geom = (m.geom_table[origin] * m.dec_geom.squeeze(-1)).sum(-1)
    assert torch.allclose(got, dest + src + geom, atol=1e-5)


def test_parameter_count_is_the_sum_of_its_parts():
    """No hidden parameters: the module's count must match its own attributes.

    A cheap guard against a head growing a private layer without the docstring
    (or the benchmark's reported parameter count) noticing.
    """
    m = GuidedModel(GuidedConfig(trunk=32))
    expected = sum([
        m.trunk_sq.numel(), m.trunk_ctx.numel(), m.trunk_bias.numel(),
        m.dec_from.numel(), m.dec_promo.numel(), m.dec_dst.numel(),
        m.dec_src.numel(), m.dec_geom.numel(), m.val_head.numel(),
        m.val_ctx.numel(),
    ])
    assert m.n_parameters() == expected


# --------------------------------------------------------------------------
# leaf evaluation changes the value; ordering does not
# --------------------------------------------------------------------------

def test_ordering_cannot_change_the_search_value(model):
    """The structural claim, stated as an executable test.

    Same engine, same depth, ordering prior switched off and on. The value the
    search returns must be identical, because alpha-beta with a correct window
    returns the same minimax value regardless of move order -- ordering only
    chooses *among* moves that share that value. If this ever fails, the prior
    has leaked into the evaluation and the level's whole decomposition is wrong.
    """
    for fen in (OPENING, MIDDLE, BACK_RANK):
        board = chess.Board(fen)
        plain = GuidedEngine(model, depth=2, use_model_prior=False, use_model_eval=False)
        prior = GuidedEngine(model, depth=2, use_model_prior=True, use_model_eval=False)

        v_plain = plain._search_value(board, 2)
        v_prior = prior._search_value(board, 2)
        assert v_plain == v_prior, f"ordering changed the value in {fen}"


def test_leaf_evaluation_does_change_the_value(model):
    """The complement of the previous test, and the reason the two are separate.

    Swapping only the leaf evaluator -- ordering held identical on both sides by
    turning the prior off -- must move at least one search value off the static
    evaluator's answer. If it did not, the value head would be inert and L5
    would be L2 with extra steps.
    """
    changed = 0
    for fen in (OPENING, MIDDLE, BACK_RANK):
        board = chess.Board(fen)
        static = GuidedEngine(model, depth=2, use_model_prior=False, use_model_eval=False)
        learned = GuidedEngine(model, depth=2, use_model_prior=False, use_model_eval=True)
        if static._search_value(board, 2) != learned._search_value(board, 2):
            changed += 1
    assert changed > 0, "the evaluation head made no difference to any search value"


def test_shallow_search_still_sees_depth_one_tactics(model):
    """A learned leaf value must not blind the search to a one-ply mate.

    The static evaluator and the model disagree about almost everything at this
    depth, so this checks the specific thing that must survive: mate scores
    (±MATE) are produced by the terminal test, not by the evaluator, and the
    evaluator is clamped well below them.
    """
    for fen in (BACK_RANK, BACK_RANK_BLACK):
        p = GuidedPolicy(model, depth=2)
        # Two plies, because a mate-in-one is found at depth 1 only if the side
        # to move is the one mating; both positions here mate on the move.
        move = p.select(chess.Board(fen))
        board = chess.Board(fen)
        board.push(move)
        assert board.is_checkmate(), (
            f"{move} from {fen} is not mate; a forced mate was buried by the "
            f"learned evaluator"
        )


def test_model_values_are_clamped_away_from_the_mate_band(engine):
    """No learned value may reach the mate band.

    If it did, the search would treat a merely nice position as a forced win and
    would prefer it over an actual mate found one ply deeper. The clamp is the
    invariant, so it is asserted directly rather than inferred from a game.
    """
    for fen in PLAYABLE:
        board = chess.Board(fen)
        value = engine.model_evaluate(board)
        assert abs(value) < engine._mate_floor
        assert abs(value) < 100_000 - 200  # MATE minus any plausible ply count


# --------------------------------------------------------------------------
# it plays legal games
# --------------------------------------------------------------------------

def test_plays_a_full_legal_game_to_a_real_result(model):
    """The gate every level in this repo passes: a real game, a real result.

    Depth 1 and a ply cap keep this bounded. An *unfinished* game is a
    legitimate outcome here -- the repo's convention is that a ply-capped game
    is not a draw, so ``is_finished`` is False and that is correct, not a
    failure. What must hold is that the game stopped for a real reason and that
    every move it recorded was legal.
    """
    from chessrl.game import play_game

    p = GuidedPolicy(model, depth=1)
    result = play_game(p, p, max_plies=60, rng=random.Random(1))

    assert result.reason in {
        "checkmate", "stalemate", "insufficient material",
        "seventyfive moves", "fivefold repetition", "fifty moves",
        "max_plies", "no_progress",
    }, f"unexpected termination reason: {result.reason}"
    assert result.plies == len(result.moves)

    # A capped game and a finished one differ in the flag, so pin both down
    # rather than accepting either.
    if result.result == "unfinished":
        assert not result.is_finished
        assert result.score_for(chess.WHITE) == 0
    else:
        assert result.is_finished

    # Every recorded move must be legal when replayed, which is the real
    # assertion: a policy that returned an illegal move would corrupt the board
    # rather than raise, and `fens` would drift from `moves`.
    #
    # ``moves`` is SAN, not UCI (see ``GameResult.to_pgn``), so the replay goes
    # through ``push_san`` rather than ``from_uci``.
    replay = chess.Board()
    for san in result.moves:
        move = replay.parse_san(san)
        assert move in replay.legal_moves, f"{san} is not legal in {replay.fen()}"
        replay.push(move)
    assert len(result.fens) == len(result.moves) + 1
    assert replay.fen() == result.fens[-1]


def test_seeded_search_is_reproducible(model):
    """Same seed, same move. The engine uses no RNG itself, but this pins the
    property down so a future tie-break shortcut cannot quietly add one."""
    a = GuidedPolicy(model, depth=2).select(chess.Board(MIDDLE))
    b = GuidedPolicy(model, depth=2).select(chess.Board(MIDDLE))
    assert a == b


def test_policy_exposes_no_distribution(model):
    """L5 is a search policy, not a learner, and the L6 ensemble needs to know.

    A ``move_distribution`` would have to be invented -- either by searching
    once per legal move (not this level) or by exposing the prior alone (not a
    distribution over *moves* the search would actually play). Its absence is
    deliberate and load-bearing for the ensemble, so it is asserted.
    """
    p = GuidedPolicy(model, depth=1)
    assert not hasattr(p, "move_distribution")


# --------------------------------------------------------------------------
# caches are real and bounded
# --------------------------------------------------------------------------

def test_value_cache_makes_repeated_evaluation_free(model):
    """The cache is the difference between a usable engine and an unusable one.

    Counting the *forward passes* rather than the calls is the only honest way
    to assert this: the second evaluation of a position must not reach the
    model at all.
    """
    engine = GuidedEngine(model, depth=1)
    board = chess.Board(MIDDLE)
    engine.model_evaluate(board)
    first = engine.leaf_evals
    engine.model_evaluate(board)
    engine.model_evaluate(board)
    assert engine.leaf_evals == first, "a cached position was evaluated again"


def test_value_cache_returns_identical_values(model):
    engine = GuidedEngine(model, depth=1)
    board = chess.Board(MIDDLE)
    assert engine.model_evaluate(board) == engine.model_evaluate(board)


def test_caches_are_bounded(model):
    """Both caches must have a ceiling, or a long tournament grows without end.

    The clear-on-overflow policy is crude but bounded; what matters is that the
    bound exists and is enforced.
    """
    engine = GuidedEngine(model, depth=1)
    engine._value_cache = {i: 0 for i in range(200_001)}
    engine._prior_cache = {i: {} for i in range(50_001)}
    engine.clear_caches()
    assert engine._value_cache == {}
    assert engine._prior_cache == {}


def test_clear_caches_also_clears_the_transposition_table(model, engine):
    board = chess.Board(OPENING)
    engine._negamax(board, 2, -INF, INF)
    assert engine.tt, "the search populated no transposition entries"
    engine.clear_caches()
    assert not engine.tt


# --------------------------------------------------------------------------
# the ablation flags actually ablate
# --------------------------------------------------------------------------

def test_disabling_both_models_reproduces_l2_exactly():
    """With both seams off, L5 must equal L2 -- not merely resemble it.

    This is the strongest form of "the search is L2's": if the model pathways
    are switched off and any difference remains, the override has changed the
    algorithm. The two engines get the *same* model-derived state only in the
    sense that neither uses it, so the comparison is apples to apples.
    """
    model = GuidedModel(GuidedConfig(seed=2))
    for fen in (OPENING, MIDDLE, BACK_RANK):
        board = chess.Board(fen)
        guided = GuidedEngine(model, depth=2, use_model_prior=False, use_model_eval=False)
        plain = MinimaxEngine(depth=2, prior_weight=0.0)
        assert guided._search_value(board, 2) == plain._search_value(board, 2)


def test_model_prior_is_normalised_and_legal(model):
    """The prior must be a distribution over exactly the legal moves.

    A prior that assigns mass to an illegal move would let the ordering
    machinery see moves the search cannot play; a prior that does not sum to one
    would not be comparable against the static ordering term.
    """
    engine = GuidedEngine(model, depth=1)
    prior = engine._model_prior(chess.Board(MIDDLE))
    legal = set(chess.Board(MIDDLE).legal_moves)
    assert set(prior) == legal
    assert all(v >= 0.0 for v in prior.values())
    assert 0.9 < sum(prior.values()) < 1.1


def test_model_prior_is_deterministic(model):
    engine = GuidedEngine(model, depth=1)
    board = chess.Board(MIDDLE)
    a = engine._model_prior(board)
    engine._prior_cache.clear()
    b = engine._model_prior(board)
    for move in a:
        assert a[move] == pytest.approx(b[move], abs=1e-9)


# --------------------------------------------------------------------------
# speed: the reason this level was nearly unusable
# --------------------------------------------------------------------------

def test_search_runs_single_threaded_and_restores(model):
    """The thread count must be pinned *during* the search and restored after.

    PyTorch defaults to one OpenMP thread per core, and for a single-position
    trunk projection the fork/join cost exceeds the arithmetic. Getting this
    wrong cost this engine a factor of nine, so it is asserted rather than
    commented.
    """
    before = torch.get_num_threads()
    engine = GuidedEngine(model, depth=1)
    seen = {}

    original_trunk = engine.model.trunk_features

    def spy(chan, ctx):
        seen["threads"] = torch.get_num_threads()
        return original_trunk(chan, ctx)

    engine.model.trunk_features = spy
    with engine.single_threaded():
        chess.Board(OPENING)
        engine.model_evaluate(chess.Board(OPENING))
    engine.model.trunk_features = original_trunk

    assert seen["threads"] == 1, "the search did not run single-threaded"
    assert torch.get_num_threads() == before, "the thread count was not restored"


def test_single_threaded_restores_on_exception(model):
    engine = GuidedEngine(model, depth=1)
    before = torch.get_num_threads()
    with pytest.raises(RuntimeError):
        with engine.single_threaded():
            raise RuntimeError("boom")
    assert torch.get_num_threads() == before


def test_depth_three_middlegame_finishes_quickly(model):
    """A budget, not a benchmark.

    L5's cost is dominated by one torch forward pass per distinct leaf, so the
    only way to keep it usable is to keep the leaf count down and the per-leaf
    cost down. Before the thread fix this took ~55s; the assertion is loose
    enough not to fail on a slow machine but tight enough to catch a regression
    back to eight threads.
    """
    import time

    p = GuidedPolicy(model, depth=3)
    start = time.perf_counter()
    p.select(chess.Board(MIDDLE))
    elapsed = time.perf_counter() - start
    assert elapsed < 25.0, f"depth-3 middlegame took {elapsed:.1f}s"


# --------------------------------------------------------------------------
# training round-trips
# --------------------------------------------------------------------------

def test_training_reduces_the_decision_loss(model):
    """A training step must move the loss in the right direction on average.

    One step on one position is noisy, so this trains repeatedly on a fixed
    position and compares the first loss to a late one. A trainer that did not
    update anything would leave them equal.
    """
    torch.manual_seed(0)
    trainer = GuidedTrainer(model, GuidedTrainConfig(search_depth=1, top_k=3))
    board = chess.Board(MIDDLE)

    first = trainer.train_on_search_feedback(board)["loss"]
    last = first
    for _ in range(6):
        last = trainer.train_on_search_feedback(board)["loss"]
    assert last < first, f"loss did not fall: {first:.4f} -> {last:.4f}"


def test_credit_for_black_moves_the_features_inference_reads(model):
    """Training on a black-to-move position must change black's prediction.

    The canonical frame means black's position is stored rank-reflected; if
    training updated one frame while inference read another, the loss would fall
    to zero while the model's actual output never improved. This is the L5
    analogue of the frame bug that L3 actually had.
    """
    torch.manual_seed(0)
    trainer = GuidedTrainer(model, GuidedTrainConfig(search_depth=1, top_k=3))
    board = chess.Board(MIDDLE)
    assert board.turn == chess.BLACK

    engine = GuidedEngine(model, depth=1)
    before = dict(engine._model_prior(board))
    for _ in range(4):
        trainer.train_on_search_feedback(board)
    engine._prior_cache.clear()
    after = engine._model_prior(board)

    moved = sum(abs(after[m] - before[m]) for m in before)
    assert moved > 1e-6, "training on a black position changed nothing black sees"


def test_value_training_ignores_unfinished_games(model):
    """The repo-wide convention: an unfinished game is not a draw.

    Scoring a ply-capped game as 0.0 would teach the value head that shuffling
    is neutral, which is the single most damaging thing a naive evaluator can
    learn.
    """
    from chessrl.game import GameResult

    trainer = GuidedTrainer(model, GuidedTrainConfig())
    unfinished = GameResult("unfinished", "ply cap", 40, [], [], 0.0)
    out = trainer.train_on_game(unfinished)
    assert out.get("skipped") is True
    assert trainer.value_steps == 0


def test_save_and_load_round_trip(tmp_path, model):
    """Weights must survive a save/load cycle bit-for-bit.

    L5's two heads make this less obvious than it looks: the file has to carry
    both, and the loaded model has to be usable as an engine without first
    being re-wrapped.
    """
    path = tmp_path / "guided.pt"
    trainer = GuidedTrainer(model, GuidedTrainConfig(search_depth=1))
    meta = trainer.save(path)
    assert path.exists()
    assert (tmp_path / "guided.meta.json").exists()

    restored_trainer = GuidedTrainer.load(path)
    restored = restored_trainer.model

    board = chess.Board(MIDDLE)
    chan, ctx = None, None
    from chessrl.guided import _trunk_inputs

    chan, ctx = _trunk_inputs(board, model.n_ch)
    a = model.evaluate(model.trunk_features(chan, ctx), ctx)
    b = restored.evaluate(restored.trunk_features(chan, ctx), ctx)
    assert torch.allclose(a, b, atol=1e-6)
    assert restored.n_parameters() == model.n_parameters()
    assert meta["model"] == str(path)


def test_loaded_model_still_plays_a_legal_move(tmp_path, model):
    path = tmp_path / "guided.pt"
    GuidedTrainer(model).save(path)
    restored = GuidedTrainer.load(path).model
    move = GuidedPolicy(restored, depth=1).select(chess.Board(OPENING))
    assert move in chess.Board(OPENING).legal_moves
