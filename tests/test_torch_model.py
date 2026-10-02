"""L4: the torch model, and the parity test that justifies rewriting L3.

The centrepiece is :func:`test_logits_match_l3_exactly`. Everything else here is
ordinary hygiene; that one test is the reason this level exists. L3 is a
hand-rolled perceptron with a hand-derived update rule. L4 is the same model in
torch. If their logits agree to float32 tolerance when loaded with the same
weights, then L3's architecture, feature order, canonical convention and weight
layout are all confirmed at once -- and L3's gradient derivation can be trusted
enough to build on.

Without that test, "L3 works" would only mean "L3 is self-consistent", which is
a much weaker claim and would not have caught the frame bug that L3 actually
had (see `tests/test_perceptron.py`).
"""

from __future__ import annotations

import numpy as np
import chess
import pytest

torch = pytest.importorskip(
    "torch",
    reason="L4 needs PyTorch, an optional extra. See chessrl.torch_model.",
)

from chessrl import masks as M
from chessrl.perceptron import L3Policy, PerceptronConfig, PerceptronScorer
from chessrl.torch_model import (
    TorchConfig,
    TorchPolicy,
    TorchScorer,
    TorchTrainer,
    TorchTrainConfig,
    canonical_squares,
    geometry_table,
)

from conftest import INTERESTING_FENS

PLAYABLE = [f for f in INTERESTING_FENS if not chess.Board(f).is_game_over()]


def reflect_square(sq: int) -> int:
    return chess.square(chess.square_file(sq), 7 - chess.square_rank(sq))


def reflect_move(move: chess.Move) -> chess.Move:
    return chess.Move(
        reflect_square(move.from_square),
        reflect_square(move.to_square),
        promotion=move.promotion,
    )


@pytest.fixture
def l3() -> L3Policy:
    return L3Policy(seed=7)


@pytest.fixture
def matched(l3):
    """An L4 model carrying L3's exact weights."""
    model = TorchScorer(TorchConfig(), seed=7)
    model.load_from_perceptron(l3.perceptron)
    return TorchPolicy(model)


# --------------------------------------------------------------------------
# the parity test
# --------------------------------------------------------------------------

def test_parameter_counts_match_l3(l3):
    """The two models must have the same number of parameters, or parity is
    not even the right question.
    """
    model = TorchScorer(TorchConfig())
    assert model.n_parameters == l3.perceptron.n_parameters


def test_parameter_shapes_match_l3(l3):
    """Every named weight must have L3's shape, not just L3's total count."""
    model = TorchScorer(TorchConfig())
    mine = model.parameter_dict()
    theirs = l3.perceptron.parameters()
    assert set(mine) == set(theirs)
    for name in theirs:
        assert mine[name].shape == theirs[name].shape, name


def test_logits_match_l3_exactly(matched, l3):
    """Load L3's weights and require every logit to agree.

    The tolerance is float32 noise, not a fudge factor: the two implementations
    perform the same sums in the same order, so agreement should be at the level
    of the last float32 bit.
    """
    worst_from = worst_to = worst_promo = 0.0
    for fen in PLAYABLE:
        board = chess.Board(fen)
        l3.scorer.set_position(board)
        matched.adapter.set_position(board)

        worst_from = max(
            worst_from,
            np.abs(l3.scorer.from_logits(board) - matched.scorer.from_logits(board)).max(),
        )
        for move in board.legal_moves:
            worst_to = max(
                worst_to,
                np.abs(
                    l3.scorer.to_logits(move.from_square)
                    - matched.scorer.to_logits(move.from_square)
                ).max(),
            )
            worst_promo = max(
                worst_promo,
                np.abs(
                    l3.scorer.promo_logits(move.from_square, move.to_square)
                    - matched.scorer.promo_logits(move.from_square, move.to_square)
                ).max(),
            )

    assert worst_from < 1e-6, worst_from
    assert worst_to < 1e-6, worst_to
    assert worst_promo < 1e-6, worst_promo


def test_move_distributions_match_l3_exactly(matched, l3):
    """The end-to-end version: same weights, same probabilities."""
    for fen in PLAYABLE:
        board = chess.Board(fen)
        np.testing.assert_allclose(
            matched.move_distribution(board),
            l3.move_distribution(board),
            atol=1e-6,
            err_msg=f"distribution diverged on {fen}",
        )


def test_greedy_choices_match_l3(matched, l3):
    """Same weights must mean the same chosen move, not merely close logits."""
    matched.greedy = l3.greedy = True
    for fen in PLAYABLE:
        board = chess.Board(fen)
        assert matched.select(board) == l3.select(board), fen


def test_weights_survive_a_round_trip_through_torch(l3):
    """L3 -> L4 -> L3 must be lossless.

    If the transfer loses precision elsewhere, the parity test could pass while
    the reverse direction silently corrupts a model.
    """
    model = TorchScorer(TorchConfig())
    model.load_from_perceptron(l3.perceptron)
    back = model.to_perceptron()
    for name, values in l3.perceptron.parameters().items():
        np.testing.assert_array_equal(
            back.parameters()[name], values, err_msg=name
        )


def test_parity_holds_without_the_context_pathway():
    """The ReLU branch is the most intricate part; check it can be disabled."""
    l3 = L3Policy(scorer=PerceptronScorer(
        PerceptronConfig(use_context=False, hidden=0), seed=3
    ))
    model = TorchScorer(TorchConfig(use_context=False), seed=3)
    model.load_from_perceptron(l3.perceptron)
    policy = TorchPolicy(model)
    assert model.h == 0
    for fen in PLAYABLE:
        board = chess.Board(fen)
        np.testing.assert_allclose(
            policy.move_distribution(board),
            l3.move_distribution(board),
            atol=1e-6,
        )


def test_parity_holds_for_the_smallest_hidden_layer():
    """A 1-unit hidden layer is the degenerate case most likely to hide an
    off-by-one in the gate."""
    l3 = L3Policy(scorer=PerceptronScorer(PerceptronConfig(hidden=1), seed=5))
    model = TorchScorer(TorchConfig(hidden=1), seed=5)
    model.load_from_perceptron(l3.perceptron)
    policy = TorchPolicy(model)
    for fen in PLAYABLE:
        board = chess.Board(fen)
        np.testing.assert_allclose(
            policy.move_distribution(board),
            l3.move_distribution(board),
            atol=1e-6,
        )


# --------------------------------------------------------------------------
# protocol conformance
# --------------------------------------------------------------------------

def test_heads_have_the_shapes_the_protocol_promises(matched):
    board = chess.Board()
    matched.adapter.set_position(board)
    assert matched.scorer.from_logits(board).shape == (64,)
    assert matched.scorer.to_logits(chess.E2).shape == (64,)
    assert matched.scorer.promo_logits(chess.A7, chess.A8).shape == (M.NUM_PROMO,)


def test_from_logits_is_not_a_square_against_square_matrix(matched):
    """Same trap as L3: the batched matmul produces a ``(64, 64)`` table."""
    assert matched.scorer.from_logits(chess.Board()).ndim == 1


def test_heads_require_a_bound_position():
    policy = TorchPolicy(TorchScorer())
    with pytest.raises(RuntimeError, match="set_position"):
        policy.scorer.to_logits(chess.E2)


def test_distribution_is_a_valid_probability_vector(matched):
    board = chess.Board()
    dist = matched.move_distribution(board)
    assert dist.shape == (M.ACTION_SPACE,)
    assert np.isclose(dist.sum(), 1.0, atol=1e-6)
    assert (dist >= 0).all()


def test_distribution_has_one_entry_per_legal_move(matched):
    board = chess.Board()
    dist = matched.move_distribution(board)
    nonzero = set(np.nonzero(dist)[0].tolist())
    assert nonzero == {M.move_to_index(m) for m in board.legal_moves}


def test_distribution_is_zero_when_the_game_is_over(matched):
    board = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
    assert board.is_checkmate()
    assert matched.move_distribution(board).sum() == 0.0


def test_selects_a_legal_move_for_both_colours(matched):
    for fen in PLAYABLE:
        board = chess.Board(fen)
        assert matched.select(board) in board.legal_moves


def test_policy_is_usable_as_a_move_policy_in_a_game(matched):
    """L4 must be a first-class member of the ladder, not a curiosity.

    Only that it *plays*: an untrained model with random weights shuffles and
    the game runs to the ply cap, exactly as untrained L3 does. Asserting a
    decisive result here would be asserting something false.
    """
    import random

    from chessrl.game import play_game
    from chessrl.policy import BlindScorer, FactoredSoftmaxPolicy

    opponent = FactoredSoftmaxPolicy(BlindScorer(), seed=1)
    result = play_game(matched, opponent, max_plies=60, no_progress_limit=30,
                       rng=random.Random(0))
    # A ply-capped game is *unfinished* by design -- that is the L0 convention
    # that stops shuffling from being scored as a draw. So assert a valid
    # termination reason, not a finished game.
    assert result.reason in ("max_plies", "no_progress", "checkmate",
                             "stalemate", "fivefold_repetition",
                             "seventyfive_moves", "insufficient_material")
    assert result.plies > 0
    # Every recorded position must be a real board, and one more FEN than moves.
    assert len(result.fens) == len(result.moves) + 1
    for fen in result.fens:
        assert chess.Board(fen).is_valid()


# --------------------------------------------------------------------------
# symmetry: the same property L3 needed, now via the adapter's scatter
# --------------------------------------------------------------------------

def test_colour_symmetry_holds(matched):
    """The canonical frame must survive the torch round trip."""
    from conftest import colour_mirror

    worst = 0.0
    for fen in PLAYABLE:
        board = chess.Board(fen)
        mirror = colour_mirror(fen)
        dist = matched.move_distribution(board)
        dist_m = matched.move_distribution(mirror)
        for move in board.legal_moves:
            mirrored = reflect_move(move)
            assert mirrored in mirror.legal_moves
            worst = max(
                worst,
                abs(dist[M.move_to_index(move)] - dist_m[M.move_to_index(mirrored)]),
            )
    assert worst < 1e-6, worst


def test_canonical_square_map_matches_l3s():
    """L4 must not have its own copy of the frame map."""
    for fen in PLAYABLE:
        board = chess.Board(fen)
        np.testing.assert_array_equal(
            canonical_squares(board),
            PerceptronScorer._canonical_squares(board),
        )


def test_geometry_table_is_built_from_l3s_function():
    """A reimplemented geometry table would make the parity test vacuous."""
    table = geometry_table()
    assert table.shape == (64, 64, 9)
    scorer = PerceptronScorer(PerceptronConfig(use_context=False))
    for from_sq in (chess.A1, chess.E4, chess.H8):
        for to_sq in (chess.A1, chess.D5, chess.H1):
            np.testing.assert_allclose(
                table[from_sq, to_sq],
                scorer._geom_features(from_sq, to_sq),
                atol=1e-6,
            )


# --------------------------------------------------------------------------
# batching: the reason to have torch at all
# --------------------------------------------------------------------------

def test_origin_head_scores_a_batch_in_one_call(matched):
    """The heads must accept a batch; that is the point of the rewrite."""
    boards = [chess.Board(f) for f in PLAYABLE]
    planes = []
    for board in boards:
        matched.adapter.set_position(board)
        planes.append(matched.adapter._chan[0])
    batch = torch.stack(planes)                      # (B, n_ch, 64)
    out = matched.model.from_logits(batch, None)
    assert out.shape == (len(boards), 64)


def test_batched_origin_logits_match_the_single_position_path(matched):
    """Batching must not change the numbers."""
    for fen in PLAYABLE:
        board = chess.Board(fen)
        matched.adapter.set_position(board)
        single = matched.model.from_logits(
            matched.adapter._chan, matched.adapter._ctx
        )[0]
        batched = matched.model.from_logits(
            torch.stack([matched.adapter._chan[0]]), None
        )[0]
        np.testing.assert_allclose(
            single.detach().numpy(), batched.detach().numpy(), atol=1e-6
        )


def test_destination_head_scores_a_batch_of_origins(matched):
    board = chess.Board()
    matched.adapter.set_position(board)
    squares = [m.from_square for m in board.legal_moves]
    idx = torch.tensor(squares, dtype=torch.long)
    out = matched.model.to_logits(matched.adapter._chan, idx)
    assert out.shape == (len(squares), 64)


# --------------------------------------------------------------------------
# autograd
# --------------------------------------------------------------------------

def test_gradients_flow_to_every_parameter():
    """A parameter that never receives a gradient is a parameter that cannot
    learn, and it would look like a slow train rather than a bug."""
    model = TorchScorer(TorchConfig(), seed=1)
    board = chess.Board()
    from chessrl.encode import board_context_features, encode

    tensor = encode(board, canonicalise=True)
    chan = torch.as_tensor(
        tensor[:model.n_ch].reshape(model.n_ch, 64), dtype=torch.float32
    ).unsqueeze(0)
    ctx = torch.as_tensor(
        board_context_features(board), dtype=torch.float32
    ).unsqueeze(0)

    loss = model.from_logits(chan, ctx).pow(2).sum()
    loss = loss + model.to_logits(
        chan, torch.tensor([chess.E2], dtype=torch.long)
    ).pow(2).sum()
    # Touch the promotion head too: it is only reachable through its own head,
    # so a loss that never calls ``promo_logits`` legitimately leaves it flat.
    loss = loss + model.promo_logits(
        chan,
        torch.tensor([chess.A7], dtype=torch.long),
        torch.tensor([chess.A8], dtype=torch.long),
    ).pow(2).sum()
    loss.backward()

    missing = [
        name for name, p in model.named_parameters()
        if p.requires_grad and (p.grad is None or p.grad.abs().sum() == 0)
    ]
    for head in ("W_from", "W_to_dst", "W_to_src", "W_geom", "promo"):
        assert head not in missing, (head, missing)


def test_context_pathway_receives_gradients():
    model = TorchScorer(TorchConfig(), seed=1)
    board = chess.Board()
    from chessrl.encode import board_context_features, encode

    tensor = encode(board, canonicalise=True)
    chan = torch.as_tensor(
        tensor[:model.n_ch].reshape(model.n_ch, 64), dtype=torch.float32
    ).unsqueeze(0)
    ctx = torch.as_tensor(
        board_context_features(board), dtype=torch.float32
    ).unsqueeze(0)
    model.from_logits(chan, ctx).pow(2).sum().backward()
    # A dead ReLU unit would give an all-zero W_ctx row; at least one must fire.
    assert model.W_ctx.grad.abs().sum() > 0
    assert model.W_hid.grad.abs().sum() > 0


def test_training_moves_the_parameters():
    trainer = TorchTrainer(TorchTrainConfig(seed=2, search_depth=2, lr=0.05))
    before = {k: v.copy() for k, v in trainer.policy.model.parameter_dict().items()}
    board = chess.Board("k2q4/8/8/8/8/8/8/3R3K w - - 0 1")
    for _ in range(3):
        trainer.train_on_search_feedback(board, depth=2)
    after = trainer.policy.model.parameter_dict()
    changed = sum(
        not np.array_equal(before[k], after[k]) for k in before
    )
    assert changed >= 3


def test_optimiser_step_reduces_the_loss_it_is_given():
    """Sanity on the whole training path: the loss must go down.

    If the sign is wrong anywhere -- in the target, the log, or the optimiser --
    the loss will increase and this catches it immediately.
    """
    trainer = TorchTrainer(TorchTrainConfig(seed=3, search_depth=2, lr=0.05))
    board = chess.Board("k2q4/8/8/8/8/8/8/3R3K w - - 0 1")
    losses = []
    for _ in range(12):
        out = trainer.train_on_search_feedback(board, depth=2)
        losses.append(out["loss"])
    assert losses[0] > losses[-1], losses


def test_search_feedback_raises_the_searched_best_move():
    """Autograd version of the L3 claim, on the same position."""
    trainer = TorchTrainer(TorchTrainConfig(seed=6, search_depth=3, lr=0.05))
    board = chess.Board("k2q4/8/8/8/8/8/8/3R3K w - - 0 1")
    grab = chess.Move.from_uci("d1d8")
    idx = M.move_to_index(grab)

    # Confirm the search really likes this move before training on it.
    targets = trainer.policy.model  # silence unused warnings in older linters
    from chessrl.search import MinimaxEngine

    engine = MinimaxEngine(depth=3, use_tt=True)
    best = None
    best_value = -10**9
    for move in board.legal_moves:
        board.push(move)
        value = -engine._negamax(board, 2, -10**9, 10**9)
        board.pop()
        if value > best_value:
            best_value, best = value, move
    assert best == grab

    before = trainer.policy.move_distribution(board)[idx]
    for _ in range(15):
        trainer.train_on_search_feedback(board, depth=3)
    after = trainer.policy.move_distribution(board)[idx]
    assert after > before, (before, after)


def test_train_on_search_feedback_reports_what_it_did():
    trainer = TorchTrainer(TorchTrainConfig(seed=6, search_depth=2))
    out = trainer.train_on_search_feedback(chess.Board(), depth=2)
    assert out["applied"] is True
    assert out["k"] > 0
    assert "loss" in out


def test_train_on_search_feedback_on_a_finished_game_is_a_no_op():
    trainer = TorchTrainer(TorchTrainConfig(seed=6))
    board = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
    out = trainer.train_on_search_feedback(board, depth=2)
    assert out["applied"] is False
    assert trainer.search_updates == 0


def test_a_trained_model_prefers_the_capture_it_learned():
    trainer = TorchTrainer(TorchTrainConfig(seed=8, search_depth=3, lr=0.05))
    board = chess.Board("k2q4/8/8/8/8/8/8/3R3K w - - 0 1")
    grab = chess.Move.from_uci("d1d8")
    for _ in range(30):
        trainer.train_on_search_feedback(board, depth=3)
    trainer.policy.greedy = True
    assert trainer.policy.select(board) == grab


# --------------------------------------------------------------------------
# determinism and persistence
# --------------------------------------------------------------------------

def test_same_seed_gives_identical_weights():
    a = TorchScorer(TorchConfig(), seed=5).parameter_dict()
    b = TorchScorer(TorchConfig(), seed=5).parameter_dict()
    for name in a:
        np.testing.assert_array_equal(a[name], b[name], err_msg=name)


def test_different_seeds_give_different_weights():
    a = TorchScorer(TorchConfig(), seed=5).parameter_dict()
    b = TorchScorer(TorchConfig(), seed=6).parameter_dict()
    assert not np.array_equal(a["W_from"], b["W_from"])


def test_policy_seed_reaches_the_model():
    """The L3 lesson, asserted here too: a seed that only reaches the sampling
    RNG leaves every initialised model identical."""
    a = TorchPolicy(seed=5).model.parameter_dict()
    b = TorchPolicy(seed=6).model.parameter_dict()
    assert not np.array_equal(a["W_from"], b["W_from"])


def test_state_dict_round_trips(tmp_path):
    policy = TorchPolicy(TorchScorer(TorchConfig(), seed=17))
    board = chess.Board()
    before = policy.move_distribution(board)

    path = tmp_path / "l4.json"
    policy.save(path)
    restored = TorchPolicy.load(path)
    np.testing.assert_allclose(
        restored.move_distribution(board), before, atol=1e-6
    )


def test_loaded_policy_is_a_torch_model_not_a_blind_scorer(tmp_path):
    policy = TorchPolicy(TorchScorer(TorchConfig(), seed=4))
    path = tmp_path / "l4.json"
    policy.save(path)
    restored = TorchPolicy.load(path)
    assert isinstance(restored.model, TorchScorer)
    assert restored.model.n_parameters == policy.model.n_parameters


def test_saved_weights_reproduce_the_same_choice(tmp_path):
    policy = TorchPolicy(TorchScorer(TorchConfig(), seed=21))
    policy.greedy = True
    board = chess.Board("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1")
    chosen = policy.select(board)

    path = tmp_path / "l4.json"
    policy.save(path)
    restored = TorchPolicy.load(path)
    restored.greedy = True
    assert restored.select(board) == chosen


def test_trainer_save_writes_a_metadata_sidecar(tmp_path):
    import json

    trainer = TorchTrainer(TorchTrainConfig(seed=2))
    trainer.train_on_search_feedback(chess.Board(), depth=1)
    paths = trainer.save(tmp_path / "l4.json")
    meta = json.loads(open(paths["meta"], encoding="utf-8").read())
    assert meta["n_parameters"] == trainer.policy.model.n_parameters
    assert meta["search_updates"] > 0


# --------------------------------------------------------------------------
# the import guard
# --------------------------------------------------------------------------

def test_torch_is_importable_in_this_environment():
    """If this fails, every other test in the module was skipped."""
    assert torch.__version__
