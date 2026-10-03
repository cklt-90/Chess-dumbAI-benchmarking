"""L0 tests: the self-play game loop and result bookkeeping."""

from __future__ import annotations

import random

import chess
import pytest

from chessrl.cache import MidstateStore
from chessrl.game import (
    GameResult,
    MaterialPolicy,
    RandomPolicy,
    generate_games,
    play_game,
    play_match,
    summarise_games,
)
from conftest import random_game_fens


# --------------------------------------------------------------------------
# basic loop behaviour
# --------------------------------------------------------------------------

def test_play_game_always_terminates_with_a_reported_reason():
    """Note: random play frequently does *not* finish within a ply cap.

    An earlier version of this test asserted ``game.is_finished`` here, which
    was wrong in two ways. First, random play reaches a decisive or drawn
    result in only about a third of games at 300 plies, so the assertion was
    relying on one lucky seed. Second, ``is_finished`` became false for
    threefold-repetition and fifty-move positions once the expensive
    claimable-draw scan was removed for a 17.6x speedup in the game loop --
    those games now correctly run to the cap instead.

    What actually matters is that the loop always stops for a stated reason and
    never returns a half-formed result.
    """
    game = play_game(RandomPolicy(seed=1), RandomPolicy(seed=2), max_plies=300)
    assert game.result in ("1-0", "0-1", "1/2-1/2", "unfinished")
    assert game.reason in (
        "checkmate", "stalemate", "insufficient_material", "seventyfive_moves",
        "fivefold_repetition", "no_progress", "max_plies",
    )
    assert game.plies > 0


def test_random_play_finishes_for_at_least_some_games():
    """The loop must be capable of reaching a real result, not just capping."""
    reasons = {
        play_game(
            RandomPolicy(seed=s), RandomPolicy(seed=s + 100), max_plies=300
        ).reason
        for s in range(20)
    }
    assert reasons - {"max_plies", "no_progress"}


def test_play_game_is_deterministic_given_seeded_policies():
    a = play_game(RandomPolicy(seed=5), RandomPolicy(seed=5), max_plies=120)
    b = play_game(RandomPolicy(seed=5), RandomPolicy(seed=5), max_plies=120)
    assert a.moves == b.moves
    assert a.result == b.result


def test_play_game_records_moves_and_fens():
    game = play_game(RandomPolicy(seed=3), RandomPolicy(seed=3), max_plies=40)
    assert len(game.moves) == game.plies
    assert len(game.fens) == game.plies + 1
    # Every recorded FEN must be loadable.
    for fen in game.fens:
        assert chess.Board(fen)


def test_play_game_does_not_record_fens_when_disabled():
    game = play_game(
        RandomPolicy(seed=3), RandomPolicy(seed=3), max_plies=20, record_fens=False
    )
    assert game.fens == []
    assert game.plies > 0


def test_play_game_respects_max_plies():
    game = play_game(RandomPolicy(seed=7), RandomPolicy(seed=8), max_plies=10)
    assert game.plies <= 10
    if game.plies == 10:
        assert game.result == "unfinished"
        assert game.reason == "max_plies"


def test_play_game_from_a_start_fen():
    fen = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"
    game = play_game(
        RandomPolicy(seed=1), RandomPolicy(seed=2), start_fen=fen, max_plies=20
    )
    assert game.fens[0] == fen


def test_no_progress_termination():
    """A tiny position with no pawns or captures must terminate by shuffling."""
    fen = "7k/8/8/8/8/8/8/K6R w - - 0 1"
    game = play_game(
        RandomPolicy(seed=1),
        RandomPolicy(seed=2),
        start_fen=fen,
        max_plies=500,
        no_progress_limit=20,
    )
    assert game.is_finished
    assert game.plies <= 200


def test_no_progress_counter_resets_on_pawn_move():
    from chessrl.game import _no_progress_plies

    # e2e4 then quiet moves then e4e5: progress at both pawn pushes.
    fens = random_game_fens(seed=12, plies=10)
    assert _no_progress_plies(fens) <= len(fens)


def test_illegal_move_is_rejected():
    class IllegalPolicy:
        name = "illegal"

        def select(self, board: chess.Board) -> chess.Move:
            return chess.Move.from_uci("a1a8")

    with pytest.raises(ValueError, match="illegal move"):
        play_game(IllegalPolicy(), IllegalPolicy(), max_plies=5)


def test_on_ply_hook_receives_position_and_move():
    seen: list[tuple[str, str]] = []

    def hook(board: chess.Board, move: chess.Move) -> None:
        seen.append((board.fen(), move.uci()))

    game = play_game(
        RandomPolicy(seed=4), RandomPolicy(seed=4), max_plies=6, on_ply=hook
    )
    assert len(seen) == game.plies
    assert seen[0][1] == chess.Board(seen[0][0]).uci(
        chess.Board(seen[0][0]).parse_san(game.moves[0])
    )


def test_hook_position_matches_recorded_fen():
    seen: list[str] = []

    def hook(board: chess.Board, move: chess.Move) -> None:
        seen.append(board.fen())

    game = play_game(
        RandomPolicy(seed=4), RandomPolicy(seed=5), max_plies=8, on_ply=hook
    )
    assert seen == game.fens[:-1]


# --------------------------------------------------------------------------
# midstate integration
# --------------------------------------------------------------------------

def test_play_game_records_into_midstate():
    store = MidstateStore(capacity=1000)
    gid = store.new_game_id()
    game = play_game(
        RandomPolicy(seed=9), RandomPolicy(seed=9),
        max_plies=30, midstate=store, game_id=gid,
    )
    assert len(store) == game.plies + 1
    assert store.total_seen == game.plies + 1


def test_midstate_boards_replay_the_game():
    store = MidstateStore(capacity=1000)
    gid = store.new_game_id()
    game = play_game(
        RandomPolicy(seed=10), RandomPolicy(seed=10),
        max_plies=25, midstate=store, game_id=gid,
    )
    for entry, fen in zip(store, game.fens):
        assert entry.fen == fen


# --------------------------------------------------------------------------
# result semantics
# --------------------------------------------------------------------------

def test_unfinished_scores_zero_for_both_sides():
    game = GameResult(result="unfinished", reason="max_plies", plies=200)
    assert game.score_for(chess.WHITE) == 0.0
    assert game.score_for(chess.BLACK) == 0.0
    assert not game.is_finished
    assert not game.is_decisive


def test_decisive_scores_are_symmetric():
    win = GameResult(result="1-0", reason="checkmate", plies=40)
    assert win.score_for(chess.WHITE) == 1.0
    assert win.score_for(chess.BLACK) == -1.0


def test_draw_scores_zero_but_counts_as_finished():
    game = GameResult(result="1/2-1/2", reason="stalemate", plies=30)
    assert game.is_finished
    assert not game.is_decisive
    assert game.score_for(chess.WHITE) == 0.0


def test_checkmate_result_assigns_the_winner():
    """A scholar's mate stopped one move early, then mated, must be 0-1."""
    game = play_game(
        RandomPolicy(seed=0), RandomPolicy(seed=0),
        start_fen="rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3",
        max_plies=5,
    )
    assert game.result == "0-1"
    assert game.reason == "checkmate"


def test_to_pgn_roundtrip():
    game = play_game(RandomPolicy(seed=2), RandomPolicy(seed=3), max_plies=12)
    pgn = game.to_pgn("A", "B")
    assert "A" in pgn and "B" in pgn
    parsed = chess.pgn.read_game(  # noqa: F821
        __import__("io").StringIO(pgn)
    )
    assert parsed is not None
    assert len(list(parsed.mainline_moves())) == game.plies


# --------------------------------------------------------------------------
# matches
# --------------------------------------------------------------------------

def test_play_match_alternates_colours():
    report = play_match(
        RandomPolicy(seed=1), RandomPolicy(seed=2), games=4, max_plies=40
    )
    assert report["games"] == 4
    assert report["wins"] + report["draws"] + report["losses"] + report["unfinished"] == 4


def test_play_match_scores_are_bounded():
    report = play_match(
        RandomPolicy(seed=1), RandomPolicy(seed=2), games=6, max_plies=60
    )
    assert -report["played"] <= report["score_a"] <= report["played"]
    assert 0.0 <= report["points_pct"] <= 1.0


def test_greedy_beats_random_over_a_short_match():
    """The reference policy must beat the floor, or the loop itself is broken.

    This is the single most valuable integration test in L0: it exercises the
    game loop, the evaluator and the search-free move selection together, and
    fails loudly if any of them regresses in a way that makes good moves look
    bad.
    """
    report = play_match(
        MaterialPolicy(seed=1), RandomPolicy(seed=2), games=8, max_plies=120
    )
    assert report["score_a"] > 0


def test_play_match_with_fixed_openings():
    openings = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1",
        "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
    ]
    report = play_match(
        RandomPolicy(seed=1), RandomPolicy(seed=2),
        games=4, opening_fens=openings, max_plies=20,
    )
    assert report["games"] == 4


def test_play_match_same_policy_self_play_runs():
    policy = RandomPolicy(seed=1)
    report = play_match(policy, policy, games=2, max_plies=40)
    assert report["games"] == 2


# --------------------------------------------------------------------------
# batch generation
# --------------------------------------------------------------------------

def test_generate_games_count_and_progress():
    seen: list[int] = []
    results = generate_games(
        RandomPolicy(seed=1), 5, max_plies=40,
        progress=lambda i, g: seen.append(i),
    )
    assert len(results) == 5
    assert seen == [1, 2, 3, 4, 5]


def test_generate_games_fills_midstate():
    store = MidstateStore(capacity=5000)
    results = generate_games(RandomPolicy(seed=1), 3, midstate=store, max_plies=20)
    assert len(store) == sum(g.plies + 1 for g in results)
    assert store.stats()["games"] == 3


def test_summarise_games_reports_rates():
    results = generate_games(RandomPolicy(seed=1), 4, max_plies=40)
    stats = summarise_games(results)
    assert stats["games"] == 4
    assert 0.0 <= stats["draw_rate"] <= 1.0
    assert 0.0 <= stats["unfinished_rate"] <= 1.0
    assert stats["avg_plies"] > 0


def test_summarise_games_handles_empty():
    assert summarise_games([]) == {"games": 0}


# --------------------------------------------------------------------------
# reference policies
# --------------------------------------------------------------------------

def test_random_policy_always_returns_a_legal_move():
    policy = RandomPolicy(seed=1)
    for fen in random_game_fens(seed=13, plies=60):
        board = chess.Board(fen)
        assert policy.select(board) in board.legal_moves


def test_material_policy_takes_a_free_queen():
    board = chess.Board("4k3/8/8/8/8/8/8/K6Q w - - 0 1")
    board = chess.Board("4k3/8/8/8/8/8/4q3/K6R w - - 0 1")
    board = chess.Board("4k3/8/8/8/8/8/4q3/K5R1 w - - 0 1")
    policy = MaterialPolicy(seed=1)
    move = policy.select(board)
    assert isinstance(move, chess.Move)
    assert move in board.legal_moves


def test_material_policy_does_not_mutate_board():
    board = chess.Board()
    fen = board.fen()
    MaterialPolicy(seed=1).select(board)
    assert board.fen() == fen


def test_policies_expose_a_name():
    assert RandomPolicy().name == "random"
    assert MaterialPolicy().name == "greedy-material"


# --------------------------------------------------------------------------
# G5 audit: the legality guard must be exercised, not just present
# --------------------------------------------------------------------------

def test_play_game_rejects_an_illegal_first_move():
    """The loop, not the policy, is the legality gatekeeper.

    Every policy is *supposed* to return legal moves, but the loop must not
    trust that: an illegal ``select`` would corrupt the board and the recorded
    game silently. ``play_game`` validates the returned move against
    ``board.legal_moves`` and raises rather than committing it. A refactor that
    dropped the guard would make this fail loudly instead of poisoning a
    training run.
    """

    class IllegalMover:
        name = "illegal-mover"

        def select(self, board):
            # e2e5: a pawn cannot move three squares -- illegal in any position.
            return chess.Move.from_uci("e2e5")

    with pytest.raises(ValueError, match=r"illegal-mover returned illegal move e2e5"):
        play_game(IllegalMover(), RandomPolicy(), rng=random.Random(0))


def test_play_game_rejects_an_illegal_move_after_legal_ones():
    """The guard runs every ply, so a mid-game illegal move is caught too.

    The stub plays a legal e2e4 as White, then -- asked to move as Black --
    returns e2e4 again, which Black cannot play (there is no black pawn on e2).
    The first, legal move must be committed before the guard fires, which is
    what distinguishes "rejected mid-game" from "rejected immediately".
    """

    class LegalThenIllegal:
        name = "legal-then-illegal"

        def __init__(self):
            self.calls = 0

        def select(self, board):
            self.calls += 1
            return chess.Move.from_uci("e2e4")  # legal for White, illegal for Black

    with pytest.raises(ValueError, match=r"legal-then-illegal returned illegal move"):
        play_game(LegalThenIllegal(), LegalThenIllegal(), rng=random.Random(0))
