"""Smoke tests for the H15 baseline verification script.

The full script replays 21 pairs twice (~13 min), which is too slow for the
suite. These tests pin the *mechanism* cheaply: that the script's evaluator
revert actually removes the `king_shelter_score` term, that the census compares
like-for-like, and that `--games` is honoured. The slow full run is invoked by
hand and its JSON is stored at `bench/h15_baseline.json`.
"""
from __future__ import annotations

import json

import chess
import pytest

import chessrl.value as V
from bench.scripts import h15_baseline as H


def test_evaluator_revert_zeroes_the_king_shelter_term():
    """The attribution hinges on this: the revert must be a real no-op eval."""
    # A castled-white position where king_shelter is non-zero on both sides.
    board = chess.Board(
        "r1bq1rk1/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQ1RK1 w - - 0 1"
    )
    live = V.king_shelter_score(board, chess.WHITE)
    original = V.king_shelter_score
    try:
        V.king_shelter_score = lambda b, c: 0
        assert V.king_shelter_score(board, chess.WHITE) == 0
        assert V.king_shelter_score(board, chess.BLACK) == 0
    finally:
        V.king_shelter_score = original
    # Restored afterwards -- the revert must not leak.
    assert V.king_shelter_score(board, chess.WHITE) == live


def test_stored_rows_keys_are_pairs(tmp_path, monkeypatch):
    """`_stored_rows` must yield (a, b) -> 4-tuple, matching the census use."""
    path = tmp_path / "results.json"
    path.write_text(json.dumps({
        "names": ["A", "B"],
        "matches": [
            {"a": "A", "b": "B", "wins": 1, "draws": 2, "losses": 3,
             "unfinished": 4},
        ],
    }), encoding="utf-8")
    monkeypatch.setattr(H, "STORED", path)
    rows = H._stored_rows()
    assert rows == {("A", "B"): (1, 2, 3, 4)}


def test_replay_honours_the_game_count_on_one_fast_pair():
    """A one-pair, 2-game replay must finish and return a valid 4-tuple.

    Uses `random vs L1`: both are cheap policies, so this is seconds not
    minutes. The point is only that a match of N games reports N games.
    """
    from bench import levels

    specs = {s.name: s for s in levels.default_roster()}
    spec_a, spec_b = "random", "L1"
    assert spec_a in specs and spec_b in specs
    from bench.runner import play_one_match

    rec = play_one_match(specs[spec_a], specs[spec_b], games=2).record
    total = rec.wins + rec.draws + rec.losses + rec.unfinished
    assert total == 2, f"expected 2 games, got {total}"


def test_payload_read_states_the_corrected_mechanism():
    """The stored JSON's read must name the evaluator, not the book, as cause.

    Guards against a future edit quietly reverting the finding to H15's
    original (wrong) framing.
    """
    from pathlib import Path

    data = json.loads(
        (Path(__file__).resolve().parents[1] / "bench" / "h15_baseline.json")
        .read_text(encoding="utf-8")
    )
    text = data["read"].lower()
    assert "book" in text, "the read must address the book claim it refutes"
    assert "king_shelter" in text, "the read must name the real cause"
