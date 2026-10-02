"""Runner and report tests: does the harness measure what it claims to?

The runner is the one part of the bench that can be wrong in a way no amount of
rating-model correctness can rescue. Four failure modes, all silent, all tested
here:

1. **Colour bias.** An odd game count, or a colour assignment that does not
   alternate, gives one side an extra game. The match then reports a systematic
   difference as if it were strength.
2. **Opening reuse.** Playing the same FEN for every game makes a six-game match
   six copies of one result. The match looks precise and measures nothing.
3. **Shared policy state.** Building one policy per round-robin instead of one
   per match lets a warm transposition table cross into the next match.
4. **Scoring unfinished games.** They must be counted separately and excluded,
   never folded into wins, losses or draws.

The tests use deliberately trivial policies (deterministic, or coin-flipping
with a seeded RNG) so that the measured quantity -- the *bookkeeping* -- is the
only thing under test, at a cost of milliseconds rather than minutes.
"""

from __future__ import annotations

import chess
import pytest

from bench.levels import LevelSpec, build, is_playable
from bench.rating import MatchRecord, fit_bradley_terry
from bench.report import (
    build_report,
    format_ablation,
    format_matrix,
    format_ratings,
    format_summary,
)
from bench.runner import BenchResult, MatchOutcome, play_one_match, run_round_robin
from chessrl import masks as M


# --------------------------------------------------------------------------
# fake policies: cheap, deterministic, and honest about their behaviour
# --------------------------------------------------------------------------

class _FirstLegal:
    """Always plays the first legal move. Fully deterministic."""

    name = "first"

    def select(self, board: chess.Board) -> chess.Move:
        return next(iter(board.legal_moves))


class _LastLegal:
    name = "last"

    def select(self, board: chess.Board) -> chess.Move:
        return list(board.legal_moves)[-1]


def _spec(name: str, factory) -> LevelSpec:
    return LevelSpec(name, factory, "test")


def _first_spec(name="first"):
    return _spec(name, lambda: _FirstLegal())


def _last_spec(name="last"):
    return _spec(name, lambda: _LastLegal())


# --------------------------------------------------------------------------
# colour balance and opening discipline
# --------------------------------------------------------------------------

def test_match_plays_an_even_number_of_games():
    """An odd request is rounded up, because an odd count biases a colour."""
    outcome = play_one_match(_first_spec(), _last_spec(), games=3)
    assert len(outcome.games) == 4
    assert outcome.record.played + outcome.record.unfinished == 4


def test_match_alternates_colours():
    """The two policies must each get exactly half the white games.

    Detected from outside the game loop, using the book: game *i* starts from
    ``OPENING_BOOK[i]``, so the policy that opens is whichever one was assigned
    White in that game. Recovered here from the recorded position rather than
    from the tally, because a tally can balance by accident while a colour
    sequence cannot.
    """
    openers: list[str] = []

    class Recorder:
        def __init__(self, name):
            self.name = name

        def select(self, board: chess.Board) -> chess.Move:
            if not board.move_stack:
                openers.append(self.name)
            return next(iter(board.legal_moves))

    a = _spec("A", lambda: Recorder("A"))
    b = _spec("B", lambda: Recorder("B"))
    outcome = play_one_match(a, b, games=4)
    assert len(outcome.games) == 4
    assert openers == ["A", "B", "A", "B"]


def test_match_uses_distinct_openings():
    """Six games must start from six books, not one FEN six times."""
    outcome = play_one_match(_first_spec(), _last_spec(), games=6)
    starts = [g.fens[0] for g in outcome.games if g.fens]
    # There are 6 book entries, so a 6-game match should use 6 distinct starts.
    assert len(set(starts)) == len(starts)


def test_a_shorter_match_than_the_book_still_varies():
    outcome = play_one_match(_first_spec(), _last_spec(), games=4)
    starts = [g.fens[0] for g in outcome.games if g.fens]
    assert len(set(starts)) == len(starts)


def test_opening_book_entries_are_all_playable():
    from bench.levels import OPENING_BOOK

    for fen in OPENING_BOOK:
        assert is_playable(fen), fen


def test_opening_book_entries_are_white_to_move():
    """The book invariant. A Black-to-move entry inverts the colour assignment
    silently, so the match is colour-biased while looking balanced."""
    from bench.levels import OPENING_BOOK, assert_white_to_move

    for fen in OPENING_BOOK:
        assert_white_to_move(fen)      # raises on a violation
        assert chess.Board(fen).turn == chess.WHITE


def test_opening_book_positions_are_distinct():
    from bench.levels import OPENING_BOOK

    positions = {chess.Board(f).board_fen() for f in OPENING_BOOK}
    assert len(positions) == len(OPENING_BOOK)


def test_a_fresh_policy_is_built_per_match():
    """The factory must be called again for each match, not once per run.

    The check is on call *count*: a spec whose factory counts invocations must
    show one call per match it plays, which is what guarantees no cache crosses
    a match boundary.
    """
    calls = {"n": 0}

    def factory():
        calls["n"] += 1
        return _FirstLegal()

    a = LevelSpec("A", factory, "test")
    b = _last_spec("B")
    run_round_robin([a, b], games=2)
    # one probe build (availability check) + one for the match
    assert calls["n"] >= 2


# --------------------------------------------------------------------------
# bookkeeping: the tallies must match the games
# --------------------------------------------------------------------------

def test_record_totals_match_the_games_played():
    outcome = play_one_match(_first_spec(), _last_spec(), games=6)
    rec = outcome.record
    assert rec.played + rec.unfinished == len(outcome.games)
    assert rec.played == rec.wins + rec.draws + rec.losses


def test_unfinished_games_are_counted_separately():
    """Two policies that shuffle forever must produce unfinished, not draws.

    ``first`` vs ``last`` on a bare-kings-ish line is not guaranteed to shuffle,
    so use a position where both sides only have king moves that repeat: the
    no-progress rule then fires and the game is unfinished.
    """
    outcome = play_one_match(
        _first_spec(), _last_spec(), games=2,
        opening_fens=["8/8/8/4k3/8/4K3/8/8 w - - 0 1"],
        no_progress_limit=4,
    )
    rec = outcome.record
    assert rec.played == 0
    assert rec.wins == rec.draws == rec.losses == 0
    assert rec.unfinished == 2


def test_zero_evidence_match_contributes_nothing_to_the_fit():
    """An all-unfinished match must not move any rating."""
    base = fit_bradley_terry(
        [MatchRecord("A", "B", wins=6)], ["A", "B"], anchor="B"
    )
    padded = fit_bradley_terry(
        [MatchRecord("A", "B", wins=6), MatchRecord("A", "C", unfinished=20)],
        ["A", "B", "C"], anchor="B",
    )
    a_base = next(r.elo for r in base.ratings if r.name == "A")
    a_pad = next(r.elo for r in padded.ratings if r.name == "A")
    assert a_pad == pytest.approx(a_base, abs=1e-9)


# --------------------------------------------------------------------------
# round-robin structure
# --------------------------------------------------------------------------

def test_round_robin_plays_every_pair_once():
    specs = [_first_spec("A"), _last_spec("B"), _first_spec("C")]
    bench = run_round_robin(specs, games=2)
    pairs = {frozenset((m.a, m.b)) for m in bench.matches}
    assert pairs == {frozenset(("A", "B")), frozenset(("A", "C")),
                     frozenset(("B", "C"))}
    assert len(bench.matches) == 3


def test_round_robin_single_entrant_plays_nothing():
    bench = run_round_robin([_first_spec("solo")], games=2)
    assert bench.matches == []
    assert bench.names == ["solo"]


def test_unavailable_entrants_are_skipped_and_reported():
    """A factory that raises must drop the entrant, not the run."""

    def broken():
        raise ImportError("torch is not installed")

    specs = [
        _first_spec("A"),
        _last_spec("B"),
        LevelSpec("L5", broken, "ladder", needs_torch=True),
    ]
    bench = run_round_robin(specs, games=2)
    assert "L5" in bench.skipped
    assert "torch is not installed" in bench.skipped["L5"]
    assert "optional extra" in bench.skipped["L5"]
    assert set(bench.names) == {"A", "B"}
    assert len(bench.matches) == 1


def test_build_reports_missing_torch_without_raising():
    def broken():
        raise ImportError("no module named torch")

    policy, err = build(LevelSpec("x", broken, "ladder", needs_torch=True))
    assert policy is None
    assert "unavailable" in err


def test_matrix_is_antisymmetric():
    specs = [_first_spec("A"), _last_spec("B"), _first_spec("C")]
    bench = run_round_robin(specs, games=2)
    grid = bench.matrix()
    idx = {n: i for i, n in enumerate(bench.names)}
    for m in bench.matches:
        i, j = idx[m.a], idx[m.b]
        fwd, rev = grid[i][j], grid[j][i]
        assert fwd.startswith(f"+{m.record.wins}-{m.record.losses}")
        assert rev.startswith(f"+{m.record.losses}-{m.record.wins}")


# --------------------------------------------------------------------------
# the report
# --------------------------------------------------------------------------

def _tiny_bench() -> BenchResult:
    specs = [_first_spec("A"), _last_spec("B")]
    return run_round_robin(specs, games=2)


def test_report_contains_every_required_block():
    bench = _tiny_bench()
    text = build_report(bench)
    assert "RATING" in text
    assert "RAW MATRIX" in text
    assert "RESIDUALS" in text
    assert "A" in text and "B" in text


def test_report_states_the_anchor():
    bench = _tiny_bench()
    text = build_report(bench, anchor="B")
    assert "'B' = 0" in text


def test_report_marks_provisional_ratings():
    """Two games each way is below the evidence threshold and must be flagged."""
    bench = _tiny_bench()
    fit = fit_bradley_terry(bench.records, bench.names, anchor="B")
    text = format_ratings(fit, min_games=4)
    assert "provisional" in text


def test_ratings_table_carries_evidence_columns():
    bench = _tiny_bench()
    fit = fit_bradley_terry(bench.records, bench.names, anchor="B")
    text = format_ratings(fit)
    for column in ("elo", "games", "W-D-L", "score%", "unf"):
        assert column in text


def test_matrix_reports_unfinished_games():
    bench = BenchResult(
        names=["A", "B"],
        matches=[MatchOutcome("A", "B", MatchRecord("A", "B", unfinished=3))],
    )
    text = format_matrix(bench)
    assert "u3" in text


def test_empty_report_degrades_gracefully():
    bench = BenchResult(names=["A", "B"], matches=[])
    text = format_summary(bench, fit_bradley_terry([], ["A", "B"]))
    assert "no decided games" in text


def test_game_count_does_not_depend_on_the_replay_log():
    """A report re-rendered from saved counts must state the same game totals.

    ``MatchOutcome.games`` holds the ``GameResult`` objects and is dropped by a
    JSON round-trip, because only the counts are persisted. Deriving the game
    total from it made a reloaded report print "0 games (108 decided)" while the
    ratings were correct -- a silent display bug in the one column a reader uses
    to judge how much evidence there is.
    """
    bench = BenchResult(
        names=["A", "B"],
        matches=[
            MatchOutcome("A", "B",
                         MatchRecord("A", "B", wins=3, draws=1, losses=1,
                                     unfinished=2),
                         games=[]),        # as after a JSON load
        ],
    )
    fit = fit_bradley_terry(bench.records, bench.names, anchor="B")
    text = format_summary(bench, fit)
    assert "7 (5 decided, 2 unfinished)" in text
    assert "-" not in text.split("games         :")[1].split("\n")[0]


def test_report_is_plain_ascii():
    """The report has to survive a terminal and a commit message."""
    bench = _tiny_bench()
    text = build_report(bench)
    text.encode("ascii")          # raises if any non-ascii slipped in


def test_ablation_table_marks_the_baseline():
    """The all-off variant must be present and be the reference row."""
    from bench.rating import MatchRecord as MR

    recs = [
        MR("L5-ab-blind", "L5-ab-prior", wins=2, losses=4),
        MR("L5-ab-blind", "L5-ab-prior+eval", wins=1, losses=5),
        MR("L5-ab-blind+eval", "L5-ab-prior+eval", wins=3, losses=3),
    ]
    names = ["L5-ab-blind", "L5-ab-prior", "L5-ab-blind+eval", "L5-ab-prior+eval"]
    fit = fit_bradley_terry(recs, names, anchor="L5-ab-blind")
    text = format_ablation(bench=_empty_bench(names), fit=fit)
    assert "baseline is 'L5-ab-blind'" in text
    assert "delta" in text


def _empty_bench(names) -> BenchResult:
    return BenchResult(names=list(names), matches=[])


# --------------------------------------------------------------------------
# ablation roster wiring
# --------------------------------------------------------------------------

def test_ablation_roster_covers_all_four_seam_combinations():
    from bench.levels import ablation_roster

    specs = ablation_roster()
    assert len(specs) == 4
    tags = {s.name for s in specs}
    assert len(tags) == 4
    for s in specs:
        assert s.needs_torch
        assert "ablation" in s.tags


def test_default_roster_names_are_unique():
    from bench.levels import default_roster

    names = [s.name for s in default_roster()]
    assert len(names) == len(set(names))


def test_the_L35_experiment_registers_all_three_arms():
    """Spec §8: the experiment is the three arms, so a missing one measures nothing.

    ``L3-flat`` in particular is not optional -- without it, a win by L3.5 is
    indistinguishable from "any change from L3 helps".
    """
    from bench.levels import default_roster

    specs = {s.name: s for s in default_roster()}
    for name in ("L3", "L3.5", "L3-flat"):
        assert name in specs, f"{name} is missing from the roster"
    # L3 and L3.5 are rungs of the ladder; the control is an ablation.
    assert specs["L3"].tier == "ladder"
    assert specs["L3.5"].tier == "ladder"
    assert specs["L3-flat"].tier == "ablation"
    assert "ablation" in specs["L3-flat"].tags
    # None of the three needs torch: the parity of the comparison depends on
    # all three being buildable in the same (torch-free) environment.
    assert not any(specs[n].needs_torch for n in ("L3", "L3.5", "L3-flat"))


def test_the_three_experiment_arms_all_build():
    from bench.levels import build, default_roster

    specs = {s.name: s for s in default_roster()}
    counts = {}
    for name in ("L3", "L3.5", "L3-flat"):
        policy, err = build(specs[name])
        assert policy is not None, f"{name} failed to build: {err}"
        counts[name] = policy.scorer.n_parameters

    # L3.5 is the lean arm and L3-flat the control; both must be a fraction of
    # L3 rather than a padded match. Asserted here as well as in
    # test_squarelocal so a roster edit cannot silently break the comparison.
    assert counts["L3"] == 7225
    assert counts["L3.5"] == 783
    assert counts["L3-flat"] < counts["L3"]
    assert counts["L3.5"] < counts["L3"]


def test_the_experiment_arms_share_one_opening_book_and_seed_convention():
    """Both arms must be driven identically or the difference is not architecture.

    The book and the seeds are supplied by the runner rather than the level, so
    this asserts that the three factories are all zero-argument and therefore
    get the same treatment from ``play_one_match``.
    """
    import inspect

    from bench.levels import default_roster

    specs = {s.name: s for s in default_roster()}
    for name in ("L3", "L3.5", "L3-flat"):
        sig = inspect.signature(specs[name].factory)
        # All parameters must have defaults, so ``factory()`` is callable.
        for param in sig.parameters.values():
            assert param.default is not inspect.Parameter.empty, (
                f"{name}.factory needs an argument ({param.name}); the runner "
                "calls it with none"
            )


# --------------------------------------------------------------------------
# L6: the ensemble
# --------------------------------------------------------------------------

def test_L6_is_registered_as_a_ladder_entrant():
    from bench.levels import default_roster

    specs = {s.name: s for s in default_roster()}
    assert "L6" in specs
    assert specs["L6"].tier == "ladder"
    assert "ensemble" in specs["L6"].tags


def test_L6_does_not_need_torch():
    """Its members are L1, L3 and L2-d2, none of which needs torch.

    This is asserted rather than assumed because the obvious way to build an L6
    is to include L4 or L5, which would silently make the whole ensemble
    unavailable on a torch-free machine -- and the bench would then report L6 as
    "unavailable" without anyone noticing it had been made optional.
    """
    from bench.levels import default_roster

    specs = {s.name: s for s in default_roster()}
    assert specs["L6"].needs_torch is False


def test_L6_builds_and_is_a_distribution_policy():
    from bench.levels import build, default_roster
    from chessrl.game import DistributionPolicy

    specs = {s.name: s for s in default_roster()}
    policy, err = build(specs["L6"])
    assert policy is not None, f"L6 failed to build: {err}"
    assert isinstance(policy, DistributionPolicy)


def test_L6_contains_both_blendable_and_voting_members():
    """The ensemble's two-stage combination is only exercised if it has both.

    L1 and L3 expose ``move_distribution``; L2-d2 exposes only ``select``. If a
    future edit dropped the search member, the voting path would become dead
    code that no bench run ever touches.
    """
    from bench.levels import build, default_roster

    specs = {s.name: s for s in default_roster()}
    policy, _ = build(specs["L6"])
    assert policy.blendable, "no blendable member; the blend stage is dead code"
    assert policy.voters, "no voting member; the vote stage is dead code"


def test_L6_starts_as_a_uniform_mixture():
    """No ledger yet, so the bench row must be an unweighted, honest mixture.

    Reporting a fitted ensemble in the matrix would be circular -- its weights
    would come from games the matrix has not played.
    """
    from bench.levels import build, default_roster

    specs = {s.name: s for s in default_roster()}
    policy, _ = build(specs["L6"])
    weights = [m.weight for m in policy.members]
    assert weights == [1.0] * len(weights)


def test_L6_produces_a_valid_distribution_from_the_opening():
    from bench.levels import build, default_roster

    specs = {s.name: s for s in default_roster()}
    policy, _ = build(specs["L6"])
    board = chess.Board()
    dist = policy.move_distribution(board)
    assert dist.sum() == pytest.approx(1.0)
    legal = {board.legal_moves}
    assert all(dist[M.move_to_index(m)] >= 0.0 for m in board.legal_moves)
    assert policy.select(board) in board.legal_moves


def test_the_ensemble_driver_agrees_with_the_bench_roster():
    """The driver and the bench row must describe the same compositional policy.

    They are separate code paths by necessity -- the driver needs fitted weights,
    the bench needs a zero-argument factory -- so the composition is duplicated.
    This test is what keeps the duplication honest: change one roster and the
    other must change with it.
    """
    import bench.ensemble_run as ER
    from bench.levels import default_roster

    specs = {s.name: s for s in default_roster()}
    bench_names = [m.name for m in build(specs["L6"])[0].members]
    assert tuple(bench_names) == ER.MEMBER_NAMES


def test_the_fitted_ensemble_is_worse_than_its_best_member():
    """A recorded negative result, locked in so it cannot be forgotten.

    The absolute weighting rule (the default) scores members on agreement with the
    played move. Under it the two blind/weak members, each just above chance, each
    earn a real vote, and their weights *sum* to more than the one member that can
    see material -- so in the blend their noise dilutes the seeing member and the
    ensemble underperforms its best component. This test asserts the *mechanism*
    (blind pair's combined weight > search member's weight) rather than the game
    result, because the game result depends on search depth and would be brittle.

    The accuracies used here are the *corrected* measurement from
    ``bench/README.md`` (after the two scoring bugs were fixed): L1 0.417, L3
    0.375, L2-d2 0.750. They replace the old bug-inflated numbers (L2-d2 had been
    scored 0.917 by the dead-code safety proxy) -- see ``test_ensemble.py``'s
    regression section for how those were caught.

    The relative weighting rule (``EnsembleConfig.relative=True``) is the structural
    fix for this: it scales each member against the set mean, so the below-mean
    members drop to ~zero and only the competent member survives. That behaviour is
    asserted by ``test_relative_weighting_collapses_to_the_best_member`` below.
    """
    from chessrl.ensemble import OutcomeLedger, accuracy_weights
    from bench.ensemble_run import MEMBER_NAMES

    # The corrected accuracies from the calibration run in the README.
    ledger = OutcomeLedger()
    ledger.note_many("L1", [True] * 10 + [False] * 14)   # 0.417
    ledger.note_many("L3", [True] * 9 + [False] * 15)    # 0.375
    ledger.note_many("L2-d2", [True] * 18 + [False] * 6) # 0.750

    weights = accuracy_weights(ledger, list(MEMBER_NAMES))
    blind = weights["L1"] + weights["L3"]
    assert blind > weights["L2-d2"], (
        "the blind members no longer outvote the search member under the absolute "
        "rule; if the weighting metric was deliberately changed, replace this test "
        "with its inverse"
    )


def test_relative_weighting_collapses_to_the_best_member():
    """The structural fix: relative mode must not let a mediocre pair dilute one
    competent member.

    With the same corrected accuracies as the test above, ``relative=True`` scales
    each member against the set mean (here 0.514). The two below-mean members drop
    to ~zero (only the floor remains) and the seeing member takes the overwhelming
    share, so the ensemble becomes its best component rather than something worse.
    """
    from chessrl.ensemble import OutcomeLedger, accuracy_weights
    from bench.ensemble_run import MEMBER_NAMES

    ledger = OutcomeLedger()
    ledger.note_many("L1", [True] * 10 + [False] * 14)
    ledger.note_many("L3", [True] * 9 + [False] * 15)
    ledger.note_many("L2-d2", [True] * 18 + [False] * 6)

    weights = accuracy_weights(ledger, list(MEMBER_NAMES), relative=True)
    blind = weights["L1"] + weights["L3"]
    assert weights["L2-d2"] > blind, (
        "relative weighting should give the competent member the dominant share, "
        f"got blind={blind:.3f} vs L2-d2={weights['L2-d2']:.3f}"
    )
    # And it should be a near-total share, not a marginal one.
    assert weights["L2-d2"] > 0.9, (
        "relative weighting should collapse almost entirely onto the best member, "
        f"got {weights['L2-d2']:.3f}"
    )


def test_an_ensemble_with_one_member_weighted_reproduces_that_member():
    """The property that makes the negative result meaningful.

    Giving the search member all the weight must reproduce it exactly -- which
    is what shows the ensemble *could* have been as good as its best member, and
    that the fitted weights, not the combination machinery, are what cost it.
    """
    import numpy as np

    from chessrl.ensemble import EnsemblePolicy
    from bench.levels import build, default_roster

    specs = {s.name: s for s in default_roster()}
    l2, _ = build(specs["L2-d2"])
    l1, _ = build(specs["L1"])

    ensemble = EnsemblePolicy(seed=0)
    ensemble.add("L1", l1, weight=0.0)
    ensemble.add("L2-d2", l2, weight=1.0)

    board = chess.Board()
    ensemble_dist = ensemble.move_distribution(board)
    # With L1 at zero weight the only contributor is the search member's vote.
    assert ensemble_dist[M.move_to_index(l2.select(board))] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# the CLI's save/load round-trip
# --------------------------------------------------------------------------

def test_json_round_trip_preserves_the_counts_and_the_ratings(tmp_path):
    """Saving and reloading must reproduce the ratings exactly.

    This is the property that makes the saved matrix worth keeping: a rating
    table can be recomputed after a fix to the fit without replaying a game. If
    the round-trip lost any count, the recomputed table would differ, and the
    difference would look like a real change in strength.
    """
    from bench.__main__ import _load, _serialise

    bench = run_round_robin([_first_spec("A"), _last_spec("B"), _first_spec("C")],
                            games=2)
    path = tmp_path / "bench.json"
    path.write_text(__import__("json").dumps(_serialise(bench)), encoding="utf-8")
    reloaded = _load(path)

    assert reloaded.names == bench.names
    assert len(reloaded.matches) == len(bench.matches)

    before = fit_bradley_terry(bench.records, bench.names, anchor="B")
    after = fit_bradley_terry(reloaded.records, reloaded.names, anchor="B")
    for a, b in zip(before.ratings, after.ratings):
        assert a.name == b.name
        assert a.elo == pytest.approx(b.elo, abs=1e-9)
        assert a.games == b.games
        assert a.unfinished == b.unfinished

    # and the re-rendered report agrees on the totals
    text_before = format_summary(bench, before)
    text_after = format_summary(reloaded, after)
    line_before = text_before.split("games         :")[1].split("\n")[0]
    line_after = text_after.split("games         :")[1].split("\n")[0]
    assert line_before == line_after
