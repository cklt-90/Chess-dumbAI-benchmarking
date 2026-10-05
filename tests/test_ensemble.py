"""L6 ensemble tests: does the combination rule do what it claims?

The ensemble is not a learner. It has no parameters of its own, and it cannot
get stronger by being trained -- only by being *weighted*. That makes it
unusually easy to test in isolation, and unusually important to test carefully,
because every way it can be wrong is a way of silently misreporting someone
else's ability:

1. **The accuracy arithmetic.** A member never asked must not be recorded as
   having been wrong, and a member that is only right half the time must not be
   worth the same as one that is never right.
2. **The weighting rules.** Three separate claims: no evidence means an average
   vote (not zero, not one), chance accuracy means half a vote, and better
   accuracy means more weight -- monotonically, and nothing else.
3. **The blend.** The output must be a genuine distribution: sum to one, support
   inside the legal moves, and reducible to a single member when that member
   holds all the weight.
4. **The vote.** A member that cannot produce a distribution must still be able
   to influence the choice, and must be rejected loudly when it votes illegally.
5. **Persistence.** Weights are matched by name, so adding a member does not
   silently reshuffle everyone else's.

The stub members below are the point of this file: they are contrived so that
the correct answer is known by hand, and the assertions are exact numbers rather
than "greater than" whenever exactness is available. Testing the ensemble with
real L1/L3 policies would tell us the two policies agree; it would not tell us
the averaging is right.
"""

from __future__ import annotations

import json
from pathlib import Path

import chess
import numpy as np
import pytest

from chessrl import masks as M
from chessrl.ensemble import (
    EnsembleConfig,
    EnsemblePolicy,
    MemberAccuracy,
    MemberKind,
    MemberSpec,
    OutcomeLedger,
    accuracy_weights,
    score_member_opinions,
)
from chessrl.game import DistributionPolicy, play_game

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


# --------------------------------------------------------------------------
# stub members: the answer is known by hand
# --------------------------------------------------------------------------

class _FixedDistribution:
    """A DistributionPolicy that puts all its mass on one chosen move.

    This is the smallest thing that satisfies the contract, which makes the
    expected mixture arithmetic exact: blending two of these gives exactly the
    two weights on the two moves, with nothing smeared over the other eighteen.
    """

    def __init__(self, move_uci: str, *, spread: float = 0.0):
        self.move_uci = move_uci
        self.spread = spread
        self.calls = 0

    def _target(self, board: chess.Board) -> chess.Move:
        intended = chess.Move.from_uci(self.move_uci)
        if intended in board.legal_moves:
            return intended
        # Fall back to a legal move so the stub can be reused on any position
        # without the test having to know the position in advance.
        return min(board.legal_moves, key=lambda m: m.uci())

    def move_distribution(self, board: chess.Board) -> np.ndarray:
        self.calls += 1
        out = np.zeros(M.ACTION_SPACE, dtype=np.float64)
        legal = list(board.legal_moves)
        target = self._target(board)
        if self.spread > 0.0:
            share = self.spread / max(1, len(legal))
            for move in legal:
                out[M.move_to_index(move)] = share
            out[M.move_to_index(target)] += 1.0 - self.spread
        else:
            out[M.move_to_index(target)] = 1.0
        return out

    def select(self, board: chess.Board) -> chess.Move:
        return self._target(board)


class _FixedMove:
    """A MovePolicy: no distribution at all. Must be voted, never blended."""

    def __init__(self, move_uci: str, *, illegal: str | None = None):
        self.move_uci = move_uci
        self.illegal = illegal
        self.calls = 0

    def select(self, board: chess.Board) -> chess.Move:
        self.calls += 1
        if self.illegal is not None:
            return chess.Move.from_uci(self.illegal)
        intended = chess.Move.from_uci(self.move_uci)
        if intended in board.legal_moves:
            return intended
        return min(board.legal_moves, key=lambda m: m.uci())


class _EmptyDistribution:
    """A blendable member that returns nothing, to exercise the empty case."""

    def move_distribution(self, board: chess.Board) -> np.ndarray:
        return np.zeros(M.ACTION_SPACE, dtype=np.float64)

    def select(self, board: chess.Board) -> chess.Move:
        return min(board.legal_moves, key=lambda m: m.uci())


def _ledger_with(entries: dict[str, tuple[int, int]]) -> OutcomeLedger:
    """Build a ledger from ``{name: (correct, predicted)}``."""
    ledger = OutcomeLedger()
    for name, (correct, predicted) in entries.items():
        ledger.register(name)
        ledger.note_many(name, [True] * correct + [False] * (predicted - correct))
    return ledger


# --------------------------------------------------------------------------
# 1. member description
# --------------------------------------------------------------------------

def test_a_member_exposing_move_distribution_is_blendable():
    spec = MemberSpec(name="d", policy=_FixedDistribution("e2e4"))
    assert spec.kind == MemberKind.DISTRIBUTION
    assert spec.is_blendable is True


def test_a_member_without_move_distribution_is_a_voter():
    spec = MemberSpec(name="m", policy=_FixedMove("e2e4"))
    assert spec.kind == MemberKind.MOVE
    assert spec.is_blendable is False


def test_kind_is_detected_from_the_policy_not_declared_by_the_caller():
    """A caller who mislabels a member must not be able to lie to the blend.

    The ensemble's whole job is to know which members it may average. If that
    were a caller-supplied flag, a typo would silently put a single move into a
    distribution as if it were a spread of probabilities. It is derived instead.
    """
    assert MemberSpec(name="x", policy=_FixedDistribution("e2e4")).kind == "distribution"
    assert MemberSpec(name="x", policy=_FixedMove("e2e4")).kind == "move"


def test_a_negative_weight_is_rejected():
    with pytest.raises(ValueError):
        MemberSpec(name="x", policy=_FixedMove("e2e4"), weight=-1.0)


def test_a_zero_weight_is_allowed():
    """Zero is a legitimate weight: it is how a member is parked, not removed."""
    assert MemberSpec(name="x", policy=_FixedMove("e2e4"), weight=0.0).weight == 0.0


# --------------------------------------------------------------------------
# 2. the accuracy ledger
# --------------------------------------------------------------------------

def test_never_asked_is_none_not_zero():
    """The single most important distinction in the ledger.

    A member that has never been exercised has *unknown* accuracy. Recording it
    as 0.0 would conflate "we don't know" with "it is always wrong", and the
    weighting rule would then starve a perfectly good member that simply had not
    been tried yet.
    """
    entry = MemberAccuracy(name="untried")
    assert entry.accuracy is None
    assert entry.predicted == 0


def test_accuracy_is_none_until_the_first_prediction():
    ledger = OutcomeLedger()
    ledger.register("a")
    assert ledger.accuracies()["a"] is None
    ledger.note("a", True)
    assert ledger.accuracies()["a"] == 1.0


def test_accuracy_is_the_simple_ratio():
    ledger = _ledger_with({"a": (3, 4)})
    assert ledger.accuracies()["a"] == pytest.approx(0.75)


def test_note_many_records_each_result():
    ledger = _ledger_with({"a": (2, 5)})
    entry = ledger.get("a")
    assert (entry.correct, entry.predicted) == (2, 5)


def test_len_counts_only_members_that_were_asked():
    ledger = _ledger_with({"a": (1, 2)})
    ledger.register("b")
    assert len(ledger) == 1


def test_total_predicted_sums_across_members():
    ledger = _ledger_with({"a": (1, 2), "b": (3, 4)})
    assert ledger.total_predicted() == 6


def test_registration_order_is_preserved():
    ledger = _ledger_with({"z": (1, 1), "a": (1, 1), "m": (1, 1)})
    assert ledger.names == ["z", "a", "m"]


def test_ledger_round_trips_through_json(tmp_path: Path):
    ledger = _ledger_with({"a": (8, 10), "b": (3, 10)})
    ledger.register("never_asked")
    path = tmp_path / "ledger.json"
    ledger.save(path)

    back = OutcomeLedger.load(path)
    assert back.names == ledger.names
    assert back.accuracies() == ledger.accuracies()
    assert back.accuracies()["never_asked"] is None
    assert back.total_predicted() == ledger.total_predicted()


def test_ledger_file_is_readable_json(tmp_path: Path):
    """The ledger is a report as much as a cache; it must be eyeball-able."""
    path = tmp_path / "ledger.json"
    _ledger_with({"a": (1, 2)}).save(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["a"]["correct"] == 1
    assert raw["a"]["predicted"] == 2


# --------------------------------------------------------------------------
# 3. deriving weights from accuracy
# --------------------------------------------------------------------------

def test_no_members_gives_no_weights():
    assert accuracy_weights(OutcomeLedger(), []) == {}


def test_weighting_a_single_member_gives_it_everything():
    weights = accuracy_weights(_ledger_with({"only": (5, 10)}), ["only"])
    assert weights["only"] == pytest.approx(1.0)


def test_zero_evidence_member_gets_the_average():
    """Rule 1: an untried member is average, not zero and not one.

    Exact by hand under ``w = acc / chance_equivalent``: the tried member has
    accuracy 0.8 and so is worth 0.8/0.5 = 1.6 votes; the untried member is
    worth the mean, 1.0. Neither is floored, so the normalised pair is
    1.6/2.6 and 1.0/2.6.
    """
    ledger = _ledger_with({"known": (8, 10)})
    ledger.register("unknown")
    weights = accuracy_weights(ledger, ["known", "unknown"])
    assert weights["known"] == pytest.approx(1.6 / 2.6)
    assert weights["unknown"] == pytest.approx(1.0 / 2.6)


def test_an_untried_member_outscores_a_perfectly_bad_one():
    """The concrete consequence of the previous test, stated as an ordering."""
    ledger = _ledger_with({"bad": (0, 10)})
    ledger.register("untried")
    weights = accuracy_weights(ledger, ["bad", "untried"])
    assert weights["untried"] > weights["bad"]


def test_all_untried_members_split_evenly():
    ledger = OutcomeLedger()
    weights = accuracy_weights(ledger, ["a", "b", "c"])
    assert all(w == pytest.approx(1 / 3) for w in weights.values())


def test_weights_always_sum_to_one():
    ledger = _ledger_with({"a": (9, 10), "b": (1, 10), "c": (5, 10)})
    assert sum(accuracy_weights(ledger, ["a", "b", "c"]).values()) == pytest.approx(1.0)


def test_better_accuracy_means_more_weight():
    """Rule 3: monotone in accuracy. Nothing else is claimed."""
    ledger = _ledger_with({"good": (9, 10), "mid": (5, 10), "poor": (2, 10)})
    weights = accuracy_weights(ledger, ["good", "mid", "poor"])
    assert weights["good"] > weights["mid"] > weights["poor"]


# --------------------------------------------------------------------------
# relative weighting: the structural fix for the "blind majority outvotes a
# seeing minority" failure. Scales each member against the set mean, so
# below-mean members drop to ~zero and only the competent member survives.
# --------------------------------------------------------------------------

def test_relative_weighting_collapses_to_the_above_mean_member():
    """The whole point of the mode: a mediocre pair cannot pool to beat one good.

    good=0.9, mid=0.5, poor=0.1; mean = 0.5. Relative raw = (0.4, 0.0, 0.0)
    after the max(0, ...) clamp, so after the floor+normalise the whole weight
    lands on ``good`` and the others keep only their (tiny) floor share.
    """
    ledger = _ledger_with({"good": (9, 10), "mid": (5, 10), "poor": (1, 10)})
    weights = accuracy_weights(ledger, ["good", "mid", "poor"], relative=True)
    assert weights["good"] > weights["mid"] + weights["poor"]
    assert weights["good"] > 0.9


def test_relative_weighting_still_discriminates_between_two_above_mean_members():
    """Not just "best wins": two genuinely good members share the weight.

    good=0.8, better=1.0, poor=0.0; mean = 0.6. Relative raw = (0.2, 0.4, 0.0),
    so ``better`` should carry roughly twice ``good`` and ``poor`` nothing.
    """
    ledger = _ledger_with({"good": (8, 10), "better": (10, 10), "poor": (0, 10)})
    weights = accuracy_weights(ledger, ["good", "better", "poor"], relative=True)
    assert weights["better"] > weights["good"] > weights["poor"]
    # Roughly proportional to the above-mean gap (0.2 vs 0.4 -> ~1:2 before floor).
    assert weights["better"] > 1.5 * weights["good"]


def test_relative_weighting_is_independent_of_an_absolute_shift():
    """Relative mode should not care that every member scored 20% higher.

    Absolute mode *does* care (a 0.7 vs 0.9 spread reads as "more decisive" than
    0.5 vs 0.7). Relative mode only reads the spread *within* the set, so shifting
    the whole roster up by a constant should leave the weights essentially
    unchanged. This is the property that makes it robust to a strong or weak
    field, and the reason it is the right fix for the blind-majority failure.
    """
    low = _ledger_with({"good": (9, 10), "mid": (5, 10), "poor": (1, 10)})
    high = _ledger_with({"good": (19, 20), "mid": (15, 20), "poor": (11, 20)})
    w_low = accuracy_weights(low, ["good", "mid", "poor"], relative=True)
    w_high = accuracy_weights(high, ["good", "mid", "poor"], relative=True)
    for name in ("good", "mid", "poor"):
        assert w_low[name] == pytest.approx(w_high[name], abs=1e-6)


def test_relative_weighting_never_produces_negative_weights():
    """The max(0, acc - mean) clamp must hold even when every member is below the
    mean is impossible, but a single below-mean member must not go negative."""
    ledger = _ledger_with({"top": (9, 10), "low": (1, 10)})
    weights = accuracy_weights(ledger, ["top", "low"], relative=True)
    assert weights["low"] >= 0.0
    assert weights["top"] > weights["low"]
    assert sum(weights.values()) == pytest.approx(1.0)


def test_chance_accuracy_is_worth_exactly_the_average():
    """Rule 2, stated exactly: ``chance_equivalent`` is the unit of accuracy.

    A member measuring exactly chance has done the job it was asked to do and no
    more, so it is worth precisely one mean vote -- the same as an untried
    member. This is the identity that makes ``chance_equivalent`` meaningful: it
    is not a threshold to clear, it is the scale's unit.
    """
    ledger = _ledger_with({"chance": (5, 10)})
    ledger.register("untried")
    weights = accuracy_weights(ledger, ["chance", "untried"])
    assert weights["chance"] == pytest.approx(weights["untried"])
    assert weights["chance"] == pytest.approx(0.5)


def test_raising_chance_equivalent_shrinks_what_any_member_is_worth():
    """The knob's actual effect: it sets how good "one vote" is.

    Only the *ratios* matter after normalisation, so raising the unit makes
    every member's score smaller relative to the mean 1.0 given to untried
    members. The observable consequence is that trying gets less rewarding, in
    relative terms, as the bar rises.
    """
    ledger = _ledger_with({"tried": (9, 10)})
    ledger.register("untried")
    lenient = accuracy_weights(ledger, ["tried", "untried"], chance_equivalent=0.25)
    strict = accuracy_weights(ledger, ["tried", "untried"], chance_equivalent=0.75)
    # The tried member is further above the mean when the unit is small.
    assert lenient["tried"] > strict["tried"]


def test_below_chance_is_worth_less_than_at_chance():
    ledger = _ledger_with({"worse": (2, 10), "chance": (5, 10)})
    weights = accuracy_weights(ledger, ["worse", "chance"])
    assert weights["worse"] < weights["chance"]


def test_a_uniformly_wrong_roster_is_still_a_valid_distribution():
    """The degenerate case must not produce NaN or an all-zero vector."""
    ledger = _ledger_with({"a": (0, 10), "b": (0, 10)})
    weights = accuracy_weights(ledger, ["a", "b"])
    assert sum(weights.values()) == pytest.approx(1.0)
    assert all(np.isfinite(w) for w in weights.values())
    # With nothing to distinguish them, they must be equal rather than arbitrary.
    assert weights["a"] == pytest.approx(weights["b"])


def test_a_perfect_member_against_a_uniformly_wrong_roster_is_not_total():
    """No member may be starved to exactly zero by the rules alone."""
    ledger = _ledger_with({"perfect": (10, 10), "awful": (0, 10)})
    weights = accuracy_weights(ledger, ["perfect", "awful"])
    assert weights["awful"] > 0.0
    assert weights["perfect"] > weights["awful"]


def test_a_zero_floor_still_discriminates():
    ledger = _ledger_with({"good": (9, 10), "bad": (0, 10)})
    weights = accuracy_weights(ledger, ["good", "bad"], floor=0.0)
    assert weights["good"] > weights["bad"]


def test_a_high_floor_flattens_the_weights():
    ledger = _ledger_with({"good": (9, 10), "bad": (0, 10)})
    tight = accuracy_weights(ledger, ["good", "bad"], floor=0.01)
    flat = accuracy_weights(ledger, ["good", "bad"], floor=10.0)
    assert flat["bad"] > tight["bad"]
    assert flat["bad"] / flat["good"] > tight["bad"] / tight["good"]


def test_high_temperature_pulls_towards_uniform():
    ledger = _ledger_with({"good": (9, 10), "bad": (1, 10)})
    sharp = accuracy_weights(ledger, ["good", "bad"], temperature=0.5)
    mild = accuracy_weights(ledger, ["good", "bad"], temperature=5.0)
    assert mild["bad"] > sharp["bad"]


def test_zero_temperature_is_rejected():
    with pytest.raises(ValueError):
        accuracy_weights(_ledger_with({"a": (1, 2)}), ["a"], temperature=0.0)


def test_negative_temperature_is_rejected():
    with pytest.raises(ValueError):
        accuracy_weights(_ledger_with({"a": (1, 2)}), ["a"], temperature=-1.0)


def test_negative_floor_is_rejected():
    with pytest.raises(ValueError):
        accuracy_weights(_ledger_with({"a": (1, 2)}), ["a"], floor=-0.1)


def test_weighting_is_order_independent():
    """The weight of a member must not depend on where it sits in the list."""
    ledger = _ledger_with({"a": (8, 10), "b": (2, 10), "c": (5, 10)})
    forward = accuracy_weights(ledger, ["a", "b", "c"])
    backward = accuracy_weights(ledger, ["c", "b", "a"])
    for name in "abc":
        assert forward[name] == pytest.approx(backward[name])


def test_chance_equivalent_is_a_declared_knob_not_a_hidden_constant():
    """Raising the chance bar must lower what a 0.5 member is worth."""
    ledger = _ledger_with({"a": (5, 10)})
    ledger.register("u")
    low = accuracy_weights(ledger, ["a", "u"], chance_equivalent=0.25)
    high = accuracy_weights(ledger, ["a", "u"], chance_equivalent=0.75)
    assert low["a"] > high["a"]


@pytest.mark.parametrize(
    "temperature",
    [1e-12, 1e-9, 1e-6, 1e-3, 0.01, 0.5, 2.0, 100.0, 1e9],
)
def test_any_positive_temperature_produces_a_valid_distribution(temperature):
    """Regression: a very small temperature used to crash, then produce NaN.

    ``temperature > 0`` is validated as legal, so the whole positive range must
    work. The original ``v ** (1/temperature)`` raised OverflowError at
    ``temperature=1e-9``; a first fix moved the overflow into ``np.exp`` and
    produced ``inf/inf = nan``; the max-shifted log-space form is what actually
    holds. This test covers both failure modes at once.
    """
    ledger = _ledger_with({"a": (9, 10)})
    ledger.register("u")
    weights = accuracy_weights(ledger, ["a", "u"], temperature=temperature)
    assert sum(weights.values()) == pytest.approx(1.0)
    assert all(np.isfinite(w) for w in weights.values()), (
        f"temperature {temperature} produced a non-finite weight: {weights}"
    )
    assert all(w >= 0.0 for w in weights.values())


def test_a_very_small_temperature_saturates_to_the_best_member():
    """The limit case, stated as behaviour rather than as absence of a crash.

    As temperature goes to zero, all weight should concentrate on the highest
    scoring member -- not on NaN, and not equally split. The default floor keeps
    a residual share for the other member, so the floor is disabled here to give
    the temperature room to do all the work; the floor's own effect is tested
    separately.
    """
    ledger = _ledger_with({"good": (9, 10), "bad": (1, 10)})
    weights = accuracy_weights(ledger, ["good", "bad"], temperature=1e-9, floor=0.0)
    assert weights["good"] > 0.99
    assert weights["bad"] < 0.01


def test_a_very_small_temperature_still_respects_the_floor():
    """Sharpening must not be able to starve a member the floor protects."""
    ledger = _ledger_with({"good": (9, 10), "bad": (1, 10)})
    weights = accuracy_weights(ledger, ["good", "bad"], temperature=1e-9)
    assert weights["bad"] > 0.0, "the floor was bypassed by a sharp temperature"


def test_sharpening_does_not_warn_about_overflow():
    """The computation must be numerically clean, not merely non-crashing."""
    import warnings

    ledger = _ledger_with({"a": (9, 10)})
    ledger.register("u")
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        weights = accuracy_weights(ledger, ["a", "u"], temperature=1e-9)
    assert np.isfinite(weights["a"])


def test_a_member_scored_at_zero_stays_at_zero_under_sharpening():
    """Pinned zeros: log(0) is -inf, which must not poison the other members.

    A member whose accuracy is zero scores a raw weight of zero, and sharpening
    must leave it there rather than converting the whole vector to NaN. The floor
    is disabled so that the pinning is what is actually under test -- with the
    default floor the member would be lifted off zero, which is the floor doing
    its job rather than the sharpening failing.
    """
    ledger = _ledger_with({"good": (9, 10), "hopeless": (0, 10)})
    weights = accuracy_weights(ledger, ["good", "hopeless"], temperature=0.5,
                               floor=0.0)
    assert np.isfinite(weights["good"])
    assert weights["hopeless"] == pytest.approx(0.0)


# --------------------------------------------------------------------------
# 4. the blend
# --------------------------------------------------------------------------

def test_the_ensemble_is_a_distribution_policy():
    """It must be a drop-in for the game loop and the bench, like L1 and L3."""
    ensemble = EnsemblePolicy(seed=0)
    ensemble.add("d", _FixedDistribution("e2e4"))
    assert isinstance(ensemble, DistributionPolicy)


def test_the_output_is_normalised():
    ensemble = EnsemblePolicy(seed=0)
    ensemble.add("a", _FixedDistribution("e2e4"))
    ensemble.add("b", _FixedDistribution("d2d4"))
    dist = ensemble.move_distribution(chess.Board(START_FEN))
    assert dist.sum() == pytest.approx(1.0)


def test_the_output_is_normalised_even_when_the_inputs_are_not():
    """The regression this test exists for.

    Weights of 1.0 each are the natural thing for a caller to write, and they do
    not sum to one. The game loop takes the distribution on trust and samples
    from it directly, so an un-renormalised mixture of two members gives the
    loop a total mass of two. It must be normalised on read.
    """
    ensemble = EnsemblePolicy(seed=0)
    ensemble.add("a", _FixedDistribution("e2e4"))
    ensemble.add("b", _FixedDistribution("d2d4"))
    ensemble.add("c", _FixedDistribution("g1f3"))
    assert sum(m.weight for m in ensemble.members) == 3.0
    dist = ensemble.move_distribution(chess.Board(START_FEN))
    assert dist.sum() == pytest.approx(1.0)


def test_the_support_is_inside_the_legal_moves():
    ensemble = EnsemblePolicy(seed=0)
    ensemble.add("a", _FixedDistribution("e2e4"))
    ensemble.add("b", _FixedMove("d2d4"))
    board = chess.Board(START_FEN)
    dist = ensemble.move_distribution(board)
    legal = {M.move_to_index(m) for m in board.legal_moves}
    nonzero = set(np.nonzero(dist)[0].tolist())
    assert nonzero <= legal


def test_illegal_moves_get_no_mass():
    ensemble = EnsemblePolicy(seed=0)
    ensemble.add("a", _FixedDistribution("e2e4", spread=0.5))
    board = chess.Board(START_FEN)
    dist = ensemble.move_distribution(board)
    legal = {M.move_to_index(m) for m in board.legal_moves}
    for idx in np.nonzero(dist)[0]:
        assert int(idx) in legal


def test_each_member_of_the_support_carries_its_own_weight():
    """Exact by hand: two point masses at equal weight split the mass in half."""
    ensemble = EnsemblePolicy(seed=0)
    ensemble.add("a", _FixedDistribution("e2e4"), weight=1.0)
    ensemble.add("b", _FixedDistribution("d2d4"), weight=1.0)
    board = chess.Board(START_FEN)
    dist = ensemble.move_distribution(board)
    e4 = M.move_to_index(chess.Move.from_uci("e2e4"))
    d4 = M.move_to_index(chess.Move.from_uci("d2d4"))
    assert dist[e4] == pytest.approx(0.5)
    assert dist[d4] == pytest.approx(0.5)
    assert (dist > 0).sum() == 2


def test_an_unequal_weight_shifts_the_odds():
    ensemble = EnsemblePolicy(seed=0)
    ensemble.add("a", _FixedDistribution("e2e4"), weight=3.0)
    ensemble.add("b", _FixedDistribution("d2d4"), weight=1.0)
    board = chess.Board(START_FEN)
    dist = ensemble.move_distribution(board)
    e4 = M.move_to_index(chess.Move.from_uci("e2e4"))
    d4 = M.move_to_index(chess.Move.from_uci("d2d4"))
    assert dist[e4] == pytest.approx(0.75)
    assert dist[d4] == pytest.approx(0.25)


def test_a_lone_member_is_reproduced_exactly():
    """Weight-1 reduction: one member holding everything gives its own answer."""
    ensemble = EnsemblePolicy(seed=0)
    member = _FixedDistribution("e2e4", spread=0.3)
    ensemble.add("only", member, weight=1.0)
    board = chess.Board(START_FEN)
    mine = ensemble.move_distribution(board)
    theirs = member.move_distribution(board)
    np.testing.assert_allclose(mine, theirs, atol=1e-12)


def test_a_zero_weight_member_is_ignored_entirely():
    ensemble = EnsemblePolicy(seed=0)
    active = _FixedDistribution("e2e4")
    idle = _FixedDistribution("d2d4")
    ensemble.add("active", active, weight=1.0)
    ensemble.add("idle", idle, weight=0.0)
    board = chess.Board(START_FEN)
    dist = ensemble.move_distribution(board)
    d4 = M.move_to_index(chess.Move.from_uci("d2d4"))
    assert dist[d4] == pytest.approx(0.0)
    assert idle.calls == 0, "a zero-weight member was consulted anyway"


def test_contributions_are_reported_and_add_up():
    """The explanation of a choice must total the probability assigned."""
    ensemble = EnsemblePolicy(seed=0)
    ensemble.add("a", _FixedDistribution("e2e4"), weight=1.0)
    ensemble.add("b", _FixedDistribution("d2d4"), weight=1.0)
    ensemble.move_distribution(chess.Board(START_FEN))
    assert sum(ensemble.last_contributions.values()) == pytest.approx(1.0)


def test_contributions_name_every_contributing_member():
    ensemble = EnsemblePolicy(seed=0)
    ensemble.add("a", _FixedDistribution("e2e4"))
    ensemble.add("b", _FixedMove("d2d4"))
    ensemble.move_distribution(chess.Board(START_FEN))
    assert set(ensemble.last_contributions) == {"a", "b"}


def test_a_member_contributing_less_mass_is_reported_as_such():
    """A member's contribution is its weight times the mass it places.

    A member with a *smaller weight* places less mass, so its contribution is
    smaller -- and the report must reflect that, because the contribution is
    what explains the choice. Note that a spread member is not automatically
    worth less than a point mass: both distributions here sum to one, so at
    equal weight they contribute equally. What changes is *where* the mass goes.
    """
    ensemble = EnsemblePolicy(seed=0)
    ensemble.add("heavy", _FixedDistribution("e2e4", spread=0.0), weight=3.0)
    ensemble.add("light", _FixedDistribution("d2d4", spread=0.0), weight=1.0)
    ensemble.move_distribution(chess.Board(START_FEN))
    assert ensemble.last_contributions["heavy"] > ensemble.last_contributions["light"]


def test_a_spread_member_places_less_mass_on_its_favourite_move():
    """The concrete difference between hedging and committing, in the output."""
    ensemble = EnsemblePolicy(seed=0)
    ensemble.add("spread", _FixedDistribution("e2e4", spread=1.0), weight=1.0)
    board = chess.Board(START_FEN)
    dist = ensemble.move_distribution(board)
    e4 = M.move_to_index(chess.Move.from_uci("e2e4"))
    # The sole member, so the output is exactly its own distribution: e4 gets
    # its one twenty-first share and nothing is concentrated anywhere.
    assert dist[e4] == pytest.approx(1.0 / len(list(board.legal_moves)))
    assert (dist > 0).sum() == len(list(board.legal_moves))


def test_a_member_returning_the_wrong_shape_is_rejected():
    class _WrongShape:
        def move_distribution(self, board):
            return np.zeros(7, dtype=np.float64)

        def select(self, board):
            return min(board.legal_moves, key=lambda m: m.uci())

    ensemble = EnsemblePolicy(seed=0)
    ensemble.add("broken", _WrongShape())
    with pytest.raises(ValueError, match="shape"):
        ensemble.move_distribution(chess.Board(START_FEN))


# --------------------------------------------------------------------------
# 5. the vote
# --------------------------------------------------------------------------

def test_a_move_only_member_still_influences_the_choice():
    ensemble = EnsemblePolicy(seed=0)
    ensemble.add("voter", _FixedMove("d2d4"), weight=1.0)
    board = chess.Board(START_FEN)
    dist = ensemble.move_distribution(board)
    d4 = M.move_to_index(chess.Move.from_uci("d2d4"))
    assert dist[d4] == pytest.approx(1.0)


def test_a_voter_is_listed_separately_from_the_blendable_members():
    ensemble = EnsemblePolicy(seed=0)
    ensemble.add("d", _FixedDistribution("e2e4"))
    ensemble.add("m", _FixedMove("d2d4"))
    assert [s.name for s in ensemble.blendable] == ["d"]
    assert [s.name for s in ensemble.voters] == ["m"]


def test_a_vote_is_weighted_like_a_blend():
    ensemble = EnsemblePolicy(seed=0)
    ensemble.add("d", _FixedDistribution("e2e4"), weight=3.0)
    ensemble.add("m", _FixedMove("d2d4"), weight=1.0)
    board = chess.Board(START_FEN)
    dist = ensemble.move_distribution(board)
    e4 = M.move_to_index(chess.Move.from_uci("e2e4"))
    d4 = M.move_to_index(chess.Move.from_uci("d2d4"))
    assert dist[e4] == pytest.approx(0.75)
    assert dist[d4] == pytest.approx(0.25)


def test_an_illegal_vote_is_rejected_loudly():
    """Silently dropping it would hide a member that had broken its contract."""
    ensemble = EnsemblePolicy(seed=0)
    # a1a8 is not legal from the opening position.
    ensemble.add("cheater", _FixedMove("e2e4", illegal="a1a8"), weight=1.0)
    with pytest.raises(ValueError, match="illegal"):
        ensemble.move_distribution(chess.Board(START_FEN))


def test_a_zero_weight_voter_is_not_consulted():
    ensemble = EnsemblePolicy(seed=0)
    voter = _FixedMove("d2d4")
    ensemble.add("quiet", voter, weight=0.0)
    ensemble.add("real", _FixedDistribution("e2e4"), weight=1.0)
    ensemble.move_distribution(chess.Board(START_FEN))
    assert voter.calls == 0


def test_a_distribution_member_is_always_more_than_a_vote():
    """Blending is strictly more informative than voting, and this shows it.

    Two members with identical weight: one commits to e4, one spreads over the
    legal moves while mildly preferring e4. If the spread member were merely
    voted for, it would place all of its weight on e4 and the two would tie. It
    does not tie -- the spread member's mass is genuinely spread.
    """
    ensemble = EnsemblePolicy(seed=0)
    ensemble.add("point", _FixedDistribution("e2e4", spread=0.0), weight=1.0)
    ensemble.add("spread", _FixedDistribution("e2e4", spread=1.0), weight=1.0)
    board = chess.Board(START_FEN)
    dist = ensemble.move_distribution(board)
    e4 = M.move_to_index(chess.Move.from_uci("e2e4"))
    # The point mass gives e4 half the vote; the spread gives it a little more
    # than its twenty-first share, so e4 lands above a half but well below one.
    assert 0.5 < dist[e4] < 1.0
    assert (dist > 0).sum() > 1


# --------------------------------------------------------------------------
# 6. selection
# --------------------------------------------------------------------------

def test_greedy_selection_takes_the_arg_max():
    ensemble = EnsemblePolicy(seed=0, greedy=True)
    ensemble.add("a", _FixedDistribution("e2e4"), weight=3.0)
    ensemble.add("b", _FixedDistribution("d2d4"), weight=1.0)
    assert ensemble.select(chess.Board(START_FEN)) == chess.Move.from_uci("e2e4")


def test_sampling_selection_returns_a_legal_move():
    ensemble = EnsemblePolicy(seed=0)
    ensemble.add("a", _FixedDistribution("e2e4"))
    ensemble.add("b", _FixedDistribution("d2d4"))
    board = chess.Board(START_FEN)
    for _ in range(50):
        assert ensemble.select(board) in board.legal_moves


def test_sampling_covers_the_support_over_many_draws():
    ensemble = EnsemblePolicy(seed=0)
    ensemble.add("a", _FixedDistribution("e2e4"))
    ensemble.add("b", _FixedDistribution("d2d4"))
    board = chess.Board(START_FEN)
    seen = {ensemble.select(board) for _ in range(200)}
    assert seen == {chess.Move.from_uci("e2e4"), chess.Move.from_uci("d2d4")}


def test_selection_is_seed_reproducible():
    def sequence(seed: int) -> list[str]:
        ensemble = EnsemblePolicy(seed=seed)
        ensemble.add("a", _FixedDistribution("e2e4"))
        ensemble.add("b", _FixedDistribution("d2d4"))
        board = chess.Board(START_FEN)
        return [ensemble.select(board).uci() for _ in range(30)]

    assert sequence(4) == sequence(4)


def test_different_seeds_give_different_sequences_eventually():
    def sequence(seed: int) -> list[str]:
        ensemble = EnsemblePolicy(seed=seed)
        ensemble.add("a", _FixedDistribution("e2e4"))
        ensemble.add("b", _FixedDistribution("d2d4"))
        board = chess.Board(START_FEN)
        return [ensemble.select(board).uci() for _ in range(60)]

    assert sequence(1) != sequence(2)


def test_an_all_zero_weight_ensemble_refuses_to_move():
    ensemble = EnsemblePolicy(seed=0)
    ensemble.add("a", _FixedDistribution("e2e4"), weight=0.0)
    with pytest.raises(ValueError, match="zero weight"):
        ensemble.select(chess.Board(START_FEN))


def test_an_ensemble_with_no_mass_refuses_to_move():
    ensemble = EnsemblePolicy(seed=0)
    ensemble.add("empty", _EmptyDistribution(), weight=1.0)
    with pytest.raises(ValueError, match="no mass"):
        ensemble.select(chess.Board(START_FEN))


def test_a_finished_position_yields_no_mass():
    """Checkmate has no legal moves; the ensemble must not invent one.

    The game loop stops before asking, but a caller can ask directly, and a
    crash here would be indistinguishable from a bug in the search. An empty
    distribution is the honest answer.
    """
    ensemble = EnsemblePolicy(seed=0)
    ensemble.add("a", _FixedDistribution("e2e4"), weight=1.0)
    mated = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
    assert mated.is_checkmate()
    assert ensemble.move_distribution(mated).sum() == pytest.approx(0.0)


# --------------------------------------------------------------------------
# 7. weights from the ledger
# --------------------------------------------------------------------------

def test_setting_weights_from_a_ledger_normalises_them():
    ledger = _ledger_with({"a": (8, 10), "b": (2, 10)})
    ensemble = EnsemblePolicy(seed=0)
    ensemble.add("a", _FixedDistribution("e2e4"))
    ensemble.add("b", _FixedDistribution("d2d4"))
    ensemble.set_weights_from_ledger(ledger)
    assert sum(m.weight for m in ensemble.members) == pytest.approx(1.0)


def test_a_better_rated_member_gets_more_of_the_blend():
    ledger = _ledger_with({"a": (9, 10), "b": (1, 10)})
    ensemble = EnsemblePolicy(seed=0)
    ensemble.add("a", _FixedDistribution("e2e4"))
    ensemble.add("b", _FixedDistribution("d2d4"))
    ensemble.set_weights_from_ledger(ledger)
    board = chess.Board(START_FEN)
    dist = ensemble.move_distribution(board)
    e4 = M.move_to_index(chess.Move.from_uci("e2e4"))
    d4 = M.move_to_index(chess.Move.from_uci("d2d4"))
    assert dist[e4] > dist[d4]


def test_a_member_missing_from_the_ledger_gets_the_average():
    """A newly added member must be usable immediately, not frozen at zero."""
    ledger = _ledger_with({"a": (9, 10), "b": (1, 10)})
    ensemble = EnsemblePolicy(seed=0)
    ensemble.add("a", _FixedDistribution("e2e4"))
    ensemble.add("b", _FixedDistribution("d2d4"))
    ensemble.add("newcomer", _FixedDistribution("g1f3"))
    weights = ensemble.set_weights_from_ledger(ledger)
    assert weights["newcomer"] > 0.0
    assert sum(weights.values()) == pytest.approx(1.0)


def test_an_empty_ledger_leaves_the_ensemble_uniform():
    ensemble = EnsemblePolicy(seed=0)
    ensemble.add("a", _FixedDistribution("e2e4"))
    ensemble.add("b", _FixedDistribution("d2d4"))
    weights = ensemble.set_weights_from_ledger(OutcomeLedger())
    assert weights["a"] == pytest.approx(0.5)
    assert weights["b"] == pytest.approx(0.5)


def test_reweighting_leaves_the_distribution_valid():
    ledger = _ledger_with({"a": (9, 10), "b": (5, 10), "c": (1, 10)})
    ensemble = EnsemblePolicy(seed=0)
    ensemble.add("a", _FixedDistribution("e2e4"))
    ensemble.add("b", _FixedDistribution("d2d4"))
    ensemble.add("c", _FixedMove("g1f3"))
    ensemble.set_weights_from_ledger(ledger)
    dist = ensemble.move_distribution(chess.Board(START_FEN))
    assert dist.sum() == pytest.approx(1.0)
    assert np.all(dist >= 0.0)


# --------------------------------------------------------------------------
# 8. persistence
# --------------------------------------------------------------------------

def test_state_dict_carries_the_weights_and_the_roster():
    ensemble = EnsemblePolicy(seed=0)
    ensemble.add("a", _FixedDistribution("e2e4"), weight=0.7)
    state = ensemble.state_dict()
    assert state["weights"] == {"a": 0.7}
    assert state["members"][0]["name"] == "a"
    assert state["name"] == "L6-ensemble"


def test_state_dict_does_not_duplicate_member_parameters():
    """Members own their own checkpoints; copying them here would drift."""
    ensemble = EnsemblePolicy(seed=0)
    ensemble.add("a", _FixedDistribution("e2e4"))
    state = ensemble.state_dict()
    assert set(state) == {"name", "members", "weights", "config"}


def test_weights_round_trip_through_a_file(tmp_path: Path):
    ensemble = EnsemblePolicy(seed=0)
    ensemble.add("a", _FixedDistribution("e2e4"), weight=0.25)
    ensemble.add("b", _FixedDistribution("d2d4"), weight=0.75)
    path = tmp_path / "ensemble.json"
    ensemble.save(path)

    fresh = EnsemblePolicy(seed=0)
    fresh.add("a", _FixedDistribution("e2e4"))
    fresh.add("b", _FixedDistribution("d2d4"))
    fresh.load_weights(path)
    assert [m.weight for m in fresh.members] == [0.25, 0.75]


def test_weights_are_matched_by_name_not_by_position(tmp_path: Path):
    """The property that makes the roster safe to extend.

    The saved ensemble has a, b. The loading ensemble has an extra member
    inserted *first*. Positional matching would give the wrong weight to every
    member from that point on; name matching keeps each one's own.
    """
    ensemble = EnsemblePolicy(seed=0)
    ensemble.add("a", _FixedDistribution("e2e4"), weight=0.3)
    ensemble.add("b", _FixedDistribution("d2d4"), weight=0.7)
    path = tmp_path / "ensemble.json"
    ensemble.save(path)

    extended = EnsemblePolicy(seed=0)
    extended.add("inserted", _FixedDistribution("g1f3"), weight=1.0)
    extended.add("a", _FixedDistribution("e2e4"))
    extended.add("b", _FixedDistribution("d2d4"))
    extended.load_weights(path)

    by_name = {m.name: m.weight for m in extended.members}
    assert by_name["a"] == pytest.approx(0.3)
    assert by_name["b"] == pytest.approx(0.7)
    assert by_name["inserted"] == 1.0, "an unknown member must keep its own weight"


def test_loading_a_missing_member_is_not_an_error(tmp_path: Path):
    ensemble = EnsemblePolicy(seed=0)
    ensemble.add("a", _FixedDistribution("e2e4"), weight=0.4)
    path = tmp_path / "ensemble.json"
    ensemble.save(path)

    fresh = EnsemblePolicy(seed=0)
    fresh.add("a", _FixedDistribution("e2e4"))
    fresh.add("brand_new", _FixedDistribution("d2d4"), weight=0.9)
    fresh.load_weights(path)
    by_name = {m.name: m.weight for m in fresh.members}
    assert by_name["a"] == pytest.approx(0.4)
    assert by_name["brand_new"] == pytest.approx(0.9)


def test_saved_state_is_readable_json(tmp_path: Path):
    path = tmp_path / "ensemble.json"
    ensemble = EnsemblePolicy(seed=0)
    ensemble.add("a", _FixedDistribution("e2e4"), weight=0.6)
    ensemble.save(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["weights"]["a"] == pytest.approx(0.6)


# --------------------------------------------------------------------------
# 9. the ensemble plays chess
# --------------------------------------------------------------------------

def test_the_ensemble_plays_a_full_legal_game_as_white():
    ensemble = EnsemblePolicy(seed=0)
    ensemble.add("a", _FixedDistribution("e2e4", spread=0.2))
    ensemble.add("b", _FixedMove("d2d4"))

    class _Sweeper:
        """Plays legal moves in a fixed order, so the game terminates fast."""

        def __init__(self):
            self.i = 0

        def select(self, board):
            moves = sorted(board.legal_moves, key=lambda m: m.uci())
            move = moves[self.i % len(moves)]
            self.i += 1
            return move

    result = play_game(ensemble, _Sweeper(), max_plies=60,
                       rng=np.random.default_rng(1))
    assert result.plies > 0
    assert result.fens
    # Every position in the record must be reachable from the start, which is
    # what "played a legal game" means operationally.
    board = chess.Board(START_FEN)
    for san in result.moves:
        board.push(board.parse_san(san))
    assert board.fen() == result.fens[-1] or not result.is_finished


def test_the_ensemble_plays_a_full_legal_game_as_black():
    ensemble = EnsemblePolicy(seed=0)
    ensemble.add("a", _FixedDistribution("e2e4", spread=0.2))
    ensemble.add("b", _FixedMove("d2d4"))

    class _Sweeper:
        def __init__(self):
            self.i = 0

        def select(self, board):
            moves = sorted(board.legal_moves, key=lambda m: m.uci())
            move = moves[self.i % len(moves)]
            self.i += 1
            return move

    result = play_game(_Sweeper(), ensemble, max_plies=60,
                       rng=np.random.default_rng(2))
    assert result.plies > 0


def test_the_ensemble_is_deterministic_under_a_fixed_seed():
    def run() -> str:
        ensemble = EnsemblePolicy(seed=13)
        ensemble.add("a", _FixedDistribution("e2e4"))
        ensemble.add("b", _FixedDistribution("d2d4"))
        board = chess.Board(START_FEN)
        return " ".join(ensemble.select(board).uci() for _ in range(40))

    assert run() == run()


# --------------------------------------------------------------------------
# 10. scoring a played game into the ledger
# --------------------------------------------------------------------------

class _GameResult:
    """A minimal stand-in for ``GameResult``, so the scoring rule is testable
    without playing a real game for every branch."""

    def __init__(self, result: str, moves: list[str], fens: list[str]):
        self.result = result
        self.moves = moves
        self.fens = fens

    @property
    def is_decisive(self) -> bool:
        return self.result in ("1-0", "0-1")

    @property
    def is_finished(self) -> bool:
        return self.result != "unfinished"


def _two_move_game() -> _GameResult:
    """1. e4 e5, White to win. Two scored positions, one per side.

    The game records three FENs (start, after e4, after e5) but scoring drops
    the final one -- there is no move left to predict in the terminal position.
    That leaves two observable plies: the start position, where White is to move
    and played e4, and the position after e4, where Black is to move and played
    e5. Both sides therefore contribute evidence, which is the point of scoring
    per position rather than per game.
    """
    board = chess.Board(START_FEN)
    fens = [board.fen()]
    for san in ("e4", "e5"):
        board.push(board.parse_san(san))
        fens.append(board.fen())
    return _GameResult("1-0", ["e4", "e5"], fens)


def test_scoring_counts_one_observation_per_non_terminal_position():
    ledger = OutcomeLedger()
    member = _FixedDistribution("e2e4")
    scored = score_member_opinions(ledger, member, _two_move_game(), name="m")
    assert scored == 2, "three FENs minus the terminal one is two observations"
    assert ledger.get("m").predicted == 2
    assert len(_two_move_game().fens) == 3


def test_scoring_records_a_hit_when_the_member_agrees_with_the_winner():
    """White played e4 and won, so a member that would also play e4 is right
    about White's ply."""
    ledger = OutcomeLedger()
    score_member_opinions(ledger, _FixedDistribution("e2e4"),
                          _two_move_game(), name="m")
    assert ledger.get("m").correct >= 1


def test_agreeing_with_the_winner_scores_better_than_disagreeing():
    """The ordering the scoring rule exists to produce, on one real game.

    A member that would play the winning side's own move must score better than
    one that would play something else. This is the property that makes the
    ledger informative; the absolute counts depend on which side is to move in
    each recorded position, so the test asserts the comparison, not the totals.
    """
    agreeing = OutcomeLedger()
    disagreeing = OutcomeLedger()
    score_member_opinions(agreeing, _FixedDistribution("e2e4"),
                          _two_move_game(), name="agree")
    score_member_opinions(disagreeing, _FixedDistribution("g1f3"),
                          _two_move_game(), name="disagree")
    assert agreeing.get("agree").accuracy > disagreeing.get("disagree").accuracy


def test_the_ledger_keeps_accumulating_across_games():
    ledger = OutcomeLedger()
    member = _FixedDistribution("e2e4")
    for _ in range(3):
        score_member_opinions(ledger, member, _two_move_game(), name="m")
    assert ledger.get("m").predicted == 6


def test_max_positions_caps_the_work():
    ledger = OutcomeLedger()
    score_member_opinions(ledger, _FixedDistribution("e2e4"),
                          _two_move_game(), name="m", max_positions=1)
    assert ledger.get("m").predicted == 1


def test_a_game_with_no_recorded_positions_is_a_no_op():
    ledger = OutcomeLedger()
    empty = _GameResult("1-0", [], [])
    assert score_member_opinions(ledger, _FixedDistribution("e2e4"),
                                 empty, name="m") == 0
    assert ledger.get("m").predicted == 0


def test_a_broken_member_does_not_kill_the_ledger():
    """One member raising mid-game must not lose the other members' evidence."""
    class _Exploding:
        def select(self, board):
            raise RuntimeError("boom")

        def move_distribution(self, board):
            raise RuntimeError("boom")

    ledger = OutcomeLedger()
    scored = score_member_opinions(ledger, _Exploding(),
                                   _two_move_game(), name="broken")
    assert scored == 0
    assert ledger.get("broken").predicted == 0


def test_an_unfinished_game_scores_on_safety_not_on_agreement():
    """There is no winner to agree with, so the rule must change.

    This is the branch that makes the ledger usable on this benchmark at all:
    the majority of games are ply-capped, and dropping them would leave almost
    no evidence.
    """
    board = chess.Board(START_FEN)
    fens = [board.fen()]
    for san in ("e4", "e5"):
        board.push(board.parse_san(san))
        fens.append(board.fen())
    unfinished = _GameResult("unfinished", ["e4", "e5"], fens)

    ledger = OutcomeLedger()
    scored = score_member_opinions(ledger, _FixedDistribution("e2e4"),
                                   unfinished, name="m")
    assert scored == 2
    assert ledger.get("m").predicted == 2


def test_scoring_a_quiet_move_in_an_unfinished_game_counts_as_safe():
    """e4 hangs nothing, so it must score as a correct prediction."""
    board = chess.Board(START_FEN)
    fens = [board.fen()]
    for san in ("e4", "e5"):
        board.push(board.parse_san(san))
        fens.append(board.fen())
    unfinished = _GameResult("unfinished", ["e4", "e5"], fens)

    ledger = OutcomeLedger()
    score_member_opinions(ledger, _FixedDistribution("e2e4"), unfinished, name="m")
    assert ledger.get("m").correct == 2


def test_scoring_ignores_positions_where_the_game_is_already_over():
    """A position after the final move has no legal continuation to be right about."""
    board = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
    result = _GameResult("0-1", [], [board.fen()])
    ledger = OutcomeLedger()
    assert score_member_opinions(ledger, _FixedDistribution("e2e4"),
                                 result, name="m") == 0


# --------------------------------------------------------------------------
# 10b. regression: the scoring rule must not hand out free hits
# --------------------------------------------------------------------------

def _scholars_mate() -> _GameResult:
    """A short decisive game White wins, for the scoring-rule tests.

    1. e4 e5 2. Bc4 Nc6 3. Qh5 Nf6 4. Qxf7#. Seven moves and eight recorded
    FENs, of which the scorer walks the first seven -- the last is the checkmate
    position and has no move left to predict. Three of the seven are Black's,
    which is what makes this the right fixture for checking that the loser's
    plies are not scored as free hits.
    """
    board = chess.Board(START_FEN)
    fens = [board.fen()]
    moves = []
    for san in ("e4", "e5", "Bc4", "Nc6", "Qh5", "Nf6", "Qxf7#"):
        board.push(board.parse_san(san))
        moves.append(san)
        fens.append(board.fen())
    return _GameResult("1-0", moves, fens)


class _RandomMember:
    """Picks a uniformly random legal move. The honest baseline for scoring."""

    def __init__(self, seed: int = 0):
        import random

        self.rng = random.Random(seed)

    def select(self, board: chess.Board) -> chess.Move:
        return self.rng.choice(list(board.legal_moves))


def test_a_random_member_scores_at_chance_on_a_decisive_game():
    """The regression this exists for: the loser's plies must not be free hits.

    An earlier scoring rule gave a hit for *disagreeing* with the losing side's
    move, on the reasoning that agreeing with a losing move is not a correct
    prediction. That inverts the question -- the member is asked what it would
    play, not whether the played move was good -- and it made every ply the
    loser moved a guaranteed hit for any member that plays something else.
    Measured, a purely random member scored **0.429** on this game.

    A random mover matches a specific one of ~20-30 legal moves, so its score
    must sit near 1/25, not near 1/2. The bound is generous (0.20) because the
    point is to reject the *category* of bug -- an order-of-magnitude error --
    not to pin the exact rate.
    """
    result = _scholars_mate()
    accuracies = []
    for seed in range(30):
        ledger = OutcomeLedger()
        score_member_opinions(ledger, _RandomMember(seed), result, name="r")
        accuracies.append(ledger.get("r").accuracy)
    mean = sum(accuracies) / len(accuracies)
    assert mean < 0.20, (
        f"a random member scored {mean:.3f} on a decisive game; near chance is "
        f"expected. This is the free-hit bug returning."
    )


def test_the_loser_side_is_scored_like_every_other_side():
    """A member that plays the actually-played move must be right on *any* ply.

    Stated as an identity rather than a rate, so it cannot be satisfied by
    accident: an oracle that always plays the move that is about to be played
    must score 1.0, regardless of which side is to move.
    """
    result = _scholars_mate()

    class _Oracle:
        """Plays the move that the game record says was played here.

        ``board.ply()`` is the index into the move list because the recorded
        positions start from the initial position, so the ply counter and the
        move index coincide.
        """

        def select(self, board: chess.Board) -> chess.Move:
            return board.parse_san(result.moves[board.ply()])

    ledger = OutcomeLedger()
    score_member_opinions(ledger, _Oracle(), result, name="oracle")
    entry = ledger.get("oracle")
    # Seven moves means eight recorded FENs; the scorer drops the last (the
    # checkmate position, which has no move left to predict) and walks the other
    # seven. All seven are scored, because every one of them still has a move to
    # make -- including the three where Black is to move.
    assert entry.predicted == 7
    assert entry.correct == 7, (
        "a member that plays exactly the played move must score every ply; "
        "if the loser's plies are scored inverted, this cannot hold"
    )


def test_a_member_that_never_matches_scores_zero_on_a_decisive_game():
    """The other end of the identity: maximal disagreement scores nothing.

    Before the fix, a member that never matched still scored ~half, because
    half the plies (the loser's) rewarded disagreement.
    """
    result = _scholars_mate()

    class _NeverMatches:
        """Plays the first legal move that is not the one that was played."""

        def select(self, board: chess.Board) -> chess.Move:
            target = result.moves[board.ply()] if board.ply() < len(result.moves) else None
            for move in sorted(board.legal_moves, key=lambda m: m.uci()):
                if board.san(move) != target:
                    return move
            return sorted(board.legal_moves, key=lambda m: m.uci())[0]

    ledger = OutcomeLedger()
    score_member_opinions(ledger, _NeverMatches(), result, name="never")
    assert ledger.get("never").correct == 0


# --------------------------------------------------------------------------
# 10c. regression: the safety proxy must actually discriminate
# --------------------------------------------------------------------------

def test_the_safety_proxy_rejects_some_moves():
    """The regression this exists for: ``_move_is_safe`` used to return True always.

    Its original form compared ``evaluate()`` before and after the move against a
    threshold of -100. Because ``evaluate`` is material + piece-square only, it
    cannot see the opponent's reply, so the swing it measures is just the mover's
    own material change: over 11,691 sampled moves the minimum observed swing was
    **-50**, never reaching -100. The function returned True for **100%** of
    moves, including 200/200 random ones, so every unfinished game gave every
    member a perfect score -- fake evidence.

    The assertion is deliberately weak in form (some moves are rejected) because
    the bug it guards against is total: a discriminator that never says no. It
    does not pin a particular rate.
    """
    import random

    from chessrl.ensemble import _move_is_safe

    rng = random.Random(0)
    rejected = total = 0
    for _ in range(120):
        board = chess.Board()
        for _ in range(rng.randint(2, 40)):
            if board.is_game_over():
                break
            board.push(rng.choice(list(board.legal_moves)))
        if board.is_game_over():
            continue
        for move in board.legal_moves:
            total += 1
            if not _move_is_safe(board, move):
                rejected += 1
    assert total > 500, "not enough sampled moves for this test to mean anything"
    assert rejected > 0, (
        "the safety proxy rejected no move out of "
        f"{total}; it is not discriminating at all, which is the dead-code bug "
        "returning (it used to accept 100% of moves)"
    )


def test_the_safety_proxy_accepts_a_quiet_developing_move():
    """The other half of the discriminator: it must not reject everything.

    A proxy that rejects all moves would pass the test above and be equally
    useless.
    """
    from chessrl.ensemble import _move_is_safe

    board = chess.Board(START_FEN)
    # e4 attacks nothing and is not attacked on arrival -- nothing can capture
    # on e4 next ply.
    assert _move_is_safe(board, chess.Move.from_uci("e2e4")) is True


def test_the_safety_proxy_rejects_a_move_into_a_free_capture():
    """A knight moved in front of an enemy pawn must be judged unsafe.

    The position has a black pawn on h6 and a white knight on f3, so Nf3g5
    walks the knight onto a square an enemy pawn attacks with nothing to win in
    return -- the clearest "hung a piece" move available in a simple position.
    """
    from chessrl.ensemble import _move_is_safe

    board = chess.Board("rnbqkbnr/pppppp2/7p/8/8/5N2/PPPPPPPP/RNBQKB1R w KQkq - 0 1")
    blunder = chess.Move.from_uci("f3g5")
    assert blunder in board.legal_moves
    assert _move_is_safe(board, blunder) is False

    # The same knight has quiet squares that pawn cannot reach.
    assert _move_is_safe(board, chess.Move.from_uci("f3e5")) is True
    assert _move_is_safe(board, chess.Move.from_uci("f3h4")) is True


def test_the_safety_proxy_is_blind_in_the_opening():
    """A limitation to pin, not a bug to fix: in the opening it rejects nothing.

    ``recapture_risk`` needs something to be attackable. From the start position
    no enemy piece can capture on any destination, so every move is "safe" and
    the fallback collapses back to the always-True behaviour it was replacing --
    for exactly the handful of positions at the top of a game.

    Measured acceptance over random positions: 0.0% rejected in the opening,
    13.0% at 20 plies, 15.7% at 40 plies. That is why the README says the
    headline result leans on unfinished games and should be trusted less than
    its weight suggests. Pinned here so the phase-dependence is a known property
    rather than a discovery.
    """
    from chessrl.ensemble import _move_is_safe

    board = chess.Board(START_FEN)
    accepted = sum(1 for move in board.legal_moves if _move_is_safe(board, move))
    assert accepted == board.legal_moves.count(), (
        "the start position has nothing attackable, so the safety proxy should "
        "accept every move; if it now rejects some, its semantics changed and "
        "the README's phase table is stale"
    )

    # And it does reject in a developed position, so the blindness is specific
    # to the opening rather than a property of the function.
    midgame = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 4 4")
    rejected = sum(1 for move in midgame.legal_moves
                   if not _move_is_safe(midgame, move))
    assert rejected > 0, (
        "a developed position must offer at least one unsafe move, else the "
        "fallback is not discriminating outside the opening either"
    )


def test_unfinished_games_still_produce_discriminating_evidence():
    """The property the safety proxy exists for, end to end.

    Unfinished games are the majority in this benchmark, so if their branch
    gives every member the same score the ledger carries no signal at all. This
    asserts a competent pair of policies and a random one do not all score
    identically.
    """
    import random

    from chessrl.game import play_game
    from chessrl.value import evaluate

    class _Greedy:
        def select(self, board):
            best, best_value = None, -1e9
            for move in board.legal_moves:
                board.push(move)
                value = -evaluate(board)
                board.pop()
                if value > best_value:
                    best_value, best = value, move
            return best

    from chessrl.game import RandomPolicy
    result = play_game(RandomPolicy(), RandomPolicy(), max_plies=60,
                       rng=random.Random(1))
    assert not result.is_finished, "fixture must be an unfinished game"

    random_ledger, greedy_ledger = OutcomeLedger(), OutcomeLedger()
    score_member_opinions(random_ledger, _RandomMember(0), result, name="r")
    score_member_opinions(greedy_ledger, _Greedy(), result, name="g")

    random_acc = random_ledger.get("r").accuracy
    greedy_acc = greedy_ledger.get("g").accuracy
    assert not (random_acc == 1.0 and greedy_acc == 1.0), (
        "both a random and a greedy member scored a perfect 1.0 on an "
        "unfinished game, which is the always-True safety proxy returning"
    )
    assert greedy_acc >= random_acc, (
        "greedy play should be at least as safe as random play"
    )


def test_scoring_a_real_game_produces_evidence_for_every_member():
    """End to end, with a real result from a real game."""
    from chessrl.game import play_game as _play

    class _Sweeper:
        def __init__(self):
            self.i = 0

        def select(self, board):
            moves = sorted(board.legal_moves, key=lambda m: m.uci())
            move = moves[self.i % len(moves)]
            self.i += 1
            return move

    result = _play(_Sweeper(), _Sweeper(), max_plies=40,
                   rng=np.random.default_rng(3))
    ledger = OutcomeLedger()
    scored = score_member_opinions(ledger, _FixedDistribution("e2e4"),
                                   result, name="m", max_positions=10)
    assert scored > 0, "a real game must yield at least one observation"
    assert ledger.get("m").predicted == scored
    assert 0.0 <= ledger.accuracies()["m"] <= 1.0


# --------------------------------------------------------------------------
# 11. the module must not drag torch in
# --------------------------------------------------------------------------

def test_the_ensemble_module_imports_without_torch():
    """L6 must stay importable on a machine with no torch.

    L4 and L5 are the only levels allowed to need it. If this module ever grows
    a top-level torch import, the whole bench becomes un-runnable for anyone who
    has only the core dependencies -- which is the documented contract.
    """
    import importlib
    import sys

    saved = sys.modules.pop("chessrl.ensemble", None)
    torch_before = sys.modules.get("torch")
    try:
        sys.modules["torch"] = None            # poison any import attempt
        module = importlib.import_module("chessrl.ensemble")
        assert module is not None
    finally:
        if saved is not None:
            sys.modules["chessrl.ensemble"] = saved
        if torch_before is not None:
            sys.modules["torch"] = torch_before
        else:
            sys.modules.pop("torch", None)


# --------------------------------------------------------------------------
# 12. config defaults are the safe ones
# --------------------------------------------------------------------------

def test_the_default_floor_keeps_members_alive():
    assert EnsembleConfig().floor > 0.0


def test_the_default_config_is_usable_with_no_evidence():
    """A freshly built ensemble must be able to move before it has a ledger."""
    ensemble = EnsemblePolicy()
    ensemble.add("a", _FixedDistribution("e2e4"))
    ensemble.add("b", _FixedDistribution("d2d4"))
    dist = ensemble.move_distribution(chess.Board(START_FEN))
    assert dist.sum() == pytest.approx(1.0)
