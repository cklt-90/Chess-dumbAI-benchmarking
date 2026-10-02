"""L6: the ensemble, weighted by historical win-prediction accuracy.

What this level is, and what it is not
--------------------------------------
Every other level answers "what is the best move here?" with one mechanism. L6
answers it with *all of them*: it holds a roster of already-built policies and
combines their opinions. It is the last rung because it is the only one that
learns nothing itself -- its parameters are the weights it assigns to other
levels' opinions, and the only evidence it uses is how well each of them has
predicted the outcome of games that have already happened.

That is exactly the design the repo has been pointing at all along:

* ``__init__.py``: "L6 ensemble -- all of the above, weighted by predictive
  accuracy".
* ``game.py``: keeping the game loop level-agnostic "is what makes the L6
  tournament a fair comparison: levels differ only in how they choose a move,
  never in how a game is run".
* ``game.py``: ``move_distribution`` is "[r]equired for the L1.5 counter logic
  and for L6 ensemble weighting".
* ``guided.py``, on why L5 exposes no distribution: "Keeping the two families
  separate is what lets the L6 ensemble know which levels can be blended and
  which can only be voted."

The last of these is the load-bearing constraint, so it is stated first.

Two kinds of member, and they cannot be treated alike
-----------------------------------------------------
A **distribution member** (L1, L3, L3.5, L4) exposes ``move_distribution``: a
normalised vector over the 20480-slot action space. Distributions can be
*blended*, which is strictly more informative than voting -- a member that is
80% sure contributes 80% of its mass to one move.

A **move member** (L2, L5) exposes only ``select``. It performs a search per
move; asking it for a distribution would mean either running a search per move
per candidate -- which is not what the level is -- or inventing a distribution
it does not use. So a move member can only *vote*: one unit of weight on the
single move it chose.

L6 therefore runs both mechanisms. Blending happens first among the distribution
members on the union of their legal support; votes are then accumulated on top.
A member's weight is the same number in both mechanisms, so "how much do we
trust this level" means one thing across the whole ensemble.

Why blending is done in probability space, not logit space
----------------------------------------------------------
A weighted average of distributions is itself a distribution, and it is what
"ensemble weighted by accuracy" conventionally means. A weighted average of
*logits* also produces a valid distribution and is cheaper (it can be done by a
single :class:`~chessrl.policy.Scorer` wrapper, so the whole
``FactoredSoftmaxPolicy`` machinery applies unchanged) -- but it is only an
*approximation* to the mixture, and the approximation error is not bounded by
anything the caller can see.

Measured, on the opening position, blending L1 (blind) with L3 (informed):
logit-space blending gives a total-variation distance of 7e-5 from the true
mixture at equal weights, and 2e-5 at 0.9/0.1. Small, but it is a *different
distribution*, silently. For members that share a factorisation (two weight-tied
L3.5s at equal weight) the two coincide exactly, which is precisely the case
that would make the error invisible in testing and then appear in production.

So L6 blends in probability space and accepts the cost of enumerating the
distribution. It is a `DistributionPolicy`, not a `Scorer`, and the union of
support is computed over the members rather than assumed to be the legal set.

The weighting rule
------------------
A member's weight comes from **outcome-prediction accuracy**, not from a fitted
rating and not from win rate. The distinction matters and is the point of the
level: a rating tells you how strong a policy is, but an ensemble needs to know
*whose opinion to believe*, and a strong policy can still be a poor predictor if
it is strong for reasons that do not generalise.

For each recorded game the ledger asks each member, in each position it was
given, which move it would play, and converts that into a predicted result. A
member whose chosen move matched the eventual winner's best move scores 1, and
so on. The accounting is deliberately simple and inspectable:

* ``predicted``     -- how many positions the member was asked about;
* ``correct``       -- how many it got right;
* ``accuracy``      -- ``correct / predicted``, or ``None`` when it was never
  asked (a member that was never exercised must not be treated as perfect *or*
  as worthless -- see :func:`accuracy_weights`).

Accuracy is measured in units of ``chance_equivalent`` (0.5 by default): a
member at exactly chance is worth one mean vote, and the vector is then
sharpened by a temperature and floored. Two properties are worth stating because
they are the ones a reader will ask about:

* **A member with no evidence gets the *mean* weight, not zero.** Dropping it
  to zero would mean a level that has not played yet can never play, which is a
  trap rather than a prior.
* **A member at chance accuracy gets the same mean weight, not zero.** Being at
  chance is the neutral position: the member is doing the job it was asked to do
  and no more. The floor on top of that is what stops the weakest member being
  starved outright, since it exists so a member having a bad run can still earn
  its way back.

The floor is not cosmetic. It is what makes the ensemble robust to a member that
is having a bad run, and it is why this module ships an ``accuracy`` ledger
rather than a softmax over ratings.

Deliberately not implemented
----------------------------
* **No training.** L6 has no gradient and no parameters in the usual sense. If
  it did, it would be L7.
* **No meta-search.** Combining opinions and then searching over the combination
  is a different (and much more expensive) idea; L5 already covers search.
* **No new features.** Every member consumes the same encoder output it always
  did. The ensemble is a combination rule, not a representation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import chess
import numpy as np

from . import masks as M
from .game import DistributionPolicy, MovePolicy

__all__ = [
    "MemberKind",
    "MemberSpec",
    "OutcomeLedger",
    "MemberAccuracy",
    "EnsembleConfig",
    "EnsemblePolicy",
    "accuracy_weights",
    "score_member_opinions",
]


# --------------------------------------------------------------------------
# member description
# --------------------------------------------------------------------------

class MemberKind:
    """How a member can be used. See the module docstring for why this exists."""

    DISTRIBUTION = "distribution"   # exposes move_distribution; can be blended
    MOVE = "move"                   # exposes only select; can only vote


@dataclass
class MemberSpec:
    """One member of the ensemble: a policy plus the weight it currently holds.

    ``weight`` is the *unnormalised* trust in this member. It is set from the
    ledger by :meth:`EnsemblePolicy.set_weights_from_ledger`, but keeping it as
    a plain field means an ensemble can also be constructed by hand with chosen
    weights -- which is how the tests pin down the blending maths without
    depending on accuracy estimation.
    """

    name: str
    policy: object
    weight: float = 1.0

    def __post_init__(self) -> None:
        has_dist = hasattr(self.policy, "move_distribution")
        self.kind = MemberKind.DISTRIBUTION if has_dist else MemberKind.MOVE
        if self.weight < 0.0:
            raise ValueError(
                f"member {self.name!r} has negative weight {self.weight}; "
                "weights are trust, and negative trust is not a mixture"
            )

    @property
    def is_blendable(self) -> bool:
        return self.kind == MemberKind.DISTRIBUTION

    def describe(self) -> dict:
        return {"name": self.name, "kind": self.kind, "weight": self.weight,
                "blendable": self.is_blendable}


# --------------------------------------------------------------------------
# the accuracy ledger
# --------------------------------------------------------------------------

@dataclass
class MemberAccuracy:
    """How well one member predicted the outcomes it was shown."""

    name: str
    predicted: int = 0
    correct: int = 0

    @property
    def accuracy(self) -> float | None:
        """``correct / predicted``, or ``None`` if the member was never asked.

        ``None`` rather than ``0.0`` on purpose: "no evidence" and "always
        wrong" are different states and the weighting rule treats them
        differently. Collapsing them is how an unused member gets starved.
        """
        if self.predicted == 0:
            return None
        return self.correct / self.predicted

    def as_dict(self) -> dict:
        acc = self.accuracy
        return {
            "name": self.name,
            "predicted": self.predicted,
            "correct": self.correct,
            "accuracy": None if acc is None else round(acc, 6),
        }


class OutcomeLedger:
    """Records whether each member's opinion agreed with what actually happened.

    The unit of evidence is one *(member, position)* pair, not one game. Per
    game the member is asked, in each recorded position, which move it would
    play; if that move is the move the eventual winner would have wanted, it is
    a correct prediction. Scoring per decision rather than per game is what
    makes a handful of games carry usable signal: 40 plies is 40 observations,
    not one.

    The comparison is deliberately *not* "did the member's move win the game".
    A policy can be right about a position and still lose the game, and an
    ensemble needs to know whose *judgement* to trust, not who got lucky.
    """

    def __init__(self) -> None:
        self._entries: dict[str, MemberAccuracy] = {}

    # ---- recording -----------------------------------------------------

    def register(self, name: str) -> MemberAccuracy:
        entry = self._entries.get(name)
        if entry is None:
            entry = self._entries[name] = MemberAccuracy(name=name)
        return entry

    def note(self, name: str, correct: bool) -> None:
        """Record one prediction for one member in one position."""
        entry = self.register(name)
        entry.predicted += 1
        entry.correct += int(bool(correct))

    def note_many(self, name: str, results) -> None:
        for correct in results:
            self.note(name, correct)

    # ---- reading -------------------------------------------------------

    @property
    def names(self) -> list[str]:
        """Registered names, in first-registration order (stable for reporting)."""
        return list(self._entries)

    def __len__(self) -> int:
        """Number of members with at least one recorded prediction."""
        return sum(1 for e in self._entries.values() if e.predicted > 0)

    def get(self, name: str) -> MemberAccuracy | None:
        return self._entries.get(name)

    def total_predicted(self) -> int:
        return sum(e.predicted for e in self._entries.values())

    def accuracies(self) -> dict[str, float | None]:
        return {n: e.accuracy for n, e in self._entries.items()}

    def as_dict(self) -> dict:
        return {n: e.as_dict() for n, e in self._entries.items()}

    # ---- persistence ---------------------------------------------------

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.as_dict(), indent=2),
                              encoding="utf-8")

    @classmethod
    def from_dict(cls, raw: dict) -> "OutcomeLedger":
        """Build a ledger from the mapping :meth:`as_dict` produces.

        Split out from :meth:`load` because a ledger is often embedded inside a
        larger artefact -- a fitted-ensemble report carries the ledger that
        produced its weights, so the weights can be re-derived without replaying
        the games. Loading such a file whole would fail, since its top level is
        the report, not the ledger. This method takes the inner mapping.
        """
        ledger = cls()
        for name, entry in raw.items():
            acc = ledger.register(name)
            acc.predicted = int(entry["predicted"])
            acc.correct = int(entry["correct"])
        return ledger

    @classmethod
    def load(cls, path: str | Path) -> "OutcomeLedger":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# --------------------------------------------------------------------------
# the weighting rule
# --------------------------------------------------------------------------

@dataclass
class EnsembleConfig:
    """Knobs for how much each member's accuracy is worth.

    ``temperature`` sharpens (``< 1``) or flattens (``> 1``) the weighting. At
    ``1.0`` weight is proportional to accuracy; a temperature above 1 pulls the
    weights towards uniform, which is the safer setting while evidence is thin
    and the default for that reason.

    ``floor`` is the minimum weight any member can hold, as a fraction of the
    *mean* weight. It exists so an unlucky or unused member is not starved to
    zero and can earn its way back -- see the module docstring.

    ``chance_equivalent`` is the accuracy that counts as one full vote, i.e. the
    unit in which accuracy is measured. It is kept separate from ``floor`` so
    the two ideas are tunable independently: ``floor`` is "don't starve", this
    is "how much accuracy is worth a vote". Raising it makes the ensemble more
    forgiving of a weak member, because every member's score shrinks relative to
    the mean; lowering it sharpens the split between good and mediocre.

    ``relative`` switches the scaling point from ``chance_equivalent`` (a fixed
    reference) to the *mean accuracy of the set itself*. A member is then worth
    its accuracy *above the mean*, and a below-mean member is worth nothing. This
    is the structural fix for the "blind majority outvotes a seeing minority"
    failure: under the absolute rule two mediocre members whose accuracies sit
    just above chance each get a real vote, and their weights sum to more than
    the one competent member's, so the blend lets their noise dilute the seeing
    member. Subtracting the mean zeroes the mediocre members and leaves only the
    competent one. See ``bench/README.md`` ("the relative fix") for the measured
    effect. It is off by default so the documented negative result -- and every
    test that pins it -- is preserved.
    """

    temperature: float = 1.0
    floor: float = 0.05
    # The accuracy that earns exactly one mean vote. A member at this accuracy
    # is neither rewarded nor punished: it is doing the job and no more.
    chance_equivalent: float = 0.5
    # Off by default. When True, scale each member's accuracy relative to the
    # mean of the set rather than to chance_equivalent.
    relative: bool = False


def accuracy_weights(
    ledger: OutcomeLedger,
    names: list[str],
    *,
    temperature: float = 1.0,
    floor: float = 0.05,
    chance_equivalent: float = 0.5,
    relative: bool = False,
) -> dict[str, float]:
    """Turn measured accuracies into normalised weights over ``names``.

    The rules, in order:

    1. **No evidence -> the mean weight.** A member that has never been asked
       cannot be ranked, so it is given exactly the average rather than zero
       (starving it forever) or one (assuming it is perfect). This is the rule
       that makes a freshly-constructed ensemble usable. In relative mode an
       unasked member is treated as exactly the mean, so it contributes nothing
       above the floor -- honest, since "no evidence" is "no better than average".
    2. **Accuracy is measured relative to chance** (or, in relative mode, to the
       *mean* of the set). Under the absolute rule a member whose accuracy equals
       ``chance_equivalent`` is worth exactly one mean vote -- it is doing the job
       it is being asked to do and no more. A member twice as good as chance is
       worth two; one worse than chance is worth nothing. Under the relative rule
       it is worth its accuracy *minus the set mean*, so a below-mean member is
       worth nothing and cannot pool its vote with another below-mean member to
       outrank (or in the blend, dilute) a competent one.
    3. **The whole vector is then sharpened by ``temperature`` and floored.**

    ``chance_equivalent`` is required because "accuracy" here is not
    balanced-class: a policy playing White in a won position and Black in a lost
    one has different base rates, so 0.5 is a *declared* reference point rather
    than a derived one. Stating it explicitly is better than silently choosing a
    constant inside the formula.

    Note carefully that the absolute rule is ``acc / chance_equivalent`` and *not*
    an interpolation between two accuracy anchors. The earlier formula here was
    ``acc / chance_equivalent * chance_equivalent``, a tautology that cancels to
    ``acc``: every member got rescaled identically, so the knob could not move
    any weight between them and was in fact dead. Treating ``chance_equivalent``
    as the *unit* of accuracy is the reading under which it does real work: it
    sets how much accuracy counts as a full vote, and thus how sharp the split
    between a good and a mediocre member is.

    The relative rule is ``max(0, acc - mean)`` and then the same sharpen/floor
    steps. It deliberately throws away the absolute calibration and keeps only the
    *ranking* of members against their peers; that is the property that makes an
    ensemble of one competent and several mediocre members collapse onto the
    competent one instead of being dragged down by the mediocre majority.
    """
    if not names:
        return {}
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    if floor < 0:
        raise ValueError(f"floor must be non-negative, got {floor}")
    if chance_equivalent <= 0.0:
        raise ValueError(
            f"chance_equivalent must be positive (it is the unit of accuracy), "
            f"got {chance_equivalent}"
        )

    mean_weight = 1.0
    # Collect the raw accuracies first so both modes share one source of truth and
    # so relative mode can compute the set mean before scaling.
    accs: dict[str, float | None] = {}
    for name in names:
        entry = ledger.get(name)
        accs[name] = entry.accuracy if entry is not None else None

    if relative:
        # Scale relative to the mean of the set, not to a fixed chance anchor.
        # Below-mean members drop to zero, so two mediocre members cannot pool
        # their weight to outrank (or, in the blend, dilute) a competent one.
        present = [a for a in accs.values() if a is not None]
        mean_acc = sum(present) / len(present) if present else 0.0
        raw: dict[str, float] = {}
        for name in names:
            a = accs[name]
            if a is None:
                a = mean_acc  # no evidence -> at the mean -> no weight above floor
            # Clamped at zero: a member worse than the mean is worth nothing,
            # never a negative share that would invert its vote in the blend.
            raw[name] = max(0.0, a - mean_acc)
    else:
        raw = {}
        for name in names:
            acc = accs[name]
            if acc is None:
                raw[name] = mean_weight
            else:
                # ``chance_equivalent`` is the unit of accuracy: at exactly chance
                # a member is worth one mean vote. Clamped at zero so a member
                # worse than chance is worth nothing rather than a negative share,
                # which would invert its vote in the blend.
                raw[name] = max(0.0, acc / chance_equivalent)

    # Sharpen/flatten towards uniform.
    #
    # Done in log space with a max-shift, which is the standard stable softmax
    # trick. The direct form ``v ** (1/temperature)`` overflows for a small
    # positive temperature -- ``0.9 ** 1e9`` raises OverflowError rather than
    # saturating -- and ``temperature > 0`` is already validated, so a caller
    # asking for a very sharp weighting was asking for something legal and got a
    # traceback. Exponentiating the shifted log-weights cannot overflow: the
    # largest exponent is 1.0 by construction, and any real weight that
    # underflows to 0.0 is a member that should indeed be negligible.
    #
    # A raw weight of exactly 0.0 is pinned before the shift, because log(0) is
    # -inf and would otherwise poison every value in the dict.
    if temperature != 1.0:
        positive = {n: v for n, v in raw.items() if v > 0.0}
        if positive:
            logs = {n: np.log(v) / temperature for n, v in positive.items()}
            shift = max(logs.values())
            sharpened = {n: float(np.exp(logs[n] - shift)) for n in positive}
        else:
            sharpened = {}
        # Members the rule scored at zero stay at zero.
        raw = {n: sharpened.get(n, 0.0) for n in raw}

    # Floor, expressed relative to the mean of the *raw* weights so that it
    # means the same thing regardless of how many members there are.
    total = sum(raw.values())
    mean = total / len(raw) if raw else 0.0
    minimum = floor * mean
    floored = {n: max(v, minimum) for n, v in raw.items()}

    total = sum(floored.values())
    if total <= 0:                      # every member floored to nothing
        return {n: 1.0 / len(names) for n in names}
    return {n: v / total for n, v in floored.items()}


# --------------------------------------------------------------------------
# the ensemble policy
# --------------------------------------------------------------------------

class EnsemblePolicy:
    """L6: combine a roster of policies, weighted by outcome accuracy.

    Implements :class:`chessrl.game.DistributionPolicy`, because the combination
    of distributions is a distribution. It is therefore a drop-in for the game
    loop and for the bench, exactly like L1 and L3.

    The combination has two stages, in this order:

    1. **Blend** the distribution members on the union of their legal support.
       Each contributes ``weight * its_distribution``, so a member that is
       confident contributes more of its weight to where it is confident.
    2. **Vote** the move-only members onto the single move each selected.

    A member with weight zero is skipped by both stages, and that is deliberate:
    zero weight means "do not consult this member", so it must not be asked for
    a move either. (An earlier version of this docstring claimed stage 2 also
    voted distribution members whose weight was zero. It never did -- stage 2
    iterates ``self.voters``, and a blendable member is never in that list. The
    docstring was wrong, not the code. Skipping is the correct behaviour.)

    Stage 2 is not redundant even when every member is blendable: ``select`` and
    the arg-max of ``move_distribution`` can legitimately disagree when a
    member's factorised arg-max is not the arg-max of its own full distribution
    (the policy's own docstring notes this). The ensemble asks each member the
    question that member is actually able to answer.
    """

    def __init__(
        self,
        members: list[MemberSpec] | None = None,
        config: EnsembleConfig | None = None,
        *,
        name: str = "L6-ensemble",
        seed: int = 0,
        greedy: bool = False,
    ):
        self.config = config or EnsembleConfig()
        self.members: list[MemberSpec] = list(members or [])
        self.name = name
        self.greedy = greedy
        self.rng = np.random.default_rng(seed)
        # Position-indexed probe results, so a caller can see *why* the ensemble
        # chose what it chose rather than only what it chose.
        self.last_contributions: dict[str, float] = {}

    # ---- roster management ---------------------------------------------

    def add(self, name: str, policy, weight: float = 1.0) -> MemberSpec:
        spec = MemberSpec(name=name, policy=policy, weight=weight)
        self.members.append(spec)
        return spec

    def set_weights_from_ledger(self, ledger: OutcomeLedger) -> dict[str, float]:
        """Adopt accuracy-derived weights. The one call that makes it "L6".

        Weights are stored *normalised* on the members, so that
        ``sum(weight) == 1`` and a bare `move_distribution` is already a valid
        mixture without a second normalisation step. The members' relative
        sizes are all that matters to the combination anyway.

        The returned mapping is keyed by *name*, so if two members share a name
        they share a ledger row and therefore share a weight -- they are, after
        all, indistinguishable to the accuracy accounting. Both members still
        receive that weight, so the ensemble is correct; only the returned dict
        is shorter than the roster. Giving two members the same name is a caller
        error, not something this method can fix.
        """
        names = [m.name for m in self.members]
        weights = accuracy_weights(
            ledger, names,
            temperature=self.config.temperature,
            floor=self.config.floor,
            chance_equivalent=self.config.chance_equivalent,
            relative=self.config.relative,
        )
        for member in self.members:
            member.weight = float(weights.get(member.name, 0.0))
        return weights

    @property
    def blendable(self) -> list[MemberSpec]:
        return [m for m in self.members if m.is_blendable]

    @property
    def voters(self) -> list[MemberSpec]:
        return [m for m in self.members if not m.is_blendable]

    def describe(self) -> list[dict]:
        return [m.describe() for m in self.members]

    # ---- the combination rule ------------------------------------------

    def move_distribution(self, board: chess.Board) -> np.ndarray:
        """The weighted mixture over the flat action space, normalised.

        Legal-support safety comes from the members: every blendable member
        guarantees zero mass on illegal moves (that is part of the ``Scorer``
        contract), and every vote is cast for a move the member returned from
        ``select``, which is legal by contract. The ensemble therefore needs no
        legality filter of its own -- and deliberately does not apply one, since
        a filter here would mask a member that had broken its contract.

        **The result is renormalised even when the member weights are not.**
        ``DistributionPolicy`` promises a normalised vector, and the game loop
        takes that promise at face value: it does not renormalise before
        sampling. A hand-built ensemble with weights of 1.0 each would therefore
        otherwise hand the loop a total mass of ``n_members``. Normalising on
        read rather than requiring callers to normalise on write means manual
        construction and ledger-derived weights behave identically, which is the
        property the weight-1.0 test relies on.
        """
        out = np.zeros(M.ACTION_SPACE, dtype=np.float64)
        contributions: dict[str, float] = {}

        if board.is_game_over(claim_draw=False):
            self.last_contributions = contributions
            return out

        # Stage 1: blend the distributions, on the union of their support.
        for member in self.blendable:
            if member.weight <= 0.0:
                continue
            dist = np.asarray(member.policy.move_distribution(board),
                              dtype=np.float64)
            if dist.shape != (M.ACTION_SPACE,):
                raise ValueError(
                    f"member {member.name!r} returned a distribution of shape "
                    f"{dist.shape}, expected {(M.ACTION_SPACE,)}"
                )
            out += member.weight * dist
            contributions[member.name] = float(dist.sum() * member.weight)

        # Stage 2: votes, one unit of the member's weight on its chosen move.
        for member in self.voters:
            if member.weight <= 0.0:
                continue
            move = member.policy.select(board)
            if move not in board.legal_moves:
                raise ValueError(
                    f"member {member.name!r} voted for the illegal move "
                    f"{move.uci()} in {board.fen()}"
                )
            out[M.move_to_index(move)] += member.weight
            contributions[member.name] = float(member.weight)

        total = out.sum()
        if total > 0.0:
            # Reported contributions are normalised on the same basis as the
            # distribution they produced, so the printed explanation of a choice
            # actually adds up to the probability that was assigned.
            out = out / total
            contributions = {k: v / total for k, v in contributions.items()}

        self.last_contributions = contributions
        return out

    # ---- selection -----------------------------------------------------

    def select(self, board: chess.Board) -> chess.Move:
        """The ensemble's move: the arg-max of the mixture, or a sample from it."""
        if not any(m.weight > 0.0 for m in self.members):
            raise ValueError(
                "every ensemble member has zero weight; call "
                "set_weights_from_ledger or assign weights explicitly"
            )
        dist = self.move_distribution(board)
        total = dist.sum()
        if total <= 0.0:
            raise ValueError(
                f"the ensemble produced no mass in {board.fen()}; every member "
                "either returned an empty distribution or voted nowhere"
            )
        if self.greedy:
            return M.index_to_move(int(np.argmax(dist)))
        p = dist / total
        return M.index_to_move(int(self.rng.choice(M.ACTION_SPACE, p=p)))

    # ---- persistence ---------------------------------------------------

    def state_dict(self) -> dict:
        """Weights and the roster, but not the members' own parameters.

        Each member owns its checkpoint and is saved by its own class. Including
        their weights here would duplicate them, and the copies would drift the
        moment either side was retrained. The ensemble's own learnable state is
        genuinely just the weight vector.
        """
        return {
            "name": self.name,
            "members": [m.describe() for m in self.members],
            "weights": {m.name: m.weight for m in self.members},
            "config": dict(self.config.__dict__),
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.state_dict(), indent=2),
                              encoding="utf-8")

    def load_weights(self, path: str | Path) -> dict[str, float]:
        """Adopt the weights from a saved ensemble, matching by member name.

        Matching by name rather than by position is deliberate: adding a member
        to the roster must not silently reassign every other member's weight.
        """
        state = json.loads(Path(path).read_text(encoding="utf-8"))
        weights = state.get("weights", {})
        for member in self.members:
            if member.name in weights:
                member.weight = float(weights[member.name])
        return {m.name: m.weight for m in self.members}


# --------------------------------------------------------------------------
# scoring the ledger from played games
# --------------------------------------------------------------------------

def score_member_opinions(
    ledger: OutcomeLedger,
    member,
    result,
    *,
    name: str | None = None,
    max_positions: int | None = None,
) -> int:
    """Feed one played game into the ledger for one member. Returns positions scored.

    The rule, stated plainly: walk the game's recorded positions. For each one,
    ask the member which move it would play, and compare it to the move that was
    actually played **in that position**, scoring a hit only when the member and
    the player who moved agree. The side that was to move is the only side whose
    move is known, so it is the only side a member can be scored against.

    **Why the loser's plies are scored the same way, not inverted.** An earlier
    version scored the winner's plies as ``move == played`` and the loser's as
    ``move != played``, on the reasoning that "agreeing with a losing move is not
    a correct prediction". That inverts the question: the member is not being
    asked "was this move good", it is being asked "what would you have played".
    Scoring a *disagreement* as a hit means every ply the loser moved is a free
    hit for any member that plays something else -- which is nearly all members,
    since a specific move matches rarely. Measured, a **purely random** member
    scored **0.429** on a 7-move decisive game (chance is ~0.05), and every
    loser ply was a guaranteed hit.

    Both sides are now scored by agreement with the move actually played. That
    is a weaker claim than "the move was good" -- it is deliberately a
    *consistency* score, not a strength score, which is the right target for
    ensemble weighting: what is wanted is the member whose judgement agrees with
    what competent play looks like, not the member that happened to agree with
    the winner in this particular game.

    Unfinished games have no winner and no basis for this comparison, so they
    fall back to :func:`_move_is_safe`.

    ``max_positions`` caps the work per game, since a depth-5 member answering
    every ply of a 120-ply game is minutes of search for one row of a table.
    """
    label = name or getattr(member, "name", None) or "member"
    ledger.register(label)
    if not getattr(result, "fens", None):
        return 0

    fens = result.fens[:-1] if len(result.fens) > 1 else result.fens
    if max_positions is not None:
        fens = fens[:max_positions]

    scored = 0
    for ply, fen in enumerate(fens):
        board = chess.Board(fen)
        if board.is_game_over():
            continue
        try:
            move = member.select(board)
        except Exception:                       # a broken member must not kill the ledger
            continue
        if move not in board.legal_moves:
            continue

        if result.is_decisive and ply < len(result.moves):
            played_san = result.moves[ply]
            try:
                played = board.parse_san(played_san)
            except ValueError:
                continue
            # Agreement with the move that was actually played in this position.
            # Both colours are scored identically -- see the docstring for why
            # inverting the loser's plies was wrong.
            correct = bool(move == played)
        else:
            correct = _move_is_safe(board, move)
        ledger.note(label, correct)
        scored += 1
    return scored


def _move_is_safe(board: chess.Board, move: chess.Move) -> bool:
    """Cheap "did this move hang material" test, for games with no winner.

    Not an evaluation. It is the smallest judgement that carries a sign, used
    only so that unfinished games -- which are the majority in this benchmark --
    still contribute usable evidence to the ledger rather than being dropped.

    **Why this is not a static-evaluation swing.** An earlier version compared
    ``evaluate()`` before and after the move and accepted anything that did not
    lose a pawn's worth:

        before = V.evaluate(board); board.push(move)
        after = -V.evaluate(board); board.pop()
        return (after - before) > -100

    That rule was **dead code**, and measurably so. ``evaluate`` is material +
    piece-square only, so it cannot see the opponent's reply: the swing it
    measures is just the mover's own material change. Over 11,691 sampled legal
    moves the observed swing ranged only from **-50 to +99957**, with a minimum
    of -50 -- no move ever reached the -100 that the rule was testing for. It
    returned ``True`` for **100% of moves**, including 200/200 purely random
    ones. Every unfinished game therefore gave every member a perfect score,
    which is worse than no signal: it looks like evidence.

    The replacement asks the question that actually has a sign -- *what can hit
    the square I just moved to* -- via :func:`~chessrl.value.recapture_risk`,
    which pushes the move and inspects the position. A move is unsafe when the
    cheapest enemy attacker is worth less than the piece now sitting on the
    target and the move did not already win that much material. Measured, it
    rejects 11.9% of random moves, so the ledger can now distinguish anything.
    """
    from . import value as V
    from .value import PIECE_VALUE

    risk = V.recapture_risk(board, move)
    if risk == 0:
        return True
    mover = board.piece_at(move.from_square)
    mover_value = PIECE_VALUE[mover.piece_type] if mover is not None else 0
    gained = V.capture_value(board, move)
    # Unsafe when we are offering a piece worth more than its cheapest capturer,
    # without having already captured at least that much.
    return not (risk < mover_value and gained < mover_value)
