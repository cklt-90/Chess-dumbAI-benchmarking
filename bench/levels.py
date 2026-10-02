"""The roster: every level in the ladder, wrapped as a named, playable policy.

Why this file exists
--------------------
The ladder is the *claim* of this repo -- that each level is strictly more
informed than the one below it. A claim like that is only worth anything if it
is measured on one harness, at once, from a fixed starting book, with the same
seeds. So every entry here is a ``MovePolicy`` and nothing else: no level gets a
custom scorer, no level gets a head start, and the names are stable because they
are the keys of every table the bench emits.

Two kinds of entry, and the distinction matters:

* **Reference levels** -- random, greedy-material. These are floors, not
  learners. ``random`` is the sanity check: if anything loses to it, that thing
  is broken, not "relatively weak".
* **Ladder levels** -- L1 upward. Each is expected to beat the one below it.
  Where that fails it is a *result*, not a bug (see ``bench/README.md`` on
  non-transitivity), so the bench records the raw matrix rather than asserting
  an ordering.

Torch levels are registered lazily. L0-L3 must stay importable without torch --
that is a stated property of the repo -- so ``build`` catches the ImportError
and reports the level as unavailable rather than failing the run.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import chess


@dataclass(frozen=True)
class LevelSpec:
    """One playable entrant.

    ``factory`` is a zero-argument callable so that a policy built for one match
    cannot carry state (a warm transposition table, a filled cache) into the
    next. That is not theoretical: L5's engine has three caches and a TT, and
    reusing an instance across a round-robin turns the tournament into a
    measurement of cache warmth. A fresh instance per match is slower and
    correct.
    """

    name: str
    factory: object  # Callable[[], MovePolicy]
    tier: str  # "reference" | "ladder"
    needs_torch: bool = False
    notes: str = ""
    tags: tuple[str, ...] = field(default=())


# --------------------------------------------------------------------------
# reference floors
# --------------------------------------------------------------------------

def _random():
    from chessrl.game import RandomPolicy

    return RandomPolicy(seed=0)


def _material():
    from chessrl.game import MaterialPolicy

    return MaterialPolicy(seed=0)


# --------------------------------------------------------------------------
# the ladder
# --------------------------------------------------------------------------

def _l1():
    from chessrl.policy import FactoredSoftmaxPolicy

    return FactoredSoftmaxPolicy(seed=0)


def _minimax(depth: int, informed: bool = False):
    """L2 at a given depth, optionally with L1's table half-informing ordering.

    ``informed`` is the L2 *wrapper* rather than the plain engine. Both are in
    the roster because they differ in a way that is easy to miss: they return
    the same *value* but can return different *moves* among equals. Putting only
    one of them in the table would hide that.
    """
    def build():
        from chessrl.search import InformedMinimaxPolicy, MinimaxPolicy

        if informed:
            return InformedMinimaxPolicy(_l1(), depth=depth)
        return MinimaxPolicy(depth=depth)

    return build


def _l3(seed: int = 0):
    from chessrl.perceptron import L3Policy

    return L3Policy(seed=seed)


def _l35(seed: int = 0):
    from chessrl.squarelocal import L35Policy

    return L35Policy(seed=seed)


def _l3_flat(seed: int = 0):
    """The control arm for the L3.5 experiment: no spatial structure at all.

    Registered as an ``ablation`` entrant rather than a ladder level because it
    is not a rung -- nobody expects a flat linear model over ``encode()`` to be
    strong. Its only job is to make an L3.5 win *mean* something: without it, a
    result favouring L3.5 cannot be distinguished from "any change from L3
    helps".
    """
    from chessrl.squarelocal import FlatPolicy

    return FlatPolicy(seed=seed)


def _l4(seed: int = 0):
    from chessrl.torch_model import TorchConfig, TorchPolicy, TorchScorer

    return TorchPolicy(TorchScorer(TorchConfig(), seed=seed))


def _l5(depth: int, *, use_model_eval: bool = True, use_model_prior: bool = True,
        seed: int = 0):
    """L5 at a given depth, with the two model seams independently switchable.

    The flags are the point. ``use_model_eval`` and ``use_model_prior`` are the
    only two places a learned model can act in a search, so the four
    combinations are not four variants of one thing -- they are the ablation
    that separates "the prior saves time" from "the evaluator changes the
    answer". The bench runs all four.
    """
    def build():
        from chessrl.guided import GuidedConfig, GuidedModel, GuidedPolicy

        model = GuidedModel(GuidedConfig(seed=seed))
        return GuidedPolicy(
            model, depth=depth,
            use_model_eval=use_model_eval,
            use_model_prior=use_model_prior,
        )

    return build


def _l6(seed: int = 0):
    """L6: the ensemble of the levels below it, weighted by outcome accuracy.

    Two things about this entry are deliberate and easy to get wrong.

    **It is not one policy, it is a roster.** The members are the same
    :func:`_l1`, :func:`_l3` and :func:`_minimax` factories the matrix already
    uses, so L6 is de-duplicated in exactly the way the ladder requires: if a
    member improves, L6 improves without being touched.

    **It starts unweighted, and that is honest rather than broken.**
    ``EnsemblePolicy`` with no ledger gives every member the mean weight, so an
    untrained L6 is a uniform mixture -- a real, playable policy, not a
    placeholder. It becomes *L6* in the sense the ladder means only once
    :func:`chessrl.ensemble.score_member_opinions` has written a ledger and
    :meth:`EnsemblePolicy.set_weights_from_ledger` has read it back. The bench
    therefore reports the untrained row, and ``bench/ensemble_run.py`` produces
    the weighted one; conflating the two would let an untrained ensemble be
    reported as if it had been fitted.

    Depth 2 is used for the search member, matching ``L2i-d2`` and the rest of
    the affordable roster: a depth-5 member would make every L6 move a
    20-second decision for no extra information at this resolution.
    """
    from chessrl.ensemble import EnsemblePolicy

    ensemble = EnsemblePolicy(seed=seed)
    ensemble.add("L1", _l1(), weight=1.0)
    ensemble.add("L3", _l3(seed), weight=1.0)
    ensemble.add("L2-d2", _minimax(2)(), weight=1.0)
    return ensemble


# --------------------------------------------------------------------------
# the roster
# --------------------------------------------------------------------------

# Depths for the search levels. Kept low on purpose: a round-robin at depth 5
# costs 20s/move for L5, which turns a full matrix into an overnight job for no
# extra information at the resolution these tables report.
SEARCH_DEPTHS = (1, 2, 3)


def default_roster() -> list[LevelSpec]:
    """The full entrant list, cheapest first.

    Ordered by expected cost so that a ``--limit`` run still produces a
    meaningful partial matrix rather than only the expensive rows. Every level
    is included at every depth that is affordable, because the depth-vs-strength
    curve is itself a result -- "deeper is better" must be visible in the data,
    not assumed.
    """
    specs = [
        LevelSpec("random", _random, "reference", notes="uniform legal mover"),
        LevelSpec("material", _material, "reference",
                  notes="1-ply greedy on static eval"),
        LevelSpec("L1", _l1, "ladder", notes="blind factored softmax, untrained"),
    ]
    for d in SEARCH_DEPTHS:
        specs.append(LevelSpec(
            f"L2-d{d}", _minimax(d), "ladder",
            notes=f"alpha-beta, plain ordering, depth {d}",
        ))
    # The informed wrapper only at one depth: it exists to demonstrate a tie
    # being broken by a prior, not to be a strength tier of its own.
    specs.append(LevelSpec(
        "L2i-d2", _minimax(2, informed=True), "ladder",
        notes="alpha-beta + L1 table as ordering prior, depth 2",
    ))
    specs.append(LevelSpec("L3", _l3, "ladder", needs_torch=False,
                           notes="factorised perceptron, per-square rows"))

    # ---- the L3.5 experiment ------------------------------------------
    #
    # Three arms at (nominally) one budget, varying only the shape of the
    # computation. The spec is explicit that all three must be present: L3.5
    # alone cannot distinguish "the tied per-square prior helps" from "any
    # change helps", and L3-flat is the arm that rules the second out.
    #
    # BUDGET NOTE. The spec's §3 asks for +-10% of L3's 7225 and its §3 layout
    # table lists 783 -- an internal contradiction no honest implementation can
    # satisfy. These three arms land at 7225 / 783 / 1445, i.e. 100% / 10.8% /
    # 20.0% of L3. The comparison is therefore *not* equal-budget in the literal
    # sense; it is "same task, same features, same signal, architecture the only
    # free variable, at the width each architecture can honestly use". The
    # deviation and its reason are recorded in ``squarelocal.py`` and in the
    # test named ``test_parameter_count_within_ten_percent_of_L3``.
    specs.append(LevelSpec(
        "L3.5", _l35, "ladder", needs_torch=False,
        notes="weight-tied per-square scorer, lean budget (783 params)",
    ))
    specs.append(LevelSpec(
        "L3-flat", _l3_flat, "ablation", needs_torch=False,
        notes="flat linear over encode(), control arm (1445 params)",
        tags=("ablation",),
    ))

    for d in SEARCH_DEPTHS:
        specs.append(LevelSpec(
            f"L5-d{d}", _l5(d), "ladder", needs_torch=True,
            notes=f"guided alpha-beta, depth {d}, untrained model",
        ))
    # L4 as a policy needs torch and has no search of its own; included so its
    # distribution-level strength is on the same axis as everything else.
    specs.append(LevelSpec("L4", _l4, "ladder", needs_torch=True,
                           notes="torch model as a distribution policy, untrained"))

    # ---- L6 ------------------------------------------------------------
    #
    # Last, because it is a combination of everything above it and is the most
    # expensive single entrant: one move costs L1 + L3 + a depth-2 search.
    #
    # This row is the *untrained* ensemble -- a uniform mixture, since no ledger
    # exists yet. That is the right thing for the matrix: every other learner in
    # the table is likewise untrained, so L6 sits on the same axis. The weighted
    # L6 is a separate artefact produced by the ensemble driver, not a bench row,
    # because its weights are derived from games the bench has not played yet.
    specs.append(LevelSpec(
        "L6", _l6, "ladder", needs_torch=False,
        notes="ensemble of L1 + L3 + L2-d2, unweighted (uniform mixture)",
        tags=("ensemble",),
    ))
    return specs


def ablation_roster() -> list[LevelSpec]:
    """L5 at depth 2 with every combination of the two model seams.

    Returned as its own roster because it answers a different question from the
    main matrix: not "which level is strongest" but "what is each seam worth".
    """
    out = []
    for prior in (False, True):
        for evalu in (False, True):
            out.append(LevelSpec(
                f"L5-ab-{'prior' if prior else 'blind'}"
                f"{'+eval' if evalu else ''}",
                _l5(2, use_model_eval=evalu, use_model_prior=prior),
                "ablation", needs_torch=True,
                notes=(f"model prior {'on' if prior else 'off'}, "
                       f"model eval {'on' if evalu else 'off'}"),
                tags=("ablation",),
            ))
    return out


def build(spec: LevelSpec) -> tuple[object | None, str | None]:
    """Instantiate one entrant. Returns ``(policy, error)``.

    Never raises. A missing optional dependency is a fact about the environment,
    not a failure of the benchmark -- and the bench has to stay runnable on a
    machine without torch, because L0-L3 are the part that is always installable.
    The error string is carried into the report so an absent row is explained
    rather than merely missing.
    """
    try:
        return spec.factory(), None
    except ImportError as exc:
        # Keep the original message. An earlier version replaced it with just
        # the exception *class* for torch levels, to signal "this is the
        # optional extra, not a bug" -- but "unavailable: ImportError" tells the
        # reader nothing about *which* module is missing, and the module name is
        # the one useful fact (it is usually torch, occasionally a transitive
        # dependency that a `pip install` would fix).
        prefix = "unavailable (optional extra): " if spec.needs_torch else "unavailable: "
        return None, f"{prefix}{exc}"
    except Exception as exc:  # pragma: no cover - surfaced in the report
        return None, f"{exc.__class__.__name__}: {exc}"


# A small fixed opening book. Playing distinct balanced positions is what makes
# a 6-game match informative rather than a coin flip: the variance comes from
# the positions, and these are chosen to be quiet (no forced tactics, roughly
# equal material, both sides developed).
#
# **Every entry must be White to move.** This is load-bearing, not stylistic.
# ``play_match`` assigns colours by alternating White and Black between the two
# policies, and ``play_game`` then dispatches the move to whichever policy owns
# ``board.turn``. If a book entry has Black to move, the *position* decides who
# opens and the colour assignment is silently inverted -- so "A plays White in
# games 0 and 2" becomes false, and an even-looking match is actually
# colour-biased. Half of the original book was Black-to-move for exactly this
# reason and nothing caught it until a test watched which policy made the first
# move. ``assert_white_to_move`` enforces it.
#
# Deliberately *not* the initial position repeated -- a single opening makes
# every game a near-copy of the others and the match reports one result with
# the illusion of six.
OPENING_BOOK: tuple[str, ...] = (
    # the initial position
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    # 1.e4 e5 2.Nf3 Nc6 3.Bc4 -- Italian, quiet
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    # 1.d4 Nf6 2.c4 e6 3.Nc3 d5 -- QGD, closed
    "rnbqkb1r/ppp2ppp/4pn2/3p4/2PP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 0 4",
    # 1.d4 Nf6 2.Nf3 c5 -- Indian, symmetrical
    "rnbqkb1r/pp1ppppp/5n2/2p5/3P4/5N2/PPP1PPPP/RNBQKB1R w KQkq - 0 3",
    # 1.e4 c5 2.Nf3 d6 -- Sicilian, quiet variation
    "rnbqkbnr/pp2pppp/3p4/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 3",
    # 1.c4 e5 2.Nc3 Nf6 -- English, balanced
    "rnbqkb1r/pppp1ppp/5n2/4p3/2P5/2N5/PP1PPPPP/R1BQKBNR w KQkq - 2 3",
)


def assert_white_to_move(fen: str) -> None:
    """Guard the book invariant. See the comment on ``OPENING_BOOK``."""
    board = chess.Board(fen)
    if board.turn != chess.WHITE:
        raise ValueError(
            f"opening book entry must be White to move, got Black: {fen}"
        )


def is_playable(fen: str) -> bool:
    """A book entry is only usable if the game is not already over."""
    board = chess.Board(fen)
    return not board.is_game_over() and board.legal_moves.count() > 0
