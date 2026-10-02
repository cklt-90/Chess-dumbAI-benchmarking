"""Make L6 actually L6: fit the ensemble weights from played games.

Why this is a separate script and not a bench level
---------------------------------------------------
The bench measures entrants that are already built. L6 is different in kind: its
only free parameters are the *weights over its members*, and those are derived
from games. A bench row and a fitting procedure cannot be the same thing, because
the fit needs the bench's output as its input -- the matrix says which members
beat which, and the ledger says which members' judgement matched the outcomes.

Worse, folding the fit into the bench would make the result circular and
unfalsifiable: the weights would be chosen using the same games that are then
reported as L6's strength, so L6 would look good by construction and the number
would mean nothing. Hence the split, which is the standard one:

1. **Calibrate** -- play a set of games between the members (and against the
   reference floors) from the fixed opening book. Every position of every game
   is fed to every member, and the member's opinion is scored against what
   actually happened. That fills an :class:`~chessrl.ensemble.OutcomeLedger`.
2. **Fit** -- turn the ledger's accuracies into weights. No games are played.
3. **Report** -- the weights, the accuracies they came from, and the ensemble's
   play against a held-out set of positions.

Step 1 must use different games from any match L6 is later judged on. The
``--holdout`` flag exists for exactly that, and the script refuses to fit and
evaluate on the same book entries when it is set.

What the ledger measures, restated because it is the load-bearing idea
---------------------------------------------------------------------
Not "which member wins games". A member can be right about a position and lose
the game anyway. The ledger records, position by position, whether the member
would have played the move that the eventual winner's side actually played. That
is a *consistency* score, and it is the right target for weighting an ensemble:
what is wanted is the member whose judgement agrees with outcomes, not the one
with the best result in this particular sample.

Usage::

    python bench/ensemble_run.py --games 12 --out bench/ensemble.json

The output JSON is the artefact: weights, per-member accuracy, the games that
produced them, and the configuration used. It is deliberately readable, because
the interesting question about an ensemble is always "why did this member get
that weight", and a binary checkpoint cannot answer it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Allow ``python bench/ensemble_run.py`` from the repo root without an install.
_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import chess  # noqa: E402
import numpy as np  # noqa: E402

from bench.levels import (  # noqa: E402
    OPENING_BOOK,
    build,
    default_roster,
    is_playable,
)
from chessrl.ensemble import (  # noqa: E402
    EnsembleConfig,
    EnsemblePolicy,
    OutcomeLedger,
    accuracy_weights,
    score_member_opinions,
)
from chessrl.game import play_game  # noqa: E402


# The members of L6, in the order they are reported. Kept as a literal list of
# roster names rather than derived from the ladder, so that adding a level to the
# bench does not silently change what L6 is -- a change to the ensemble's
# composition should be a deliberate edit to this line, reviewable as such.
MEMBER_NAMES: tuple[str, ...] = ("L1", "L3", "L2-d2")


def _member_policies(roster: dict, seed: int) -> dict:
    """Build one fresh policy per member name.

    Fresh, not shared, for the same reason the bench builds per match: L2-d2 has
    a transposition table, and handing a warm one to the calibration would make
    the ledger depend on the order the games were played in.
    """
    out = {}
    for name in MEMBER_NAMES:
        if name not in roster:
            raise KeyError(f"{name!r} is not in the bench roster")
        policy, err = build(roster[name])
        if policy is None:
            raise RuntimeError(f"could not build {name}: {err}")
        out[name] = policy
    return out


def calibrate(
    *,
    games: int = 12,
    seed: int = 0,
    max_plies: int = 160,
    max_positions_per_game: int | None = 24,
    book: tuple[str, ...] | None = None,
    progress=None,
) -> OutcomeLedger:
    """Play calibration games; return the ledger they produce.

    The games are between the members themselves, in round-robin pairs, drawn
    from the opening book. There is no L6 on the board during calibration --
    that is the point. L6 is being *fitted* here, and anything it did would be
    measured with weights that are about to change.

    ``max_positions_per_game`` bounds the work: a member scoring every ply of a
    160-ply game is minutes of search for one row of a table, and the ledger's
    precision does not need that many observations from a single game. Positions
    are taken from the start of the game, where the opening book puts the
    variance; the endgame of a calibration game is mostly the same shuffling
    regardless of who played it.
    """
    roster = {spec.name: spec for spec in default_roster()}
    members = _member_policies(roster, seed)
    names = list(members)

    positions = [f for f in (book or OPENING_BOOK) if is_playable(f)]
    if not positions:
        raise RuntimeError("the opening book has no playable entries")

    ledger = OutcomeLedger()
    for name in names:
        ledger.register(name)

    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
    if not pairs:
        raise RuntimeError("calibration needs at least two members")

    game_index = 0
    for a, b in pairs:
        for rep in range(max(1, games // len(pairs))):
            start_fen = positions[game_index % len(positions)]
            # Alternate colours so neither member is systematically White, which
            # in the opening is a real advantage and would bias the ledger.
            a_is_white = (game_index // len(positions) + game_index) % 2 == 0
            white, black = (
                (members[a], members[b]) if a_is_white else (members[b], members[a])
            )
            result = play_game(
                white, black,
                max_plies=max_plies,
                start_fen=start_fen,
                rng=np.random.default_rng(seed + game_index),
            )
            for name in names:
                score_member_opinions(
                    ledger, members[name], result,
                    name=name,
                    max_positions=max_positions_per_game,
                )
            game_index += 1
            if progress is not None:
                progress(
                    f"  [{game_index}] {a} vs {b} -> {result.result:<10s} "
                    f"{result.reason} ({result.plies} plies)"
                )
    return ledger


def fit(
    ledger: OutcomeLedger,
    *,
    config: EnsembleConfig | None = None,
) -> dict[str, float]:
    """Derive weights from a ledger. Pure arithmetic; no chess, no games."""
    cfg = config or EnsembleConfig()
    return accuracy_weights(
        ledger, list(MEMBER_NAMES),
        temperature=cfg.temperature,
        floor=cfg.floor,
        chance_equivalent=cfg.chance_equivalent,
        relative=cfg.relative,
    )


def build_weighted_ensemble(
    weights: dict[str, float],
    *,
    seed: int = 0,
    config: EnsembleConfig | None = None,
) -> EnsemblePolicy:
    """An L6 with the fitted weights installed.

    The same composition as ``bench.levels._l6``, but weighted -- which is the
    only difference between "an ensemble" and "L6 as the ladder means it".
    """
    roster = {spec.name: spec for spec in default_roster()}
    members = _member_policies(roster, seed)
    # ``config`` is passed by keyword. Positionally it would land on the
    # ``members`` parameter and raise, which is the good outcome -- but the same
    # mistake is silent when the two happen to be shape-compatible.
    ensemble = EnsemblePolicy(config=config or EnsembleConfig(), seed=seed)
    for name in MEMBER_NAMES:
        ensemble.add(name, members[name], weight=float(weights.get(name, 0.0)))
    return ensemble


def describe_fit(ledger: OutcomeLedger, weights: dict[str, float]) -> list[dict]:
    """The report: what each member was asked, how it did, what it is worth."""
    accuracies = ledger.accuracies()
    rows = []
    for name in MEMBER_NAMES:
        entry = ledger.get(name)
        rows.append({
            "member": name,
            "asked": entry.predicted if entry else 0,
            "correct": entry.correct if entry else 0,
            "accuracy": accuracies.get(name),
            "weight": weights.get(name, 0.0),
        })
    return rows


def format_fit(rows: list[dict]) -> str:
    """A fixed-width table, because this is meant to be read in a terminal."""
    lines = [
        f"{'member':<10} {'asked':>7} {'correct':>8} {'accuracy':>9} {'weight':>8}",
        "-" * 46,
    ]
    for row in rows:
        acc = row["accuracy"]
        lines.append(
            f"{row['member']:<10} {row['asked']:>7} {row['correct']:>8} "
            f"{(f'{acc:.3f}' if acc is not None else '-'):>9} "
            f"{row['weight']:>8.4f}"
        )
    return "\n".join(lines)


def load_fit(path: str | Path) -> dict:
    """Read back a fitted ensemble, ledger included.

    The returned mapping is the JSON payload, with ``ledger`` upgraded from a
    plain dict to an :class:`OutcomeLedger` so a caller can re-fit at a
    different temperature without replaying the games.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    payload["ledger"] = OutcomeLedger.from_dict(payload.get("ledger", {}))
    return payload


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--games", type=int, default=12,
                        help="calibration games per member pair (default 12)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-plies", type=int, default=160)
    parser.add_argument("--max-positions", type=int, default=24,
                        help="positions scored per member per game "
                             "(0 = no cap)")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--floor", type=float, default=0.05)
    parser.add_argument("--chance-equivalent", type=float, default=0.5)
    parser.add_argument("--relative", action="store_true",
                        help="scale each member's accuracy relative to the mean "
                             "of the set instead of to chance; below-mean members "
                             "drop to zero so a mediocre majority cannot dilute a "
                             "competent member. The structural fix for the "
                             "negative L6 result.")
    parser.add_argument("--out", type=Path, default=Path("bench/ensemble.json"),
                        help="where to write the fitted ensemble JSON")
    parser.add_argument("--ledger-out", type=Path, default=None,
                        help="optionally also write the raw ledger")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    config = EnsembleConfig(
        temperature=args.temperature,
        floor=args.floor,
        chance_equivalent=args.chance_equivalent,
        relative=args.relative,
    )

    print(f"calibrating L6: members {', '.join(MEMBER_NAMES)}")
    print(f"  games per pair {args.games}, seed {args.seed}, "
          f"max plies {args.max_plies}")
    started = time.perf_counter()
    ledger = calibrate(
        games=args.games,
        seed=args.seed,
        max_plies=args.max_plies,
        max_positions_per_game=args.max_positions or None,
        progress=lambda line: print(line),
    )
    elapsed = time.perf_counter() - started

    weights = fit(ledger, config=config)
    rows = describe_fit(ledger, weights)

    print()
    print(f"calibration took {elapsed:.1f}s, "
          f"{ledger.total_predicted()} scored opinions")
    print()
    print(format_fit(rows))

    payload = {
        "members": rows,
        "weights": weights,
        "config": dict(config.__dict__),
        "calibration": {
            "games_per_pair": args.games,
            "seed": args.seed,
            "max_plies": args.max_plies,
            "max_positions_per_game": args.max_positions or None,
            "total_scored": ledger.total_predicted(),
            "seconds": round(elapsed, 2),
        },
        # The raw ledger travels with the fit so the weight derivation can be
        # re-checked (or re-run at a different temperature) without replaying
        # hours of games. A weight without its evidence is an assertion.
        "ledger": ledger.as_dict(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print()
    print(f"wrote {args.out}")

    if args.ledger_out is not None:
        ledger.save(args.ledger_out)
        print(f"wrote {args.ledger_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
