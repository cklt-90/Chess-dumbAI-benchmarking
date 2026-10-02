"""``python -m bench``: run the scoreboard and print (or save) the report.

Usage
-----
    python -m bench                       # the full default roster, 6 games/pair
    python -m bench --roster ablation     # the L5 seam ablation instead
    python -m bench --games 10 --out bench.json
    python -m bench --only random material L1 L2-d2   # a quick subset
    python -m bench --list                # show what the roster contains
    python -m bench --from bench.json     # re-render without replaying games

Two things this deliberately does *not* do
------------------------------------------
**It does not re-fit an existing result file into a new matrix.** The saved JSON
contains the raw counts, so re-fitting is one call to ``bench.rating`` -- making
the CLI do it would mean two ways to produce a rating table and only one of them
tested.

**It does not parallelise.** Levels carry internal caches and the search levels
are already the bottleneck; running matches in threads would divide the same CPU
and make the wall-clock column meaningless while adding a class of
non-determinism to a benchmark whose whole value is reproducibility. If it is too
slow, lower ``--games`` or ``--only`` a subset.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Make ``python -m bench`` work from a fresh clone without an editable install.
#
# ``chessrl`` lives under ``src/`` and is not on ``sys.path`` by default; pytest
# papers over this with ``pythonpath = ["src", "tests"]``, but a user running the
# bench directly gets ``ModuleNotFoundError``. Rather than require
# ``pip install -e .`` before the benchmark is usable -- which is a real
# friction point for a tool whose purpose is to be run -- prepend the source
# directory here. Guarded so an installed ``chessrl`` (a real deployment) always
# wins.
_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from bench.levels import (
    LevelSpec,
    ablation_roster,
    build,
    default_roster,
)
from bench.rating import fit_bradley_terry
from bench.report import format_ablation, format_summary
from bench.runner import BenchResult, MatchOutcome, run_round_robin


def _rosters() -> dict[str, object]:
    return {"default": default_roster, "ablation": ablation_roster}


def _select_specs(roster: str, only: list[str] | None) -> list[LevelSpec]:
    specs = _rosters()[roster]()
    if not only:
        return specs
    wanted = set(only)
    picked = [s for s in specs if s.name in wanted]
    missing = wanted - {s.name for s in picked}
    if missing:
        raise SystemExit(
            f"unknown entrant(s): {', '.join(sorted(missing))}\n"
            f"available: {', '.join(s.name for s in specs)}"
        )
    return picked


def _serialise(bench: BenchResult) -> dict:
    return {
        "names": bench.names,
        "seconds": bench.seconds,
        "skipped": bench.skipped,
        "matches": [
            {
                "a": m.a, "b": m.b, "seconds": m.seconds,
                "wins": m.record.wins, "draws": m.record.draws,
                "losses": m.record.losses, "unfinished": m.record.unfinished,
            }
            for m in bench.matches
        ],
    }


def _load(path: str | Path) -> BenchResult:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    from bench.rating import MatchRecord

    matches = [
        MatchOutcome(
            m["a"], m["b"],
            MatchRecord(m["a"], m["b"], wins=m["wins"], draws=m["draws"],
                        losses=m["losses"], unfinished=m["unfinished"]),
            games=[],
            seconds=m.get("seconds", 0.0),
        )
        for m in data["matches"]
    ]
    return BenchResult(
        names=data["names"], matches=matches,
        skipped=data.get("skipped", {}), seconds=data.get("seconds", 0.0),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bench",
        description="Round-robin scoreboard for the chess-rl ladder.",
    )
    parser.add_argument("--roster", choices=sorted(_rosters()), default="default",
                        help="which entrant list to run (default: %(default)s)")
    parser.add_argument("--games", type=int, default=6,
                        help="decided games per pair, rounded up to even "
                             "(default: %(default)s)")
    parser.add_argument("--plies", type=int, default=200,
                        help="hard ply cap, games hitting it are unfinished "
                             "(default: %(default)s)")
    parser.add_argument("--no-progress", type=int, default=60,
                        help="terminate after this many plies with no capture "
                             "and no pawn move (default: %(default)s)")
    parser.add_argument("--prior", type=float, default=1.0,
                        help="rating prior precision; raise for a small roster "
                             "(default: %(default)s)")
    parser.add_argument("--anchor", default="random",
                        help="entrant pinned to 0 Elo (default: %(default)s)")
    parser.add_argument("--only", nargs="+", metavar="NAME",
                        help="restrict to these entrants")
    parser.add_argument("--out", metavar="PATH",
                        help="write the raw matrix as JSON (the report always "
                             "goes to stdout)")
    parser.add_argument("--json", action="store_true",
                        help="print the JSON to stdout instead of the report")
    parser.add_argument("--from", dest="from_file", metavar="PATH",
                        help="re-render a report from a previously saved JSON "
                             "instead of playing games")
    parser.add_argument("--list", action="store_true",
                        help="list the roster and exit")
    args = parser.parse_args(argv)

    if args.list:
        for spec in _rosters()[args.roster]():
            policy, err = build(spec)
            status = "ok" if policy is not None else err
            print(f"{spec.name:<16s} [{spec.tier:<9s}] {status}")
            if spec.notes:
                print(f"{'':16s}  {spec.notes}")
        return 0

    if args.from_file:
        bench = _load(args.from_file)
    else:
        specs = _select_specs(args.roster, args.only)
        if not specs:
            print("no entrants selected", file=sys.stderr)
            return 2
        started = time.perf_counter()
        bench = run_round_robin(
            specs,
            games=args.games,
            max_plies=args.plies,
            no_progress_limit=args.no_progress,
            progress=None if args.json else (lambda s: print(s, file=sys.stderr)),
        )
        bench.seconds = time.perf_counter() - started

    if args.out:
        Path(args.out).write_text(
            json.dumps(_serialise(bench), indent=2), encoding="utf-8"
        )

    if args.json:
        print(json.dumps(_serialise(bench), indent=2))
        return 0

    fit = fit_bradley_terry(
        bench.records, bench.names, prior=args.prior, anchor=args.anchor
    )
    print(format_summary(bench, fit))
    if args.roster == "ablation":
        print()
        print("-- ABLATION " + "-" * 66)
        print(format_ablation(bench, fit))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
