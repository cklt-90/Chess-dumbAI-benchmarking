"""H13 / H3 -- does folding unfinished games into draws move the rating?

H3 warns that scoring a ply-capped (unfinished) game as a draw is the trap that
teaches a naive policy that shuffling is neutral. H13 worries that 6 games/pair
leaves the standard errors too wide to read. Both are answerable from a *stored*
result matrix, because the runner serialises the raw counts -- so this script
re-fits ``bench/results-2026-10-02.json`` twice:

* ``include_unfinished_as_draws=False`` (the repo default -- unfinished excluded)
* ``include_unfinished_as_draws=True``  (the H3 trap: each unfinished = 1/2-1/2)

and reports the per-entrant rating / standard-error shift and the rank churn.
No games are replayed. The patch in ``bench/rating.py`` is the only change.
"""
import json
import sys
from pathlib import Path

import numpy as np

# bench/ is not on the path for a raw run; rating.py only needs numpy.
_BENCH = Path(__file__).resolve().parents[1]
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

from rating import MatchRecord, fit_bradley_terry  # noqa: E402


def _load_records(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    names = data["names"]
    records = [
        MatchRecord(
            m["a"], m["b"],
            wins=m["wins"], draws=m["draws"],
            losses=m["losses"], unfinished=m["unfinished"],
        )
        for m in data["matches"]
    ]
    return names, records


def _rank_map(order):
    return {n: i for i, n in enumerate(order)}


def main():
    path = _BENCH / "results-2026-10-02.json"
    names, records = _load_records(path)

    off = fit_bradley_terry(records, names, include_unfinished_as_draws=False)
    on = fit_bradley_terry(records, names, include_unfinished_as_draws=True)
    off_by = {r.name: r for r in off.ratings}
    on_by = {r.name: r for r in on.ratings}

    print(f"{'entrant':<16}{'Elo excl':>10}{'Elo incl':>10}"
          f"{'SE excl':>9}{'SE incl':>9}{'dElo':>8}")
    print("-" * 62)
    for name in names:
        o, n = off_by[name], on_by[name]
        print(f"{name:<16}{o.elo:>10.1f}{n.elo:>10.1f}"
              f"{o.stderr:>9.1f}{n.stderr:>9.1f}{n.elo - o.elo:>8.1f}")

    ro, rn = _rank_map([r.name for r in off.ratings]), _rank_map([r.name for r in on.ratings])
    churn = sum(1 for n in names if ro[n] != rn[n])
    print(f"\nRank changed for {churn}/{len(names)} entrants "
          f"(excl order vs incl order).")

    moves = sorted(
        ((n, on_by[n].elo - off_by[n].elo) for n in names),
        key=lambda t: abs(t[1]), reverse=True,
    )
    print("Largest rating moves (incl - excl):")
    for n, d in moves[:5]:
        print(f"  {n:<16} {d:+.1f}")

    se_off = np.mean([off_by[n].stderr for n in names])
    se_on = np.mean([on_by[n].stderr for n in names])
    print(f"\nMean SE: excl = {se_off:.1f}, incl = {se_on:.1f} "
          f"({(1 - se_on / se_off) * 100:.1f}% smaller when unfinished counted).")
    print(f"SE range excl: {min(off_by[n].stderr for n in names):.0f}"
          f"..{max(off_by[n].stderr for n in names):.0f} "
          f"-- the H13 'too few games' window.")


if __name__ == "__main__":
    main()
