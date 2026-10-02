"""L1 diagnostics: measuring what a naive policy actually does.

The point of a benchmark is to produce numbers you can act on, and the most
useful numbers here are not the win rate. A blind policy's win rate against
itself is meaningless (it should be 50% by symmetry). What you want to know is:

* **How often does a game terminate for an informative reason?** If most games
  hit the ply cap, the learner is receiving almost no signal, regardless of how
  long it has been training. This is the single most important diagnostic for
  L1, and it is invisible in a win-rate table.
* **How sharp is the policy?** Entropy over a fixed probe set. Entropy falling
  while the finished-game rate stays flat is the fingerprint of a policy that
  has learned to be confidently bad.
* **Which actions has it actually seen?** A parameter with no updates is either
  useless or untested, and those need different responses.

Kept separate from the trainer so it can be run against any policy, including
ones that were loaded from disk rather than trained in this process.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import chess
import numpy as np

from . import masks as M
from .game import DistributionPolicy, MovePolicy, play_game
from .policy import entropy, perplexity

# A fixed probe set. Deliberately small and phase-diverse: opening, developed
# middlegame, and a sparse endgame. Fixed so that entropy measured after 10
# games and after 10,000 games is comparable.
PROBE_FENS = (
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
)


@dataclass
class PlayDiagnostics:
    """Aggregate behaviour of a policy over a batch of self-play games."""

    games: int = 0
    reasons: Counter = field(default_factory=Counter)
    results: Counter = field(default_factory=Counter)
    plies: list[int] = field(default_factory=list)
    captures: list[int] = field(default_factory=list)
    checks: list[int] = field(default_factory=list)
    unique_positions: list[int] = field(default_factory=list)

    @property
    def informative_rate(self) -> float:
        """Fraction of games that ended decisively.

        This is the number that matters for L1. A decisive game carries a real
        reward signal; a capped or drawn one carries none. If this is near
        zero, no amount of additional training will help, and the fix is a
        better exploration scheme rather than more games.
        """
        if not self.games:
            return 0.0
        decisive = self.results.get("1-0", 0) + self.results.get("0-1", 0)
        return decisive / self.games

    @property
    def finished_rate(self) -> float:
        """Fraction of games that ended by any real chess rule."""
        if not self.games:
            return 0.0
        return 1.0 - self.results.get("unfinished", 0) / self.games

    @property
    def contact_rate(self) -> float:
        """Average captures per 100 plies.

        The blunt measure of whether the two sides ever interact. A blind
        sampling policy scores near zero here, which is precisely why its games
        run to the ply cap: nothing is ever exchanged, so nothing ever resolves.
        """
        total_plies = sum(self.plies)
        if not total_plies:
            return 0.0
        return 100.0 * sum(self.captures) / total_plies

    def summary(self) -> dict:
        return {
            "games": self.games,
            "reasons": dict(self.reasons),
            "results": dict(self.results),
            "informative_rate": round(self.informative_rate, 3),
            "finished_rate": round(self.finished_rate, 3),
            "mean_plies": round(np.mean(self.plies), 1) if self.plies else 0.0,
            "mean_captures": (
                round(np.mean(self.captures), 1) if self.captures else 0.0
            ),
            "mean_checks": (
                round(np.mean(self.checks), 1) if self.checks else 0.0
            ),
            "contact_rate_per_100_plies": round(self.contact_rate, 2),
            "mean_unique_positions": (
                round(np.mean(self.unique_positions), 1)
                if self.unique_positions
                else 0.0
            ),
        }


def diagnose_games(
    policy: MovePolicy,
    n_games: int = 20,
    *,
    max_plies: int = 150,
    no_progress_limit: int = 60,
    seed: int = 0,
    start_fens: list[str] | None = None,
) -> PlayDiagnostics:
    """Play self-play games and measure how the policy actually behaves.

    Records captures and checks per game as well as the termination reason,
    because "unfinished" has several causes that need different fixes: a
    repetition-heavy game says the policy is deterministic, while a
    capture-free game says it is not exploring into contact.
    """
    import random

    diag = PlayDiagnostics()
    rng = random.Random(seed)

    for i in range(n_games):
        fen = None
        if start_fens:
            fen = start_fens[i % len(start_fens)]
        game = play_game(
            policy, policy,
            max_plies=max_plies,
            no_progress_limit=no_progress_limit,
            rng=rng,
            start_fen=fen,
        )
        diag.games += 1
        diag.reasons[game.reason] += 1
        diag.results[game.result] += 1
        diag.plies.append(game.plies)
        diag.unique_positions.append(len(set(game.fens)))

        # Recount captures and checks by replaying, since the loop does not
        # record them and adding that to the hot path would slow every level
        # down for the benefit of diagnostics only.
        captures = checks = 0
        if game.fens:
            board = chess.Board(game.fens[0])
            for san in game.moves:
                move = board.parse_san(san)
                if board.is_capture(move):
                    captures += 1
                if board.gives_check(move):
                    checks += 1
                board.push(move)
        diag.captures.append(captures)
        diag.checks.append(checks)

    return diag


def policy_sharpness(
    policy: DistributionPolicy, fens: list[str] | None = None
) -> dict:
    """Entropy and perplexity of the policy over a probe set.

    Perplexity is the readable form: it is the effective number of moves the
    policy is choosing between. At initialisation a blind policy over ~30 legal
    moves has perplexity near 30; a policy that has collapsed onto one move has
    perplexity near 1.

    One subtlety worth knowing when reading ``uniformity``: a *factorised*
    softmax with zero logits is uniform per factor, not uniform over the move
    set. The origin factor spreads mass evenly over origins, and each origin
    then spreads its own share over its own destinations, so a square with more
    legal moves has each individual move weighted less. ``uniformity`` is
    therefore exactly 1.0 only where every origin has the same number of
    destinations -- true in the opening and in symmetric middlegames, false
    otherwise. It is never above 1.0, since uniform-over-moves is the
    maximum-entropy case.

    A policy that only implements ``select`` -- :class:`chessrl.game.RandomPolicy`
    and :class:`chessrl.game.MaterialPolicy` are the two here -- has no
    distribution to measure, and the difference is informative rather than an
    error. Rather than crash, report ``available: False`` with the reason, so a
    comparison table can show "n/a" for those rows instead of losing the whole
    run.
    """
    if not hasattr(policy, "move_distribution"):
        return {
            "available": False,
            "reason": (
                f"{type(policy).__name__} exposes only select(), not "
                "move_distribution(); sharpness is undefined for it"
            ),
            "probes": [],
            "mean_entropy": 0.0,
            "mean_perplexity": 0.0,
        }

    fens = fens or list(PROBE_FENS)
    entries = []
    for fen in fens:
        board = chess.Board(fen)
        if board.is_game_over(claim_draw=False):
            continue
        dist = policy.move_distribution(board)
        positive = dist[dist > 0]
        if positive.size == 0:
            continue
        normalised = positive / positive.sum()
        n_legal = board.legal_moves.count()
        entries.append({
            "fen": fen,
            "entropy": round(entropy(normalised), 4),
            "perplexity": round(perplexity(normalised), 2),
            "legal_moves": n_legal,
            # A perfectly uniform policy has perplexity == legal_moves. The
            # ratio below 1 measures how far the policy has departed from
            # uniform, which is more interpretable than entropy alone.
            "uniformity": round(perplexity(normalised) / n_legal, 3),
        })

    mean_perp = (
        float(np.mean([e["perplexity"] for e in entries])) if entries else 0.0
    )
    return {
        "available": True,
        "probes": entries,
        "mean_entropy": (
            round(float(np.mean([e["entropy"] for e in entries])), 4)
            if entries else 0.0
        ),
        "mean_perplexity": round(mean_perp, 2),
    }


def action_coverage(
    policy: DistributionPolicy, n_games: int = 10, seed: int = 0
) -> dict:
    """How much of the action space the policy has actually exercised.

    Distinguishes "untouched because never legal" from "untouched because never
    sampled". Those look identical in the weight table and are the difference
    between a policy that has converged and one that has barely explored.
    """
    import random

    rng = random.Random(seed)
    offered = np.zeros(M.ACTION_SPACE, dtype=bool)
    taken = np.zeros(M.ACTION_SPACE, dtype=bool)

    for _ in range(n_games):
        board = chess.Board()
        while not board.is_game_over(claim_draw=False) and len(board.move_stack) < 150:
            mask = M.legal_move_mask(board)
            for f in np.nonzero(M.legal_from_mask(mask))[0]:
                for t in np.nonzero(mask[f].any(axis=1))[0]:
                    for slot in np.nonzero(mask[f, t])[0]:
                        offered[
                            (int(f) * M.NUM_SQUARES + int(t)) * M.NUM_PROMO
                            + int(slot)
                        ] = True
            move = policy.select(board)
            taken[M.move_to_index(move)] = True
            board.push(move)

    return {
        "offered_slots": int(offered.sum()),
        "taken_slots": int(taken.sum()),
        "action_space": M.ACTION_SPACE,
        "offered_fraction": round(offered.sum() / M.ACTION_SPACE, 5),
        "taken_fraction_of_offered": round(
            taken.sum() / offered.sum(), 4
        ) if offered.any() else 0.0,
    }


def compare_policies(
    policies: dict[str, MovePolicy],
    *,
    games: int = 10,
    max_plies: int = 150,
    seed: int = 0,
) -> dict:
    """Run the same diagnostic over several policies for a side-by-side table.

    This is the shape of the L6 benchmark reduced to its simplest form and
    applied only to L1-era policies, which is useful long before the ensemble
    exists: it makes the gap between "blind", "random" and "one-ply greedy"
    concrete rather than assumed.
    """
    out = {}
    for name, policy in policies.items():
        diag = diagnose_games(
            policy, games, max_plies=max_plies, seed=seed
        )
        entry = diag.summary()
        entry["sharpness"] = policy_sharpness(policy)
        out[name] = entry
    return out


def format_comparison(table: dict) -> str:
    """Render :func:`compare_policies` output as a fixed-width text table.

    Exists because the whole point of these diagnostics is to be read by a
    human deciding what to do next, and a dict dump is not readable. Kept in
    this module rather than the CLI so tests can assert on the rendered form.
    """
    header = (
        f"{'policy':<20}{'inform':>8}{'finished':>10}{'plies':>8}"
        f"{'capt':>7}{'chk':>6}{'contact':>9}{'perplex':>9}"
    )
    lines = [header, "-" * len(header)]
    for name, entry in table.items():
        sharp = entry.get("sharpness", {})
        perp = (
            f"{sharp['mean_perplexity']:.1f}"
            if sharp.get("available")
            else "n/a"
        )
        lines.append(
            f"{name:<20}"
            f"{entry['informative_rate']:>8.2f}"
            f"{entry['finished_rate']:>10.2f}"
            f"{entry['mean_plies']:>8.0f}"
            f"{entry['mean_captures']:>7.1f}"
            f"{entry['mean_checks']:>6.1f}"
            f"{entry['contact_rate_per_100_plies']:>9.1f}"
            f"{perp:>9}"
        )
    return "\n".join(lines)
