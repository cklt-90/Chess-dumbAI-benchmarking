"""Turning a win matrix into a strength ranking.

Why not just report the raw matrix
----------------------------------
A round-robin of *k* entrants produces *k*(k-1)/2 match results, and the matrix
containing them is the honest artefact -- it is the data, and it should always
be emitted. But it is not readable as a ranking, for two reasons:

1. **Not all wins are equal.** Beating the random mover is worth nothing;
   beating L2-d3 is worth a lot. A raw win count cannot tell the difference, so
   it rewards entrants that happened to draw easy opponents.
2. **Transitivity is not guaranteed.** In game-playing benchmarks it is common
   for A to beat B and B to beat C while C beats A. A ranking that assumes a
   total order will misreport this as noise. The fitted model at least exposes
   it, because the residuals get large where the ordering breaks down.

The model
---------
The standard one, and the right one here: **Bradley-Terry**, which says the
probability that A beats B is

    P(A beats B) = 1 / (1 + 10 ** ((r_B - r_A) / 400))

with a draw handled as half a win to each side. That is exactly the logistic
model behind Elo, so the fitted ``r`` values *are* Elo ratings on the familiar
400-point scale, and they are fitted from the whole matrix at once rather than
updated pairwise.

Fitting is **maximum likelihood by damped Newton**, plus a small L2 pull towards
zero. Three details that matter:

* **The prior is necessary, not decorative.** An entrant that won every game
  has an infinite MLE. Without regularisation the fit runs to infinity and the
  rating is meaningless -- and in a small round-robin that case is common. The
  prior is also what identifies the rating *level*: only differences are
  constrained by the data, and the penalty on the parameters is what pins the
  mean.
* **Newton, not gradient ascent.** The objective -- a logistic log-likelihood
  plus a quadratic prior -- is concave, so Newton is globally convergent and
  there is no step size to tune. A fixed-step ascent does not merely need
  tuning, it *overshoots the optimum on the first iteration* of any lopsided
  match, because the gradient at zero is ``n_games * 0.5`` and for a 10-game
  sweep that is a larger step than the optimum itself.
* **Ratings are reported relative to the field mean, then anchored so
  ``random`` sits at 0.** Absolute Elo is arbitrary; the anchor makes the table
  readable ("50 points above random") without changing any difference.

Where the parameters actually live
----------------------------------
The fit optimises in **logistic units**, not Elo points, and this is the one
design decision that is easy to get wrong in a way the numbers will not reveal.

Define the natural parameter ``theta = r * ln(10) / 400``. Then
``P(A beats B) = 1 / (1 + exp(-(theta_A - theta_B)))``, an ordinary logistic
model with no scale factor anywhere. Two consequences:

* The gradient of the count-weighted log-likelihood with respect to ``theta``
  is ``n_games * (1 - p)`` -- **order 1 per game**, independent of the Elo
  scale. The prior, being a unit-variance L2 pull, is then also order 1 and the
  two terms are commensurate, so the Newton system is well-conditioned.
* If instead you optimise ``r`` directly and penalise ``r ** 2``, the prior is
  measured in *Elopoints squared* and swamps the likelihood by four orders of
  magnitude: at a 100-point gap the penalty is 2500 while three games of
  evidence contribute about 2. The optimum collapses to ~0 and every entrant
  fits at zero. This was not a tuning error, it was a units error, and it
  produced a perfectly converged all-zeros table that looked like a bug in the
  tournament.

So: optimise ``theta``, regularise ``theta``, convert to ``r = theta / c`` only
at the end (for reporting, the standard errors, and the anchor). The Elo scale
``c = ln(10)/400`` appears in exactly two places -- the sigmoid that converts a
rating difference to a probability, and the final division -- and nowhere in the
optimiser.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# The Elo scale factor. 400 points means a 10:1 expected score, by convention,
# and it is what makes the fitted numbers comparable to published ratings.
_ELO_SCALE = 400.0

# The conversion between Elo points and the logistic (natural) parameter:
#     theta = r * _LOGISTIC_SCALE,   r = theta / _LOGISTIC_SCALE
# so that P(A beats B) = sigmoid(theta_A - theta_B) = 1/(1+10**(-(r_A-r_B)/400)).
#
# Everything internal works in theta. See the module docstring for why mixing
# the two units is the failure mode this file had to be written twice to learn.
_LOGISTIC_SCALE = float(np.log(10.0)) / _ELO_SCALE


def _sigmoid(delta: np.ndarray) -> np.ndarray:
    """Logistic expected score for a *logistic-unit* difference.

    Takes ``theta_A - theta_B`` (not Elo points) and is written with the
    numerically stable two-branch form, because the naive
    ``1 / (1 + exp(-x))`` overflows for large negative ``x`` -- which is exactly
    where a lopsided match sits, and where an overflow turns the whole fit into
    ``nan`` under ``filterwarnings = ["error::RuntimeWarning"]``.
    """
    delta = np.asarray(delta, dtype=np.float64)
    out = np.empty_like(delta)
    pos = delta >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-delta[pos]))
    ez = np.exp(delta[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def _elo_to_theta(r: np.ndarray) -> np.ndarray:
    return np.asarray(r, dtype=np.float64) * _LOGISTIC_SCALE


def _theta_to_elo(theta: np.ndarray) -> np.ndarray:
    return np.asarray(theta, dtype=np.float64) / _LOGISTIC_SCALE


@dataclass
class MatchRecord:
    """One match between two entrants, as the fit consumes it.

    Counts, not just a total. An earlier version stored only ``score_a`` (the
    usual 1/0.5/0 convention) and tried to recover wins and draws from it
    afterwards, which is impossible -- the total cannot distinguish "2 wins"
    from "1 win and 2 draws". The per-outcome counts are kept so the table can
    report them directly and the fit never has to guess.

    Unfinished games (ply cap, no-progress cap) are held separately and
    **excluded** from the fit. That is the repo-wide convention -- an
    unfinished game says nothing about who was better, and scoring it as a draw
    is what teaches a naive policy that shuffling is neutral -- and it has to be
    enforced here too, because the fit will happily consume a fake draw.
    """

    a: str
    b: str
    wins: int = 0           # games A won
    draws: int = 0          # games drawn
    losses: int = 0         # games A lost
    unfinished: int = 0     # excluded from the fit and from `played`

    @property
    def played(self) -> int:
        """Decided games only. Unfinished games are not evidence."""
        return self.wins + self.draws + self.losses

    @property
    def score_a(self) -> float:
        return self.wins + 0.5 * self.draws

    @property
    def win_rate_a(self) -> float:
        return self.score_a / self.played if self.played else 0.0


@dataclass
class Rating:
    name: str
    elo: float
    games: int
    wins: int
    draws: int
    losses: int
    unfinished: int
    stderr: float = 0.0
    notes: str = ""

    @property
    def points_pct(self) -> float:
        played = self.wins + self.draws + self.losses
        return (self.wins + 0.5 * self.draws) / played if played else 0.0


@dataclass
class FitResult:
    ratings: list[Rating]
    log_likelihood: float
    iterations: int
    converged: bool
    anchor: str
    # Pairwise residuals: (a, b, observed_score, expected_score, games). Large
    # residuals are where the Bradley-Terry assumption is failing -- usually a
    # genuine cycle rather than noise, which is why they are surfaced.
    residuals: list[tuple[str, str, float, float, int]] = field(default_factory=list)


def expected_score(r_a: float, r_b: float) -> float:
    """Bradley-Terry expected score of A against B, ratings in Elo points."""
    return float(_sigmoid(np.array([(r_a - r_b) * _LOGISTIC_SCALE]))[0])


def fit_bradley_terry(
    records: list[MatchRecord],
    names: list[str],
    *,
    anchor: str = "random",
    prior: float = 1.0,
    iterations: int = 200,
    tol: float = 1e-9,
    include_unfinished_as_draws: bool = False,
) -> FitResult:
    """Maximum-likelihood Elo ratings from a set of match records.

    ``records`` should contain one entry per (unordered) pair. A symmetric pair
    is not required: ``play_match`` already alternates colours, so a single
    record per pair is the correct granularity and duplicating it in both
    directions would double-count.

    ``prior`` is the strength of the pull towards zero, in logistic units -- it
    is a Gaussian precision on each rating. It bounds the fit, and its size sets
    how much evidence is needed to reach a large rating. With ``prior=1`` the
    calibration is:

    ==================  ==================
    clean sweep of...   fits at about
    ==================  ==================
    3 games             +225 Elo
    6 games             +306 Elo
    10 games            +370 Elo
    20 games            +460 Elo
    100 games           +680 Elo
    500 games           +911 Elo
    ==================  ==================

    which is the ``sqrt(log n)`` growth a unit-variance prior is supposed to
    give. Raise ``prior`` for a small roster, lower it for a large one.

    There is deliberately **no step size**. The objective is concave, so the fit
    uses damped Newton and is globally convergent; a fixed-step ascent has to be
    tuned and still overshoots on the first iteration of a lopsided match. See
    ``hessian`` for the derivation.
    """
    if not records:
        return FitResult([], 0.0, 0, True, anchor)

    index = {n: i for i, n in enumerate(names)}
    k = len(names)
    theta = np.zeros(k, dtype=np.float64)

    # Flatten to (i, j, score, weight) rows once. The draw case contributes half
    # a win to each side, which under Bradley-Terry is exactly the same as
    # weighting a win and a loss each by 0.5 -- so a drawn match is expressed as
    # two half-weight observations rather than a special-cased outcome.
    obs: list[tuple[int, int, float, float]] = []
    for rec in records:
        # Unfinished games carry no information about who was better; by default
        # they are excluded from the fit. When ``include_unfinished_as_draws`` is
        # set (the H3 experiment), each unfinished game is folded in as a half-win
        # to each side -- the exact "score a ply-capped game as a draw" trap H3
        # warns against -- so the fit can quantify how much that choice moves the
        # line. The default path is unchanged: ``eff_draws == rec.draws`` and an
        # all-unfinished record is still skipped.
        eff_draws = rec.draws + (rec.unfinished if include_unfinished_as_draws else 0)
        if rec.played <= 0 and eff_draws <= 0:
            continue
        i, j = index.get(rec.a), index.get(rec.b)
        if i is None or j is None or i == j:
            continue
        if rec.wins:
            obs.append((i, j, float(rec.wins), 1.0))
        if rec.losses:
            obs.append((j, i, float(rec.losses), 1.0))
        if eff_draws:
            obs.append((i, j, float(eff_draws), 0.5))
            obs.append((j, i, float(eff_draws), 0.5))
    if not obs:
        return FitResult([], 0.0, 0, True, anchor)

    ii = np.array([o[0] for o in obs], dtype=np.int64)
    jj = np.array([o[1] for o in obs], dtype=np.int64)
    ss = np.array([o[2] for o in obs], dtype=np.float64)
    hh = np.array([o[3] for o in obs], dtype=np.float64)

    def log_likelihood(par: np.ndarray) -> float:
        """Weighted Bernoulli log-likelihood plus the unit-variance prior.

        Both terms are in logistic units, so they are comparable term by term:
        the prior contributes ``0.5 * prior * theta**2`` per parameter, which is
        the log of a N(0, 1/prior) density -- i.e. ``prior`` is a precision.
        """
        p = np.clip(_sigmoid(par[ii] - par[jj]), 1e-300, 1.0)
        ll = float((hh * ss * np.log(p)).sum())
        centred = par - par.mean()
        ll -= 0.5 * prior * float((centred ** 2).sum())
        return ll

    def gradient(par: np.ndarray) -> np.ndarray:
        """Gradient of the count-weighted log-likelihood, in logistic units.

        Each observation is ``ss`` games that all came out the same way, so the
        term is ``ss * log(p)`` and its derivative wrt ``theta`` is
        ``ss * (1 - p)``. The count multiplies the *residual*, giving
        ``n_games * (1 - p)`` -- order 1 per game, independent of the Elo scale,
        which is the whole reason for working in logistic units.

        Writing this as ``(ss - p)`` instead -- treating ``ss`` as a Bernoulli
        success count against a single probability -- is wrong by
        ``ss * p`` worth of gradient and moves the fixed point. It is a
        tempting simplification because for ``ss = 1`` the two agree exactly,
        so it passes any test that only plays single games.
        """
        p = _sigmoid(par[ii] - par[jj])
        g = np.zeros(k, dtype=np.float64)
        contrib = hh * ss * (1.0 - p)
        np.add.at(g, ii, contrib)
        np.add.at(g, jj, -contrib)
        g -= prior * (par - par.mean())  # the prior, also order 1
        return g

    def hessian(par: np.ndarray) -> np.ndarray:
        """Negative Hessian of the log-likelihood (the observed information).

        The second derivative of ``ss * log sigmoid(theta)`` is
        ``ss * p * (1 - p)`` -- the count multiplies here too, for the same
        reason it multiplies the gradient. Both terms are in logistic units, so
        no Elo scale factor appears.

        There is a gauge freedom: only rating *differences* are identified, so
        the raw likelihood Hessian is singular along the all-ones direction. The
        prior is what removes it -- an L2 penalty on the parameters is exactly
        what makes the level identified -- and it does so by construction, since
        ``prior > 0`` makes the matrix positive definite. Do **not** try to
        handle the gauge by projecting the Hessian into the sum-zero subspace
        afterwards: that deletes the very diagonal the prior contributed and
        leaves ``[[w, -w], [-w, w]]``, which is singular, which makes the
        Newton solve fail and silently degrade the whole fit to un-scaled
        gradient steps. (That fallback was the last bug in this function.)
        """
        p = _sigmoid(par[ii] - par[jj])
        w = hh * ss * p * (1.0 - p)
        h = np.zeros((k, k), dtype=np.float64)
        np.add.at(h, (ii, ii), w)
        np.add.at(h, (jj, jj), w)
        np.add.at(h, (ii, jj), -w)
        np.add.at(h, (jj, ii), -w)
        h += prior * np.eye(k)
        return h

    # Damped Newton, in logistic units.
    #
    # The objective -- a logistic log-likelihood plus a quadratic prior -- is
    # **concave**, so Newton's method is globally convergent from any start
    # provided each step is damped to remain feasible. That matters here because
    # the obvious alternative does not work: plain gradient ascent with a fixed
    # step *overshoots the optimum on the very first iteration* for any lopsided
    # match (the gradient at zero is ``n_games * 0.5``, which for a 10-game
    # sweep is a step larger than the optimum itself), and a naive backtracking
    # line search then crawls back the wrong way at a step of 1e-4, converging
    # to nothing over 20000 iterations. Newton has no step size to get wrong.
    #
    # The damping factor just guarantees the objective never decreases: start at
    # the full Newton step and halve until it improves. Because the problem is
    # concave the full step is almost always accepted, so this is a formality
    # rather than the load-bearing part.
    #
    # History of this function, so the next person does not repeat it. Four
    # separate bugs, each of which produced a plausible, converging, wrong
    # table -- which is why the tests below assert against analytic values
    # rather than merely that the fit terminates:
    # * Elo-unit gradient ascent with an ``r ** 2`` prior: the prior is ~10^4
    #   times stronger than the data at any real rating gap, so everything fits
    #   at 0.0. (Units error -- see the module docstring.)
    # * Elo-unit ascent with the ratings re-centred each iteration: re-centring
    #   a mean-zero prior gradient is a no-op on it, so the prior's contribution
    #   is deleted and a 10-0 sweep "converges" to 3127 Elo.
    # * Projecting the Hessian into the sum-zero subspace to fix the gauge:
    #   deletes the prior diagonal, leaves a singular matrix, makes the solve
    #   raise, and silently degrades Newton to un-scaled gradient steps.
    # * Gradient and Hessian written as ``(ss - p)`` and ``p*(1-p)``: correct
    #   for a single game and wrong for a count, since the term is
    #   ``ss * log(p)`` whose derivative carries the count. Passes any test that
    #   only plays one game per pair.
    #
    # The current form: damped Newton on a concave objective, so there is no
    # step size to tune and convergence is single-digit iterations.
    ll = log_likelihood(theta)
    converged = False
    used = 0
    prev = theta.copy()
    for it in range(1, iterations + 1):
        g = gradient(theta)
        h = hessian(theta)
        try:
            step_dir = np.linalg.solve(h, g)
        except np.linalg.LinAlgError:                # pragma: no cover
            # Fall back to the pseudo-inverse, never to the raw gradient: a raw
            # gradient step has the wrong units (it is a direction, not a step)
            # and silently turns Newton into badly-scaled descent, which is how
            # this function previously "converged" to the wrong answer.
            step_dir = np.linalg.pinv(h) @ g
        scale = 1.0
        for _ in range(60):
            trial = theta + scale * step_dir
            cand = log_likelihood(trial)
            if cand >= ll:
                break
            scale *= 0.5
        else:                                        # cannot improve at all
            converged = True
            used = it
            break
        drift = float(np.abs(trial - theta).max())
        theta = trial
        ll = cand

        used = it
        prev = theta.copy()
        if drift < tol:
            converged = True
            break

    r = _theta_to_elo(theta)
    r = r - r.mean()
    if anchor in index:
        r = r - r[index[anchor]]

    # Standard errors from the diagonal of the observed information.
    #
    # This is the one place the Elo scale legitimately reappears: the
    # information is the second derivative of the *logistic* log-likelihood, so
    # it must be converted back to Elo units before taking the square root.
    # Working in theta and forgetting the conversion understates every standard
    # error by 174x, which would make a 6-game match look like a precise
    # measurement.
    p = _sigmoid(theta[ii] - theta[jj])
    info = np.zeros(k, dtype=np.float64)
    np.add.at(info, ii, hh * ss * p * (1.0 - p))
    np.add.at(info, jj, hh * ss * p * (1.0 - p))
    info += prior
    se = 1.0 / (_LOGISTIC_SCALE * np.sqrt(np.maximum(info, 1e-300)))

    agg = {n: dict(games=0, wins=0, draws=0, losses=0, unfinished=0)
           for n in names}
    for rec in records:
        if rec.a in agg:
            agg[rec.a]["games"] += rec.played
            agg[rec.a]["wins"] += rec.wins
            agg[rec.a]["draws"] += rec.draws
            agg[rec.a]["losses"] += rec.losses
            agg[rec.a]["unfinished"] += rec.unfinished
        if rec.b in agg:
            agg[rec.b]["games"] += rec.played
            agg[rec.b]["wins"] += rec.losses       # A's losses are B's wins
            agg[rec.b]["draws"] += rec.draws
            agg[rec.b]["losses"] += rec.wins
            agg[rec.b]["unfinished"] += rec.unfinished

    ratings = [
        Rating(
            name=n,
            elo=float(r[index[n]]),
            stderr=float(se[index[n]]),
            games=agg[n]["games"],
            wins=agg[n]["wins"],
            draws=agg[n]["draws"],
            losses=agg[n]["losses"],
            unfinished=agg[n]["unfinished"],
        )
        for n in names
    ]
    ratings.sort(key=lambda x: -x.elo)

    residuals = []
    for rec in records:
        if rec.played <= 0:
            continue
        i, j = index.get(rec.a), index.get(rec.b)
        if i is None or j is None:
            continue
        residuals.append((
            rec.a, rec.b, rec.win_rate_a,
            expected_score(r[i], r[j]), rec.played,
        ))
    residuals.sort(key=lambda t: -abs(t[2] - t[3]))

    return FitResult(ratings, ll, used, converged, anchor, residuals)
