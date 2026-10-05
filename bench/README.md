# bench — the round-robin scoreboard

A benchmark that only produces a leaderboard is not a benchmark, it is a
leaderboard. This package aims to produce a leaderboard *plus* the evidence
needed to distrust it: the raw win/loss matrix, the game counts behind every
rating, the standard errors, the pairwise residuals that show where the ordering
breaks, and the number of games that were abandoned rather than decided.

## Running it

```bash
python -m bench                       # full default roster, 6 games per pair
python -m bench --list                # what is in the roster, and is it buildable
python -m bench --only random L1 L2-d2 --games 10
python -m bench --roster ablation     # the L5 seam ablation instead
python -m bench --out results.json    # save the raw matrix
python -m bench --from results.json   # re-render the report, no games replayed
```

`--from` is the important one. The saved JSON holds the *counts*, not the
ratings, so a rating table can always be recomputed from it after a fix to the
fit or a change of prior. Nothing about the ranking is baked into the run.

## What is being measured, and what is not

Every entrant is a `MovePolicy` and nothing else. No level gets a custom scorer,
a head start, or an easier book. The two reference floors exist to make failures
visible:

- `random` — uniform legal mover. **Anything that loses to this is broken**, not
  "relatively weak". A loss here is a bug report.
- `material` — 1-ply greedy on static evaluation. The dumb baseline that any
  real search must beat.

A match is **even games, colours alternating, one distinct book opening per
game, fixed seed**. The book is White-to-move only, and that is load-bearing:
the colour assignment decides who opens, so a Black-to-move entry would invert
it and produce a colour-biased match that looks balanced. `assert_white_to_move`
enforces it and a test watches which policy actually makes the first move.

Unfinished games (ply cap, no-progress cap) are counted, displayed, and
**excluded from every rating**. This is the repo-wide convention: an unfinished
game carries no information about who was better, and scoring one as a draw is
what teaches a naive policy that shuffling is a defence.

## How the rating is fitted

Bradley-Terry maximum likelihood over the whole matrix at once,
`P(A beats B) = 1 / (1 + 10 ** ((r_B - r_A) / 400))`, with a draw as half a win
to each side. The fitted `r` values are Elo on the familiar 400-point scale.

Two things about the fit are worth stating because they are easy to get wrong
and produce a *converged, plausible, wrong* table:

1. **It optimises in logistic units, not Elo points.** `theta = r * ln(10)/400`.
   The gradient of the count-weighted likelihood is then `n_games * (1 - p)`,
   order 1 per game, and a unit-variance prior is commensurate with it. Penalising
   `r ** 2` instead makes the prior ~10^4 times stronger than the data at any real
   rating gap, and every entrant fits at 0.0.
2. **A count is not a probability.** Each observation is `n` games that came out
   the same way, so the term is `n * log(p)` and its derivative carries the `n`.
   Writing `(ss - p)` is correct for one game and wrong for a count; it passes any
   test that only plays single games.

The solver is damped Newton, not gradient ascent. The objective is concave so
Newton is globally convergent and there is no step size to tune — a fixed-step
ascent overshoots the optimum on the first iteration of any lopsided match.

The `prior` (default `1.0`) is a Gaussian precision on each rating. It is
necessary, not decorative: an entrant that won every game has an infinite MLE.
Its size sets the evidence needed for a large rating — with `prior=1`, a clean
sweep fits at roughly +225 Elo for 3 games, +370 for 10, +680 for 100, +911 for
500, which is the `sqrt(log n)` growth it should have.

`tests/test_rating.py` checks these against a bisection oracle computed
independently of the fit, because "it converged" is worthless as an assertion
when every bug in the fit produced a converged answer.

## Reading the table

```
 #  entrant             elo   +/- (se)  games      W-D-L  score%  unf  note
 1  L2-d1             524.4      103.5     18   17-1-0      97.2%    0
 2  material          255.0       84.4     18    9-1-8      52.8%    0
```

- **`elo`** is a difference from the anchor (`random` = 0 by default). Absolute
  Elo is arbitrary; the anchor makes the table readable without changing any
  difference.
- **`+/- (se)`** is the standard error from the observed information. It is on
  the Elo scale. A gap smaller than the two standard errors is not a result.
- **`unf`** counts unfinished games. A large number here means the level is
  avoiding commitment, which is itself a finding.
- **`note`** flags ratings fitted from too few decided games (`provisional`).
  A provisional rating is shown, not hidden — dropping it would hide evidence —
  but it must not be quoted as a result.

`residuals` are the pairs the ratings fit worst. Large residuals are the honest
signal that the matrix is not a ladder. In game-playing benchmarks cycles are
common (A beats B, B beats C, C beats A) and no set of ratings fits all three;
the residual list is what makes that visible instead of reporting a tidy
ordering over a contradictory matrix.

## The L5 ablation

`--roster ablation` runs L5 at depth 2 with every combination of its two model
seams, `use_model_eval` and `use_model_prior`. These are the only two places a
learned model can act inside L2's search, and they do different things:

- **prior** changes move ordering — which move among equals is tried first. It
  cannot change the search *value*, only which of several equally-valued moves
  is returned, and its main payoff is nodes saved.
- **eval** replaces the leaf evaluation, which changes the value at every leaf
  and therefore the answer itself.

If `+eval` is worth more than `+prior`, the model's value head is doing more work
than its policy head, and the next round of training should be aimed there. The
table reports both against the all-off baseline, in Elo, with the `Δ` caveat
that anything inside one standard error is not a result.

### The first ablation result, and how to read it

`bench/ablation-2026-10-02.json` (L5 depth 2, untrained model, 6 games/pair):

```
variant                           elo      delta      se  games
L5-ab-prior                      85.2      +91.4     81.8     16
L5-ab-prior+eval                 29.5      +35.7     87.4     12
L5-ab-blind                      -6.2       +0.0     79.3     16
L5-ab-blind+eval               -108.4     -102.2     92.9     12
```

**Every standard error is larger than every pairwise difference.** The table
looks like an ordering — `prior` best, `blind+eval` worst — and it is not one.
The mechanism is visible in the raw matrix:

- Three of the six matches are draws or 3-3 splits `(blind vs prior 1-1=2, blind
  vs blind+eval 3-3, prior vs prior+eval 3-3)`. The four variants are
  indistinguishable head to head.
- `blind+eval`'s entire negative rating comes from **one** match: it lost 0-6 to
  `L5-ab-prior`. That is the largest residual in the table (observed 0.000,
  expected 0.247).
- `blind+eval vs prior+eval` produced **6 unfinished games out of 6** and took
  772s — nearly four times the next slowest. Two model-eval searches with no
  prior to guide ordering shuffle into the ply cap every time.

So the honest conclusion is a **negative result**: with an untrained model,
neither seam demonstrably improves strength, and a leaf evaluator that replaces
the material heuristic *without* a prior to order moves makes the search both
worse and far slower. That is precisely what the value-head work (Task #9) is
meant to change, and this table is its baseline.

The lesson for the reader: a rating table without its evidence columns and its
matrix is a made-up ordering. The differences here are 35-100 Elo on standard
errors of 79-93 across 12-16 decided games per entrant. Nothing is established.

## The L3.5 experiment: three arms, one budget, one free variable

`L3.5` is the rung that varies **the shape of the computation** at fixed input,
fixed output and (nominally) fixed budget. It exists because every other rung
varies *how much information* the learner gets (L1 blind, L3 informed), *how
much search it does* (L2, L5), or *which backend it runs on* (L4). None of them
isolates architecture.

Three arms are registered, and all three must be present or the comparison
measures nothing:

| arm | tier | params | what it varies |
|---|---|---|---|
| `L3` | ladder | 7609 (100%) | baseline: separate weight row per square |
| `L3.5` | ladder | 787 (10.3%) | the per-square function is **weight-tied** |
| `L3-flat` | ablation | 1573 (20.7%) | **control**: no spatial structure at all |

`L3-flat` is the control and is not optional. Without it, a win by L3.5 cannot
be distinguished from "any change from L3 helps". It is a flat linear model over
the flattened encoder output with a rectifier; measured, its origin head is
*exactly constant* across squares (`std = 0.0`), which is precisely the spatial
structure it gives up, and it is only viable at all because one hidden unit over
a 24x64 input already costs 1536 weights.

### The budget is not literally equal, and that is stated rather than hidden

Spec §3 asks the three arms to land within +-10% of L3's 7609 parameters, then
its own layout table lists **787**. Those two demands contradict each other and
no honest implementation satisfies both. The arms above land at 100% / 10.3% /
20.7%.

Closing the gap would require the tied arm to run a ReLU of width ~649 over a
**9-dimensional** context input -- 5841 parameters of deliberate
over-parameterisation chosen to move a number rather than to model anything,
which is the "quietly match by padding" the same paragraph forbids. So the arm
is lean, `h_ctx` remains the budget knob, and the claim it supports is the
honest one: *same task, same features, same signal, architecture the only free
variable, at the width each architecture can actually use.*

### What the tying costs, found before the match was ever run

Two results came out of building the level, and both are recorded here because
they change how the eventual table must be read.

**1. The spec's `lr_shared = lr/64` silently disables the tied heads.**
A tied row receives credit from every square, so its rate must be normalised --
that part of the spec is right. But at the default `lr=0.05` and a typical
credit of `0.5`, the step is `QUANT * 0.05/64 * 0.5 = 0.4` quantisation grid
units, and `np.round` sends that to **zero**. Measured: after 30 full games and
14,400 credit events, `w_shared`, `w_dst_shared`, `w_geom` and `promo` were
bit-identical to their initialisation, while the context path -- which runs at
the full rate -- moved by 1.9. The level as literally specified was untrainable
in its main pathway, and the ablation would have blamed weight tying for what
was a fixed-point artifact.

The fix is error feedback: sub-grid remainders are banked per tensor and added
to the next update, so `~2.5` events of `0.4` units produce one real step. The
*rate* is exactly what the spec asked for; the rounding just stops throwing
updates away. After the fix all eight tensors move. Asserted by
`test_sub_grid_updates_are_buffered_not_discarded` and
`test_undivided_learning_rate_actually_diverges`.

**2. Tying makes some choices genuinely ambiguous, and the ambiguity is often
chess-correct.** `L3` gives the b1 and g1 knights *different* origin logits,
because it owns an independent weight row per square; greedy play can therefore
prefer one. `L3.5` applies one row to both, so their origin logits are **exactly
equal**, and greedy selection falls back to array order. The mirror-image
position may then pick the other member of the tie.

That is not a canonicalisation bug. Nc3 and Nf3 are genuinely equivalent from
the opening, and any symmetric evaluation ties them -- `L3.5` is *right* to tie
and `L3` only breaks it by initialisation noise. The consequence is real
nonetheless: `L3`'s test
`test_mirrored_positions_select_mirrored_moves_when_greedy` cannot be ported to
`L3.5`, and the invariant that replaces it is that *probabilities* mirror
exactly (residual 1.5e-08, same standard as L3), with either tied move acceptable
where a tie exists.

The deeper limitation the investigation surfaced is more interesting than the
tie itself. `_geom_features` carries `dest_rank` but **no destination-file
feature**, and it takes `abs(tf - ff)` for the file difference. So `b1->c3` and
`g1->h3` produce byte-identical geometry vectors, and because two knights also
have identical channel vectors and the tied destination row is origin-blind,
the model cannot express "a knight belongs near the centre, not on the rim".
`L3` can, because `W_to_dst` is indexed by destination square. This is the first
place where weight tying is *observably* weaker, it is a missing-feature
limitation rather than a tying limitation per se, and it is the concrete form of
the hypothesis's stated weak point in spec §1: a per-square function cannot see
a fact that is about the *relationship* between squares.

It is reported here rather than patched around, because adding a destination
file feature would change the feature set and break the fixed-input premise the
whole comparison rests on. Whether the limitation costs measurable strength is
exactly the question the match is for.

## L6: the ensemble, and why its weights cannot come from the bench

`L6` is not a learner. It has no parameters of its own — only *weights over its
members*, which are derived from played games. That makes it structurally
different from every other entrant, and the difference has to be handled rather
than papered over.

### Two objects, not one

| object | built by | weights | reported as |
|---|---|---|---|
| `L6` (bench row) | `levels._l6` | uniform (no ledger yet) | a matrix row, like any other |
| weighted L6 | `bench/ensemble_run.py` | fitted from a ledger | the driver's JSON + table |

The bench row is a **uniform mixture** of `L1`, `L3` and `L2-d2`. That is the
honest thing to put in the matrix, because every other learner in the matrix is
likewise untrained — L1's own row is an untrained L1. The fitted L6 is a separate
artefact, produced by a separate script, because its weights depend on games that
have to be played *before* it can be evaluated. Folding the fit into the bench
would make the number circular: the weights would be chosen using the same games
that are then quoted as L6's strength.

### How the weights are derived

`bench/ensemble_run.py` plays calibration games between the members, then asks
every member, in every recorded position, which move it would play. A member's
opinion is scored against the move the **eventual winner's side actually played**
in that position. Unfinished games — the majority here — have no winner, so they
are scored on whether the move hung material instead.

That is a *consistency* score, not a strength score, and it is the right target
for weighting: an ensemble wants the member whose judgement agrees with outcomes,
not the one that happened to win this sample. Accuracy is measured relative to
`chance_equivalent` (0.5 by default), which is the *unit* of accuracy: a member
at exactly chance is worth one mean vote.

### The first fitted result, and it is a warning

Calibration with 4 games per pair, 24 positions per game (72 scored opinions):

```
member       asked  correct  accuracy   weight
----------------------------------------------
L1              24       14     0.583   0.2745
L3              24       15     0.625   0.2941
L2-d2           24       22     0.917   0.4314
```

The ordering is exactly what the ladder predicts — depth-2 search is the member
whose judgement agrees with outcomes far more often than the untrained blind
ones. **And the fitted weights still make the ensemble worse.**

Measured against the greedy-material floor:

| configuration | as White | as Black |
|---|---|---|
| fitted weights (0.27 / 0.29 / 0.43) | unfinished, 100 plies | **1-0 checkmate, 59 plies** |
| `L2-d2` given all the weight | **1-0 checkmate, 39 plies** | (same policy) |
| `L2-d2` alone, no ensemble | **1-0 checkmate, 39 plies** | 0-1 checkmate, 40 plies |

The two blind members together hold **0.569** of the weight, so they outvote the
one member that can see material. The failure is visible in the opening the
fitted ensemble plays as White:

```
b4 Nc6 d3 Nxb4 Nc3 Nxd3+ Qxd3 ...
```

It hangs a pawn, then another, then loses a third for a queen — the signature of
policies with no material term at all. The accuracy metric does not punish this,
because a blind policy still *agrees with the winner's move* on the many
positions where one move is obvious. Agreement is not competence, and this is
the concrete form of that gap.

**This is a negative result about the weighting rule, not about ensembles.** It
says the accuracy proxy is too weak to separate "sees the board" from "does not",
and therefore that a fitted ensemble can be *worse* than its best member.

### Two scoring bugs made that result worse than it needed to be

Both were found after the table above was produced, and both made the ledger
report **fake evidence** — not merely noisy evidence.

**1. `_move_is_safe` was dead code that returned `True` always.** It compared
`evaluate()` before and after a move against a threshold of −100. Because
`evaluate` is material + piece-square only it cannot see the opponent's reply,
so the "swing" it measures is just the mover's own material change. Over 11,691
sampled legal moves the observed swing ranged only from **−50 to +99957** — the
minimum never reached −100, so the rule accepted **100%** of moves, including
200/200 random ones. Every unfinished game (the majority in this benchmark) gave
every member a perfect score.

Replaced with a `recapture_risk`-based test: a move is unsafe when the cheapest
enemy attacker is worth less than the piece now on the target square and the move
did not already win that much. It is genuinely alive now, but its power is
**phase-dependent** and worth stating precisely, because it bounds how much the
unfinished games can contribute:

```
positions sampled         moves accepted   rejected
opening (0 plies)          1200/1200        0.0%
midgame (20 plies)         1657/1904       13.0%
late    (40 plies)         1545/1833       15.7%
```

In the opening nothing is attackable yet, so *every* move is "safe" and the
fallback is still blind — exactly the case the old dead code was. In real
blind-vs-blind games, which run long, the picture is better: the move-level safe
rate is 0.80–0.85 and positions where *every* move is safe are only 8–20%. So
unfinished games do carry signal — but far less than decisive ones, where the
agreement branch spreads members over the whole 0–1 range (a member replaying
the played moves scores 1.00; a random one ~0.05) while this fallback squeezes
them all into 0.80–0.85. The headline result below leans on the unfinished
games; that is the reason to trust it less than its weight suggests.

**2. The losing side's plies were scored as free hits.** The rule read
`correct = (move != played)` when the side to move was the eventual loser, on the
reasoning that agreeing with a losing move is not a correct prediction. That
inverts the question: the member is asked *what it would play*, not *whether the
played move was good*. Scoring a disagreement as a hit made every ply the loser
moved a guaranteed hit. Measured, a **purely random** member scored **0.429** on
a 7-move decisive game, where chance is ~0.05. After the fix the same member
scores **0.046**.

### The corrected result, which is still negative

With both bugs fixed the same calibration gives:

```
member       asked  correct  accuracy   weight
----------------------------------------------
L1              24       10     0.417   0.2703
L3              24        9     0.375   0.2432
L2-d2           24       18     0.750   0.4865
```

The blind members lost a third of their accuracy once their free hits were
removed, and the search member's share rose from 0.431 to 0.487. That is the
metric doing its job. **But the blind pair still holds 0.514** — a majority over
the one member that can see — so the fitted ensemble still fails to beat
`L2-d2` alone (unfinished both colours, against checkmate in 39/40).

So the honest conclusion is sharper than "the metric is weak". Under the default
(absolute) rule, **the weighting lets a blind majority outvote a seeing minority**,
and the ensemble is reliably worse than its best member. The structural fix for
that — make the comparison relative so two mediocre members cannot pool to beat one
competent one — **is implemented** as `EnsembleConfig.relative=True` and exposed via
`--relative` (see the section above). The other suggested fix, *weight by something
that has a sign for material*, is **not** done: the ledger still measures agreement
with the played move, not material gain. That remains the open lever if a future
member (e.g. a trained L5 value head) is to be judged on what it actually wins
rather than on whether it agrees with whoever moved.

The `L2-d2`-only row is the control that proves the machinery is sound: the
ensemble reproduces its sole weighted member exactly, so the combination is not
what costs the strength.

### The relative fix (implemented, opt-in)

The structural flaw named above — *two mediocre members pooling their weight to
outrank one competent one* — is exactly what `EnsembleConfig.relative=True`
addresses. Instead of scaling each member's accuracy against a fixed
`chance_equivalent`, it scales against the **mean accuracy of the set**. A member
below the mean becomes worth nothing, so below-mean members cannot sum to more
than the one that can see.

Run it with:

```
python -m bench.ensemble_run --relative
```

On the same calibration the corrected table becomes:

```
member       asked  correct  accuracy   weight (relative)
--------------------------------------------------------
L1             144       52     0.361   0.0161
L3             144       60     0.417   0.0161
L2-d2          144      111     0.771   0.9677
```

The blind pair's combined share drops from 0.514 to 0.032, and the search member
takes 0.968. Head-to-head against `L2-d2` alone, the relative ensemble now plays
*like* `L2-d2` (sometimes draws, sometimes loses — the same spread the control
produces), whereas the absolute ensemble **consistently lost** to it. So the fix
converts the negative result ("ensemble worse than its best member") into a
neutral one ("ensemble equals its best member").

It does **not** make the ensemble *better* than its best member, and it was never
going to. L1 and L3 are blind/weak learners; they are not *complementary* to a
search, so there is no surplus to harvest by combining them. The honest reading
of the benchmark is now: **a naive ensemble of naive learners is, at best, its
strongest component, and only if the weighting rule refuses to let the weak
majority dilute it.** Relative weighting is the rule that refuses. It is off by
default so the documented negative result (and every test that pins it) is
preserved; the absolute rule remains the "as specified" behaviour.

## Layout

| file | responsibility |
|------|----------------|
| `levels.py` | the roster, the opening book, lazy torch-safe construction |
| `runner.py` | playing matches; produces raw counts and nothing else |
| `rating.py` | Bradley-Terry fit; the only place ratings are computed |
| `report.py` | formatting; ASCII, fixed-width, evidence travels with numbers |
| `ensemble_run.py` | calibrating L6's weights from played games and writing the fit |
| `__main__.py` | the CLI |

The separation matters: the runner never computes a rating, so the ranking can
always be recomputed from saved counts without replaying a game.
