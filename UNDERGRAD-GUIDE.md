# A Year-2 Undergrad's Guide to `chess-rl-bench`

*Written 2026-10-06. This is the "why" document. If you want the "how do I run
it", read `ONBOARDING.md` instead. If you want the list of open questions, read
`HYPOTHESES.md`. Read this one first. §11 places the project in the external
literature (toy-scale RL, self-play collapse, search vs learning, negative
results).*

---

## 0. One sentence

This repo is not trying to build a good chess engine. It is trying to **measure
how much strength you get from each ingredient of machine learning**, one
ingredient at a time, on the same toy problem, at a scale small enough that you
can hold the whole thing in your head.

That last clause is the entire point. Everything below follows from it.

---

## 1. The research question, stated properly

Imagine you want to build a chess player out of a neural network. You will
immediately face a dozen design decisions, and the textbooks will give you
answers like "use a factorised action space", "use search", "use a value head",
"train by self-play". Those answers are mostly true — but they are true at
*Google-scale*. AlphaZero had 5000 TPUs and millions of games. A lot of what
the literature reports is about what happens when you have that much compute.
Almost nobody reports what happens when you have a laptop and ten minutes.

So the question here is:

> **At the "toy" scale — a few thousand parameters, a few hundred games,
> numpy on a CPU — which of these standard ingredients actually still help, and
> which of them are promises the literature quietly makes only to people with
> data centres?**

A secondary, sharper version:

> **When one of these ingredients *stops* helping, where exactly is the
> boundary? Can we point at the specific mechanism that breaks?**

That second framing is what makes this a research project rather than a
homework exercise. "L3 beats L1 with training" is a homework answer.
"The self-play training signal collapses to exactly zero reward at this scale,
and here is why" is a research finding.

### Why a negative result is the good outcome

The register (`HYPOTHESES.md`) says this explicitly: *"a negative is a result,
not a failure — several are expected to fail, and those are the informative
ones."*

This is a genuine methodological stance, not a consolation prize. If you run an
experiment where 20 of your 20 games end in a draw and you conclude "the
experiment failed", you have learned nothing. If you run the same experiment and
conclude "at these caps, blind self-play drifts to bare kings every single time,
so the outcome-RL signal is identically zero — the arm cannot learn by
construction", you have learned something that a dozen other people would have
wasted a GPU-week discovering. The repo has done exactly this (see §6). **Your
job as a newcomer is to be the person who notices the second thing.**

---

## 2. The mental model: a ladder

The repo is built around a **ladder of learners**, each rung adding exactly one
idea. They all share the same interface, the same features, and the same game
loop, so you can compare them fairly.

```
L6  ensemble          blend/vote over all of the below, weighted by track record
L5  guided search     alpha-beta whose move ordering + leaf eval come from L4
L4  torch net         the SAME function as L3, in a different backend (parity test)
L3  perceptron        learns from the board; minimax feedback signal
L3.5 weight-tied      per-square heads sharing weights (the representation axis)
L2  alpha-beta        real search: negamax + TT + iterative deepening + quiescence
L1.5 trainer          the LEARNING for L1: diminishing credit, pruning
L1  blind softmax     a probability table with NO board input at all
L0  foundation        bitboards, move masks, encoding, evaluation, the game loop
```

Read that bottom-to-top as "each rung adds one idea". That is the design
principle: **one free variable at a time**. If L3 beats L1, and L3 differs from
L1 only by seeing the board, then you have isolated the value of *seeing the
board*. If you changed three things at once you would learn nothing.

### The two interfaces everything implements

This is the single most important piece of code to understand. Two protocols:

```python
# Search levels (L2, L5) — pick one move
class MovePolicy:
    def select(self, board: chess.Board) -> chess.Move: ...

# Learners (L1, L3, L3.5, L4, L6) — emit a probability over all moves
class DistributionPolicy:
    def move_distribution(self, board: chess.Board) -> np.ndarray: ...
```

The game loop (`play_game`) accepts either. That is what makes the whole ladder
comparable: same referee, same clock, same starting positions, different player.
A `DistributionPolicy` is just a `MovePolicy` that samples from its own
distribution.

**And here is the trick that keeps the code small.** Every learner is split into:

- a **`Scorer`** — three little functions that produce raw numbers (logits)
  for the three factors of a move: *which square to move from*, *which square to
  move to*, *which piece to promote to*. A blind scorer returns constants. A
  perceptron returns board-dependent values. A torch net returns tensors.
- the shared **`FactoredSoftmaxPolicy`** — which takes those logits, masks out
  illegal moves, and turns them into a real probability distribution.

So "writing a new learner" almost always means "writing a new `Scorer`" — maybe
40 lines — not a new policy. This is why L3 and L4 can be guaranteed to be the
same function: they plug different scorers into the identical policy.

---

## 3. The five ideas, explained as if you haven't seen them

Each of these is a rung of the ladder. I'll give the intuition, not the code.

### 3.1 L1 — the blind policy (and why blindness is the point)

A chess move is a triple `(from_square, to_square, promotion)`. Naively you
could learn a probability for each of the 64 × 64 × 5 = 20480 possible triples.
This is a terrible idea at small scale, and understanding *why* is the first
real lesson.

A given position has maybe 30 legal moves. So the other 20450 slots never get
updated in that position. To give each slot enough updates to learn, you would
need an enormous number of games. The table is mostly empty forever.

The fix is **factorisation** — split the probability via the chain rule:

```
P(move) = P(from) · P(to | from) · P(promo | from, to)
```

Now `P(to | from)` is a 64×64 table that is reused across *every* origin square.
Learning "b1→c3 is good" and "g1→f3 is good" can share the fact that both are
knight-to-central-ish moves. This is **statistical sharing**: the same data
teaches more. (Hypothesis H16 asks whether this is the real reason factorisation
wins, or whether it is secretly a capacity effect — that question is genuinely
open. Nobody has separated the two.)

L1 takes factorisation and goes one further: it **removes the board input
entirely**. Its logits are three fixed tables. It sees only the legal-move mask,
to know which moves are even possible. This sounds useless. It is not:

- It is the **floor**. If a policy that can see the board *loses* to one that
  cannot, the informed policy is broken. L1 is a null hypothesis you can lose to.
- It makes all the *learning machinery* testable in isolation. Credit
  assignment, masking, pruning, counters, saving weights — you can debug all of
  it with a learner whose behaviour you can predict by hand.

The `BlindScorer` starts at all-zero logits, which makes its initial
distribution uniform over legal moves — identical to random play. That is
deliberate: L1-at-initialisation is exactly `RandomPolicy`, so you can diff them.

### 3.2 L2 — search, and the one thing search cannot do

L2 is a classic **negamax alpha-beta** engine. If you take one algorithms course
you will see this; here's the version that matters for this project:

- **Negamax** — evaluate a position from the side-to-move's perspective, and
  at each node pick the move that maximises *your* value, which is the negative
  of the opponent's best reply. Two lines of recursion.
- **Alpha-beta pruning** — if you have already found a response that gives the
  opponent value 5, and you're currently exploring a move where they can get 7,
  stop: you already know this move is worse. Prune.
- **Move ordering** — alpha-beta prunes better if you try likely-good moves
  first. This matters a lot: bad ordering means a *lot* of wasted nodes.
- **Transposition table** — cache positions you have evaluated. Different move
  orders reach the same position; don't evaluate it twice.
- **Iterative deepening** — search depth 1, then 2, then 3… reusing the
  previous depth's results to order the next. Sounds wasteful, is not.
- **Quiescence search** — do not stop searching in the middle of a capture
  sequence. "Horizon effect": at a fixed depth you can be one ply from losing a
  queen and not see it. Continuing on captures-only until things are quiet
  blunts this.

**The subtle claim in this repo that is worth understanding** (it's H1, and it's
the kind of thing that separates a careful person from a careless one):

> Changing move ordering changes *which* move alpha-beta returns among moves
> that are tied, but never changes the *value* it returns.

The value-invariance is a theorem (Knuth & Moore, 1975) — a correct-window
alpha-beta cannot change the value. But the *move* can change, because the first
move that reaches `alpha` is the one that's kept, so among equals the first
tried wins. The repo measured this concretely: in one position, both Nf6 and Nc6
evaluate to exactly 90, and the informed wrapper returns Nc6 instead of Nf6
*because L1's blind table weights the b8 origin slightly higher* — a tie broken
on no chess grounds. Which is, in fact, the only channel a prior can act
through. This is a beautiful small result; make sure you understand why it is
not a bug.

### 3.3 The forward pass: what a learner actually sees

This is `encode.py` and it's the module the whole benchmark rests on. Every
learner sees the position as a stack of **planes** — 2D 8×8 arrays — plus a few
"value" planes.

There are 24 binary planes (each cell is a yes/no fact about that square):

- 12 piece planes (6 piece types × 2 colours), so the network can see what's
  where;
- `occupied`, `to_move_attacks`, `opponent_attacks`, `to_move_defends`,
  `opponent_defends`, `contested` — the *relational* facts, which squares are
  attacked by whom;
- `capture_targets`, `promotion_rank_w/b`, `en_passant`, `castle_w/b`.

Plus 4 continuous value planes (centipawn quantities): `capture_value`,
`danger_value`, `attack_balance`, `own_danger`.

Two conventions you must not break:

1. **Coordinates are python-chess.** `0 = a1`, `63 = h8`. Row 0 of a plane is
   rank 1. No flipping anywhere except display.
2. **Channel order is append-only.** L3's weight matrix and L4's checkpoint are
   *indexed by channel position*. Insert a plane in the middle and a saved model
   loads fine and plays nonsense. This is the single nastiest way to break this
   repo and the `ONBOARDING.md` warns about it for good reason.

The value planes are off by default for training: they cost ~40× the binary
planes (they call into python-chess attack generation per move) and the binary
planes carry most of the signal.

### 3.4 The canonicalisation trick (the most important idea to internalise)

A chess position comes with a colour to move. If you train a network on raw
positions, it has to learn the game twice — once as White, once as Black. That
halves your data.

The fix: transform every position into a **canonical frame** where white is
always the side to move. The repo does this with `bitboard.canonical`.

**The trap:** the obvious transformation is a 180° rotation. It is wrong. A 180°
rotation maps a8 → h1, but the a-file and the h-file are *not* interchangeable —
chess has no left-right symmetry, only a top-bottom (rank) symmetry. So
canonicalisation is a **rank reflection plus a colour-block swap**, not a
rotation. If you get this wrong, positions encode identically that aren't
mirror images, and the model gets silently confused.

`apply_canonical` maps a *move* (which lives in real-board coordinates) into the
canonical frame and back. It is its own inverse.

There is a test helper, `tests/conftest.py::colour_mirror`, that takes a position
and produces its true mirror — swapping colours, reflecting ranks, and correctly
carrying castling rights and the en-passant target square. Every symmetry bug in
this repo's history was caught by comparing `encode(pos)` against
`encode(colour_mirror(pos))`. This has bitten **four separate times**. If you
write a symmetry test, use this helper and check it's still right.

### 3.5 The value head, the ensemble, and the training signal

- **A value head** is a second output of the network: not "which move", but
  "who's winning, as a number". L5 uses it as the leaf evaluation inside search:
  instead of a hand-coded material+PST score, ask the network. (Hypothesis H11:
  does a *trained* value head beat the heuristic? Not testable until the head is
  trained.)
- **An ensemble (L6)** combines several players. Two ways to combine: *blend*
  the members that give you a probability distribution (average them), *vote*
  the members that only give you a move. This is why `MemberKind` is *derived*
  from the member type rather than declared by the caller — you can't vote a
  distribution and you can't blend a single move.
- **The training signal** is the deepest axis. L1.5 and L3 learn from some
  target. Where does that target come from? Options: the game's final result
  (outcome RL), a stronger search's chosen moves (supervision), the learner's
  own value estimates (bootstrapping), random play (a control). This axis —
  "what signal makes a learner" — is the newest and most open part of the
  project (Groups G and H in `HYPOTHESES.md`).

---

## 4. How a game is played and scored (the parts that bite)

`game.py` is the referee. You point two policies at it and it plays a game.

Three conventions are load-bearing, and two of them were the source of real
bugs:

**Unfinished games are not draws.** The loop has a ply cap and a no-progress
cap. If a game hits a cap, it is `unfinished`, its result is *excluded from the
rating fit*, and it carries no training signal. If you scored it as a draw, you
would be teaching a shuffling policy that stalling is as good as playing — the
exact opposite of what you want. (H3 asks whether this convention actually
matters; H14 asks whether a high `unf` count is a free proxy for "untrained".)

**The game starts from a fixed opening book.** `play_match` plays from a set of
FENs, alternating colours. This is how you get a *low-variance* comparison out
of few games: without it, all games start from the same opening and your sample
is really one sample repeated. (H15: a bug in the book indexing may mean older
results aren't comparable to newer ones. Verify before quoting any stored
baseline.)

**Deeper is not automatically stronger, and untrained is not stronger at all.**
This is the honesty gate of the whole project. No learner beats the level below
it until it is *trained*:

- untrained L3 draws blind L1 — six draws, all by fivefold repetition;
- untrained L5 does not beat L2;
- a test asserting "L3 beats L1" failed 0–6, and **the test was wrong, not the
  code**. The architecture was credited where training hadn't happened.

So the tests assert the *mechanism* — does it play a full legal game, is it
seed-reproducible, does training change the preference — not that a random-weight
model wins. If you find yourself writing a test that expects an untrained model
to be strong, stop.

---

## 5. The methodology: how to do experiments here

This is the part that matters most for your career as a researcher, and it is
written down with unusual care. Read it twice.

### 5.1 The evidence standard

When you claim something, you must have ruled out the boring explanations first:

1. Is it an implementation bug? (Test the derivative / the rounding / the frame.)
2. Is it an evaluator bug? (Is the score function right?)
3. Is it a data bug? (Right positions, right labels, right split?)
4. Is it a metric bug? (Are you measuring what you think?)
5. Is it a protocol bug? (Right seeds, right comparison, right control?)

Only *then* is it a finding about chess or learning. This ordering is not
pedantry — every class on that list has produced a false result in this repo's
history.

### 5.2 The standard workflow, in order

```
1. Regression first.  Write a test that FAILS on the current code in the
   expected direction. This proves the test can detect the bug.
2. Minimal fix.       Smallest possible change. One file at a time.
3. Smoke test.        Tiny budget, under 5 minutes. Wiring check ONLY.
4. Frozen experiment. Pre-register the endpoint and the effect size you'd
   call meaningful. Freeze the data. Report uncertainty.
5. Conclude.          Only now is a claim allowed.
```

Steps 1–3 are cheap and catch most disasters. Step 4 is where the discipline
lives. A "smoke test" is not a result — it proves the plumbing works, nothing
more. The repo's own memory says: *"Do not infer learning or strength from tiny
samples; use a pre-specified, adequately powered held-out experiment with
uncertainty reporting before making hypothesis claims."*

### 5.3 The statistical hygiene

- **Always report the standard error.** A rating without an SE is noise. The
  repo's Elo SE formula is roughly `347/√(n(1−d))` — at 6 games/pair that's
  ~80 Elo, so a 6-game run cannot establish a sub-100-Elo gap. This is H13, and
  the answer is "yes, 6 games is too few".
- **Never quote a rating without its decided-game count.** A rating fitted from
  <10 decided games is provisional.
- **A result inside one standard error is "not established", not the direction
  you preferred.**
- **Convergence is not correctness.** Four separate bugs in the rating fit all
  converged to plausible-looking answers. Assert against an independent oracle.
- **Fix the budget before crediting architecture.** If model A has 10× the
  parameters of model B, a win tells you about capacity, not topology. (H9 is
  the control that exists purely to hold this line.)

### 5.4 The regression workflow for numeric bugs

For a learning bug, the strongest evidence is a **finite-difference check on the
actual forward quantity**:

1. Write a deterministic test that computes the true derivative by finite
   differences and compares it to your update rule.
2. Run it against the buggy code. It must fail, in the direction you predicted.
3. Fix.
4. Run it again. It passes.
5. Run the whole module.

Step 2 is the one people skip, and it's the one that makes the test meaningful —
without it you don't know the test can fail at all. The repo's most recent work
(today, 2026-10-06) fixed exactly this class of bug: the shared
context-to-hidden weight update was missing the ReLU gate coefficient, and the
finite-difference regression caught it.

---

## 6. The worked example: the self-play signal collapse

This is the best story in the repo. Study it; it's the template for how to think
here.

**The plan** was to train L3 by self-play: let the model play itself, reward
winning, and watch it improve. Standard reinforcement learning.

**What actually happened.** The model played 20 games against a frozen copy of
itself. **19 or 20 hit the ply cap** — they were all `unfinished`. Unfinished
games train only through a weak "shaped reward" branch (score the final
position's material, squashed). Measured mean signal: 0.053. Barely any
gradient.

So the obvious fix: raise the cap so games finish naturally. Do that, and
**40 out of 40 games end in `insufficient_material`** — both sides trade
everything off and drift to bare kings. Median length 288 plies. And every one
is a **draw**.

Finished draws return reward exactly `(0.0, 0.0)`. So:

> **Total reward mass over 40 games: 0.00. Zero learning signal.**

Decisive games across every cap tested (120/200/300/400): **0 out of 20 at each**.

**The finding.** The arm cannot learn strength *by construction*, at this scale,
with these caps. Short cap → weak shaped-only gradient. Long cap → zero signal.
There is no setting of the knob that works.

**Why this is a great result, not a failed experiment.** You have located a
regime boundary and explained the mechanism. That is knowledge. And notice what
it cost: no GPU. The whole thing was numpy on a laptop. Someone with a data
centre would never have found this, because at scale the games wouldn't all be
draws — the bug, if it exists there, is hidden behind enough compute to brute
force past it.

**What it changed.** The Kaggle plan
(`bench/KAGGLE-PLAN.md`) now says: **do not spend Kaggle hours on the self-play
arm until the signal is fixed locally.** The only arm worth running at scale is
the supervised one — imitate a search. That is a concrete, valuable redirection
that came out of a "negative" result.

**Three candidate fixes are listed** (opening diversity from varied start FENs;
adjudicating truncated games by material as a *decisive* reward; a small
randomness floor so the policy doesn't always pick the equal trade). Any of them
is a good first project for you.

---

## 7. The current frontier: the L3 held-out saga

This is what the repo is working on *right now* (2026-10-05/06). It is subtle
and it is the best illustration of §5 in action.

**The setup.** Train L3 to imitate the top-5 moves of a depth-3 search. Measure
on a held-out set (positions the model never trained on).

**First result (the pilot).** After 500 positions seen, the probability mass the
model puts on the teacher's *top-5 set* rose by +0.061, with a tight interval
above the pre-registered threshold. Looks like a win!

**But the metrics disagreed.** At the same checkpoint:
- weighted cross-entropy **got worse by +0.83 nats**;
- policy entropy **fell by 0.81 nats** (the model got much more confident).

**The per-position diagnosis.** Break it down position by position:
- 107 of 128 held-out positions had *worse* cross-entropy — so it's broad, not
  a few outliers;
- the "confident in the wrong move" signature appears: entropy collapses
  (3.35 → 1.67) while the teacher's best-move probability goes to ~0.

**The matched ablation.** Two hypotheses: is the target rule wrong, or is the
step size too big? Add a third arm (`max_scaled`) that keeps the *same* target
rule but matches the *magnitude* of the sum-rule arm. Result:

```
CE damage at 500 positions:
  max          +0.830     (the production rule)
  max_scaled   +0.095     (same rule, matched magnitude)
  sum          −0.0006    (the alternative rule)
```

Matching magnitude alone removes **~89%** of the damage. So **step size
dominates**; the target rule adds a small real residue. The earlier framing
("the target rule is wrong") was **over-stated by the person who wrote it, and
they corrected themselves**. That self-correction is the culture of this repo.

**The dose-response sweep.** Vary only the step multiplier:

```
max@1      CE +0.8296    entropy −0.809    topk +0.0611
max@0.5    CE +0.1996    entropy −0.256    topk +0.0362
max@0.25   CE +0.0211    entropy −0.050    topk +0.0161
max@0.125  CE −0.0037    entropy −0.0096   topk +0.0071
max@0.0625 CE −0.0070    entropy −0.0043   topk +0.0055
```

A clean monotone curve. The CE damage shrinks smoothly with the step and turns
slightly negative at `≤ 0.125`. This is the signature of **overshoot from too
large a step**, not a wrong target — a wrong target would not be cured by
shrinking the same push.

**The objective question (the punchline).** The user asked: which objective do
we actually optimise — maximise set-mass, or keep CE/entropy stable? The answer,
quantified:

> Only cross-entropy is a legitimate training objective. Set-mass is a
> *recall-style / satisficing* diagnostic — it rises when the policy spreads
> probability across the listed set, *even onto low-ranked members*. **Entropy
> is not a goal at all** — it is a diagnostic for detecting degeneration.

Decomposing the top-5 gain into best-move vs non-best-move:

```
max@1:  +0.0611 = best +0.0261 (42.7%) + non-best +0.0350
max@0.5: +0.0362 = best +0.0159 (44.1%) + non-best +0.0203
... (best-move share flat at ~40-44% across the entire sweep)
```

Most of the "gain" is *non-best* mass at every step size — it is largely the
policy becoming less diffuse, not learning the teacher's ranking. So the pilot's
set-mass gain is **not evidence of learning**, and shrinking the step loses the
gain without losing anything real: the discarded +0.061 was ~57% non-best mass
that co-occurred with the CE blow-up.

**The honest signal** is the best-move mass (0.0429 → 0.0690 at the top step),
and it tracks the step size. That is the quantity to optimise if a *real*
improvement is wanted.

**Every claim in this section is caveated**: one frozen corpus, one held-out
set, exploratory, *not a scale signal, not a bug fix, not wired into
production*. The intervals cover initialization seeds only — not corpus
resampling. **Do not promote to Kaggle.** This is exactly the discipline §5
describes, applied honestly.

---

## 8. Reading the code: a guided tour

**Start here, in this order.** Each step is one file and one idea.

1. `src/chessrl/encode.py` — read the module docstring, then `CHANNELS`. This
   defines the feature space everyone shares. If you understand this file, you
   understand what every learner sees.
2. `src/chessrl/masks.py` — `legal_move_mask` and `move_to_index` /
   `index_to_move`. The action space and how moves are addressed.
3. `src/chessrl/policy.py` — the `Scorer` protocol and `FactoredSoftmaxPolicy`.
   This is the interface that makes the whole ladder comparable.
4. `src/chessrl/game.py` — `play_game`, `play_match`, `GameResult`. The referee.
5. `bench/levels.py` — the roster: every level as a named `LevelSpec`. This is
   where you register a new player.
6. `bench/__main__.py` + `bench/runner.py` + `bench/rating.py` — the tournament:
   play matches, fit Bradley-Terry ratings, print a table.
7. `src/chessrl/search.py` — L2, the engine. Read it after you've seen the game
   loop, so the terms have context.

Then, for depth: `perceptron.py` (L3), `perceptron.py`/`squarelocal.py` (the
representation axis), `guided.py` (L5), `ensemble.py` (L6).

### Where the learning actually is

- **`policy.py`** — the factored softmax and masking. Understand why illegal
  moves are *masked* (`-inf` logit) rather than *punished*: masking gives exactly
  zero probability, zero gradient, and leaves the parameters untouched *and
  counted as untouched*, which is what makes pruning safe (see `ProbCounter`).
- **`perceptron.py`** — L3's forward and backward pass. This is the file where
  the current bug-fix work happened; the multi-head structure and the "midstate
  incremental updates" are the two things to read carefully.
- **`train_naive.py`** — L1.5. "Diminishing credit" (a move made earlier in the
  game gets less of the blame/credit for the outcome) and the probability
  counters. Note: counters must be *stripped before saving weights*, so a saved
  model doesn't carry training state.
- **`cache.py`** — `PositionCache` (don't re-encode the same position) and
  `MidstateStore` (a stratified corpus of game positions by phase). The corpus
  is what you train on when you don't want to re-generate games.

### Traps already paid for (don't pay for them again)

Read `ONBOARDING.md`'s "Traps that already bit us" section. A few highlights:

- **Verify a "clearly best move" before asserting the engine finds it.**
  `3qk3/8/8/8/8/8/8/3QK3` looks like a free queen; `Qxd8 Kxd8` is a trade and
  the true value is 0. The engine was right; the test was wrong.
- **The king's piece-square table is in a separate branch.** Forgetting to
  rank-mirror *it* (as well as the other tables) for Black made the start
  position evaluate to +50 instead of 0.
- **L5 frame discipline.** Read and write board quantities in *one* frame. The
  prior was all zeros for Black because masks were built from raw squares while
  head outputs are canonical. `canon` is the identity for White, so the bug was
  invisible until you played the other colour.
- **A too-small learning rate on a quantised model silently disables it.** The
  L3.5 tied heads get credit from every square, so the rate had to be divided —
  but at `lr=0.05` each step fell *below* the `1/QUANT` grid and `np.round` sent
  it to zero. Thirty games later, five of eight tensors were *bit-identical to
  initialisation* and the loss looked fine. Fixed with error feedback.
- **Check that a hypothesis's "obvious" feature exists.** L3.5's geometry
  features carry destination *rank* but no destination *file*, and use
  `abs(tf - ff)` — so `b1→c3` and `g1→h3` produce byte-identical vectors, and
  the model literally cannot say "a knight belongs near the centre". The missing
  feature, not the weight tying, is what limits it.
- **A null result on untrained weights is not a finding about architecture.**
  The first L3/L3.5/L3-flat run was 11-of-12 unfinished and said nothing.

---

## 9. Your first week: a concrete plan

**Day 1 — get it running.**
```bash
./.venv/Scripts/python.exe -m pytest         # 460 tests, ~3m30s
./.venv/Scripts/python.exe -m pytest tests/test_bitboard.py   # one module
python -m bench --only random material L2-d1 --games 20
```
Read `bench/README.md` to learn how to read the rating table (and, importantly,
what the benchmark *cannot* establish). Read `HYPOTHESES.md` before running any
comparison.

**Day 2 — read the five core files** (§8, steps 1–5). Write down, in your own
words, what a `Scorer` is and why L3 and L4 can be guaranteed identical.

**Day 3 — pick a cheap hypothesis and run it.** The docs say H17, H18, H19, H21
need no new levels. `bench/CHEAP-HARVEST.md` gives you ready scripts:
- **H21** — does an *untrained* L3 already prefer central moves? Pure script,
  no games, near-zero cost. A single afternoon. This is the best first project:
  it teaches you `move_distribution`, the probe set, and correlation-vs-null
  reasoning, all with no training.
- **H18** — at depth 2–3, are the engine's mistakes *tactical* (a capture is
  nearby) or *positional*? Also a pure script, no model.

**Day 4 — reproduce a number.** Take the L3 target-rule ablation, re-run it, and
confirm you get the same CE deltas. If you don't, something in your environment
differs and you've learned how to detect that. This is the skill that makes
everything else trustworthy.

**Day 5 — read the Kaggle plan** (`bench/KAGGLE-PLAN.md`). Understand why the
self-play arm is blocked and why the supervised arm is the one worth running at
scale. Then read the three candidate fixes at the end of §6 and pick the one you
find most convincing.

**After week 1** — the low-hanging fruit is picked; the real work is
training-shaped. Options: implement one of the three self-play signal fixes and
test it locally under 5 minutes; run the H23 castling census with a *trained*
L3 (the headline Kaggle test); or design the confirmatory corpus-replication
experiment that the L3 saga still needs.

---

## 10. What you're actually learning here

Strip away the chess and this is a course in **how to do empirical computer
science**:

- Isolating **one free variable** at a time, and building the control that makes
  the isolation real (L3-flat for topology; the random-walk arm for training
  signal; `max_scaled` for step size).
- Recognising when a **metric is lying to you** (set-mass rising while CE blows
  up) and choosing the objective that is actually legitimate.
- Distinguishing an **implementation bug** from a **domain finding** — and the
  discipline of ruling out the former before claiming the latter.
- **Pre-registering** an endpoint, then reporting the interval honestly, then
  saying "not established" when it's inside one standard error.
- Finding the **regime boundary**: not "does X help", but "at what scale does X
  stop helping, and through what mechanism".
- Treating a **negative result as a deliverable**, and correcting your own
  over-statements in writing when the evidence moves (as the L3 author did).

The chess is the vehicle. The reason it's *chess* is that it's the one domain
where the strongest possible baseline is known (search), the features are
discrete and inspectable, and every move is legal or not — so you can always
tell whether a failure is your code or the world.

---

## 11. Where this sits in the literature

You named four threads. Each one is a real, citable body of work, and — this is
the useful part — **each one is a place where this repo can produce a
contribution, because your scale is different from the scale at which the
literature was established.** Below I give the thread, the key sources, what the
field actually knows, and the specific gap this project occupies.

A general warning first, stated in `hypothesis-lit-review/report.md`: at
toy/numpy scale (~10³ params, hundreds of games) most published verdicts are
**silent or inverted**. So a local result here is often *novel by construction*,
not a replication. That is the portfolio value, and it also means you must not
casually claim "we confirmed the literature" — you probably didn't; you measured
a different regime.

---

### 11.1 Small-scale RL: papers on toy environments where standard tricks fail

**The problem this thread addresses.** Most RL benchmarks are expensive and
noisy, which makes them bad at *diagnosing* why an algorithm fails. The field
responded by building deliberately small, controlled testbeds whose failure
modes are legible.

Key sources:

- **Henderson et al., "Deep Reinforcement Learning that Matters"**
  (arXiv:1709.06560, AAAI 2018). The canonical reproducibility paper. Shows that
  reported RL results vary wildly with *seed*, *hyperparameters*, and *codebase* —
  to the point that a reported improvement is often inside the run-to-run noise.
  Gives concrete recommendations (multiple seeds, report variance, standardise
  reporting). **This is the direct ancestor of this repo's "always report the
  SE" rule (H13).** When the repo says "6 games/pair is too few", it is applying
  Henderson's lesson to a game score instead of an episode return.
- **Osband et al., "Behaviour Suite for Reinforcement Learning" (bsuite)**
  (arXiv:1908.03568, DeepMind, 2019). ~20 tiny experiments, each isolating one
  core capability (credit assignment, exploration, memory, generalisation).
  bsuite's stated goal is exactly this project's: *"collect clear, informative
  and scalable problems that capture key issues"* in RL, so agents can be
  diagnosed rather than merely ranked. **`chess-rl-bench` is bsuite's philosophy
  applied to one domain (chess) with a level ladder instead of a capability
  matrix.** If you want a model for how to present results, read bsuite's
  scorecard paper.
- **Abbas et al., "Loss of Plasticity in Continual Deep Reinforcement Learning"**
  (arXiv:2303.01486 / PMLR v232). Documents a real failure mode at small scale:
  networks trained continually can *lose the ability to learn* — their
  activations collapse and they stop adapting. Includes a mitigation (CReLU
  activations). Worth knowing because **"the model looks like it's training but
  isn't"** is precisely the family of bug this repo hit with L3.5's sub-grid
  learning rate (§8: five of eight tensors bit-identical to initialisation while
  the loss looked unremarkable).

**What the project adds.** Existing toy-RL work is mostly about *algorithms*
(does DQN beat PPO on this gridworld). This repo's ladder varies *representational
and signal* choices — factorised vs flat action space (H16), weight-tied vs
per-square (H8–H10), self-play vs supervision vs random-walk (H24–H26) — at a
budget where each choice is individually inspectable. That axis is under-covered
at this scale.

---

### 11.2 Self-play collapse: why self-play degenerates without diversity or reward shaping

**The problem this thread addresses.** Self-play is the engine behind AlphaZero,
but it can spiral: the agent trains against a copy of itself, so any bad habit
becomes the *opponent's* habit too, and the two co-adapt into a degenerate
equilibrium. This is the single most relevant thread to this repo, because the
repo's best story (§6) is exactly a self-play collapse.

Key sources:

- **Czarnecki et al., "Real World Games Look Like Spinning Tops"**
  (arXiv:2004.09468, NeurIPS 2020). The theoretical backbone. Argues that real
  games have a *nested, transitive* structure (a "spinning top" of strategy
  sets), and shows that **population-based self-play converges correctly only if
  the population is large enough to span the relevant strategy sets**. A single
  self-copy is the smallest possible population — so the theory predicts it can
  get stuck. This paper explains *why* the repo's lone self-copy drifts to bare
  kings.
- **Silver et al., "Mastering Chess and Shogi by Self-Play"**
  (arXiv:1712.01815, 2017 — the AlphaZero paper). The success story, and the
  source of the asymmetry: AlphaZero also injects *Dirichlet noise at the root*
  and *temperature sampling*, which are diversity mechanisms. Read it against
  the Czarnecki paper to see what the noisy version buys.
- **DiGiovanni, Zell et al., "Survey of Self-Play in Reinforcement Learning"**
  (arXiv:2107.02850, 2021). A map of the field: which games self-play works on,
  and the failure cases. Good entry point if you want breadth.
- **Recent LLM self-play work** (e.g. "Survive or Collapse: The Asymmetric Roles
  of Data Gating and Reward Grounding in Self-Play RL", and the ICLR 2026
  "diversity collapse" line). Warns that self-play RL without verifiable reward
  grounding *degrades over time* — early gains erode. Different domain (LLMs),
  same structural failure, and useful for showing the pattern is general.

**How this maps to the repo — precisely.** The repo's self-play result (§6) is:

> 40/40 games end in `insufficient_material`, all draws, median 288 plies,
> **0.00 total reward mass**, 0/20 decisive at every cap.

Read against Czarnecki et al., this is a **degenerate equilibrium**: the single
self-copy co-adapted to "trade everything, drift to bare kings" because there
was no population diversity and no noise to break the symmetry. The paper
predicts the mechanism; the repo *measured it at a regime 5–6 orders of
magnitude below where the theory was tested.* **The three candidate fixes the
Kaggle plan lists are exactly the three standard diversity mechanisms**: varied
start positions (population diversity), material adjudication of truncated games
(reward grounding / shaping), and a randomness floor (Dirichlet noise / temperature).
So the repo's fix list is not invented — it is the literature's remedy list,
being re-derived and tested at toy scale. That is a clean, defensible framing.

**The hypothesis to watch:** H24 (round-robin opponents beat self-play) is a
direct test of the Czarnecki claim at small scale. H26 (random-walk control) is
what stops you crediting "diversity" when the real cause is "more games".

---

### 11.3 Search vs function approximation in small domains

**The problem this thread addresses.** Two ways to play a game: *search* a
game tree (brute force + pruning + an eval), or *learn* a policy that maps
position → move directly. Which wins, and when, is a foundational question.

Key sources:

- **Knuth & Moore (1975), "An Analysis of Alpha-Beta Pruning."** The origin of
  the alpha-beta theorem the repo relies on (H1: ordering cannot change a
  search's *value*, only *which* tied move is returned). This is not just
  background — the repo's H1 is literally a restatement plus a measured
  illustration of this 1975 result at toy scale.
- **The "brute force vs knowledge" debate** (Computer History Museum, "Brute
  Force vs Knowledge") — the historical arc of engine design, from hand-coded
  chess knowledge to the recognition that deep search + a simple eval beats
  shallow search + clever eval. This is the entire justification for the ladder
  putting *depth* (L2) below *learning* (L3): H20 ("positional knowledge only
  pays once tactics are handled") is the local, measured form of this.
- **Ruoss et al., "Grandmaster-Level Chess Without Search"**
  (arXiv:2402.04494, DeepMind, 2024). The modern inversion: a 270M-parameter
  transformer trained by supervised learning on 10M games reaches grandmaster
  level with *no search at all*. Crucially, the paper's own scaling study shows
  the claim **fails at ≤10M parameters** — below that, search wins. This is the
  single most important citation for this project, because **it tells you where
  the regime boundary is, and the repo sits far below it.** The literature
  review calls this out explicitly: at toy scale the searchless claim is
  *inverted*, which is exactly the boundary this project is positioned to
  measure from the other side.

**What the project adds.** The repo has both arms on one harness: L2 is pure
search, L3/L4 are pure function approximation, L5 is search *guided by* a
learned model. H12 (does the model prior save nodes without changing the value?)
and H11 (does a trained value head beat the material heuristic?) are the two
seams. At toy scale the expected result — search wins, because the learned
policy can't be trained enough to compete — is *itself* the finding, provided
you can point at the mechanism (which the repo does: see the reward-collapse
diagnosis).

**Caveat to keep in mind** (this is H22, and it's a real trap): at this scale,
branching factor, search quality and eval quality are **perfectly confounded**.
The repo's search levels don't learn and its learners don't search. So you
*cannot* cleanly attribute a difference to "search space size". H22 exists
specifically to stop someone re-proposing that claim. Read it before you write
anything about branching factor.

---

### 11.4 Negative results in ML

**The problem this thread addresses.** ML rewards positive results. A paper
reporting "we tried X, it didn't work" gets rejected, so the failure is
rediscovered independently by dozens of labs. This is a structural, known
pathology with an active reform movement.

Key sources:

- **Karl et al., "Position: Embracing Negative Results in Machine Learning"**
  (arXiv:2406.03980, ICML 2024, PMLR v235). The definitive position paper, and
  the one to cite. Main claims:
  - predictive performance alone is a **faulty metric** for a publication's
    worth, and sets bad incentives;
  - it distinguishes two negative-result types — **NMNR** (a novel method that
    doesn't beat SOTA) and **EMNR** (an existing/SOTA method shown to fail where
    it was assumed to work, e.g. replication or failure-mode analysis);
  - it makes **eight concrete proposals** (special tracks, discussion of failed
    attempts in challenge papers, replication incentives, teaching negative
    results, review-process reform, etc.).
  - **This project's results are predominantly EMNR**: "self-play was assumed to
    work; at this scale it produces zero signal" is an existing-method negative
    result. That is the *respected* category, not the weak one. Use the
    vocabulary.
- **The "I Can't Believe It's Not Better" (ICBINB) workshops**, hosted at
  NeurIPS (2020, 2023; PMLR v239). The 2023 edition was literally titled
  *"Failure Modes in the Age of Foundation Models."* This is the venue where
  this kind of work is actually presented. There are also ICBINB spinoffs
  (e.g. ICBINB-BIO at NeurIPS 2026) — evidence the movement is spreading.
- **"Workshop on Insights from Negative Results in NLP"** — the NLP-side
  equivalent, if you want a second venue.
- **Historical proof that negative results pay off:** the review lists Bengio
  et al. (1994) on vanishing gradients (→ LSTM) and Szegedy/Goodfellow on
  adversarial examples. Both were "this doesn't work / this is broken" findings
  that reshaped the field.

**How this maps to the repo.** Three of the repo's headline results are exactly
the EMNR category the Karl paper describes:

1. **The self-play signal collapse** (§6) — an existing method (naive self-play
   RL) shown to produce identically zero signal at this scale.
2. **The L3 set-mass illusion** (§7) — a *metric* shown to reward a degenerate
   solution, meaning an apparent positive result was not evidence of learning.
   This is a methodological negative result.
3. **The target-rule over-claim correction** (§7) — the author concluded "the
   target rule is wrong", then a matched ablation showed step size explained
   ~89% of it and the author **publicly corrected themselves in writing**.
   Self-correction to a more mundane explanation is the *highest* form of this
   practice.

**The honest gap to acknowledge.** As of this writing, these results live in
`HYPOTHESES.md` and the memory logs, not in a paper. The register explicitly
notes that a separate **prove/not-prove document does not yet exist**. So the
"negative results" thread is the one where the project's *rhetoric* (§1 is
written in exactly this spirit) has outrun its *artefacts*. That is the most
interesting gap for a newcomer to close: the project already does negative
science; what it lacks is the write-up that lets anyone else benefit from it.

---

### 11.5 The synthesis: what makes this project's position distinctive

Put the four threads together and a coherent thesis emerges:

> **The standard ingredients of game-playing RL — self-play, factorisation,
> learned value heads, searchless policies — were validated at scales many
> orders of magnitude above this one, and at least four of them (per Ruoss et
> al. and per this repo's own measurements) *invert or go silent* at toy scale.
> This project maps the boundary from the small side, using a controlled ladder
> and pre-registered falsification conditions, and publishes the failures as
> results.**

Each thread contributes one leg:

| Thread | Literature anchor | Repo's role |
|---|---|---|
| Toy-scale RL | Henderson 2018; bsuite 2019; Abbas 2023 | A toy testbed with a *representational* axis, not just an algorithmic one |
| Self-play collapse | Czarnecki 2020; AlphaZero 2017; DiGiovanni 2021 | Measured a degenerate equilibrium and its zero-signal consequence |
| Search vs learning | Knuth & Moore 1975; Ruoss 2024 | Both arms on one harness; measures the regime boundary Ruoss defines |
| Negative results | Karl 2024; ICBINB workshops | Produces EMNR-class findings, but hasn't written them up yet |

**Two rules for using this section.** First, cite the *specific* claim, not the
vibe — "Ruoss et al. show the searchless claim fails below ~10M parameters" is
useful; "deep learning is amazing for chess" is not. Second, when you find a
result that *contradicts* a source here, do not assume you have found a bug —
check the regime. A difference between your 10³-parameter model and a 270M one
is a *regime boundary*, which is the whole point of the project.

---

## Appendix: the documents, and what each is for

| File | Role |
|---|---|
| `ONBOARDING.md` | The "how do I run it" sheet: setup, the five rules, module map, traps |
| `HYPOTHESES.md` | The investigation register: every claim, its falsification condition, its status |
| `bench/README.md` | How to read a rating table, and what the benchmark *cannot* establish |
| `bench/CHEAP-HARVEST.md` | Decisive experiments needing no training, with ready scripts |
| `bench/KAGGLE-PLAN.md` | What to run at scale, what it costs, what a result would mean |
| `hypothesis-lit-review/report.md` | How each hypothesis relates to the published literature |
| `CLAIMS-CHANGELOG.md` | Append-only ledger of claims/caveats that changed status — retired, revised, relocated |
| §11 of this file | The four external literature threads, with citations and the repo's position in each |
| `.workbuddy-ai/memory/*.md` | The blow-by-blow history, including every bug fixed |

**The one habit to build:** before you trust any claim in this repo — including
one you just made — ask "what's the standard error, and what was held constant?"
If you can't answer both, you don't have a result yet.
