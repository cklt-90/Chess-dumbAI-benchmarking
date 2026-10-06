# Design note: stacked factored-softmax entry ("L1-bilinear" / proposed L1.9)

*Evaluating a proposal from 2026-10-06. Status: **analysis only, not implemented.***

> **See also:** `DESIGN-tier2-template.md`. That document is a *pre-registered
> container for tier-2 entries*, held empty until tier 1 produces a located,
> evidenced defect. This note is used there as the **illustrative instance** of
> the shape — it is **not** an approved build, and its own Step-0 defect (a
> structural limitation of the factored chain) is stated in the code but not yet
> *measured* to matter at matched budget. The missing parameter-matched control
> (arm C) is the step the original proposal omitted.

## The proposal

> A new rung based on L1. L1's three tables act as a **bias** that gets masked
> and softmaxed. Then a **linear layer over all possible moves → all possible
> moves** is placed on top of that, and masked + softmaxed again.

Restated in the repo's terms: keep L1's factored logits, but add a second stage
whose score for move *i* can depend on the scores of moves *j* — then re-do
mask + softmax.

## What is genuinely good about this idea

Three things, and they are not small.

1. **It is the first proposal that varies the *computation shape* at the
   distribution level rather than the feature level.** Every existing rung
   varies what the learner *sees* (L1 blind, L3 informed), *how much it
   searches* (L2, L5), or *which backend* (L4). L3.5 varies the shape of the
   per-square computation. Your idea varies the shape *across the action axis
   itself* — and nothing on the ladder currently does that. That is a real gap.

2. **It targets a limitation the repo has already documented.** The factorised
   chain `P(from)·P(to|from)·P(promo|from,to)` cannot express any *interaction
   between moves*. The destination head is a function of the origin alone; no
   move's probability can depend on another move's score. Under a factorised
   model, the greedy pick is greedy *per factor* and is not even guaranteed to
   be the arg-max of the joint distribution (see `_argmax_factors` in
   `policy.py`, which admits this in its docstring). A mixing layer is the
   standard way to fix exactly that. **Your intuition is pointing at a real,
   named defect.**

3. **"Bias, then mix, then re-softmax" is a coherent architecture** — it is
   essentially one attention-free transformer block over the action axis, where
   L1 provides the positional/identity prior and the linear layer provides
   cross-move interaction. That is a legitimate design, not a naive one.

## The problem: parameter count, and it is fatal at this scale

The proposal says "all possible moves → all possible moves". The action space is

```
ACTION_SPACE = 64 × 64 × 5 = 20480
```

A dense linear layer over it is `20480 × 20480` = **419,430,400 parameters**.
For scale:

| Model | Parameters |
|---|---|
| whole L1 (three tables) | 64 + 4096 + 20480 = **24,640** |
| L3 perceptron | **7,609** |
| L3.5 | **787** |
| **your dense mixing layer** | **419,430,400** |

Your mixing layer would be **~17,000× the entire L1 model** and ~55,000× L3.
This is not "a bit expensive". It is the exact failure mode that
`policy.py`'s docstring already warns about for the *flat* softmax:

> *"A flat softmax over 20480 slots is a table you cannot fill… a given position
> only ever touches ~30 of them, which means each weight sees very few updates
> and learns almost nothing."*

A dense 20480×20480 mixing matrix is that same problem, squared: each of the 419M
weights is touched by only the handful of positions where *both* its row's move
and its column's move are legal. It would never train. So the idea as literally
stated cannot work here — but that is a statement about *density*, not about the
*concept*, and the concept survives if you keep the interaction **factorised
too**.

## The salvageable version: interact on the factors, not the flat space

The fix is the same move L1 already makes — factorise the new layer. Instead of
one 20480×20480 matrix, give each factor its own small mixing matrix:

```
from-stage:   logits_from' = W_from_mix  @ logits_from     # 64×64     = 4,096
to-stage:     logits_to'   = W_to_mix    @ logits_to       # 64×64     = 4,096
promo-stage:  logits_promo'= W_promo_mix @ logits_promo    # 5×5       =    25
```

Total added: **8,217 parameters** — less than L1's own tables, ~108% of L3. That
is affordable, and it is a real model: the destination score for `from=e2` can
now be a learned blend of *all* destinations rather than an independent lookup,
which is the cheap, structured version of the interaction you wanted.

Two design decisions then have to be made deliberately:

- **Where does the mixing happen relative to the mask?** Masking zeroes illegal
  moves. If you mix *then* mask, the mixing matrix still sees legal-logit-shaped
  inputs — this is the natural order and matches your description. If you mix
  *after* masking, illegal entries are already `-inf` and would poison the
  blend. **Mix before mask**, on raw logits, at every stage.
- **Is the mixing matrix shared across stages or per-stage?** Per-stage (as
  above) is simpler; a single shared matrix across all three would be closer to
  the "one tied mixing rule" philosophy of L3.5 and would add only 4,096. That
  choice is itself a small experiment.

## What this would actually be *for* — and the honest caveat

Here is where you have to be careful, because the repo has a specific rule about
this (H16, and the anti-patterns list):

> **H16** asks whether factorisation buys *trainability* or *expressiveness*.
> The flat model is "a table you cannot fill" — that is an argument about *update
> counts*, an optimisation claim. Nobody usually separates the two, so the
> claimed advantage is doing double duty.

Your stacked model is a way to **test that separation directly**. If a small
factorised mixing layer (8k params) beats plain L1, you have shown that
cross-move interaction has *expressiveness* value at this scale, not just an
update-count effect. That is a genuinely publishable-shaped result and it is
close to H16's open question.

**But** — and this is the trap the register exists to prevent — you must not
report "the stacked model wins" without holding the budget constant. At 24,640
+ 8,217 = ~32.8k parameters you would be *bigger than L3* (7,609). So a win
would be confounded with capacity, exactly the thing H9 (the `L3-flat` control)
was built to rule out. The clean experiment is therefore:

1. **Arm A** — plain L1 (24,640 params). The control.
2. **Arm B** — L1 + factorised mixing (~32.8k). Does interaction help?
3. **Arm C** — L1 padded with *irrelevant* capacity to ~32.8k (extra table
   entries), so B−C isolates the *structure* of the mixing from its *size*.

Without arm C you cannot distinguish "cross-move mixing helps" from "more
parameters help". That is H9's rule applied to your idea.

## Where it sits on the ladder, if built

It is **not** an L2 replacement and **not** a new top rung. It is a *sibling of
L1*: blind, factorised, no board input, but with one extra linear interaction.
I would name it **L1.9** or `L1-mix` and register it as an **`ablation`**
entrant, not a `ladder` entrant — because it is not claimed to beat L1, it is
there to *measure whether interaction matters*, which is a Group C/E question,
not a Group B "is the next rung higher" question.

This matters for honesty: `bench/levels.py`'s docstring is explicit that a
ladder entry is "expected to beat the one below it", and that where that fails
it is a result, not a bug. Putting L1.9 in the ladder would assert something you
don't yet believe. Put it in the ablation roster.

## What I would do before writing any code

The failure mode to avoid is building the model to succeed (H8–H10 exist *before*
L3.5 for exactly this reason). So:

1. **Write the hypothesis down first, with the kill condition.** Something like:
   > *A cross-move interaction term (factorised mixing, ≥1 shared matrix across
   > the three heads) improves L1's held-out teacher agreement beyond a
   > parameter-matched control, at equal data budget.*
   > **Falsified by:** arm B within 1 SE of arm C (structure of the mixing adds
   > nothing beyond its size), or B below A.
2. **Add it to `HYPOTHESES.md`** as a Group C/E entry with that condition, *before*
   implementing. The repo's own rule: "writing the kill condition before
   building the thing is what stops the thing being built to succeed."
3. **Only then** write the `Scorer` — and note it is a small job: a
   `MixedScorer` implementing `from_logits`/`to_logits`/`promo_logits`, holding
   L1's tables plus the three mixing matrices. It plugs into the *existing*
   `FactoredSoftmaxPolicy` unchanged. No new policy class.

## Verdict

- **As stated (dense 20480×20480):** infeasible — 419M params, the exact
  "table you cannot fill" trap L1 was built to escape.
- **Factorised over the three heads (~8k params):** coherent, affordable, and it
  answers a real open question (H16's trainability-vs-expressiveness split, and
  the documented no-cross-move-interaction limitation of the factored chain).
- **Do not skip the budget-matched control.** Without it the result is
  uninterpretable, and this repo will (correctly) refuse to let you claim it.
