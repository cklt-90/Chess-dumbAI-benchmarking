# Tier-2 design: repairs and verifications under fixed constraints

*Status: **an empty slot with a pre-registered shape — a reserve, not a plan and
not a workflow to run yet.***

> **Read this first.** Tier 2 has **no content** until tier 1 has produced
> located, evidenced defects. This document is deliberately a *container*, not a
> to-do list: it says what a tier-2 entry must look like *when one is warranted*,
> so that the first one is specified correctly instead of retrofitted. Do not
> "build tier 2" — there is nothing to build until there is a defect to repair.
>
> "Defect" is broader than "the model is wrong". It includes **experimental-design
> limitations** — see §0.1. A tier-1 result that cannot be *interpreted* (because
> its interval covers only init seeds, or its control was missing) is a defect
> too, and repairing it is a tier-2 job.
>
> The one worked example (`DESIGN-l1-mixing-layer.md`) is an **illustration of
> the shape**, not a commitment to build it. It is included because a pattern
> with no instance is too abstract to be checkable; it is not an endorsement.

---

## 0. What this document is, and is not

**Is:** a pre-registered shape for a class of future entries — the constraint set
they must inherit, the control they must carry, and the rule that they are
*evidence-driven* (designed after a tier-1 result, never before).

**Is not:** a workstream, a schedule, or a claim that any specific tier-2 entry
is needed. **Tier-2 work is gated on tier 1 being complete enough to point at
something.** If tier 1 has not located a defect with evidence, tier 2 has nothing
to do and this document should stay unopened.

The correct posture toward it is: *written down so it is not re-invented later,
and otherwise inert until a defect warrants opening it.*

**One entry is now open (2026-10-07):**
`DESIGN-tier2-move-conditioning.md` — a flavour (a) repair entry, triggered by
the H23 behavioural result (L3's move score has a two-square receptive field and
cannot condition castling on the position). Note its framing: it registers the
**problem as a gate on further conditional evaluation** and explicitly leaves the
**choice of repair open**. Registering an entry does not mean the container has
become a to-do list; the Step 0 gate still applies to every future entry.

### 0.1 Two kinds of tier-1 output that justify a tier-2 entry

"Located defect" was too narrow. A tier-2 entry is warranted when tier 1 leaves
either of these:

**(a) A mechanism defect.** A located limitation of a *model*: a computation that
provably cannot express something the task needs. Example: the factored chain
`P(from)·P(to|from)·P(promo)` cannot express cross-move interaction. Repairing it
means changing the design of the computation under the same information budget.

**(b) A design defect / unresolved verification.** A located limitation of the
*experiment* — the evidence exists but cannot be interpreted, or a claim was
established only under conditions too narrow to trust. Repairing it means
building the missing control, replicating over the missing axis, or running the
next-most-important verification the result raised. **This is a first-class
reason to plan tier 2, not a lesser one.**

The repo has a live instance of (b) right now, and it is the honest gap recorded
on 2026-10-06: *the L3 held-out pilot's intervals cover initialization seeds
only; they are conditional on one frozen corpus and held-out set.* That is not a
model defect — the model may be fine. It is a **design defect**: the claim cannot
be generalised because corpus sampling was never replicated. The tier-2 response
would be a corpus-replication arm (fresh positions, same protocol, same budget),
which is a *verification* entry, not an *improvement* entry.

So tier 2 has two flavours:

| Flavour | Tier-1 output it answers | What the entry does |
|---|---|---|
| **Repair** | a mechanism defect (a) | change the computation, same information budget |
| **Verification** | a design defect (b) | add the missing control / replicate the missing axis |

Both must still carry the constraint set (§3) and the matched control (§4 Step 4).
A verification entry additionally has to state *which claim it is testing* and
*what result would retire it*.

**Where a verification entry's output goes.** Its result is typically "this
caveat no longer applies" (or "it survives"), which is a *status transition* and
does not fit `HYPOTHESES.md`'s hypothesis format. Those land in
**`CLAIMS-CHANGELOG.md`** at the repo root — the append-only ledger of claims and
caveats that changed status. The changelog records the *transition*; a future
prove/not-prove ledger would record the *current state*. Use both if they
diverge.

---

## 1. The idea

Right now the ladder has one direction of travel: **add information or add
capability.** L1 → L2 adds search. L1 → L3 adds board vision. L3 → L4 changes
backend. Each new rung is *more* than the one below it, and the benchmark asks
"is the new rung stronger?".

The proposal is a **second, orthogonal tier**:

> **Tier 2** entries are *improved* versions of a tier-1 entry that keep the
> **same information and design constraints** as the level below them. They are
> **designed in response to what the tier-1 experiments found**, not invented in
> advance.

So tier 1 asks *"what does each ingredient buy?"*. Tier 2 asks a sharper
question:

> **"Given what we learned, can we build a better model *without changing the
> rules* — same inputs, same signal, same budget class, same action space — by
> changing only the thing the evidence pointed at?"**

That is a different and harder question, and it is the one that produces
transferable knowledge. "Add a neural net and it gets stronger" teaches little.
"Under a fixed information budget, the specific defect the ablation located can
be repaired, and here is the repair's measurable value" teaches a lot.

## 2. Why this is worth having (the failure it prevents)

The repo already has the anti-pattern that motivates it, written down in
`HYPOTHESES.md`:

> **Anti-pattern: crediting architecture for a parameter-count win.** Fix the
> budget first (H9), then let topology be the free variable.

And the L3.5 experiment (`bench/README.md`) is the existing hero case: three
arms, one budget, one free variable, with `L3-flat` as the mandatory control so
that "any change helps" cannot masquerade as "this change helps".

Tier 2 would generalise that discipline into a **repeatable shape**. The value is
that a tier-2 entry is *evidence-driven*: you may not design it until a tier-1
ablation has told you what to fix, and you may not claim it works until it beats
its own parameter-matched control at equal data budget. **But that discipline only
bites once there is something to fix — see §0.**

## 3. The constraints a tier-2 entry must inherit

This is the load-bearing part. An "improved" entry is only interesting if it is
genuinely constrained — otherwise it is just a stronger rung with a new name.
A tier-2 entry inherits, from the tier-1 level it improves:

| Constraint | Meaning | Example (from L1) |
|---|---|---|
| **Information budget** | Identical inputs. No new features, no board access if the base had none. | L1 is *blind* — a tier-2 L1 improvement may **not** look at the board. |
| **Signal / training** | Same target and same data budget (games / positions seen). | Same corpus, same number of updates per position. |
| **Action space** | Same `ACTION_SPACE`, same masking rules. | 20480, factorised `(from,to,promo)`. |
| **Budget class** | Same order of magnitude of parameters — stated as a *fraction with its reason*, as L3.5 does, never quietly padded. | "within ~1.5× of L1's 24,640" — declared, not assumed. |
| **Protocol** | Same `MovePolicy`/`DistributionPolicy` interfaces, same game loop, same book, ≥20 games/pair, SE always reported. | Unchanged. |

**The one free variable is the design of the computation.** Everything else is
held fixed. That is what makes a tier-2 result interpretable.

### 3.1 The control vocabulary — and what "disproof-style" means

A caveat first, in the spirit of this repo: **"disproof-style control" is not a
standard term of art.** I coined it in conversation on 2026-10-06; it does not
appear in the experimental-design literature under that name. What follows is the
definition I meant, so it can be used precisely or replaced by the standard term
if one exists.

Controls differ by *what they are built to prove*:

| Control | Built so that… | If it behaves as designed, you learn… |
|---|---|---|
| **Positive control** | it *should* show an effect | the experiment can detect an effect at all (the harness works) |
| **Negative control** | it *should* show no effect | the observed effect is not an artifact of the setup |
| **Disproof-style control** | it *reproduces the effect by a known-spurious route* | if it matches the treatment, **the treatment's claimed mechanism is refuted** |

The disproof-style control is the sharpest of the three, and it is the one this
repo keeps reinventing. Its logic: *build an arm that gets the same measured win
for the wrong reason. If the treatment does not beat it, the reason you believed
was doing the work was not doing the work.*

**The repo's own worked instance is `max_scaled`** in
`bench/scripts/l3_target_rule_ablation.py` (2026-10-06). The claim was "the `max`
target rule is wrong". The disproof-style control kept the `max` rule *and*
matched the alternative arm's aggregate update magnitude. It reproduced most of
the measured difference (`max_scaled` +0.095 vs `sum` −0.0006, against `max`
+0.830) — thereby **refuting the target-rule explanation** and relocating the
cause to step size. That single control overturned the original framing.

In the tier-2 three-arm experiment (§4 Step 4), **arm C is the disproof-style
control**: `base + irrelevant capacity` reproduces the *size* of the improvement
without its *structure*. If B ≈ C, the structural claim dies and only "more
parameters help" survives. That is why arm C is mandatory and why B must beat C,
not merely A.

**What makes a control "disproof-style" rather than merely negative:** a negative
control is passive (it should do nothing); a disproof-style control is *active* —
it is engineered to succeed by the wrong mechanism, so that its success is
falsifying rather than reassuring. In clinical-trial language it is closer to an
*active comparator* than to a placebo.

## 4. The sequence, *when a tier-2 entry is warranted*

**Step 0 is a gate, not a step.** If it is not satisfied, the rest does not apply
and no tier-2 work should exist. The sequence is deliberately front-loaded with
hypothesis authoring, because the repo's precedent (H8–H10 were written *before*
L3.5 existed) is that writing the kill condition in advance is what stops the
thing being built to succeed.

```
Step 0 — GATE: a tier-1 experiment has produced a located defect, with evidence.
         EITHER kind counts (§0.1):
           (a) a mechanism defect — the computation cannot express something
               the task needs ("the factored chain has no cross-move term");
           (b) a design defect — the evidence exists but cannot be interpreted,
               or a claim rests on one frozen corpus / one unreplicated axis.
         Each must be demonstrated, not asserted. If neither is met, STOP.
         There is no tier-2 entry to design.

Step 1 — author the tier-2 hypothesis in HYPOTHESES.md, BEFORE implementing.
         State: the constraint set it inherits, the single changed variable,
         the falsification condition, and the parameter-matched control arm.

Step 2 — declare the budget. State the inherited budget and the tier-2 budget
         as a fraction, with the reason for any deviation. (L3.5 precedent.)

Step 3 — implement the `Scorer` (or policy), reusing the shared machinery.
         A tier-2 learner is normally a new Scorer plugged into the EXISTING
         policy — see the L1-mixing example.

Step 4 — the three-arm matched experiment:
           A  base tier-1 entry        (control)
           B  tier-2 improved entry    (the change)
           C  disproof-style control: base + the WRONG mechanism's advantage,
              matched to B's size (the H9 control; §3.1)
         B must beat C, not just A, or the result is a parameter-count win.
         A **verification** entry (flavour b) instead adds the missing axis —
         e.g. a second frozen corpus — and reports whether the claim survives it.

Step 5 — register as an **ablation**-tier entrant, not a ladder rung.
         A ladder entry asserts "beats the one below it". A tier-2 entry asserts
         "the repaired design beats a matched control under the SAME inputs".
         Different claim, different tier.
         A **verification** entry need not be an entrant at all — a script plus a
         results JSON is often the right artefact, as the cheap-harvest items are.

Step 6 — report with SE, decided-game counts, and the residuals. A result
         inside one SE is "not established", never the preferred direction.
```

## 4.1 The one open entry, for reference

`DESIGN-tier2-move-conditioning.md` is the first entry to clear Step 0. It is
worth reading as a *worked instance of the gate*, not just of the shape — in
particular its §"Demonstrated vs suggestive" table, which separates the part of
the defect established by construction from the part that is only suggestive,
and its refusal to name the repair. Both are habits a future entry should copy.

## 5. Worked example (illustrative only): `DESIGN-l1-mixing-layer.md`

The L1 mixing-layer proposal is shown below mapped onto the sequence, purely so
the shape of a tier-2 entry is checkable rather than abstract. **It is an
illustration, not a commitment** — no decision has been made to build it, and its
own Step 0 defect is the one that is closest to being met but is not yet
*measured* at matched budget.

- **Step 0 — located defect (closest to gate-satisfying, but not closed).** The
  factored chain `P(from)·P(to|from)·P(promo)` cannot express any interaction
  between moves; `_argmax_factors` admits greedy-per-factor is not joint-argmax.
  This is a *structural* limitation of the design, stated in the code — but the
  gate asks for evidence that repairing it *matters*, which does not exist yet.
- **Step 1 — hypothesis (would be).** *"A cross-move interaction term (factorised
  mixing, ≥1 shared matrix across the three heads) improves L1's held-out teacher
  agreement beyond a parameter-matched control, at equal data budget."*
  Falsified by arm B within 1 SE of arm C, or B below A.
- **Step 2 — budget (would be).** L1 = 24,640. Factorised mixing adds 8,217
  (per-stage) or 4,096 (shared). Declared up front.
- **Step 3 — implementation (would be).** A `MixedScorer` — L1's three tables
  plus the mixing matrices — plugged into the **existing**
  `FactoredSoftmaxPolicy`. No new policy class.
- **Step 4 — three arms.** A = plain L1; B = L1 + mixing; C = L1 padded with
  irrelevant capacity to B's size. This is the arm the proposal *originally
  omitted* and the pattern now requires.
- **Step 5 — tier.** `ablation`, because it is not claimed to beat L1.
- **Step 6 — reporting rule.** SE plus decided games, as always.

The design note's own history is the useful part: the first framing was
"dense 20480×20480" (fatal, 419M params) and the correction to a factorised
mixing layer only became *visible* once the constraint set was written down and
the budget had to be declared. **Declaring the budget (§3) caught the design
error.** That is the pattern earning its keep on its first pass — which is the
argument for having the shape on record *before* tier 2 has any content.

## 6. How tier 2 relates to the existing groups

- **Group B (the ladder)** asks "is each rung above the one below?" — a *tier-1*
  question.
- **Group C (representation axis)** is the existing tier-2-shaped work: L3 vs
  L3.5 vs L3-flat. Tier 2 is not a new method — it is "make Group C's already-used
  method the standard way to introduce any improved model". Group C is the proof
  the method is sound; the container just names it so it is reused rather than
  reinvented.
- **H9 (budget vs topology)** and **H16 (trainability vs expressiveness)** are
  *tier-1* hypotheses that a tier-2 entry would be the natural instrument for. A
  tier-2 result is what would convert an H9/H16 "mixed verdict across regimes"
  into a local answer — **once one exists**.
- The container should therefore reference, not duplicate, the existing
  hypotheses. Its job is to add the **constraint-set declaration and the
  three-arm control** as a required part of the record, when a tier-2 entry is
  eventually warranted.

## 7. What could go wrong (and the guards)

| Risk | Guard |
|---|---|
| **Tier 2 is "built" speculatively, before any defect exists** | **§0 and Step 0 are gates.** No located, evidenced defect ⇒ no tier-2 entry. This is the primary failure mode, and the reason this document is a reserve rather than a plan. |
| "Improved" silently means "bigger / more informed" | Constraint table (§3) must be filled in, and the information budget is the strictest row. |
| The improvement is a parameter-count effect | Arm C (H9 control) is mandatory, not optional. |
| The entry is designed to succeed | Hypothesis + kill condition authored in Step 1, before Step 3. |
| The result is claimed at 6 games/pair | ≥20 games/pair, SE reported, else "not established". |
| Tier 2 quietly becomes a new ladder rung and inflates the ladder claim | Register as `ablation` tier; the ladder claim stays untouched. |
| The "located defect" is asserted rather than measured | Step 0 requires evidence, not narrative. (Contrast: H10's located defect was found by construction and labelled as such — *missing-feature*, not proven *tying*.) |
| A **design** defect (flavour b) is treated as second-class, so the experiment's own limitations are never fixed | §0.1 makes design defects a first-class trigger. An uninterpretable result is as good a reason to plan tier 2 as a broken model. |
| The disproof-style control is replaced by a *weaker* control (e.g. just "no change") | §3.1: arm C must be engineered to succeed by the **wrong** mechanism, not merely to do nothing. A passive control cannot refute a mechanism claim. |
| A verification entry is judged as if it were an improvement entry ("it didn't get stronger, so it failed") | Flavour (b) is judged by *what it retires* (a claim, a caveat), not by Elo gained. State the retirement condition in Step 1. |

## 8. One-line summary

> **Tier 1** discovers which ingredient matters, by adding one at a time.
> **Tier 2** is a *pre-registered container* — held empty until a tier-1
> experiment produces a located, evidenced defect, of **either** kind: a broken
> mechanism (repair) **or** an uninterpretable / unreplicated result
> (verification). It then says how the response must be specified: one variable
> changed, against a **disproof-style** control engineered to win by the wrong
> mechanism, under an unchanged information budget. Tier 1 answers "what helps?";
> tier 2, *when warranted*, answers "why — and is the result even trustworthy?"
