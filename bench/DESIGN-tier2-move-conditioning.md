# Tier-2 entry: the move score cannot condition on board-wide context

*Status: **registered as a gate.** Not a plan, not a chosen repair. This note
records a located defect and asserts that **it must be repaired before further
*conditional* evaluation is interpretable.** It deliberately does **not** assert
that "the tier-2 upgrade" *is* the fix — the repair is an open choice, to be
made in Step 3.*

> **See also:** `DESIGN-tier2-template.md` (the container this entry inherits its
> shape and constraints from; §0.1(a) mechanism defect), `HYPOTHESES.md` §H23
> (the behavioural half that produced the defect), `CLAIMS-CHANGELOG.md`
> (2026-10-07 entries), `bench/scripts/h23_behavioural.py`.

---

## What this note is asking for

The user's framing, which this note adopts verbatim as its scope:

> **Do not conclude that the tier-2 upgrade is exactly this change. Conclude that
> this problem should be fixed before further evaluation.**

So the two commitments made here are:

1. **A defect is located and evidenced** (§Step 0). This satisfies the template's
   Step 0 gate, with the caveats listed there.
2. **A class of further evaluation is gated on repairing it** (§The gate). The
   gate is stated precisely: it blocks *conditional* claims, not all claims.

What is **not** committed: which repair to build. §Candidate repairs lists
options with their budget cost and explicitly defers the choice.

---

## Step 0 — the located defect

### The defect, by construction

`L3Scorer` computes a move's score as

```
score(from, to) = from_logit(from) + to_logit(to | from) + promo_slot

from_logit(f)   = W_from[f] . chan[:, f] + b_from[f] + (W_hid @ relu(W_ctx @ ctx))[f]
to_logit(t | f) = W_to_dst[t] . chan[:, t]
                + W_to_src[t] . chan[:, f]
                + W_geom[t] . geom(f, t)
```

The complete set of inputs to `score(from, to)` is therefore:

| Input | Width |
|---|---|
| `chan[:, from]` — encoder channels at the origin square | 24 |
| `chan[:, to]` — encoder channels at the destination square | 24 |
| `geom(from, to)` — geometry features | 9 |
| `ctx` — position-global scalars | 9 |

**Nothing else reaches it.** A threat on the other side of the board, a hanging
piece elsewhere, an open file three ranks away: none of these can alter this
move's score except via the nine global scalars (material ratio, material
balance, total material, check, mobility, castling rights, three phase weights).

**An important qualifier, stated so the defect is not overstated.** The 24
channel planes are not just piece identity — they include `to_move_attacks`,
`opponent_attacks`, `to_move_defends`, `opponent_defends`, `contested` and
`capture_targets`. So the destination square's vector *does* say "is this square
attacked / defended / contested". The receptive field is two squares, but those
two squares carry real tactical annotation. The defect is the *absence of
board-wide context*, not the absence of any local information.

### The evidence that it binds

All from `bench/scripts/h23_behavioural.py`, 2026-10-07, Stockfish depth-12
labels on 583 curated castling-legal positions, 8 initialization seeds.

1. **The learner does not condition castling.** Trained on a castling-rich
   corpus, held-out castling mass is flat across the teacher's castling rank:
   0.587 / 0.525 / 0.544 / 0.575 / 0.551 / 0.530 for ranks 1 / 2–3 / 4–5 / 6–10
   / 11–20 / >20 (Spearman ρ = −0.12 at `sw=1.0`, −0.25 at `sw=0.125`).
2. **It is not a corpus artifact — mostly.** A MIXED corpus (supplying the
   negative examples the RICH-only corpus lacked) raises discrimination
   (castling mass where the teacher wants it minus where it does not) from
   **+0.009** to **+0.047**, ~5×. So part of the flatness was my own design. But
   +0.047 against a conditioned model's ~0.5 is under a tenth of what is needed.
3. **It is not a missing negative gradient.** Supplying explicit negative credit
   for castling wherever the teacher rejected it (124/244 updates confirmed
   applied) suppresses castling **globally**, not conditionally — 0.539 → 0.325
   on `rich_held` and 0.492 → 0.301 on `poor_held`, the same drop on both
   (discrimination +0.025; −0.002 for the larger corpus). Mechanically expected:
   one logit per `(from, to)` pair, so credit generalises across positions.
4. **The information is on the board but unreachable.** Linear probe, held-out
   AUC for "does the teacher want castling here":

   | Features the probe may use | dims | held-out AUC (95% bootstrap) |
   |---|---|---|
   | the 9 global scalars | 9 | 0.629 [0.53, 0.73] |
   | + channels at the castling squares | 57 | 0.606 [0.50, 0.71] |
   | whole-board channel means (**invisible to the logit**) | 24 | **0.743 [0.65, 0.82]** |

   The probe carries a planted-signal positive control (AUC > 0.95 on a label
   that is a threshold of one of its own features), so a low AUC here means "no
   signal", not "broken probe".

### What is demonstrated vs what is suggestive — read this before citing

| Claim | Status |
|---|---|
| The receptive field is two squares + nine scalars | **Demonstrated by construction** (code above). Not in dispute. |
| A trained L3 does not condition castling | **Demonstrated** (8 seeds, two step sizes, rank-bucketed). |
| Corpus contrast and negative gradients do not fix it | **Demonstrated** (matched arms). |
| Board-wide features separate the classes better than visible ones | **Suggestive.** 0.743 [0.65, 0.82] vs 0.606 [0.50, 0.71] — the intervals **overlap**. The comparison is the right direction and the right instrument, but it is not established at this n (34 held-out positives). |
| The defect is the *binding* constraint on castling | **Not established.** It is the best-supported remaining explanation after controls 2 and 3; it is not proven to be the only one. |

The template's Step 0 asks for a located defect *with evidence*. The gate is
satisfied for the defect as **constructed and behaviourally demonstrated**; it is
*not* satisfied for the stronger claim that widening the receptive field will fix
it. **That stronger claim is what the entry would test**, which is the correct
order.

---

## The gate: what must be repaired before further evaluation

This is the operative part of the note. The claim is narrow and testable:

> **Any evaluation whose metric requires the learner to make a
> position-conditional decision — "does it play X *when X is good*" — is not
> interpretable until the move score can see board-wide context. Running more of
> those evaluations first measures the architecture, not the learner.**

### Blocked until repaired

- **H23's behavioural half** (does a trained learner castle where castling is
  good). This is the evaluation that produced the defect; further runs of the
  same shape will reproduce the same flat result for the same structural reason.
- **Any "conditioning" metric of this shape**: castling mass vs teacher rank,
  discrimination between teacher-wants / teacher-rejects held-out sets, and the
  same construction applied to en-passant or any other context-dependent move.
- **Any claim that a change "taught the learner when to play X"** — as opposed
  to "raised how often it plays X", which is a different and weaker claim.

### Not blocked

- **Aggregate strength**: match results, win rates, ≥20 games/pair with SE. A
  match does not ask for a per-position conditional decision.
- **Aggregate imitation metrics on a fixed corpus**: top-k mass, weighted CE,
  policy entropy. These average over positions.

  *But note the repo's own history:* those aggregate metrics have already
  produced two false positives — the pilot's "+0.061 top-5 set-mass gain" was
  mostly non-best-move mass, and the `sw=1.0` CE deterioration was over-
  sharpening. "Not blocked" means *interpretable*, not *sufficient*.

- **H25's primary endpoint** (held-out best-move mass vs Stockfish) is
  aggregate, so it is not blocked — but its interpretation ("a master supervises
  better") should not be read as evidence of conditional play.

### Gate-lift condition (provisional, to be fixed in Step 1)

The gate lifts when a repair (arm B) achieves discrimination that is
**significantly greater than its parameter-matched control (arm C)** and is a
material fraction of what the teacher itself separates. The exact threshold is
**deliberately left un-numbered here**: it requires first measuring the
teacher's own rich/poor castling-mass separation, which has not been measured.
Setting the number before that measurement would be the failure mode the
template warns about — designing the entry to succeed.

---

## Candidate repairs — open, not chosen

Listed with cost so the choice is budgeted rather than assumed. **None is
preferred here.** The `L3` scorer is 7,609 parameters
(`W_from` 1536, `b_from` 64, `W_to_dst` 1536, `W_to_src` 1536, `W_geom` 576,
`W_ctx` 288, `W_hid` 2048, `promo` 25); `n_ch = 24`, `n_ctx = 9`, `h = 32`.

| # | Repair | What it adds | Rough cost | Risk |
|---|---|---|---|---|
| **A** | Extend `ctx` with pooled board features (per-channel means / counts) | board-wide summary into the existing context MLP | tiny (24 × h on `W_ctx`) | pooling throws away *where*; may not help tactically |
| **B** | Enable `VALUE_CHANNELS` (`capture_value`, `danger_value`, `attack_balance`, `own_danger`) at training time — currently off by default for cost | richer per-square annotation | small (`n_ch` 24 → 28) | still per-square, not board-wide; ~40× encode cost |
| **C** | Low-rank board embedding: `score += u_to · (A · vec(board))` | genuine board-wide, position-specific term | moderate — needs a spatially pooled or factorised `A` or it blows past budget (24 × 64 = 1536 per rank) | the one that actually widens the field; must be budgeted carefully |
| **D** | Replace the factored scorer for this decision | full board conditioning | large | changes the level's identity; likely a ladder question, not tier 2 |

**A and B are cheap and may be insufficient. C is the one that addresses the
defect as stated. D probably violates the constraint set.** Choosing between
them is Step 3's job, after Step 1 fixes the hypothesis and Step 2 the budget.

---

## Constraint set inherited (template §3)

| Constraint | Inherited value |
|---|---|
| **Information budget** | **This is the row the entry changes — and it must be declared as such.** The entry's entire point is that the base's information budget is too narrow for conditional decisions. A repair that widens it is admissible *only* if the widening is stated, bounded, and controlled by arm C (§Step 4), which adds capacity **without** widening the input. If the entry cannot be specified that way, it is a ladder rung, not tier 2 — register it as such and stop. |
| **Signal / training** | Same cached Stockfish (depth-12, top-5) labels; same corpus, same splits, same `split_seed`, same epochs, same `search_weight` sweep. |
| **Action space** | Unchanged: 20480, factorised `(from, to, promo)`. |
| **Budget class** | L3 = 7,609. Declared as a fraction with a reason (L3.5 precedent) — **not** quietly padded. |
| **Protocol** | Same `DistributionPolicy` interface, same evaluation script, same 8 seeds, seed-conditional 95% t intervals, `interval_scope` stated. |

---

## Step 1 — hypothesis, drafted for `HYPOTHESES.md` (not yet registered)

Per the template, this must be authored *before* implementation. Drafted here;
**registering it in `HYPOTHESES.md` is the next required step and is not done by
this note.**

> **H-tier2-move-conditioning.** Giving L3's move score access to board-wide
> context — beyond the two squares the move touches and the nine global scalars
> — improves held-out *conditional* agreement with a master teacher, measured as
> castling mass on positions where the teacher wants castling minus positions
> where it does not, **beyond a parameter-matched control that adds the same
> capacity without widening the input**, at equal data budget.
>
> **Falsified by:** arm B within 1 SE of arm C, or B at or below A.

Secondary endpoints (pre-registered, not primary): weighted CE and entropy on
the same held-out sets (to detect the over-sharpening signature that already
produced one false positive), and Spearman ρ of castling mass against the
teacher's castling rank.

## Step 2 — budget

To be declared when a repair is chosen. Rule: state L3's 7,609 and the repair's
total as a fraction, with the reason for any deviation beyond ~1.5×. Repair C's
cost is dominated by the rank of the board embedding and must be computed before
it is selected, not after.

## Step 3 — implementation

A new `Scorer` plugged into the **existing** `L3Policy`, per the template. No new
policy class, no change to `ACTION_SPACE`, no change to the encoder's channel
order (append-only — see `encode.py`'s module docstring; reordering invalidates
saved L3/L4 weights).

## Step 4 — three arms (plus controls)

| Arm | Construction | What it tests |
|---|---|---|
| **A** | base L3 scorer | control |
| **B** | L3 + the chosen repair | the change |
| **C** | **disproof-style control** — L3 padded with the *same* added capacity, but fed only the **existing** input (e.g. extra hidden units in the context MLP, `h` 32 → larger). More parameters, **same input bottleneck.** | If B ≈ C, the win was capacity, not board-width. **B must beat C, not merely A.** |
| **D** | *positive control* — the identical harness on a decision that per-square features *can* make, to show the instrument detects conditioning when it exists | guards against "the experiment cannot see any effect" |

Arm C is mandatory under template §3.1 and is the sharpest arm here: it is
engineered to succeed by the wrong mechanism (capacity) precisely so that its
success would refute the structural claim.

A **verification** sub-item is attached, of flavour (b): *does the defect bind
for non-castling moves?* The receptive-field fact is general by construction,
but the evidence that it binds is castling-specific. One probe on a second move
type decides whether this entry is about castling or about the architecture.

## Step 5 — tier

`ablation`, not a ladder rung. The ladder claim stays untouched. The verification
sub-item is a script plus a results JSON; its output lands in
`CLAIMS-CHANGELOG.md` (a status transition), per template §0.1.

## Step 6 — reporting

SE, decided-game counts, residuals, seed-conditional intervals with
`interval_scope`. A result inside one SE is "not established", never the
preferred direction.

---

## Risks and guards

| Risk | Guard |
|---|---|
| **This note is read as "tier 2 = widen the receptive field"** | The banner and §What this note is asking for state the opposite. The repair is open (§Candidate repairs); only the *problem* is committed. |
| The gate is read as blocking *all* further work | §The gate lists what is not blocked, explicitly. |
| The overlapping AUC intervals are cited as proof | §Demonstrated vs suggestive tabulates the four claims and their status. |
| The repair is judged on aggregate metrics, which already produced two false positives | Primary endpoint is the **conditional** (discrimination) metric; CE/entropy are secondary and pre-registered as *degeneration detectors*, not goals. |
| Arm C is replaced by a passive "no change" control | Template §3.1: arm C must be engineered to win by the wrong mechanism. |
| The threshold is set after seeing the result | Gate-lift condition is explicitly left un-numbered until the teacher's own separation is measured. |
| Budget creep (repair C blows past 7,609) | Step 2 requires the fraction be declared before Step 3. |

## Open questions

1. Is the defect castling-specific or architectural? (Verification sub-item,
   Step 4.)
2. Is the blanket "castle whenever legal" habit specific to castling, or generic
   to any strongly-targeted move? Untested; needs a third corpus targeting a
   different move.
3. Does widening the receptive field help *aggregate* metrics at all, or only
   conditional ones? If only conditional, the gate is narrower than this note
   assumes and should be restated.

---

## Provenance

Triggered by the 2026-10-07 H23 behavioural work: `bench/h23_behavioural_mixed.json`
(8 arms across corpus and step size), `bench/h23_behavioural_conditioning.json`
(castling mass vs teacher rank), `bench/h23_feature_probe.json` (three-variant
separability probe with bootstrap intervals). Harness:
`bench/scripts/h23_behavioural.py`; tests `tests/test_h23_behavioural.py` (24).
