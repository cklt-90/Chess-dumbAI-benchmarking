# Board-wide context in a factored move scorer

A tier-2 entry into `chess-rl-bench`, opened 2026-10-07.

*papers have abstracts; so does this. Read the abstract for the claim, §1 for
what is established versus merely indicated, and §6 for the one decision this
document deliberately does not make.*

---

## Abstract

`L3` scores a move as a sum of per-square terms fitted independently. Its
complete input is the encoder channels at the move's **origin** and
**destination** squares plus nine position-global scalars — a two-square
receptive field. This document claims that **this receptive field is a binding
constraint on L3's ability to make position-conditional decisions**, and that a
repair which widens it — under an otherwise unchanged information budget — will
lift materially more of the gate than a parameter-matched control that adds the
same capacity without widening the input.

The claim is **narrow and pre-registered**. It does not say a specific repair is
the answer (§6), does not assert the defect is the *only* binding constraint
(§1), and does not generalise from castling alone (§5). It is falsified by the
repair failing to beat its own matched control: if a padded `L3` with the same
added capacity and the *unchanged* input does as well, the win was capacity, not
board-width, and this entry is refuted.

**Status of the claim today:** a located, behaviourally-demonstrated defect
(§1) and a **partial** verification (§5, new 2026-10-07). The verification
supports the claim's direction but is honestly marginal, and is reported as
such.

---

## 1. The defect: what is established, what is only suggested

The `L3Scorer` computes a move's score as

```
score(from, to) = from_logit(from) + to_logit(to | from) + promo_slot

from_logit(f)   = W_from[f] · chan[:, f] + b_from[f] + (W_hid @ relu(W_ctx @ ctx))[f]
to_logit(t | f) = W_to_dst[t] · chan[:, t]
                + W_to_src[t] · chan[:, f]
                + W_geom[t] · geom(f, t)
```

so the complete input to `score(from, to)` is:

| Input | Width |
|---|---|
| `chan[:, from]` — encoder channels at the origin | 24 |
| `chan[:, to]` — encoder channels at the destination | 24 |
| `geom(from, to)` — geometric features | 9 |
| `ctx` — position-global scalars | 9 |

**Nothing else reaches it.** A threat across the board, a hanging piece
elsewhere, an open file three ranks away: none of these can alter this move's
score except through the nine global scalars (material ratio, material balance,
total material, check, mobility, castling rights, three phase weights).

**A qualifier, so the defect is not overstated.** The 24 planes are not just
piece identity — they include `to_move_attacks`, `opponent_attacks`,
`to_move_defends`, `opponent_defends`, `contested` and `capture_targets`. The
destination vector *does* say "is this square attacked / defended / contested".
The receptive field is two squares, but those two squares carry real tactical
annotation. **The defect is the absence of board-wide context, not the absence
of local information.**

### Established vs indicated — read before citing

| Claim | Status |
|---|---|
| The receptive field is two squares + nine scalars | **Demonstrated by construction** (code above). Not in dispute. |
| A trained L3 does not condition castling on the teacher's rank | **Demonstrated** (8 seeds, two step sizes, rank-bucketed; ρ = −0.12 / −0.25). |
| Corpus contrast and negative gradients do not fix it | **Demonstrated** (matched arms). |
| Board-wide features separate the classes better than visible ones | **Suggestive.** AUC 0.743 [0.65, 0.82] vs 0.606 [0.50, 0.71]: the intervals **overlap**. Right direction, right instrument, not established at this n. |
| The defect binds for a class *other* than castling | **Partially supported, new 2026-10-07** — captures condition weakly, castling is flat, paired difference excludes zero. See §5, including the caveats. |
| The defect is the *binding* constraint on castling | **Not established.** It is the best-supported remaining explanation after the controls; it is not proven to be the only one. |
| Widening the field will lift the gate | **This is what the entry tests.** Asserted nowhere. |

---

## 2. The claim

> **H-tier2-move-conditioning.** Giving L3's move score access to board-wide
> context — beyond the two squares the move touches and the nine global scalars —
> improves held-out *conditional* agreement with a master teacher, measured as
> castling mass on positions where the teacher wants castling minus positions
> where it does not, **beyond a parameter-matched control that adds the same
> capacity without widening the input**, at equal data budget.
>
> **Falsified by:** arm B (the repair) within 1 SE of arm C (the matched
> control), or B at or below arm A (the base).

**Pre-declared before any arm is built.** This is the Step 1 authoring the
tier-2 template requires, and it is written here before Step 3 — the template's
own rule, justified by the H8–H10 precedent (the kill condition was written
*before* L3.5 existed, which is what stopped L3.5 being built to succeed).

**Secondary endpoints** (pre-registered, not primary): weighted cross-entropy
and policy entropy on the same held-out sets — as **degeneration detectors**,
not goals — and Spearman ρ of castling mass against the teacher's castling rank.
The detector matters: aggregate imitation metrics in this repo have already
produced two false positives (the top-5 set-mass gain, and the `sw=1.0` CE
deterioration), both over-sharpening.

---

## 3. The constraint set this entry inherits

From `bench/DESIGN-tier2-template.md` §3, filled in for this entry.

| Constraint | Inherited value | Note |
|---|---|---|
| **Information budget** | **This is the row the entry changes — declared as such.** The entry's whole point is that the base's budget is too narrow for conditional decisions. A widening is admissible **only** if stated, bounded, and controlled by arm C, which adds capacity **without** widening the input. If the entry cannot be specified that way, it is a ladder rung, not tier 2. | The strictest row. |
| **Signal / training** | Same cached Stockfish (depth-12, top-5) labels; same corpus, same splits, same `split_seed`, same epochs, same `search_weight`. | Unchanged. |
| **Action space** | Unchanged: 20480, factorised `(from, to, promo)`. | Unchanged. |
| **Budget class** | L3 = **7,609** params. The repair's total is declared as a **fraction with a reason** (§4) — not quietly padded. | L3.5 precedent. |
| **Protocol** | Same `DistributionPolicy` interface, same evaluation script, same 8 seeds, seed-conditional 95% t intervals, `interval_scope` stated. | Unchanged. |

**The one free variable is the design of the computation.** That is what makes
the result interpretable.

### 3.1 What this entry does *not* change

- The encoder's channel order (append-only — reordering invalidates every saved
  L3/L4 weight; `COLOUR_PAIRED_CHANNELS` in `encode.py` lists the colour-carrying
  channels, and any new colour-specific channel must be added there).
- The action space, the masking rules, the canonicalisation, the game loop.
- The level's identity as a **factored** scorer: it stays a sum of per-move
  terms. The entry widens the *input*, not the *factorisation*.

---

## 4. Budget

To be declared in full when a repair is chosen (§6 names none). The rule is
fixed here so it binds later:

> **State L3's 7,609 and the repair's total as a fraction, with the reason for
> any deviation beyond ~1.5×.**

The cheapest candidate (pooled board features into the existing context MLP)
costs on the order of `24 × h` on `W_ctx`; the low-rank board embedding's cost is
**dominated by the embedding rank** and **must be computed before the repair is
selected, not after** — a repair chosen for its narrative and then budgeted is
the failure mode this section exists to prevent.

---

## 5. Verification already run: is the defect castling-specific? (2026-10-07)

The template's Step 4 asks the tier-2 entry to attach a verification sub-item:
*does the defect bind for moves other than castling?* The receptive-field fact is
general by construction, but the evidence that it **binds** was castling-specific.
One probe on a second move class decides whether this entry is about castling or
about the architecture.

**Design.** Two classes with opposite local-information coverage:

- **captures** — locally decidable: the destination square carries
  `capture_targets`, `opponent_defends`, `contested`;
- **castling** — board-wide: it depends on king safety that cannot reach the
  score.

One **shared** held-out set of **194** positions where *every* class is legal
(this fixes an earlier design defect: a per-class split left the comparison
measured on a 4-of-48 overlap — a spurious paired effect of −0.225 [−1.200,
0.000]). 8 seeds, MultiPV 20, engine depth 12, two step sizes. Metric:
`class_mass / (n_class / n_legal)`, 1.0 = indifferent.

**Three-way pre-registered read** — fixed before the run:

| Outcome | Implication for this entry |
|---|---|
| captures condition, castling does not | defect is **board-wide context** |
| neither conditions | defect is the **training signal**, not the receptive field — subject changes |
| both condition | castling flatness was a **corpus artifact** |

**Result — branch 1.**

| class | step | ρ | 95% CI | excludes 0 |
|---|---|---|---|---|
| castling | `sw=1.0` | +0.022 | [−0.107, +0.181] | no |
| castling | `sw=0.125` | −0.033 | [−0.163, +0.127] | no |
| capture | `sw=1.0` | **−0.192** | [−0.311, −0.023] | yes |
| capture | `sw=0.125` | **−0.204** | [−0.352, −0.065] | yes |

Paired difference (capture ρ − castling ρ), same 194 positions: **+0.206**
[+0.003, +0.410] at `sw=1.0` and **+0.194** [+0.007, +0.381] at `sw=0.125`.

**What this settles, and what it does not.**

- It **selects the subject**: this entry is about *board-wide context*, not the
  training signal. Branch 1 is the pre-registered branch for that conclusion.
- It is **marginal**. Both paired lower bounds are barely off zero (+0.003,
  +0.007), and captures are only *weakly* negative (ρ ≈ −0.20). This is branch 1,
  not a strong branch 1.
- The classes are **differently shaped** (castling rank-1 share 0.010 across 21
  ranks; captures 0.650 across 16), so a flat ρ on captures would have been
  *weaker* evidence than the same ρ on castling.
- Therefore: **the receptive field is *a* binding constraint on castling, and is
  not established to be the only one.** Any result in §7 must carry that caveat.

Source: `bench/h23_two_class_conditioning.json`; instrument validity:
`bench/h23_class_rank_variance.json`.

---

## 6. The decision this document deliberately does not make

**It does not name the repair.** The candidates are listed below with cost so the
choice is budgeted rather than assumed; **none is preferred here.** Choosing is
Step 3's job, after Step 2 has fixed the budget.

`L3` is 7,609 parameters: `W_from` 1536, `b_from` 64, `W_to_dst` 1536, `W_to_src`
1536, `W_geom` 576, `W_ctx` 288, `W_hid` 2048, `promo` 25; `n_ch = 24`,
`n_ctx = 9`, `h = 32`.

| # | Repair | What it adds | Rough cost | Risk |
|---|---|---|---|---|
| **A** | Extend `ctx` with pooled board features (per-channel means / counts) | board-wide summary into the existing context MLP | tiny (24 × h on `W_ctx`) | pooling throws away *where*; may not help tactically |
| **B** | Enable `VALUE_CHANNELS` (`capture_value`, `danger_value`, `attack_balance`, `own_danger`) at training time — currently off by default for cost | richer per-square annotation | small (`n_ch` 24 → 28) | still per-square, not board-wide; ~40× encode cost |
| **C** | Low-rank board embedding: `score += u_to · (A · vec(board))` | genuine board-wide, position-specific term | moderate — needs a spatially pooled or factorised `A` or it blows past budget (24 × 64 = 1536 per rank) | the one that actually widens the field; must be budgeted carefully |
| **D** | Replace the factored scorer for this decision | full board conditioning | large | changes the level's identity; likely a ladder question, not tier 2 |

**A and B are cheap and may be insufficient. C addresses the defect as stated.
D probably violates the constraint set.** The label letters here are *not* the
arm letters of §7 and must not be confused with them.

---

## 7. The experiment

### 7.1 Arms

| Arm | Construction | What it tests |
|---|---|---|
| **A** | base L3 scorer | control |
| **B** | L3 + the chosen repair | the change |
| **C** | **disproof-style control** — L3 padded with the *same* added capacity, fed only the **existing** input (e.g. extra hidden units in the context MLP, `h` 32 → larger). More parameters, **same input bottleneck.** | If B ≈ C, the win was capacity, not board-width. **B must beat C, not merely A.** |
| **D** | *positive control* — the identical harness on a decision that per-square features *can* make, to show the instrument detects conditioning when it exists | guards against "the experiment cannot see any effect" |

**Arm C is mandatory** (template §3.1) and is the sharpest arm here: it is
engineered to succeed by the wrong mechanism — capacity — precisely so that its
success would refute the structural claim. A passive "no change" control cannot
refute a mechanism claim.

### 7.2 Gate-lift condition (deliberately un-numbered)

The gate lifts when the repair (arm B) achieves discrimination **significantly
greater than its parameter-matched control (arm C)** and is a **material fraction
of what the teacher itself separates**.

The threshold is **left un-numbered here on purpose.** It requires first
measuring the teacher's own rich/poor castling-mass separation, which has not
been measured. Setting the number before that measurement is the failure mode the
template warns about — designing the entry to succeed.

### 7.3 Implementation

A new `Scorer` plugged into the **existing** `L3Policy` (template Step 3). No new
policy class, no change to `ACTION_SPACE`, no change to the encoder's channel
order (§3.1).

### 7.4 Tier

`ablation`, **not a ladder rung.** A ladder entry asserts "beats the one below
it"; this entry asserts "the repaired design beats a matched control under the
*same* inputs". Different claim, different tier. The ladder claim stays untouched.

### 7.5 Reporting

SE, decided-game counts, residuals, seed-conditional intervals with
`interval_scope`. **A result inside one SE is "not established", never the
preferred direction.**

---

## 8. What is gated, and what is not

Tier 2's value depends on the gate being a gate and not a wall. This is the
operative part.

### Blocked until repaired

- **H23's behavioural half** — the evaluation that produced the defect. Further
  runs of the same shape will reproduce the same flat result for the same
  structural reason.
- **Any "conditioning" metric of this shape** — castling mass vs teacher rank,
  discrimination between teacher-wants / teacher-rejects sets, and the same
  construction on en passant or any context-dependent move.
- **Any claim that a change "taught the learner when to play X"** — as opposed to
  "raised how often it plays X", a weaker claim.

### Not blocked

- **Aggregate strength**: match results, win rates, ≥20 games/pair with SE. A
  match does not ask for a per-position conditional decision.
- **Aggregate imitation metrics on a fixed corpus**: top-k mass, weighted CE,
  policy entropy. These average over positions.

  *But note the repo's history:* those metrics have already produced **two** false
  positives — the pilot's "+0.061 top-5 set-mass gain" was largely non-best-move
  mass, and the `sw=1.0` CE deterioration was over-sharpening. **"Not blocked"
  means interpretable, not sufficient.**

- **H25's primary endpoint** (held-out best-move mass vs Stockfish) is aggregate,
  so it is not blocked — but its interpretation ("a master supervises better")
  must not be read as evidence of conditional play.

---

## 9. Risks and guards

| Risk | Guard |
|---|---|
| This document is read as "tier 2 = widen the receptive field" | §2 states the claim is what the entry **tests**; §6 names no repair; §1 tabulates demonstrated vs indicated. |
| The gate is read as blocking *all* further work | §8 lists what is not blocked, explicitly. |
| The overlapping AUC intervals are cited as proof | §1 tabulates the six claims and their status; the AUC row is marked suggestive. |
| The verification's marginal branch-1 is over-read | §5 states the marginality, the shape mismatch, and that "only constraint" is *not* established. |
| The repair is judged on aggregate metrics | Primary endpoint is the **conditional** metric; CE/entropy are pre-registered as degeneration detectors. |
| Arm C is replaced by a passive control | §7.1 (template §3.1): arm C must win by the wrong mechanism. |
| The threshold is set after seeing the result | §7.2 leaves it un-numbered until the teacher's separation is measured. |
| Budget creep (the repair blows past 7,609) | §4 requires the fraction be declared before the repair is selected. |
| Castling-only evidence is generalised | §5 ran the two-class verification for exactly this reason, and reports its marginality. |

---

## 10. Open questions

1. **Is the defect castling-specific or architectural?** *Partially answered
   (§5): captures condition weakly, castling flat — branch 1, marginal.*
2. **Is a blanket "castle whenever legal" habit specific to castling, or generic
   to any strongly-targeted move?** Untested; needs a third corpus targeting a
   different move.
3. **Does widening the receptive field help *aggregate* metrics at all, or only
   conditional ones?** If only conditional, the gate is narrower than this entry
   assumes and should be restated.
4. **Does the repair beat its own matched control once the step size is sane?**
   Given the over-sharpening history in this repo, the experiment must run at a
   step size that does *not* itself produce the artifact — otherwise a repair
   effect could be confounded with an overshoot effect.

---

## Provenance

Triggered by the 2026-10-07 H23 behavioural work and the 2026-10-07 two-class
conditioning verification.

- `bench/DESIGN-tier2-move-conditioning.md` — the gate entry this document
  expands (Step 1 drafted there; this is the standalone paper-shaped version).
- `bench/DESIGN-tier2-template.md` — the container and constraint set.
- `bench/h23_two_class_conditioning.json` (+ `h23_class_rank_variance.json`) —
  the §5 verification.
- `bench/h23_behavioural_mixed.json`, `h23_behavioural_conditioning.json`,
  `h23_feature_probe.json` — the castling evidence behind §1.
- `HYPOTHESIS-REVIEW-2026-10-07.md` §5 — the census in which this entry's subject
  was fixed.
- Harness: `bench/scripts/h23_behavioural.py`; tests `tests/test_h23_behavioural.py`.
