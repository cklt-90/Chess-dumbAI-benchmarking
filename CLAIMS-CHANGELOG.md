# CLAIMS-CHANGELOG.md

A chronological ledger of **claims and caveats that changed status** — raised,
retired, revised, relocated, downgraded, upgraded.

## Why this file exists, and why it is not called `CHANGELOG.md`

Three existing artefacts record history, and none of them covers this:

| Artefact | Holds | Why it is not this |
|---|---|---|
| `git log` | code changes | Code history is already tracked. A conventional `CHANGELOG.md` would duplicate it. |
| `.workbuddy-ai/memory/YYYY-MM-DD.md` | raw work, append-only, per-day | No single view. A claim revised three times over a week is three unlinked entries. |
| `HYPOTHESES.md` | guesses + falsification conditions + current status | It holds the *current* status. It does not record *transitions*, and its format is a hypothesis, not a retirement. |

What is missing is the answer to: **"this claim/caveat — what happened to it?"**
That is what a changelog is for, applied to claims rather than code. Hence the
name.

**Relationship to the promised prove/not-prove document.** `HYPOTHESES.md` says
a separate *prove / not-prove* document is "to be created", to hold verdicts.
This file is its companion, not its replacement: the ledger would hold the
**current state** of each hypothesis, this file holds the **transitions** between
states. A verification entry's output ("this caveat no longer applies") lands
here; the resulting verdict lands in the ledger.

## How to use it

- **Append-only.** Never edit or delete an entry. A later entry may supersede an
  earlier one — say so in the later entry, do not rewrite history.
- **One entry per status change**, not per experiment. An experiment that changes
  nothing does not appear here.
- **A retirement needs evidence.** "We no longer believe X" requires the result
  that killed it, named. An unevidenced retirement is a deletion with extra steps.
- **Retired is a success.** This is the repo's stated stance: a negative is a
  result. A caveat retired by evidence is progress, and it is recorded as such.

## Entry format

```markdown
## YYYY-MM-DD — <short title>
- **Claim/caveat**: <as it stood before>
- **Change**: retired | revised | relocated | downgraded | upgraded | introduced
- **Was** → **Now**: <old status> → <new status>
- **Because**: <the evidence, named>
- **Source**: <file / commit / results JSON>
- **Knock-on**: <what else this affects, or "none identified">
```

---

## 2026-10-02 — `chessboard` PyPI package cannot be used

- **Claim/caveat**: the project spec referenced the PyPI package `chessboard` for
  board representation.
- **Change**: retired
- **Was** → **Now**: assumed dependency → rejected dependency
- **Because**: reading its source established it is a puzzle-placement solver —
  no move generation, no turns, no pawns. It cannot represent a game.
- **Source**: `ONBOARDING.md`; `src/chessrl/bitboard.py` docstring
- **Knock-on**: `python-chess` is the rules authority repo-wide; a convention
  that has held since.

## 2026-10-02 — "Move ordering can change a search value" (overclaim)

- **Claim/caveat**: an earlier framing implied move ordering could change the
  value alpha-beta returns.
- **Change**: revised
- **Was** → **Now**: "ordering can change the value" → "ordering cannot change
  the *value*; it chooses among moves that *share* that value"
- **Because**: value-invariance is a theorem (Knuth & Moore 1975). The observable
  effect is that the first move to reach `alpha` is kept, so among tied moves the
  ordering decides. Measured: Nf6 and Nc6 both evaluate to exactly 90 at depth 3;
  the plain engine returns Nf6, the informed wrapper returns Nc6 because L1's
  table weights the b8 origin higher (0.0385 vs 0.0256).
- **Source**: `ONBOARDING.md` ("The ordering invariant, stated precisely");
  `.workbuddy-ai/memory/MEMORY.md`
- **Knock-on**: tests assert the value invariant and that any disagreement is a
  genuine tie — not that the move is always identical.

## 2026-10-02 — L3.5 spec: `lr_shared = lr / 64` silently disables the model

- **Claim/caveat**: the design spec stated the tied heads' learning rate must be
  divided by the fan-out (`lr / 64`).
- **Change**: revised (the rule is right; the *value* was wrong)
- **Was** → **Now**: `lr_shared = lr/64` is sufficient → it must clear the
  quantisation grid, or steps round to zero
- **Because**: at `lr=0.05` the divided rate put each step *below* the `1/QUANT`
  grid and `np.round` sent it to zero. Measured: 30 games and 14,400 updates left
  five of eight tensors **bit-identical to initialisation**, while the loss looked
  unremarkable. Fixed with error feedback (bank the sub-grid remainder).
- **Source**: `src/chessrl/squarelocal.py` docstring; `ONBOARDING.md` traps
- **Knock-on**: general rule recorded — any quantised model with a divided
  learning rate needs this check.

## 2026-10-02 — H23 special-move census: "blind/L3 never castle" needs a filtered test

- **Claim/caveat**: the 20-game self-play census (castling ≈ 0 for blind/L3) was
  offered as a diagnosis of the castling feature.
- **Change**: downgraded
- **Was** → **Now**: evidence about the feature → **not** evidence about the
  feature (every game ended by fivefold repetition)
- **Because**: a self-play census in which all games end in repetition cannot
  isolate a feature. Castling must be tested on a filtered, oracle-labelled
  position set, not inferred from self-play.
- **Source**: `.workbuddy-ai/memory/MEMORY.md` (2026-10-05 special-moves section)
- **Knock-on**: the behavioural half of H23 is tested on a curated set; the
  mechanical half stands.

## 2026-10-05 — Self-play is not an informative training arm at this scale

- **Claim/caveat**: L3 trained by self-play outcome-RL would improve with more
  games.
- **Change**: retired (for this regime)
- **Was** → **Now**: "more self-play games = more learning" → the arm produces
  **zero signal by construction** at these caps
- **Because**: with current caps, 19–20 of 20 games hit the ply cap
  (`unfinished`), training only via a weak shaped reward (mean |reward| ≈ 0.053).
  With a lenient cap, **40/40 games end `insufficient_material`, all draws**, and
  finished draws return exactly `(0.0, 0.0)` — **total reward mass 0.00 over 40
  games**. Decisive games: 0/20 at each of caps 120/200/300/400.
- **Source**: `bench/KAGGLE-PLAN.md` ("Self-play signal finding");
  `.workbuddy-ai/memory/2026-10-05.md`
- **Knock-on**: the `selfplay` and `both` Kaggle arms are blocked pending a
  signal fix; the supervised arm is the only one worth running at scale.

## 2026-10-06 — "The `max` target rule is wrong"

- **Claim/caveat**: the L3 pilot's cross-entropy deterioration was caused by the
  `max` normalisation of the training target (`weight / max(weights)`).
- **Change**: relocated (partly retired)
- **Was** → **Now**: "the target rule is wrong" → **step size dominates**;
  a small target-scale residue remains
- **Because**: a magnitude-matched third arm (`max_scaled`, same `max` rule,
  `search_weight * 0.3742`) removed ~**89%** of the CE damage (0.830 → 0.095),
  while the alternative rule (`sum`) gave −0.0006. A dose-response sweep
  (`max@1` +0.830 → `max@0.0625` −0.007) confirms overshoot from too large an
  effective step — a wrong target would not be cured by shrinking the same push.
- **Source**: `bench/l3_target_rule_ablation.json`; `bench/l3_stepex_sweep.json`;
  `.workbuddy-ai/memory/2026-10-06.md`
- **Knock-on**: the earlier framing was **over-stated by its own author and
  corrected in writing**. `max` remains production code; the ablation is not
  wired in.

## 2026-10-06 — The pilot's top-5 set-mass "gain" is not evidence of learning

- **Claim/caveat**: the L3 pilot's +0.061 top-5 probability-mass gain at 500
  positions seen was treated as a learning improvement.
- **Change**: downgraded
- **Was** → **Now**: "set-mass gain = learning" → ~57% of the gain is *non-best*
  mass at every step size; it is largely the policy becoming less diffuse
- **Because**: decomposing the gain into best-move vs non-best-move shows the
  best-move share is **flat at ~40–44% across the whole sweep**, and the gain
  co-occurs with a weighted-CE blow-up and an entropy collapse. Only
  cross-entropy is a legitimate training objective; set-mass is a recall-style
  diagnostic and entropy is a degeneration detector, not a goal.
- **Source**: `.workbuddy-ai/memory/2026-10-06.md` ("Objective framing");
  `bench/l3_stepex_sweep.json`
- **Knock-on**: the honest signal is best-move mass (0.0429 → 0.0690 at `max@1`).

## 2026-10-06 — The L3 held-out pilot's intervals are seed-conditional only

- **Claim/caveat**: the pilot's seed-paired 95% interval on the top-5 mass gain
  ([0.06089, 0.06136]) was stated as an uncertainty statement.
- **Change**: downgraded (scope narrowed)
- **Was** → **Now**: "the effect is established" → the interval covers
  **initialization seeds only**; it is conditional on one frozen corpus and
  held-out set
- **Because**: corpus/held-out sampling was never replicated, so the interval
  does not cover dataset uncertainty. It must not be generalised to freshly
  sampled positions.
- **Source**: `.workbuddy-ai/memory/2026-10-06.md` ("Matched held-out pilot")
- **Knock-on**: this is a **design defect**, not a model defect — the live
  instance of tier-2 flavour (b). The tier-2 response is a corpus-replication arm.
  See `bench/DESIGN-tier2-template.md` §0.1.

## 2026-10-06 — "Disproof-style control" is not a standard term

- **Claim/caveat**: the phrase "disproof-style control" was used in conversation
  as if it were an established term of art.
- **Change**: introduced (flagged non-standard)
- **Was** → **Now**: — → an author's coinage, defined locally
- **Because**: it does not appear in the experimental-design literature under
  that name. Defined in `bench/DESIGN-tier2-template.md` §3.1 as: *a control
  engineered to reproduce the effect by a known-spurious route, so that matching
  it refutes the treatment's claimed mechanism.* The repo's own instance is
  `max_scaled` (see the 2026-10-06 target-rule entry above).
- **Source**: `bench/DESIGN-tier2-template.md` §3.1
- **Knock-on**: if the standard name is found (nearest: *active comparator*),
  replace the coinage.

## 2026-10-06 — Tier-2 is a container, not a workflow

- **Claim/caveat**: the first draft of the tier-2 document read as a runnable
  workflow, implying tier-2 entries could be built now.
- **Change**: revised
- **Was** → **Now**: "a workflow to run" → **an empty pre-registered container**,
  gated on a tier-1 result producing a located defect
- **Because**: tier 2 has no content until tier 1 points at something. Building it
  speculatively is itself the primary failure mode.
- **Source**: `bench/DESIGN-tier2-template.md` §0
- **Knock-on**: Step 0 is a gate; the document should stay unopened until a
  defect exists.

## 2026-10-07 — "A true master supervises better than a weak master"

- **Claim/caveat**: training L3 on a *true* master (Stockfish) labels would beat
  training on the *weak* master (depth-3 search) labels, at matched budget.
- **Change**: retired (for this regime)
- **Was** → **Now**: "a stronger teacher makes a stronger student" → **the whole
  weak-vs-true difference was an over-sharpening artifact of the step size**
- **Because**: at `search_weight=1.0` the true-master arm lost best-move mass
  (−0.0048) but won cross-entropy (−0.336) and entropy (+0.259) — a mixed result
  whose signature is over-sharpening. Re-running the identical comparison at
  `search_weight=0.125` (same corpus, same seeds) collapsed the difference to
  nothing: best-move gap **−0.00010** (48× smaller), CE gap −0.0048 (70×
  smaller). Teacher choice makes no measurable difference once the step is sane.
- **Source**: `bench/l25_master_arm.json`; `bench/l25_master_arm_sw0125.json`;
  `.workbuddy-ai/memory/2026-10-07.md`
- **Knock-on**: the same lesson as the 2026-10-06 target-rule entry, applied to
  *teacher choice* instead of *target rule* — an apparent factor effect that is
  really a step-size/overshoot artifact. Reinforces H20 (capacity/tactics bind
  at low budget). **H25 itself is not settled**: master-vs-*self-play* remains
  blocked on the retired self-play signal; this entry covers only the
  weak-vs-true sub-question.

## 2026-10-07 — "L3 cannot castle (missing feature)"

- **Claim/caveat**: H23's present-tense text states the encoder has **no**
  castling-rights channel, i.e. castling failure is a feature/capacity gap.
- **Change**: revised (user-confirmed; text was stale)
- **Was** → **Now**: "missing feature" → **signal/reward gap** — the ability was
  always present; there was no incentive to use it
- **Because**: verified in code — `encode.CHANNELS` has 24 planes including
  `castle_w`/`castle_b` (22/23) and `COLOUR_PAIRED_CHANNELS` contains `(22, 23)`.
  The planes landed with the H23-fix; the H23 hypothesis text predates it. User
  confirms: "always had ability to, but not any incentive to notice it."
- **Source**: `src/chessrl/encode.py` (verified); `.workbuddy-ai/memory/2026-10-07.md`
- **Knock-on**: `HYPOTHESES.md` §H23 "Concrete, checkable defect found while
  testing" is stale and should be reconciled. The behavioural half (does a
  *trained* L3 castle?) remains open and is now testable with a master-trained
  checkpoint.

## 2026-10-07 — "A trained L3 castles, so the under-play is a signal/coverage gap"

- **Claim/caveat**: H23's behavioural half would resolve as a pure *signal*
  gap — the corpus never presents the castling choice, so supplying a teacher
  that castles should fix it.
- **Change**: revised (half right, and the useful half is the one that fails)
- **Was** → **Now**: "under-play is a signal gap; teach it and it castles" →
  **under-play is a signal-coverage gap, but teaching it yields a *blanket*
  castling habit, not the teacher's positional judgement**
- **Because**:
  1. *The registered test's premise barely exists.* On 583 random-play
     castling-legal positions, castling is the **best** move for L2-d3 in
     **8/583** and for Stockfish (d12) in **7/583**; on 273 quiet-play positions,
     **1/273**. Castling is usually a *good* move, rarely the *single best* move.
     "Both wings open" is rarer still: 10/583 (random), 4/273 (quiet).
  2. *No level castles.* Every level sits at ~**2.5%** castling mass on
     castling-legal positions, which is the uniform no-preference baseline
     (1–2 castling moves out of ~35 legal ≈ 3–6%).
  3. *The learner CAN be pushed to castle.* Training on a castling-rich corpus
     (castling in Stockfish's top-5; 60 positions × 4 epochs) raises held-out
     castling mass **0.028 → 0.556** at `sw=1.0` and **0.028 → 0.136** at
     `sw=0.125`. So the *features* are not the binding constraint for
     **expressing** castling.
  4. *But it is a blanket habit.* The same model castles at the same rate on
     held-out positions where Stockfish does **not** want castling:
     `rich_held` **+0.528** [+0.527, +0.529] vs `poor_held` **+0.521**
     [+0.520, +0.522] — statistically indistinguishable. In those POOR
     positions castling is typically the **17th-best** move (median), **~670 cp**
     worse than best, with 159/167 gaps ≥100 cp. So the learned behaviour is
     actively harmful where the teacher rejected castling, i.e. it is *not* the
     teacher's policy. The matched POOR-trained control does the opposite
     (slight suppression, −0.011).
  5. *And the mass does not track the teacher's rank at all.* Conditioning test
     (train on RICH; measure castling mass on 198 held-out positions bucketed by
     Stockfish's castling rank, MultiPV 20):

     | arm | rank 1 | 2–3 | 4–5 | 6–10 | 11–20 | >20 | Spearman ρ |
     |---|---|---|---|---|---|---|---|
     | rich@sw1 | 0.587 | 0.525 | 0.544 | 0.575 | 0.551 | 0.530 | **−0.12** |
     | rich@sw0.125 | 0.134 | 0.140 | 0.140 | 0.128 | 0.118 | 0.114 | **−0.25** |

     Flat from rank 1 to rank >20. A model that had learned the teacher's
     *judgement* would show a strongly negative ρ; at most there is a very weak
     partial conditioning at the smaller step.
- **Source**: `bench/scripts/h23_behavioural.py` (+ `tests/test_h23_behavioural.py`);
  `bench/h23_behavioural_stockfish.json`; `bench/h23_behavioural_quiet_stockfish.json`;
  `bench/h23_behavioural_train_stockfish.json`; `bench/h23_castling_rank_poor.json`;
  `bench/h23_behavioural_conditioning.json`; `.workbuddy-ai/memory/2026-10-07.md`
- **Knock-on**:
  - The binding constraint is the ability to **condition** castling on the
    position, not the ability to express it. "Add castling to the corpus" is not
    a sufficient fix at this architecture/scale.
  - The blanket-bias control is load-bearing: without it, "a trained L3 castles"
    (0.028 → 0.556) would have been reported as a positive result. Same
    metric-validity lesson as the set-mass and step-size entries above.
  - The earlier "master-trained L3 does not castle" reading was a **corpus
    coverage artifact** (the pilot corpus has castling in the teacher's top-5 in
    **3/500** positions), not a feature or signal limitation.
  - `HYPOTHESES.md` §H23 should be reconciled: the falsification criterion
    ("positions where castling is clearly best") has almost no support in any
    position distribution tested.

---

## 2026-10-07 — "conditioning is the binding constraint" — PARTLY MY OWN CORPUS DESIGN

- **Claim/caveat**: the preceding entry attributes the flat castling-vs-rank
  result to the model's inability to *condition* castling on the position.
- **Was** → **Now**: "the binding constraint is conditioning (capacity)" →
  **"roughly a fifth of the flatness was a corpus-contrast artifact of my own
  design; the rest is real, and part of the remainder is architectural — the
  castling logit cannot see most of the board."**
- **Why the correction**: three controls were run after the headline.
  1. *No negative example ever existed.* Every position in the curated set has
     castling **legal** (that is how the set is built), so a RICH-only training
     corpus perfectly confounds "castling is legal" with "castling is
     targeted". The learner literally cannot see a counterexample. Adding a
     MIXED corpus (RICH + POOR together) raises discrimination
     (castling mass on `rich_held` minus on `poor_held`) from **+0.009** to
     **+0.047** (volume-matched) — a ~5× improvement, so part of the flat
     result was the corpus, not the model. Two MIXED sizes were reported
     (volume-matched and RICH-dose-matched) so the conclusion does not depend
     on which nuisance was controlled.
  2. *But the improvement is far short of conditioning.* +0.047 means the
     model casts ~0.54 where the teacher wants it and ~0.49 where it does not.
     A conditioned model would be ~0.55 vs ~0.03. Achieved discrimination is
     under a tenth of what conditioning requires.
  3. *An explicit negative gradient does not fix it.* Supplying the missing
     half of the update (negative credit for castling wherever the teacher
     rejected it — 124/244 updates confirmed applied) suppresses castling
     **globally**, not conditionally: `mixed_half+neg@sw1` goes
     0.539→0.325 on `rich_held` and 0.492→0.301 on `poor_held`, i.e. the same
     drop on both. Discrimination does not improve (+0.025; −0.002 for
     `mixed_full+neg`). This is mechanically expected: the factored scorer
     computes one logit per (from, to) pair, so credit — positive or negative
     — generalises across every position with a king on e1.
  4. *The information is largely unreachable, not merely unlearned.* A linear
     probe on the features the castling logit is actually a function of
     (the 9 global context scalars, plus the encoder channels at the castling
     origin and destination squares) separates RICH from POOR at held-out AUC
     **0.629 [0.53, 0.73]** (context only) and **0.606 [0.50, 0.71]** (with
     local channels) — CIs that touch chance. The same probe on whole-board
     channel means reaches **0.743 [0.65, 0.82]**. So there *is* signal on the
     board that the castling logit structurally cannot see. The CIs overlap,
     so "architecture blocks it" is **suggestive, not established**.
- **Structural fact established by reading, not inference**: the production
  update path (`train_on_targets` / the pilot's `_apply_cached_targets`)
  applies credit **only to moves in the target list**. A move the teacher did
  not choose receives no gradient at all, so "do not play this" is learnable
  only indirectly, by raising alternatives and letting softmax renormalisation
  act. This bounds what any corpus can teach.
- **Source**: `bench/scripts/h23_behavioural.py` (added MIXED arms,
  `apply_negative_castling`, `probe_features` mode with three feature variants,
  `roc_auc`, `bootstrap_auc_ci`); `tests/test_h23_behavioural.py` (24 tests,
  incl. a planted-signal positive control proving the probe can fit);
  `bench/h23_behavioural_mixed.json`; `bench/h23_feature_probe.json`
- **Knock-on**:
  - "Add castling to the corpus" is necessary but **not sufficient**, and
    "add a negative class" is also not sufficient. Both were tested.
  - The remaining gap is best described as **architectural**: the castling
    decision has no board-wide input. Any fix that keeps the factored
    (from, to) scorer will reproduce this.
  - Do not read the AUC comparison as proof: the two intervals overlap.

---

## 2026-10-07 — tier 2 opened: one entry registered, as a GATE not as a fix

- **What changed**: `DESIGN-tier2-template.md` was an empty pre-registered
  container ("do not build tier 2 until a defect exists"). A first entry is now
  open: **`bench/DESIGN-tier2-move-conditioning.md`**.
- **The framing the user specified, and which the note adopts verbatim**: *do not
  conclude that the tier-2 upgrade IS this change; conclude that this problem
  should be fixed before further evaluation.* Accordingly the note:
  - commits to the **defect** (located, with evidence, and with a
    demonstrated-vs-suggestive table separating what is established from what is
    only indicated);
  - commits to a **gate**: conditional evaluations — "does the learner play X
    *when X is good*"? — are not interpretable until the move score can see
    board-wide context. It also lists what is **not** blocked (aggregate
    strength, aggregate imitation metrics), because a gate that blocks
    everything is not a gate;
  - **declines to name the repair.** Four candidates are listed with budget
    cost; the choice is deferred to Step 3.
- **Claim/caveat**: the entry clears the template's Step 0 gate for the defect as
  *constructed and behaviourally demonstrated*. It explicitly does **not** clear
  the gate for the stronger claim that widening the receptive field will fix it —
  that stronger claim is what the entry would **test**, which is the correct
  order. The gate-lift threshold is left deliberately **un-numbered** until the
  teacher's own rich/poor separation is measured, so the entry cannot be tuned
  to succeed.
- **Source**: `bench/DESIGN-tier2-move-conditioning.md`; cross-linked from
  `DESIGN-tier2-template.md` (§0 banner + new §4.1), `DESIGN-h25-true-master.md`
  (gate interaction: H25's aggregate endpoint is *not* blocked, but must not be
  read as conditional play), and `HYPOTHESES.md` §H23 (new item 5).
- **Open**: Step 1 (authoring the hypothesis in `HYPOTHESES.md`) is drafted in
  the note but **not yet registered** — the template requires it be authored
  before implementation, and no implementation is being started.

---

## 2026-10-07 — the prove/not-prove ledger exists; the merge question is settled

- **Claim/caveat**: the register promises a separate prove/not-prove document,
  and this changelog's own footer flags it as an open question ("changelog vs
  stateful ledger — merge or not?").
- **Change**: introduced
- **Was** → **Now**: "to be created" → **`PROOF-LEDGER.md` created, standalone**
- **Because**: the three artefacts answer three different questions and two of
  them need *opposite* write policies — the register and the ledger rewrite in
  place; the changelog is append-only. Merging the ledger into the changelog
  would force a stateful verdict table into an append-only log.
- **Source**: `PROOF-LEDGER.md`; `HYPOTHESES.md` header; this file's footer
- **Knock-on**: the footer's open question is answered; the ledger carries a state
  row per hypothesis with an explicit **basis** column, and names the flags
  (DQ-3, H15) that make several states *conditional*.

## 2026-10-07 — H23's stale "no castling-rights channel" text deleted

- **Claim/caveat**: `HYPOTHESES.md` §H23 stated, in the present tense, that the
  encoder has no castling-rights channel — the paragraph was kept as a
  struck-through quote so `test_docs_consistency.py` stayed green.
- **Change**: retired (delete-and-replace, user-chosen)
- **Was** → **Now**: "kept as history under a superseded marker" → **replaced
  with current text**
- **Because**: the user chose delete-and-replace over strike-through. Verified in
  code: `CHANNELS` (24 binary) includes `castle_w`/`castle_b` at 22/23,
  registered in `COLOUR_PAIRED_CHANNELS`. The replacement keeps the
  `` `CHANNELS` (24 binary) `` phrase in a *present-tense, true* sentence, so
  `test_hypotheses_channel_note_matches_code` stays green without modification.
- **Source**: `HYPOTHESES.md` §H23; verified against `src/chessrl/encode.py`
- **Knock-on**: the older review's C4 (same stale claim) is a *dated snapshot* and
  must not be quoted as current — flagged in `HYPOTHESIS-REVIEW-2026-10-07.md` S-2.

## 2026-10-07 — tier-2 subject fixed: board-wide context, marginally

- **Claim/caveat**: the H23 behavioural work located a two-square receptive field
  but could not say whether the defect was **castling-specific** or
  **architectural** — the Step-4 verification sub-item the tier-2 entry attached.
- **Change**: upgraded (partially; the verification was run)
- **Was** → **Now**: "defect located; scope unknown" → **pre-registered branch 1:
  the defect is board-wide context** — but **marginally**
- **Because**: two-class conditioning on one **shared** 194-position held-out set
  (captures = locally decidable; castling = board-wide) with 8 seeds, two step
  sizes: capture ρ = −0.192 [−0.311, −0.023] / −0.204 [−0.352, −0.065] (excludes
  0) vs castling ρ = +0.022 [−0.107, +0.181] / −0.033 [−0.163, +0.127] (includes
  0); paired difference +0.206 [+0.003, +0.410] / +0.194 [+0.007, +0.381].
- **Honest limits**: both paired lower bounds are barely off zero and captures are
  only *weakly* negative, so the receptive field is *a* binding constraint, **not
  established to be the only one.** An earlier per-class split left a 4-of-48
  overlap and was discarded; the two classes are also differently shaped.
- **Source**: `bench/h23_two_class_conditioning.json`;
  `bench/h23_class_rank_variance.json`; `bench/TIER2-BOARD-WIDE-CONTEXT.md`
- **Knock-on**: `bench/TIER2-BOARD-WIDE-CONTEXT.md` is the standalone tier-2
  paper; its subject is board-wide context, and it carries this marginality
  explicitly.

## 2026-10-07 — Adam amplifies the overshoot rather than curing it

- **Claim/caveat**: the L3 imitation overshoot is a step-magnitude problem, so a
  standard scale-invariant optimiser (Adam) should remove it.
- **Change**: retired (for this regime)
- **Was** → **Now**: "use a standard optimiser" → **Adam amplifies the artifact
  ~23× and its failure *is* the diagnosis**
- **Because**: in the frozen-corpus ablation (`adam` at `lr × 1`, identical target
  rule and credit) CE = **+19.2** and entropy = **−2.14** against the `max@1`
  control's +0.83 / −0.81; best-over-top-k **fell** (0.252 → 0.222). This is the
  pre-registered *second* branch: the overshoot is **aggregate across ranks, not
  per-parameter in scale**, so a per-parameter scale-invariant step cannot shrink
  it. All three SGD controls reproduced exactly.
- **Note**: `adam@0.01` is **not** evidence Adam works at small `lr` — its step
  (1e-4) is below the fixed-point grid and `np.round` zeroes it (the X5 trap).
- **Source**: `bench/l3_target_rule_ablation.json`;
  `.workbuddy-ai/memory/2026-10-07.md`
- **Knock-on**: `L3Config.search_weight` is still **1.0**, the damaging value —
  the step-size finding remains unwired into production.

## 2026-10-07 — a byte sha is not a behaviour guard

- **Claim/caveat**: the `bench/scripts/l3_*` provenance guards used a **byte** sha
  of `perceptron.py` to detect behaviour drift.
- **Change**: revised (guard strengthened)
- **Was** → **Now**: "byte sha mismatch ⇒ behaviour changed" → **a byte mismatch
  alone is not evidence of a behaviour change**
- **Because**: the guard fired (`byte_matches: false`) on a refactor that was
  verified inert *executably* — replaying the pilot trajectory reproduced the
  top-k delta to 15 significant figures (0.061126762900030385 vs …04). Also ruled
  out line endings (pure LF). Fixed by adding a **behaviour digest**: replay the
  first 32 cached records through the update path and hash the resulting weights.
- **Source**: `bench/scripts/l3_target_rule_ablation.py`; `tests/test_l3_target_rule_ablation.py`
- **Knock-on**: **not yet propagated to the other three byte-sha guards** — named
  as open work in the review (§7, S-7).

---

## 2026-10-09 — H15: the baseline is non-comparable, but the cause is the evaluator, not the book

- **Claim/caveat**: H15 states the 2026-10-02 baseline is not comparable to
  later results **because of the opening-book fix** (`book[i // 2]` → `book[i]`,
  and a book with Black-to-move entries).
- **Change**: revised (supported in outcome, corrected in mechanism)
- **Was** → **Now**: "incomparable because of the book fix" → **"incomparable
  because of the `king_shelter_score` term added to `value.evaluate()`"** — the
  book fix is *not* the cause
- **Because**: a 6-game-per-pair replay of all 21 stored pairs leaves **9 rows
  differing**; zeroing `king_shelter_score` in-process restores **7 of the 9**
  exactly. The book is ruled out twice: `bench/runner.py` was committed
  **already fixed**, in the **same commit** as the baseline (`efcc242`), and
  replaying with the pre-fix `book[i // 2]` does not reproduce either
  (`0-2-4` → `0-0-6`). The 2 rows that do not restore both involve `L3`, whose
  input width changed after the baseline (`TOTAL_CHANNELS`(28) →
  `BINARY_CHANNELS`(24)) alongside the `grad_ctx` chain-rule fix.
- **Source**: `bench/h15_baseline.json`; `bench/scripts/h15_baseline.py`;
  `.workbuddy-ai/memory/2026-10-09.md`
- **Knock-on**: the re-baseline boundary is the **evaluator** change, not the
  book change. Every rating in `bench/results-2026-10-02.json` is confirmed
  incomparable to today's harness; a future re-baseline must state which side of
  `king_shelter_score` it sits on. This makes the 2026-10-03 review's X1 ("all
  ratings are conditional on one unverified file") concrete: the file is now
  verified, and the reason is located.

---

## Open question for this file

**Is a changelog the right form, or should it be a stateful ledger?** This file
is chronological: it answers "what changed, and when?". The prove/not-prove
document the register promises would be stateful: "what do we currently believe
about each hypothesis?". They may want to merge, with the changelog as the
transition log feeding a verdict table. Not decided.
