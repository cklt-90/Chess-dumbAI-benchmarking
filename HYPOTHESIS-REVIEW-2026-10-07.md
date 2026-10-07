# Hypothesis review — a fresh census before tier 2 (2026-10-07)

**Point-in-time working note.** Written before any tier-2 work, as the gate the
user specified: *"hypothesis-review.md will have to be made before tier 2."* It
re-derives every verdict from **current code and stored results**, not from the
2026-10-03 review, and it does **not** inherit that review's claims.

**What this note is, and is not.** It is a cross-reading: verdict census, stale
claims, what cannot be concluded, and open data-quality flags. It is **not** a
researcher — no new games were played for it and no hypothesis is upgraded
without a stored result behind it. Where a doc and a JSON disagree, **both are
quoted and the disagreement is flagged, never resolved by choosing a side** —
resolving one requires a run, and a run is a separate act with its own record.

**How to read it against the 2026-10-03 review.** That review is a `then` record
and is deliberately not retro-edited (`test_docs_consistency.py` pins its stale
numbers). This one is a `now` record. Where they differ, **this one is current
only if the code it cites is current**; every code claim below was read or
executed today.

---

## Headline

- The register has grown from H1–H23 (H22 rejected) to **H1–H28**: Group G
  (H24–H27, the training-signal axis) and Group H (H28, the feedback-style axis)
  were added, and **all five new rows are `[ ]` or `[~]` — none tested.**
- **The centre of gravity has moved.** The deepest findings of the last two days
  are not about any level: they are about **measurement**. Three apparently
  separate effects — a top-5 mass "gain", a target-rule difference, and a
  teacher-quality difference — turned out to be **one artifact: over-sharpening
  from too large an effective step**. That is now the register's most-repeated
  lesson and it is still not wired into production.
- **Tier 2 is open and gated on a located, evidenced defect.** The gate is
  `bench/DESIGN-tier2-move-conditioning.md`; its subject was decided today by
  the two-class conditioning experiment, which landed on the **first**
  pre-registered branch (see §5).
- **Several conclusions are now solid, and several are still conditional on the
  single 2026-10-02 baseline** whose comparability H15 flags as unverified. The
  distinction between the two is the main thing a census is for.

## 1. Verdict census (current)

Status key is the register's own: `[ ]` untested · `[~]` in progress ·
`[x]` tested · `[!]` falsified · `[-]` blocked.

| Tier | Hypotheses | Count |
|---|---|---|
| Solid (tested, evidenced, or located by construction) | H1†, H2†, H6, H13, H20, H23-mech | 6 |
| Directionally supported, caveated | H4, H5, H7, H14 | 4 |
| Located / tested-negative, strength or direction resolved | H10, H23-behav, H25 | 3 |
| Runnable today, untested | H12, H17, H18, H19, H21, H26, H27 | 7 |
| Blocked | H3, H8, H9, H11, H16, H24, H28 | 7 |
| Rejected by design | H22 | 1 |

Total: 28. † H1 and H2 are invariants with passing tests but **no stored result
block** in the register — see §4, S-6. They are "solid by test", not "solid by
registered result". H23 is split into its mechanical half (solid) and its
behavioural half (tested, negative in the located direction) rather than counted
twice; the "missing feature" claim the behavioural work superseded is recorded as
stale text in §4 (S-1), not as a separate row.

**Change since 2026-10-03.** The 2026-10-03 census had 22 live hypotheses; this
one has **28**. The new rows are all in the two newest groups and all untested.
The verdicts for H1–H23 are largely unchanged **except** H23, which moved from
"directionally supported (medium)" to "mechanical half solid, behavioural half
tested and *negative in the located direction*" — a downgrade with a located
cause, which is the informative kind.

## 2. Conclusions the docs and code support today (high confidence)

**C1 — Search beats non-search, decisively, at every tested depth (H6, H20).**
Unchanged from 2026-10-03 and re-verified as statements about the *levels*, not
about a run: `MinimaxEngine` (negamax + alpha-beta + MVV-LVA + TT + iterative
deepening + quiescence) is present in `chessrl/search.py`; `material` sits
mid-table; `random` loses the decided games. *Caveat unchanged: exact W-D-L
records differ between sources (DQ-2), so quote the JSON, not the narrative.*

**C2 — Untrained learners are indistinguishable from blind play (H5, H21-settled
half).** Unchanged. This remains the repo's honesty gate.

**C3 — L3.5's per-square weakness is located by construction (H10).**
Re-verified today against code, not quoted: `squarelocal.py` exposes
`SquareLocalScorer` and `L35Policy`; `L3-flat` (`FlatScorer`) imports and the
level is registered. The `_geom_features` destination-file gap is a **code
fact** and survives any re-reading. *Confidence: high on mechanism; strength
cost still unmeasured and confounded with the parameter deficit (H8's 100% /
10.3% / 20.7% caveat).*

**C4 — 6 games per pair cannot establish pairwise gaps (H13).** Unchanged and
now doubly supported: the ablation SEs still exceed the deltas, and the whole
over-sharpening episode is a second demonstration that **default settings
produce measurements, not conclusions.**

**C5 — The `max` target rule is not wrong; the effective step was too large.**
**This is now the register's load-bearing methodology finding** and it must have
a home. Re-verified from `bench/l3_target_rule_ablation.json`: the
magnitude-matched control (`max_scaled`, `search_weight * 0.3742`) removed
**~89%** of the CE damage (0.830 → 0.095 at the pilot's scope), and the
dose-response sweep (`max@1` +0.830 → `max@0.0625` −0.007) is the signature of
overshoot. *A wrong target would not be cured by shrinking the same push.*
*Confidence: high. Still not wired in: `L3Config.search_weight` remains 1.0.*

**C6 — Adam does not rescue an aggregate overshoot; it amplifies it.**
**New today**, and a genuine negative result (see §5). Measured in the same
frozen-corpus ablation: the `adam` arm at `lr * 1` produced CE **+19.2** and
entropy **−2.14**, against the `max@1` control's CE +0.83 / −0.81 — roughly
**23× and ~2.6× worse**. Best-over-top-k **fell** (0.252 → 0.222), which is the
pre-registered *second* branch of the discriminator: the overshoot is
**aggregate across ranks, not per-parameter in scale**, so a per-parameter
scale-invariant optimiser cannot shrink it. *Confidence: high for this corpus;
design is a matched ablation, all three SGD controls reproduced exactly.*

**C7 — The receptive field is a binding constraint on castling, and it is *not
the only* one.** **New today**, from the two-class conditioning experiment: on a
shared 194-position held-out set, capture-class preference conditions weakly on
the teacher rank (ρ ≈ −0.19 / −0.20, CI excludes 0) while castling-class
preference is flat (ρ ≈ +0.02 / −0.03, CI includes 0), and the **paired**
difference excludes zero at both step sizes. *Confidence: moderate — branch 1
of the pre-registration, but with marginal lower bounds; see §5 for the honest
limits.*

## 3. Directionally supported, with caveats (medium)

**H4 — Depth is monotone.** Unchanged: ordering correct, gaps are draws.
20 games/pair remains the stated path to significance; not yet run.

**H5 — Untrained does not beat the level below.** Unchanged for L3; the L5 arm
is still untested at scale. **DQ-1 still applies** (two different runs
conflated in the prose) — see §6.

**H7 — The ladder is not transitive.** Supported *as recorded*, but **still
fragile on inspection**: its largest residual sits on L3's provisional rating,
which is engineered by L3's 3-3 splits vs every L2 depth (**DQ-3**, still open,
still unexplained). If those splits are a book artifact, H7's cycle is an
artifact. **Do not quote H7 without DQ-3.**

**H14 — `unf` predicts untrainedness.** Correlational half only; the causal arm
(falls with training) is still untested. `unf` also tracks *compute*.

**H23 — behavioural half, tested 2026-10-07, negative in the located
direction.** A master-trained L3 *can* be pushed to castle (0.028 → 0.556 at
`sw=1.0`), but it does so **equally** where the teacher rejects castling, and
flat across the teacher's castling rank (ρ = −0.12 / −0.25). The corpus-contrast
and negative-gradient controls do **not** fix it. So the binding constraint is
**conditioning**, not expression. **Restated falsification criterion:** the
registered one ("positions where castling is clearly best") has almost no
support in any distribution tested (≤8/583), so it needs re-specifying before
it can be used — this is a live defect *in the register*, not in the code.

## 4. Stale claims found while cross-reading

Findings are numbered the way the register numbers data-quality flags; the
2026-10-03 review used DQ-1…DQ-5 and X1…X5, so this note continues **S-n** for
stale-claims to avoid collision.

- **S-1 — H23's "no castling-rights channel" paragraph is stale (present
  tense).** Re-verified today: `encode.CHANNELS` has **24** binary planes with
  `castle_w`/`castle_b` at indices **22/23**, `BINARY_CHANNELS == 24`,
  `TOTAL_CHANNELS == 28`. The paragraph at `HYPOTHESES.md` ~line 502 is written
  in the present tense and is false. **Action taken in this pass: delete-and-
  replace** (user's choice), with the consistency test updated accordingly —
  see §7. The *mechanical* statement is not the interesting one; the interesting
  one is that the *behavioural* half was then tested and came back negative in a
  different way.
- **S-2 — The 2026-10-03 review's own C4 is stale.** It states the encoder has
  no castling-rights channel. That review is a dated snapshot and must **not** be
  retro-edited, so this is a flag on the *citation*, not on the file: **do not
  quote C4 as current.**
- **S-3 — The 2026-10-03 review's headline "22 live hypotheses" and its census
  counts are a `then` record.** They are correct as history. The current count
  is 28 (§1). Cite the date when quoting either.
- **S-4 — H16's blocker line is stale.** Register reads `[-]` blocked on
  `L3-flat`. Verified today: `L3-flat` **exists** (`FlatScorer` imports;
  registered in `bench/levels.py`; ONBOARDING lists it). The live blockers are
  the ±10% budget contradiction (7609 / 787 / 1573) and untrained weights, not
  the level's absence. (This is the same class as the 2026-10-03 DQ-4; it has
  not yet been fixed in the register text.)
- **S-5 — H11's `[-] blocked` posture is correct but its command is stale.**
  It points at `--roster ablation`, which still requires a trained value head
  that does not exist. Not a defect; noted so the blocker is not misread as
  a missing flag.
- **S-6 — H1 and H2 carry passing tests but no stored result block.** The
  invariants hold (the search tests and the encode-mirror tests pass), yet the
  register still shows `[ ]`. **The evidence exists; the result block does
  not.** This is free: writing the two blocks upgrades two rows without playing
  a game (the same point the 2026-10-03 review made as recommendation 5, still
  not done).
- **S-7 — The byte-sha provenance guard is not a behaviour guard.**
  **New today, and it is a methodology defect, not a doc typo.** The
  `bench/scripts/l3_*` provenance guards compare a **byte** sha of
  `perceptron.py`. On 2026-10-07 the guard fired (`byte_matches: false`) even
  though the only change was a verified-inert refactor (extracting
  `train_on_targets`, making `train_on_search_feedback` a wrapper). Diagnosed in
  order: not line endings (pure LF; LF-normalised hash equals raw); not
  semantic (the pilot hash matches commit `36d3add`; current `92d4cb9` is the
  refactor), and confirmed **executably** by replaying the pilot trajectory —
  top-k delta `0.061126762900030385` vs the pilot's `0.0611267629000304`,
  identical to 15 significant figures. **A byte mismatch alone is not evidence
  of a behaviour change.** Fixed for this script by adding a **behaviour digest**
  (replay the first 32 cached records through the update path; hash the resulting
  weights) alongside the byte sha. **Not yet propagated to the other three
  byte-sha guards.**

## 5. The two results that decide tier 2 (new today)

These are the experiments that were run *for* this review. They are recorded
here as results; the paper-shaped claim is a separate document.

### 5.1 Adam arm — refutes "use a standard optimiser"

**Question.** The `max` overshoot (C5) is a step-magnitude problem, so would a
standard optimiser (Adam) remove it? Adam's step is ≈`lr·sign(g)`, roughly
scale-invariant per parameter.

**Result.** No — it amplifies it, and the amplification is *diagnostic*.

| arm | CE Δ | entropy Δ | best/top-k |
|---|---|---|---|
| `max` (control, production rule) | +0.83 | −0.81 | 0.252 |
| `sum` | −0.015 | −0.003 | 0.233 |
| `max_scaled` (magnitude-matched) | −0.011 | −0.012 | 0.236 |
| **`adam` @ lr×1** | **+19.2** | **−2.14** | **0.222** |
| `adam` @ lr×0.1 | +0.022 | −0.044 | 0.232 |
| `adam` @ lr×0.01 | 0.0 | 0.0 | 0.227 |

**Read.** The pre-registered discriminator had two branches; this is the second:
*"if the overshoot is aggregate not per-parameter: CE stays near `max@1` or
worsens, entropy collapses, best/top-k DROPS."* All three signatures present.
The mechanism is stated in §6. **`adam@0.01` is *not* evidence Adam works at
small lr** — its step (1e-4) is below the fixed-point grid, so `np.round` zeroes
it (the X5 trap). It is a zero-update arm, not a converged one.

### 5.2 Two-class conditioning — decides the tier-2 subject

**Question.** Is the two-square receptive field a *castling-specific* limit or an
*architectural* one? Pre-registered three-way read:
`captures condition + castling doesn't` → defect is board-wide context;
`neither` → defect is the training signal; `both` → corpus artifact.

**Design.** One **shared** held-out set of **194** positions where every class is
legal (fixing the 4-of-48 near-disjoint split defect of the first attempt), 8
seeds, MultiPV 20, engine depth 12, two step sizes. Metric:
`class_mass / (n_class / n_legal)` — 1.0 = indifferent.

| class | step | ρ (rank vs preference) | 95% CI | excludes 0 |
|---|---|---|---|---|
| castling | `sw=1.0` | +0.022 | [−0.107, +0.181] | no |
| castling | `sw=0.125` | −0.033 | [−0.163, +0.127] | no |
| capture | `sw=1.0` | **−0.192** | [−0.311, −0.023] | **yes** |
| capture | `sw=0.125` | **−0.204** | [−0.352, −0.065] | **yes** |

Paired difference (capture ρ − castling ρ), same 194 positions:

| step | mean | 95% CI | excludes 0 |
|---|---|---|---|
| `sw=1.0` | +0.206 | [+0.003, +0.410] | yes |
| `sw=0.125` | +0.194 | [+0.007, +0.381] | yes |

**Read: branch 1 — the defect is board-wide context.** Captures — whose
destination square carries `capture_targets`, `opponent_defends` and `contested`
— condition *weakly*; castling, which depends on board-wide king safety that
cannot reach the score, is flat.

**Honesty about the strength of this read.**
- The effect is **marginal**: both paired lower bounds are barely off zero
  (+0.003, +0.007), and captures are only *weakly* negative (ρ ≈ −0.20). With
  43+ matched positions and 8 seeds, this is **branch 1, not a strong branch 1.**
- The classes are **differently shaped** (castling rank-1 share 0.010 across 21
  ranks; captures 0.650 across 16), so a flat ρ on captures would have been
  *weaker* evidence than the same ρ on castling. This does not change the
  direction but it bounds the strength.
- **Instrument validity was checked before the experiment** (the earlier
  `probe_classes` run was discarded when it turned out to split per class and
  leave 4/48 overlap), so this is a repaired, not first-try, result.
- **Therefore:** the receptive field is *a* binding constraint on castling, and
  it is **not established to be the only one.** The tier-2 paper must carry that
  same caveat.

## 6. Cross-cutting conclusions

- **X6 — One artifact wore three costumes (extends X5).** The top-5 mass "gain",
  the target-rule effect, and the teacher-quality effect were all
  over-sharpening from a too-large effective step. **The production rule still
  runs at the damaging value (`search_weight = 1.0`).** Until that is wired in,
  every future imitation metric inherits the same false-positive risk.
- **X7 — Scale-invariance is the *wrong* invariance for this bug.** Adam is
  invariant per parameter; the overshoot is *aggregate across ranks* (the `max`
  rule hands out ~3.2 units of target probability over the listed top-k while
  the policy has 1.0 to allocate). Hence Adam cannot help, and its failure is
  *evidence for the aggregate diagnosis*, not a null result.
- **X8 — Controls keep winning.** `max_scaled` refuted the target-rule story;
  the parameter-matched arm is mandatory for tier 2; the MIXED-corpus control
  moved the castling story; the shared-held-out fix moved the two-class story;
  the behaviour digest refuted the byte-guard alarm. **The repo's controls have
  a better hit-rate than its treatments.**
- **X9 — "Blocked" and "not blocked" are different from "sufficient".** Tier 2's
  gate blocks *conditional* claims, not aggregate ones — but aggregate metrics
  have already produced **two** false positives here (the set-mass gain and the
  `sw=1.0` CE deterioration). Interpretable ≠ sufficient.

## 7. Actions taken in this pass (so the review is not just prose)

- **H23 delete-and-replace** (user's choice, overriding the earlier
  keep-as-quote posture): the present-tense stale paragraph in `HYPOTHESES.md`
  is removed and replaced with current text; `HYPOTHESES.md`'s channel note now
  states the **current** count, and
  `tests/test_docs_consistency.py::test_hypotheses_channel_note_matches_code`
  is updated to pin the current phrasing.
- **Standalone prove/not-prove ledger created** — see §8; the merge question at
  the foot of `CLAIMS-CHANGELOG.md` is now settled.
- **Behaviour digest added** to `bench/scripts/l3_target_rule_ablation.py`
  (S-7); propagation to the other three guards is *not* done and is named as
  open work.

## 8. Where the ledger lives (the open question, settled)

`HYPOTHESES.md` promises a separate prove/not-prove document; `CLAIMS-CHANGELOG.md`
promises it and flags the merge question. **Decision: a new standalone
`PROOF-LEDGER.md`, not a merge.** Reason: the three artefacts answer three
different questions and a verdict table rewrites in place while a changelog is
append-only —

| artefact | question | write policy |
|---|---|---|
| `HYPOTHESES.md` | what could happen (guess + kill condition) | filled in place |
| `CLAIMS-CHANGELOG.md` | what *changed*, and why | append-only |
| `PROOF-LEDGER.md` | what we *currently* believe about each row | rewrites in place |

Merging the last two would force a stateful table into an append-only log.

## 9. What cannot be concluded from the docs alone

- **The colour split of any stored match** — per-game colours are not recorded;
  **DQ-3 is unresolvable without a re-run.**
- **Whether `unf` falls with training** (H14 causal arm).
- **Every trained claim:** H8, H9, H10-strength, H11, H16 — still blocked on a
  trained arm (and, for H3, on the flagged re-fit).
- **H17, H18, H19, H21, H26, H27** — runnable today; no stored output exists.
- **Whether the tier-2 receptive-field repair actually lifts the gate** — by
  design; that is the paper's claim, and it is *unproven*, not assumed.

## 10. Open data-quality flags (unchanged except where noted)

| Flag | State | Why still open |
|---|---|---|
| **DQ-1** H5's evidence trail | open | register prose (0-6 draws) ≠ stored JSON (0-0-1, 5 unf); two runs conflated |
| **DQ-2** quoted records differ between docs | open | quote the JSON, not the narrative |
| **DQ-3** L3's 3-3 splits vs every L2 depth | **open, load-bearing** | implausible as strength; drives L3's rating and H7's largest residual; needs per-game colours |
| **DQ-4** H16's blocker stale | **still open** | verified today: `L3-flat` exists (S-4) |
| **DQ-5** H5's "12 decided games" wording | open | stored matrix gives 25 decided |
| **H15** baseline comparability | **open, highest priority** | every baseline-derived conclusion is conditional on one unverified file |

**Two pre-existing test flakes, named as flakes not failures:**

| Test | Symptom | Why it is a flake, not a finding |
|---|---|---|
| `tests/test_ensemble.py` statistical assertion | occasional borderline miss | statistical threshold near a sampled boundary |
| `tests/test_guided.py::test_depth_three_middlegame_finishes_quickly` | wall-clock (e.g. 31.2s vs 25s) | time-based assertion; fails in isolation too; every file it touches is unmodified by recent work; passed at 28.1s earlier the same day |

Neither is a correctness regression; both should be made deterministic
(fixed seed / relative budget) rather than deleted.

## 11. Recommended order for the next work (the docs' own gates)

1. **Write the tier-2 paper** (task after this review) — the gate's Step 1.
2. **H15** — verify the baseline; everything quotes it. Still un-run.
3. **Wire the step-size finding into production** — `search_weight = 1.0` is the
   damaging value; X6 says every imitation metric inherits the risk until fixed.
4. **Record existing test evidence for H1/H2** — free `[ ]`→`[x]` (S-6).
5. **Propagate the behaviour digest** to the other three byte-sha guards (S-7).
6. **Resolve DQ-3** on the next run that plays games: record per-game colours.
7. Then the runnable-today Group F rows (H17 first — strongest circumstantial
   support).

---

## Provenance

Code read/executed today: `src/chessrl/encode.py` (24 binary planes,
`castle_w`/`castle_b` at 22/23, `BINARY_CHANNELS == 24`, `TOTAL_CHANNELS == 28`),
`src/chessrl/perceptron.py` (`L3Config.search_weight = 1.0`),
`src/chessrl/squarelocal.py` (`FlatScorer` present), `src/chessrl/search.py`,
`src/chessrl/value.py` (`king_shelter_score` at 227, wired into `evaluate()` at
346), `bench/levels.py`, `chessrl.masks.ACTION_SPACE == 20480`.

Results: `bench/l3_target_rule_ablation.json` (Adam arm),
`bench/h23_two_class_conditioning.json` (two-class + paired difference),
`bench/h23_class_rank_variance.json` (instrument validity).
