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

---

## Open question for this file

**Is a changelog the right form, or should it be a stateful ledger?** This file
is chronological: it answers "what changed, and when?". The prove/not-prove
document the register promises would be stateful: "what do we currently believe
about each hypothesis?". They may want to merge, with the changelog as the
transition log feeding a verdict table. Not decided.
