# TODO — document updates (temporary)

**Temporary working note.** Delete once the three documents below exist. It
records *what must be written and in what order*, not results — results live in
`bench/*.json` and the memory logs.

---

## Order matters

```
experiments (Adam arm, two-class conditioning)
        |
        v
1. HYPOTHESIS-REVIEW (new)   <- must exist BEFORE any tier-2 work
        |
        v
2. TIER-2 HYPOTHESIS (new, standalone, paper-shaped)
        |
        v
3. PROVE / NOT-PROVE LEDGER (still missing)
```

---

## 1. New hypothesis review

**Why it must come first.** Tier 2 is gated on a *located, evidenced* defect.
Deciding whether the located defect is the right one is exactly what a review
does. Writing the tier-2 paper before the review would fix the framing before
the evidence has been re-censused.

**Precedent:** `HYPOTHESES-REVIEW-2026-10-03.md` — a temporary cross-reading
note. It did three things worth repeating: censused verdicts into tiers, listed
what *cannot* be concluded from the docs alone, and raised data-quality flags
**without resolving** doc-vs-JSON disagreements.

**That precedent must not be reused verbatim.** Its own C4 ("the encoder has no
castling-rights channel") is now stale — `encode.CHANNELS` has 24 planes
including `castle_w`/`castle_b` at 22/23, and `value.evaluate()` includes
`king_shelter_score`. Any new review must re-derive its claims from current
code, not inherit them.

**Must contain:**

- A fresh verdict census over H1–H28 against *current* code and stored results.
- The stale-claim list, including stale claims in the **older review itself**.
- What the docs cannot conclude, stated explicitly.
- The known data-quality flags that remain open: H5/H6/H7 prose records that do
  not match `bench/results-2026-10-02.json`; L3's 3-3 splits with every L2 depth
  (DQ-3), still unexplained and still load-bearing for H7; H15's baseline
  comparability.
- The two pre-existing test flakes, named as flakes rather than failures
  (`test_ensemble.py` statistical assertion, `test_guided.py` wall-clock).
- **The guard finding from 2026-10-07**: `bench/scripts/l3_*` provenance guards
  compare a *byte* sha of `perceptron.py`, so they fired on a refactor that was
  verified behaviour-preserving. Note the fix (behaviour digest) and that a byte
  mismatch alone is not evidence of a behaviour change.

**Must NOT:** resolve a doc-vs-JSON disagreement by choosing a side without a
run; promote a conditional interval to a general one; treat "not blocked" as
"sufficient".

---

## 2. Tier-2 hypothesis — a new standalone document

**Framing the user asked for:** like a **newly opened paper**, not a rehash or
rework of `HYPOTHESES.md`. It is *not* a new row appended to the register.

**Seed:** `bench/DESIGN-tier2-move-conditioning.md` Step 1 holds a drafted
hypothesis. That is the starting point, not the deliverable. The paper must
absorb whatever the two-class conditioning experiment finds — in particular, if
captures condition and castling does not, the claim is about *board-wide
context*; if neither conditions, the defect is in the training signal and the
paper's subject changes.

**Must contain:**

- Claim, and the falsification condition stated before any result.
- The located defect, with the demonstrated/suggestive split kept explicit
  (the receptive field is demonstrated by construction; that it *binds* is not).
- Budget: L3's 7,609 params and the repair's total as a **declared fraction**
  with a reason. Repair C's cost depends on the embedding rank and must be
  computed before selection, not after.
- Arms **A** (base) / **B** (repair) / **C** (disproof-style control — same
  added capacity, unchanged input) / **D** (positive control). B must beat C,
  not merely A.
- The gate-lift threshold, left **un-numbered until the teacher's own rich/poor
  separation is measured** — deliberate, to stop the entry being designed to
  succeed.
- What is blocked vs not blocked. Conditional claims blocked; aggregate metrics
  not blocked — with the caveat that aggregate metrics have already produced two
  false positives in this repo.

**Must NOT:** name the repair (that is Step 3); claim generality from
castling-only evidence; describe itself as a plan or workflow.

---

## 3. Prove / not-prove ledger — still missing

`HYPOTHESES.md`'s header promises verdicts are tracked in "a separate document
(to be created)". `CLAIMS-CHANGELOG.md` holds **transitions** (what changed
status and why); the ledger would hold **current state** (what each hypothesis
stands at today). Companions, not substitutes. The merge question is flagged
open at the end of `CLAIMS-CHANGELOG.md` and should be settled here.

---

## Open questions to settle while writing

1. Does the ledger live in `HYPOTHESES.md`'s promised slot, or merge with
   `CLAIMS-CHANGELOG.md`? Pick one and record the reason.
2. The two-class experiment's outcome decides whether the tier-2 paper's subject
   is "board-wide context" or "the training signal". Do not draft past it.
3. Should `HYPOTHESES.md` §H23's stale paragraphs be struck through or deleted?
   Current posture keeps them as faithful quotes so
   `test_docs_consistency.py::test_hypotheses_channel_note_matches_code` stays
   green.
