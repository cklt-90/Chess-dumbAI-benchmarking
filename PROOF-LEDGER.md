# PROOF-LEDGER.md

**What we currently believe about each hypothesis.** One row per hypothesis, one
state, re-derived from the register, the changelog, and the stored results — not
from memory. This is the document `HYPOTHESES.md`'s header promises as the
"separate document (to be created)".

## Why this file exists, and how it differs from its companions

Three artefacts, three questions. Merging any two loses information:

| Artefact | Question it answers | Write policy |
|---|---|---|
| `HYPOTHESES.md` | *what could happen* (guess + falsification condition) | filled in place |
| `CLAIMS-CHANGELOG.md` | *what changed*, and why | **append-only** — never rewritten |
| `PROOF-LEDGER.md` (this) | *what we currently believe* about each row | **rewrites in place** |

**Decision recorded 2026-10-07 (user-approved): standalone, not a merge.** The
merge question flagged at the foot of `CLAIMS-CHANGELOG.md` is now settled: a
stateful verdict table rewrites in place, while a changelog is append-only, so
folding the ledger into the changelog would force the opposite write policies
into one file. The ledger is the *state*; the changelog is the *transition*; the
register is the *guess*.

**Rule for using it.** A row may move up only on a stored result or a code fact,
never on narrative. Where a row's support is a code fact rather than a run, the
"basis" column says so. **Do not quote a row without its basis.**

---

## State table

State vocabulary (matching the register's status key, plus one):

- **supported** — tested, evidence in the repo, direction holds
- **located** — mechanism established by reading/code, strength unmeasured
- **partial** — tested, direction holds but caveated / conditional
- **open** — runnable, untested
- **blocked** — cannot be answered without a prerequisite that does not exist
- **refuted** — tested and failed, or superseded by evidence
- **n/a** — not a hypothesis

| ID | State | Basis | Caveat to carry |
|---|---|---|---|
| **H1** | supported | code fact: search tests pass (ordering invariant) | no stored result block yet; free to record |
| **H2** | supported | code fact: encode-mirror tests pass (~1e-08 / exact 0.0 on ep plane) | no stored result block yet; free to record |
| **H3** | blocked | needs the flagged re-fit (`unfinished`-as-draws) | methodological; un-run |
| **H4** | partial | baseline: 578 / 533 / 485 monotone | `d3 vs d2` was draws; needs 20 games/pair |
| **H5** | partial | baseline: L3 vs L1 no wins | L5 arm untested; **DQ-1** (two runs conflated) |
| **H6** | supported | baseline: `material` loses to every L2 depth; `random` 24/24 decided | **DQ-2** records differ between docs |
| **H7** | partial | baseline: largest residual `material vs L3` | **fragile on DQ-3** — sits on L3's provisional rating |
| **H8** | blocked | needs trained budget-matched arm; not budget-matched (100 / 10.3 / 20.7%) | do not read a loss as "tying is worse" |
| **H9** | blocked | same trained-arm prerequisite | `L3-flat` exists (verified 2026-10-07); the blocker is untrained weights |
| **H10** | located | code fact: `_geom_features` lacks a destination-file term; takes `abs(tf-ff)` | missing-feature, not proven tying; strength cost unmeasured |
| **H11** | blocked | needs trained value head | baseline (untrained) recorded and negative; do not re-run untrained |
| **H12** | open | runnable: node counters on a fixed position set | cleanest mechanistic claim; never measured at scale |
| **H13** | supported | baseline: every ablation SE > every delta | general threshold untested |
| **H14** | partial | baseline: correlational (`unf` tracks untrained/blind) | causal arm (falls with training) untested |
| **H15** | open | runnable: re-run one pre-fix configuration | **highest priority — every baseline-derived row is conditional on it** |
| **H16** | blocked | needs `L3-flat` at matched budget **and** trained weights | blocker text in the register is **stale** (S-4): the level exists |
| **H17** | open | runnable: transition-proximity stratifier, no new model | strongest circumstantial support; **best bet** |
| **H18** | open | runnable: classify L2-vs-depth-5 disagreements | general principle settled; local magnitude open |
| **H19** | open | runnable: full round-robin, check top-3 residuals all involve `material` | one edge is not a structural pattern |
| **H20** | supported | baseline: `material` mid-table, beats L1, loses to all L2 depths | `SETTLED`; do not re-derive |
| **H21** | partial | settled half: untrained-vs-untrained ≈ random | open half: do random weights imply random *preferences*? un-run |
| **H22** | n/a | rejected by design: not falsifiable at this scale | do not re-propose |
| **H23-mech** | supported | mask from `board.legal_moves`; 100% of legal special moves admitted | definitional |
| **H23-behav** | refuted (in the located direction) | 2026-10-07: trained L3 castles but **un-conditioned** (ρ = −0.12 / −0.25); corpus + negative-gradient controls fail | "under-play is a signal-coverage gap" holds; "teach it and it conditions" failed |
| **H24** | blocked | needs per-paradigm driver (thin) | hooks exist (`opponent=` override) |
| **H25** | partial | 2026-10-07: master-trained L3 **does** castle → resolves feature-vs-signal toward **signal** | learned behaviour is a blanket habit; master-vs-self-play still blocked |
| **H26** | open | runnable: random-vs-random corpus, bench vs untrained clone | weak-material signal expected; run it *first* |
| **H27** | open | needs `AP` parameter + driver | generalises H26 into a randomness dial |
| **H28** | blocked | needs driver + value head (styles 3–6) | styles 1–2 need only a credit rule |

---

## The rows that moved most recently

- **H23-behav → refuted (located direction), 2026-10-07.** The registered
  falsification criterion ("positions where castling is clearly best") has almost
  no support in any distribution tested (≤8/583). The criterion itself needs
  re-specifying before it can be used — a defect *in the register*, not the code.
- **H25 → partial (feature-vs-signal resolved), 2026-10-07.** A true master
  supplied; the learner castles, so the fork resolves toward **signal** — but not
  the teacher's *judgement*.
- **H10 → located (re-verified against code), 2026-10-07.** `SquareLocalScorer`
  and `FlatScorer` both import; the geometry gap is a code fact.
- **H9/H16 blocker corrected, 2026-10-07.** `L3-flat` **exists** — the blocker is
  untrained weights and the budget contradiction, not a missing level (S-4).

## The findings that are *not* rows (methodology, but load-bearing)

These are not hypotheses; they are things the repo has learned about *measuring*,
and every row's basis inherits them.

- **Over-sharpening artifact (C5/X6).** Three apparent effects — the top-5
  set-mass "gain", the `max` target-rule difference, the teacher-quality
  difference — were one artifact: too large an effective step. `search_weight`
  is still **1.0**, the damaging value.
- **Adam does not fix an aggregate overshoot (C6) — it amplifies it.** CE +19.2
  vs the control's +0.83; best/top-k *fell*. Per-parameter scale-invariance is
  the wrong invariance for an aggregate target-mass mismatch.
- **A byte sha is not a behaviour guard (S-7).** The `l3_*` provenance guards
  fired on a verified-inert refactor. Fixed by adding a behaviour digest to
  `l3_target_rule_ablation.py`; **not yet propagated to the other three guards.**
- **Receptive field is a binding constraint on castling, not the only one
  (C7/§5).** Two-class verification: captures condition weakly (ρ ≈ −0.20),
  castling flat (ρ ≈ 0), paired difference excludes zero — but marginally.

---

## Open data-quality flags (block row upgrades)

| Flag | Rows it blocks | State |
|---|---|---|
| **DQ-1** H5 evidence trail inconsistent | H5 | open |
| **DQ-2** records differ between docs | H6, and any baseline quote | open — quote the JSON |
| **DQ-3** L3's 3-3 splits vs every L2 depth | H7, L3's rating | **open, load-bearing** — needs per-game colours |
| **DQ-4** H16 blocker stale | H16 | open (S-4) |
| **DQ-5** H5 "12 decided" wording | H5 | open |
| **H15** baseline comparability | **every baseline-derived row** | open — highest priority |

---

## One-line summary

> **The register** holds the guesses. **The changelog** holds what changed.
> **This ledger** holds where each guess stands today, with the basis that lets a
> reader decide whether to trust the state — and it names the flags (DQ-3, H15)
> that make several current states conditional rather than settled.
