# Hypothesis review — what the existing docs can conclude (2026-10-03)

**Temporary working note.** No games were played and no Python was run for this
review. Every number below is quoted from `HYPOTHESES.md`, `bench/README.md`,
`ONBOARDING.md`, `bench/results-2026-10-02.json` and
`bench/ablation-2026-10-02.json`. Where two sources disagree, both are quoted
and the disagreement is flagged rather than resolved (see §4). This note is
point-in-time: delete or fold it into the register once the next run lands.

---

## Headline

- Of the 22 live hypotheses (H1–H23, H22 rejected), **6 carry a tested verdict,
  5 are partially supported, 2 are blocked, and 9 are untouched.**
- **Five conclusions are solid today** (§2) — but almost every tested verdict
  leans on the single 2026-10-02 baseline, whose comparability H15 flags as
  *unverified*. All baseline-derived conclusions are real but **conditional**.
- The deepest findings were not produced by matches at all: H10's mechanism and
  H23's encoder gap were **found by construction** — reading the code, not
  playing games. The register's best evidence is sometimes free.
- Cross-reading surfaced **5 data-quality flags** (§4), including prose records
  that do not match the stored JSON, and one stored pattern (L3 splitting 3-3
  with *every* L2 depth) that is implausible as strength and may be a book
  artifact — which would mean H7's "cycle" is measuring the opening book, not
  the policies.

## 1. Verdict census

| tier | hypotheses | count |
|---|---|---|
| Conclusion supported (high confidence) | H6, H20, H23-mech, H10-mech, H13 | 5 |
| Directionally supported (medium, caveated) | H4, H5, H7, H14, H23-behav, H11-baseline | 6 |
| Not established / open | H1, H2, H3, H12, H15, H17, H18, H19, H21 | 9 |
| Blocked (training) | H8, H9, H10-strength, H11, H16 | 5 arms |
| Rejected by design | H22 | 1 |

## 2. Conclusions the docs already support (high confidence)

**C1 — Search beats non-search, decisively, at every tested depth (H6, H20).**
`random` lost all 24 decided baseline games; `material` lost to every L2 depth
in the baseline (0-2-4, 0-0-6, 0-0-6) and sits mid-table (255) above L1 (165).
Also the entire history of the field, which is why H20 is `SETTLED`.
*Confidence: high. Caveat: exact W-D-L records differ between sources (DQ-2).*

**C2 — Untrained learners are indistinguishable from blind play (H5 L3-arm,
H21 settled half).** Four independent observations agree: L3 vs L1 produced no
wins (0-6 in the register's account; 0-0-1 + 5 unf in the stored JSON); the
H8/H9 first run was 11-of-12 unfinished; L6's blind members scored near chance
(0.417 / 0.375 after the ledger fix); untrained L5 does not beat L2. This is
the repo's honesty gate and it currently holds on every arm tested.
*Confidence: high. The L5 arm of H5 remains untested at scale.*

**C3 — L3.5's per-square weakness is located, not hypothetical (H10
mechanism).** `_geom_features` has no destination-file term and takes
`abs(tf - ff)`, so `b1→c3` and `g1→h3` are byte-identical; two knights carry
identical channel vectors; the tied destination row is origin-blind. The model
**cannot express "a knight belongs near the centre"**, which L3 can. Found by
construction, before any match. *Confidence: high on mechanism; the strength
cost is unmeasured and confounded — it is a missing-feature limitation, not
proven to be a tying limitation.*

**C4 — Special moves are legal everywhere but played almost only by search
(H23).** The mask admits 100% of legal special moves (definitional — it is
built from `board.legal_moves`). Behaviourally: L2-d2 castled in 3/3 games;
L1, L3, material, random castled ≤3 times in 6 (1, 1, 3, 1). En passant
appeared twice across all L1/L3 games, zero elsewhere. The gap is located:
`evaluate()` has no castling/en-passant/mobility term, and the encoder has no
castling-rights channel (only one scalar in `board_context_features`).
Castling needs a *reason*; only search has one. *Confidence: high on the
mechanical half and the located gap; medium on the behavioural rates (3–6
games per level).*

**C5 — Six games per pair cannot establish pairwise gaps (H13).** In the L5
ablation every standard error (79–93) exceeded every pairwise delta (36–102),
and the apparent ordering (`prior` best, `blind+eval` worst) is demonstrably
noise: three of six matches were draws or 3-3 splits, and `blind+eval`'s whole
negative rating comes from one 0-6 match. *Confidence: high for the ablation
roster; the general threshold is untested.* This is the meta-conclusion: at
default settings the harness produces **measurements, not conclusions**.

## 3. Directionally supported, with caveats (medium confidence)

**H4 — Depth is monotone.** Ratings 578 / 533 / 485 order correctly, but
d3 vs d2 was +1-0=4 — the gap is draws. "Deeper is stronger" is supported;
"deeper wins decisively" is not. 20 games/pair is the stated minimum to move
SE below the gap.

**H5 — Untrained does not beat the level below.** Supported for L3 (no wins
in either account), L5 arm untested. See DQ-1: the two quoted records differ.

**H7 — The ladder is not transitive.** Supported as recorded: largest residual
is `material vs L3` (observed 0.500, expected 0.259). **Fragile on inspection:**
that edge sits on L3's *provisional* rating, which is engineered by L3's 3-3
splits against all three L2 depths — an implausible strength signal for an
untrained model (DQ-3). If those splits are a book/colour artifact, the cycle
is an artifact too.

**H14 — `unf` predicts untrainedness.** Correlational support is clean
(random 12, L1 10, L3 11 vs L2-d1 0, L2-d3 1). `unf` also tracks *compute*:
the slowest recorded match (772s, ~4× the next) was 6/6 unfinished. The causal
arm (falls with training) is untested and is what would make `unf` a free
training-progress signal.

**H11 — Baseline only.** With an untrained model the seams demonstrably do
nothing (+eval worst at −102; `blind+eval vs prior+eval` 6/6 unfinished at
772s — eval without prior is not just weak, it is slow, because the prior is
what makes the eval affordable). This says **nothing** about the trained
claim, and even the untrained *direction* rests on one 0-6 match (the
table's largest residual). Blocked on Task #9; do not re-run untrained.

## 4. Data-quality flags found while cross-reading

- **DQ-1 — H5's evidence trail is inconsistent.** Register prose: "L3 vs L1
  was 0-6, all draws by fivefold repetition." Stored baseline: 0-0-1 with 5
  unfinished. Two different runs are conflated. The verdict survives (zero L3
  wins in both), but the citation should name its run.
- **DQ-2 — Quoted records differ between documents.** H6 prose: "L2-d1 17-1-0
  vs material." Stored baseline: 4-2-0 in 6 games. The README sample row
  (L2-d1 524.4 ±103.5, 17-1-0, 18 games) matches neither the stored JSON row
  (19-11-6 over 36 games) nor H4's quoted 485. Multiple runs are quoted
  interchangeably — H15's incomparability hazard is already present *in the
  prose*. Quote the JSON, not the narrative.
- **DQ-3 — L3's 3-3 splits vs every L2 depth.** An untrained perceptron
  splitting 3-3 with depth-1, depth-2 *and* depth-3 alpha-beta is not a
  plausible strength signal. It is consistent with "the book decided the
  games" (wins by the side the book favours), which the aggregate JSON cannot
  confirm or deny — **per-game colours are not recorded**. These splits drive
  both L3's provisional rating and H7's largest residual, so two hypotheses
  lean on one suspicious pattern. Follow-up: record per-game colours and count
  wins by side-to-move.
- **DQ-4 — H16's blocker is stale.** It reads "blocked on `L3-flat`", but
  `L3-flat` is implemented (Group C banner, ONBOARDING). The live blockers are
  the ±10% budget contradiction (783 / 1445 / 7225) and untrained weights.
- **DQ-5 — Minor wording.** H5 says L3's rating is "fitted from 12 decided
  games out of 25"; the stored matrix gives 25 decided games (12W-1D-12L,
  50.0% score). "12" conflates wins with decided games.

## 5. Cross-cutting conclusions (visible only by reading across hypotheses)

- **X1 — Every tested verdict inherits H15's caveat.** The baseline may
  straddle the `book[i]` fix. Order of operations: H15 → re-baseline →
  re-quote every `[x]`. Until then, all ratings are *conditional on one
  unverified file*.
- **X2 — The learnable signal lives near irreversible transitions
  (circumstantial, multi-source).** In the L6 ledger, decisive games spread
  member accuracy across 0–1 while unfinished games squeeze everyone into
  0.80–0.85; blind policies shuffle into caps (`unf` 10–12); the no-progress
  rule already encodes the same fact as a *stopping* rule. **H17 is the
  best-supported untested hypothesis** and is runnable today with a sampling
  wrapper.
- **X3 — Two representation gaps were found the same way: by construction.**
  No castling-rights channel (H23) and no destination-file geometry (H10).
  Both fixes are append-only-safe. Together they mean current learner weakness
  is partly an *input* weakness, not an *algorithm* weakness — and the next
  representation work is already specified.
- **X4 — Aggregation is as dangerous as learning.** L6's default (absolute)
  weights let a blind majority (0.514) outvote the one member that sees
  material; the relative rule fixes it structurally (blind share → 0.032) but
  can only reach *parity* with the best member, because blind members have no
  surplus to contribute. Two ledger bugs produced *fake* evidence (a random
  member scoring 0.429 ≈ 9× chance; a dead safety check accepting 200/200
  random moves). Any future ensemble claim must state which weighting rule and
  which ledger version it used.
- **X5 — Quantised training needs a movement assertion.** L3.5's `lr/64` rule
  put each update at 0.4 grid units, which `np.round` sends to zero: five of
  eight tensors stayed bit-identical through 30 games / 14,400 updates while
  the loss looked normal. Error feedback is the recorded fix; the general rule
  — *any divided learning rate on a quantised model must be checked for actual
  movement* — belongs in every future trainer.

## 6. What cannot be concluded from the docs alone

- The colour split of any stored match (per-game colours are not recorded) —
  DQ-3 is unresolvable without a re-run.
- Whether `unf` falls with training (H14's causal arm).
- Every trained claim: H8, H9, H10-strength, H11, H16.
- H17, H18, H19, H21 — all runnable today; no stored output exists.
- H3 (unfinished-games convention) — needs the flagged re-fit.

## 7. Recommended order for the next runs (the docs' own gates)

1. **H15** — verify the baseline; everything quotes it.
2. **H4 at `--games 20`** — the only verdict-adjacent claim with a known path
   to significance.
3. **H21, H12** — cheapest: no games / node counters on a fixed position set.
4. **H17** — strongest circumstantial support; needs a sampling wrapper, not a
   level.
5. **Record existing test evidence as results for H1/H2** — the symmetry
   numbers (exact 0.0 on the en-passant plane; 1.5e-08 for L3/L3.5) and the
   ordering invariant (Nf6 = Nc6 = 90) already exist; writing the result
   blocks upgrades two `[ ]` to `[x]` without playing a game.
6. **Training (Task #9)** — unblocks H8/H9/H10-strength/H11/H16. Until then,
   re-running those arms is explicitly an anti-pattern.
