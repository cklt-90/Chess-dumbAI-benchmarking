# HYPOTHESES.md

A register of testable claims about `chess-rl-bench`. Each entry states the
hypothesis, **the measurement that would falsify it**, the command to run, and
where to record the result.

The purpose is to stop "which level is best" from being the only question asked.
A ladder produces a leaderboard; a set of falsifiable claims produces knowledge.
Every hypothesis here is written so that **a negative is a result**, not a
failure — several are expected to fail, and those are the informative ones.

## How to use this

1. Pick a hypothesis. **Read the falsification condition before running
   anything** — if you cannot state what result would kill it, it is not a
   hypothesis, it is a wish.
2. Run the command. Save raw output to `bench/` as JSON where possible
   (`--out`), never just a terminal screenshot.
3. Fill in the result block in place. Do not delete a falsified hypothesis —
   strike it through and record what actually happened. **A dead hypothesis with
   evidence is worth more than a live one without.**
4. If a result is inside one standard error, write "not established", not the
   direction you preferred.

Baselines already on disk: `bench/results-2026-10-02.json` (7 entrants),
`bench/ablation-2026-10-02.json` (L5 seams). Re-render either with
`python -m bench --from <file>` — no games replayed.

**Status key**: `[ ]` untested · `[~]` in progress · `[x]` tested (see result) ·
`[!]` falsified · `[-]` blocked / not answerable yet

**Groups**: A foundations (is the code sound?) · B the ladder (is each rung
above the one below?) · C the representation axis (L3.5) · D value head and
search seams · E methodology (is the benchmark sound?) · F **claims about chess
and learning itself** (is a common belief true?)

**Runnable today**: H17, H18, H19, H21 need no new levels — see Group F. If you
want a result on the board this week, start there rather than with L3.5.

**Recently tested**: H23 (special moves) — mechanical half supported, behavioural
half has a concrete gap (no castling-rights channel in the encoder; no
castling/en-passant term in `evaluate()`). See Group F.

---

## Group A — the foundations (are the levels even sound?)

### H1 — Ordering cannot change a search value
- **Claim**: For `MinimaxEngine`, changing move ordering changes *which* move is
  returned among equals but never the value at a fixed depth and full window.
- **Falsified by**: any pair of orderings returning different root values at the
  same depth on the same position. This is a correctness invariant, not a
  performance claim; a failure is a bug in the engine, not a finding.
- **Command**: `pytest tests/test_search.py -k ordering_or_value -v`
- **Watch out**: the test must use `_root_with_value`, not `select` — two
  searches can agree on the move and disagree on the value, or vice versa.
- **Status**: `[ ]`

### H2 — Canonicalisation makes one parameter set serve both colours
- **Claim**: `encode(pos)` and `encode(colour_mirror(pos))` agree to ~1e-08 for
  every level, and a greedy policy selects mirrored moves.
- **Falsified by**: any position where the two encodings diverge beyond float
  noise. Non-symmetric positions are the interesting ones — the opening array is
  symmetric and hides rank/file transposition bugs.
- **Command**: `pytest tests/test_encode.py -k mirror -v` and
  `pytest tests/test_perceptron.py -k symmetry -v`
- **Why it matters**: a failure here silently halves the training data (the
  model learns from one colour only). This is the single most expensive class of
  bug in the repo and has already bitten four times.
- **Status**: `[ ]`

### H3 — Unfinished games carry no signal
- **Claim**: excluding unfinished games from ratings is right; including them as
  draws would change the ordering.
- **Falsified by**: a re-fit including `unf` as half-points producing the same
  ordering within one standard error — i.e. the exclusion does not matter.
- **Command**: `python -m bench --from bench/results-2026-10-02.json` and
  compare against a re-fit with unfinished counted as draws. (Requires a small
  patch to `rating.py` to switch the convention; do it behind a flag.)
- **Watch out**: this is a *methodological* hypothesis. If excluding them makes
  no difference, the convention is still defensible (it is the conservative
  choice) but the claim "it matters" is not supported.
- **Status**: `[ ]`

---

## Group B — the ladder (is each rung above the one below?)

### H4 — Depth is monotone (L2 deeper beats L2 shallower)
- **Claim**: `L2-d3 > L2-d2 > L2-d1` in fitted Elo, and each head-to-head is won.
- **Falsified by**: any inversion, or a gap inside two standard errors.
- **Baseline evidence**: already **partially supported** — `578 / 533 / 485`
  with monotone ordering, but `d3 vs d2` was `+1-0=4` (draws, not wins). So the
  claim "deeper is stronger" holds; "deeper wins more decisively" does not.
- **Command**: `python -m bench --only L2-d1 L2-d2 L2-d3 --games 20`
- **Honest reading**: an ordering that is monotone but whose gaps are draws is
  weak evidence. 20 games/pair is the minimum to move the standard error below
  the gap.
- **Status**: `[~]` (ordering observed, significance not established)

### H5 — Untrained learners do not beat the level below
- **Claim**: with random weights, L3 does not beat L1, and L5 does not beat L2.
- **Falsified by**: an untrained L3 winning a match against blind L1.
- **Baseline evidence**: **supported** — L3 vs L1 was 0-6, all draws by
  fivefold repetition. L3's rating (440) is fitted from 12 decided games out of
  25, so it is provisional.
- **Command**: `python -m bench --only L1 L3 --games 10`
- **Why it is in the register**: this is the claim that stops a reader crediting
  architecture where training hasn't happened. It is the repo's honesty gate.
- **Status**: `[x]` — supported for L3; L5 arm untested at scale

### H6 — L2 beats every non-searching level
- **Claim**: L2 at depth ≥1 beats L1, material, and random.
- **Falsified by**: a loss to `material` or `random`. **A loss to `random` is a
  bug report, not a result.**
- **Baseline evidence**: **supported** — L2-d1 17-1-0 vs material
  (97.2%), and random lost 24/24 decided.
- **Command**: `python -m bench --only random material L2-d1 --games 20`
- **Status**: `[x]`

### H7 — The ladder is not transitive (cycles exist)
- **Claim**: there is at least one A>B, B>C, C>A cycle, so no rating ordering
  is fully faithful.
- **Falsified by**: all pairwise residuals small (< 0.1 in win-probability
  terms) across a full round-robin.
- **Baseline evidence**: **supported** — largest residual was
  `material vs L3` (observed 0.500, expected 0.259): material beats L3 head to
  head while rating below it.
- **Command**: `python -m bench --games 20` and read the `residuals` section.
- **Why it matters**: if cycles are real, "the ladder" is a metaphor and the
  HYPOTHESES register is the honest artefact.
- **Status**: `[x]`

---

## Group C — the representation axis (the reason L3.5 exists)

> **L3.5 is now implemented.** `src/chessrl/squarelocal.py` holds the tied arms
> (`SquareLocalScorer`, `L35Policy`, `L35Trainer`) and the `L3-flat` control;
> `bench/levels.py` registers all three. The design doc that specified it has
> been retired — its findings now live in `bench/README.md` ("The L3.5
> experiment") and in the module docstrings, and the claims that turned out
> **wrong** are recorded on H8 and H10 below.
>
> Two spec claims were falsified during implementation and are noted per
> hypothesis: the `lr_shared = lr/64` rule silently zeroed the tied heads at the
> fixed-point grid (fixed by error feedback), and the geometry feature set cannot
> express origin/destination file asymmetry.
>
> **These are still the hypotheses to run.** They were written before the level
> existed, which is why they are worth keeping: the falsification conditions were
> fixed in advance and the implementation did not get to move them.

### H8 — A weight-tied per-square function matches the per-square-weights model
- **Claim**: at a comparable parameter budget, L3.5 is not worse than L3 by more
  than one standard error.
- **Falsified by**: L3.5 losing to L3 by more than ~1 SE over ≥20 games/pair.
- **Command**: `python -m bench --only L3 L3.5 --games 20`
- **Budget caveat, measured**: the arms are **not** budget-matched — they sit at
  7225 / 783 / 1445 (100% / 10.8% / 20.0%). The original ±10% demand was
  self-contradictory (see `bench/README.md`). So a *win* by L3.5 is strong
  evidence for tying, but a *loss* is confounded by the 9x parameter deficit and
  **cannot** be read as "tying is worse". Read H8 alongside H9 for that reason.
- **This is a *decision* hypothesis.** A match (even a slight loss inside noise)
  vindicates the tied bias and licenses spending the freed parameters elsewhere.
  A clear loss says the per-square weight rows were earning their keep — which
  is itself an interesting statement about chess geometry.
- **Status**: `[~]` runnable, untrained null result recorded (see below)

### H9 — Parameter budget, not topology, dominates at this scale
- **Claim**: a flat MLP over `encode()` at matched parameter count does not beat
  the factorised models.
- **Falsified by**: `L3-flat` beating both L3 and L3.5 at equal budget.
- **Command**: `python -m bench --only L3 L3.5 L3-flat --games 20`
- **Measured, and it is the strongest evidence for the control's value**:
  `L3-flat`'s origin head is **exactly constant across squares (std = 0.0)** —
  it gives up spatial structure entirely. That is what makes it a control rather
  than a weaker copy of L3.
- **Why it is the control**: without it, an L3.5 win is indistinguishable from
  "any change at all helps". This is the load-bearing arm of the experiment.
- **Status**: `[~]` runnable, untrained null result recorded (see below)

### H10 — Global structure is not recoverable per-square
- **Claim (the *interesting* one)**: a per-square function cannot express facts
  that are irreducibly about several squares ("open file" is a fact about seven
  other squares), so tying must lose *some* strength on positions distinguished
  only by global structure.
- **ALREADY PARTLY SUPPORTED, from implementation alone** — a rare case of a
  hypothesis confirmed before any match. `_geom_features` carries `dest_rank` but
  **no destination-file term**, and takes `abs(tf - ff)`, so `b1→c3` and `g1→h3`
  produce **byte-identical** geometry vectors. Two knights also have identical
  channel vectors, and the tied destination row is origin-blind. The model
  therefore **cannot express "a knight belongs near the centre, not on the rim"**
  — which L3 can, via its per-destination `W_to_dst` row.
  So the stated weak point is **concrete and located**, not hypothetical.
- **Falsified by**: L3.5 **matching** L3 specifically on a filtered set of
  positions whose best move is determined by a global fact (open file, passed
  pawn, weak square complex, king safety). Given the above, this now looks
  unlikely — which makes it the more worth running.
- **Command**: needs a curated position set, not the opening book. Build ~50
  positions where the best move is chosen by a single named global feature, then
  compare top-1 agreement of L3 vs L3.5 against depth-4 search as the oracle.
- **Caveat**: the located limitation is a *missing-feature* problem, not strictly
  a *tying* problem. Adding a destination-file feature would change the fixed
  feature set the comparison rests on, which is why it was reported and not
  patched. Distinguishing the two is itself worth a result.
- **Status**: `[~]` mechanism located; strength cost unmeasured

### H8/H9 first run — null, and correctly so
- **Result (2026-10-02)**: **not established**, as expected. 4 games/pair,
  11 of 12 games unfinished.
- Untrained L3, L3.5 and `L3-flat` are all equally ignorant; they shuffle into
  the ply cap. This is the same behaviour H5 documents for L3 vs L1.
- **Interpretation**: the comparison is wired correctly end-to-end and says
  nothing until weights are trained. **Do not report this as a finding about
  tying** — a null result on untrained weights is a statement about the harness,
  not about architecture.
- **Follow-up**: H8/H9 require a trained budget-matched arm. Track with H11's
  training workstream; the "same task, same features, same signal, architecture
  the only free variable" framing is only meaningful once training exists.

---

## Group D — the value head and the search seams

### H11 — A trained value head beats the material heuristic as a leaf eval
- **Claim**: L5 with `use_model_eval` and a *trained* value head outperforms
  `use_model_eval=False`.
- **Falsified by**: no improvement after training, or an improvement inside one
  SE.
- **Baseline evidence**: with an **untrained** model the result is **negative** —
  `+eval` alone was worst (`-102` delta), and `blind vs prior+eval` produced 6
  unfinished games out of 6 at 772s. Every SE exceeded every difference.
- **Command**: `python -m bench --roster ablation --games 20` after Task #9.
- **Prerequisite**: this hypothesis is **not testable until the value head is
  trained**. Do not run the ablation again on an untrained model and call the
  result new.
- **Status**: `[-]` blocked on training (baseline recorded)

### H12 — The prior saves nodes but cannot change the answer
- **Claim**: `use_model_prior` reduces node counts at fixed depth; it does not
  change the root value except among equals.
- **Falsified by**: a node-count *increase*, or a value change on a position
  where the tied moves are not actually tied.
- **Command**: node counts are not in the bench report — measure directly:
  run `GuidedEngine` with the flag on/off on a fixed position set and compare
  `.nodes`, plus `_root_with_value` for the value.
- **Why it matters**: this is the cleanest mechanistic claim in the repo and it
  is currently only asserted in tests, never measured at scale.
- **Status**: `[ ]`

---

## Group E — methodology (is the benchmark itself sound?)

### H13 — 6 games per pair is too few to establish anything
- **Claim**: at the default `--games 6`, most pairwise gaps are inside 2 SE and
  therefore not results.
- **Falsified by**: a 6-game run whose gaps all exceed 2 SE.
- **Baseline evidence**: **supported** — every L5 ablation SE (79-93) exceeded
  every pairwise delta (35-100).
- **Command**: compare `python -m bench --games 6` vs `--games 30` on the same
  subset and tabulate how many gaps cross 2 SE.
- **Implication if supported**: the repo's own README says a small gap is not a
  result; this hypothesis quantifies *how* small is too small for this harness,
  which is the number every future run needs.
- **Status**: `[x]` for the ablation roster; general case untested

### H14 — The `unf` column predicts untrainedness
- **Claim**: high unfinished counts track untrained / blind policies, because
  they shuffle instead of committing.
- **Falsified by**: a *trained* policy with a high `unf` count, or a blind
  policy with a low one.
- **Baseline evidence**: **supported** — `random` 12, `L1` 10, `L3` 11 (all
  untrained/blind) vs `L2-d3` 1, `L2-d1` 0.
- **Command**: read the `unf` column in any run; for the stronger claim, track it
  across a training run and check it falls.
- **Why it matters**: if `unf` falls with training, it is a free training
  progress signal that needs no labels.
- **Status**: `[x]` (correlational); causal claim untested

### H15 — The opening book change altered earlier results
- **Claim**: the `book[i]` fix (was `book[i // 2]`, and the book had
  Black-to-move entries) makes results from before the fix **not directly
  comparable** to results after it.
- **Falsified by**: re-running a pre-fix configuration and reproducing the
  pre-fix numbers.
- **Command**: `python -m bench --only L2-d1 material --games 20` and compare to
  the stored `results-2026-10-02.json` row for that pair.
- **Why it is here**: `bench/results-2026-10-02.json` is the baseline for
  future comparisons and it may straddle the fix. **Verify before quoting it as
  a baseline.** This is exactly the kind of silent incomparability that makes a
  register worth keeping.
- **Status**: `[ ]` — **do this one before trusting any stored baseline**

---

## Group F — claims about chess and learning itself

Groups A–E ask "is my harness sound". Group F asks a different question: **is a
widely-held belief about chess or about learning actually true?** These are
claims about the *domain*, not about the code. A failure here is not a bug
report — it is a finding, and the more interesting kind.

Each entry carries an **evidence status**: `OPEN` (genuinely undetermined),
`CONTESTED` (partly established, worth a local re-test), or `SETTLED` (the field
knows the answer; listed for completeness so nobody re-derives it). Do not spend
a week on a `SETTLED` row.

Runnable-today marks whether existing code can answer it: **yes**, **needs a
dataset**, or **needs the level** (a new arm such as `L3-flat` or L3.5).

### H16 — Factorisation buys trainability, not expressiveness
- **Evidence status**: `OPEN`
- **Claim**: the three-factor softmax `P(from) · P(to | from) · P(promo | ...)`
  beats a flat softmax over 20480 *at equal parameters and equal updates per
  legal move* — i.e. the win is statistical sharing, not extra capacity.
- **Falsified by**: a flat model at equal parameter count and equal
  updates-per-legal-move matching the factorised model. That would say
  factorisation is a trainability convenience, not an expressiveness gain —
  a more mundane claim than the literature implies.
- **Why it is open, not settled**: the usual defence of factorisation is
  "20480 slots is too sparse to fill". That is an argument about *update counts*,
  which is an optimisation claim, not an expressiveness claim. Nobody usually
  separates the two, so the claimed advantage is doing double duty.
- **Command**: needs `L3-flat` (same arm as H9). Match parameter count and
  normalise learning rate per legal move, then
  `python -m bench --only L3 L3-flat --games 20`.
- **Runnable today**: needs the level
- **Status**: `[-]` blocked on `L3-flat`

### H17 — Irreversible moves are the load-bearing transitions
- **Evidence status**: `OPEN` — and this is the one I would bet on
- **Claim**: the value function changes at pawn moves and captures; positions
  near one of those carry most of the learnable signal. Therefore **a policy
  trained only on positions within N plies of a pawn move or capture learns
  faster per game than one trained uniformly** over the same corpus.
- **Falsified by**: uniform training reaching the same strength per game as
  transition-focused training. If it does, the no-progress counter is a
  termination heuristic and nothing more.
- **Why it matters if true**: your `no_progress_limit` already encodes this
  structural fact as a *stopping rule*. H17 says the same fact should be a
  *sampling rule* — the benchmark would be privileged to information it is
  currently using only to end games.
- **Command**: `MidstateStore.record_game` then `sample()`, which already
  supports phase stratification. Add a transition-proximity stratifier and
  compare corpus mixes. No new model needed.
- **Runnable today**: **yes** (needs a sampling wrapper, not a new level)
- **Status**: `[ ]`

### H18 — The horizon effect dominates L2's errors at low depth
- **Evidence status**: `CONTESTED` → treat as `OPEN` locally
- **Claim**: at depth 2–3, L2's *mistakes* (move differs from depth-5) are
  dominated by unresolved capture sequences, not positional misjudgement. If
  true, quiescence is where the remaining strength is; if false, more depth is.
- **Falsified by**: sampling L2 vs depth-5 disagreements and finding them mostly
  non-tactical (no capture available at or near the divergence point).
- **Known context, honestly stated**: that quiescence blunts the horizon effect
  is textbook and `search.py` already says so. What is *not* established here is
  the **magnitude at your specific depths** — your own measurement was that
  quiescence was roughly node-neutral, which says nothing about error
  composition. So the local claim is open even though the general principle is
  settled.
- **Command**: use `diagnostics.compare_policies` to enumerate disagreements, then
  classify each by whether a capture is legal within 2 plies of the divergence.
- **Runnable today**: **yes** (classification script; no new model)
- **Status**: `[ ]`

### H19 — Non-transitivity is structural, and `material` is its cause
- **Evidence status**: `OPEN`
- **Claim**: H7 showed cycles exist. H19 says *how*: the cycle runs through
  `material` specifically, because a shallow greedy evaluator exploits an
  informed-but-weak policy's uncertainty in a way a blind policy cannot.
- **Falsified by**: cycles distributed randomly across pairs with no
  `material`/`L3` concentration. H7 survives; **H19 dies**, and the ladder
  metaphor becomes less salvageable, not more.
- **Baseline evidence for the `material`–`L3` edge**: already visible —
  `material vs L3` was the largest residual (observed 0.500, expected 0.259).
  One edge is not a structural pattern; the hypothesis is about whether *all*
  the large residuals involve `material`.
- **Command**: `python -m bench --games 30` and check whether the top-3 residuals
  all involve `material`.
- **Runnable today**: **yes**
- **Status**: `[ ]`

### H20 — Positional knowledge only pays once tactics are handled
- **Evidence status**: `SETTLED` — listed for completeness, do not investigate
- **Claim**: at naive scale, a 1-ply greedy material player sits mid-table
  because tactics dominate before positional understanding pays off.
- **Already supported by your own data**: `material` at 255 Elo beats L1 (165)
  and loses to every L2 depth. This is also the entire history of chess engines.
- **Why the row exists anyway**: it is the load-bearing assumption behind
  *search depth being on the ladder at all*. Worth one line for the reader who
  wonders why depth comes before cleverness.
- **Command**: nothing new; read the existing table.
- **Runnable today**: n/a
- **Status**: `[x]` (settled, evidenced locally)

### H21 — The encoder leaks useful geometry before any learning
- **Evidence status**: `CONTESTED` — the interesting half is open
- **Claim**: an **untrained** model still prefers central pawn pushes to a-file
  pushes, measurably above chance, because `encode.py`'s channel semantics and
  the factorised structure carry a weak prior before a single update.
- **Falsified by**: an untrained policy showing no systematic preference for
  central/forward moves over a large probe set — i.e. `policy_sharpness`
  perplexity ≈ legal-move count and no spatial bias.
- **Half that is settled**: untrained-vs-untrained matches are ~random (your L3
  evidence, 0-6 draws). Do not re-test that.
- **Half that is open**: random *weights* do not imply random *preferences*. The
  factorised softmax with zero logits is uniform *per factor*, not per move —
  documented in `policy_sharpness` — so the geometry of the move set itself
  imposes structure. Whether that structure *points anywhere useful* is the
  question.
- **Command**: `policy_sharpness(L3Policy(seed=n))` over the probe set, then
  correlate per-move probability against a centrality/forwardness score.
  Cheap: no games needed.
- **Runnable today**: **yes**
- **Status**: `[ ]`

### H22 — Scale-free difficulty: rejected, do not add
- **Evidence status**: not a hypothesis
- **Tempting claim**: chess's branching factor (~35) implies some particular
  learning difficulty, so a learner's failure is attributable to search-space
  size.
- **Why it is rejected**: not falsifiable at this scale. Your search levels do
  not learn and your learners do not search, so branching factor, search quality
  and eval quality are perfectly confounded. Any correlation you find would have
  three equally good explanations. **Recorded here so it does not get
  re-proposed.**

### H23 — Special moves are legal but rare, and one of them is unevaluated
- **Evidence status**: `CONTESTED` → **half tested below**
- **Claim (mechanical half)**: every special move (castling, en passant,
  under-promotion) is *admitted* by the move mask and playable by every level.
- **Claim (behavioural half)**: they are **played far less often than real chess
  would warrant**, because nothing in the evaluation rewards or penalises them.
- **Tested 2026-10-02 — mechanical half: SUPPORTED.** `legal_move_mask` admits
  100% of legal moves including `O-O`/`O-O-O`, ep, and all four promotions.
  The mask is generated from `board.legal_moves`, so it cannot be otherwise.
- **Tested 2026-10-02 — behavioural half: PARTLY SUPPORTED, with a concrete
  gap.** Self-play special-move census (ply cap 120-160):

  | level | castling | en passant | promotion |
  |---|---|---|---|
  | L1 blind | 1 | 2 | 136 |
  | L3 perceptron | 1 | 2 | 105 |
  | material 1-ply | 3 (3/6 games) | 0 | 1 |
  | random | 1 (1/6 games) | 0 | 0 |
  | L2-d2 minimax | 6 (3/3 games) | 0 | 0 |

  L2 castles in every game; the blind/untrained policies almost never do. That
  split is the real finding: castling needs a *reason*, and only the searching
  levels have one.

- **The gap, isolated — the evaluator has no special-move term.**
  `value.evaluate()` is `material_value + piece_square_score` and nothing else.
  It has **no castling term, no en-passant term, no mobility term.** Two
  consequences worth stating plainly:
  1. **Castling is valued only incidentally**, via the king PST (king is safer on
     g1 than e1). The *connectivity* of the rooks, the pawn shield, and the
     tempo cost are all invisible. A castling move is therefore chosen only when
     the PST delta happens to be positive — which is why it works at all, and
     why it is underplayed relative to a real engine.
  2. **En passant is invisible to the eval** except as a material capture of a
     pawn. It was played **twice in 80+ self-play games** at L1/L3 and *zero*
     times for material/random/L2 in this sample — consistent with "never
     rewarded beyond the immediate pawn".

- **Concrete, checkable defect found while testing**: the encoder has **no
  castling-rights channel.** `CHANNELS` (22 binary) contains no `castling_*`
  entry; `en_passant` is channel 21 and `promotion_rank_w/b` are 19/20. So a
  learner sees `has_castling_rights(me)` only as a single scalar in
  `board_context_features` (`encode.py:424`), **not as a spatial fact about which
  squares the rights attach to.** A network cannot express "my king can still
  castle kingside" as a board pattern, which is exactly the information needed
  to decide *whether* to castle. This is a plausible partial explanation of the
  behavioural result above.
- **Verified incidentally (no bug)**: the `en_passant` plane survives
  canonicalisation correctly — black-to-move `f6` becomes plane `f3` after rank
  reflection, matching `colour_mirror`, with `encode(pos) == encode(mirror(pos))`
  to **exactly 0.0**. Promotion slots have **no built-in queen bias**: uniform
  promo logits give `[0.25, 0.25, 0.25, 0.25]` over N/B/R/Q.
- **Falsified by**: a learner demonstrating castling/ep in proportion to a
  depth-3 search on a filtered set of positions where castling is clearly best.
- **Command**: filtered position set (rights intact, both wings open, ≥4 plies
  from any capture), then compare each level's castle rate against L2-d3 as
  oracle. Also: `python -m bench --games 20` and count `O-O` in recorded SAN.
- **Runnable today**: **yes**
- **Status**: `[~]` mechanical half `[x]` supported; behavioural half tested, gap
  identified, no fix attempted

---

## Result template

Copy this under a hypothesis when you test it:

```markdown
**Result (YYYY-MM-DD)**: <verdict: supported / falsified / not established>
- Command: `<exact command>`
- Raw output: `bench/<file>.json`
- Key numbers: <the delta and the SE, always both>
- Games behind it: <decided games per entrant — a rating with <10 is provisional>
- Interpretation: <what this changes about what we believed>
- Follow-up: <the next hypothesis this result creates, if any>
```

## Anti-patterns this register exists to prevent

- **Quoting a rating without its standard error.** The ablation table looked
  like an ordering and was noise.
- **Treating convergence as correctness.** Four consecutive bugs in the rating
  fit all converged to plausible answers. Assert against an independent oracle.
- **Crediting architecture for a parameter-count win.** Fix the budget first
  (H9), then let topology be the free variable.
- **Running a blocked hypothesis.** H11 cannot be answered by re-running the
  ablation on an untrained model. H8-H10 cannot be answered before L3.5 exists.
- **Deleting a falsified hypothesis.** The negatives are the valuable entries.
- **Investigating a `SETTLED` row.** H20 and the settled half of H21 are known;
  re-deriving them costs a week and teaches nothing. The register labels them so
  a future reader does not.
- **Confusing a domain claim with a code claim.** Group F failures are findings;
  Group A failures are bugs. Do not file one as the other.

## Adding a hypothesis

Before adding a row, answer these three, in the entry:

1. **What result would falsify it?** If you cannot say, it is not a hypothesis.
2. **Is it already settled?** If the field knows, label it `SETTLED` and move on.
3. **Can existing code answer it?** Mark runnable-today, or name the blocker.
   A hypothesis blocked on a level that does not exist is still worth recording —
   writing the kill condition *before* building the thing is what stops the
   thing being built to succeed (this is why H8–H10 exist while L3.5 does not).
