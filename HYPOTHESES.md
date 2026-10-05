# HYPOTHESES.md

A register of testable claims about `chess-rl-bench`. Each entry states the  
hypothesis, **the measurement that would falsify it**, the command to run, and  
where to record the result.

The purpose is to stop "which level is best" from being the only question asked.  
A ladder produces a leaderboard; a set of falsifiable claims produces knowledge.  
Every hypothesis here is written so that **a negative is a result**, not a  
failure — several are expected to fail, and those are the informative ones.

## How to use this

> **Verification note.** This register records *guesses* (what could happen) and,
> where available, *literature context* (consolidated in the *Literature review
> linkage (2026-10-04)* appendix at the end of this file, sourced from
> `hypothesis-lit-review/report.md`). The actual **prove / not-prove** of each
> hypothesis against experiments is tracked in a *separate document* (to be
> created) — do not record conclusive verdicts here until that document exists.

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
and learning itself** (is a common belief true?) · G the training-signal axis  
(self-play vs round-robin vs master-graded, with a random-walk control)

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
  7609 / 787 / 1573 (100% / 10.3% / 20.7%). The original ±10% demand was  
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
  castling-rights channel.** `CHANNELS` (24 binary) contains no `castling_*`
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

## Group G — the training-signal axis (which signal makes a learner?)

> Group C varied the *function* (representation) at a fixed signal. This group
> varies the *signal* (what the learner is trained on) at a fixed function. The
> learner is held constant — same architecture, same parameter budget, **and the
> same data budget (games / positions seen)** — so a difference is attributable
> to the signal, not the learner and not the volume. That is the H9 rule applied
> to data rather than parameters, and it is why the random-walk arm (H26) is the
> control: without it, "trained" is indistinguishable from "touched more games".
>
> **Do-ability, assessed against the code (2026-10-03).** Every paradigm maps to
> an existing hook; only one has a real blocker:
>
> | paradigm | signal type | hook that already exists | blocker |
> |---|---|---|---|
> | self-play (incumbent) | outcome RL vs frozen self-copy | `train_naive.train()` frozen-copy opponent | none — runnable today |
> | round-robin | outcome RL vs a rotating opponent pool | `train_naive.train(opponent=...)` override | none — needs a thin opponent-cycle scheduler |
> | graded by master | supervised imitation of a stronger player | `L3Trainer.train_on_search_feedback` (currently fed depth-5 search) | **a true master is not vendored** — depth-5 is a built-in *weak* master, runnable today; a real engine / master games is the blocker |
> | random walks (control) | outcome RL on noise games | `RandomPolicy` generates the corpus | none — runnable today |
>
> What is genuinely missing is a **per-paradigm driver** (train N games with
> signal S, then bench against fixed reference opponents, report rating ± SE) —
> the analogue of `bench/ensemble_run.py`. The hooks exist; the driver does not.
> That is a small build, not a research blocker.
>
> **None of these is settled.** The wider field expects opponent diversity and
> supervision to help *eventually*; whether they help *at this budget and this
> architecture* is precisely the open question, and it is why the control matters
> more than the prediction. Related: H9 (the control is load-bearing), H20
> (positional knowledge pays only after tactics), H23 (the master arm is the one
> that would teach castling — L3 already wires `can_castle` into its origin
> scoring, so it has the *features* to absorb supervision; the open question is
> whether the *signal* is what it lacks).

### H24 — Opponent diversity: round-robin beats self-play at equal data budget
- **Claim**: at equal games seen and identical architecture, a learner trained
  against a rotating pool of opponents (random / material / L2 from the roster)
  beats the same learner trained by pure self-play, by more than ~1 SE on a
  common reference set.
- **Falsified by**: self-play matching or beating round-robin within 1 SE —
  opponent diversity buys nothing at this scale.
- **Why it might hold**: the frozen self-copy converges to the learner's own
  style, so self-play rarely produces positions the learner would not itself
  reach. A diverse pool forces responses to unfamiliar play and covers more of
  the position space.
- **Why it might not (the honest null)**: if the binding constraint is the
  *eval/reward* signal (H23's gap — the learner cannot tell a good move from a
  bad one), then better opponents do not help, because the learner cannot read
  the lesson in the games either way.
- **Command**: needs the per-paradigm driver. `train_naive.train(opponent=...)`
  already accepts the override; a round-robin arm cycles a pool through it.
- **Read against**: H26 — if random-walk training also helps, the diversity
  effect is confounded.
- **Status**: `[~]` hooks runnable today; driver thin

### H25 — Master-graded beats self-taught at equal positions seen
- **Claim**: at equal positions seen, a learner trained by imitating a master's
  moves beats the same learner trained by self-play outcome-RL, by more than ~1 SE.
- **Falsified by**: self-play matching master-graded within 1 SE — naive RL
  suffices and supervision buys nothing. A draw here is a *strong* "naive is
  enough" result, not a non-result.
- **This is a *decision* hypothesis.** It prices the naive premise the whole
  ladder is built on. A clear master win says the self-taught framing understates
  the architecture's ceiling; a draw vindicates naive RL at this scale.
- **Weak master runnable today**: `train_on_search_feedback` already imitates
  depth-5 search, which is a built-in weak master — a first run needs no new
  data. The **strong version is blocked** on a vendored master (engine or master
  games) plus a shim turning master moves into the `(move, weight)` feedback
  format the hook expects.
- **Castling cross-ref (H23)**: this is the paradigm that would teach castling —
  a master that castles supplies the reward self-play lacks. A master-trained L3
  that *still* does not castle indicts the features (no castling plane); one that
  *does* indicts the self-play reward. The arm doubles as the feature-vs-signal
  diagnostic left open in H23.
- **Status**: `[-]` weak-master (depth-5) runnable today; true master blocked on data


### H26 — The random-walk control: noise carries no exploitable signal
- **Claim**: a learner trained on games between two `RandomPolicy` players does
  **not** beat an untrained learner, by more than ~1 SE.
- **Falsified by**: random-walk-trained beating untrained by more than 1 SE —
  which would mean the learner extracts structure even from noise, and *every*
  comparison above is confounded by "more games = better" rather than "better
  signal".
- **Why it is the control** (the L3-flat of this axis, cf. H9): without it, a
  self-play or master win is indistinguishable from "any training at all helps".
  This is the load-bearing arm — run it *first*.
- **A subtlety worth stating**: random-vs-random outcomes are not pure noise.
  Random play slightly favours captures and checks, so the learner may pick up a
  faint material-greed signal. A *small* random-walk gain is therefore expected
  and is itself informative; the claim is only that it cannot reach the trained
  levels. If it *does*, the benchmark's training loop is the finding, not any
  paradigm.
- **Command**: `RandomPolicy` exists; generate random-vs-random games, train on
  them, bench against an untrained clone. Runnable today (thin driver).
- **Status**: `[~]` runnable today

---

### H27 — Crow Search Algorithm (CSA) randomness as a training-signal dial
- **Claim:** a training-game generator whose move sampling is parameterised by a
  Crow Search Algorithm-style exploration knob (awareness probability
  `AP ∈ [0,1]`, flight length `fl`) lets a learner explore *new* moves off its
  self-play attractor. Intermediate `AP` reaches fixed strength per game *faster*
  than either endpoint at equal data budget: `AP=0` ≈ pure self-play (incumbent),
  `AP=1` ≈ pure random-walk (the H26 control).
- **Falsified by:** pure self-play (`AP=0`) or pure random-walk (`AP=1`) matching
  the interpolated arm within ~1 SE on a common reference set; or a monotonic
  improvement as `AP` goes 0→1 (no intermediate optimum).
- **Why it matters:** generalises H26 from a single baseline into a *continuous*
  randomness spectrum, and directly tests the intuition that "differing amounts
  of randomness allow exploring new moves." Other randomness-injection mechanisms
  (temperature sampling, random-move injection, random-legal-position restarts
  seeded from saved game positions) are alternative dials for the same axis.
- **Command:** needs a `noise`/`AP` parameter on `RandomPolicy`/opponent move
  sampling + the per-paradigm driver (same blocker as H24/H25/H26). Sweep
  `AP ∈ {0, 0.25, 0.5, 0.75, 1.0}`, train N games each at matched budget, bench
  vs fixed references.
- **Runnable today:** partially — `RandomPolicy` exists; needs the `AP` parameter
  + the driver.
- **Status:** `[ ]`

## Group H — the feedback-style axis (what target does the learner train on?)

> Six candidate feedback styles, all at the fixed function / fixed data budget of
> Group G, so any difference is attributable to the *signal form*, not the learner
> or the volume. The random-start seeding (variant) is an orthogonal corpus-init
> mechanism layerable on any style. Runs need the per-paradigm driver (Group G
> blocker); see H24/H25/H26.

### H28 — Feedback style changes learning rate at equal data budget
- **Claim:** at matched games/positions seen and identical architecture, the *form*
  of the training signal changes how fast and how far a learner improves, and the
  ranking among the six styles is non-trivial (no style dominates).
- **Falsified by:** every style landing within ~1 SE of every other on a common
  reference set — i.e. signal form is irrelevant at this scale (a strong
  "naive-RL-is-enough" result).
- **Styles (arms):**
  1. **Outcome** — win/lose terminal signal only (the incumbent self-play target).
  2. **Discounted outcome** — win/lose with a *decaying* credit further back in
     the game (temporal credit assignment / eligibility traces).
  3. **Master-graded value** — position strength after 0..n *best-play* moves,
     labelled by a master (depth-5 search is the built-in weak master today).
  4. **Self-graded value** — position strength after 0..n *generated* moves, graded
     by the learner's own (bootstrapped) value head.
  5. **Blended / DAgger-style** — train on the learner's own generated positions
     but *graded by the master* (dataset aggregation, Ross et al. 2011),
     iteratively; generalises 3 & 4 and fixes covariate shift.
  6. **Exploration-bonus / intrinsic-motivation** — reward visiting state–action
     pairs the current policy rates as novel (count- or ensemble-disagreement-
     based); ties directly back to H27/CSA's exploration dial.
- **Corpus-init variant:** recurring training from *random legal positions* saved
  from previous games (diverse restarts), layerable on any arm above.
- **Command:** needs the per-paradigm driver extended so each style is a selectable
  `feedback=` mode; sweep styles at matched budget, bench vs fixed references.
- **Runnable today:** partially — 1–2 need only a credit rule; 3–6 need the
  value-head + driver; the variant needs a position-store.
- **Status:** `[ ]`

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

---

## Literature review linkage (2026-10-04)

Consolidated from `hypothesis-lit-review/report.md` (15 validated items). Each
entry below maps a register hypothesis to the literature. The harness sits at
**toy/numpy scale** (~10³ params, hundreds of self-play games); the recurring
finding is that most literature verdicts are *silent or inverted* at this scale,
so a local test is often **novel by construction**, not a replication. The
register above stays the record of *guesses*; the prove/not-prove lives in a
future document.

- **H17 transitions** — *partially supports.* Mechanism (non-uniform sampling
  accelerates value learning: PER, Ape-X, prioritized sweeping) established
  toy→Atari; the *irreversibility* criterion is unpublished at every scale
  (AZ/Lc0 use uniform replay). Local regime = white space.
  Details + suggested test: `hypothesis-lit-review/report.md#h17-transitions`
- **H21 arch-priors** — *supports premise, silent on payoff.* Structure carries a
  prior before training (DIP, spectral bias, WANN) established at small-net;
  *unpublished* at numpy scale; swamped by training at AZ/Lc0.
  `…/report.md#h21-arch-priors`
- **H23 special-moves** — *supports.* "Legal but invisible" is a known failure
  pattern; every serious representation encodes castling explicitly (AZ: 4
  constant planes). Holds across all regimes; toy/numpy has neither plane nor
  term — exactly the local gap. `…/report.md#h23-special-moves`
- **H18 horizon** — *partially supports.* Berliner's horizon effect + quiescence
  are settled at engine scale; magnitude at depth 2–3 with a material+PST eval is
  *untested* (AZ/Lc0 dissolve the question). `…/report.md#h18-horizon`
- **H19 non-transitivity** — *partially supports.* Cycles confirmed at >1B-game
  human scale and AI-population scale; the *greedy-material-hub* structural claim
  is novel/untested anywhere. `…/report.md#h19-nontransitivity`
- **H20 tactics-first** — *supports*, but **regime-bound**: Ruoss 2024 inverts it
  at 270M params (~5 orders of magnitude above local scale). Local ladder premise
  is safe. `…/report.md#h20-tactics-first`
- **H16 factored-action** — *partially supports.* Factorisation is universal in
  strong nets, but no published work isolates statistical sharing from capacity at
  *matched* params/updates (H16's condition). No evidence at toy/numpy scale.
  `…/report.md#h16-factored-action`
- **H08 weight-tying** — *supports* (structural, not controlled-untied baselines);
  established small-net & AZ/Lc0; **untested at toy/numpy (~10³ params)**. NNUE's
  per-king-slice untying is the boundary. `…/report.md#h08-weight-tying`
- **H09 budget-vs-topology** — *mixed; verdict flips across regimes.* Budget-
  dominates holds ~10⁶–10¹⁰+ params; topology-matters holds in matched-parameter
  game-net comparisons. **No measurement at ~10³ params.**
  `…/report.md#h09-budget-vs-topology`
- **H10 relational-limits** — *partially supports.* Per-square functions cannot
  express irreducibly relational facts (supports small-net & engine scale); silent
  at toy/numpy. Local caveat: missing-feature, not proven tying, problem.
  `…/report.md#h10-relational-limits`
- **H11 learned-leaf-eval** — *supports direction*; every published success used
  search-derived/supervised labels (KnightCap, NNUE). **Silent at numpy scale**;
  cheapest success used thousands of FICS games / ~175M positions. Cheapest local
  arm: supervised value head via `train_on_search_feedback`.
  `…/report.md#h11-learned-leaf-eval`
- **NNUE** — *supports* (engine-scale; displaced handcrafted eval since ~2020, via
  *supervised* training). Transferable to local = king-relative features +
  supervised-from-search signal; accumulator/quantization are a regime gap.
  `…/report.md#nnue`
- **H12 move-ordering** — *supports.* Value-invariance is a **theorem** (Knuth &
  Moore 1975), all regimes; node-saving magnitude is canonical at engine/AZ scale
  but *unmeasured locally*. `…/report.md#h12-move-ordering`
- **H13 / H3 rating-methodology** — *supports; regime-free.* SE formula
  `347/√(n(1−d))` predicts ~82 Elo vs observed 79–93, so 6 games/pair cannot
  establish sub-100-Elo gaps. H3 (unfinished-as-draws) untested locally.
  `…/report.md#h13-rating-methodology`
- **searchless-gm (Ruoss 2024)** — *supports at 10⁸-param/10⁷-game regime;
  explicitly inverted at ≤10M params.* At the local toy/numpy regime the claim
  *fails* — that is precisely its local role. `…/report.md#searchless-gm`
