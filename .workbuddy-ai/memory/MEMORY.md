# Project: chess-rl-bench

A benchmark harness for naive RL approaches to chess. Workspace root:
`C:/Users/Trinity/Programmes/Chess-player`.

## Purpose
Compare a ladder of increasingly informed learners on the same task, all behind
one interface, so results are comparable. The point is the *benchmark*, not a
strong engine.

## Non-negotiable conventions

### Rules layer
`python-chess` (1.11+) is the authority for move generation, legality, check,
terminal conditions. Do not reimplement chess rules. Everything else is numpy.

Do **not** use the PyPI package `chessboard`. It is a puzzle-placement solver
with no move generation, no turns and no pawns. This was established by reading
its source; the original project spec mistakenly referenced it.

### Coordinate and plane conventions
- Square index = python-chess numbering. `0 = a1`, `63 = h8`.
- Plane arrays are `(channels, 8, 8)` with **row = rank, row 0 = rank 1**.
  No flipping anywhere except at display time.
- Channel order is **append-only**. L3 weight matrices and L4 checkpoints index
  channels by position, so reordering invalidates saved models.
  `COLOUR_PAIRED_CHANNELS` in `encode.py` lists channels carrying colour
  identity; any new colour-specific channel must be added there.

### Canonicalisation
`chessrl.bitboard.canonical` is a **rank reflection plus colour-block swap**.
It is NOT a 180 rotation: a 180 rotation maps a8 -> h1 and is not a chess
symmetry (the a-file and h-file are not interchangeable). Every learner sees
positions in canonical form so one parameter set serves both colours.
`apply_canonical` maps move-space quantities between frames; it is its own
inverse.

### Action space
`ACTION_SPACE = 20480`, index `(from_square * 64 + to_square) * 5 + promo_slot`
where slot 0 = not-a-promotion, 1-4 = N/B/R/Q. Masks are `(64, 64, 5)` bool.
Always address moves through `masks.move_to_index` / `index_to_move` so the
flat and factorised views stay consistent.

### Environment
`.venv` in the repo root. `numpy`, `chess`, `pytest`. torch is an **optional
extra** (L4/L5 only) and must not be imported by L0-L3 modules.
Run tests: `./.venv/Scripts/python.exe -m pytest`.

## Architecture
Every level implements `chessrl.policy`-style protocols:
- `MovePolicy.select(board) -> Move` for search-based levels.
- `DistributionPolicy.move_distribution(board) -> ndarray` for learners.

`encode.py` is shared by all levels. Resist forking it per level: the benchmark
is only meaningful if every learner sees the same features, so they can be
ablated.

## The level ladder
- **L0** foundation: bitboard, masks, cache, encode, value, game loop. DONE.
- **L1** blind factored softmax over `(from, to, promo)`, no board input. DONE.
- **L1.5** diminishing-credit updates, per-combination prob counters, prune
  below 0.01. Counters must be stripped before saving weights. DONE.
- **L2** alpha-beta minimax, optimised; then a *non-learning* wrapper that
  half-informs L1's probabilities. Must have no trainable parameters. DONE.
  Lives in `chessrl/search.py` (`MinimaxEngine`, `MinimaxPolicy`,
  `InformedMinimaxPolicy`).
- **L3** perceptron from encoded boardstate, int-friendly, midstate incremental
  updates, minimax-depth-5 top-5 feedback.
- **L4** the same model in torch, behind the same protocol. Needs a parity test
  against L3.
- **L5** alpha-beta whose ordering and leaf eval come from L4.
- **L6** ensemble of all of the above, weighted by historical win-prediction
  accuracy, plus the tournament bench.

## L2 search: what may and may not be claimed

`MinimaxEngine` is negamax + alpha-beta with MVV-LVA ordering, a zobrist
transposition table, iterative deepening and quiescence. Measured effect of each
at depth 4 from the opening (nodes): full 2049, no-TT 3254 (TT saves 37%),
no-iterative-deepening 2665, quiescence roughly neutral in nodes but it exists
to blunt the horizon effect, not to save time.

**The ordering invariant, stated precisely** (an earlier overclaim was corrected
here): ordering cannot change the *value* a correct-window alpha-beta returns,
but it *does* choose among moves that share that value, because the first move
to reach alpha is kept. Measured: in `rnbqkbnr/pppp1ppp/8/4pP2/8/8/PPPPP1PP/RNBQKBNR
b KQkq f6 0 3`, Nf6 and Nc6 both evaluate to exactly 90 at depth 3; the plain
engine returns Nf6 by board order, the wrapper returns Nc6 because L1's blind
table weights the b8 origin slightly higher (0.0385 vs 0.0256) — a tie resolved
on no chess grounds, which is the *only* channel a prior has. Tests assert the
value invariant and that any disagreement is a genuine tie, not that the move
is always identical.

`InformedMinimaxPolicy` holds no parameters and exposes no `train`/`state_dict`.
The trainer accepts it via `opponent=`.

### L2 measured strength
- Beats L1 blind, random and greedy-material 100% (18/18 across three matches),
  always by decisive results, no unfinished games.
- Deeper beats shallower: d2 beats d1, d3 beats d2 (never loses).
- Self-play finishes 50-75% of games; contact rate 21.4/100 plies, between
  blind play (15.8) and greedy-material (33.5).
- Finds Qxf7#, Qh4# and back-rank mates at depth 2.

## Testing standard
Every level is gated on: plays a full legal game to a real result, is
seed-reproducible, and loses to the level above it in a head-to-head match.
`tests/conftest.py::colour_mirror` is the load-bearing helper for symmetry
tests -- keep it correct, including castling rights and the en-passant target.

## Evidence standard (user, 2026-10-05)
- The goal is to prove or disprove hypotheses; negative results and justified
  changes of direction are useful outcomes.
- Rule out implementation, evaluator, data, metric, and protocol defects before
  interpreting a result as a domain finding.
- Smoke-test new code first, then run the smallest pre-specified experiment
  with enough independent observations and uncertainty reporting to decide.
  Label exploratory pilots as non-conclusive.

## Known traps
- Piece-square tables are written White's-perspective and every lookup must
  rank-mirror for Black, **including the king**, which is in a separate branch.
  Forgetting it made the start position evaluate to +50.

### Special moves — current state (verified 2026-10-05)

- `legal_move_mask` comes from `board.legal_moves`, so castling, en passant, and
  all promotions are legal in the action mask.
- `encode.CHANNELS` has 24 binary planes; `castle_w`/`castle_b` are appended at
  22/23 and mark whether each colour has any castling right (not which wing).
- `value.evaluate()` now includes `king_shelter_score`. The 2026-10-02
  material+PST-only description and census are pre-fix history, not current code.
- The trained-L3 20-game census had zero castling, but every game ended by
  fivefold repetition; it does not diagnose the feature. Test castling on a
  filtered, oracle-labelled position set instead of inferring from self-play.
- En-passant canonicalisation and equal promotion-slot priors were checked; keep
  unfinished games out of ratings. `MidstateStore.sample` is without replacement
  and caps at corpus size.

## Bug-fix evidence workflow

- For numeric learning bugs, test the derivative of the actual forward quantity
  with deterministic finite differences; first show the regression fails on the
  old implementation, then make the smallest correction and rerun the focused
  test plus its full module.
- Treat smoke runs as wiring checks only. Assert non-empty train/held-out splits
  and use dtype-appropriate numeric tolerances. Do not infer learning or strength
  from tiny samples; use a pre-specified, adequately powered held-out experiment
  with uncertainty reporting before making hypothesis claims.
- Human-approved plans and prose guards establish scope, not mechanical
  correctness. Independent LLM review is supplementary; report unavailable
  review honestly and rely on executable evidence for correctness claims.
- For learning diagnostics, cache identical training/held-out labels and use them
  across initialization seeds; checkpoint by positions seen and compute the
  primary paired effect at the independent-seed level. Intervals from shared,
  frozen corpora are conditional on those corpora; add dataset/position sampling
  uncertainty before generalizing beyond them.
- Predeclare one primary endpoint and its meaningful effect, but report
  complementary set-based and weight-sensitive teacher metrics plus policy
  entropy. A gain in top-k set mass alongside worse weighted cross-entropy or
  collapsing entropy is mixed evidence, not a go signal for expensive scaling.
  Treat pilot plug-in sample sizes as planning aids, never as confirmation.
