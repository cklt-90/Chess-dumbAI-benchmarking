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

## Known traps
- Piece-square tables are written White's-perspective and every lookup must
  rank-mirror for Black, **including the king**, which is in a separate branch.
  Forgetting it made the start position evaluate to +50.

### Special moves: legal, but underplayed (measured 2026-10-02)

- Masks admit **everything** (castling, en passant, all 4 promotions) because
  `legal_move_mask` is built from `board.legal_moves`. There is no legality gap.
- `value.evaluate()` is **material + PST only** — no castling term, no en-passant
  term, no mobility term. Castling is rewarded only incidentally via the king PST,
  so the rooks' connectivity, the pawn shield and the tempo cost are invisible.
- `encode.py` CHANNELS (22 binary) has **no castling-rights channel**. En passant
  is channel 21; promotion ranks 19/20. `has_castling_rights` appears only as one
  scalar in `board_context_features`, so a learner cannot express "I may still
  castle kingside" as a board pattern. Likely a partial cause of underplay.
- Measured self-play census: L2-d2 castles **3/3 games**; material 3/6; random
  1/6; L1 and L3 **1 each**. En passant: 2 total across all L1/L3 games, 0 for
  material/random/L2. So searching levels castle, blind/untrained ones do not —
  castling needs a *reason*, and only search has one.
- Verified **not** a bug: the `en_passant` plane canonicalises correctly
  (black f6 -> plane f3 after rank reflection; `encode(pos) ==
  encode(colour_mirror(pos))` exactly 0.0). Promotion slots have **no queen
  bias** — uniform logits give 0.25 each over N/B/R/Q.
- Fixing this means appending channels (append-only rule) — a castling-rights
  plane per colour, and if castling is to be *preferred*, an eval term, since
  PST alone will not do it.
- Unfinished games (ply cap, no-progress cap) must not be scored as draws; that
  teaches a naive policy that shuffling equals winning. `GameResult.score_for`
  returns 0 for them and `is_finished` is False.
- `MidstateStore.sample` samples without replacement and caps at the corpus
  size. Duplicates would silently reweight the phase mix.
