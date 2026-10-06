# ONBOARDING.md

Context to be productive in `chess-rl-bench` in one read.

## What this is

A **benchmark harness**, not an engine. It implements a ladder of increasingly
informed learners for chess, all behind one `Policy` protocol, so their strength
is comparable on one harness with fixed seeds and a fixed opening book. The
deliverable is *measurement*, not a strong player.

Repo root: `C:/Users/Trinity/Programmes/Chess-player`.

## Setup

```bash
./.venv/Scripts/python.exe -m pytest        # 460 tests, ~3m30s (torch tests dominate)
./.venv/Scripts/python.exe -m pytest tests/test_bitboard.py   # one module
```

- Everything is `src/chessrl/`, layout is `src/`.
- Deps: `numpy>=2.0`, `chess>=1.11`, `pytest`. **torch is an optional extra**
  (L4/L5 only). L0-L3 must import without torch — never add a top-level torch
  import to a lower-level module.
- `pip install -r requirements.txt` for the numpy-only benchmark.

## The five rules you must not break

1. **python-chess is the rules authority.** Move generation, legality, check,
   terminal conditions. Do not reimplement chess rules. Everything else is numpy.
2. **Coordinates are python-chess.** `0 = a1`, `63 = h8`, `rank = sq >> 3`,
   `file = sq & 7`. Plane arrays are `(channels, 8, 8)` with **row 0 = rank 1**.
   No flipping anywhere except at display time.
3. **Channel order is append-only.** `encode.py::CHANNELS` is indexed by position
   in L3 weight matrices and L4 checkpoints. Appending is safe; inserting or
   reordering silently invalidates saved models. If a new channel carries colour
   identity, add it to `COLOUR_PAIRED_CHANNELS`.
4. **Canonicalisation is a rank reflection + colour-block swap, NOT a 180
   rotation.** `chessrl.bitboard.canonical`. A 180 maps a8 -> h1, which is not a
   chess symmetry: the a-file and h-file are not interchangeable. Every learner
   sees canonical positions so one parameter set serves both colours.
   `apply_canonical` converts move-space quantities between frames and is its
   own inverse — use it rather than hand-rolling a flip.
5. **Address moves through `masks.move_to_index` / `index_to_move`.** Flat index
   is `(from * 64 + to) * 5 + promo_slot` over `ACTION_SPACE = 20480`; slot 0 =
   not-a-promotion, 1-4 = N/B/R/Q. Masks are `(64, 64, 5)` bool. This keeps the
   flat and factorised views consistent.

## Module map

| File | Role |
|---|---|
| `__init__.py` | `WHITE`, `BLACK`, `PLANE_ORDER` (12 piece planes; `plane ^ 6` flips colour) |
| `bitboard.py` | L0. `board_to_planes`, `canonical`, `apply_canonical`, `attack_map`, `defended_map`, packing |
| `masks.py` | L0. `legal_move_mask` `(64,64,5)`, flat-index converters, state masks |
| `encode.py` | L0. **Shared by every level.** `encode()` -> `(28, 8, 8)` float32: 24 binary + 4 value planes |
| `value.py` | L0. Material, PSTs, `evaluate`, `capture_value`, `recapture_risk`, `phase_weights` |
| `cache.py` | L0. `PositionCache` (packed-plane keys), `MidstateStore` (stratified phase corpus) |
| `game.py` | L0. `play_game`, `play_match`, `GameResult`; `RandomPolicy`, `MaterialPolicy` |
| `policy.py` | L1. `Scorer` protocol, `masked_softmax`, `FactoredSoftmaxPolicy`, `BlindScorer` |
| `train_naive.py` | L1.5. `FactoredSoftmaxTrainer`, `credit_curve`, `ProbCounter`, pruning |
| `search.py` | L2. `MinimaxEngine` (negamax + alpha-beta + TT + iterative deepening + quiescence), `MinimaxPolicy`, `InformedMinimaxPolicy` |
| `perceptron.py` | L3. `PerceptronScorer`, `L3Policy`, `L3Trainer` (incl. `train_on_targets` / `train_on_master`) |
| `master.py` | L3 supervision. `EngineMaster` (UCI, e.g. Stockfish), `PgnMaster` (master-games corpus), weight converters. H25's true-master source; inert until a binary/PGN is supplied |
| `squarelocal.py` | L3.5. `SquareLocalScorer` (weight-tied per-square heads), `L35Policy`, `L35Trainer`, plus the `L3-flat` control |
| `torch_model.py` | L4. `TorchScorer` (+ `TorchScorerAdapter`), `TorchPolicy`, `TorchTrainer` — parity-tested against L3 |
| `guided.py` | L5. `GuidedModel` (shared trunk + policy head + value head), `GuidedEngine`, `GuidedPolicy`, `GuidedTrainer` |
| `ensemble.py` | L6. `MemberSpec`/`MemberKind` (blendable vs vote-only), `OutcomeLedger`, `accuracy_weights`, `EnsemblePolicy`, `score_member_opinions` |
| `diagnostics.py` | Shared. `diagnose_games`, `policy_sharpness`, `action_coverage`, `compare_policies` |
| `bench/levels.py` | The roster: every level as a named `LevelSpec`, opening book, ablation roster |
| `bench/runner.py` / `rating.py` / `report.py` | Match loop, Bradley-Terry fit, ASCII tables. `python -m bench` |
| `bench/ensemble_run.py` | Fits L6's weights from calibration games. `python bench/ensemble_run.py` |

Root-level docs: `HYPOTHESES.md` (the investigation register — read this before
running any comparison), `bench/README.md` (how to read a rating table, and what
the benchmark cannot establish).

## Interfaces

Two protocols, both duck-typed by the game loop:

- `MovePolicy.select(board) -> chess.Move` — search levels (L2, L5).
- `DistributionPolicy` adds `move_distribution(board) -> ndarray` — learners.

`play_game` accepts either. Note that L6 is a `DistributionPolicy` that *holds*
both kinds: it blends the members that expose a distribution and votes the ones
that do not. That is why `MemberKind` is derived from the member rather than
declared by the caller.

`play_match(a, b, games=..., opening_fens=...)` alternates colours and plays from
a fixed book, which is how a low-variance comparison is obtained from few games.

Learners divide into a `Scorer` (three logit heads: `from`, `to | from`,
`promo | from,to`) plus the shared `FactoredSoftmaxPolicy`. **L3 and L4 exist to
be the same function in two backends** — a rewrite that isn't bit-identical to
its predecessor is only "self-consistent". Swapping the model means writing a
`Scorer`, not a new policy.

`PerceptronScorer` is hand-rolled and passes the exact same three head shapes as
`TorchScorer`. The value head (L5) is shared across policy and eval heads via a
trunk; the two seams (`use_model_prior` = ordering, `use_model_eval` = leaf
value) are independently switchable, and that ablation is a result in its own
right.

## Level ladder and honest status

| Level | State | Notes |
|---|---|---|
| L0 foundation | done | bitboard, masks, cache, encode, value, game |
| L1 blind factored softmax | done | no board input; probability floor, not strength |
| L1.5 trainer | done | diminishing credit, per-combination counters, prune < 0.01; strip counters before saving weights |
| L2 alpha-beta | done | `MinimaxEngine`; `InformedMinimaxPolicy` holds no parameters |
| L3 perceptron | done | 7609 params, ~1.8 ms/position, symmetry exact to 1.5e-08 |
| L3.5 weight-tied per-square | done | 787 params (10.3% of L3); `L3-flat` control at 1573. **Not budget-matched** — see `bench/README.md` |
| L4 torch | done | parity with L3 is exact (diff 0.0); this is the reason it exists |
| L5 guided alpha-beta | done | 95083 params; untrained, does not beat L2 |
| L6 ensemble + tournament bench | **implemented; negative under default rule, neutral under relative rule** | blends L1/L3, votes L2-d2; weights fitted by `bench/ensemble_run.py`. Default (absolute) weights make it **worse than L2-d2 alone** — the blind pair holds 0.514 and outvotes the one member that sees material. `EnsembleConfig.relative=True` (`--relative`) fixes this structurally: below-mean members drop to ~0 and the ensemble equals its best member. See `bench/README.md` |

**Read this before writing a strength claim.** No learner beats the level below
it until it is *trained*. Untrained L3 draws blind L1 (six draws by fivefold
repetition); untrained L5 does not beat L2. The tests assert the *mechanism*
(is a full legal game played, is it seed-reproducible, does training change the
preference), not that a random-weight model wins. A test asserting "L3 beats L1"
failed 0-6 and the test was wrong, not the code.

L6 is the sharpest case of this. It is fully implemented, its weights are
reproducible from a stored ledger, and under the default (absolute) rule it still
loses to its own best member — because the metric it weights on measures
*agreement with the winner's move*, which a policy with no material term scores
well on by accident. The relative rule (`relative=True`) removes that deficit by
refusing to let the weak majority dilute the strong member, so the ensemble then
equals its best member rather than falling below it. "Implemented" and "an
improvement" are different claims and L6 is only the first.

## Testing standard

Each level is gated on: plays a full legal game to a real result, is
seed-reproducible, and loses to the level above it in a head-to-head match.

`tests/conftest.py::colour_mirror` is the load-bearing helper: it swaps colours
and reflects ranks while carrying castling rights and the en-passant target. Every
symmetry bug in this repo was caught by comparing `encode(pos)` against
`encode(colour_mirror(pos))`. Keep it correct — an early version dropped castling
rights and produced *false* failures that cost real time to diagnose.

## Traps that already bit us

- **Verify a "clearly best move" position with the engine before asserting the
  engine finds it.** `3qk3/8/8/8/8/8/8/3QK3` looks like a free queen; `Qxd8 Kxd8`
  is a trade and the true value is 0. The engine was right, the test was wrong.
- **Value planes are off by default** in `encode()`. They cost ~40x the binary
  planes. On for L2 and inference; off for corpus building.
- **Unfinished games are not draws.** The ply cap and no-progress cap are the
  cheap replacement for `claim_draw=True` (which scans the whole move stack and
  cost 39% of training time). `GameResult.score_for` returns 0 for unfinished
  games and `is_finished` is False. Scoring shuffling as a draw is how a naive
  policy is accidentally taught that stalling is winning.
- **The king PST is in a separate branch** from the other piece-square tables.
  Forgetting to rank-mirror it for Black made the start position evaluate to
  +50 instead of 0, biasing every search toward White.
- **`MidstateStore.sample` samples without replacement**, capped at corpus size.
  Duplicates would silently reweight the phase mix.
- **Frame discipline in L5.** Read and write board quantities in *one* frame.
  The prior was all zeros for Black because masks were built from raw squares
  while head outputs are canonical. `canon` is the identity for White, so it is
  invisible until you play the other colour.
- **A width sweep removes the coincidence a shape bug hides behind.** L5's head
  shapes were only legal when `trunk == 64`; `test_heads_work_at_trunk_widths_
  other_than_64` is the test that catches it.
- **`GuidedEngine.single_threaded()` is not in `__init__`.** torch's OpenMP fork
  cost exceeds the arithmetic at batch size 1 (depth-3 middlegame 55.3s -> 5.9s),
  but the thread count is process-global and a trainer in the same process does
  benefit from parallelism.
- **A too-small learning rate on a quantised model silently disables it.** L3.5's
  tied heads receive credit from every square, so the rate must be divided by the
  fan-out — but at `lr=0.05` that put each step *below* the `1/QUANT` grid, and
  `np.round` sent it to zero. Measured: 30 games and 14,400 updates left five of
  eight tensors **bit-identical to initialisation**, while the loss looked
  unremarkable. Fixed with error feedback (bank the sub-grid remainder per tensor).
  **Any quantised model with a divided learning rate needs this check.**
- **Check that a hypothesis's "obvious" feature exists before trusting it.**
  L3.5's `_geom_features` carries `dest_rank` but no destination-*file* term, and
  uses `abs(tf - ff)` — so `b1->c3` and `g1->h3` produce byte-identical vectors
  and the model cannot represent "a knight belongs near the centre". The missing
  feature, not the weight tying, is what limits it. This is the concrete form of
  H10 and was found by construction, not by a match.
- **A null result on untrained weights is not a finding about architecture.**
  The first L3/L3.5/L3-flat run was 11-of-12 unfinished and said nothing.
  Untrained models shuffle into the ply cap; only training makes the comparison
  meaningful.

## Where to look first

- New to the repo: `encode.py` docstring, then `bench/levels.py`.
- Adding a level: implement `MovePolicy`/`DistributionPolicy`, register a
  `LevelSpec` in `bench/levels.py`.
- Debating an algorithm change to L2: the measured node effects (`--` full
  2049 / no-TT 3254 / no-iterative-deepening 2665 at depth 4) are in
  `.workbuddy-ai/memory/MEMORY.md`; quote those or re-measure, do not guess.
- Deeper history and every bug fixed to date: `.workbuddy-ai/memory/2026-10-02.md`
  and `.workbuddy-ai/memory/MEMORY.md`.
