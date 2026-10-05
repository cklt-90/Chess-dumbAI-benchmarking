# Cheap-harvest run sheet — decisive experiments that need no training

Every item here is runnable today but several need a *small script* or a *tiny
code addition* (flagged per item). All are **white-space at naive numpy scale**
per `hypothesis-lit-review/report.md`, so a result either way is a contribution,
not a replication. Run from the repo root with the managed interpreter:

```
C:/Users/Trinity/.workbuddy-ai/binaries/python/versions/3.13.12/python.exe -m bench ...
```

Imports used below (all verified against `src/chessrl/*`):

```python
from chessrl.diagnostics import policy_sharpness, PROBE_FENS
from chessrl.perceptron import L3Policy
from chessrl.search import MinimaxEngine
from chessrl.cache import MidstateStore
from chessrl.rating import fit_bradley_terry, MatchRecord
```

---

## H21 — untrained architecture prior points anywhere useful? (NEEDS SCRIPT)

**Verified:** `policy_sharpness(policy, fens=None)` exists; it returns per-probe
`entropy`/`perplexity`/`uniformity` but **not** per-move probabilities, so the
probe-set correlation the register wants needs a direct `move_distribution` loop.

**Script** (`bench/scripts/h21_prior.py`): for `n >= 20` seeds, for each PROBE_FEN
and each legal move, read `L3Policy(seed=n).move_distribution(board)[move_to_index]`
and correlate mean log-prob against (a) centrality = `-dist(square, centre)` and
(b) forwardness = rank-advance for the side to move. Test against a permutation
null over move labels.

```python
import numpy as np, chess
from chessrl.perceptron import L3Policy
from chessrl.diagnostics import PROBE_FENS
from chessrl import masks as M

def centrality(sq):
    f, r = chess.square_file(sq), chess.square_rank(sq)
    return -((f-3.5)**2 + (r-3.5)**2)**0.5
def forwardness(sq, colour):
    r = chess.square_rank(sq)
    return r if colour == chess.WHITE else 7 - r

def correlate(seed):
    pol = L3Policy(seed=seed)
    cents, forws, logps = [], [], []
    for fen in PROBE_FENS:
        b = chess.Board(fen)
        if b.is_game_over(claim_draw=False):
            continue
        dist = pol.move_distribution(b)
        for mv in b.legal_moves:
            idx = M.move_to_index(mv)
            cents.append(centrality(mv.to_square))
            forws.append(forwardness(mv.to_square, b.turn))
            logps.append(float(np.log(dist[idx] + 1e-12)))
    return np.corrcoef(logps, cents)[0,1], np.corrcoef(logps, forws)[0,1]

seeds = range(20)
res = [correlate(s) for s in seeds]
print("centrality r:", np.mean([r[0] for r in res]))
print("forwardness r:", np.mean([r[1] for r in res]))
# Permutation null: shuffle move labels within each board, recompute, repeat 200x.
```

**Readout:** mean correlation ± spread across seeds vs the null. Decisive either
way at near-zero cost. **No games, no training.**

---

## H18 — at depth 2–3, are mistakes tactical or positional? (NEEDS SCRIPT)

**Verified:** `compare_policies` exists but does self-play diagnostics, **not**
disagreement classification. `MinimaxEngine(depth=d)._root_with_value(board, ...)[0]`
returns the chosen move, so we can diff depths directly.

**Script** (`bench/scripts/h18_horizon.py`): on a fixed position set, get the move
from `MinimaxEngine(d2)`, `MinimaxEngine(d3)`, `MinimaxEngine(d5)`; record
disagreements d2-vs-d5 and d3-vs-d5; for each divergence, classify by whether a
capture is legal within 2 plies of the divergence point. Report the tactical
fraction.

```python
import chess
from chessrl.search import MinimaxEngine

def move_at(board, depth):
    return MinimaxEngine(depth=depth, use_tt=False).select(board)

def capture_near(board, plies=2):
    seen = set()
    def rec(b, d):
        if d == 0:
            return False
        for m in b.legal_moves:
            if b.is_capture(m):
                return True
            b.push(m); r = rec(b, d-1); b.pop()
            if r: return True
        return False
    return rec(board.copy(), plies)

fens = [  # a small diverse set
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
]
tactical = 0; total = 0
for fen in fens:
    b = chess.Board(fen)
    d2 = move_at(b, 2); d5 = move_at(b, 5)
    if d2 != d5:
        total += 1
        if capture_near(b):
            tactical += 1
print(f"tactical-fraction of disagreements: {tactical}/{total}")
```

**Readout:** tactical fraction of disagreements. The literature settles the
*principle*; this measures the *magnitude at your depths*. **No new model.**

---

## H17 — transition-focused sampling learns faster? (DONE)

**Verified:** `MidstateStore.record_game(fens, game_id)` and `.sample(n, rng, mix)`
exist; `sample` stratifies by **phase** (`mix`) but had **no transition-proximity
stratifier**.

**Code added:**
- `Midstate` gained `near_transition: bool = False`.
- `record_game` tags each position by the linking move (capture or pawn push).
- `MidstateStore.sample_near_transition(n, rng, mix)` prefers flagged positions
  (phase-mixed within the flagged pool, then backfills), degrading gracefully to
  uniform `sample` when the flagged pool is exhausted.

**Smoke-test** (`bench/scripts/h17_transitions.py`, 8 random games → 320 positions,
42% near-transition):
- uniform sample: 41% near-transition
- transition sample: 66% near-transition (ceiling = 133 flagged / 200 requested)
- matched-update probe agreement (greedy == depth-2): uniform `0.00→0.25`,
  transition `0.00→0.25`, delta **+0.00** — inconclusive at this tiny corpus.

**Readout:** the *mechanism* works (sampler preferentially returns flagged
positions). The *learning-rate payoff* is not yet demonstrated — decisive H17
needs a larger corpus and games-to-fixed-accuracy measurement. Locked by
`tests/test_cheap_harvest.py` regression tests (tagging correctness + sampler
preference).

---

## H12 — does the prior save nodes without changing the value? (THEOREM + PROXY)

**Verified:** `MinimaxEngine._root_with_value` returns `(move, value)`; value
invariance under reordering is a **theorem** (Knuth & Moore 1975) — no test
needed. Node-saving magnitude at numpy scale: `MinimaxEngine(prior_weight=w)`
counts `engine.nodes` for a given board/depth.

**Two options:**
1. **Proxy (numpy-scale, today):** compare `MinimaxEngine(depth=4, prior_weight=0)`
   vs `prior_weight=200` (with an `InformedMinimaxPolicy`-style L1 prior) on a
   fixed position set; assert `_root_with_value` is identical and report node
   counts. This measures the *L1-wrapper* prior, not the L5 model prior.
2. **Faithful (needs torch/L5):** run `GuidedEngine(use_model_prior=on/off)` per
   the register's H12 and compare `.nodes` + `_root_with_value`.

**Readout:** node-count ratio at fixed depth; value must be bit-identical.

---

## H13 / H3 — is 6 games/pair too few? do unfinished games matter? (NEEDS PATCH)

**Verified:** `fit_bradley_terry(records, names, ...)` excludes `unfinished` via
`MatchRecord.played`; there is **no flag** to fold unfinished into draws.

**Code to add** (one flag): add `include_unfinished_as_draws: bool = False` to
`fit_bradley_terry`; when set, add `0.5 * unfinished` to each side's score and
include it in `played`.

**Command:** re-fit from the stored baseline:
```
python -m bench --from bench/results-2026-10-02.json
```
and a second pass with the flag on (after the patch), then compare the ordering
and the standard errors. H13 is already supported by the stored SEs (79–93 >
deltas 36–102); this *quantifies* the draw-inclusion effect for H3.

**Readout:** ordering delta and SE with/without unfinished; H13's "too few games"
claim gets a number.

---

## H23-fix — castling is legal but underplayed (DONE)

**Verified:** the mask admits castling (definitional). The gap is real:
`encode.py::CHANNELS` had **no castling-rights plane** and `value.evaluate()` had
**no shelter term** (per `ONBOARDING.md` / `HYPOTHESES.md` H23).

**Code added (append-only):**
- `encode.py`: appended `castle_w` (22) and `castle_b` (23) as full-board
  constant planes, registered in `COLOUR_PAIRED_CHANNELS` so canonicalisation
  swaps them correctly (plane 22 == side-to-move's rights).
- `value.py`: added `king_shelter_score` — a small potential-based term
  rewarding pawn shield, penalising open king files, and favouring wing files
  (castled kings). Scaled by game phase so it fades in the endgame. Kept small
  (~15 cp max) so it nudges without overriding material or creating
  alpha-beta tie-breaking regressions.

**Census post-fix** (`bench/scripts/h23_census.py`, 6 games/level, 120 plies):

| level | castling | en passant | promotion |
|---|---|---|---|
| random | 2 | 0 | 3 |
| L1-blind | 0 | 0 | 30 |
| material | **9** | 1 | 0 |
| L2-d2 | **12** | 0 | 0 |
| L3-untrained | 0 | 0 | 12 |

Baseline (pre-fix): material 3/6, L2-d2 6 (3/3 games), blind/L3 ≤1/6.
- **Material castling 3→9 (3×)** and **L2-d2 6→12 (2×)** — the shelter term gives
  search and greedy-material a *reason* to castle.
- **Blind (L1) and untrained L3 stay at ~0** — the encode channel adds
  *capacity* but untrained random weights do not create behaviour; the locked
  census test (`test_diagnostics.py::test_the_castling_census_search_castles_blind_does_not`)
  still passes, confirming the invariants.

**No-regression match:** L2-d2 vs material 1-ply → **6/6 wins** (unchanged).

**Test impact:** parameter counts moved (L3 7225→7609, L3.5 783→787) because
`BINARY_CHANNELS` increased from 22→24; the pinned layout test and budget
assertions were updated accordingly. The shelter term was tuned down from an
initial ~45 cp to ~15 cp max to avoid flipping alpha-beta tie-breaking in the
`test_search.py` depth-ladder tests (deeper still beats shallower, zero losses
restored).

---

## Priority order (recommended)

1. **H21** — pure script, near-zero cost, no training. Run first.
2. **H18** — pure script, no training. Run second.
3. **H12** — theorem already settled; proxy node-count is a quick third.
4. **H13/H3** — one-line patch, re-fit from stored JSON.
5. **H17** — small stratifier addition, then matched-update comparison.
6. **H23-fix** — encode/value edits + census; the one with real model-input change.

Items 1–4 are scripts/patches only. Items 5–6 touch the learner internals, so
do them after the no-risk ones land.
