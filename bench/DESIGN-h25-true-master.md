# Design note: H25 — graded by a *true* master (engine / master games)

*Status: **design + shim implemented; no run.** The master itself (a UCI engine
binary or a master-games PGN) is not vendored — supplying it is the remaining
step, and it is deliberately left to the user.*

> **See also:** `HYPOTHESES.md` §H25 (the claim), §H11 (graded by a master),
> §H23 (castling — this arm answers its behavioural half), `DESIGN-tier2-template.md`
> (this is a tier-1 arm, not a tier-2 repair).

## Why this note exists

H25's register entry says the strong version is "blocked on a vendored master
(engine or master games) plus a shim turning master moves into the
`(move, weight)` feedback format the hook expects." This note specifies that
shim and the experiment, so that when a master is supplied nothing has to be
re-invented — and so the arm is *pre-registered* rather than built to succeed
(the repo's standing rule: write the kill condition before building the thing).

## The claim (H25, unchanged)

> At equal positions seen, a learner trained by imitating a master's moves beats
> the same learner trained by self-play outcome-RL, by more than ~1 SE.

**Falsified by:** self-play matching master-graded within 1 SE. A draw here is a
*strong* "naive RL suffices" result, not a non-result.

**This is a decision hypothesis.** It prices the naive premise the whole ladder
rests on.

## What "true master" means — two sources, one interface

The existing weak master is the built-in depth-5 (here, depth-3 in the pilot)
minimax, fed through `L3Trainer.search_feedback`. A *true* master is one of:

| Source | What it supplies | What it costs |
|---|---|---|
| **UCI engine** (e.g. Stockfish) | top-k moves + centipawn scores per position | an engine binary; per-position `analyse` time |
| **Master-games PGN** | the moves actually played from a position, by frequency | a PGN corpus; sparse coverage of arbitrary positions |

Both are normalised to the *same* contract `search_feedback` already returns:

```
targets: list[(chess.Move, float)]   # weight in (0, 1], best == 1.0
```

so the training update path is unchanged — only the *source* of the targets
differs. That is what makes this a matched comparison rather than a new learner.

## The shim

Implemented in `src/chessrl/master.py`. A `MasterSource` is anything with:

```python
def targets(self, board: chess.Board) -> list[tuple[chess.Move, float]]: ...
```

Two concrete sources:

- **`EngineMaster(engine_path, top_k, limit)`** — runs a UCI engine via
  `chess.engine`, takes MultiPV top-k, converts scores to weights.
- **`PgnMaster(pgn_path, top_k)`** — indexes a master-games corpus by position
  key, returns move frequencies.

Conversion rules (pure, unit-tested, no engine required):

- **Engine:** `weight = exp((cp_mover - best_cp_mover) / margin)`, scores taken
  from the **side-to-move's** point of view (matching `search_feedback`, which
  negates the negamax value into the mover's frame). `margin` defaults to 300cp,
  the same "clearly better" margin `search_feedback` uses — so a true-master arm
  differs from the weak-master arm in the *labels*, not in the squash.
- **PGN:** `weight = count(move) / max(count over the position's moves)`, so the
  most-played master move gets 1.0 and the rest scale down.

## The matched control (mandatory, or the result is uninterpretable)

Three arms, one budget, one free variable:

| Arm | Signal | Role |
|---|---|---|
| **weak master** | `search_feedback` at depth 3 | the incumbent weak-master baseline (already runnable) |
| **true master** | `EngineMaster` / `PgnMaster` | the treatment |
| **random-walk** (H26) | outcome RL on `RandomPolicy` games | the load-bearing null — run *first* |

Without H26, "master-graded beats self-play" is indistinguishable from "any
training at all helps". Without the weak-master arm, a win is indistinguishable
from "more compute per label". Both controls are already built.

## What this changes versus the depth-3 teacher we have been diagnosing

The last several experiments diagnosed imitation of a depth-3 minimax — the
*weak* master. A true master changes two things:

1. **Label quality.** A depth-3 search's top-5 is frequently tactical noise at
   the margin; an engine's top-5 with scores is a real preference distribution.
2. **The target-rule question.** `search_feedback` squashes engine *scores* the
   same way, but a master-games source supplies **frequencies**, not scores — a
   genuinely different `p_target` shape. The target-rule / step-size findings
   (2026-10-06) were measured on the score-squash path; they should not be
   assumed to transfer to the frequency path without re-measurement.

## The blocked step (the user's)

Supply one of:

- a **Stockfish (or other UCI) binary path**, or
- a **master-games PGN** (e.g. a Lichess/TCEC elite dump).

Neither is vendored and neither is installed by this note. The shim is inert
until one is provided.

## Kill condition (restated, for the pre-registration)

`true master` within 1 SE of `weak master` **and** within 1 SE of `random-walk`
⇒ supervision buys nothing at this scale, and the naive premise stands. Any
result to the contrary must survive the two controls above before it is claimed.
