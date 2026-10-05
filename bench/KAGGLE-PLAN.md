# Kaggle plan — turning cheap-harvest results into scale experiments

This document does two things the user asked for:

1. **Suggestions / todos** derived from what the cheap-harvest run sheet
   actually showed (see `CHEAP-HARVEST.md`, all items `DONE`).
2. **Kaggle next steps** — the experiments that need CPU we do not have
   locally, written as concrete, reusable scripts.

The framing comes from the literature review (`hypothesis-lit-review/report.md`):
this harness is **toy / numpy scale** (~10³ params, hundreds of self-play
games), and most published verdicts are *silent or inverted* at that scale. So
the value of a Kaggle run is **measuring the regime boundary**, not confirming
the literature. A local result is novel by construction; that is the portfolio
differentiator, and the README should lead with it.

---

## 1. What the cheap-harvest results imply (todos)

| Result | What it means | Todo |
|---|---|---|
| **H21** — untrained L3 has a central prior (centrality r = +0.07) but **no** forward prior (+0.005) | The encoder leaks geometry, but the payoff is silent until something is *trained*. | Train L3 and re-measure the prior: does training amplify the central bias into play? (tie to the supervised arm) |
| **H18** — depth-2/3 errors are ~100% **tactical** (capture within 2 plies) | The horizon effect, not positional misjudgement, is where L2's remaining strength lives. Quiescence blunts it. | A quiescence-aware L2 variant is the cheap strength win; more depth helps more than a better eval at this scale. Candidate L2 tweak. |
| **H12** — the L1 blind prior saves **~0%** nodes (ratio 1.000) | The non-learning wrapper's prior is a *tie-breaker, not a search accelerator* at numpy scale. | The **model** prior (L5 `use_model_prior`) is the only ordering accelerator worth measuring — run the *faithful* H12 arm (torch) on Kaggle, not the L1 proxy. |
| **H13 / H3** — counting unfinished as draws fakes precision (SE 74→65, ratings collapse) | The "exclude unfinished" convention is validated; including them is dishonest. | **All future benches use ≥20 games/pair and always report the SE.** Record this as a standing methodology rule. |
| **H17** — transition sampler works, payoff **inconclusive** | The mechanism (preferential sampling near captures/pawn pushes) is proven; the learning-rate gain needs a bigger corpus. | Run transition-focused vs uniform at Kaggle scale; measure *games-to-fixed-accuracy*, not a single snapshot. |
| **H23-fix** — castling channels + shelter term: material 3→9, L2-d2 6→12; **blind/L3 stay ~0** | The fix gave learners *capacity*; behaviour still needs *training*. Castling needs a reason, and only a trained/supervised signal supplies one. | **Train L3 and re-run the castling census** (`h23_census.py`). Does a trained L3 castle now? This is the H23 behavioural follow-up and the headline Kaggle test. |

The single most important todo: **the cheap-harvest proved the harness is
sound and located the gaps; the gaps are all training-shaped, so the next phase
is training, run where the CPU lives.**

---

## 2. Kaggle experiment design (the training-signal axis, Group G/H)

The script `bench/kaggle/kaggle_l3_train.py` reuses the existing, tested
trainers (`L3Trainer`) — there is no new learning code, only orchestration — so
a Kaggle run is identical to a local run, just bigger. Every arm is numpy-only
(no torch), matching the repo's stated property that L0–L3 run without the
optional extra.

| Arm (`--mode`) | Signal | Hypothesis | Why it is the right Kaggle job |
|---|---|---|---|
| `selfplay` | outcome RL vs frozen self-copy | H24 baseline / H26 reference | The incumbent. Establishes what naive self-play alone buys at scale. |
| `supervised` | imitate a search on a position corpus (depth **2–3**) | **H25 / H11** | The literature review's flagged first-run arm: a built-in weak master supplies the label. Cheapest *trained* path; also the arm most likely to make L3 castle (H23). **Depth 5 is not viable at scale** — see cost note below. |
| `randomwalk` | outcome RL on random-vs-random games | **H26 control** | Run **first**. If this beats untrained, every comparison above is confounded by "more games = better". The load-bearing null. |
| `both` | selfplay then supervised | H24 + H25 combined | Recommended single Kaggle job: learn from play, then sharpen with search. |

Recommended Kaggle sweep (each arm is a separate job / different seed):

1. `randomwalk` — 4000 games, `--eval-against untrained`. **Must not beat untrained.** If it does, stop and fix the loop.
2. `supervised` — **4k positions, `--search-depth 3`, `--eval-against material`** (~1.6 h on Kaggle CPU at the measured rate). Does imitation reach material/L2-d1?
3. `selfplay` — **BLOCKED** until the signal fix below is tested locally. Do not spend Kaggle hours here yet.
4. `both` — as above; the self-play half contributes nothing until step 3 is fixed, so it is currently equivalent to `supervised` plus wasted time.

### Cost note (measured — local vs Kaggle CPU)

Per-position labelling is the dominant expense and scales steeply with search
depth. Local rates and the **measured Kaggle CPU rate** (2026-10-05 validation
run, 300 positions at depth 3 took 443.75 s):

| depth | s/pos (local) | s/pos (Kaggle CPU, measured) | positions | Kaggle time |
|---|---|---|---|---|
| 2 | ~0.04 | ~0.25 (est., 6× local) | 4k | ~17 min |
| 3 | ~0.24 | **~1.48** (measured) | 4k | **~1.6 h** |
| 3 | ~0.24 | ~1.48 (measured) | 6k | ~2.5 h |
| 3 | ~0.24 | ~1.48 (measured) | 24k | **~10 h** |
| 4 | ~4.5 | ~27 (est.) | — | not viable |

**The earlier "24k positions ≈ 1.6 h" estimate was ~6× optimistic** — it used the
local rate. On a Kaggle CPU, depth-3 labelling runs at ~1.5 s/pos, so 24k
positions is ~10 h, not 1.6 h. (Kaggle's session cap is 12 h; a 24k job would
fit in one session but leave no headroom.)

Recommended first scale run: **~4k positions at depth 3 (~1.6 h)**, which is
already 13× the validation budget, rather than 24k in one shot. Record the
per-arm `seconds` from `results.json` and revise again — the rate is the single
most important number for planning.

Self-play training is cheap (~0.15 s/game → 4000 games ≈ 10 min), *but see the
self-play finding below: more self-play games are not useful until the signal
problem is fixed.* A full Kaggle job is therefore gated almost entirely by the
supervised labelling depth: use **depth 2–3** at scale, and reserve depth 4–5
for tiny local probes. The "weak master" in H25 is therefore a depth-2/3 search
at this regime — which is honest: relative to an *untrained* L3 it is still a
strong, real teacher.

### Self-play signal finding (measured locally, 2026-10-05)

L3 self-play **does not generate a usable outcome-RL signal at this regime**, and
the reason is a cap/signal trap, not a bug:

- With the current caps (`max_plies=120`, `no_progress_limit=60`), **19–20 of 20
  games hit the ply cap** → all `unfinished`. They still train, but *only* via
  the `shaped_reward` branch in `L3Trainer.white_reward` (score the final
  position's material, `tanh(eval/1000)*0.3`). Measured mean |shaped reward| on
  unfinished games ≈ **0.053** — a weak, material-only gradient.
- With a lenient cap (300 plies), the same self-play games terminate *naturally*
  — **40/40 by `insufficient_material`**, median 288 plies — and every one is a
  **draw**. `white_reward` returns exactly `(0.0, 0.0)` for finished draws, so
  **total reward mass = 0.00 over 40 games: zero learning signal.**
- Decisive games across every cap tested: **0/20** at each of 120/200/300/400.
  Blind-vs-blind L3 trades everything off and drifts to bare kings.

So the two settings are both dead ends: short cap → weak shaped-only gradient;
long cap → zero signal. The arm as configured cannot learn strength. This is why
the validation run's self-play arm (18,544 updates) showed no measurable gain.

**Consequence for the sweep:** the `selfplay` and `both` arms are currently
*not* informative at scale — more games just burn time. Before spending Kaggle
hours on them, fix the signal, e.g. one of:
1. **Exploration/opening diversity** — play self-play from varied `start_fen`
   draws (midstate store) so games are not all identical trade-fests.
2. **Adjudicate truncated games by material** as a *decisive* reward (a shaped
   terminal, not a soft per-ply nudge), so a truncated game still yields ±1.
3. **Reduce blunder-free drift** — a small temperature/randomness floor so the
   policy does not always pick the (equal) trade.

Until (1)/(2)/(3) are tested locally, the **supervised arm is the only arm
worth running at Kaggle scale** — and it is also the H23 headline arm.

Then drop the saved `l3_<mode>.json` into `python -m bench --only L3 <refs>`
(trained L3 is just an `L3Policy`; build a small spec for it) to get a proper
Bradley-Terry rating **with SE** over the whole roster.

### H23 follow-up (the headline)
After any *trained* run, re-run `bench/scripts/h23_census.py` with the trained
L3 in the roster. The pre-fix census showed blind/L3 castle ~0; if a
supervised-trained L3 casts, that is the first behavioural proof the castling
channel + shelter term were the missing piece (feature vs signal resolved:
it was the signal — self-play had no reason to castle, the master does).

### H17 follow-up
Add a `--transition-focused` flag to the supervised/selfplay corpus collection
that uses the new `MidstateStore.sample_near_transition` instead of uniform
random positions, and compare games-to-fixed-accuracy against uniform. That is
the decisive H17 test the tiny local corpus could not deliver.

---

## 3. How to run

Local smoke test (proves the script end-to-end — save/load round-trip, both
training arms, a 20-game eval — in minutes, before you upload):

```bash
python bench/kaggle/kaggle_l3_train.py --mode both --games 20 --positions 300 \
    --search-depth 4 --eval-against untrained --eval-games 20 --seed 1 \
    --out-dir bench/kaggle/out_smoke
```

Kaggle scale:

```bash
# Supervised only (~1.6 h at the measured Kaggle rate). Depth 3, 4k positions.
python bench/kaggle/kaggle_l3_train.py --mode supervised --positions 4000 \
    --search-depth 3 --eval-against material --eval-games 40 --seed 7 \
    --out-dir /kaggle/working/out
```

(Do **not** run depth 5 at scale — it is ~300 h. And do not run the `selfplay`
or `both` arms at scale until the self-play signal finding above is addressed.)

Kaggle environment notes: drop the repo on the notebook filesystem; the script
bootstraps `src/` and `bench/` onto `sys.path` from its own location.
`pip install python-chess numpy` if absent. **Do not install torch** for this
script — it is never imported.

---

## 4. What a result would mean (prove / not-prove mapping)

These are the verdicts the portfolio README and the separate *prove/not-prove*
document will record. The Kaggle runs are what supply the evidence:

- **H26 (control):** random-walk == untrained  ⇒  "more games" is not the driver. If random-walk *beats* untrained, the benchmark's training loop is the finding, not any paradigm — a strong, novel, negative.
- **H25 (supervised):** trained-by-search beats self-play within ~1 SE  ⇒  naive RL is *not* enough; supervision is what L3 lacked (also resolves H23's feature-vs-signal question). A draw  ⇒  "naive is enough at this scale" — itself a clean result.
- **H11 (learned leaf):** a supervised L3 used as L5's `use_model_eval` beating `use_model_eval=False`  ⇒  the trained value head earns its place. Blocked until L5 exists in the bench; the L3 half is runnable now.
- **H24 (diversity):** round-robin (diverse opponents) beats self-play  ⇒  opponent diversity helps. Needs the per-paradigm opponent scheduler (thin build, noted in HYPOTHESES.md Group G).
- **H17 (transitions):** transition-focused reaches fixed strength in fewer games than uniform  ⇒  the no-progress counter should also be a *sampling* rule. Inconclusive at local scale; this is the decisive test.

---

## 5. Standing methodology rule (from H13/H3)

> Every benchmark run reports its **standard error**, and uses **≥20 games/pair**.
> Unfinished games are excluded from the rating fit (they carry no signal); they
> are reported as a column, never silently scored as draws.
