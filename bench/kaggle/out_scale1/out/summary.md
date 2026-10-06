# L3 training run

**mode:** `supervised`  **seed:** 7

## Training
### supervised  (6659.22s)
- positions_requested: 4000
- positions_applied: 3999
- search_depth: 3

## Evaluation
L3-supervised-trained  vs  greedy-material  (40 games)

- wins: 0  draws: 1  losses: 33  unfinished: 6
- points_pct (wins / played): -0.971

> `points_pct` counts only decisive wins; the bench's own Bradley-Terry fit (which scores draws as 0.5) is the authoritative metric. Re-run the trained model through `python -m bench` for that.