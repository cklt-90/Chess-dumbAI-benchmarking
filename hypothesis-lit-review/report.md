# Hypothesis-based literature review for chess-rl-bench

Compiled from 15 validated result files (`results/*.json`, 100% field coverage). Fields marked uncertain by the research agents are omitted from the body and listed per item.

## Table of Contents

1. [h17-transitions](#h17-transitions) — partially supports - non-uniform sampling demonstrably accelerates value learning per transition (PER, Ape-X, prioritized sweeping), but published gains key on TD-error magnitude or goal relabeling, not on move irreversibility; no published work tests irreversible-move proximity as the sampling criterion. | General mechanism established at toy-scale control through DQN/Atari scale (small-net and above); the chess-specific irreversibility variant is untested at every scale, including the AZ/Lc0 regime which uses uniform replay windows.
2. [h21-arch-priors](#h21-arch-priors) — supports the premise, silent on the payoff - structure-without-trained-weights carrying usable priors is established (Deep Image Prior, spectral bias, weight-agnostic networks), but no published work measures spatial move-preference bias of an untrained policy over chess moves; whether the local prior points anywhere useful is untested. | Established at small-net scale (DIP: single randomly-initialized convnets; spectral bias: MLPs; WANN: small evolved topologies); at toy/numpy factorised-softmax scale the mechanism is simpler - a deterministic function of input statistics times random weights - but unpublished; nothing extends to AZ/Lc0-scale because those priors are swamped by training.
3. [h23-special-moves](#h23-special-moves) — supports - every serious chess representation since Shannon-era engines encodes castling rights as explicit input (AZ's 119-plane stack includes 4 constant castling planes; Giraffe hand-encodes castling features), and sparse-event RL results (HER) confirm rare delayed-payoff events are under-learned without explicit representation or reward; 'legal but invisible' is a known, diagnosed failure pattern. | Holds across all regimes: engine-scale handcrafted evals reward castled positions indirectly via king-shelter/storm terms; AZ/Lc0-scale learned systems receive castling rights as input planes and learn the value from outcomes; toy/numpy-scale has neither - which is exactly the local gap.
4. [h18-horizon](#h18-horizon) — partially supports - the direction of the claim is canonical: Berliner identified the horizon effect as the characteristic failure of fixed-depth search, and quiescence search exists precisely because leaf evaluation amid unresolved captures is the dominant error source of fixed-depth alpha-beta. But no published work measures the tactical-vs-positional composition of a shallow engine's disagreements with a deeper reference at these specific depths, so the magnitude claim is silent. | Principle established at engine-scale: hand-crafted eval, alpha-beta at 5-15 plies, from Berliner's 1970s analyses through mature open-source engines. Untested at toy/numpy-scale depth 2-3. At AZ/Lc0-scale the question largely dissolves: a learned value net plus MCTS has no fixed-depth horizon in the classical sense, so that lineage is silent on the claim.
5. [h19-nontransitivity](#h19-nontransitivity) — partially supports - chess non-transitivity is confirmed empirically at billion-game human scale and Elo's failure under cycles is proven, supporting H7's general claim that no rating ordering is fully faithful; but the specific structural claim of H19 (cycles run through a shallow greedy-material player) is novel and untested anywhere in the literature. | Established at human population scale (Lichess/FICS, >1 billion games, Sanjaya et al. 2022) and at AI population scale (AlphaZero agents, StarCraft II, Hex; Czarnecki et al. 2020; Balduzzi et al. 2018). Untested at toy/numpy-scale round-robins of 4-8 weak agents - the local regime - though nothing in the theory excludes it; cycles are a property of the payoff matrix, not of strength.
6. [h20-tactics-first](#h20-tactics-first) — supports - both the historical record (brute-force type-A programs with dominant material terms beat knowledge-rich selective programs for four decades) and the modern stress test (Ruoss et al. 2024) agree that at low search/eval budget tactical-material correctness is the binding constraint, and positional sophistication pays only once tactics are secured. | Established from toy/numpy-scale through engine-scale historically (Shannon 1950 through the brute-force era, Chess 4.x to Deep Blue). The boundary moved only with Ruoss et al. 2024: at 270M parameters trained on ~15 billion Stockfish-16 action-values, a pure policy without search handles tactics at grandmaster level - but that regime (small-net >> naive scale, supervised by a search engine) is unreachable and irrelevant at numpy scale.
7. [h16-factored-action](#h16-factored-action) — partially supports — shared/factorised move parameterisations are universal in strong game-playing networks, but no published work isolates statistical sharing from capacity at matched parameters and matched updates per legal move, which is exactly H16's condition. | Holds as folk wisdom at small-net (<=10M params) and AZ/Lc0-scale; weakens at the searchless 270M-param scale, where a flat 1968-way softmax (Ruoss et al. 2024) reaches grandmaster level without any factorisation. No published evidence exists at toy/numpy-scale (~10^3 params).
8. [h08-weight-tying](#h08-weight-tying) — supports — weight sharing across board positions is the most established inductive bias in game-playing networks: every strong published chess net ties massively, and the one matched-parameter comparison (GNN vs CNN) has the more-tied model winning. The support is structural, not from controlled untied baselines. | Established at small-net (<=10M params) and AZ/Lc0-scale; untested at toy/numpy-scale (~10^3 params). At engine scale, NNUE shows structured partial untying (a separate input slice per king position) is worth its parameter cost — the boundary of the tying claim.
9. [h09-budget-vs-topology](#h09-budget-vs-topology) — mixed — above ~10^6 parameters loss is a smooth function of budget and topology shifts constants rather than exponents (supports 'budget dominates'); in the only matched-parameter game-playing comparisons, inductive bias adds large gains on top of budget (contests it). At ~10^3 parameters the literature has no measurement at all. | The verdict flips across regimes. Budget-dominates holds in the Kaplan/Hoffmann regime (~10^6-10^10+ parameters, language modelling) and across Ruoss's 9M->136M->270M ladder at fixed data. Topology-matters holds in matched-parameter game-net comparisons (Rigaux & Kashima GNN vs CNN; ChessFormer position encodings) and at engine scale (NNUE). Toy/numpy-scale: no published verdict exists.
10. [h10-relational-limits](#h10-relational-limits) — partially supports - literature agrees per-square/per-object functions cannot express irreducibly relational facts and that working chess systems always carry cross-square machinery; it does not isolate weight tying from missing features, which is exactly the local caveat. | Supports at small-net (<=10M params: AlphaGateau GNN vs CNN) and engine scale (Giraffe's global/square-centric feature modalities; handcrafted engine evals with pawn-structure and king-safety terms). Silent at toy/numpy scale.
11. [h11-learned-leaf-eval](#h11-learned-leaf-eval) — supports - every chess-learning system that completed training, from KnightCap's linear eval to NNUE, replaced or beat the handcrafted leaf eval; the local untrained negative is consistent with the literature (nobody reports an untrained value head helping), but the literature is silent at numpy scale. | Supports at small-net scale (KnightCap linear eval ~tens of thousands of weights; Giraffe ~4-layer MLP) and at engine/AZ scale (NNUE, AlphaZero, MuZero). Silent at toy/numpy scale: the cheapest published success still used thousands of FICS games or ~175M self-play positions.
12. [nnue](#nnue) — supports - NNUE's displacement of handcrafted eval in essentially all top engines since 2020 is the canonical demonstration of the claim; it is the strongest single data point in the H11 lineage and it used supervised training, not MCTS-scale RL. | Engine-scale (NNUE + deep alpha-beta search, CPU SIMD, millions of evals/sec/thread). Silent at toy/numpy scale: the speed mechanisms (incremental accumulator, int8/int16 quantization) presuppose a deep search tree and huge eval counts the local harness does not have.
13. [h12-move-ordering](#h12-move-ordering) — supports - the invariance half is proven (alpha-beta computes the same root value under any ordering; Knuth & Moore 1975), and the node-saving half is the most exploited efficiency result in game search, though all published magnitudes come from heuristic stacks or neural priors far above the local perceptron-scale prior. | Value invariance: all regimes (exact theorem for fixed-depth alpha-beta without inexact pruning). Node savings: engine-scale canonical (TT + MVV-LVA + killer + history ordering is load-bearing in every strong alpha-beta engine) and AZ/Lc0-scale for learned priors inside MCTS (PUCT). At toy/numpy scale (depth<=5, no TT, ~10^3-param prior) the theorem still holds but the saving factor shrinks with depth; magnitude there is unpublished and unmeasured locally.
14. [h13-rating-methodology](#h13-rating-methodology) — supports - for H13: rating theory (Glickman's guide; Glicko-2's RD; WHR; Fishtest's SPRT practice) puts the standard error of a handful of games per pair at tens-to-hundreds of Elo, matching the observed 79-93, so 6 games/pair cannot establish sub-100-Elo gaps. For H3: partially supports - the BT-with-ties literature treats genuine draws as an informative third outcome but offers no sanction for counting TRUNCATED games as draws; exclusion is the conservative, precedent-backed convention, and no published work tests this harness-specific convention directly. | Regime-free: the statistics are scale-invariant. The same 1/sqrt(games) standard errors, draw-rate corrections, and SPRT machinery govern a toy numpy roster and Fishtest's 10^4-10^5-game engine tests. Verdicts hold identically across toy/numpy, small-net, AZ/Lc0, and engine scales; only the draw rate (which enters the SE constant) shifts with strength.
15. [searchless-gm](#searchless-gm) — supports - at 270M params trained on 10M Stockfish-annotated games, a pure action-value policy reaches 2895 Lichess blitz Elo vs humans and 93.5% on 10k puzzles with zero search; but the same paper bounds the claim: it trails Stockfish 16 by ~407 Elo and AlphaZero+400-sim MCTS by ~200 Elo, and its 9M sibling manages only ~2007 Elo - the search-vs-eval division blurs only past a scale floor ~5 orders of magnitude above the local harness. | Verdict holds only in the 10^8-param / 10^7-game / 10^10-datapoint regime - a new published point between small-net and AZ self-play scale, distinguished by being supervised from Stockfish rather than trained by RL. Explicitly inverted at <=10M-param scale: the paper's own 9M model scores ~2007 tournament Elo, and its scaling ablation shows overfitting at 10k games for models >=7M params. At the local toy/numpy regime the claim fails, which is precisely its local role.

## h17-transitions

### Register
**register_link**
  > H17 - 'the value function changes at pawn moves and captures; positions near one of those carry most of the learnable signal; a policy trained only on positions within N plies of a pawn move or capture learns faster per game than one trained uniformly.' Local status: OPEN, untested, runnable today (HYPOTHESES.md). Circumstantial support: decisive games spread member accuracy 0-1 vs 0.80-0.85 in unfinished; the no_progress_limit already encodes the same fact as a stopping rule; the 2026-10-03 review calls it 'the best-supported untested hypothesis'.

**register_implication**
  > Keep OPEN but annotate: literature supports the general mechanism (non-uniform sampling accelerates value learning per update) while the specific irreversibility criterion is unpublished; no status change justified until the matched-update local test runs.

### Verdict
**literature_verdict**
  > partially supports - non-uniform sampling demonstrably accelerates value learning per transition (PER, Ape-X, prioritized sweeping), but published gains key on TD-error magnitude or goal relabeling, not on move irreversibility; no published work tests irreversible-move proximity as the sampling criterion.

**scale_regime**
  > General mechanism established at toy-scale control through DQN/Atari scale (small-net and above); the chess-specific irreversibility variant is untested at every scale, including the AZ/Lc0 regime which uses uniform replay windows.

### Evidence
**key_sources**
  - Schaul et al. 2016, Prioritized Experience Replay (arXiv:1511.05952) - canonical result: replaying high-|TD-error| transitions more often beats uniform sampling per update; establishes 'sample where value changes fastest'.
  - Horgan et al. 2018, Distributed Prioritized Experience Replay / Ape-X (arXiv:1803.00933) - prioritization compounds with distributed acting; SOTA Atari results, but confounded with data scale.
  - Andrychowicz et al. 2017, Hindsight Experience Replay (arXiv:1707.01495) - converting uninformative transitions into signal (goal relabeling); standard remedy when informative events are rare.
  - Moore & Atkeson 1993, Prioritized Sweeping - ancestor method: concentrate backups where expected value change is largest; H17 is its chess-structural variant with a free-to-compute priority.
  - Syzygy/DTZ endgame tablebases (de Man, Fiekas; python-chess docs) - chess-specific evidence that irreversible moves (capture/pawn) are exactly where WDL can change irreversibly, privileged by a dedicated metric.

**literature_findings**
  > Consensus: sampling transitions non-uniformly, in proportion to how much they can change the learned value, accelerates value learning per update across tabular, small-net, and Atari-scale DQN regimes. PER formalizes the priority as |TD-error| with proportional or rank-based sampling; Ape-X shows the same idea survives distribution to hundreds of actors; HER shows an orthogonal lever - relabeling so that 'failed' transitions carry signal. Chess-specific structural support is indirect but real: tablebase work defines DTZ precisely because pawn moves and captures are the irreversible boundaries at which game-theoretic value can reset, and fifty-move/no-progress rules institutionalize the same boundary. Active disagreements and limits: (1) TD-error prioritization provably overweights stochastic or high-variance transitions and needs importance-sampling correction with beta annealed to 1 to avoid biased gradients; (2) whether prioritization still helps once the buffer is enormous and mostly self-generated (AZ-lineage) is less clear - AZ, MuZero, and Lc0 all use uniform sampling from a recent window, so at engine-scale the field effectively declined the PER lever; (3) no work isolates a structural, domain-knowledge priority such as irreversibility, so H17's specific criterion is genuinely novel rather than a replication.

**matched_controls**
  > PER vs uniform DQN: matched (same network, buffer size, hyperparameters; only sampling distribution changed). HER vs DDPG baseline: matched (same algorithm and hyperparameters; only replay relabeling changed). Ape-X vs Rainbow/prior baselines: NOT matched (360 actors, ~100x frames; prioritization entangled with throughput).

**known_failure_modes**
  > PER without importance-sampling weight annealing (beta -> 1) changes the effective update distribution and biases gradients; TD-error priority locks onto stochastic/high-variance transitions, which then get replayed indefinitely; hard prioritization shrinks effective batch diversity; priorities go stale as the value updates; HER-style relabeling presumes a goal-conditioned objective and does not transfer to a win-probability target like chess.

### Mapping
**mapping_to_local**
  > Local claim narrows the literature: instead of an error-based priority (which requires a trained value function to even compute), H17 proposes a structural priority - proximity to an irreversible move - computable for free at record time, closer in spirit to prioritized sweeping with domain knowledge than to PER. The literature supports the mechanism (sample where value changes) but is silent on this criterion. Local circumstantial evidence aligns: the no_progress_limit already treats irreversibility as a stopping rule and DTZ treats it as a value boundary; H17 asks the same fact to work as a sampling rule.

**scale_limits**
  > All PER/HER evidence comes from value learning with function approximation on dense-reward Atari or goal-conditioned robotics, with buffers of 1e5-1e6 transitions; nothing speaks directly to perceptron/factorised-softmax policies trained on numpy-scale midstate corpora of self-play games capped by a no-progress rule, where outcomes are binary and delayed.

**small_scale_replication**
  > Not yet checked - a cheap local experiment exists (MidstateStore.record_game plus a transition-proximity stratifier on top of the existing phase stratifier; no new model needed). The literature cannot supply the irreversibility criterion itself (regime gap), but the mechanism-level question is locally decidable today.

**confounds_flagged**
  > Circumstantial local evidence (member accuracy spread 0-1 in decisive vs 0.80-0.85 in unfinished games) is confounded by which member types reach decisive results and by game length; published PER gains are entangled with alpha/beta hyperparameter tuning; Ape-X gains are entangled with ~100x data throughput; Atari's dense per-step rewards make 'informative transition' far more frequent than in chess with terminal-only outcomes.

**suggested_local_test**
  > Two arms at matched gradient-update counts over the same recorded games: (a) uniform corpus sampling vs (b) sampling only positions within N plies (N in {1,3,5}) of a pawn move or capture; readout is games-to-fixed-accuracy on a held-out probe set. Optional third arm once a value head exists: PER-style TD-error priority, to rank the free structural priority against the canonical error-based one.

### Uncertain (skipped above)
- quantified_effect


## h21-arch-priors

### Register
**register_link**
  > H21 - 'an untrained model still prefers central pawn pushes to a-file pushes, measurably above chance, because encode.py's channel semantics and the factorised structure carry a weak prior before a single update.' Local status: CONTESTED - settled half: untrained-vs-untrained matches are ~random (L3 0-6 draws; 2026-10-03 review cites four independent observations); open half: random weights do not imply random preferences - the factorised softmax with zero logits is uniform per factor, not per move.

**register_implication**
  > Keep CONTESTED but reword the open half: literature establishes that architecture x initialization carries a functional prior before any update (DIP, spectral bias, WANN), so the question narrows from 'is there a prior' to 'does the factorised prior align with useful chess geometry' - the probe-set correlation test, not more untrained matches.

### Verdict
**literature_verdict**
  > supports the premise, silent on the payoff - structure-without-trained-weights carrying usable priors is established (Deep Image Prior, spectral bias, weight-agnostic networks), but no published work measures spatial move-preference bias of an untrained policy over chess moves; whether the local prior points anywhere useful is untested.

**scale_regime**
  > Established at small-net scale (DIP: single randomly-initialized convnets; spectral bias: MLPs; WANN: small evolved topologies); at toy/numpy factorised-softmax scale the mechanism is simpler - a deterministic function of input statistics times random weights - but unpublished; nothing extends to AZ/Lc0-scale because those priors are swamped by training.

### Evidence
**key_sources**
  - Ulyanov et al. 2018, Deep Image Prior (arXiv:1711.10925) - a randomly-initialized convnet's structure alone is a usable image prior with zero training data: architecture as prior, before any learning.
  - Rahaman et al. 2019, On the Spectral Bias of Neural Networks (arXiv:1806.08734) - deep ReLU networks are biased toward low-frequency functions; initialization and architecture jointly determine what is 'easy' to represent.
  - Xu, Zhang, Luo 2022, Overview: frequency principle/spectral bias in deep learning (arXiv:2201.07395) - consolidates the F-Principle across architectures and datasets; the bias is generic, not task-specific.
  - Gaier & Ha 2019, Weight Agnostic Neural Networks (arXiv:1906.04358) - architectures evolved to perform RL tasks with one shared random weight; topology alone can carry task competence.
  - Saxe et al. 2011, On Random Weights and Unsupervised Feature Learning - random-weight convnets plus pooling yield features useful for recognition; untrained structure is not neutral.

**literature_findings**
  > Consensus across three independent lines: (1) architecture x initialization encodes a functional prior before any gradient update - convnets encode locality and translation equivariance, MLPs encode smoothness/low-frequency preference (spectral bias), and even per-layer normalization and fan-in scaling shape which functions are likely at init; (2) these priors are practically useful for statistics resembling natural signals (DIP for images, random conv features for recognition, WANN for small control tasks); (3) 'untrained' therefore never means 'unstructured'. Limits and disagreements: all evidence is on continuous input manifolds (images, robot states, audio); the demonstrated priors exploit spatial smoothness, which a discrete 8x8 board with legal-move masking only partially has; spectral bias is established for regression-style losses and is less characterized for softmax policies over discrete action sets; WANN architectures were evolved, so their competence is selected, not generic. Nothing published measures whether an untrained policy over chess moves prefers central/forward moves - the local falsification target has no literature precedent.

**matched_controls**
  > DIP vs learned-prior methods: not matched-task comparisons (single-image optimization vs training-set priors; different data assumptions by design). WANN: architecture search with weight-sharing, not a random-architecture control. Spectral bias: controlled synthetic experiments, matched. None of the cited work is a matched comparison of untrained chess policies; the local probe test would be the first of its kind.

**known_failure_modes**
  > Random-feature quality depends strongly on input statistics (whitening/normalization), so the 'prior' is partly a property of the encoder's input scaling, not the architecture alone; initialization schemes (He/Xavier) themselves encode smoothness, confounding 'structure vs init' attribution; the frequency principle is characterized mainly for MSE-trained regression, not softmax policies over discrete move sets; DIP-style priors fail on noise-like or high-frequency targets; WANN results are topology-searched and do not show arbitrary random topologies carry useful priors.

### Mapping
**mapping_to_local**
  > Local hypothesis is the discrete-action analogue of the same claim: factorised P(from) x P(to|from) x P(promo) with random weights induces a non-uniform distribution over moves purely from move-set geometry - the register already notes zero logits give uniformity per factor, not per move. The literature says such structural priors are real and sometimes useful (supports the premise); it is silent on whether this particular prior prefers good moves (centrality/forwardness), which is exactly the open local half.

**scale_limits**
  > All cited priors operate on continuous or smooth input statistics; chess's discrete board, legal-move mask, and outcome-only supervision sit outside their demonstrated regime. At AZ/Lc0-scale any init prior is quickly dominated by training signal, so the question only has a window of relevance at small/numpy scale - which is precisely where no published measurement exists.

**small_scale_replication**
  > Not yet checked - a near-zero-cost local experiment exists (policy_sharpness over a fixed probe set across seeds, correlated against centrality/forwardness scores; no games needed). This is decisively answerable locally; the literature contributes the framing, not the measurement.

**confounds_flagged**
  > The initialization distribution itself carries part of the spatial prior (fan-in scaling smooths over input dimensions), so 'encoder semantics' and 'init scheme' are confounded; the legal-move mask already biases toward pieces with more targets; per-move probability is a product of per-factor probabilities, so apparent 'preference' can be an artifact of move-set factorization rather than position content; single-seed probes are noise - need seed distributions and a permutation null over move labels.

**suggested_local_test**
  > Run policy_sharpness(L3Policy(seed=n)) for n >= 20 seeds over a fixed probe set; score every legal move by target-square centrality (distance to board centre) and forwardness (rank advance for the moving side); rank-correlate mean move log-probability against both scores and test against a permutation null over move labels. Decisive either way at near-zero cost and no games played.

### Uncertain (skipped above)
- quantified_effect


## h23-special-moves

### Register
**register_link**
  > H23 - mechanical half: every special move (castling, en passant, under-promotion) is admitted by the move mask - SUPPORTED 2026-10-02, definitional (mask built from board.legal_moves). Behavioural half: special moves are played far less than real chess warrants because nothing in evaluate() rewards them - PARTLY SUPPORTED: L2-d2 castled 6 times (3/3 games); blind/untrained/random/material <=3 in 6; en passant twice in 80+ L1/L3 games. Gap located: no castling term in evaluate(), no castling-rights channel in encode() (one scalar in board_context_features only).

**register_implication**
  > Mechanical half stays SUPPORTED; behavioural half can move toward SUPPORTED for castling once the fix is made and the census re-run - the literature gives the canonical repair (castling-rights input planes per AZ; king-safety eval terms per engine practice) and the warning (shape via consequences, not per-move bonuses).

### Verdict
**literature_verdict**
  > supports - every serious chess representation since Shannon-era engines encodes castling rights as explicit input (AZ's 119-plane stack includes 4 constant castling planes; Giraffe hand-encodes castling features), and sparse-event RL results (HER) confirm rare delayed-payoff events are under-learned without explicit representation or reward; 'legal but invisible' is a known, diagnosed failure pattern.

**scale_regime**
  > Holds across all regimes: engine-scale handcrafted evals reward castled positions indirectly via king-shelter/storm terms; AZ/Lc0-scale learned systems receive castling rights as input planes and learn the value from outcomes; toy/numpy-scale has neither - which is exactly the local gap.

### Evidence
**key_sources**
  - Silver et al. 2017, Mastering Chess and Shogi by Self-Play (AlphaZero, arXiv:1712.01815) - castling rights are 4 of the 7 constant input planes; the canonical representational fix for exactly the local encoder gap.
  - Andrychowicz et al. 2017, Hindsight Experience Replay (arXiv:1707.01495) - the standard sparse-event remedy: when informative events are rare and unrewarded, re-represent or relabel so they carry signal.
  - Lai 2015, Giraffe (arXiv:1509.01549) - even a fully learned evaluation hand-encodes global features including castling rights; learned does not mean representation-free.
  - Ng, Harada & Russell 1999, Policy Invariance Under Reward Transformations - potential-based shaping: the correct way to add a castling incentive without distorting the optimal policy.
  - Stockfish handcrafted eval (king-safety shelter/storm terms) - engine-scale convention: castling is rewarded through its consequences (king shelter, pawn shield), not as a move bonus.

**literature_findings**
  > Consensus on representation: castling rights are first-class state in every serious chess program, for a hard reason - future castling legality is not derivable from piece placement alone (it depends on king/rook move history), so a Markov state must include it; AZ encodes it as 4 constant planes, Giraffe as handcrafted global features, and all engines track it in the board state. Consensus on behavior: sparse events with delayed payoff are systematically under-produced by learners lacking explicit representation or reward - HER exists precisely for this pattern, and classical RL treats 'legal but unrewarded' actions as effectively invisible to greedy learners. Engine practice adds the third leg: handcrafted evals never bonus the castling move itself; they reward the consequences (king shelter, pawn storm, rook connectivity), which is why searching levels 'find a reason' to castle. Active limit: no published matched-parameter ablation measures the isolated contribution of castling planes/terms - the convention rests on representational necessity and long practice, not controlled ablation evidence.

**matched_controls**
  > Not matched. AZ, Giraffe and engine evals adopt castling representation on Markov-necessity grounds; no published comparison holds architecture, compute and data constant while removing castling input planes or castling/king-safety eval terms. HER's sparse-reward comparisons are matched but in robotics, not chess.

**known_failure_modes**
  > A per-move castling bonus invites reward hacking (castling into attack to collect the bonus) unless the reward is potential-based (Ng et al. 1999); HER-style relabeling does not transfer - castling is not a goal state; globally constant input planes couple poorly with per-square weight sharing unless broadcast into positional features; the local census is tiny (3-6 games per level), so behavioural rates have very wide binomial intervals; under-promotion rates are additionally deflated by the 120-160 ply cap truncating late endgames.

### Mapping
**mapping_to_local**
  > The local gap is precisely the representational omission the literature standardizes against: encode() lacks castling-rights planes where AZ has 4 constant ones, and evaluate() lacks king-safety terms where engines carry shelter/storm weights. The local behavioural split (L2-d2 castled 3/3 games; blind/untrained <=1/6) matches the literature mechanism exactly: search converts the king-PST delta into a reason; blind policies have neither term nor search, so the event is unrewarded - the sparse-event pattern HER diagnoses.

**scale_limits**
  > AZ-lineage systems learn castling's value from millions of games with deep search; at numpy scale with hundreds of self-play games and no king-safety term, neither representation alone nor outcome signal alone may suffice, and the literature supplies no effect size at this data budget. Engine-scale king-safety weights are tuned for deep search evals, not for a material+PST perceptron.

**small_scale_replication**
  > Partly checked - mechanism located by construction and the census is done. What remains is the fix-and-recount arm: add castling-rights planes and/or a shelter-based eval term at fixed self-play budget. That is 'not yet checked' (cheap), not a regime gap.

**confounds_flagged**
  > 3-6 games per level make castling rates extremely noisy; opening book / starting-position choices change castling opportunity; the 120-160 ply cap truncates promotions; 'real chess would warrant' is convention-dependent (master castling rates vary by era and repertoire); color balance and mirror positions in the small sample can skew per-level counts.

**suggested_local_test**
  > Two cheap arms at fixed self-play budget: (a) add 4 constant castling-rights planes to encode() plus a king-shelter term in evaluate(), then re-run the special-move census; (b) potential-based shaping with Phi = king-shelter score, so castling is rewarded only through consequences (per Ng et al. 1999). Readout: castling rate of blind/untrained levels vs the baseline <=1/6, plus a no-regression match against material 1-ply.

### Uncertain (skipped above)
- quantified_effect


## h18-horizon

### Register
**register_link**
  > H18 - 'At depth 2-3, L2's mistakes vs depth-5 are dominated by unresolved capture sequences, not positional misjudgement. If true, quiescence is where the remaining strength is; if false, more depth is.' Local status: CONTESTED, treated as OPEN locally. Principle settled (quiescence exists to blunt it; search.py already says so); magnitude at these depths unmeasured. Local quiescence measured roughly node-neutral, which says nothing about error composition (HYPOTHESES.md H18; 2026-10-03 review).

**register_implication**
  > Keep OPEN; no status change justified. The literature settles the principle (Berliner 1973; quiescence's raison d'etre) but contains no error-composition measurement at depth 2-3 with a material+PST eval, so the decisive evidence remains the local classification test. Consider adding one line to H18 noting that mature-engine node-share figures (50-90% in quiescence) do not transfer to depth 2-3, so the local 'roughly node-neutral' qsearch measurement is consistent with, not contrary to, the literature.

### Verdict
**literature_verdict**
  > partially supports - the direction of the claim is canonical: Berliner identified the horizon effect as the characteristic failure of fixed-depth search, and quiescence search exists precisely because leaf evaluation amid unresolved captures is the dominant error source of fixed-depth alpha-beta. But no published work measures the tactical-vs-positional composition of a shallow engine's disagreements with a deeper reference at these specific depths, so the magnitude claim is silent.

**scale_regime**
  > Principle established at engine-scale: hand-crafted eval, alpha-beta at 5-15 plies, from Berliner's 1970s analyses through mature open-source engines. Untested at toy/numpy-scale depth 2-3. At AZ/Lc0-scale the question largely dissolves: a learned value net plus MCTS has no fixed-depth horizon in the classical sense, so that lineage is silent on the claim.

### Evidence
**key_sources**
  - Berliner 1973, Some Necessary Conditions for a Master Chess Program (IJCAI-73) - names and analyzes the horizon effect: a fixed-depth program defers inevitable loss beyond its search depth by making concessions, mistaking postponement for avoidance; argues fixed-depth brute force cannot reach master level.
  - Berliner 1974, Chess as Problem Solving: The Development of a Tactics Analyzer (CMU PhD thesis) - the constructive follow-up: tactical (capture/check) sequences must be resolved to quiescence rather than truncated at a fixed horizon.
  - Shannon 1950, Programming a Computer for Playing Chess (Phil. Mag. 41) - type-A (full-width fixed-depth) vs type-B (selective, 'forceful variations' deep) strategies; the seed of quiescence: evaluation should be applied only at relatively quiescent positions.
  - Quiescence Search, Chess Programming Wiki - the standard remedy: extend leaves until no captures (and sometimes checks) are pending; standing-pat, SEE and delta pruning as the cost controls; documents the 50%-90% node share in practice.

**literature_findings**
  > Consensus, settled since the 1970s: a fixed-depth alpha-beta that evaluates every leaf statically blunders when the leaf is tactically unstable - material can be won or lost just beyond the horizon - and Berliner's horizon-effect analysis shows the failure is systematic (the search actively prefers lines that push loss past its depth, paying concessions to do so). The remedy, quiescence search, is universal in serious engines: keep expanding forcing moves (captures, sometimes checks) until the position is quiet, then evaluate. Shannon had already framed the principle in 1950 as the type-B strategy with evaluation restricted to quiescent positions. Where the literature is genuinely thin: (1) error composition - no study samples a shallow engine's disagreements with a deeper reference and classifies them tactical vs positional, which is exactly H18's local measurement; (2) magnitude at very shallow depths - quiescence costs and benefits are documented at engine depths and in engine-author lore, not at depth 2-3 with a minimal material+PST eval; (3) the AZ/MuZero lineage replaced static leaf eval with a learned value net, softening the horizon concept without measuring it. Active disagreements: none on the principle; practitioner debate concerns qsearch scope (captures only vs checks, pruning aggressiveness), not whether unresolved captures are the dominant fixed-depth error class.

**known_failure_modes**
  > Quiescence explosion: with many captures available, qsearch fails to terminate usefully, especially if checks are included; standing-pat plus SEE/delta pruning are the standard brakes, and pruning reintroduces a horizon inside qsearch. Reference bias: classifying 'mistakes' against a depth-5 reference inherits depth-5's own horizon at ply 5, so some 'positional' disagreements are the reference's tactical misses. Tactical and positional errors are entangled: a mis-evaluated quiet position can surface as what looks like a tactical miss one ply later, so any classification scheme needs a fixed operational criterion (e.g., capture legal within 2 plies of the divergence) and should report borderline rates. Node-neutrality measurements are sensitive to whether quiescence nodes are counted in the budget at all.

### Mapping
**mapping_to_local**
  > The local claim is exactly Berliner's principle instantiated at the smallest scale: fixed-depth alpha-beta's errors should concentrate where its static eval is least trustworthy, i.e., amid unresolved captures. The register already concedes the principle and asks for the magnitude, which the literature cannot supply - the local experiment (enumerate d2/d3-vs-d5 disagreements with diagnostics.compare_policies, classify by capture-near-divergence) is a measurement the field never published. The local node-neutral qsearch result matches the literature's cost picture at low depth (qsearch is cheap when few captures are pending) and is orthogonal to the error-composition question, so it neither supports nor contests H18.

**scale_limits**
  > All quantified quiescence lore (node shares, pruning thresholds, strength deltas) comes from engines searching 5+ plies with hand-crafted or NNUE evals; nothing published says how the tactical fraction of errors behaves at depth 2-3 with a perceptron-scale material+PST eval, where the eval's weakness may inflate the positional-error share relative to engine regimes. The AZ-lineage value-net results do not transfer: a learned eval absorbs some tactics statically, which is precisely the regime difference H18 probes.

**small_scale_replication**
  > 'Not yet checked', not a regime gap: the decisive classification is cheap and runnable today with existing tooling (diagnostics.compare_policies plus a capture-proximity classifier; no new model). The literature contributes the principle and the failure-mode list, not the number.

**confounds_flagged**
  > Depth-5 reference has its own horizon (reference bias). Disagreement sampling is confounded by which positions the two depths reach - opening book or fixed starting positions skew the mix toward tactical or quiet phases. Node accounting conventions (qnodes inside or outside the budget) silently drive 'node-neutral' claims. Draw and unfinished-game handling changes which disagreements ever surface in playouts. Move-ordering quality affects which disagreements appear first. At these depths the eval function is shared across arms, so eval-driven errors are common-mode and drop out of disagreement counts - biasing the measured mix toward search-driven (tactical) errors; the classifier must acknowledge this built-in tilt toward the claim.

**suggested_local_test**
  > Run the register's test: use diagnostics.compare_policies to enumerate all positions where L2 at depth 2 (and separately depth 3) disagrees with depth 5 over a fixed game set; classify each divergence as tactical if a capture is legal within 2 plies of the divergence point (extend to checks in a second pass), else positional; report the tactical fraction with a bootstrap CI. Decision rule: tactical fraction >= 2/3 confirms horizon dominance and prioritizes quiescence work; a near-even or positional majority means more depth or better eval is where the remaining strength is. Companion arm, cheap because local qsearch is node-neutral: play d2 vs d2+qsearch and d3 vs d3+qsearch at matched node budget (>=20 games/pair) - a qsearch strength gain is the strength-side corroboration that the disagreements were tactically driven.

### Uncertain (skipped above)
- matched_controls
- quantified_effect


## h19-nontransitivity

### Register
**register_link**
  > H19 (+H7) - 'Cycles exist, so no rating ordering is fully faithful (H7); the cycle runs through the greedy-material level specifically, because a shallow greedy evaluator exploits an informed-but-weak policy's uncertainty in a way a blind policy cannot (H19).' Local status: H7 SUPPORTED as recorded (largest residual material vs L3: observed 0.500, expected 0.259) but fragile - rests on L3's provisional rating and unexplained 3-3 splits vs all L2 depths. H19 OPEN (HYPOTHESES.md; 2026-10-03 review).

**register_implication**
  > Keep H7 SUPPORTED-with-fragility-note and H19 OPEN; no status change. The literature confirms non-transitivity is a real, measurable property of chess (peaking at intermediate skill) and that Elo/BT ratings are systematically unfaithful under cycles, which strengthens the register's framing - but it says nothing about cycles concentrating on a greedy-material player, so H19's structural claim remains locally decidable only. Consider one edit to HYPOTHESES.md: cite Sanjaya et al. 2022's peak at Elo 1300-1700 as the human-scale analogue of 'intermediate players are the hub of cycles'.

### Verdict
**literature_verdict**
  > partially supports - chess non-transitivity is confirmed empirically at billion-game human scale and Elo's failure under cycles is proven, supporting H7's general claim that no rating ordering is fully faithful; but the specific structural claim of H19 (cycles run through a shallow greedy-material player) is novel and untested anywhere in the literature.

**scale_regime**
  > Established at human population scale (Lichess/FICS, >1 billion games, Sanjaya et al. 2022) and at AI population scale (AlphaZero agents, StarCraft II, Hex; Czarnecki et al. 2020; Balduzzi et al. 2018). Untested at toy/numpy-scale round-robins of 4-8 weak agents - the local regime - though nothing in the theory excludes it; cycles are a property of the payoff matrix, not of strength.

### Evidence
**literature_findings**
  > Consensus: non-transitivity is a measurable, structural property of chess, not sampling noise. The spinning-top picture (Czarnecki et al. 2020) says a game's strategy space decomposes into a transitive strength axis and a cyclic bulk; across games the cyclic bulk is largest at intermediate skill - strong players converge on a few dominant strategies, weak players are uniformly dominated, and the middle is where style counter-picking thrives. Sanjaya et al. 2022 confirmed the same shape from real human chess data (first non-AI measurement), located the peak at Elo 1300-1700, and argued the peak coincides with where players plateau: to progress you must beat many mutually-countering styles. On evaluation, Balduzzi et al. 2018 showed Elo/BT-type ratings systematically misreport cyclic populations and can be manipulated by adding redundant agents, and proposed Nash averaging (max-entropy Nash equilibrium of the empirical payoff matrix) as the cycle-robust alternative; Nash clustering extends this to discover the cyclic components themselves. Active disagreements and limits: (1) how much of measured non-transitivity is real vs an artifact of matrix completion - Sanjaya et al. filled missing matchups with Elo-predicted scores, injecting transitivity by construction into sparse cells; (2) human data entangles strength with opponent choice (matchmaking), which AI round-robins avoid; (3) no published work tests whether cycles concentrate on a particular style of player (e.g., greedy material) - H19's hub claim is genuinely novel.

**matched_controls**
  > Sanjaya et al. 2022: observational but internally controlled - same rating pool, uniform monthly sampling, two-way (White/Black) matchups, robustness across years and across transitive-strength measures; the Elo-matrix completion step is a self-inflicted transitivity bias to note, not a matched-control failure. Czarnecki et al. 2020: full round-robin payoff matrices over trained agent populations - the design the local bench replicates at micro scale. Balduzzi et al. 2018: methodological analysis on given payoff matrices, controls not applicable.

**known_failure_modes**
  > Bradley-Terry/Elo non-identifiability on disconnected or sparse comparison graphs; residual significance without multiple-comparison correction reads noise as cycles, especially at 6-30 games per pair - the local H7 evidence (0.500 vs 0.259) rests on small samples and a provisional rating, exactly this hazard. Color asymmetry (White's first-move advantage) inflates apparent cycles if colors are not balanced per pair. Matrix completion with Elo-predicted scores (Sanjaya et al.) suppresses measured non-transitivity for missing cells - conclusions from sparse local matrices have the same gap. Nash averaging's max-entropy choice is a convention; different equilibria give different orderings. RPS-cycle counting via A^3 diagonal is sensitive to the win/loss threshold used to build the adjacency matrix.

### Mapping
**mapping_to_local**
  > H7 maps directly: the local residuals analysis is the micro-scale instance of what Balduzzi et al. formalized - a BT fit over a small round-robin, with large residuals flagging exactly the cycles the literature says to expect; the literature upgrades H7 from 'anomaly' to 'expected at intermediate skill'. H19 has no literature counterpart: no one has tested whether cycles in a mixed population concentrate on a shallow greedy-material player, and the mechanism (greedy evaluation exploiting an informed-but-weak policy's uncertainty) is a local hypothesis about style interaction that the spinning-top geometry is compatible with but silent on. The literature does sharpen H19's plausibility: the non-transitivity peak at intermediate skill matches the local roster, where material and the low depths are precisely the intermediate band.

**scale_limits**
  > Human-scale results (>1B games, 1500-bin matrices) and AI-population results (AZ, StarCraft) involve players far stronger and populations far larger than the local 4-8-agent roster; the spinning-top shape is a population-level statistic and says nothing about which agent in a tiny roster becomes a cycle hub. Nothing published measures non-transitivity among near-beginner-strength agents at all - the bottom of the top is unsampled even in the human data (ratings below ~1300 are sparse on Lichess/FICS).

**small_scale_replication**
  > 'Not yet checked', not a regime gap: the decisive test is the register's own - 30 games per pair round-robin, check whether the top-3 residuals all involve material. Nash averaging and Nash clustering can be computed on the local payoff matrix for free (a few lines of numpy), so the literature's tools transfer even where its numbers do not.

**confounds_flagged**
  > Small game counts per pair (6-30) make residual rankings unstable - the local H7 result already shifted with roster revisions. Provisional ratings (L3's) distort expected scores. Color balance per pair; opening/start-position choice concentrates matchups in tactical or quiet phases, biasing which pairs cycle. Unfinished-game handling (excluded vs counted as draws) changes residuals - interacts with H3. In the human studies, matchmaking and Elo-matrix completion are the silent drivers to check before importing any magnitude. Local roster composition (which levels exist) determines whether 'material is the hub' is a property of material or of the roster - a cycle hub can be manufactured by the choice of opponents.

**suggested_local_test**
  > Two-part, matching the register and adding the literature's tools. (1) H19 falsifier as registered: python -m bench --games 30 round-robin; if the top-3 BT residuals all involve material, the structural claim survives; if residuals scatter across pairs with no material/L3 concentration, H19 dies and H7 is noise-level. (2) Literature import: fit the payoff matrix with Nash averaging (Balduzzi 2018) alongside BT, run Nash clustering (Czarnecki 2020) on the local matrix, and bootstrap residuals (resample games within pairs) to attach CIs - this converts the fragile 0.500/0.259 edge into a statement with error bars and tests whether material remains a hub across both rating methods. Mechanism probe (optional): score material vs L2 at depths 1-5 separately; H19's mechanism predicts material's edge over informed-but-weak policies is largest at intermediate depth, not monotone.

### Uncertain (skipped above)
- key_sources
- quantified_effect


## h20-tactics-first

### Register
**register_link**
  > H20 - 'At naive scale, a 1-ply greedy material player sits mid-table because tactics dominate before positional understanding pays off.' Local status: SETTLED locally (material 255 Elo > L1 165, loses to every L2 depth) and historically; listed for completeness, do not investigate. The row is the load-bearing assumption behind search depth being on the ladder at all (HYPOTHESES.md H20; 2026-10-03 review).

**register_implication**
  > No change - keep SETTLED. The literature framing agrees with the local reading: material/tactics dominate at shallow depth and positional terms pay only after tactics are handled; the one nuance worth a single line in HYPOTHESES.md is the Ruoss 2024 scale-boundary - a big enough learned policy can absorb tactics statically, so H20 is a claim about the naive-scale regime, not about chess in principle.

### Verdict
**literature_verdict**
  > supports - both the historical record (brute-force type-A programs with dominant material terms beat knowledge-rich selective programs for four decades) and the modern stress test (Ruoss et al. 2024) agree that at low search/eval budget tactical-material correctness is the binding constraint, and positional sophistication pays only once tactics are secured.

**scale_regime**
  > Established from toy/numpy-scale through engine-scale historically (Shannon 1950 through the brute-force era, Chess 4.x to Deep Blue). The boundary moved only with Ruoss et al. 2024: at 270M parameters trained on ~15 billion Stockfish-16 action-values, a pure policy without search handles tactics at grandmaster level - but that regime (small-net >> naive scale, supervised by a search engine) is unreachable and irrelevant at numpy scale.

### Evidence
**literature_findings**
  > Consensus: at fixed shallow search and a simple evaluation, correctness on tactics - overwhelmingly material - is the binding constraint on strength, and positional terms yield returns only after tactical blunders are suppressed. Shannon posed the search-vs-knowledge question in 1950; the brute-force era answered it empirically: full-width searchers with material-dominated evaluations dominated knowledge-heavy selective programs from the 1970s through Deep Blue, and evaluation terms beyond material (king safety, pawn structure, mobility) were added only on top of already-tactically-sound search. Engine-development lore is unambiguous that material is the heaviest single eval term and that quiescence (tactics at the leaves) is prerequisite to any positional refinement paying. The modern nuance is Ruoss et al. 2024: given enough capacity and a search-engine teacher, a static policy can internalize tactics to grandmaster level, blurring the search-vs-evaluation division - but the same paper shows this fails at smaller scale, which restores H20's claim exactly in the regime the local bench occupies. Active disagreements: none on the naive-scale claim; the live frontier question is where the scale boundary sits (Ruoss's 'only at sufficient scale'), which is orthogonal to the local ladder.

**matched_controls**
  > Historical evidence is not matched-control: era comparisons entangle search depth with hardware, eval terms, and engineering quality - the brute-force victory is a consistent trend, not a controlled experiment. Ruoss et al. 2024 is the exception: model-size and dataset-size ablations at fixed recipe show the sharp scale dependence of searchless strength, a controlled demonstration that H20's claim holds below the boundary and fails above it.

**known_failure_modes**
  > Importing the historical trend without its regime: 'tactics first' is a budget-allocation statement, and at sufficient model scale the allocation flips (Ruoss) - quoting engine history against learned policies overstates it. Ruoss's model is supervised by Stockfish-16 action-values, so 'no search' means no search at inference; search is in the training loop - using it as evidence that search is dispensable smuggles in the teacher's search. Material-dominated mid-table placement is roster-dependent: the local 255 Elo is anchored by the other levels, not absolute. Human-pedagogy versions of the claim ('chess is 99% tactics', attributed to Teichmann) concern human learning and do not transfer to engines or RL agents.

### Mapping
**mapping_to_local**
  > The local measurement is the literature in miniature: material (255) beating L1 (165) and losing to every L2 depth is exactly the type-A story - one ply of tactical greed beats blind policy, and depth beats greed. The register's settled status is consistent with every source found; the only refinement the literature adds is the explicit scale boundary (Ruoss), which the register already notes as the row's raison d'etre ('why depth comes before cleverness'). No divergence to record.

**scale_limits**
  > Everything cited supports the claim at or above naive scale; nothing in the literature contradicts it at numpy scale either - Ruoss's counterexample requires ~10^8-10^9 more parameters and a search-engine teacher, regimes the local bench cannot reach. The literature cannot say where the boundary lies between naive scale and Ruoss scale (no intermediate data points), but that gap does not touch the local claim.

**small_scale_replication**
  > Nothing to replicate: the local result already exists (material 255 > L1 165, loses to all L2 depths) and matches the historical and modern framing. This is the settled case the fields.yaml calls 'verify the literature framing only; do not over-invest'.

**confounds_flagged**
  > Roster dependence: 'mid-table' is defined by which levels exist. Local Elo anchored arbitrarily; gaps (255 vs 165) are within-roster comparisons, not calibrated to human or CCRL scales. Historical era comparisons confound depth with hardware and engineering. Ruoss's numbers are blitz vs humans under specific time controls; importing the Elo number across pools is unsafe. Draw handling and the no-progress rule shape where a greedy-material player lands on the local ladder.

**suggested_local_test**
  > None - SETTLED; per the register, read the existing table. If the row is ever reopened, the minimal literature-implied check is an eval ablation at fixed depth: L2 with material-only eval vs L2 with material+PST at matched depth and node budget (>=20 games/pair); H20 predicts the PST delta is small relative to the material-vs-L1 and depth-vs-depth deltas - a large PST gain at depth 2 would be the surprising, register-worthy result.

### Uncertain (skipped above)
- key_sources
- quantified_effect


## h16-factored-action

### Register
**register_link**
  > H16 — 'P(from)*P(to|from)*P(promo) beats a flat 20480-way softmax at equal parameters and equal updates per legal move — statistical sharing, not capacity.' Local status: OPEN. Live blockers: parameter-budget mismatch (783/1445/7225 across arms) and untrained weights (2026-10-03 review).

**register_implication**
  > Keep OPEN. The register row in HYPOTHESES.md still reads 'blocked on L3-flat', which is stale (L3-flat now exists); edit the status line to name the live blockers — the 783/1445/7225 budget mismatch and untrained weights. The literature does not run the matched-parameter comparison, so the local experiment remains the decisive evidence and no verdict change is warranted.

### Verdict
**literature_verdict**
  > partially supports — shared/factorised move parameterisations are universal in strong game-playing networks, but no published work isolates statistical sharing from capacity at matched parameters and matched updates per legal move, which is exactly H16's condition.

**scale_regime**
  > Holds as folk wisdom at small-net (<=10M params) and AZ/Lc0-scale; weakens at the searchless 270M-param scale, where a flat 1968-way softmax (Ruoss et al. 2024) reaches grandmaster level without any factorisation. No published evidence exists at toy/numpy-scale (~10^3 params).

### Evidence
**key_sources**
  > 1) Silver et al. 2017, AlphaZero (arXiv:1712.01815) — 8x8x73 policy planes under one flat softmax over 4672 entries; the parameterisation is shared across squares by convolution, i.e. factorised by construction even though the distribution is flat. 2) Monroe & Lc0 team 2024, Mastering Chess with a Transformer Model (arXiv:2409.12272) — attention policy head with logit(move) = q(from).k(to)/sqrt(d), an explicit bilinear from/to factorisation with illegal moves masked; adopted without ablation against the planes head. 3) McIlroy-Young et al. 2020, Maia (arXiv:2006.01855) — the canonical policy-prior quality metric (top-1 accuracy ~53% with no search), using the AlphaZero-style 8x8x73 head. 4) Ruoss et al. 2024, Grandmaster-Level Chess Without Search (arXiv:2402.04494) — flat softmax over the 1968 UCI-indexed actions suffices at 270M params and 10M games; the scale boundary of the factorisation claim. 5) Rigaux & Kashima 2024, Enhancing Chess RL with Graph Representation (arXiv:2410.23753) — edge-featured GAT emits moves as edges, a generic factorised-by-graph output that beats the CNN at similar parameter counts.

**literature_findings**
  > Consensus: every strong published chess policy shares parameters across the move space instead of learning independent per-move logits. AlphaZero's 8x8x73 head is a single softmax, but its weights are conv filters over move-types shared across all from-squares — the factorisation lives in the parameterisation, not the distribution. The Lc0 transformer head makes the from x to factorisation explicit as a query-key dot product, on the stated grounds that it is simple and works; the paper contains no comparison against the planes head and never mentions it, so the field's move toward factorisation is an uncontrolled design adoption, not an ablated result. Maia fixes the evaluation metric (top-1 move-matching accuracy) that any such comparison should report. The counterpoint is Ruoss et al. 2024: at 270M parameters with 15.3B Stockfish annotations, a flat 1968-way softmax over UCI moves reaches ~2895 blitz, showing factorisation is not necessary given enough capacity and data — which is precisely H16's 'statistical sharing, not capacity' framing, read from the other end. Active disagreement: none in print; the field simply never runs the matched flat control, so the claimed advantage is asserted, not measured. The '20480 slots are too sparse to fill' defence of factorisation is an optimisation argument (update counts), never separated from expressiveness — the confound H16 names.

**matched_controls**
  > No — none of the cited comparisons is matched on parameters and updates per legal move. ChessFormer adopts the attention head with no planes-head ablation; Ruoss uses the flat head with no factorised ablation; AlphaZero was never compared to a parameter-matched flat head; Rigaux & Kashima match parameters but compare body architectures, not action factorisations. H16's exact control (equal parameters, equal updates per legal move) exists nowhere in the published record.

**known_failure_modes**
  > Product-of-factors softmax places mass on illegal from-to pairs, so masking must be applied per move after the product — and a masked renormalised product is no longer the independent product, which changes the gradient statistics the sharing argument relies on. Promotions do not fit the from x to schema (the ChessFormer paper does not describe promotion handling; AlphaZero needs 9 extra planes), so the promo factor is not interchangeable across designs. 'Equal updates per legal move' is non-standard: published nets update shared weights once per position, not per legal move, so published gradient and convergence statistics do not transfer to the local normalisation. Finally, tied move-type planes give untrained nets strong geometric biases at init (the H21 direction), which can masquerade as learned structure in short training runs.

### Mapping
**mapping_to_local**
  > The local claim narrows the literature: it accepts the universal sharing practice and asks the harder question the field skips — is the win statistical sharing or capacity? The local 20480 = 64 from x 64 to x 5 promo matches the AZ/Lc0 move granularity, so the local flat arm is a faithful local analogue of the published flat baseline nobody ran. The local arms at 783/1445/7225 parameters do not yet satisfy the claim's own matched-budget condition, which mirrors the literature's gap exactly: both the field and the local harness currently compare factorised vs flat at unmatched budgets.

**scale_limits**
  > All cited nets have >=10^6 parameters and train on >=10^5 positions; neither the AlphaZero planes result, the ChessFormer efficiency claims, nor the Ruoss flat-softmax result transfers to 783-7225 parameters updated by naive outcome RL. The 270M flat result bounds the claim from above — factorisation is at most a small-scale convenience — but says nothing about whether it is a real convenience at ~10^3 parameters.

**small_scale_replication**
  > This is 'not yet checked', not 'literature cannot say at this scale': the matched flat-vs-factorised comparison is cheap locally once weights are trained, and the literature's silence is due to nobody running the control, not to a regime barrier.

**confounds_flagged**
  > Published strength claims confound head design with body scale, data volume, and search in the loop (MCTS policy improvement hides prior weaknesses). Lc0 BT4 numbers are raw-policy but not head-ablated. Ruoss's flat result is confounded with 15.3B Stockfish action-value annotations (supervision, not RL) and with human-vs-bot rating miscalibration. Maia's ~53% top-1 is supervised human-move matching, not an RL outcome measure, so it calibrates the metric, not the training claim.

**suggested_local_test**
  > First fix the 783/1445/7225 mismatch (widen the factorised arm's per-square rows or narrow the flat head) so both arms sit at equal parameters. Then train both to equal updates per legal move — not per position — and report (a) top-1 agreement against a depth-4 search oracle, the Maia metric, and (b) bench Elo +/- SE at >=20 games/pair. Falsifier per the register: the flat model at equal parameters and equal updates per legal move matching the factorised model. Report parameter counts and update counts alongside the result, since the literature shows this is the comparison nobody else publishes.

### Uncertain (skipped above)
- quantified_effect


## h08-weight-tying

### Register
**register_link**
  > H8 — 'A weight-tied per-square function matches the per-square-weights model at comparable budget.' Local status: IN PROGRESS. Untrained null recorded (11/12 games unfinished — says nothing). Budget confound: the tied arm has 10.8% of baseline parameters (783 vs 7225), so a loss cannot be read as 'tying is worse' (2026-10-03 review; HYPOTHESES.md H8).

**register_implication**
  > Keep IN PROGRESS; no verdict change. The literature strongly favours tying as a default but is silent at the ~10^3-parameter regime and never runs tied-vs-untied at matched tiny budgets, so the local experiment is still the decisive one. Consider adding one line to H8 noting NNUE as the structured-partial-untying counterexample (per-king-slice input layer) so the register records where untying demonstrably pays.

### Verdict
**literature_verdict**
  > supports — weight sharing across board positions is the most established inductive bias in game-playing networks: every strong published chess net ties massively, and the one matched-parameter comparison (GNN vs CNN) has the more-tied model winning. The support is structural, not from controlled untied baselines.

**scale_regime**
  > Established at small-net (<=10M params) and AZ/Lc0-scale; untested at toy/numpy-scale (~10^3 params). At engine scale, NNUE shows structured partial untying (a separate input slice per king position) is worth its parameter cost — the boundary of the tying claim.

### Evidence
**key_sources**
  > 1) LeCun et al. 1989, Backpropagation Applied to Handwritten Zip Code Recognition — origin of weight sharing: shared kernels enforce translation equivariance and cut free parameters while improving generalisation at fixed data. 2) LeCun et al. 1998, Gradient-Based Learning Applied to Document Recognition (Proc. IEEE) — the canonical statement that sharing is a geometric/variance prior, not a capacity argument. 3) Silver et al. 2017, AlphaZero (arXiv:1712.01815) — shared conv trunk and conv policy head; no per-square weights anywhere in the state of the art. 4) Rigaux & Kashima 2024 (arXiv:2410.23753) — GAT shares node and edge functions across all squares (stronger, permutation-equivariant tying than a CNN) and beats the CNN at similar parameter counts. 5) Monroe & Lc0 team 2024 (arXiv:2409.12272) — attention shares Q/K projections across all 64 square tokens; position information is re-injected through shared relative encodings rather than untied weights. 6) Nasu 2018, NNUE (github.com/ynasu87/nnue) — the counterexample nuance: the input layer is deliberately untied across king slices ('heavily overparametrized for all king placements'), affordable only because the differential accumulator makes it cheap to evaluate.

**literature_findings**
  > Consensus: at matched budgets, translation-equivariant sharing wins because the same local features are useful on every square, while an untied per-square weight row sees training signal only for positions where that square matters — the statistical-sharing argument H8 makes locally, stated by LeCun in 1989 for images. The GNN result pushes the direction further: even the grid topology assumption can be dropped, and the more aggressively tied model still wins at matched parameters. The literature's answer to the known weakness of tying (a tied function cannot express square-indexed facts — the H10 direction: knight-on-rim needs square identity) is not to untie but to re-inject position as input: ChessFormer adds relative position encodings, NNUE unties structurally by king slice while keeping in-slice sharing. No published work shows untied per-square weights winning at matched budget on a board game. Active disagreement: none on tying itself; the open axis is how much position information to re-inject and how (absolute vs relative encodings, NNUE-style slices), which is a question about features, not about whether to share weights.

**known_failure_modes**
  > Tied models cannot express square-indexed facts — locally concrete as H10: no destination-file feature and abs(tf-ff) make b1->c3 and g1->h3 byte-identical — but this is a missing-feature failure, separable from tying itself; importing 'tying wins' without checking the feature set imports the confound. Tying also couples updates: one gradient step changes behaviour on all squares, so a per-move learning-rate rule that silently zeroes the shared head (the local lr/64 bug, fixed by error feedback) is the classic import hazard. At extreme tying (permutation equivariance) the net is blind to square identity unless position is re-injected as input. NNUE's untying works only because the incremental accumulator exists; without differential updates the same design is computationally infeasible, so 'untying pays there' does not travel to architectures without it.

### Mapping
**mapping_to_local**
  > The local claim matches the literature direction precisely: tied L3.5 is the CNN/GNN-side arm, and L3's per-square W_to_dst rows are the untied baseline the field would not bother to build — the local experiment sharpens the question by adding the matched-budget control the literature skips. But as currently wired the local comparison tests 'tying at 9x less budget' (783 vs 7225 parameters), not 'tying at matched budget', so the local harness currently replicates the literature's characteristic confound rather than resolving it.

**scale_limits**
  > Published tying results live at >=10^5 parameters with convolution or attention compute and large supervised or self-play corpora; nothing published says whether sharing still wins when the entire model has ~10^3 weights and each per-square row is a few dozen parameters trained by naive outcome RL. NNUE's success is at small compute, not small parameter count, so it does not transfer either — in both directions the numpy regime is unsampled.

**small_scale_replication**
  > 'Not yet checked', not a regime gap: a budget-matched tied-vs-untied run at ~10^3 parameters is cheap locally once the trained budget-matched arm exists (the H11/Task #9 training workstream). The literature contributes direction and the position-reinjection caveat, not the number.

**confounds_flagged**
  > CNN/GNN comparisons confound tying with receptive field and depth. ChessFormer's position-encoding ablation shows the encoding, not the tying, can be the dominant term — a tying result and an encoding result are easy to conflate. NNUE's per-king-slice untying is confounded with its incremental-update engineering motive. All published nets train with search in the loop or massive supervised data, not naive outcome RL, so their gradient statistics do not transfer to the local trainer.

**suggested_local_test**
  > Build the budget-matched tied arm — pad L3.5's shared scorer to ~7225 parameters (wider shared rows or a second shared layer) — and after training run L3 vs L3.5 at >=20 games/pair. Decision rule per the register: L3.5 within 1 SE of L3 at matched budget vindicates tying and licenses spending the freed parameters elsewhere; a clear loss at matched budget — not at the current 10.8% budget — is the informative negative, and would locally contradict the literature's default for the first time at this scale. Report both arms' parameter counts and updates per position, and check the tied arm actually varies across inputs after training before interpreting any null.

### Uncertain (skipped above)
- matched_controls
- quantified_effect


## h09-budget-vs-topology

### Register
**register_link**
  > H9 — 'A flat MLP at matched parameter count does not beat the factorised models; budget, not topology, dominates at this scale.' Local status: IN PROGRESS. Control arm verified mechanically (L3-flat origin head exactly constant across squares, std = 0.0 — it gives up spatial structure entirely). Strength question not established (2026-10-03 review; HYPOTHESES.md H9).

**register_implication**
  > Keep IN PROGRESS; no verdict change. The literature is two-sided for this claim — scaling laws and the searchless-chess model ladder side with budget, while the only matched-parameter game comparison (GNN vs CNN) and NNUE side with bias — and it is silent at ~10^3 parameters, so the local result will be new evidence either way. Optionally add to H9's row that the literature's direction priors conflict, so a clean local result in either direction is quotable against a named source.

### Verdict
**literature_verdict**
  > mixed — above ~10^6 parameters loss is a smooth function of budget and topology shifts constants rather than exponents (supports 'budget dominates'); in the only matched-parameter game-playing comparisons, inductive bias adds large gains on top of budget (contests it). At ~10^3 parameters the literature has no measurement at all.

**scale_regime**
  > The verdict flips across regimes. Budget-dominates holds in the Kaplan/Hoffmann regime (~10^6-10^10+ parameters, language modelling) and across Ruoss's 9M->136M->270M ladder at fixed data. Topology-matters holds in matched-parameter game-net comparisons (Rigaux & Kashima GNN vs CNN; ChessFormer position encodings) and at engine scale (NNUE). Toy/numpy-scale: no published verdict exists.

### Evidence
**key_sources**
  > 1) Kaplan et al. 2020, Scaling Laws for Neural Language Models (arXiv:2001.08361) — loss is a power law in parameters, data, and compute with no topology term; the strongest statement of 'budget dominates', in a regime starting far above local scale. 2) Hoffmann et al. 2022, Training Compute-Optimal Large Language Models (arXiv:2203.15556) — compute-optimal budget allocation between parameters and data; architecture details shift constants, not the scaling relation. 3) Rigaux & Kashima 2024, Enhancing Chess RL with Graph Representation (arXiv:2410.23753) — the direct counterexample: at similar parameter counts the GNN beats the CNN and gains strength an order of magnitude faster; a measurable topology term at small-net scale. 4) Ruoss et al. 2024, Grandmaster-Level Chess Without Search (arXiv:2402.04494) — the cleanest budget ladder in chess at fixed data (9M/136M/270M), plus the caveat that their ConvNet baseline lost to the Transformer but was untuned, i.e. not a matched topology comparison. 5) Nasu 2018, NNUE (github.com/ynasu87/nnue) — at engine scale a hand-picked king-relative sparse bias with incremental updates displaced both handcrafted eval and larger general networks at roughly fixed compute: the bias, not the budget, was the contribution.

**literature_findings**
  > The scaling-law side: for language modelling and for the searchless-chess ladder, once data is fixed, parameters buy strength smoothly and predictably, and architecture changes within a working family shift constants rather than exponents — the formal version of 'budget dominates'. But every published matched-parameter comparison in games lands on the other side: Rigaux & Kashima hold parameter count fixed and find a large topology term (GNN over CNN); ChessFormer holds the 6M body fixed and finds position encoding — a pure prior choice — dominating parameter doublings; NNUE holds compute roughly fixed and finds a hand-designed bias decisive at engine scale. The synthesis the literature implies: budget dominates given a sufficient inductive bias; below some bias threshold, parameters are wasted. A flat MLP at matched budget is precisely the proposed test of that threshold — and no published game study fills that cell. Active disagreement: scaling-law universality is contested for structured/game domains, and Chinchilla-style allocation has never been validated below ~10^6 parameters; Ruoss also shows the budget direction is data-dependent (models >=7M overfit a 10K-game set, and the effect disappears at 100K-1M games), so even the sign of the budget effect can flip at tiny data.

**matched_controls**
  > Split. Rigaux & Kashima explicitly match parameter count (GNN vs CNN) and find a topology effect — the only cited study controlling the variable H9 holds fixed. Ruoss's ladder matches data and varies budget, finding a budget effect — but fixes the action space and architecture family throughout. Neither study crosses the other's control, and nothing varies both at ~10^3 parameters. NNUE vs handcrafted eval is matched on neither parameters, data, nor training pipeline. The local design (vary topology, hold budget) is exactly the control structure the literature uses once — and that once found topology mattering.

**known_failure_modes**
  > Scaling-law imports carry the regime caveat: power laws fitted at 10^6-10^10 parameters have no warrant at 10^3 parameters, and Chinchilla allocation assumes data and parameters are both freely scalable, which numpy-scale RL violates. Flat-MLP controls can fail silently by symmetry — the local std = 0.0 origin head is exactly this: the control must be checked to vary something, else 'no difference' is uninformative and 'worse' is vacuous. The budget effect inverts by overfitting at tiny data (Ruoss: >=7M parameters overfit 10K games), so 'budget dominates' is a data-conditional statement. Finally, topology comparisons confound sample efficiency with asymptotic strength: Rigaux & Kashima's 'faster' mixes the two, and a control matched on updates can still be unmatched on wall-clock or vice versa.

### Mapping
**mapping_to_local**
  > The local claim is the small-scale conjunction the literature leaves open: scaling laws supply the supportive prior (budget should dominate), the GNN and NNUE results supply the hostile prior (bias can dominate budget), and the local flat-MLP arm instantiates precisely the 'budget without bias' cell no published game study fills. The mechanically verified constant origin head (std = 0.0) makes the local control cleaner than any published flat baseline — it gives up spatial structure entirely, so a local result in either direction is a real measurement of the bias term, not of a partially-working control.

**scale_limits**
  > Everything cited runs at >=10^6 parameters and >=10^5 games (NNUE excepted on compute, not on parameter count). At ~10^3 parameters trained by naive outcome RL, gradient noise — not loss-surface topology and not capacity — may be the binding constraint, a regime none of the cited work samples and none of its fitted laws covers.

**small_scale_replication**
  > Pure 'not yet checked': the local experiment is the replication, and there is no regime gap to hide behind — the literature's two direction priors conflict precisely at this scale, which is what makes the local run worth doing. The only import the literature offers is the checklist of controls (matched budget, matched updates, verified-varying control).

**confounds_flagged**
  > Ruoss's budget ladder is confounded with 15.3B Stockfish action-value annotations (supervision, not RL) and a fixed flat action space. NNUE comparisons confound bias with search integration and training pipeline. The GNN-vs-CNN result is confounded with its training-speed framing ('faster' mixes sample efficiency and wall-clock). Scaling-law exponents are fitted on next-token LM loss, not Elo. And any local null on untrained weights is a statement about the harness (as the 11/12-unfinished first run showed), not about budget or topology.

**suggested_local_test**
  > After the trained budget-matched arm exists (the H8 fix), run L3 vs L3.5 vs L3-flat at >=20 games/pair, reporting parameters, updates per position, and top-1 agreement against a depth-4 oracle for each arm. Decision rule: L3-flat matching the factorised arms at equal budget kills the topology side locally and supports 'budget dominates' — and in passing falsifies H16's 'sharing, not capacity' framing; a clean L3-flat loss confirms bias-over-budget at ~10^3 parameters, locally replicating the Rigaux & Kashima direction at three orders of magnitude below their scale. Before interpreting any null, verify the trained control's origin head varies (std > 0), repeating the mechanical check that validated the untrained arm.

### Uncertain (skipped above)
- quantified_effect


## h10-relational-limits

### Register
**register_link**
  > H10 - claim: 'A per-square function cannot express facts irreducibly about several squares (open file, knight-on-rim); weight tying must lose some strength there.' Local status (HYPOTHESES.md / 2026-10-03 review): MECHANISM LOCATED by construction - _geom_features carries no destination-file term and abs(tf-ff) makes b1->c3 and g1->h3 byte-identical; strength cost unmeasured; caveat recorded: a missing-feature problem, not proven a tying problem.

**register_implication**
  > Keep status [~] but upgrade framing: the literature converges that per-object/per-square functions cannot express relational facts without cross-square machinery, so the located mechanism is the expected failure, not a surprising one. Suggested one-line edit to HYPOTHESES.md: 'Mechanism located; literature (Santoro 2017, Rigaux & Kashima 2024, Giraffe 2015) treats per-square inexpressibility of relational facts as settled; the open local question is narrowed to (a) the strength cost and (b) tying-vs-missing-feature, both untested.'

### Verdict
**literature_verdict**
  > partially supports - literature agrees per-square/per-object functions cannot express irreducibly relational facts and that working chess systems always carry cross-square machinery; it does not isolate weight tying from missing features, which is exactly the local caveat.

**scale_regime**
  > Supports at small-net (<=10M params: AlphaGateau GNN vs CNN) and engine scale (Giraffe's global/square-centric feature modalities; handcrafted engine evals with pawn-structure and king-safety terms). Silent at toy/numpy scale.

### Evidence
**key_sources**
  - Santoro et al. 2017, 'A simple neural network module for relational reasoning' (arXiv:1706.01427) - canonical proof that per-object feedforward structure fails relational tasks; the RN computes pairwise g_theta(o_i, o_j), i.e. irreducibly two-object facts, which is the formal version of 'open file is a fact about seven other squares'.
  - Rigaux & Kashima 2024, 'Enhancing Chess Reinforcement Learning with Graph Representation' / AlphaGateau (arXiv:2410.23753, NeurIPS 2024) - direct small-scale chess test: edge-feature GNN beats grid CNN at similar parameter counts and learns an order of magnitude faster; chess board modelled as a graph with relational edges, not per-square channels.
  - Lai 2015, Giraffe (arXiv:1509.01549) - the strongest pre-NNUE learned eval deliberately added square-centric (attack/defend maps) and position-centric (global) modalities alongside piece-centric features; restricted first-layer connectivity kept the modalities separate, an explicit architectural admission that per-piece/per-square views are insufficient alone.
  - Silver et al. 2018, AlphaZero (Science) - counterpoint boundary: whole-board convolutional trunk still aggregates cross-square information every layer, i.e. even the 'grid' baseline is not per-square in the H10 sense; H10's target (strictly per-square tied function) is weaker than any published chess learner.

**literature_findings**
  > Three independent literatures say the same thing. (1) Relational reasoning: Santoro et al. showed that CNN/MLP baselines which process objects independently fail CLEVR-style relational questions, while a module computing pairwise relations solves them; the RN's contribution is precisely the irreducibly multi-object term g(o_i, o_j). 'Open file' (a fact about 8 squares), 'knight on the rim' (a fact about a piece relative to board topology), and 'weak square complex' are exactly this class of fact. (2) Chess RL at small scale: AlphaGateau replaces the grid with a graph whose edges carry relational features (attack, defense, ray relations) and reports both a win over parameter-matched CNNs and an order-of-magnitude training speedup at sub-AlphaZero sizes - the closest published analogue to the local L3.5 question, and it comes down on the side that cross-square structure must be representable. (3) Learned evaluation: Giraffe's authors found piece-centric features alone performed only 'at a reasonable level' and added square-centric attack/defend maps plus global features (side to move, castling rights, material configuration) as separate modalities; every handcrafted engine eval likewise carries global terms (pawn structure, king safety, space). Consensus: no serious chess system relies on strictly per-square functions. Active disagreement: none on the mechanism; the open empirical question is the *size* of the strength loss at tiny scale, which nobody has measured. Critically for the local caveat, none of these sources separates 'the feature set lacks the relational channel' from 'weight tying prevents expressing it' - published systems fix both at once by adding relational structure.

**matched_controls**
  > AlphaGateau claims similar parameter counts between GNN and CNN arms (matched-budget, the one published chess comparison close to the H8/H10 protocol), but not matched training compute per arm in the ablation sense, and edge features are hand-supplied domain knowledge. RN-vs-baseline comparisons are task-matched, not architecture-matched. No published study runs the local experiment: same features, same budget, tied vs untied, relational probes isolated.

**known_failure_modes**
  > RN/GNN pairwise computation is O(n^2) in objects (4096 square pairs for chess) - fine here but the cited efficiency claims do not transfer to larger boards. AlphaGateau's edge features are hand-engineered domain knowledge, so 'graph beats grid' partly credits the feature engineer, not the architecture; the same confound attaches to any local fix that adds a destination-file feature. Giraffe's global features were hand-crafted inputs: the result shows learned evals *need* global information, not that the network discovers it. Finally, inexpressibility results (RN-style) are about representational capacity and say nothing about whether the missing facts matter at a given playing level - the strength cost is always an empirical question.

### Mapping
**mapping_to_local**
  > Agreement is unusually direct: the local mechanism (no destination-file term; abs(tf-ff) making b1->c3 and g1->h3 byte-identical; tied destination row origin-blind) is an instance of the per-object limitation Santoro formalized, and AlphaGateau's matched-budget GNN win predicts L3.5 should lose strength specifically on global-fact positions. Divergence: the literature's remedy is always to add relational machinery (edges, global features), which the local protocol forbids because it would change the fixed feature set the comparison rests on - the local experiment is therefore testing a purer (and more artificial) version of the claim than any published system embodies.

**scale_limits**
  > All cited chess systems are >=100K parameters with training compute orders of magnitude above the local harness; AlphaGateau itself uses an AlphaZero-style MCTS+RL pipeline. Nothing in the literature prices the strength cost of per-square inexpressibility at 783-7225 parameters trained (if at all) by naive self-play - both the mechanism and the cost could behave differently when the model is too small to have learned the relational facts anyway.

**small_scale_replication**
  > Not yet checked, and cheap: the register already specifies the experiment (~50 curated positions where the best move hinges on one named global feature; compare L3 vs L3.5 top-1 agreement against a depth-4 oracle). This is a 'not yet checked' gap, not a regime gap.

**confounds_flagged**
  > AlphaGateau's pipeline (MCTS, self-play RL, edge features) is so far above the local harness that 'GNN beats CNN' may not transfer down; its edge features inject the relational knowledge by hand, which is precisely the thing being tested. Giraffe/NNUE-era eval comparisons conflate eval quality with search depth and node speed. Locally: using depth-4 search as oracle imports the oracle's own horizon errors into the 'global fact' labels, and position-set curation (who decides the best move is 'determined by' an open file) is a subjective label channel.

**suggested_local_test**
  > Two arms, as the register implies: (1) curated global-fact probe set (open file, passed pawn, weak complex, king safety, knight-rim), top-1 agreement of L3 vs L3.5 with a depth-4 oracle - measures the strength cost of the located mechanism; (2) a patched arm adding a destination-file geometry feature (breaking abs(tf-ff) symmetry) at matched parameter count - separates the missing-feature explanation from the tying explanation, the distinction the literature cannot make.

### Uncertain (skipped above)
- quantified_effect


## h11-learned-leaf-eval

### Register
**register_link**
  > H11 - claim: 'L5 with use_model_eval and a trained value head outperforms use_model_eval=False (the material+PST heuristic leaf eval).' Local status (HYPOTHESES.md / 2026-10-03 review): BLOCKED on training (Task #9). Untrained baseline NEGATIVE: +eval worst (-102 Elo delta), every standard error exceeds every delta; eval-without-prior also ~4x slower.

**register_implication**
  > Keep [-] blocked locally, but annotate the register: at engine scale this question is SETTLED in the claim's favour (NNUE's +~130 Elo under identical search; Giraffe's parity with handcrafted evals), so H11 is no longer 'will a trained eval win' but 'does the win survive at numpy scale with naive outcome-RL training'. Suggested one-line edit: 'Engine-scale verdict settled positive (NNUE, Giraffe); local question is the training-budget floor, not the direction.'

### Verdict
**literature_verdict**
  > supports - every chess-learning system that completed training, from KnightCap's linear eval to NNUE, replaced or beat the handcrafted leaf eval; the local untrained negative is consistent with the literature (nobody reports an untrained value head helping), but the literature is silent at numpy scale.

**scale_regime**
  > Supports at small-net scale (KnightCap linear eval ~tens of thousands of weights; Giraffe ~4-layer MLP) and at engine/AZ scale (NNUE, AlphaZero, MuZero). Silent at toy/numpy scale: the cheapest published success still used thousands of FICS games or ~175M self-play positions.

### Evidence
**key_sources**
  - Thrun 1995, NeuroChess (NIPS 7) - first neural board eval learned from game outcomes (TD + explanation-based learning); demonstrated feasibility but only weak club strength, establishing the eval-quality-vs-search-depth tradeoff the whole lineage struggles with.
  - Baxter, Tridgell, Weaver 2000, 'Learning to Play Chess using Temporal Differences' / TD-Leaf (arXiv:cs/9901001) and KnightCap (arXiv:cs/9901002) - the cleanest matched evidence for H11: same engine, same search, only the eval trained; +500 Elo in 308 FICS games from outcome-based TD-Leaf alone.
  - Lai 2015, Giraffe (arXiv:1509.01549) - learned eval from scratch (material-only bootstrap, TD-Leaf self-play, ~175M positions) reaching parity with handcrafted evals of top engines; explicit that the NN eval is slower per node and must pay for its depth loss.
  - Silver et al. 2017/2018, AlphaZero - value head replacing handcrafted eval entirely at scale; sets the ceiling but its 44M-game self-play budget defines the regime boundary the local harness cannot approach.
  - Schrittwieser et al. 2020, MuZero (arXiv:1911.08265) - matched AlphaZero on chess with a learned model and value head, confirming the value-head-as-leaf-eval design is not AZ-specific.
  - Nasu 2018 / Stockfish 2020, NNUE - the decisive engine-scale data point: supervised-from-search learned eval, no MCTS-scale RL, +~130 Elo under identical search; the cheapest training signal that has ever won this comparison.

**literature_findings**
  > The direction of H11 is one of the most consistently confirmed claims in computer chess: whenever training completes, a learned leaf eval beats material-plus-heuristic evals, usually by hundreds of Elo. KnightCap is the canonical matched comparison (identical search, trained vs untrained/tuned eval) and also documents the prerequisites: TD-Leaf (backpropagating through the minimax search to leaf values, not just game outcomes), an opponent pool (FICS) rather than pure self-play, and thousands of games. Giraffe established that the learned eval can match human-engineered evals feature-for-feature, but its own tradeoff analysis is the cautionary finding for H11: a better-but-slower eval can lose strength overall because it searches fewer nodes - directly analogous to the local 'eval-without-prior ~4x slower' observation. NeuroChess is the lineage's documented partial failure: learning from outcomes alone produced only weak-club play, which is why every successor added structure (TD-Leaf's search-level targets, Giraffe's feature design, NNUE's search-derived labels, AZ's MCTS policy improvement). AlphaZero/MuZero settle the question at the high-budget end but are regime markers, not methods, for the local harness. NNUE settled it at engine scale with the cheapest signal: supervised training on positions labelled by the engine's own shallow search - no outcome RL, no MCTS. Consensus: a trained eval wins. Active disagreement: how cheap the training signal can be (outcome RL vs search-derived labels vs imitation), which is precisely the axis the local harness sits at the extreme cheap end of. Nobody has published a trained value head at ~10^3-10^4 parameters with naive numpy self-play; the local untrained negative result (untrained +eval worst, -102) matches the universal implicit finding that an untrained or barely-trained head is harmful, not neutral.

**matched_controls**
  > KnightCap: matched - same program and search before/after training; the strongest small-scale control in the lineage (pool-rating caveat aside). Stockfish 11 vs 12 (NNUE): matched - same search framework, eval swapped; the strongest engine-scale control. Giraffe vs other engines: NOT matched (search depth, node speed, and eval all differ; the paper acknowledges this). AlphaZero vs Stockfish: NOT matched (hardware, search type, time controls contested). For the local claim the transferable matched pairs are KnightCap (small) and NNUE (large); both support H11.

**known_failure_modes**
  > TD-Leaf can drift or collapse without anchored outcomes and enough game diversity; KnightCap's 2150 is a FICS pool rating, vulnerable to pool drift and opponent selection. Better-eval-slower-search tradeoff: eval gains can be eaten by lost depth (Giraffe, KnightCap both report this) - a value head must be cheap relative to the heuristic it replaces, which is exactly where the local +eval arm already shows a 4x slowdown. Untrained or undertrained value heads are actively harmful as leaf evals (consistent with the local -102 delta). AZ-style value heads are only useful inside MCTS-style search with a policy prior; a bare value head at depth 1-2 is a different (weaker) device. NNUE-style supervised-from-search training inherits the teacher engine's blind spots and requires periodic re-distillation.

### Mapping
**mapping_to_local**
  > The literature agrees with the local claim's direction so strongly that the local result, once training exists, would be surprising only if negative. The narrowing the literature forces: (1) 'trained' must mean a real training run with a search-aware or search-derived signal - KnightCap is the floor, and it still used TD-Leaf through its own search plus a live opponent pool; (2) the eval must be speed-competitive - the local 4x slowdown is a first-class threat to the claim even if accuracy improves, per Giraffe's explicit tradeoff; (3) the untrained baseline negative is expected, not informative - consistent with the register's own instruction not to rerun the untrained ablation.

**scale_limits**
  > Nothing published operates near the local regime: ~10^3-10^4 parameters, numpy, naive outcome-RL self-play, depth<=5 search, no opponent pool, no search-derived labels. The cheapest literature success (KnightCap) is roughly 3+ orders of magnitude more training signal, and its eval was linear over hand-designed features rather than a learned head. The literature cannot say whether any training budget at this scale suffices, nor whether outcome-RL (the local signal, Group G) can substitute for TD-Leaf/search-derived labels.

**small_scale_replication**
  > Regime gap, not 'not yet checked': the blocker is Task #9 (training infrastructure), and the literature says the cheap version of the experiment still requires a search-aware training signal the harness does not yet produce. No published numpy-scale replication exists to shortcut it.

**confounds_flagged**
  > FICS pool ratings (KnightCap) drift and are not engine-normalized. Giraffe and AZ comparisons entangle eval quality with node speed, search algorithm, and hardware. NNUE's +130 Elo is self-play against its own predecessor and conflates net architecture with two years of training-data pipeline work; the net was also trained on labels from the same engine family (self-distillation). For the local harness the literature imports two silent assumptions: an opponent pool (KnightCap, AZ) and search-level training targets (TD-Leaf, NNUE labels) - naive single-copy self-play outcome RL has neither, and the literature offers no clean success under exactly those conditions.

**suggested_local_test**
  > After Task #9: rerun 'python -m bench --roster ablation --games 20' with the trained value head vs use_model_eval=False, reporting node/sec alongside Elo so the Giraffe tradeoff is visible. The literature implies a cheaper, higher-prior arm first: label MidstateStore positions with the depth-5 search value (NNUE-style teacher labels; the repo already wires train_on_search_feedback to depth-5) and fit the value head supervised - the 'learned eval without MCTS-scale training' lesson - before attempting pure outcome-RL training, which the literature (NeuroChess) flags as the weakest signal in the lineage.

### Uncertain (skipped above)
- quantified_effect


## nnue

### Register
**register_link**
  > No register row - supplement item informing H11 (learned leaf eval), H8 (weight tying), H9 (budget vs topology). Claim: 'Efficiently updatable king-relative shared features with differential accumulator updates displaced handcrafted eval in top engines - learned eval without MCTS-scale training.' Local status: not in the register; flagged by the 2026-10-04 outline supplement as the biggest gap in the H11 lineage.

**register_implication**
  > Add a one-line cross-reference under H11 (not a new row): 'Engine-scale SETTLED positive - NNUE (Nasu 2018; Stockfish 12, 2020, +~130 Elo under identical search) is the existence proof that a learned leaf eval beats handcrafted eval without MCTS-scale training; training signal was supervised-from-search, which the repo can imitate cheaply via depth-5 labels (train_on_search_feedback).' H8/H9 get the same one-liner: NNUE is massive weight tying (king-relative shared first layer) winning at the highest level, i.e. tying wins when the features are right.

### Verdict
**literature_verdict**
  > supports - NNUE's displacement of handcrafted eval in essentially all top engines since 2020 is the canonical demonstration of the claim; it is the strongest single data point in the H11 lineage and it used supervised training, not MCTS-scale RL.

**scale_regime**
  > Engine-scale (NNUE + deep alpha-beta search, CPU SIMD, millions of evals/sec/thread). Silent at toy/numpy scale: the speed mechanisms (incremental accumulator, int8/int16 quantization) presuppose a deep search tree and huge eval counts the local harness does not have.

### Evidence
**key_sources**
  - Nasu 2018, NNUE original (github.com/ynasu87/nnue; integrated into the YaneuraOu shogi engine by Motohiro Isozaki, May 2018) - invention of the efficiently-updatable architecture: sparse king-relative inputs, shared first-layer weights, differential accumulator updates on make/unmake move.
  - Noda 2019, Stockfish port - NNUE adapted from shogi to chess (June 2019), leading to the Stockfish 12 release (Sept 2020), the +~130 Elo event that displaced handcrafted eval in top engines.
  - Stockfish NNUE documentation (nnue-pytorch docs/nnue.md, mirrored on the Pikafish wiki) - the canonical technical reference: three principles (few non-zero inputs, minimal input change between consecutive evals, integer-quantized inference), sparse linear layers, ClippedReLU, shallow 2-4 layer design, accumulator calculus.
  - Lai 2015, Giraffe (arXiv:1509.01549) - conceptual predecessor: learned eval at parity with handcrafted evals but too slow per node; NNUE is the engineering answer to Giraffe's eval-quality-vs-node-speed tradeoff.
  - Silver et al. 2018, AlphaZero - the regime NNUE explicitly undercuts: NNUE matched the 'learned eval wins' conclusion with supervised training on search labels, at a small fraction of AZ's self-play budget.

**literature_findings**
  > NNUE is the resolution of the eval-quality-vs-search-speed tension that runs through the whole H11 lineage (NeuroChess -> KnightCap -> Giraffe). Its three design principles are documented in the Stockfish NNUE reference: (1) the network must have relatively few non-zero inputs - extreme sparsity (~0.1%) caps the cost of a full first-layer evaluation even when the layer is huge; (2) inputs should change as little as possible between consecutive evaluations - a single move alters only a few king-relative features, so the expensive first layer (the accumulator) is updated differentially on make/unmake move instead of recomputed; (3) the network must be simple enough for low-precision integer inference - shallow (2-4 layers), linear + ClippedReLU, int8/int16 SIMD, giving millions of evals/sec/thread on CPU. The features are king-relative and shared: HalfKP encodes (own king square, piece type, piece square) with one shared weight row per feature - i.e. the same massive weight tying H8 tests, with the king position supplying the global reference frame that a purely per-square encoding lacks (the H10 point). Training signal: supervised regression to engine search scores over billions of positions (Stockfish nets are trained on positions labelled by Stockfish's own search, periodically re-generated - self-distillation), which is the 'learned eval without MCTS-scale training' in the claim; no outcome RL and no MCTS are involved. Consequences: adopted by Stockfish (SF12, +~130 Elo self-play), Komodo, Ethereal, and effectively every strong engine after 2020; handcrafted evals disappeared from top play within a season. Active disagreements are about training-data generation (teacher strength, position sampling, score blending with game results) and architecture variants (HalfKA, horizontally mirrored perspectives, larger accumulators), not about the core claim, which is as close to settled as anything in engine chess. Known limitation honestly stated in the docs: the accumulator trick only pays inside a deep search tree with make/unmake; for one-shot position evaluation it is optional.

**matched_controls**
  > Stockfish 11 vs Stockfish 12 is a genuinely matched comparison - same search framework, same hardware pool, eval swapped - making NNUE the strongest matched-control evidence in the entire learned-eval literature (KnightCap is the small-scale analogue). Caveat: the comparison still conflates the architecture with the training-data pipeline built around it, and fishtest self-play inflates gaps relative to independent-pool measurements (which put it nearer +100).

**known_failure_modes**
  > Quantization error accumulates with depth, forcing the shallow 2-4 layer design; NNUE knowledge lives in the first layer, and deepening shows diminishing returns. Incremental accumulator correctness is fragile: king moves, castling, and en-passant can require full or partial refreshes, and make/unmake desync bugs are a notorious defect class. Training-data dependence: nets distilled from the engine's own search inherit the teacher's blind spots and degrade out-of-distribution (extreme material imbalances, fortress positions); periodic re-distillation is required. Sparsity and incrementality assumptions are chess-specific and do not transfer to domains without near-static inputs. For import into the local harness: none of the speed machinery matters at depth<=5 numpy scale - the transferable content is the training signal and the feature design, not the accumulator.

### Mapping
**mapping_to_local**
  > For H11: NNUE settles the direction (learned leaf eval beats handcrafted) at engine scale with the cheapest successful training signal in the literature - supervised labels from the engine's own search - which maps directly onto the repo's existing train_on_search_feedback (depth-5) hook and the Group G 'graded by master' arm. For H8: NNUE is weight tying vindicated at the highest level - one shared first-layer weight row per king-relative feature - supporting the register's suspicion that tying wins when the feature set is right. For H9: NNUE cuts against 'budget dominates' at engine scale - a shallow, tiny-by-deep-learning-standards network won via inductive bias (king-relative features, sparsity), not parameter count; topology/bias dominated budget there. For H10: king-relative features are the published fix for per-square inexpressibility - the king square gives every feature a global anchor. Divergence: none of NNUE's performance mechanisms (accumulator, quantization, SIMD) apply to a numpy depth<=5 harness; only the representation and training-signal lessons transfer.

**scale_limits**
  > NNUE's speed claims presuppose millions of leaf evals per game inside a deep alpha-beta tree with make/unmake; at the local scale (a few hundred evals per game at depth 1-5) the differential accumulator buys nothing. Its training scale (billions of search-labelled positions, GPU trainers, fishtest fleets) is 6+ orders of magnitude above the local harness, and its strength verdict is entangled with engine-scale search depth. The literature cannot say whether king-relative shared features matter at 10^3-10^4 parameters, only that they dominate at 10^7+.

**small_scale_replication**
  > Regime gap for the mechanism (accumulator/quantization - nothing cheap to check locally), but 'not yet checked' for the transferable lesson: king-relative shared features and search-derived supervised labels are both implementable at numpy scale, and the repo already has the label pipeline (depth-5 search feedback) and a midstate store.

**confounds_flagged**
  > The +130 Elo is self-play against the engine's own predecessor (inflated vs independent pools, ~+100), and bundles architecture with two years of data-pipeline iteration. NNUE nets are self-distilled - trained on labels from the same engine family - so 'learned beats handcrafted' partly measures the teacher's depth, not learning per se. Hardware asymmetry: fishtest gains are measured on heterogeneous CPU fleets where integer NNUE is favoured. Adoption by 'all top engines' is also a network effect (shared tooling, nnue-pytorch), not 20 independent confirmations.

**suggested_local_test**
  > Do not replicate NNUE mechanics. Import the two transferable lessons into the H11/H25 arms: (1) king-relative feature arm - add a variant of the local encoder where piece-square features are expressed relative to the own-king square (cheap: a re-indexing of existing planes) and test whether the tied L3.5 arm recovers strength on the H10 global-fact probe set; (2) search-labelled supervised arm - train the value head on MidstateStore positions labelled by depth-5 search values (the NNUE training-signal lesson, already wired via train_on_search_feedback) and compare against pure outcome-RL at equal positions seen, which prices 'learned eval without MCTS-scale training' at numpy scale.

### Uncertain (skipped above)
- quantified_effect


## h12-move-ordering

### Register
**register_link**
  > H12 - claim: 'use_model_prior reduces node counts at fixed depth; it does not change the root value except among equals.' Local status (HYPOTHESES.md / 2026-10-03 review): OPEN - mechanism unit-tested only (tie example Nf6 = Nc6 = 90 exactly); node-saving magnitude never measured, and node counts are not in the bench report.

**register_implication**
  > Keep OPEN but split the row. The value-invariance half is a theorem (Knuth & Moore 1975), not an open question: exact fixed-depth alpha-beta returns the minimax value under any ordering, and ordering can only change WHICH equal-valued move is returned. Suggested one-line edit: 'Value invariance settled by theory (K&M 1975); the open half is empirical - node ratio prior-on/prior-off at depths 1-5 on a fixed position suite, plus prior top-1 agreement with depth-5 best moves.'

### Verdict
**literature_verdict**
  > supports - the invariance half is proven (alpha-beta computes the same root value under any ordering; Knuth & Moore 1975), and the node-saving half is the most exploited efficiency result in game search, though all published magnitudes come from heuristic stacks or neural priors far above the local perceptron-scale prior.

**scale_regime**
  > Value invariance: all regimes (exact theorem for fixed-depth alpha-beta without inexact pruning). Node savings: engine-scale canonical (TT + MVV-LVA + killer + history ordering is load-bearing in every strong alpha-beta engine) and AZ/Lc0-scale for learned priors inside MCTS (PUCT). At toy/numpy scale (depth<=5, no TT, ~10^3-param prior) the theorem still holds but the saving factor shrinks with depth; magnitude there is unpublished and unmeasured locally.

### Evidence
**key_sources**
  - Knuth & Moore 1975, 'An analysis of alpha-beta pruning', Artificial Intelligence 6(4):293-326 - proves alpha-beta returns the minimax value for any move ordering (ordering changes efficiency only, never the answer, up to which equal-valued move is selected) and derives the perfect-ordering 2b^(d/2) and random-ordering b^(3d/4) node bounds.
  - Chess Programming Wiki, 'Move Ordering' (history heuristic, killer moves, MVV-LVA, hash move) - the standard engine ordering stack; ordering quality is treated there as the dominant determinant of alpha-beta efficiency after the transposition table.
  - Silver et al. 2017, 'Mastering the game of Go without human knowledge' (Nature 550:354-359) - the neural policy prior inside PUCT replaces handcrafted ordering; the prior, not the value head, decides where simulations go.
  - Silver et al. 2018, AlphaZero (Science 362:1140-1144) - 80k simulations/move beating Stockfish 8 establishes how much node reduction a strong learned prior buys at scale against a heuristic-ordering alpha-beta engine.
  - McIlroy-Young et al. 2020, Maia, 'Aligning Superhuman AI with Human Behavior' (KDD '20; local copy tmp-maia.pdf) - quantifies what a good learned prior is: top-1 move accuracy above 52% vs a 46% engine ceiling; the accuracy metric the local prior's ordering quality depends on.

**literature_findings**
  > Two halves. (1) Invariance: Knuth & Moore 1975 proved alpha-beta returns exactly the minimax value of the full tree for any move ordering; ordering is purely an efficiency device. The only freedom ordering retains is among exactly tied optimal moves - it can select a different member of the tie set, which is precisely the local claim's 'except among equals' clause, and the repo's tie unit test (Nf6 = Nc6 = 90) is an instance of the theorem's edge case. This is uncontested consensus and the reason every engine reorders moves freely. (2) Magnitude: the theoretical gap between perfect and random ordering is a factor b^(d/4) in node count - exponential in depth, which is why ordering matters more the deeper the search. In practice every strong alpha-beta engine orders TT-move-first, then winning captures by MVV-LVA/SEE, then killer/history for quiet moves, and treats ordering as the highest-leverage component after the TT itself. Learned priors play the same role inside MCTS: AlphaGo Zero's and AlphaZero's PUCT formula lets the network prior allocate simulations, and AZ's 80k-nodes-per-move vs Stockfish's 70M-nodes-per-second comparison is the canonical demonstration that a good prior substitutes for node volume. Maia 2020 quantifies prior quality directly: a network trained for the right objective exceeds 52% top-1 move accuracy while strength-optimised engines cap at 46% for human-move prediction - i.e. prior quality is trainable and objective-dependent. Active disagreement: none on invariance; on magnitude, real engines deviate from the clean b^(d/4) picture because transposition-table cutoffs, aspiration windows, and null-move/LMR/futility pruning couple ordering back into the returned value at finite depth - inexact search CAN change the answer - which is why the local claim must stay scoped to exact fixed-depth search, as the repo's GuidedEngine is.

**matched_controls**
  > Knuth & Moore: matched by construction - same tree, same evaluator, only ordering varies; it is a theorem, not an experiment. Engine heuristic stacks: mostly NOT published as matched ablations; the field treats ordering's value as established engineering folklore and no canonical 'Elo lost without the history heuristic' number exists. AlphaZero vs Stockfish: NOT matched - search family (MCTS vs alpha-beta), hardware, and node budgets all differ; it shows priors can substitute for nodes, not the isolated effect of ordering. Maia: matched across its own rating bins (same architecture and training, different data slices), but its comparison to Stockfish/Leela is NOT matched in objective - Maia was trained to predict humans, the engines to win.

**known_failure_modes**
  > (1) Invariance fails once search is inexact: with aspiration windows, TT cutoffs, or null-move/LMR/futility pruning, ordering interacts with pruning decisions and can change the returned value at finite depth (search instability) - the theorem covers only exact fixed-depth alpha-beta. (2) A miscalibrated learned prior can order WORSE than simple MVV-LVA and increase node counts; prior quality, not prior-ness, drives savings. (3) Among exact ties, ordering changes WHICH equal move is played - harmless for the value but it changes the trajectory, so 'cannot change the answer' holds per-position, not per-match. (4) Quiescence search breaks the clean node-count story via variable-depth extensions. (5) At tiny depths (d<=2) the perfect-vs-random ordering factor is a small constant - the exponential advantage needs depth to exist. (6) TT reuse makes 'nodes at fixed depth' protocol-sensitive (warm vs cold table gives different counts for identical ordering).

### Mapping
**mapping_to_local**
  > The local claim is the theorem plus an empirical magnitude question, and the literature fully supports that framing. Divergence is only in machinery: published learned priors sit inside MCTS (PUCT), while the local use_model_prior reorders moves inside plain fixed-depth alpha-beta - mechanistically closer to the history heuristic than to AlphaZero. The narrowing the literature forces: (a) 'cannot change the root value' is guaranteed only for exact fixed-depth search - if the harness ever adds cross-iteration TT cutoffs, aspiration, or pruning, the claim must be re-scoped; (b) the expected magnitude at numpy scale is modest: at d<=5 the perfect-vs-random factor is at most single-digit-to-low-double-digit x, and the local prior's actual top-1 agreement with search (never measured) caps how close to 'perfect' its ordering can be; (c) the tie unit test the repo already has is exactly the theorem's 'among equals' clause, so the falsification condition should be a value change on a NON-tied position.

**scale_limits**
  > All published magnitudes come from regimes with deep search (d>=8), transposition tables, and either hand-tuned heuristics or 10^7-10^8-parameter policy nets. Nothing measures what a ~10^3-parameter perceptron prior buys at depth<=5 without a TT - the exact local regime. Maia/AZ accuracy numbers show neural priors can be excellent, but their training budgets (millions of games) exceed the local self-play corpus by 4+ orders of magnitude; the literature cannot say how accurate a naive-RL prior gets, hence cannot bound the local node saving.

**small_scale_replication**
  > Not yet checked, and cheap - this is a measurement, not a regime gap. The register's own command (run GuidedEngine with the flag on/off on a fixed position set, compare .nodes and _root_with_value) is decisive and runnable today; no literature substitute exists because nobody publishes at this scale.

**confounds_flagged**
  > Node counts are protocol-sensitive: warm vs cold transposition table, move-count limits, quiescence on/off, and tie-breaking RNG all change .nodes without changing ordering quality - the local measurement must fix all of them. Published ordering magnitudes assume deep search and a TT. Maia's accuracy is measured against HUMAN moves, not against search-best moves; top-1 agreement with a depth-5 engine's choices is a different (for the local purpose more relevant) target. AZ's 80k-vs-70M comparison conflates prior quality with value-head quality and the MCTS-vs-alpha-beta family difference.

**suggested_local_test**
  > Run the register's command as a node-ratio experiment: a fixed seeded suite of ~100 positions (mixed opening/midgame/endgame), GuidedEngine use_model_prior on vs off, depths 1-5, cold TT per run. Report (a) the node ratio per (position, depth) - the literature predicts ~1x at d=1 growing toward a single-digit factor at d=5 for a mediocre prior - and (b) _root_with_value agreement to float tolerance on every position: the theorem predicts exact agreement except reordered ties, and any value change on a non-tied position falsifies the claim. One cheap add-on the literature implies: measure the prior's top-1 agreement with the depth-5 best move (Maia-style move-matching against the engine itself); that single number bounds how close to perfect ordering the prior can ever get and turns H12 from pass/fail into a quantified statement.

### Uncertain (skipped above)
- quantified_effect


## h13-rating-methodology

### Register
**register_link**
  > H13 + H3. H13 claim: 'at the default --games 6, most pairwise gaps are inside 2 SE and therefore not results.' H13 status: [x] SUPPORTED for the ablation roster - every L5 ablation SE (79-93) exceeded every pairwise delta (35-100); general case untested. H3 claim: 'excluding unfinished games from ratings is right; including them as draws would change the ordering.' H3 status: [ ] untested - needs a flagged re-fit of rating.py counting unfinished as draws.

**register_implication**
  > H13: extend the [x] to the general case - the analytic SE formula below plus one --games 6 vs --games 30 crossing tabulation settles it for any roster. H3: keep [ ] but reframe - the Bradley-Terry-with-ties literature supports EXCLUDING truncated games (they are censored outcomes, not draws, and injecting draws pulls ratings together), so the re-fit is a sensitivity check on a defensible convention, not an open methodological risk. Suggested one-line edit to H3: 'Exclusion convention defensible per BT-with-ties literature (truncated != drawn); quantify sensitivity via flagged re-fit.'

### Verdict
**literature_verdict**
  > supports - for H13: rating theory (Glickman's guide; Glicko-2's RD; WHR; Fishtest's SPRT practice) puts the standard error of a handful of games per pair at tens-to-hundreds of Elo, matching the observed 79-93, so 6 games/pair cannot establish sub-100-Elo gaps. For H3: partially supports - the BT-with-ties literature treats genuine draws as an informative third outcome but offers no sanction for counting TRUNCATED games as draws; exclusion is the conservative, precedent-backed convention, and no published work tests this harness-specific convention directly.

**scale_regime**
  > Regime-free: the statistics are scale-invariant. The same 1/sqrt(games) standard errors, draw-rate corrections, and SPRT machinery govern a toy numpy roster and Fishtest's 10^4-10^5-game engine tests. Verdicts hold identically across toy/numpy, small-net, AZ/Lc0, and engine scales; only the draw rate (which enters the SE constant) shifts with strength.

### Evidence
**literature_findings**
  > (H13, sample size) Every model in the rating lineage - Elo's normal model, Bradley-Terry (logistic), Glicko/Glicko-2, Bayeselo, WHR - estimates a latent strength from win/draw/loss outcomes, so the standard error of a rating difference scales as 1/sqrt(games) with a constant set by the draw rate. The derived constant (SE ~ 347/sqrt(n(1-d)) per player near equality) reproduces the register's observed 79-93 Elo at the local roster geometry (~36 games/arm, d~0.5) and implies 6 games per pair cannot separate gaps below ~160-190 Elo at 2 SE - H13's 'not results' verdict is exactly what the statistics predict, and the remedy is mechanical: more games, or sequential testing. Glickman's guide supplies the framing (parameter vs estimate; the statistician's role includes identifying 'a reasonable sample size so that estimates are not likely to vary much'); Glicko-2 makes it operational (publish an RD next to every rating); Fishtest shows the industrial solution - GSPRT with explicit [Elo0, Elo1] tolerances at alpha=beta=0.05, normalized Elo so expected duration is independent of draw rate and opening book, the pentanomial model over paired game outcomes (ll/ld/dd-wl/wd/ww) for a substantial variance reduction over trinomial statistics, a chi-squared screen for anomalous workers, and explicit estimation of opening-book RMS bias, acknowledging book bias as a first-class confound. WHR demonstrates the value of joint fits over ALL games: its prediction edge over incremental systems (Elo/Glicko/TrueSkill) comes precisely from not discarding information, and its table quantifies the gain (55.793% vs 55.121-55.698% test prediction on millions of games). (H3, unfinished games) The BT-with-ties literature (Rao & Kupper 1967; Davidson 1970) models genuine draws as a third outcome with its own parameter; engine fitters (Bayeselo, ordo) either model draws explicitly or count them as half-points - but 'draw' everywhere means an agreed or rule-based terminal result, never a truncation. No published rating methodology scores truncated/censored games as draws, and the statistical consequence of doing so is known from the model itself: injected draws pull the two ratings toward each other and inflate apparent draw propensity, biasing gaps toward zero - Bayeselo's drawElo exists precisely because draw propensity is informative about strength. Consensus: H13 is correct and quantifiable; H3's exclusion convention is the defensible default with the only methodological precedent, and the local re-fit is a sensitivity analysis no paper can substitute for. Active disagreement: how to model genuine draws (plain half-points vs Davidson ties vs drawElo) - all camps agree genuine draws are informative; none address harness-truncated games.

**matched_controls**
  > WHR vs Elo/Glicko/TrueSkill/Bayeselo/decayed-history: matched - same KGS train (726k games) and test (2.3M games) splits, same prediction-rate metric, paired-data variance reduction; the cleanest rating-method comparison available. Pentanomial vs trinomial in Fishtest: matched on the same game streams, with an accounting identity isolating opening-book RMS bias. Glicko-2 vs Glicko-1: matched by design (same update structure plus RD/volatility). BT vs BT-with-ties (Davidson): nested models on the same data. For H3 specifically: NO matched comparison exists anywhere - no rating paper compares 'exclude truncated games' vs 'count them as draws' because truncated games do not occur in published datasets; the local flagged re-fit would be the first.

**known_failure_modes**
  > BT non-identifiability on disconnected comparison graphs: an arm playing only one opponent borrows that opponent's scale. Perfect or zero scores send MLE ratings to +/-infinity without a prior - WHR uses one virtual win plus one virtual loss against a rating-0 player; Bayeselo uses a Gaussian prior. The draw-as-half-points convention biases ratings when draw propensity correlates with strength (Bayeselo's drawElo documents this). SPRT optional stopping invalidates fixed-sample confidence intervals read post-hoc; conversely fixed-sample fits are invalid if the run was peeked at and stopped ad hoc. Pool drift: roster-internal Elos are anchored only relative to each other (WHR's KGS numbers are pool-absolute; local fits are not). The pentanomial i.i.d.-pairs assumption breaks under correlated openings, which Fishtest estimates explicitly as book RMS bias. Counting censored games as draws biases gaps toward equality - conservative against false claims of difference, anti-conservative against false claims of equality. Multiple comparisons: with ~21 pairs in a 7-arm roster, one spurious 2-SE gap is expected by chance at 6 games/pair.

### Mapping
**mapping_to_local**
  > The literature agrees with H13 so directly that the local numbers are a plug-in verification: observed SE 79-93 vs the analytic ~82 at the same geometry. The local harness diverges from published practice in three ways the literature flags: (1) no uncertainty (RD/SE) published alongside each fitted rating - Glicko-2's operational norm; (2) no sequential testing - fixed --games 6 is the worst-of-both-worlds design Fishtest abandoned; (3) the `unf` truncation outcome does not exist in any published model, making H3 a genuinely harness-local question. On H3 the literature narrows the claim: excluding unfinished games is not merely conservative, it is the only convention with precedent (censored data is excluded or modelled as censored, never scored as a draw), and the literature predicts the sensitivity will be small where `unf` counts are low and concentrated in weak arms (random 12, L1 10, L3 11 per H14).

**scale_limits**
  > Rating statistics are scale-free, so little is lost in transfer: the SE math, SPRT machinery, and draw-bias analysis apply verbatim at numpy scale. What the literature cannot provide: (1) the local draw rate and its arm-dependence, which set the SE constant (published draw rates are for strong engines or humans, not perceptron players); (2) any precedent for `unf` truncations - published datasets contain terminal results only; (3) calibration of the roster-internal Elo to any external scale. None of these block H13 or H3; they only mean the constants must be measured locally.

**small_scale_replication**
  > Not yet checked, and both checks are cheap: (a) H3's flagged re-fit (a one-line convention switch in rating.py, per the register) is a sensitivity experiment no literature can substitute; (b) H13's general case is settled by tabulating 2-SE crossings in one --games 6 vs --games 30 comparison, or analytically from the BT Hessian the fitter already computes. No regime gap exists for this item.

**confounds_flagged**
  > Draw handling is the flagged confound inside every cited model: Davidson ties, half-points, and drawElo give different rating spreads from the same results. Opening books / starting positions: Fishtest treats book bias as a measurable RMS quantity; the local fixed-start-position design has an analogous, unmeasured bias. Colour asymmetry at 6 games/pair (3 white / 3 black) is confounded with any first-move advantage; the pentanomial model exists partly to cancel it via pairing. Time/search-budget asymmetry across arms (the +eval arm ~4x slower per the H11 baseline) can interact with any node or move caps. Sequential tests are invalid under ad-hoc early stopping that ignores the boundaries. Unfinished counts correlate with untrained arms (H14: random 12, L1 10, L3 11), so an unf-as-draws re-fit differentially penalises weak arms - a confound that would masquerade as a convention effect.

**suggested_local_test**
  > Two cheap experiments, both implied directly by the literature. (1) H3: implement the register's flagged re-fit - count `unf` as half-points behind a flag in rating.py, re-fit bench/results-2026-10-02.json, and report the maximum rating displacement in units of the fitted SEs. The BT-with-ties literature predicts displacements biased toward equality and largest for high-unf arms (random/L1/L3); 'ordering unchanged within 1 SE' is the plausible outcome, keeping the convention defensible but unimportant. (2) H13 general case: run --games 6 vs --games 30 on the same subset and tabulate how many pairwise gaps cross 2 SE (the register's own command), then publish the analytic per-pair SE (347/sqrt(n(1-d))) next to the README's thresholds; optionally adopt Glicko-2-style RD reporting or an SPRT-style [Elo0, Elo1] tolerance framing for future runs - the full Fishtest machinery is overkill at this scale, but its tolerance statement is exactly how the harness should express 'how small is too small'.

### Uncertain (skipped above)
- key_sources
- quantified_effect


## searchless-gm

### Register
**register_link**
  > No register row. Register ref: stress-tests H20 ('Positional knowledge only pays once tactics are handled' - SETTLED at naive scale), bounds H12 ('The prior saves nodes but cannot change the answer' - OPEN), informs H16 ('Factorisation buys trainability, not expressiveness' - OPEN, blocked on L3-flat). Local role per the 2026-10-03 review: the scale-regime boundary - the result needs 270M parameters and 10M games, unreachable at naive numpy scale.

**register_implication**
  > No new row needed; optionally annotate H20's 'why the row exists' note: at sufficient scale a pure policy absorbs tactics (Ruoss et al. 2024: 2895 Lichess blitz Elo, no search), so H20 is a statement about the naive-scale regime, not about policies in principle. Suggested one-line edit to H20: 'SETTLED at naive scale; inverted at 270M-param/10M-game scale (Ruoss et al. 2024) - the boundary itself is the finding.'

### Verdict
**literature_verdict**
  > supports - at 270M params trained on 10M Stockfish-annotated games, a pure action-value policy reaches 2895 Lichess blitz Elo vs humans and 93.5% on 10k puzzles with zero search; but the same paper bounds the claim: it trails Stockfish 16 by ~407 Elo and AlphaZero+400-sim MCTS by ~200 Elo, and its 9M sibling manages only ~2007 Elo - the search-vs-eval division blurs only past a scale floor ~5 orders of magnitude above the local harness.

**scale_regime**
  > Verdict holds only in the 10^8-param / 10^7-game / 10^10-datapoint regime - a new published point between small-net and AZ self-play scale, distinguished by being supervised from Stockfish rather than trained by RL. Explicitly inverted at <=10M-param scale: the paper's own 9M model scores ~2007 tournament Elo, and its scaling ablation shows overfitting at 10k games for models >=7M params. At the local toy/numpy regime the claim fails, which is precisely its local role.

**quantified_effect**
  > 270M-param decoder-only transformer (16 layers, d=1024), 10M Lichess games annotated by Stockfish 16 at 50ms per legal move into 15.32B action-value points, trained on 128 TPUv5 chips for 20M steps: 2895 Lichess blitz Elo vs humans (174 games), 2299 (±14) internal tournament Elo, 93.5% exact-solution accuracy on 10,000 Lichess puzzles (rated up to 2867), 69.4% action accuracy and Kendall tau 0.300 vs the Stockfish oracle. Scale ladder (tournament Elo / puzzles): 9M = 2007 (±15) / 85.5%; 136M = 2224 (±14) / 92.1%; 270M = 2299 (±14) / 93.5%. Comparisons (tournament Elo): AlphaZero policy-net-only 1620 (±22), AZ value-net-only 1853 (±16), AZ + 400 MCTS sims 2502 (±15), Stockfish 16 (50ms) 2706 (±20); GPT-3.5-turbo-instruct 66.5% puzzles. Objective ablation at equal games (9M model): action-value 83.3% puzzles > state-value 77.5% > behavioral cloning 65.7%; at equal data points (40M, Appendix B.2) the AV-vs-SV edge vanishes (+252 vs +264 relative Elo) - AV wins mainly by yielding ~30x more supervision points per game (15.32B vs ~530M).

### Evidence
**key_sources**
  - Ruoss et al. 2024, 'Grandmaster-Level Chess Without Search' (arXiv:2402.04494, NeurIPS 2024) - the item's core: supervised distillation of Stockfish 16 action-values into a 270M transformer; 2895 Lichess blitz vs humans, 93.5% puzzle solve, beats AlphaZero's policy and value nets without MCTS, remains ~200 Elo below AZ+MCTS and ~407 below Stockfish 16; v2 retitled 'Amortized Planning with Large-Scale Transformers'.
  - ChessBench dataset, github.com/google-deepmind/searchless_chess - 10M games (part CC0 from Lichess), 15B Stockfish-16 annotations with action-value / state-value / behavioral-cloning targets (1.1TB action-value train split), plus 10k puzzles and 9M/136M/270M checkpoints; the public artifact defining the data floor for the result.
  - Silver et al. 2018, AlphaZero (Science 362:1140-1144) - the baseline isolating what search adds: in Ruoss et al.'s own tournament the AZ-family nets score 1620 (policy only) / 1853 (value only) / 2502 (+400 sims MCTS) - the cleanest published search-vs-no-search ladder at neural scale.
  - McIlroy-Young et al. 2020, Maia (KDD '20; local copy tmp-maia.pdf) - the policy-only predecessor at small-net scale: supervised human-move prediction tops 52% top-1 accuracy but was never strength-optimal; Ruoss et al. show the same supervised-policy recipe reaches GM level when the teacher is Stockfish action-values instead of human moves and scale is raised ~10x.

**literature_findings**
  > The paper establishes that tactics are absorbable into a pure policy at sufficient scale: greedy selection over predicted Stockfish action-values solves 93.5% of 10k Lichess puzzles (sequences up to ~2800 rating, entirely greedily - 'solving the puzzle sequences relies entirely on having good value estimates that can be used greedily') and plays at 2895 Lichess blitz vs humans, with no search, rollouts, or domain-specific architecture. Three qualifications the same paper states explicitly, which together define the current consensus. (1) The division blurs but does not vanish - 'perfect distillation is still beyond reach': Stockfish 16 at the same 50ms/move oracle leads by ~407 tournament Elo, and AlphaZero with 400 MCTS sims leads the searchless model by ~200 Elo, so search still adds substantial strength on top of even a 270M policy. (2) The result is scale-gated: Elo and puzzle accuracy rise monotonically from 9M to 136M to 270M; at 10k training games models >=7M params overfit; 'strong chess performance only arises at sufficient scale' is the paper's own framing. (3) 'Without search' has precise scope: the claim covers the action-value and behavioral-cloning policies; the state-value policy (evaluate all states reachable by legal moves) is conceded to be 'a version of 1-step search', and actual play used three non-search patches - softmax temperature 0.005 for the first 5 full-moves (variety), a threefold-repetition win% override, and a Stockfish fallback when the top 5 moves all predict >99% wins (to stop mate-line dithering). On representation: the action-value objective (classify Stockfish's Q(s,a) into 128 win-prob bins; argmax expected value over legal moves) beats state-value and behavioral cloning per GAME mainly because it yields ~30x more supervision points per game; at equal DATA POINTS the advantage disappears - a direct lesson for the local H16 question about whether representation wins are statistical or expressiveness wins. Active disagreement: essentially none on the numbers; the stated open question is whether the remaining gap to Stockfish closes with more scale.

**matched_controls**
  > Objective ablation (AV vs SV vs BC): matched at equal games (9M model) AND re-run at equal data points (40M, Appendix B.2) - the paper's own best control, and the one that changes the conclusion (AV's edge is supervision density, not objective magic). Model-size ladder 9M/136M/270M: matched (same data and recipe). Dataset-size scaling: matched at fixed model sizes. vs AlphaZero: NOT matched - AZ has 27.6M params trained on 44M self-play games with full PGN-history input and a different teacher (MCTS self-play RL vs Stockfish-16 supervision), so the +679 Elo over AZ's policy net entangles scale, data, and objective. vs Stockfish 16: matched in oracle time budget (same 50ms-per-move evaluation protocol) but not in algorithm class. Lichess 2895 vs internal 2299: the paper itself flags human-vs-bot pool miscalibration - the SAME model differs by ~600 Elo across pools.

**known_failure_modes**
  > Pool miscalibration: the same 270M policy is 2895 vs humans and 2299 vs bots on Lichess - a ~600-Elo reminder that the headline number is pool-relative. Distillation ceiling: the student inherits the teacher's blind spots and stays ~407 Elo below it; value estimates generalize imperfectly (Kendall tau 0.300 vs oracle 1.0; action accuracy 69.4%). 'No search' asterisks: threefold-repetition and >99%-endgame Stockfish patches plus opening temperature were needed for robust play; the state-value variant is arguably 1-step search. Scale floor: at 9M params the same recipe yields ~2007 Elo, and <=10k games induce overfitting at >=7M params - the technique does not degrade gracefully toward numpy scale. Data dependence: the 15.32B teacher labels are themselves engine-search-derived (50ms of Stockfish per legal move), so the policy is distilled search - 'searchless' applies at inference, not at data generation (the v2 'amortized planning' framing). Puzzle scoring is exact-sequence match, stricter than and different from playing strength.

### Mapping
**mapping_to_local**
  > H20 (SETTLED locally): the paper stress-tests the claim's scope, not its content - at naive scale tactics dominate and a 1-ply material player sits mid-table, exactly as the register says; Ruoss et al. show that past a ~10^8-param / ~10^10-datapoint floor the same 1-ply greedy selection becomes GM-strength BECAUSE the eval has absorbed tactics. The local row should read as regime-bounded, which it already implicitly is. H12 (OPEN): the paper bounds what a policy prior can be - AZ's raw policy net (1620) and value net (1853) sit ~900/~650 Elo below their own MCTS version, showing even excellent learned priors leave large value on the search table; and the 270M model's 69.4% action accuracy / tau=0.300 against its own teacher quantifies the prior imperfection any node-saving story must live with. H16 (OPEN): the AV-vs-SV-vs-BC ablation is the closest published analogue to 'factorisation buys trainability, not expressiveness' - BC's deficit vanished when data points (not games) were equalized, i.e. the win came from supervision density per position, a statistical-sharing story, not a capacity story. The divergence: nothing here is outcome-RL; all supervision is engine-derived, which the local harness deliberately lacks.

**scale_limits**
  > The entire result lives ~5 orders of magnitude above the local regime on every axis: 270M vs ~10^3-10^4 params; 10M games / 15.32B annotated points vs the harness's self-play corpus; 128 TPUv5 chips vs numpy; Stockfish-16 teacher labels vs outcome-only signal. The paper's own scaling section is the explicit boundary: below ~10M params and ~1M games its recipe produces 2000-Elo-class play with overfitting - still far above local scale. It cannot say whether any policy-first approach works at numpy scale; it only certifies that the naive-scale ceiling on policy quality (H12's prior, H20's tactics) is a scale phenomenon, not a policy-in-principle phenomenon.

**small_scale_replication**
  > Regime gap, not 'not yet checked': no numpy-scale replication of GM-without-search is possible by construction - the claim is about scale. The cheap local echo that IS available: a Maia-style move-matching measurement of the local policy against the harness's own depth-5 search (top-1 agreement), which quantifies prior quality - the same quantity that, scaled up, makes searchless play possible. That experiment belongs to H12, not here.

**confounds_flagged**
  > Teacher-student confound: all supervision is Stockfish-16-derived, so 'searchless' is inference-only - search is amortized into the training data. Pool confound: 2895 (humans) vs 2299 (bots) for the identical policy. Oracle confound: Stockfish at 50ms per LEGAL MOVE evaluated independently is weaker per position than 50ms-per-move real play and has no cross-move consistency. Opening variety was injected (temperature 0.005 for the first 5 full-moves), so reported strength is not pure argmax play. The endgame Stockfish fallback patches a pure-policy failure mode (mate-line dithering). Puzzle/train overlap was checked (1.33% of puzzle initial states in training) and memorization cannot explain generalization, but tau=0.300 shows ranking of unseen positions is far from the teacher. The AZ comparisons use a fresh tournament (avoiding the 2018 hardware/time-control controversies) but param/data/history-input asymmetries remain.

**suggested_local_test**
  > Nothing new to run for this item itself - its local role is a boundary marker. The one transferable measurement the paper implies for the harness: quantify prior/eval quality directly - top-1 action agreement and a Kendall-tau-like ranking correlation between the local policy/value scores and depth-5 search values over a fixed position set (e.g. from MidstateStore). Ruoss et al. show these two numbers (action accuracy, Kendall tau) are the honest proxies for how much of search has been absorbed into the network; at local scale they convert H12's node-saving question and H20's tactics question into measurable quantities without any new arms.

