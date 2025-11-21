# Methodology

This section details the methodology underlying a modular Battleship agent that blends classical probabilistic reasoning, neural pattern recognition, and Monte Carlo occupancy sampling, with blend weights evolved by a genetic algorithm (GA). We describe datasets and training, model architectures, inference and decision flow, GA optimization, evaluation protocols, statistical validity, and reproducibility. The central novelty is an offline GA that learns robust per‑source blend weights which are then applied at runtime by Agent 4.

## Problem setting and environment
- Board: 10×10 grid; fleet sizes: [5, 4, 3, 3, 2].
- Game rules: standard Battleship; ships placed uniformly at random subject to non‑overlap; agent observes only its shot outcomes (hit/miss/sunk).
- Turn structure: on each turn, the agent selects an unplayed cell; the opponent returns outcome and sunk status; the agent updates its internal state and proceeds until game end.

## Agent overview
The agent computes multiple per‑cell score maps every turn and blends them into a master probability grid used to select the next move.

- Density (classical counting): counts how many legal placements of every remaining ship cover each cell.
- Neural heatmap (CNN): predicts per‑cell ship presence based on the current 3‑plane board encoding (miss/hit/unknown). Recomputed every turn.
- Monte Carlo occupancy: samples complete, globally consistent ship layouts consistent with observations and counts cell occupancy frequency.
- Information gain: approximates expected information gain via entropy × hit probability.
- Parity mask (hunt mode): multiplicative mask that probes only one modular class defined by the minimum remaining ship length.
- Neighbor‑of‑hits bonus (targeting): additive boost to orthogonal neighbors of current hits to efficiently finish ships.

Agent 4 is a thin subclass of Agent 3 that loads GA‑tuned blend weights from `models/ga_weights.json` at initialization and otherwise inherits identical decision logic.

## Data generation and supervised training (neural heatmap)
- Dataset: Generated via self‑play using `generate_dataset.py`. For each move state:
  - Input `state_tensor`: 10×10×3 float16 planes for miss, hit, unknown.
  - Target `ship_grid`: 10×10 int8 mask of true ship locations.
  - Samples are streamed as pickled tuples (state, target) to `data/battleship_supervised.pkl`.
- Model: a compact fully convolutional network trained by `train_heatmap.py`:
  - Architecture: Conv(64, 3×3, relu, same) × 3 → Conv(1, 1×1, sigmoid, same).
  - Loss: binary cross‑entropy; Optimizer: Adam(1e‑3); Epochs ≈ 12; Batch = 256.
  - Train/val split: 90/10; persistence: `models/battleship_heatmap.h5`.
- Inference contract: given the live 3‑plane board encoding, output a 10×10 heatmap aligned to board indices; mask to legal moves before blending. The model gives per‑cell probabilities (not normalized over the board).

## Deterministic density computation
Each turn, the agent enumerates all in‑bounds horizontal/vertical placements for each remaining ship size, rejecting any placement overlapping known misses or sunk cells, and requiring consistency with active hits. For every legal placement, it increments the covered unknown cells. Summed across ships, this yields a smooth, structural prior over plausible cells. A parity mask based on the minimum remaining ship length gates legal probe squares in hunt mode, accelerating coverage while preserving completeness.

## Monte Carlo occupancy sampling
To respect inter‑ship constraints, the agent repeatedly samples entire fleets: it randomly places remaining ships without overlap, enforcing miss/sunk constraints and requiring that all current hits lie under placed ships. Each accepted layout increments an occupancy counter for the cells it uses. The grid is normalized to [0,1] by its maximum. The sampling budget scales up in endgame to sharpen estimates when few squares remain.

## Information gain estimation
Uncertainty per cell is computed using binary entropy on Monte Carlo hit probabilities. Expected information gain is approximated as entropy × hit probability, prioritizing shots that both are likely to hit and reduce uncertainty. If Monte Carlo is unavailable (e.g., early game sparsity), IG falls back to contributing little or nothing.

## Blending and decision policy
Let D, N, MC, IG denote the four maps (each normalized to [0,1] over legal moves). Let w_d, w_n, w_mc, w_ig be the blend weights. The master grid is constructed in a stable, scale‑preserving sequence:

1) Initialize grid = 0.
2) Add density: grid += w_d · D.
3) Blend neural: grid = (1 − w_n)·grid + w_n · (N · grid.max()).
4) Blend Monte Carlo: grid = (1 − w_mc)·grid + w_mc · (MC · grid.max()).
5) Blend information gain: grid = (1 − w_ig)·grid + w_ig · (IG · grid.max()).
6) Hunt‑mode parity: grid *= parity_mask if min ship size ≥ 2.
7) Neighbor bonus: grid += bonus concentrated on orthogonal neighbors of active hits, scaled by current grid.max().
8) Invalidate already‑played or illegal cells (−∞), then pick an argmax among remaining cells (breaking ties randomly among the top).

This design maintains numeric stability, preserves prior structure, and allows later sources to tilt the distribution without overwhelming scale.

## Genetic algorithm for weight learning (novelty)
We formulate the per‑source blend weights as a real‑valued chromosome in [0,1]^5: {density, neural, montecarlo, information_gain, opponent_model}. Although opponent_model is not directly blended in this version, it remains available for future extensions. The GA optimizes these weights offline using multi‑opponent self‑play.

- Fitness: 100 × win_rate − avg_moves_to_win, encouraging agents that win often and quickly.
- Opponents: a fixed set, e.g., `AIPlayer`, `AIPlayer2`, and `UltimateBattleshipAgent` (configurable).
- Evaluation: each chromosome plays multiple games per opponent; winners and move counts are aggregated; only scalar fitness returns from workers to bound memory.
- Operators: uniform per‑gene crossover (rate ≈ 0.25), Gaussian mutation (σ≈0.12, clamped to [0,1]), and elitism (~10%).
- Parallelism and safety: evaluation runs in a multiprocessing pool with finite worker lifetimes to recycle memory; periodic garbage collection; checkpoints written every few generations to `models/ga_weights.json`.
- Deployment: Agent 4 loads `ga_weights.json` at startup and uses these weights for all games until the file is replaced.

### Why GA for blending?
- Non‑convex objective: interactions among sources and game dynamics create complex, non‑differentiable landscapes; GA explores them without gradient information.
- Robustness: GA can escape local optima created by adversarial opponent mix or training bias in the neural component.
- Interpretability: learned weights remain transparent, allowing ablations and targeted improvements.

### Example GA outcome
A typical evolved set emphasizes Monte Carlo in late game while keeping density and neural as structural priors, with IG often reduced toward zero if it fails to improve on the chosen opponent set. This is consistent with MC’s advantage under tight constraints and density/NN’s smoother guidance early.

## Experimental protocols
- Baselines: compare Agent 4 (GA‑tuned) against Agent 3 (default weights), `AIPlayer2` (NN + heuristics), `AIPlayer` (classical), and a strong scripted “Ultimate” agent.
- Match scheduling: fixed opponent placement seeds when comparing agents to reduce variance; identical starting conditions per matchup where applicable.
- Game counts: ≥1000 games per opponent for stable estimates; report per‑opponent win rates and average moves to win.
- Ablations: set each weight to 0 in turn (D / N / MC / IG / OPP) holding others fixed to quantify marginal contribution.
- Confidence intervals: bootstrap (10k resamples) for win rate and moves to win; report 95% CIs.
- Efficiency metrics: time per game and memory footprint (optional) to show practical feasibility.

## Statistical validity and threats to validity
- Paired comparisons: use identical RNG seeds across agent comparisons for opponent placements to enable paired statistical tests.
- Distributional shift: self‑play data for the NN may bias priors; GA partially mitigates by tuning against diverse opponents; include more opponents to improve generalization.
- Overfitting to evaluation set: mitigate by holding out opponents for final reporting or rotating opponent pools across GA epochs.
- Stochasticity: report standard errors and CIs; when differences are small, apply non‑parametric tests (e.g., Wilcoxon signed‑rank) on paired outcomes.

## Reproducibility
- Environment: Python 3.9; TensorFlow 2.17.0; use a dedicated virtual environment with pinned `requirements.txt`. Ensure system `tkinter` availability if running the UI.
- Artifacts: publish `models/battleship_heatmap.h5`, `models/ga_weights.json`, and evaluation CSVs/logs.
- Seeds: fix Python/NumPy/TensorFlow and game placement seeds during evaluation; document all seeds used.
- Scripts: dataset generation (`generate_dataset.py`), training (`train_heatmap.py`), GA optimization (`ga_optimizer.py`), and agent play/evaluation (`game.py`, `AI_agent3.py`, `AI_agent4.py`).

## Implementation details and safeguards
- Logging: per‑module logs in `logs/` (agent decisions, GA progress, setup) for traceability.
- Memory safety in GA: finite worker lifetimes and garbage collection after each game; only scalar results passed back to the driver.
- Numerical stability in blending: normalization and max‑scaled convex mixing to avoid dominance by any single source’s absolute scale.
- Endgame handling: specialized exhaustive searches when ≤4 ship squares remain; parity and targeting logic to accelerate ship completion.

## Limitations and future work
- The current blend excludes a direct opponent‑model map; “opponent_model” is reserved for future integration (e.g., learned placement priors by opponent class).
- The neural heatmap is supervised on self‑play distributions; adding data augmentation and opponent‑conditioned features may reduce bias.
- Information gain uses a simple entropy × probability approximation; more precise VOI estimation (e.g., via rollout or Bayesian updates) could improve mid‑game choices.
- Meta‑learner: the project includes a prototype meta‑learner (`meta_learner.py`) capable of per‑turn dynamic weight adjustment; integrating it end‑to‑end and comparing against GA‑fixed weights is promising future work.

## Summary
We present a modular inference pipeline for Battleship and a GA‑based procedure to learn robust blend weights across heterogeneous sources. The combination of a deterministic density prior, a learned neural prior, and globally consistent Monte Carlo occupancy, augmented by information gain, parity gating, and targeted bonuses, yields a practical and statistically grounded decision system. The GA discovers weightings that transfer across opponents and game phases, providing both performance and interpretability.
