## Agent 3: Multi-Strategy + Meta-Learning + Opponent Modeling

This document specifies `AI_agent3.py` (class `AIAgent3`) in full detail: state, algorithms, learning, and decision policy.

### Additions over Agent 2
- Meta-weights `meta_weights = {density, neural, montecarlo, information_gain, opponent_model}` controlling linear blending of sub-grids.
- Opponent modeling: persistent profiles (`models/opponent_profiles.pkl`) storing placement and attack tendencies, orientation bias, clustering.
- Graph-based ship representation (NetworkX) for reasoning over contiguity and alignment; used to prioritize targeting.
- Information-theoretic scoring: per-cell entropy via MC posterior and expected information gain grid.
- Continuous learning hooks: move logs, performance metrics persisted to `models/ai_performance_metrics.pkl`.

### State and logs
- Inherits Agent 2 state; adds `ship_graph`, `entropy_grid`, `performance_metrics`, `board_states_history`, and `move_log`.
- Robust logging and error handling guard optional deps (TensorFlow, SciPy, NetworkX), with graceful fallbacks.

### Decision policy f: S → A
1) Endgame guards identical in spirit to Agent 2, with tuned helpers `_exhaustive_endgame_search`, `_target_final_ship_exhaustive`, `_select_final_square_overall`.

2) Target mode
- `_select_target_mode_move_enhanced()` computes `target_axis_hits`; proposes endpoints or neighbours; prioritizes with NN heatmap when available; integrates graph connectivity scores when NetworkX is available.

3) Hunt mode master probability
- `compute_master_probability_grid()` builds:
  - Density grid D as in Agent 2.
  - Adaptive parity Π with rotating offset.
  - Neural grid N via `_get_neural_heatmap()`.
  - Monte Carlo grid MC via `_run_monte_carlo_simulations()`.
  - Information gain grid I: `I[r,c] = entropy[r,c] * p_hit[r,c]` where `p_hit` derives from MC samples and `entropy` from either SciPy’s `entropy` or a local fallback.
- Blend: `G = w_d * D̂ + w_n * N̂ + w_mc * MĈ (+ w_i * Î)` scaled to current magnitude. Neighbour bonus around unsunk hits is then added.
- Select argmax over `available_moves` with random tie-break.

### Opponent modeling
- Profiles structure: dict with keys `placement_tendencies`, `attack_patterns`, `ship_orientations`, `clustering_tendency`.
- During sunk events, orientation counts are updated; post-game learning can update attack patterns; profiles saved to disk.
- Profiles are currently used to inform future enhancements (and can be integrated into the density prior or a dedicated opponent prior plane).

### Information theory
- `update_information_metrics()` derives an MC posterior over ship occupancy, computes per-cell binary entropy, and caches `entropy_grid`.
- `_compute_information_gain_grid()` approximates expected information gain by weighting entropy with `p_hit` from MC.

### Graph reasoning (optional)
- `initialize_ship_graph()` constructs a grid graph; `update_ship_graph(move, result)` removes edges through misses, constrains neighbourhood around hits/sunk structures; targeting may prioritize moves maintaining alignment with current `target_axis_hits`.

### Learning hooks
- `learn_from_game(game_result, opponent_moves=None)`:
  - Updates `performance_metrics` counters and persists to `models/ai_performance_metrics.pkl`.
  - Optionally updates opponent profile from move sequences; persists `opponent_profiles.pkl`.
- Continuous board-state capture for post-hoc analysis (`board_states_history`).

### Neural model initialization
- Robust loader mirrors Agent 2 path logic and prints directory listing to aid debugging; gracefully disables NN when missing.

### Reproducibility and determinism
- Blends and Monte Carlo share random sources with Agent 2; set Python RNG for repeatability of parity offsets, sampling, and tie-breaks.

### Empirical behaviour
- With tuned meta-weights, Agent 3 outperforms Agent 2 via better mid/late-game targeting and information-gain prioritization. Graph and opponent-modeling provide additional structure when deps are available.
