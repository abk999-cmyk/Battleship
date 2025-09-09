## Agent 2: Probabilistic Monte Carlo + Neural Heatmap

This document specifies `agents/AI_agent2.py` (class `AIPlayer2`) in scientific detail: state, algorithms, and decision policy.

### State representation
- Board size `BOARD_SIZE = 10`; ship sizes from `core.board.SHIP_SIZES`.
- Belief grid `result_grid ∈ {−1,0,1}^{10×10}`: −1=miss, 0=unknown, 1=hit.
- Action set `available_moves ⊂ [0..9]×[0..9]` maintained as a set for O(1) deletions.
- Hit memory `hits` for active, unsunk fragments; `sunk_ships_info` with coordinates and sizes; `hit_count` tracks confirmed ship squares.
- Mode `mode ∈ {hunt, target}`; in target mode, `target_axis_hits` stores a maximal collinear subset, `target_blocked_ends` marks dead-end probes.
- Remaining ships `remaining_ship_sizes` with cached `min_remaining_ship_size`, `max_remaining_ship_size`.
- Optional neural model `nn_model: f(10×10×3)→10×10` loaded from `models/battleship_heatmap.h5` if TF available.

### Observation encoding
- Miss plane M, Hit plane H, Unknown plane U from `result_grid`; stacked into tensor X∈ℝ^{10×10×3}. The CNN outputs P∈[0,1]^{10×10}. Non-legal cells are zeroed.

### Decision policy f: S → A
1) Endgame guards
- If squares remaining = 1, `_select_final_square_overall()` returns the final square by prioritizing neighbours of hits and line-extension.
- If only one ship remains, `_target_final_ship_exhaustive()` enumerates all placements of that ship consistent with constraints (misses, sunk cells, active hits) and selects the most frequent cell (tie-breaks by NN heatmap if present).
- If ≤4 ship squares remain, `_exhaustive_endgame_search_from_original()` recursively enumerates multi-ship placements to frequency-rank candidate cells.

2) Target mode
- `_select_target_mode_move_enhanced()`:
  - Determine axis from `target_axis_hits` (horizontal/vertical) else try four orthogonal neighbours of the most recent hit.
  - Generate endpoints (front/back) candidates, prune off-board/illegal and blocked ends, prefer NN-weighted candidates when available.
  - On miss at an endpoint, `_mark_blocked_if_miss_at_target_end()` flips the corresponding `target_blocked_ends` bit.

3) Hunt mode master probability
- `compute_master_probability_grid()` computes
  - Density term D by counting legal placements of every remaining ship over the board under constraints (misses, sunk cells, covering hits when present); normalized.
  - Parity mask Π for stride = `min_remaining_ship_size` with rotating offset based on `turn_count`; applied when no active hits.
  - Neural term N from CNN heatmap, blended with weight α (0.6 in hunt, 0.2 in target) scaled to current max of the combined grid.
  - Monte Carlo term MC via `_run_monte_carlo_simulations_enhanced()`: sample placements for remaining ships avoiding conflicts and covering `hits`; add with weight β (0.3 if no hits else 0.5).
  - Local adjacency bonus B: for each unsunk hit, boost its four neighbours by 2× current max.
- Final grid G = normalize(max to 1). Best actions A* = argmax G restricted to `available_moves`; random tie-break.

### Monte Carlo simulation
- Draw up to `MC_SAMPLES_BASE` placements (tripled in late endgame). For each ship: attempt up to 20 random legal placements avoiding misses/sunk/overlap; accept sample only if all `hits` are covered. Accumulate occupancy counts over `available_moves`; return normalized occupancy as MC grid.

### Complexity and safety
- All exhaustive routines guard with constraints and memoization; parity reduces search branching in early hunt; MC guards against zero-coverage by returning None.
- Defensive fallbacks: if grids flatten or illegal choices arise, `_fallback_hunt_move()` returns a parity-respecting random move, else uniform random from `available_moves`.

### Update dynamics
- After attack, `update_state(move, result, opponent_board)` updates `result_grid`, `hit_count`, `hits`, mode switching, target axis, sunk bookkeeping (removing ship size, pruning hits), and end-blocking on misses.

### Interfaces used by engine
- `take_turn(opponent_board) -> str` chooses move, executes `board.attack`, records `last_move`, `last_result`, and returns result string; returns "no_moves" if exhausted.
- `view_display()` provides a textual board view with meta-state for debugging.

### Reproducibility knobs
- Determinism depends on Python RNG for parity offsets and tie-breaks; seed Python’s `random` for repeatability.
- Neural predictions are deterministic for a fixed model and environment; MC randomness derives from Python RNG.

### Empirical behaviour
- Early game: parity + density dominate; NN provides prior shape; MC weight lower without hits.
- Mid game: target mode alternates with hunt; axis inference stabilizes; MC contributes.
- Late game: exhaustive routines and MC dominate; NN weight reduced.
