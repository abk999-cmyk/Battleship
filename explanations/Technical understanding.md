# Technical Understanding

This document summarizes component architectures and provides high-level pseudocode for the Battleship AI system, focusing on Agent 3/4. It is intended as a technical companion to the methodology.

## Components and architectures

### 1) Neural network (heatmap CNN)
- Input: 10×10×3 tensor [miss, hit, unknown].
- Architecture:
  - Conv2D(64, 3×3, relu, same)
  - Conv2D(64, 3×3, relu, same)
  - Conv2D(64, 3×3, relu, same)
  - Conv2D(1, 1×1, sigmoid, same)
- Output: 10×10 map in [0,1], aligned to board cells.

### 2) Density (classical counting)
- Deterministic enumeration of all legal placements for remaining ships.
- Sums coverage counts per cell, respecting misses/sunk and consistency with hits.

### 3) Monte Carlo occupancy
- Randomly sample whole‑fleet placements consistent with constraints.
- Count per‑cell occupancy over accepted layouts; normalize by max.

### 4) Information gain (entropy × hit probability)
- Entropy from Monte Carlo probabilities; score = H(p) × p.

### 5) Parity mask
- k = min remaining ship size; allow cells where (r+c) % k == offset.
- Offset rotates over time.

### 6) Neighbor‑of‑hits bonus
- Additive boost to orthogonal neighbors of active hits; scaled by current grid.max().

### 7) Blending
- Sources: Density (D), Neural (N), Monte Carlo (MC), Information Gain (IG).
- Weights: w_d, w_n, w_mc, w_ig from GA (Agent 4).
- Combine via stable, scale‑preserving convex blends; apply parity and neighbor bonus.

### 8) Targeting mode
- Maintain line axis through hits; extend at unblocked ends.

### 9) Endgame solvers
- Special cases: only 1 ship square left; or 1 ship remains; or ≤4 squares remain.
- Use exhaustive checks and placement frequency when feasible.

### 10) Genetic Algorithm (GA) for weights
- Chromosome in [0,1]^5: {density, neural, montecarlo, information_gain, opponent_model}.
- Fitness = 100×win_rate − avg_moves_to_win; variation via crossover/mutation; elitism.

### 11) Data pipeline (supervised NN)
- Self‑play generates (state_tensor, ship_grid) stream; trained with BCE/Adam.

### 12) Meta‑learner (optional prototype)
- MLP that outputs per‑turn weight adjustments given a 20‑dim state feature vector.

## Pseudocode

### A) Heatmap CNN inference
```
function neural_heatmap(result_grid, model):
    miss ← (result_grid == -1)
    hit  ← (result_grid ==  1)
    unk  ← (result_grid ==  0)
    X    ← stack([miss, hit, unk])           # 10×10×3
    Y    ← model.predict(X)[0]               # 10×10 or 10×10×1
    return mask_to_legal(Y)
```

### B) Density grid
```
function density_grid(state):
    grid ← zeros(10,10)
    for ship_len in remaining_ship_sizes(state):
        for each in‑bounds start and orientation:
            coords ← cells_covered(start, ship_len, orientation)
            if coords intersect misses or sunk: continue
            if not consistent_with_hits(coords, state.hits): continue
            for (r,c) in coords:
                if is_unknown(state, r, c): grid[r,c] += 1
    return normalize_by_max(grid)
```

### C) Monte Carlo occupancy
```
function mc_occupancy(state, num_samples):
    occ ← zeros(10,10)
    success ← 0
    for s in 1..num_samples:
        placed ← sample_full_fleet(state)         # respects misses/sunk; no overlap
        if placed == None: continue
        if not state.hits ⊆ placed.cells: continue
        success += 1
        for (r,c) in placed.cells:
            if is_legal(state, r, c): occ[r,c] += 1
    if success == 0: return None
    return occ / max(occ)
```

### D) Information gain
```
function information_gain(state):
    mc ← mc_occupancy(state, budget)
    if mc is None: return zeros(10,10)
    for each cell (r,c):
        p ← mc[r,c]
        H ← -p*log2(p) - (1-p)*log2(1-p) if 0<p<1 else 0
        ig[r,c] ← H * p
    return normalize_by_max(ig)
```

### E) Parity mask
```
function parity_mask(state):
    k ← min_remaining_ship_size(state)
    if k <= 1: return ones(10,10)
    offset ← (state.turn // (BOARD_SIZE // k + 1)) % k
    mask ← zeros(10,10)
    for r,c: if (r+c) % k == offset: mask[r,c] = 1
    return mask
```

### F) Neighbor‑of‑hits bonus
```
function neighbor_bonus(state, base_grid):
    bonus ← zeros_like(base_grid)
    mag ← max(base_grid) * 2 if max(base_grid)>0 else 1
    for (rh,ch) in state.hits:
        if part_of_sunk(rh,ch,state): continue
        for (dr,dc) in [(0,1),(0,-1),(1,0),(-1,0)]:
            r,c ← rh+dr, ch+dc
            if in_bounds(r,c) and is_unknown(state,r,c): bonus[r,c] += mag
    return bonus
```

### G) Blending and move selection
```
function select_move(state, weights):
    grid ← zeros(10,10)
    D ← density_grid(state)
    grid ← grid + weights.density * D

    N ← neural_heatmap(state.result_grid, model)
    if N not None:
        grid ← (1-weights.neural)*grid + weights.neural*(N * max(grid))

    MC ← mc_occupancy(state, budget)
    if MC not None:
        grid ← (1-weights.montecarlo)*grid + weights.montecarlo*(MC * max(grid))

    IG ← information_gain(state)
    if IG not None:
        grid ← (1-weights.information_gain)*grid + weights.information_gain*(IG * max(grid))

    if state.mode == "hunt" and min_ship_size(state) >= 2:
        grid ← grid * parity_mask(state)

    if state.hits:
        grid ← grid + neighbor_bonus(state, grid)

    invalidate_illegal(grid, state)
    return argmax_random_tie(grid)
```

### H) Targeting mode move
```
function target_move(state):
    hits_axis ← determine_axis(state.hits)
    if aligned(hits_axis):
        ends ← unblocked_axis_ends(hits_axis, state)
        candidates ← neighbors_at_ends(ends)
    else:
        candidates ← orthogonal_neighbors(state.hits)
    return prioritize_with_nn_or_random(candidates)
```

### I) Endgame solvers
```
if squares_remaining(state) == 1:
    return best_neighbor_of_hits_or_argmax(grid)
if len(remaining_ship_sizes)==1:
    return best_cell_by_final_ship_frequency(state)
if 1 < squares_remaining(state) <= 4:
    return exhaustive_search(state)
```

### J) Genetic Algorithm (weights)
```
initialize population of chromosomes w ∈ [0,1]^5
for gen in 1..G:
    fitnesses ← parallel_map(evaluate, population)
    elites ← top_k(population, fitnesses, k=ceil(0.1*pop))
    next_pop ← elites
    while |next_pop| < pop:
        pa,pb ← select_parents(population, fitnesses)
        child ← uniform_crossover(pa,pb, rate=0.25)
        child ← gaussian_mutation(child, sigma=0.12, clamp=[0,1])
        next_pop.add(child)
    population ← next_pop
    if gen % SAVE_EVERY==0: save_best(population, path="models/ga_weights.json")
```

### K) Dataset generation
```
for game in 1..GAMES:
    g ← BattleshipGame(AIPlayer, AIPlayer)
    while not g.over and g.turns<100:
        p ← g.current
        X ← encode_state(p.turn_board_state())     # 10×10×3
        y ← encode_true(p.board)                   # 10×10
        write_pickle((X,y))
        g.step()
```

### L) Meta‑learner (prototype)
```
model ← MLP(inputs=20) → [Dense 64 + BN + Dropout] → [Dense 32 + BN + Dropout] → Dense 5 (tanh)
weights_adjustment ← model(features(state))     # in [-1,1]
meta_weights ← clip(base_weights + scale*weights_adjustment, [0,1])
```

## Notes
- Agent 4 loads GA‑tuned weights at startup and uses them for all games until retrained.
- Opponent modeling is present in the codebase but not directly blended into the grid; future work can add an opponent prior map and corresponding weight.
