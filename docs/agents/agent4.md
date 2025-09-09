## Agent 4: GA-Optimized AIAgent3

`agents/AI_agent4.py` defines `AIAgent4`, a thin subclass of `AIAgent3` that automatically loads genetic-algorithm optimized meta-weights at initialization. No decision logic changes are introduced; only the blending weights are evolved.

### Architecture
- Inherits all mechanisms from Agent 3 (density, neural, Monte Carlo, information gain, opponent modeling, graph reasoning).
- On `__init__`, calls `_load_ga_weights()` which attempts to read `models/ga_weights.json` and merges any recognized keys into `self.meta_weights`.

### GA integration
- `training/ga_optimizer.py` evolves weight vectors over generations using a tournament-like GA:
  - Chromosome: `{density, neural, montecarlo, information_gain, opponent_model}` ∈ [0,1]^5
  - Fitness: `100 * win_rate − avg_moves_to_win` measured against a set of fixed opponents (Agent1, Agent2, Ultimate).
  - Operators: uniform crossover per gene (rate 0.25), Gaussian mutation (σ≈0.12 with clamping to [0,1]), elitism (~10%).
  - Evaluation: each chromosome plays multiple games per opponent via `BattleshipGame` in isolated worker processes; only scalar fitness returns to the driver to bound memory usage.
  - Checkpoints: best weights saved to `models/ga_weights.json` every `SAVE_EVERY` generations.

### Runtime behaviour
- When the GA file is present, Agent 4 immediately reflects evolved weights without code changes.
- If not present, Agent 4 falls back to Agent 3 default meta-weights with an informational log line.

### Typical evolved weights (example)
```json
{
  "density": 0.49,
  "neural": 0.60,
  "montecarlo": 0.97,
  "information_gain": 0.00,
  "opponent_model": 0.51
}
```

### Scientific interpretation
- The GA strongly favours Monte Carlo late-game accuracy and keeps density as a structural prior; opponent_model receives moderate weight; information_gain may reduce if MC already captures most value-of-information structure given the constraints.

### Reproducibility
- GA runs are stochastic; set seeds at the Python level per worker if deterministic evolution is required for ablation studies.
