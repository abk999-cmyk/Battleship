## Data Schemas

### Dataset stream (`data/battleship_supervised.pkl`)
- Format: pickled sequence of tuples `(state_tensor, ship_grid)`.
- `state_tensor`: `float16[10,10,3]` with channels [miss, hit, unknown].
- `ship_grid`: `int8[10,10]` with 1 at true ship coordinates.

### Dashboard move logs (`data/all_moves.csv` and per-game `game_*.json`, `game_*_moves.csv`)
- Per-game JSON keys: `game_id`, `player1`, `player2`, `winner`, `winner_idx`, `move_count`, `duration`, `timestamp`.
- Per-move CSV fields: `game_id`, `move_idx`, `player_idx`, `player_type`, `row`, `col`, `result`.
- Aggregate `all_moves.csv` concatenates per-move entries across games with identical schema.

### App logs (`game_logs/metrics.csv`)
- Columns: `time`, `winner`, `moves`, `duration_sec`.

### Model artifacts
- `models/battleship_heatmap.h5` (CNN prior); optional `models/battleship_heatmap_finetuned.h5` (RL refined).
- `models/ga_weights.json` (GA meta-weights). Example fields: `density`, `neural`, `montecarlo`, `information_gain`, `opponent_model`.
- `models/ai_performance_metrics.pkl` (Agent 3 metrics dict).
- `models/opponent_profiles.pkl` (opponent behaviour profiles).
