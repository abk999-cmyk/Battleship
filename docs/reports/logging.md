## Logging and Error Handling

### Logging
- Global logs: `logs/` directory; dashboard logs to `logs/dashboard.log`, agents and training scripts log to their respective files (e.g., `logs/ai_agent3.log`, `logs/opponent_model.log`, `logs/meta_learner.log`).
- Metrics: `battleship_app.py` logs per-batch metrics to `game_logs/metrics.csv`; dashboard writes game metadata and moves to `data/`.

### Error handling
- Agents (AIAgent3): defensive imports for TensorFlow/SciPy/NetworkX; graceful degradation when deps absent.
- Dashboard: thread-safe UI updates; exception hook displays messagebox and logs traces; matplotlib uses Agg backend to avoid UI-thread conflicts.
- GA optimizer: worker lifetimes and explicit GC to prevent memory bloat; main process only receives scalar fitness.

### Recommendations
- Centralize logging levels via environment variable (e.g., `LOGLEVEL=INFO`); rotate logs for long runs.
- Capture software/hardware versions in experiment metadata for reproducibility.
