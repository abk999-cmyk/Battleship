## Evaluation Protocols and Metrics

### Batch evaluation
- `main.py` benchmarks `AIAgent4` against a suite of baseline opponents (Naive1–10 and Ultimate). For each opponent, it plays `NUM_GAMES_PER_AGENT` games (default 1000), tracking wins, losses, and average moves to win.
- Results serialized to `testing_results.csv` with fields [opponent, main_ai_wins, main_ai_losses, avg_moves_to_win].

### Dashboard analytics
- `battleship_dashboard.py`’s `GameEngine` logs per-game JSON metadata and per-move CSVs to `data/` and maintains `all_moves.csv`.
- `AnalyticsManager` computes win rates by agent and average moves; provides heatmap visualizations of attack frequencies and can render win-rate bar charts.

### Core metrics
- Win rate by matchup; average moves to victory; per-move hit rate; distribution of game lengths; time per move (optional via logs).

### Statistical treatment
- Report mean±95% CI via bootstrapping over games for win rate and moves to win; compare agents via paired tests when matched on seeds/opponents.
- Stratify by opponent type to surface failure modes (e.g., clustering opponents vs spread-out).

### Reproducible benchmark commands
- Supervised pretrain → RL fine-tune → GA evolve → batch evaluate (see `repro/guide.md`).
- Ensure identical `requirements.txt` and TensorFlow 2.17.0; seed Python RNG across scripts.
