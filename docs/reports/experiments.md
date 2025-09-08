## Experiment Reports and Results

This document aggregates how to report results and links to existing summaries.

### Source results
- `results.md`: narrative of batch simulation performance.
- `ai_evolution_story.md`: evolutionary timeline and qualitative analysis.

### Reporting template
- Benchmark settings: opponent suite, games per opponent, hardware, seeds, TF version.
- Metrics: win rate with 95% CI; average moves to win; per-move hit rate; length distribution.
- Ablations: disable each sub-grid (N/MC/D/I/OPP) and report deltas; report GA weight vectors.
- Plots: win-rate bar chart; attack-frequency heatmaps; weight trajectories if meta-learner integrated.

### Reproduction checklist
- venv with `requirements.txt`; verify TF 2.17.0.
- Train supervised model (or use provided); optional RL fine-tune; evolve GA; confirm `models/ga_weights.json` present for Agent 4.
- Run `main.py --games 1000` and store `testing_results.csv`.
- Archive `data/` CSV/JSON logs; include `models/` hashes and training configs.
