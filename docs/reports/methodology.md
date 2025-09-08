## Methodology for Experiments and Ablations

### Experimental setup
- Agents: AIAgent4 as primary; baselines: AIAgent3, AIPlayer2, AIPlayer, Ultimate, Naive1–10.
- Environment: 10×10 board; classic ship sizes [5,4,3,3,2]; random non-overlapping placement.
- Hardware: specify CPU/GPU, OS, Python version; TensorFlow 2.17.0.
- Software: `requirements.txt` under virtualenv; versions pinned.

### Protocols
- Supervised pretrain → optional RL fine-tune → GA evolve → batch evaluate (1000+ games per opponent).
- For ablations, fix seeds and evaluate with each sub-grid disabled or weight=0 (N / MC / D / I / OPP), one at a time.
- Report confidence intervals via bootstrapping (10k resamples) on win rate and moves to win.

### Metrics
- Primary: win rate by matchup; secondary: average moves to win; per-move hit rate; distribution of game lengths.
- Efficiency: time per game; memory footprint (optional).

### Statistical validity
- Use identical opponent placement RNG seeds across agent comparisons to reduce variance.
- Perform paired analyses where possible; adjust for multiple comparisons when reporting many ablations.

### Releasing artifacts
- Commit `testing_results.csv`, GA weights (`models/ga_weights.json`), and model hashes; store training configs.
