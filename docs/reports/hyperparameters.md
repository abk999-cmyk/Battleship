## Hyperparameters and Compute Budgets

### Supervised CNN
- Epochs: 12; Batch: 256; LR: 1e-3; Opt: Adam; Params: ~50–100K.
- Compute: CPU/GPU optional; fits in seconds–minutes on laptop CPU for small datasets.

### RL Fine-tune
- Games: 20k (configurable); Workers: CPU cores; Batch: 256 games per gradient; LR: 1e-4.
- Compute: multi-hour on CPU; checkpoint every 1000 processed games.

### GA Evolution
- Pop: 40; Gens: 50; CPU workers: ~half of cores; GAMES_PER_OPP: 5–10 for throughput.
- Compute: hours to days depending on evaluation volume.

### Agent runtime
- MC samples: base 1500; triples in endgame; parity stride = min remaining ship size.
- NN blending α: 0.6 hunt / 0.2 target (Agent 2); GA may override effective weighting.

### Notes
- For constrained machines, reduce GA population/gens and RL games; prefer GA first for strong gains.
