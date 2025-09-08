## Limitations and Threats to Validity

- Training distribution bias: Supervised CNN trained on self-play (Agent 1) induces policy bias; mitigated via RL fine-tuning and GA blending.
- Opponent diversity: Synthetic opponents may not reflect real human strategies; consider profiling real games via dashboard logs.
- Monte Carlo approximations: Sampling may under-cover rare ship configurations; endgame exhaustive routines help.
- Determinism: Multiprocessing and stochastic sampling require careful seeding; residual nondeterminism may persist across platforms.
- Overfitting GA weights: Fitness measured on a fixed opponent suite; include hold-out opponents or cross-validate.
- Optional dependencies: Some features (graph, SciPy) degrade gracefully; comparisons should report which features were active.
