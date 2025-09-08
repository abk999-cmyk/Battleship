## Meta-Learner

Describes the dense network in `meta_learner.py` that predicts strategy weight adjustments from a compact state feature vector.

### Model
- Input: `META_FEATURES=20` engineered scalars (e.g., progress, mode, remaining ships, min ship size, etc.).
- Body: Dense(64, relu) + BN + Dropout(0.2) → Dense(32, relu) + BN + Dropout(0.2).
- Output: Dense(5, tanh) → adjustments for [density, neural, montecarlo, information_gain, opponent_model] in [−1,1].
- Loss: MSE; Optimizer: Adam(1e-3); epochs≈50 with ES/MC/TB callbacks. Saved to `models/meta_learner.h5`.

### Data
- Synthetic generator produces targets reflecting phase-appropriate strategy emphasis (early density, late MC, targeting mode bias); optional real data loader consumes `data/game_states.pkl`.

### Inference and coupling
- Produces deltas to apply over base meta-weights; could be smoothed and bounded before blending.
- Intended to adapt Agent 3’s blending in real time; current code provides model and trainer; wiring into Agent 3 is straightforward.

### Research considerations
- Ablate each feature group; study stability of weight trajectories and effect on win rate.
- Consider policy-gradient training against win-rate as a downstream objective.
