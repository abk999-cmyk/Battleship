## Training Pipelines

This document formalizes all training/evolution procedures present in the repository.

### Supervised pretraining (heatmap prior)
- Scripts: `training/generate_dataset.py` (multiprocess self-play sampler), `training/train_heatmap.py` (Keras trainer).
- Dataset: pickled stream of (X,Y) where X=10×10×3 planes (miss/hit/unknown), Y=10×10 binary mask of true ship cells.
- Training: shallow CNN (≈4 conv layers) with BCE, Adam(1e-3), 12 epochs, batch 256; outputs `models/battleship_heatmap.h5`.
- Purpose: learn a spatial prior over ship occupancy conditional on partial observations.

### RL fine-tuning (REINFORCE over logits)
- Script: `training/rl_finetune.py`.
- Setup: load `models/battleship_heatmap.h5`; spawn W worker processes running self-play with `AI_agent.AIPlayer` using the model as a policy over cells.
- Trajectories: per-visit state tensors and chosen actions (flattened cell indices); returns terminal reward R∈{0,1}.
- Loss: vectorized REINFORCE over all (s,a) with shared return R; gradients aggregated per `--batch` games; optimizer Adam(1e-4).
- Checkpoints: periodic `models/rl_finetune_*.h5`; final `models/battleship_heatmap_finetuned.h5`.
- Purpose: align the heatmap logits with win-rate via on-policy sampling.

### Genetic Algorithm (meta-weight evolution)
- Script: `training/ga_optimizer.py`; objective evaluates AIAgent3's blending weights.
- Chromosome: 5 meta-weights ∈ [0,1]. Operators: crossover (p=0.25), Gaussian mutation (σ≈0.12), elitism (10%).
- Evaluation: matches against opponents (Agent1, Agent2, Ultimate) for `GAMES_PER_OPP` games; fitness = 100×win_rate − avg_moves.
- Memory control: workers return only scalars; maxtasksperchild recycles memory; explicit GC per game.
- Output: `models/ga_weights.json` loaded by Agent 4 at runtime.

### Meta-learner (supervised proxy)
- Script: `agents/meta_learner.py` trains a dense regressor to output weight adjustments given engineered state features.
- Current integration: model training and IO provided; runtime coupling to AIAgent3 is intended but optional.

### Opponent modeling
- `agents/opponent_model.py` trains a CNN on synthetic/real opponent tendencies, saving `models/opponent_model.h5`; utilities manage `models/opponent_profiles.pkl`.

### Data and logs
- Datasets in `data/`; models in `models/`; training logs/plots in `logs/`.
- Dashboard and apps also write per-game JSON/CSV under `data/`, consumable by analytics.

### Reproducibility and compute
- Seed Python RNG before GA or RL runs for deterministic sampling; ensure consistent TensorFlow version (2.17.0).
- Multiprocessing uses spawn; avoid GPU contention by pinning CPU where appropriate; use smaller batches on laptops.
