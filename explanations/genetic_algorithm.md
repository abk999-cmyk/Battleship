# Genetic Algorithm — How the Blend Weights Are Learned

Simple metaphor: we treat each weight as a volume knob. A population of knob settings “plays” matches; the best‑sounding mixes survive, breed, and mutate until we find a strong set of knobs.

## What it tunes
- A vector in [0,1]: {density, neural, montecarlo, information_gain, opponent_model}.
- In Agent 4, these become `self.meta_weights` loaded from `models/ga_weights.json` at startup.

## How candidates are scored (fitness)
- Play multiple games vs fixed opponents.
- Fitness = 100 × win_rate − average_moves_to_win.
- Optimizes for winning often and quickly.

## How new candidates are created
- Random initialization in [0,1].
- Mutation: add small Gaussian noise per gene, clamp to [0,1].
- Crossover: per gene, pick from parent A or B (e.g., 25% take from B).
- Elitism: keep top ~10% unchanged each generation.

## Why information_gain might be 0
- In current results, Density+NN+MC already perform strongly; GA pushes IG down if it doesn’t help on the evaluation set.
- Keep the knob—if IG improves later, a new GA run can raise it.

## Practical notes
- Checkpoints write best weights to `models/ga_weights.json`.
- Agent 4 loads them automatically; weights stay constant during a game.
- To generalize better, include diverse opponents and re‑run the GA.
