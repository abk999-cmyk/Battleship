# Neural Network Heatmap — Pattern Recognition from Experience

Simple metaphor: a seasoned sonar operator sees today’s hits, misses, and fog, and “paints” where ships usually hide in patterns like this.

## What it computes each turn
- Build a 10×10×3 image: [miss plane, hit plane, unknown plane].
- Run a small CNN; output a 10×10 map with scores in [0,1].
- Mask out illegal/already‑played cells, then normalize.

## Why it helps
- Learns priors from many games: straight ships, typical extensions of partial hits, realistic spacing.
- Smooth, low‑noise guidance when evidence is sparse; great for early/mid game.

## Tiny example
- Two horizontal hits with water above/below → the network boosts left/right neighbors, forming a ridge along that row.

## Notes
- Recomputed every move from the live board state.
- It doesn’t enforce whole‑fleet consistency; Monte Carlo does. Blending corrects biases while keeping learned patterns.
