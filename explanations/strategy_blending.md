# Strategy Blending — One Map from Many Signals

Simple metaphor: a mixing desk with four channels (Density, Neural, Monte Carlo, Info Gain), plus two shapers: a parity gate (hunt mode) and a neighbor‑of‑hits boost (targeting).

## Blend steps each turn (hunt mode)
1) Start grid = 0.
2) Density: add (density_norm × w_density).
3) Neural: convex blend to preserve scale: grid = (1−w_nn)·grid + w_nn·(nn_norm · grid.max()).
4) Monte Carlo: same style blend with w_mc.
5) Information Gain: same style blend with w_ig (often small or 0).
6) Parity mask (if min ship ≥ 2): grid *= mask.
7) Invalidate illegal cells (−∞) and pick argmax.

## Targeting extras
- If there are active hits, add a neighbor bonus to orthogonal neighbors, scaled by current grid.max(), to finish ships quickly.

## Why this works
- Normalization makes sources comparable.
- Convex blending keeps the shape while letting sources steer the peak.
- Parity accelerates early exploration; neighbor bonus accelerates completion once a ship is found.

## Weights in Agent 4
- Weights come from `self.meta_weights` (GA‑tuned via `models/ga_weights.json`) and stay fixed during a game.
