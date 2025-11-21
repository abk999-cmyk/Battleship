# Monte Carlo Occupancy — Whole‑Fleet Sampling (not MCTS)

Simple metaphor: repeatedly place the entire remaining fleet in the fog in every way that fits all clues. Count how often each square ends up under a ship. More hits in sampling → higher score.

## What it computes each turn
1) Gather constraints: misses, sunk cells, active hits, remaining ship sizes.
2) Randomly place all remaining ships (h/v) without overlap and respecting clues.
3) Accept only layouts that cover all known hits.
4) For each accepted layout, add +1 to every occupied, unshot cell.
5) Repeat many times; divide by the max to get a 0–1 heatmap.

## Tiny example
- Two adjacent hits at (r,c) and (r,c+1), misses above/below.
- Many accepted layouts extend left/right; few (or none) extend vertically due to misses.
- Left/right neighbors accumulate counts → become the hottest next shots.

## How it differs from density
- Density: counts per‑ship “could‑fit” placements independently (fast, local).
- Monte Carlo: samples whole fleets that must co‑exist (global consistency).

## Strengths and limits
- Strengths: respects inter‑ship constraints; excels mid/late game precision.
- Limits: stochastic and costlier; early game can be flat or noisy.
