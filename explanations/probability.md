# Probability Density and Parity — Deterministic Counting

Simple metaphor: slide a ship‑shaped stencil over the board. Every legal place the stencil can sit, stamp the cells under it. Do this for every remaining ship and both orientations. Cells stamped most often are where a ship most often could fit.

## What it computes each turn
1) Start a 10×10 grid of zeros.
2) For each remaining ship length and orientation, try every in‑bounds placement.
3) Reject placements that intersect known misses or sunk cells, and require consistency with active hits.
4) For every legal placement, add +1 to each covered unknown cell.
5) Sum across ships; optionally normalize by the max to 0–1.

## Parity mask (efficient hunting)
- Let k = smallest remaining ship length. Probe only cells where (row + col) % k == offset.
- Rotate the offset over time to sweep the board efficiently without missing any k‑length ship.
- In hunt mode the master grid is multiplied by this mask.

## Tiny example (4×4, one length‑3 ship)
- Hit at (1,2), miss at (1,1). Horizontal through (1,2) is illegal (hits miss), so only vertical placements count. The column around (1,2) accumulates 1–2 stamps, making it hottest.

## How it differs from Monte Carlo
- Density counts independent “could‑fit” placements per ship (fast, deterministic).
- Monte Carlo samples whole fleets that must coexist without overlap (globally consistent).

## Strengths and limits
- Strengths: fast, smooth structural prior; great early game with parity.
- Limits: can overrate cells that look good for one ship but block others (MC corrects this).
