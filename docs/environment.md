## Game Environment and Mechanics

This document specifies the Battleship environment as implemented in `board.py`, `ship.py`, `player.py`, and the headless engine `game.py`.

### Board
- Size: 10×10 (configurable in `Board(size=10)`).
- Representation: `grid: np.ndarray[int]` where 0 unknown, 1 miss, 2 hit.
- Ship sizes: `[5, 4, 3, 3, 2]` (`SHIP_SIZES`).
- Placement: `place_ships()` random, non-overlapping, axis-aligned; `place_ships_manual()` supports guided placement.
- `ship_lookup`: dict[(r,c)→Ship] for O(1) hit resolution.
- `attack(coord)` returns one of {"hit","sunk","miss","already","invalid"} with robust tolerance for tuple/list/ndarray/int index inputs.
- `all_ships_sunk()` closes out a game when all `Ship.hits` cover `Ship.size`.
- `display(reveal=False)` provides human-readable board; `deep_copy()` ensures safe snapshotting for simulation.

### Ship
- Tracks size, coordinates, orientation, and per-cell hits. `is_sunk()` checks completion; `check_hit()` mutates state.

### Player
- Base `Player` stores `name`, `board`; helpers: `turn_board_state()` (agent-visible result grid) and `true_ship_grid()` (label mask for dataset generation).
- `HumanPlayer` optionally allows manual placement and CLI-driven firing.

### Game engine (`BattleshipGame`)
- Minimal driver for AI-vs-AI training/eval: maintains `current`, `opponent`, `move_count`, `winner`.
- `play()` loops `step()` until `is_over()`; `step()` delegates to agent `take_turn()`, then `post_move()` updates turns and checks terminal condition.
- `step_manual(move)` allows external RL drivers to apply chosen actions directly while still updating agent state if supported.
- No-legal-move forfeit: if `current.available_moves` is empty on handover, `winner` defaults to the opponent.

### Semantics and invariants
- A move changes board state exactly once; repeated targeting of the same cell returns "already" and is filtered by agents.
- `result_grid` is agent-side belief state separate from the true board; agents must keep it consistent with attacks.

### Telemetry
- Apps write CSV logs per move and aggregate summaries to `data/` and `logs/`. See `battleship_dashboard.py` for detailed schema.
