## API and Module Reference (Key Classes/Functions)

### board.py
- `class Board(size=10)`: methods `place_ships`, `place_ships_manual`, `attack(coord)`, `all_ships_sunk`, `display(reveal=False)`, `deep_copy()`; attributes `grid`, `ships`, `ship_lookup`.
- `SHIP_SIZES = [5,4,3,3,2]`.

### ship.py
- `class Ship(size, coordinates, orientation)`: methods `is_sunk()`, `check_hit(move_coord)`.

### player.py
- `class Player(name)`: attributes `board`; helpers `turn_board_state()`, `true_ship_grid()`.
- `class HumanPlayer(Player, manual_setup='y'|'n')`: `take_turn(opponent_board)`.

### game.py
- `class BattleshipGame(player1, player2)`: `play()`, `step()`, `step_manual(move)`, `is_over()`, attributes `current`, `opponent`, `winner`, `move_count`, `turns`.

### AI_agent.py (Agent 1)
- `class AIPlayer(Player)`: baseline agent with hunt/target logic, probability grid, optional neural heatmap.

### AI_agent2.py (Agent 2)
- `class AIPlayer2(Player)`: see docs `agents/agent2.md`. Public: `take_turn()`, `view_display()`.

### AI_agent3.py (Agent 3)
- `class AIAgent3(Player)`: see docs `agents/agent3.md`. Public: `take_turn()`, `view_display()`, `learn_from_game()`.

### AI_agent4.py (Agent 4)
- `class AIAgent4(AIAgent3)`: GA weight loader; see `agents/agent4.md`.

### AI_testing_agents.py
- `BaseNaiveAgent` and Naive1–10, `UltimateBattleshipAgent`: deterministic/simple baselines for benchmarking.

### Training scripts
- `generate_dataset.py`: CLI `--games`, `--workers`, `--out`.
- `train_heatmap.py`: trains CNN from dataset file.
- `rl_finetune.py`: CLI `--games`, `--workers`, `--batch`.
- `ga_optimizer.py`: CLI `--pop`, `--gens`, `--cpus`.
- `meta_learner.py`, `opponent_model.py`: training entry points; see `nn/` docs.

### Apps and entry points
- `battleship_dashboard.py`: run directly to launch GUI.
- `main.py`: batch evaluation CLI with `--watch/--opp/--delay/--games`.
