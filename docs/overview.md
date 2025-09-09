## System Overview

This repository implements a full Battleship AI research stack: deterministic game engine, multiple AI agents, training pipelines (supervised, RL, GA), a rich Tk-based research dashboard, and supporting utilities for opponent modeling and meta-learning. This document provides a high-level architecture and conceptual map to all modules.

### Top-level components
- Game core: `core/board.py`, `core/ship.py`, `core/player.py`, `core/game.py` (see `environment.md`).
- Agents: `agents/AI_agent.py` (v1), `agents/AI_agent2.py` (v2 → `agents/agent2.md`), `agents/AI_agent3.py` (v3 → `agents/agent3.md`), `agents/AI_agent4.py` (v4 → `agents/agent4.md`), `agents/AI_testing_agents.py` (baselines).
- Training/data: `training/generate_dataset.py`, `training/train_heatmap.py`, `training/rl_finetune.py`, `training/ga_optimizer.py`, `agents/meta_learner.py`, `agents/opponent_model.py` (see `training/pipelines.md`, `nn/heatmap_model.md`, `nn/meta_learner.md`, `nn/opponent_model.md`).
- Apps: `apps/battleship_dashboard.py` (research GUI → `ui/dashboard_webapp.md`), `apps/battleship_app.py`, `apps/battleship_app_play.py`.
- Orchestration: `apps/main.py` (batch evaluation → `evaluation/metrics.md`), `scripts/setup.py`/`scripts/setup_tensorflow.py`.

### Architecture diagram
- Environment (Board, Ships) → Game Engine (`BattleshipGame`) → Agents (choose move) → Board updates → Logs/Analytics.
- Training pipelines read/write `data/`, `models/`, `logs/`.
- Dashboard reads models and data, visualizes and runs simulations.

### Key data contracts
- Board grid semantics: 0 unknown, 1 miss, 2 hit; `ship_lookup` mapping coordinates→`Ship`.
- Agent interface: `take_turn(opponent_board)` updates `last_move`, `last_result`; maintains `available_moves`, `result_grid`, `hits`, `sunk_ships_info`.
- Heatmap NN I/O: input tensor 10×10×3 (miss, hit, unknown), output 10×10 probability map.

### Reproducibility
See `repro/guide.md` for venv setup, deterministic seeds, hardware notes, and script commands.
