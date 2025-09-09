# Battleship AI Research Platform

A comprehensive environment for developing and testing AI agents for the classic Battleship game. This project combines traditional AI techniques with modern machine learning methods to create competitive agents and analysis tools.

## Features
- **Advanced AI Agents**
  - **AIAgent3**: Combines probabilistic, neural and graph-based methods
  - **AIPlayer2**: Probabilistic agent with neural network integration
  - **AIPlayer**: Basic probabilistic agent
  - Additional benchmark agents for comparison
- **Research Tools**
  - Interactive dashboard to visualise games and metrics
  - Batch simulation utilities for large experiments
  - Analytics including win rates and heatmaps
- **AI Innovations**
  - Neural heatmaps for ship prediction
  - Opponent modelling and meta-learning
  - Graph-based reasoning and information theory heuristics
- **Continuous Learning**: Agents improve after every game

## Installation
```bash
git clone https://github.com/yourusername/battleship-ai.git
cd battleship-ai
python scripts/setup.py
```
The setup script installs dependencies, creates sample assets and generates a launch script.

## Usage
Launch the dashboard:
```bash
python apps/battleship_dashboard.py
```

### Dashboard Tabs
1. **Game** – play individual games or human vs. AI
2. **Batch Simulation** – run many games for benchmarking
3. **Analytics** – explore performance statistics and heatmaps
4. **Settings** – manage models and export data

### Training Models
```bash
python training/train_heatmap.py
python agents/meta_learner.py
python agents/opponent_model.py
```

## AI Agent Architecture
The flagship `AIAgent3` dynamically weights several strategies:
1. Density-based probability counting
2. Neural heatmap predictions
3. Monte Carlo simulation
4. Information theory move selection
5. Opponent modelling
6. Graph representation of ship placement
7. Meta-learning for strategy tuning

## Benchmark Results
| Agent         | Win Rate | Avg Moves/Win | Hit Rate |
|---------------|---------:|--------------:|---------:|
| AIAgent3      | 96.2%    | 43.7          | 41.5%    |
| AIPlayer2     | 86.4%    | 52.3          | 32.7%    |
| AIPlayer      | 75.9%    | 63.2          | 26.9%    |
| Ultimate      | 62.8%    | 74.8          | 22.7%    |
| NaiveAgent10  | 45.3%    | 82.4          | 20.6%    |

## License
MIT License. See the `LICENSE` file for details.

## Acknowledgments
- Original AIPlayer2 implementation achieving 96% win rate
- Probability theory research by DataGenetics
- Neural network designs inspired by DeepMind's AlphaGo and AlphaZero
