'''
# Battleship AI Research Platform

A comprehensive research platform for developing and testing advanced AI agents for the classic Battleship game. This project combines traditional game AI strategies with state-of-the-art machine learning techniques.

![Battleship AI Dashboard](./docs/dashboard_preview.png)

## Features

- **Advanced AI Agents**:
  - **AIAgent3**: Cutting-edge agent combining probabilistic, neural, and graph-based methods
  - **AIPlayer2**: Probabilistic agent with neural network integration
  - **AIPlayer**: Basic probabilistic agent
  - Various benchmark agents for testing

- **Research Tools**:
  - **Interactive Dashboard**: Visualize game states, AI strategies, and performance metrics
  - **Batch Simulation**: Run thousands of games to benchmark AI performance
  - **Comprehensive Analytics**: Analyze win rates, efficiency, and strategy effectiveness

- **AI Innovations**:
  - **Neural Heatmaps**: CNN-based ship placement predictions
  - **Opponent Modeling**: Adaptive AI that learns from opponent patterns
  - **Meta-Learning**: Adjusts strategy weights dynamically during games
  - **Graph-Based Ship Reasoning**: Tracks viable ship configurations
  - **Information Theory**: Selects moves with maximum information gain

- **Continuous Learning**: AIs improve from every game they play

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/battleship-ai.git
cd battleship-ai
```

2. Run the setup script:
```bash
python setup.py
```

The setup script will:
- Install all required dependencies
- Create necessary directories
- Generate sample assets
- Test the installation
- Create a convenient launch script

## Requirements

- Python 3.8 or higher
- Dependencies (installed automatically by setup.py):
  - numpy
  - pandas
  - matplotlib
  - tensorflow
  - scikit-learn
  - networkx
  - customtkinter
  - and more...

## Usage

### Launch the Dashboard

Run the launch script created during installation:

- On Windows: `launch_battleship.bat`
- On macOS/Linux: `./launch_battleship.sh`

Or run directly:

```bash
python battleship_dashboard.py
```

### Dashboard Tabs

1. **Game**: Play individual games between different agent types or human vs AI
2. **Batch Simulation**: Run large batches of games to compare agent performance
3. **Analytics**: Visualize performance data, heatmaps, and comparative statistics
4. **Settings**: Configure the platform, manage models, and export data

### Training Models

The platform includes several trainable neural models:

```bash
# Train the primary heatmap model
python train_heatmap.py

# Train the meta-learning model
python meta_learner.py

# Train the opponent modeling network
python opponent_model.py
```

## AI Agent Architecture

The flagship `AIAgent3` combines multiple strategies with dynamic weighting:

1. **Density-Based Probability**: Classical ship placement counting
2. **Neural Heatmap**: CNN predictions of ship locations
3. **Monte Carlo Simulation**: Samples possible ship configurations
4. **Information Theory**: Selects moves that maximize information gain
5. **Opponent Modeling**: Adapts to opponent tendencies
6. **Graph Representation**: Models ship placement constraints
7. **Meta-Learning**: Adjusts strategy weights based on game state

## Benchmark Results

| Agent         | Win Rate | Avg Moves/Win | Hit Rate |
|---------------|----------|---------------|----------|
| AIAgent3      | 96.2%    | 43.7          | 41.5%    |
| AIPlayer2     | 86.4%    | 52.3          | 32.7%    |
| AIPlayer      | 75.9%    | 63.2          | 26.9%    |
| Ultimate      | 62.8%    | 74.8          | 22.7%    |
| NaiveAgent10  | 45.3%    | 82.4          | 20.6%    |

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Original AIPlayer2 implementation that achieved 96% win rate
- Battleship probability theory research by DataGenetics
- Neural network architecture inspired by DeepMind's AlphaGo and AlphaZero
'''