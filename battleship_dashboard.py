"""
Battleship AI Research Dashboard

A clean, high-performance dashboard for visualizing, testing, and analyzing
Battleship AI agents with integrated neural network support.

Features:
- Interactive game visualization
- Human vs AI gameplay
- Batch simulation for AI benchmarking
- Performance analytics and visualization
- Support for neural network-based AI agents
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import time
import os
import json
import csv
from pathlib import Path
import threading
import logging
from datetime import datetime
import random
import uuid
import sys
import traceback

# Set up matplotlib with thread-safe backend
import matplotlib
matplotlib.use('Agg', force=True)  # Use Agg backend for thread-safety
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Import AI components - only import what we use
from player import HumanPlayer
from board import Board
from AI_agent import AIPlayer
from AI_agent2 import AIPlayer2
from AI_agent4 import AIAgent4

# Ship sizes
SHIP_SIZES = [5, 4, 3, 3, 2]  # Directly define here instead of importing

# Try to import AIAgent3 if available
HAS_AI_AGENT3 = False
AIAgent3 = None  # Define outside try/except to avoid undefined variable warnings
try:
    from AI_agent3 import AIAgent3
    HAS_AI_AGENT3 = True
    print("AI_agent3 imported successfully")
except ImportError:
    print("AI_agent3 not available. Advanced AI features will be disabled.")

# Import testing agents
from AI_testing_agents import (
    NaiveAgent1, NaiveAgent2, NaiveAgent3, NaiveAgent4,
    NaiveAgent5, NaiveAgent6, NaiveAgent7, NaiveAgent8,
    NaiveAgent9, NaiveAgent10, UltimateBattleshipAgent
)

# Setup directories
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
for dir_path in [MODELS_DIR, DATA_DIR, LOGS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'dashboard.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('BattleshipDashboard')

# Constants
BOARD_SIZE = 10
CELL_SIZE = 40
WATER_COLOR = '#1E90FF'   # Water/ocean
SHIP_COLOR = '#808080'    # Ship
HIT_COLOR = '#FF4C4C'     # Hit marker
MISS_COLOR = '#4C72B0'    # Miss marker

# Map agent classes to names
AGENT_CLASSES = {
    "AI_agent4": AIAgent4,
    "AI_Agent2": AIPlayer2,
    "AI_Agent1": AIPlayer,
    "Ultimate": UltimateBattleshipAgent,
    "Naive1": NaiveAgent1,
    "Naive2": NaiveAgent2,
    "Naive3": NaiveAgent3,
    "Naive4": NaiveAgent4,
    "Naive5": NaiveAgent5,
    "Naive6": NaiveAgent6,
    "Naive7": NaiveAgent7,
    "Naive8": NaiveAgent8,
    "Naive9": NaiveAgent9,
    "Naive10": NaiveAgent10,
    "Human": HumanPlayer
}

# Add AI_Agent3 if available
if HAS_AI_AGENT3:
    AGENT_CLASSES["AI_Agent3"] = AIAgent3


class BoardCanvas(tk.Canvas):
    """Canvas for displaying a Battleship board"""

    def __init__(self, parent, board_size=BOARD_SIZE, cell_size=CELL_SIZE, **kwargs):
        width = board_size * cell_size
        height = board_size * cell_size
        super().__init__(
            parent,
            width=width,
            height=height,
            bg='#2C3E50',
            highlightthickness=0,
            **kwargs
        )

        # Board properties
        self.board_size = board_size
        self.cell_size = cell_size
        self.cell_padding = 2

        # Board state
        self.cells = {}  # Cell rectangles
        self.overlays = {}  # Overlays like ships, hits, misses
        self.board = None  # Reference to game board
        self.show_ships = False  # Whether to show ships
        self.show_probabilities = False  # Whether to show probability heatmap
        self.agent = None  # AI agent for probability visualization

        # Draw initial grid
        self._draw_grid()

    def _draw_grid(self):
        """Draw the initial empty board grid"""
        for r in range(self.board_size):
            for c in range(self.board_size):
                x1 = c * self.cell_size
                y1 = r * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size

                # Create water cell
                cell = self.create_rectangle(
                    x1, y1, x2, y2,
                    fill=WATER_COLOR,
                    outline='#2C3E50',
                    width=1,
                    tags=f"cell_{r}_{c}"
                )
                self.cells[(r, c)] = cell

                # Add row/col indicators around the grid
                if r == 0:
                    self.create_text(
                        x1 + self.cell_size/2,
                        y1 - 8,
                        text=str(c),
                        fill='white',
                        font=('Arial', 8)
                    )
                if c == 0:
                    self.create_text(
                        x1 - 8,
                        y1 + self.cell_size/2,
                        text=str(r),
                        fill='white',
                        font=('Arial', 8)
                    )

    def bind_to_board(self, board, agent=None):
        """Associate with a board and optionally an agent"""
        self.board = board
        self.agent = agent
        self.refresh()

    def refresh(self):
        """Refresh the board display"""
        if not self.board:
            return

        # Clear any existing overlays
        for overlay_id in list(self.overlays.values()):
            self.delete(overlay_id)
        self.overlays = {}

        # Draw ships if enabled
        if self.show_ships and hasattr(self.board, 'ship_lookup'):
            self._draw_ships()

        # Draw hits and misses
        if hasattr(self.board, 'grid'):
            self._draw_hits_misses()

        # Draw probability heatmap if enabled and agent has the capability
        if self.show_probabilities and self.agent and hasattr(self.agent, 'compute_probability_grid'):
            try:
                prob_grid = self.agent.compute_probability_grid()
                self._draw_probability_heatmap(prob_grid)
            except Exception as e:
                logger.error(f"Error computing probability grid: {e}")

    def _draw_ships(self):
        """Draw ships on the board"""
        for (r, c), ship in self.board.ship_lookup.items():
            # Create ship rectangle
            x1 = c * self.cell_size + self.cell_padding
            y1 = r * self.cell_size + self.cell_padding
            x2 = (c + 1) * self.cell_size - self.cell_padding
            y2 = (r + 1) * self.cell_size - self.cell_padding

            ship_id = self.create_rectangle(
                x1, y1, x2, y2,
                fill=SHIP_COLOR,
                outline='',
                tags=f"ship_{r}_{c}"
            )
            self.overlays[(r, c, 'ship')] = ship_id

    def _draw_hits_misses(self):
        """Draw hits and misses on the board"""
        for r in range(self.board_size):
            for c in range(self.board_size):
                cell_value = self.board.grid[r, c]

                if cell_value == 1:  # Miss
                    self._draw_miss(r, c)
                elif cell_value == 2:  # Hit
                    self._draw_hit(r, c)

    def _draw_hit(self, r, c):
        """Draw a hit marker at position r,c"""
        # Center of the cell
        x = c * self.cell_size + self.cell_size / 2
        y = r * self.cell_size + self.cell_size / 2
        radius = self.cell_size / 3

        # Draw hit marker (red circle)
        hit_id = self.create_oval(
            x - radius, y - radius,
            x + radius, y + radius,
            fill=HIT_COLOR,
            outline='white',
            width=2,
            tags=f"hit_{r}_{c}"
        )
        self.overlays[(r, c, 'hit')] = hit_id

        # Add X in the circle
        line1_id = self.create_line(
            x - radius/2, y - radius/2,
            x + radius/2, y + radius/2,
            fill='white',
            width=2,
            tags=f"hit_x1_{r}_{c}"
        )
        line2_id = self.create_line(
            x - radius/2, y + radius/2,
            x + radius/2, y - radius/2,
            fill='white',
            width=2,
            tags=f"hit_x2_{r}_{c}"
        )
        self.overlays[(r, c, 'hit_x1')] = line1_id
        self.overlays[(r, c, 'hit_x2')] = line2_id

    def _draw_miss(self, r, c):
        """Draw a miss marker at position r,c"""
        # Center of the cell
        x = c * self.cell_size + self.cell_size / 2
        y = r * self.cell_size + self.cell_size / 2
        radius = self.cell_size / 4

        # Draw miss marker (blue circle)
        miss_id = self.create_oval(
            x - radius, y - radius,
            x + radius, y + radius,
            fill=MISS_COLOR,
            outline='white',
            width=1,
            tags=f"miss_{r}_{c}"
        )
        self.overlays[(r, c, 'miss')] = miss_id

    def _draw_probability_heatmap(self, prob_grid):
        """Draw probability heatmap overlay"""
        if prob_grid is None:
            return

        # Check if there are any positive values
        max_prob = prob_grid.max()
        if max_prob <= 0:
            return

        # Normalize the grid
        norm_grid = prob_grid / max_prob

        # Create colormap
        colormap = plt.cm.get_cmap('viridis')

        # Draw overlay for available moves
        if hasattr(self.agent, 'available_moves'):
            available_moves = self.agent.available_moves

            for r in range(self.board_size):
                for c in range(self.board_size):
                    if (r, c) in available_moves:
                        prob_value = float(norm_grid[r, c])  # Ensure float type
                        if prob_value > 0.01:  # Only draw significant probabilities
                            # Get color from colormap
                            color_rgba = colormap(prob_value)
                            color_hex = "#{:02x}{:02x}{:02x}".format(
                                int(color_rgba[0]*255),
                                int(color_rgba[1]*255),
                                int(color_rgba[2]*255)
                            )

                            # Draw overlay rectangle
                            x1 = c * self.cell_size + self.cell_padding
                            y1 = r * self.cell_size + self.cell_padding
                            x2 = (c + 1) * self.cell_size - self.cell_padding
                            y2 = (r + 1) * self.cell_size - self.cell_padding

                            overlay_id = self.create_rectangle(
                                x1, y1, x2, y2,
                                fill=color_hex,
                                stipple="gray50",  # For transparency
                                tags=f"prob_{r}_{c}"
                            )
                            self.overlays[(r, c, 'prob')] = overlay_id

                            # Add text label for high probability cells
                            if prob_value > 0.5:
                                text_id = self.create_text(
                                    x1 + (x2-x1)/2,
                                    y1 + (y2-y1)/2,
                                    text=f"{prob_value:.2f}",
                                    fill="white",
                                    font=("Arial", 8, "bold"),
                                    tags=f"prob_text_{r}_{c}"
                                )
                                self.overlays[(r, c, 'prob_text')] = text_id

    def highlight_cell(self, r, c, color='#3498DB', duration=500):
        """Temporarily highlight a cell for visual feedback"""
        # Calculate cell coordinates
        x1 = c * self.cell_size + 1
        y1 = r * self.cell_size + 1
        x2 = (c + 1) * self.cell_size - 1
        y2 = (r + 1) * self.cell_size - 1

        # Create highlight rectangle
        highlight = self.create_rectangle(
            x1, y1, x2, y2,
            outline=color,
            width=3,
            tags=f"highlight_{r}_{c}"
        )

        # Schedule removal of highlight
        self.after(duration, lambda: self.delete(highlight))

    def toggle_ships(self):
        """Toggle ship visibility"""
        self.show_ships = not self.show_ships
        self.refresh()
        return self.show_ships

    def toggle_probabilities(self):
        """Toggle probability heatmap"""
        self.show_probabilities = not self.show_probabilities
        self.refresh()
        return self.show_probabilities

    def clear(self):
        """Reset the board display"""
        # Clear all overlays
        for overlay_id in list(self.overlays.values()):
            self.delete(overlay_id)
        self.overlays = {}

        # Clear all cells
        for cell_id in list(self.cells.values()):
            self.delete(cell_id)
        self.cells = {}

        # Redraw grid
        self._draw_grid()


class GameEngine:
    """Engine for managing game state and simulations"""

    def __init__(self, callback=None):
        """Initialize the game engine"""
        self.player1 = None
        self.player2 = None
        self.current_player = None
        self.opponent = None
        self.game_over = False
        self.winner = None
        self.move_count = 0
        self.start_time = None
        self.callback = callback
        self.last_move = None
        self.last_result = None
        self.game_id = str(uuid.uuid4())[:8]

        # For batch simulations
        self.paused = False
        self.simulation_thread = None
        self.batch_stats = {
            'games_played': 0,
            'player1_wins': 0,
            'player2_wins': 0,
            'avg_moves': 0,
            'total_time': 0
        }
        self.moves_history = []

    def setup_game(self, player1_type, player2_type, player1_name="Player 1", player2_name="Player 2"):
        """Set up a game with the given player types"""
        try:
            # Create players
            if player1_type not in AGENT_CLASSES or player2_type not in AGENT_CLASSES:
                raise ValueError(f"Invalid player type: {player1_type} or {player2_type}")

            p1_class = AGENT_CLASSES[player1_type]
            p2_class = AGENT_CLASSES[player2_type]

            # Initialize with proper parameters
            if player1_type == "Human":
                self.player1 = p1_class(player1_name, manual_setup='n')
            else:
                self.player1 = p1_class(player1_name)

            if player2_type == "Human":
                self.player2 = p2_class(player2_name, manual_setup='n')
            else:
                self.player2 = p2_class(player2_name)

            # Reset game state
            self.current_player = self.player1
            self.opponent = self.player2
            self.game_over = False
            self.winner = None
            self.move_count = 0
            self.start_time = time.time()
            self.game_id = str(uuid.uuid4())[:8]
            self.moves_history = []

            logger.info(f"Game set up: {player1_type}({player1_name}) vs {player2_type}({player2_name})")
            return True

        except Exception as e:
            logger.error(f"Error setting up game: {e}")
            return False

    def make_move(self, row, col):
        """Process a move at the given coordinates"""
        if self.game_over:
            return "game_over"

        # Make the move
        result = self.opponent.board.attack((row, col))
        self.last_move = (row, col)
        self.last_result = result

        # Log the move
        player_idx = 1 if self.current_player == self.player1 else 2
        self.moves_history.append({
            'game_id': self.game_id,
            'move_idx': self.move_count,
            'player_idx': player_idx,
            'player_type': self.current_player.__class__.__name__,
            'row': row,
            'col': col,
            'result': result
        })

        self.move_count += 1

        # Update AI player state if possible
        if hasattr(self.current_player, 'update_state'):
            try:
                self.current_player.update_state((row, col), result, self.opponent.board)
            except Exception as e:
                logger.error(f"Error updating player state: {e}")

        # Check for game over
        if self.opponent.board.all_ships_sunk():
            self.game_over = True
            self.winner = self.current_player
            self._handle_game_over()
            return "game_over"

        # Switch players
        self.current_player, self.opponent = self.opponent, self.current_player
        return result

    def ai_move(self):
        """Make a move for the current AI player"""
        if self.game_over or not self.current_player:
            return None

        # Skip if human player
        if self.current_player.__class__.__name__ == "HumanPlayer":
            return None

        try:
            # Let the AI choose its move
            result = self.current_player.take_turn(self.opponent.board)

            # Get the coordinates that were played
            if hasattr(self.current_player, 'last_move'):
                row, col = self.current_player.last_move
                self.last_move = (row, col)
                self.last_result = result

                # Log the move
                player_idx = 1 if self.current_player == self.player1 else 2
                self.moves_history.append({
                    'game_id': self.game_id,
                    'move_idx': self.move_count,
                    'player_idx': player_idx,
                    'player_type': self.current_player.__class__.__name__,
                    'row': row,
                    'col': col,
                    'result': result
                })

                self.move_count += 1

                # Check for game over
                if self.opponent.board.all_ships_sunk():
                    self.game_over = True
                    self.winner = self.current_player
                    self._handle_game_over()
                    return (row, col), result, True  # Move, result, game_over

                # Switch players
                self.current_player, self.opponent = self.opponent, self.current_player
                return (row, col), result, False  # Move, result, game_over

            return None

        except Exception as e:
            logger.error(f"Error during AI move: {e}")
            return None

    def _handle_game_over(self):
        """Process end-of-game activities"""
        duration = time.time() - self.start_time

        # Log game statistics
        logger.info(f"Game over: {self.winner.name} wins in {self.move_count} moves, {duration:.1f}s")

        # Update batch statistics
        self.batch_stats['games_played'] += 1
        if self.winner == self.player1:
            self.batch_stats['player1_wins'] += 1
        else:
            self.batch_stats['player2_wins'] += 1

        # Update average moves
        previous_total = (self.batch_stats['avg_moves'] *
                         (self.batch_stats['games_played'] - 1))
        self.batch_stats['avg_moves'] = (previous_total + self.move_count) / self.batch_stats['games_played']

        # Update total time
        self.batch_stats['total_time'] += duration

        # Save game data
        self._save_game_data(duration)

        # Trigger learning if available
        self._handle_learning()

        # Notify callback
        if self.callback:
            self.callback("game_over", {
                "winner": self.winner.name,
                "player_idx": 1 if self.winner == self.player1 else 2,
                "move_count": self.move_count,
                "duration": duration,
                "game_id": self.game_id
            })

    def _save_game_data(self, duration):
        """Save game data to files"""
        try:
            # Prepare game metadata
            game_data = {
                'game_id': self.game_id,
                'player1': self.player1.__class__.__name__,
                'player2': self.player2.__class__.__name__,
                'winner': self.winner.__class__.__name__,
                'winner_idx': 1 if self.winner == self.player1 else 2,
                'move_count': self.move_count,
                'duration': duration,
                'timestamp': datetime.now().isoformat()
            }

            # Ensure we have moves to save
            if not self.moves_history:
                logger.warning("No moves to save")
                return

            # Save game metadata as JSON
            with open(DATA_DIR / f"game_{self.game_id}.json", 'w') as f:
                json.dump(game_data, f, indent=2)

            # Save moves as CSV
            moves_file = DATA_DIR / f"game_{self.game_id}_moves.csv"
            with open(moves_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.moves_history[0].keys())
                writer.writeheader()
                writer.writerows(self.moves_history)

            # Update master moves file
            master_file = DATA_DIR / "all_moves.csv"
            file_exists = master_file.exists()

            with open(master_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.moves_history[0].keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerows(self.moves_history)

            logger.info(f"Game data saved: {self.game_id}")

        except Exception as e:
            logger.error(f"Error saving game data: {e}")

    def _handle_learning(self):
        """Handle learning from game results for AI agents"""
        # Trigger learning for winner if supported
        if hasattr(self.winner, 'learn_from_game'):
            try:
                self.winner.learn_from_game("win")
            except Exception as e:
                logger.error(f"Error in winner learn_from_game: {e}")

        # Trigger learning for loser if supported
        loser = self.opponent
        if hasattr(loser, 'learn_from_game'):
            try:
                loser.learn_from_game("loss")
            except Exception as e:
                logger.error(f"Error in loser learn_from_game: {e}")

    def start_batch_simulation(self, player1_type, player2_type, num_games, progress_callback=None):
        """Start a batch simulation in a separate thread"""
        if self.simulation_thread and self.simulation_thread.is_alive():
            return False  # Already running

        # Reset batch statistics
        self.batch_stats = {
            'games_played': 0,
            'player1_wins': 0,
            'player2_wins': 0,
            'avg_moves': 0,
            'total_time': 0
        }

        # Start simulation thread
        self.paused = False
        self.simulation_thread = threading.Thread(
            target=self._run_batch_simulation,
            args=(player1_type, player2_type, num_games, progress_callback),
            daemon=True
        )
        self.simulation_thread.start()
        return True

    def _run_batch_simulation(self, player1_type, player2_type, num_games, progress_callback):
        """Run multiple games for batch simulation"""
        for game_idx in range(num_games):
            if self.paused:
                logger.info("Batch simulation paused")
                break

            # Setup game
            success = self.setup_game(
                player1_type,
                player2_type,
                f"{player1_type}-{game_idx}",
                f"{player2_type}-{game_idx}"
            )

            if not success:
                logger.error(f"Failed to set up game {game_idx}")
                continue

            # Play until game over
            while not self.game_over and not self.paused:
                self.ai_move()

                # Small delay to prevent CPU overload
                time.sleep(0.001)

            # Update progress
            if progress_callback and not self.paused:
                try:
                    progress_callback(game_idx + 1, num_games, self.batch_stats.copy())
                except Exception as e:
                    logger.error(f"Error in progress callback: {e}")

        logger.info(f"Batch simulation completed: {self.batch_stats['games_played']}/{num_games} games")

        # Final callback
        if progress_callback and not self.paused:
            try:
                progress_callback(
                    self.batch_stats['games_played'],
                    num_games,
                    self.batch_stats.copy(),
                    finished=True
                )
            except Exception as e:
                logger.error(f"Error in final progress callback: {e}")

    def pause_simulation(self):
        """Pause the current simulation"""
        self.paused = True
        return True

    def resume_simulation(self):
        """Resume a paused simulation"""
        self.paused = False
        return True

    def stop_simulation(self):
        """Stop the current simulation"""
        self.paused = True
        self.simulation_thread = None
        return True


class AnalyticsManager:
    """Manager for data analytics"""

    def __init__(self):
        """Initialize analytics manager"""
        self.game_data_cache = {}
        self.summary_cache = None

    def load_all_game_data(self, refresh=False):
        """Load data from all recorded games"""
        if not refresh and self.game_data_cache:
            return self.game_data_cache

        try:
            # Check if we have a master file
            master_file = DATA_DIR / "all_moves.csv"
            if master_file.exists():
                all_moves = []
                with open(master_file, 'r', newline='') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        all_moves.append(row)

                # Load game metadata
                games = {}
                for game_file in DATA_DIR.glob("game_*.json"):
                    try:
                        with open(game_file, 'r') as f:
                            game_data = json.load(f)
                            games[game_data['game_id']] = game_data
                    except Exception as e:
                        logger.error(f"Error loading game file {game_file}: {e}")

                logger.info(f"Loaded {len(all_moves)} moves from {len(games)} games")

                # Cache the data
                self.game_data_cache = {'moves': all_moves, 'games': games}
                return self.game_data_cache
            else:
                logger.warning("No game data file found")
                return None
        except Exception as e:
            logger.error(f"Error loading game data: {e}")
            return None

    def get_performance_summary(self, refresh=False):
        """Get a summary of agent performance"""
        # Check cache first
        if self.summary_cache is not None and not refresh:
            return self.summary_cache

        # Load data
        data = self.load_all_game_data(refresh)
        if not data:
            return None

        moves_data = data['moves']
        games_data = data['games']

        # Count games and wins by agent
        player_stats = {}
        for game_id, game in games_data.items():
            player1 = game['player1']
            player2 = game['player2']
            winner = game['winner']

            # Initialize player stats
            for player in [player1, player2]:
                if player not in player_stats:
                    player_stats[player] = {
                        'games': 0,
                        'wins': 0
                    }

            # Update game counts
            player_stats[player1]['games'] += 1
            player_stats[player2]['games'] += 1

            # Update win count
            player_stats[winner]['wins'] += 1

        # Calculate win rates
        for player, stats in player_stats.items():
            if stats['games'] > 0:
                stats['win_rate'] = stats['wins'] / stats['games']
            else:
                stats['win_rate'] = 0

        # Calculate average moves by winner
        avg_moves = {}
        for game_id, game in games_data.items():
            winner = game['winner']
            move_count = game['move_count']

            if winner not in avg_moves:
                avg_moves[winner] = []

            avg_moves[winner].append(move_count)

        # Convert to averages
        for winner, moves_list in avg_moves.items():
            avg_moves[winner] = sum(moves_list) / len(moves_list)

        # Create summary
        summary = {
            'total_games': len(games_data),
            'player_stats': player_stats,
            'avg_moves': avg_moves
        }

        # Cache summary
        self.summary_cache = summary
        return summary

    def create_win_rate_chart(self, frame):
        """Create a win rate chart"""
        # Get performance summary
        summary = self.get_performance_summary()
        if not summary:
            ttk.Label(frame, text="No game data available").pack(pady=20)
            return None

        # Extract win rates
        player_stats = summary['player_stats']
        players = []
        win_rates = []

        for player, stats in player_stats.items():
            if stats['games'] >= 5:  # Only include players with enough games
                players.append(player)
                win_rates.append(stats['win_rate'])

        if not players:
            ttk.Label(frame, text="Not enough data for win rate chart").pack(pady=20)
            return None

        # Create figure
        fig, ax = plt.subplots(figsize=(8, 5))

        # Create bar chart
        bars = ax.bar(players, win_rates, color='#3498DB')

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom')

        # Customize chart
        ax.set_ylim(0, 1.0)
        ax.set_ylabel('Win Rate')
        ax.set_title('Win Rate by Agent Type')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        # Create canvas
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()

        return canvas

    def create_heatmap(self, frame, agent_type=None):
        """Create attack frequency heatmap"""
        # Load game data
        data = self.load_all_game_data()
        if not data:
            ttk.Label(frame, text="No game data available").pack(pady=20)
            return None

        # Filter by agent type if specified
        moves_data = data['moves']
        if agent_type and agent_type != "All":
            filtered_moves = [move for move in moves_data if move['player_type'] == agent_type]
        else:
            filtered_moves = moves_data

        if not filtered_moves:
            ttk.Label(frame, text=f"No move data available for {agent_type}").pack(pady=20)
            return None

        # Create frequency grid
        grid = np.zeros((BOARD_SIZE, BOARD_SIZE))
        for move in filtered_moves:
            r, c = int(move['row']), int(move['col'])
            grid[r, c] += 1

        # Normalize
        grid_max = grid.max()
        if grid_max > 0:
            grid = grid / grid_max

        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))

        # Create heatmap
        im = ax.imshow(grid, cmap='viridis')

        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Attack Frequency', rotation=-90, va="bottom")

        # Add labels
        ax.set_xticks(np.arange(BOARD_SIZE))
        ax.set_yticks(np.arange(BOARD_SIZE))
        ax.set_xticklabels(range(BOARD_SIZE))
        ax.set_yticklabels(range(BOARD_SIZE))
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')

        title = 'Attack Frequency Heatmap'
        if agent_type and agent_type != "All":
            title += f' - {agent_type}'
        ax.set_title(title)

        # Add grid lines
        ax.set_xticks(np.arange(-.5, BOARD_SIZE, 1), minor=True)
        ax.set_yticks(np.arange(-.5, BOARD_SIZE, 1), minor=True)
        ax.grid(which='minor', color='w', linestyle='-', linewidth=1)

        plt.tight_layout()

        # Create canvas
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()

        return canvas

    def clear_cache(self):
        """Clear analytics cache and force reload"""
        self.game_data_cache = {}
        self.summary_cache = None
        # Force reload on next request
        self.load_all_game_data(refresh=True)
        logger.info("Analytics cache cleared and data reloaded")


class BattleshipDashboard:
    """Main dashboard application"""

    def __init__(self, root):
        """Initialize the dashboard"""
        self.root = root
        self.root.title("⚓ Battleship AI Research Dashboard ⚓")
        self.root.geometry("1280x800")
        self.root.minsize(1000, 700)

        # Initialize components
        self.game_engine = GameEngine(callback=self.handle_game_event)
        self.analytics = AnalyticsManager()

        # UI state variables
        self.player1_type = tk.StringVar(value="AI_Agent2" if not HAS_AI_AGENT3 else "AI_Agent3")
        self.player2_type = tk.StringVar(value="AI_Agent1")
        self.player1_name = tk.StringVar(value="AI-2" if not HAS_AI_AGENT3 else "AI-3")
        self.player2_name = tk.StringVar(value="AI-1")
        self.animation_speed = tk.IntVar(value=500)
        self.show_ships = tk.BooleanVar(value=True)
        self.show_probabilities = tk.BooleanVar(value=False)
        self.batch_size = tk.IntVar(value=100)
        self.status_message = tk.StringVar(value="Welcome to Battleship AI Dashboard")
        self.opponent_ai_type = tk.StringVar(value="AI_Agent2" if not HAS_AI_AGENT3 else "AI_Agent3")

        # Define UI elements that are referenced elsewhere (fixes outside __init__ warnings)
        self.notebook = None
        self.game_tab = None
        self.batch_tab = None
        self.analytics_tab = None
        self.play_tab = None
        self.board1_frame = None
        self.board2_frame = None
        self.board1_canvas = None
        self.board2_canvas = None
        self.game_status_text = None
        self.batch_status_text = None
        self.play_status_text = None
        self.history_view = None
        self.progress_var = None
        self.progress_bar = None
        self.results_tree = None
        self.batch_viz_frame = None
        self.human_canvas = None
        self.ai_canvas = None
        self.human_cells = {}
        self.ai_cells = {}
        self.analytics_agent_var = None
        self.analytics_notebook = None

        # Build UI
        self.build_ui()

        logger.info("Dashboard initialized")

    def build_ui(self):
        """Build the main UI"""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create tabs
        self.game_tab = ttk.Frame(self.notebook)
        self.batch_tab = ttk.Frame(self.notebook)
        self.analytics_tab = ttk.Frame(self.notebook)
        self.play_tab = ttk.Frame(self.notebook)

        self.notebook.add(self.game_tab, text="Game")
        self.notebook.add(self.batch_tab, text="Batch Simulation")
        self.notebook.add(self.analytics_tab, text="Analytics")
        self.notebook.add(self.play_tab, text="Play vs AI")

        # Build each tab
        self.build_game_tab()
        self.build_batch_tab()
        self.build_analytics_tab()
        self.build_play_tab()

        # Status bar
        status_bar = ttk.Frame(self.root)
        status_bar.pack(fill=tk.X, padx=10, pady=(0, 10))

        ttk.Label(status_bar, textvariable=self.status_message).pack(side=tk.LEFT)
        ttk.Label(status_bar, text="v1.0.0").pack(side=tk.RIGHT)

    def build_game_tab(self):
        """Build the game play tab"""
        # Main frame
        game_frame = ttk.Frame(self.game_tab)
        game_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left control panel
        control_frame = ttk.Frame(game_frame, width=200)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        # Player selection
        setup_frame = ttk.LabelFrame(control_frame, text="Game Setup")
        setup_frame.pack(fill=tk.X, pady=(0, 10))

        # Player 1
        ttk.Label(setup_frame, text="Player 1:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Combobox(setup_frame, textvariable=self.player1_type, values=list(AGENT_CLASSES.keys()), width=10).grid(row=0, column=1, padx=5, pady=5)
        ttk.Entry(setup_frame, textvariable=self.player1_name, width=10).grid(row=0, column=2, padx=5, pady=5)

        # Player 2
        ttk.Label(setup_frame, text="Player 2:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Combobox(setup_frame, textvariable=self.player2_type, values=list(AGENT_CLASSES.keys()), width=10).grid(row=1, column=1, padx=5, pady=5)
        ttk.Entry(setup_frame, textvariable=self.player2_name, width=10).grid(row=1, column=2, padx=5, pady=5)

        # Game controls
        controls_frame = ttk.LabelFrame(control_frame, text="Game Controls")
        controls_frame.pack(fill=tk.X, pady=10)

        ttk.Button(controls_frame, text="New Game", command=self.start_new_game).pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(controls_frame, text="Step", command=self.step_game).pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(controls_frame, text="Auto Play", command=self.auto_play).pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(controls_frame, text="Stop", command=self.stop_auto_play).pack(fill=tk.X, padx=5, pady=5)

        # Visualization controls
        viz_frame = ttk.LabelFrame(control_frame, text="Visualization")
        viz_frame.pack(fill=tk.X, pady=10)

        ttk.Checkbutton(viz_frame, text="Show Ships", variable=self.show_ships,
                       command=self.toggle_ships).pack(anchor=tk.W, padx=5, pady=5)

        ttk.Checkbutton(viz_frame, text="Show Probabilities", variable=self.show_probabilities,
                       command=self.toggle_probabilities).pack(anchor=tk.W, padx=5, pady=5)

        # Animation speed
        ttk.Label(viz_frame, text="Animation Speed:").pack(anchor=tk.W, padx=5, pady=(5, 0))
        ttk.Scale(viz_frame, from_=50, to=2000, variable=self.animation_speed,
                 orient=tk.HORIZONTAL).pack(fill=tk.X, padx=5, pady=5)

        # Game status
        status_frame = ttk.LabelFrame(control_frame, text="Game Status")
        status_frame.pack(fill=tk.X, pady=10)

        self.game_status_text = tk.Text(status_frame, height=6, width=25, wrap=tk.WORD)
        self.game_status_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.game_status_text.insert(tk.END, "Game not started.\nPress 'New Game' to begin.")
        self.game_status_text.config(state=tk.DISABLED)

        # Game boards container
        boards_frame = ttk.Frame(game_frame)
        boards_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Board 1 (Player 1)
        self.board1_frame = ttk.LabelFrame(boards_frame, text="Player 1 Board")
        self.board1_frame.grid(row=0, column=0, padx=10, pady=10)

        self.board1_canvas = BoardCanvas(self.board1_frame)
        self.board1_canvas.pack(padx=10, pady=10)

        # Board 2 (Player 2)
        self.board2_frame = ttk.LabelFrame(boards_frame, text="Player 2 Board")
        self.board2_frame.grid(row=0, column=1, padx=10, pady=10)

        self.board2_canvas = BoardCanvas(self.board2_frame)
        self.board2_canvas.pack(padx=10, pady=10)

        # Move history
        history_frame = ttk.LabelFrame(boards_frame, text="Move History")
        history_frame.grid(row=1, column=0, columnspan=2, sticky=tk.EW, padx=10, pady=10)

        # Treeview for move history
        self.history_view = ttk.Treeview(
            history_frame,
            columns=("turn", "player", "move", "result"),
            show="headings",
            height=6
        )

        self.history_view.heading("turn", text="Turn")
        self.history_view.heading("player", text="Player")
        self.history_view.heading("move", text="Move")
        self.history_view.heading("result", text="Result")

        self.history_view.column("turn", width=50)
        self.history_view.column("player", width=100)
        self.history_view.column("move", width=100)
        self.history_view.column("result", width=100)

        self.history_view.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def build_batch_tab(self):
        """Build the batch simulation tab"""
        # Main frame
        batch_frame = ttk.Frame(self.batch_tab)
        batch_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Top controls
        controls_frame = ttk.Frame(batch_frame)
        controls_frame.pack(fill=tk.X, pady=(0, 10))

        # Batch settings
        settings_frame = ttk.LabelFrame(controls_frame, text="Batch Settings")
        settings_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        # Player selection
        ttk.Label(settings_frame, text="Player 1:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Combobox(settings_frame, textvariable=self.player1_type,
                    values=[k for k in AGENT_CLASSES.keys() if k != "Human"],
                    width=10).grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(settings_frame, text="Player 2:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Combobox(settings_frame, textvariable=self.player2_type,
                    values=[k for k in AGENT_CLASSES.keys() if k != "Human"],
                    width=10).grid(row=1, column=1, padx=5, pady=5)

        # Batch size
        ttk.Label(settings_frame, text="Games:").grid(row=0, column=2, sticky=tk.W, padx=(15, 5), pady=5)
        ttk.Spinbox(settings_frame, from_=1, to=10000, textvariable=self.batch_size, width=6).grid(
            row=0, column=3, sticky=tk.W, padx=5, pady=5)

        # Batch controls
        batch_control_frame = ttk.LabelFrame(controls_frame, text="Controls")
        batch_control_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=5)

        ttk.Button(batch_control_frame, text="Run Batch", command=self.start_batch_simulation).pack(
            fill=tk.X, padx=5, pady=5)
        ttk.Button(batch_control_frame, text="Pause", command=self.pause_batch_simulation).pack(
            fill=tk.X, padx=5, pady=5)
        ttk.Button(batch_control_frame, text="Resume", command=self.resume_batch_simulation).pack(
            fill=tk.X, padx=5, pady=5)
        ttk.Button(batch_control_frame, text="Stop", command=self.stop_batch_simulation).pack(
            fill=tk.X, padx=5, pady=5)

        # Batch status
        status_frame = ttk.LabelFrame(controls_frame, text="Status")
        status_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        self.batch_status_text = tk.Text(status_frame, height=5, width=30, wrap=tk.WORD)
        self.batch_status_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.batch_status_text.insert(tk.END, "Ready to run batch simulation.")
        self.batch_status_text.config(state=tk.DISABLED)

        # Progress bar
        progress_frame = ttk.Frame(batch_frame)
        progress_frame.pack(fill=tk.X, pady=(0, 10))

        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(
            progress_frame,
            orient=tk.HORIZONTAL,
            length=100,
            mode='determinate',
            variable=self.progress_var
        )
        self.progress_bar.pack(fill=tk.X, padx=5, pady=5)

        # Results section
        results_frame = ttk.LabelFrame(batch_frame, text="Batch Results")
        results_frame.pack(fill=tk.BOTH, expand=True)

        # Results table
        self.results_tree = ttk.Treeview(
            results_frame,
            columns=("agent", "games", "wins", "win_rate", "avg_moves"),
            show="headings",
            height=2
        )

        self.results_tree.heading("agent", text="Agent")
        self.results_tree.heading("games", text="Games")
        self.results_tree.heading("wins", text="Wins")
        self.results_tree.heading("win_rate", text="Win Rate")
        self.results_tree.heading("avg_moves", text="Avg Moves")

        self.results_tree.column("agent", width=100)
        self.results_tree.column("games", width=60, anchor=tk.CENTER)
        self.results_tree.column("wins", width=60, anchor=tk.CENTER)
        self.results_tree.column("win_rate", width=80, anchor=tk.CENTER)
        self.results_tree.column("avg_moves", width=80, anchor=tk.CENTER)

        self.results_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Initialize rows
        self.results_tree.insert('', 'end', iid='p1', values=(self.player1_type.get(), 0, 0, "0%", 0))
        self.results_tree.insert('', 'end', iid='p2', values=(self.player2_type.get(), 0, 0, "0%", 0))

        # Visualization
        viz_frame = ttk.LabelFrame(batch_frame, text="Visualization")
        viz_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        # Create container for visualization
        self.batch_viz_frame = ttk.Frame(viz_frame)
        self.batch_viz_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Add "Waiting for data" label
        ttk.Label(self.batch_viz_frame, text="Run a batch simulation to see results").pack(pady=20)

    def build_analytics_tab(self):
        """Build the analytics tab"""
        # Main frame
        analytics_frame = ttk.Frame(self.analytics_tab)
        analytics_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Top controls
        controls_frame = ttk.Frame(analytics_frame)
        controls_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(controls_frame, text="Agent:").pack(side=tk.LEFT, padx=(0, 5))

        self.analytics_agent_var = tk.StringVar(value="All")
        agent_combo = ttk.Combobox(controls_frame, textvariable=self.analytics_agent_var,
                                  values=["All"] + list(AGENT_CLASSES.keys()),
                                  width=10)
        agent_combo.pack(side=tk.LEFT, padx=(0, 10))

        ttk.Button(controls_frame, text="Update", command=self.update_analytics).pack(side=tk.LEFT)
        ttk.Button(controls_frame, text="Clear Cache", command=self.clear_analytics_cache).pack(side=tk.LEFT, padx=5)

        # Analytics content
        content_frame = ttk.Frame(analytics_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)

        # Create notebook for different analytics views
        self.analytics_notebook = ttk.Notebook(content_frame)
        self.analytics_notebook.pack(fill=tk.BOTH, expand=True)

        # Summary tab
        self.summary_tab = ttk.Frame(self.analytics_notebook)
        self.analytics_notebook.add(self.summary_tab, text="Summary")

        # Win rates tab
        self.win_rates_tab = ttk.Frame(self.analytics_notebook)
        self.analytics_notebook.add(self.win_rates_tab, text="Win Rates")

        # Heatmaps tab
        self.heatmaps_tab = ttk.Frame(self.analytics_notebook)
        self.analytics_notebook.add(self.heatmaps_tab, text="Heatmaps")

        # Initialize analytics views
        self.update_analytics()

    def build_play_tab(self):
        """Build the human vs AI play tab"""
        # Main frame
        play_frame = ttk.Frame(self.play_tab)
        play_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Controls frame
        controls_frame = ttk.Frame(play_frame, width=200)
        controls_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        # AI selection
        setup_frame = ttk.LabelFrame(controls_frame, text="Game Setup")
        setup_frame.pack(fill=tk.X, pady=(0, 10))

        # AI opponent selection
        ttk.Label(setup_frame, text="AI Opponent:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Combobox(
            setup_frame,
            textvariable=self.opponent_ai_type,
            values=[k for k in AGENT_CLASSES.keys() if k != "Human"],
            width=12,
            state="readonly"
        ).grid(row=0, column=1, padx=5, pady=5)

        # Game controls
        action_frame = ttk.LabelFrame(controls_frame, text="Game Controls")
        action_frame.pack(fill=tk.X, pady=10)

        ttk.Button(action_frame, text="New Game", command=self.start_human_game).pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(action_frame, text="Restart", command=self.restart_human_game).pack(fill=tk.X, padx=5, pady=5)

        # Status display
        status_frame = ttk.LabelFrame(controls_frame, text="Game Status")
        status_frame.pack(fill=tk.X, pady=10, expand=True)

        self.play_status_text = tk.Text(status_frame, height=10, width=25, wrap=tk.WORD)
        self.play_status_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.play_status_text.insert(tk.END, "Click 'New Game' to begin playing against AI.")
        self.play_status_text.config(state=tk.DISABLED)

        # Board area
        board_frame = ttk.Frame(play_frame)
        board_frame.pack(side=tk.LEFT, pady=10, padx=10, fill=tk.BOTH, expand=True)

        # Human board
        self.human_board_frame = ttk.LabelFrame(play_frame, text="Your Fleet")
        self.human_board_frame.pack(side=tk.RIGHT, pady=10, padx=10, fill=tk.BOTH)

        self.human_canvas = tk.Canvas(
            self.human_board_frame,
            width=10*CELL_SIZE,
            height=10*CELL_SIZE,
            bg='#1E2A38',
            highlightthickness=0
        )
        self.human_canvas.pack()

        # AI board (for targeting)
        self.ai_board_frame = ttk.LabelFrame(play_frame, text="Enemy Waters")
        self.ai_board_frame.pack(side=tk.LEFT, pady=10, padx=10, fill=tk.BOTH)

        self.ai_canvas = tk.Canvas(
            self.ai_board_frame,
            width=10*CELL_SIZE,
            height=10*CELL_SIZE,
            bg='#1E2A38',
            highlightthickness=0
        )
        self.ai_canvas.pack()

        # Bind click event to enemy board
        self.ai_canvas.bind("<Button-1>", self.on_ai_board_click)

    def start_new_game(self):
        """Start a new game"""
        # Get player types and names
        player1_type = self.player1_type.get()
        player2_type = self.player2_type.get()
        player1_name = self.player1_name.get()
        player2_name = self.player2_name.get()

        # Set up game
        success = self.game_engine.setup_game(
            player1_type,
            player2_type,
            player1_name,
            player2_name
        )

        if success:
            # Clear board displays
            self.board1_canvas.clear()
            self.board2_canvas.clear()

            # Update board titles
            self.board1_frame['text'] = f"{player1_name}'s Board"
            self.board2_frame['text'] = f"{player2_name}'s Board"

            # Bind boards to players
            self.board1_canvas.bind_to_board(self.game_engine.player1.board, self.game_engine.player1)
            self.board2_canvas.bind_to_board(self.game_engine.player2.board, self.game_engine.player2)

            # Apply visibility settings
            self.apply_visibility_settings()

            # Clear move history
            for item in self.history_view.get_children():
                self.history_view.delete(item)

            # Update game status
            self.update_game_status(f"Game started: {player1_name} vs {player2_name}")
            self.status_message.set(f"New game: {player1_name} vs {player2_name}")

            # Update UI state
            self.update_ui_state()

            # If first player is AI, auto-step
            if self.game_engine.current_player.__class__.__name__ != "HumanPlayer":
                self.root.after(500, self.step_game)

        else:
            messagebox.showerror("Error", "Failed to start new game")

    def step_game(self):
        """Execute a single AI step"""
        if self.game_engine.game_over:
            self.status_message.set("Game is over. Start a new game.")
            return

        current_player = self.game_engine.current_player

        # Skip if human player
        if current_player.__class__.__name__ == "HumanPlayer":
            self.status_message.set(f"It's {current_player.name}'s turn (Human). Click the opponent's board.")
            return

        # Execute AI move
        move_result = self.game_engine.ai_move()

        if move_result:
            (row, col), result, game_over = move_result

            # Update the board display
            is_p1_turn = current_player == self.game_engine.player1
            board_canvas = self.board2_canvas if is_p1_turn else self.board1_canvas
            board_canvas.refresh()
            board_canvas.highlight_cell(row, col, color=HIT_COLOR if result in ['hit', 'sunk'] else MISS_COLOR)

            # Add to move history
            turn_number = (self.game_engine.move_count + 1) // 2
            self.history_view.insert(
                '', 0,
                values=(turn_number, current_player.name, f"({row}, {col})", result)
            )

            # Update status
            if game_over:
                self.update_game_status(
                    f"Game over! {current_player.name} wins!\n" +
                    f"Moves: {self.game_engine.move_count}\n" +
                    "Press 'New Game' to play again."
                )
                self.status_message.set(f"Game over! {current_player.name} wins!")

                # Show both boards
                self.board1_canvas.show_ships = True
                self.board2_canvas.show_ships = True
                self.board1_canvas.refresh()
                self.board2_canvas.refresh()

                # Update UI state
                self.update_ui_state()

                # Show game summary dialog
                duration = time.time() - self.game_engine.start_time
                messagebox.showinfo(
                    "Game Over",
                    f"{current_player.name} wins!\n\n" +
                    f"Moves: {self.game_engine.move_count}\n" +
                    f"Duration: {duration:.1f} seconds"
                )

            else:
                # Update for next player
                self.update_game_status(
                    f"Turn {turn_number}: {current_player.name}\n" +
                    f"Move: ({row}, {col})\n" +
                    f"Result: {result}"
                )

                # If next player is AI, schedule next move
                next_player = self.game_engine.current_player
                if next_player.__class__.__name__ != "HumanPlayer":
                    self.status_message.set(f"{next_player.name}'s turn...")
                else:
                    self.status_message.set(f"{next_player.name}'s turn. Click the opponent's board.")

    def auto_play(self):
        """Auto-play the game until completion"""
        if self.game_engine.game_over:
            return

        current_player = self.game_engine.current_player

        # Skip if human player
        if current_player.__class__.__name__ == "HumanPlayer":
            self.status_message.set(f"Can't auto-play for human player ({current_player.name})")
            return

        # Execute one step
        self.step_game()

        # If not over, schedule next step
        if not self.game_engine.game_over:
            next_player = self.game_engine.current_player

            if next_player.__class__.__name__ != "HumanPlayer":
                self.root.after(self.animation_speed.get(), self.auto_play)

    def stop_auto_play(self):
        """Stop auto-play"""
        # Set game_over to prevent further moves
        self.game_engine.game_over = True
        self.update_game_status("Game stopped by user.\nPress 'New Game' to start again.")
        self.status_message.set("Game stopped")

        # Update UI state
        self.update_ui_state()

    def start_batch_simulation(self):
        """Start a batch simulation"""
        # Get settings
        player1_type = self.player1_type.get()
        player2_type = self.player2_type.get()
        num_games = self.batch_size.get()

        # Validate
        if player1_type == "Human" or player2_type == "Human":
            messagebox.showinfo("Invalid Selection", "Can't use Human players in batch simulation")
            return

        if num_games < 1:
            messagebox.showinfo("Invalid Input", "Number of games must be at least 1")
            return

        # Start the simulation
        success = self.game_engine.start_batch_simulation(
            player1_type,
            player2_type,
            num_games,
            self.update_batch_progress
        )

        if success:
            # Update status
            self.update_batch_status(f"Running batch simulation:\n{player1_type} vs {player2_type}\n{num_games} games")
            self.status_message.set(f"Batch simulation started: {num_games} games")

            # Reset progress
            self.progress_var.set(0)

            # Update results tree
            self.results_tree.item('p1', values=(player1_type, 0, 0, "0%", 0))
            self.results_tree.item('p2', values=(player2_type, 0, 0, "0%", 0))
        else:
            self.update_batch_status("Failed to start batch simulation.\nAnother batch may be running.")
            self.status_message.set("Failed to start batch simulation")

    def pause_batch_simulation(self):
        """Pause the current batch simulation"""
        if self.game_engine.pause_simulation():
            self.update_batch_status("Batch simulation paused.\nPress 'Resume' to continue.")
            self.status_message.set("Batch simulation paused")

    def resume_batch_simulation(self):
        """Resume a paused batch simulation"""
        if self.game_engine.resume_simulation():
            self.update_batch_status("Batch simulation resumed.")
            self.status_message.set("Batch simulation resumed")

    def stop_batch_simulation(self):
        """Stop the current batch simulation"""
        if self.game_engine.stop_simulation():
            self.update_batch_status("Batch simulation stopped.\nPress 'Run Batch' to start a new batch.")
            self.status_message.set("Batch simulation stopped")

    def update_batch_progress(self, current, total, stats, finished=False):
        """
        Thread-safe version of batch progress update that schedules UI updates
        in the main thread.

        Args:
            current: Current game number
            total: Total games to run
            stats: Current statistics dictionary
            finished: Whether the batch is complete
        """
        # Schedule the actual UI update in the main thread
        self.root.after(0, lambda: self._update_batch_progress_main_thread(current, total, stats, finished))

    def _update_batch_progress_main_thread(self, current, total, stats, finished=False):
        """
        Updates batch progress indicators in the main thread.

        Args:
            current: Current game number
            total: Total games to run
            stats: Current statistics dictionary
            finished: Whether the batch is complete
        """
        try:
            # Update progress bar
            progress = (current / total) * 100
            self.progress_var.set(progress)

            # Update status
            if not finished:
                self.update_batch_status(
                    f"Running batch simulation:\n" +
                    f"Progress: {current}/{total} games\n" +
                    f"Player 1 wins: {stats['player1_wins']}\n" +
                    f"Player 2 wins: {stats['player2_wins']}"
                )
            else:
                self.update_batch_status(
                    f"Batch simulation complete.\n" +
                    f"Games played: {stats['games_played']}/{total}\n" +
                    f"Total time: {stats['total_time']:.1f}s"
                )
                self.status_message.set(f"Batch simulation finished: {stats['games_played']} games")

            # Update results tree
            if stats['games_played'] > 0:  # Avoid division by zero
                p1_win_rate = stats['player1_wins'] / stats['games_played']
                p2_win_rate = stats['player2_wins'] / stats['games_played']

                self.results_tree.item('p1', values=(
                    self.player1_type.get(),
                    stats['games_played'],
                    stats['player1_wins'],
                    f"{p1_win_rate:.1%}",
                    f"{stats['avg_moves']:.1f}"
                ))

                self.results_tree.item('p2', values=(
                    self.player2_type.get(),
                    stats['games_played'],
                    stats['player2_wins'],
                    f"{p2_win_rate:.1%}",
                    f"{stats['avg_moves']:.1f}"
                ))

            # Create visualization if complete
            if finished and stats['games_played'] > 0:
                # The create_batch_visualization method will handle threading concerns
                self.create_batch_visualization(stats)

        except Exception as e:
            # Log error
            logger.error(f"Error updating batch progress: {e}")
            logger.error(traceback.format_exc())

            # Update status with error
            self.status_message.set(f"Error updating batch progress: {str(e)}")

    def create_batch_visualization(self, stats):
        """
        Thread-safe version of batch visualization creation.
        Schedules the actual visualization creation in the main thread.

        Args:
            stats: Batch statistics dictionary
        """
        # Schedule the actual visualization in the main thread
        self.root.after(10, lambda: self._create_batch_visualization_main_thread(stats))

    def _create_batch_visualization_main_thread(self, stats):
        """
        Creates the batch visualization in the main thread to avoid
        matplotlib warnings and errors.

        Args:
            stats: Batch statistics dictionary
        """
        try:
            # Clear previous visualization
            for widget in self.batch_viz_frame.winfo_children():
                widget.destroy()

            # Create figure with pie chart
            fig, ax = plt.subplots(figsize=(6, 4))

            # Create pie chart of wins
            labels = [self.player1_type.get(), self.player2_type.get()]
            sizes = [int(stats['player1_wins']), int(stats['player2_wins'])]  # Convert to int to fix float warning

            if sum(sizes) > 0:  # Avoid empty pie chart
                explode = (0.1, 0) if sizes[0] > sizes[1] else (0, 0.1)

                ax.pie(
                    sizes,
                    explode=explode,
                    labels=labels,
                    autopct='%1.1f%%',
                    shadow=True,
                    startangle=90
                )
                ax.axis('equal')
                plt.title('Win Distribution')

                # Create canvas
                canvas = FigureCanvasTkAgg(fig, master=self.batch_viz_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            else:
                ttk.Label(self.batch_viz_frame, text="No games completed").pack(pady=20)

        except Exception as e:
            # Log error and show a message in the frame
            logger.error(f"Error creating batch visualization: {e}")
            logger.error(traceback.format_exc())

            ttk.Label(self.batch_viz_frame,
                     text=f"Error creating visualization: {str(e)}").pack(pady=20)

    def update_analytics(self):
        """Update analytics displays"""
        # Get selected agent
        agent = self.analytics_agent_var.get()

        # Update summary tab
        self.update_summary_tab()

        # Update win rates tab
        self.update_win_rates_tab()

        # Update heatmaps tab
        self.update_heatmaps_tab(agent)

    def update_summary_tab(self):
        """Update the summary analytics tab"""
        # Clear previous content
        for widget in self.summary_tab.winfo_children():
            widget.destroy()

        # Get summary data
        summary = self.analytics.get_performance_summary()
        if not summary:
            ttk.Label(self.summary_tab, text="No game data available").pack(pady=20)
            return

        # Create summary frame
        summary_frame = ttk.Frame(self.summary_tab)
        summary_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Key metrics section
        metrics_frame = ttk.LabelFrame(summary_frame, text="Key Metrics")
        metrics_frame.pack(fill=tk.X, pady=(0, 10))

        # Total games
        ttk.Label(metrics_frame, text="Total Games:").grid(row=0, column=0, sticky=tk.W, padx=10, pady=5)
        ttk.Label(metrics_frame, text=str(summary['total_games']), font=('Arial', 10, 'bold')).grid(
            row=0, column=1, sticky=tk.W, padx=10, pady=5)

        # Best agent
        ttk.Label(metrics_frame, text="Best Agent:").grid(row=0, column=2, sticky=tk.W, padx=10, pady=5)

        # Find best agent
        best_agent = None
        best_win_rate = 0

        for agent, stats in summary['player_stats'].items():
            if stats['games'] >= 5 and stats['win_rate'] > best_win_rate:
                best_agent = agent
                best_win_rate = stats['win_rate']

        # Display best agent
        best_agent_text = f"{best_agent} ({best_win_rate:.1%})" if best_agent else "N/A"
        ttk.Label(metrics_frame, text=best_agent_text, font=('Arial', 10, 'bold')).grid(
            row=0, column=3, sticky=tk.W, padx=10, pady=5)

        # Agent performance table
        table_frame = ttk.LabelFrame(summary_frame, text="Agent Performance")
        table_frame.pack(fill=tk.BOTH, expand=True)

        # Create treeview
        performance_tree = ttk.Treeview(
            table_frame,
            columns=("agent", "games", "wins", "win_rate", "avg_moves"),
            show="headings"
        )

        performance_tree.heading("agent", text="Agent")
        performance_tree.heading("games", text="Games")
        performance_tree.heading("wins", text="Wins")
        performance_tree.heading("win_rate", text="Win Rate")
        performance_tree.heading("avg_moves", text="Avg Moves")

        performance_tree.column("agent", width=100)
        performance_tree.column("games", width=60, anchor=tk.CENTER)
        performance_tree.column("wins", width=60, anchor=tk.CENTER)
        performance_tree.column("win_rate", width=80, anchor=tk.CENTER)
        performance_tree.column("avg_moves", width=80, anchor=tk.CENTER)

        performance_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Add data rows
        for agent, stats in summary['player_stats'].items():
            avg_moves = summary['avg_moves'].get(agent, 0)
            performance_tree.insert('', 'end', values=(
                agent,
                stats['games'],
                stats['wins'],
                f"{stats['win_rate']:.1%}",
                f"{avg_moves:.1f}" if avg_moves else "N/A"
            ))

    def update_win_rates_tab(self):
        """Update the win rates tab"""
        # Clear previous content
        for widget in self.win_rates_tab.winfo_children():
            widget.destroy()

        # Create chart
        canvas = self.analytics.create_win_rate_chart(self.win_rates_tab)

        if canvas:
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def update_heatmaps_tab(self, agent=None):
        """Update the heatmaps tab"""
        # Clear previous content
        for widget in self.heatmaps_tab.winfo_children():
            widget.destroy()

        # Controls
        controls_frame = ttk.Frame(self.heatmaps_tab)
        controls_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(controls_frame, text="Agent:").pack(side=tk.LEFT, padx=(0, 5))

        agent_combo = ttk.Combobox(controls_frame, textvariable=self.analytics_agent_var,
                                  values=["All"] + list(AGENT_CLASSES.keys()),
                                  width=10)
        agent_combo.pack(side=tk.LEFT, padx=(0, 10))

        ttk.Button(controls_frame, text="Generate",
                  command=lambda: self.update_heatmaps_tab(self.analytics_agent_var.get())
                 ).pack(side=tk.LEFT)

        # Create heatmap
        canvas = self.analytics.create_heatmap(self.heatmaps_tab, agent)

        if canvas:
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def clear_analytics_cache(self):
        """Clear the analytics cache and reload data"""
        # Reset all cached data
        self.analytics.game_data_cache = {}
        self.analytics.summary_cache = None

        # Force reload data on next request
        self.analytics.load_all_game_data(refresh=True)

        # Update all tabs to reflect cleared cache
        self.update_summary_tab()
        self.update_win_rates_tab()
        self.update_heatmaps_tab(self.analytics_agent_var.get())

        self.status_message.set("Analytics cache cleared and data reloaded")

    def start_human_game(self):
        """Start a new human vs AI game"""
        # Create players
        self.human_player = HumanPlayer("You", manual_setup='n')

        ai_type = self.opponent_ai_type.get()
        if ai_type in AGENT_CLASSES:
            ai_class = AGENT_CLASSES[ai_type]
            self.ai_opponent = ai_class("AI Opponent")
        else:
            messagebox.showerror("Error", f"Unknown AI type: {ai_type}")
            return

        # Reset game state
        self.game_turn = "human"  # human goes first
        self.game_over = False
        self.move_count = 0

        # Clear boards
        self.draw_human_board(reveal=True)
        self.draw_ai_board()

        # Update status
        self.update_play_status("Game started! Click on the enemy board to fire.")
        self.status_message.set("Human vs AI game started")

    def restart_human_game(self):
        """Restart human vs AI game"""
        if hasattr(self, 'human_player') and hasattr(self, 'ai_opponent'):
            self.start_human_game()
        else:
            self.update_play_status("No game in progress. Click 'New Game' to start.")

    def update_play_status(self, message):
        """Update play status text"""
        self.play_status_text.config(state=tk.NORMAL)
        self.play_status_text.delete("1.0", tk.END)
        self.play_status_text.insert(tk.END, message)
        self.play_status_text.config(state=tk.DISABLED)

    def draw_human_board(self, reveal=True):
        """Draw human player's board"""
        self.human_canvas.delete('all')
        self.human_cells = {}

        if not hasattr(self, 'human_player'):
            return

        size = 10
        for r in range(size):
            for c in range(size):
                x = c * CELL_SIZE
                y = r * CELL_SIZE
                rect = self.human_canvas.create_rectangle(
                    x, y, x + CELL_SIZE, y + CELL_SIZE,
                    fill=WATER_COLOR, outline='black'
                )
                self.human_cells[(r, c)] = rect

                # Draw ships if reveal is True
                if reveal and (r, c) in self.human_player.board.ship_lookup:
                    self.human_canvas.create_rectangle(
                        x + 5, y + 5, x + CELL_SIZE - 5, y + CELL_SIZE - 5,
                        fill=SHIP_COLOR, outline=''
                    )

        # Draw any hits and misses on human board
        for r in range(size):
            for c in range(size):
                val = self.human_player.board.grid[r, c]
                if val == 1:  # Miss
                    self._mark_cell(self.human_canvas, (r, c), "miss")
                elif val == 2:  # Hit
                    self._mark_cell(self.human_canvas, (r, c), "hit")

    def draw_ai_board(self):
        """Draw AI opponent's board"""
        self.ai_canvas.delete('all')
        self.ai_cells = {}

        if not hasattr(self, 'ai_opponent'):
            return

        size = 10
        for r in range(size):
            for c in range(size):
                x = c * CELL_SIZE
                y = r * CELL_SIZE
                rect = self.ai_canvas.create_rectangle(
                    x, y, x + CELL_SIZE, y + CELL_SIZE,
                    fill=WATER_COLOR, outline='black'
                )
                self.ai_cells[(r, c)] = rect

    def on_ai_board_click(self, event):
        """Handle click on AI board to make a move"""
        if not hasattr(self, 'human_player') or not hasattr(self, 'ai_opponent') or self.game_over:
            return

        if self.game_turn != "human":
            self.update_play_status("Wait for your turn!")
            return

        # Convert click to board coordinates
        r = event.y // CELL_SIZE
        c = event.x // CELL_SIZE

        # Make the move
        result = self.ai_opponent.board.attack((r, c))
        self.move_count += 1

        # Update board display
        self._mark_cell(self.ai_canvas, (r, c), result)

        # Update status
        self.update_play_status(f"Your shot at ({r},{c}): {result.upper()}")

        # Check for game over
        if self.ai_opponent.board.all_ships_sunk():
            self.game_over = True
            self.update_play_status(f"Game over! You win in {self.move_count // 2 + 1} turns!")
            return

        # AI turn
        self.game_turn = "ai"
        self.root.after(1000, self.ai_make_move)

    def ai_make_move(self):
        """Let AI make its move"""
        if self.game_over:
            return

        result = self.ai_opponent.take_turn(self.human_player.board)
        move = self.ai_opponent.last_move
        self.move_count += 1

        # Update board
        self.draw_human_board(reveal=True)

        # Update status
        self.update_play_status(f"AI shot at {move}: {result.upper()}\nYour turn!")

        # Check for game over
        if self.human_player.board.all_ships_sunk():
            self.game_over = True
            self.update_play_status(f"Game over! AI wins in {self.move_count // 2} turns!")
            return

        # Back to human turn
        self.game_turn = "human"

    def _mark_cell(self, canvas, coord, result):
        """Mark a cell with hit or miss"""
        r, c = coord
        x, y = c * CELL_SIZE, r * CELL_SIZE

        if result in ["hit", "sunk"]:
            # Draw hit marker (red circle with X)
            canvas.create_oval(
                x + 10, y + 10,
                x + CELL_SIZE - 10, y + CELL_SIZE - 10,
                fill=HIT_COLOR, outline="white", width=2
            )
            canvas.create_line(
                x + 15, y + 15,
                x + CELL_SIZE - 15, y + CELL_SIZE - 15,
                fill="white", width=2
            )
            canvas.create_line(
                x + CELL_SIZE - 15, y + 15,
                x + 15, y + CELL_SIZE - 15,
                fill="white", width=2
            )
        elif result == "miss":
            # Draw miss marker (blue circle)
            canvas.create_oval(
                x + 15, y + 15,
                x + CELL_SIZE - 15, y + CELL_SIZE - 15,
                fill=MISS_COLOR, outline="white", width=1
            )

    def apply_visibility_settings(self):
        """Apply visibility settings to boards"""
        # Apply ship visibility
        if self.show_ships.get():
            self.board1_canvas.show_ships = True
            self.board2_canvas.show_ships = True
        else:
            self.board1_canvas.show_ships = False
            self.board2_canvas.show_ships = False

        # Apply probability visualization
        if self.show_probabilities.get():
            self.board1_canvas.show_probabilities = True
            self.board2_canvas.show_probabilities = True
        else:
            self.board1_canvas.show_probabilities = False
            self.board2_canvas.show_probabilities = False

        # Refresh displays
        self.board1_canvas.refresh()
        self.board2_canvas.refresh()

    def toggle_ships(self):
        """Toggle ship visibility"""
        self.board1_canvas.toggle_ships()
        self.board2_canvas.toggle_ships()
        self.show_ships.set(self.board1_canvas.show_ships)

    def toggle_probabilities(self):
        """Toggle probability display"""
        if not self.game_engine.player1:
            messagebox.showinfo("Not Available", "Start a game first to use probability visualization")
            self.show_probabilities.set(False)
            return

        # Check if neural models are available
        has_neural = (
            (self.game_engine.player1.__class__.__name__ in ["AIPlayer2", "AIAgent3"] or
            self.game_engine.player2.__class__.__name__ in ["AIPlayer2", "AIAgent3"]) and
            hasattr(self.game_engine.player1, 'nn_model') and
            self.game_engine.player1.nn_model is not None
        )

        if not has_neural and not self.show_probabilities.get():
            messagebox.showinfo("Limited Feature",
                             "Full probability visualization requires neural models.\n"
                             "Basic visualization will be shown.")

        self.board1_canvas.toggle_probabilities()
        self.board2_canvas.toggle_probabilities()
        self.show_probabilities.set(self.board1_canvas.show_probabilities)

    def update_game_status(self, message):
        """Update game status text widget"""
        self.game_status_text.config(state=tk.NORMAL)
        self.game_status_text.delete("1.0", tk.END)
        self.game_status_text.insert(tk.END, message)
        self.game_status_text.config(state=tk.DISABLED)

    def update_batch_status(self, message):
        """Update batch status text widget"""
        self.batch_status_text.config(state=tk.NORMAL)
        self.batch_status_text.delete("1.0", tk.END)
        self.batch_status_text.insert(tk.END, message)
        self.batch_status_text.config(state=tk.DISABLED)

    def handle_game_event(self, event, data):
        """Handle events from the game engine"""
        if event == "game_over":
            # Game over event handling
            self.update_ui_state()
        elif event == "move_made":
            # Move made event handling
            pass

    def update_ui_state(self):
        """Update UI based on current state"""
        # Update board titles
        if self.game_engine.player1 and self.game_engine.player2:
            self.board1_frame['text'] = f"{self.game_engine.player1.name}'s Board"
            self.board2_frame['text'] = f"{self.game_engine.player2.name}'s Board"

    def run(self):
        """Run the dashboard application"""
        try:
            self.root.mainloop()
        except Exception as e:
            logger.error(f"Error in mainloop: {e}")
            traceback.print_exc()


# Main function
def main():
    """Main function to run the dashboard application"""
    # Create root window
    root = tk.Tk()

    try:
        # Setup error handling
        def show_error(exception_type, exception_value, exception_traceback):
            """Show error message for uncaught exceptions"""
            error_msg = f"An unexpected error occurred:\n\n{exception_value}"
            logger.error(f"Uncaught exception: {error_msg}")
            logger.error(traceback.format_exc())
            messagebox.showerror("Error", error_msg)

        # Set custom exception handler
        sys.excepthook = show_error

        # Create dashboard
        dashboard = BattleshipDashboard(root)

        # Run dashboard
        dashboard.run()

    except Exception as e:
        logger.error(f"Error initializing dashboard: {e}")
        traceback.print_exc()
        messagebox.showerror("Initialization Error", f"Failed to start dashboard: {e}")
        root.destroy()


if __name__ == "__main__":
    main()