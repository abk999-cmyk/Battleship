import random
import numpy as np
import datetime
import time
import os
import pickle
import logging
from typing import List, Set, Tuple, Dict, Optional, Any, Union
from collections import defaultdict, Counter
import warnings
from player import Player
from board import SHIP_SIZES

# ----------------------------------------------------------------------
# Global constants
# ----------------------------------------------------------------------
BOARD_SIZE: int = 10
TOTAL_SHIP_SQUARES: int = sum(SHIP_SIZES)
MC_SAMPLES_BASE: int = 1500
PARITY_MIN_SHIP_SIZE: int = 2

# Set up directories and paths exactly like AI_agent2 does
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))  # Base directory
MODELS_SUBDIR = "models"  # Same subdirectory name as AI_agent2
if not os.path.exists(os.path.join(MODEL_DIR, MODELS_SUBDIR)):
    os.makedirs(os.path.join(MODEL_DIR, MODELS_SUBDIR), exist_ok=True)

LOG_DIR = os.path.dirname(os.path.abspath(__file__))  # Base directory for logs
LOGS_SUBDIR = "logs"  # Subdirectory for logs
if not os.path.exists(os.path.join(LOG_DIR, LOGS_SUBDIR)):
    os.makedirs(os.path.join(LOG_DIR, LOGS_SUBDIR), exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, LOGS_SUBDIR, 'ai_agent3.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('AI_Agent3')

# Try importing optional dependencies with proper error handling
TF_AVAILABLE = False
SCIPY_AVAILABLE = False
NETWORKX_AVAILABLE = False

# Attempt to import TensorFlow for neural network model
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model

    TF_AVAILABLE = True
    logger.info("TensorFlow imported successfully")
except ImportError as e:
    logger.warning(f"TensorFlow not available. Neural models will be disabled. Error: {str(e)}")

# Attempt to import scipy
try:
    from scipy.stats import entropy

    SCIPY_AVAILABLE = True
except ImportError as e:
    logger.warning(f"SciPy not available. Some statistical functions will be disabled.")

# Attempt to import networkx for graph representation
try:
    import networkx as nx

    NETWORKX_AVAILABLE = True
except ImportError as e:
    logger.warning(f"NetworkX not available. Graph-based ship representation will be disabled.")


# Define simple entropy function to use if scipy is not available
def simple_entropy(p):
    """Simple binary entropy calculation without scipy"""
    if 0 < p < 1:
        return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    return 0.0


# ----------------------------------------------------------------------
# AI Agent Class with all enhancements
# ----------------------------------------------------------------------
class AIAgent3(Player):
    def __init__(self, name: str, continuous_learning: bool = True, opponent_modeling: bool = True):
        """
        Enhanced Battleship AI with advanced features.
        """
        super().__init__(name)
        self.board_size = BOARD_SIZE
        self.board.place_ships()

        # Core state variables
        self.available_moves = {(r, c) for r in range(self.board_size) for c in range(self.board_size)}
        self.result_grid = np.zeros((self.board_size, self.board_size), dtype=int)  # 0: unknown, 1: hit, -1: miss
        self.hits = []  # Current hits on a ship not yet sunk
        self.sunk_ships_info = []  # Store info about ships we've sunk (coords, size)
        self.hit_count = 0  # Total squares confirmed as part of any ship (hit or sunk)
        self.mode = "hunt"  # "hunt" or "target"
        self.remaining_ship_sizes = sorted(list(SHIP_SIZES), reverse=True)
        self._update_min_max_ship_sizes()

        # For tracking targeting direction
        self.target_axis_hits = []  # Hits that form the current target line
        self.target_blocked_ends = {"front": False, "back": False}  # For line targeting

        # Move history for training and analysis
        self.move_log = []
        self.turn_count = 0
        self.board_states_history = []  # Store board states for later analysis

        # Feature flags
        self.continuous_learning = continuous_learning
        self.opponent_modeling = opponent_modeling

        # Initialize neural models - using exact approach from AI_agent2
        self.nn_model = None  # Primary heatmap model
        self.init_neural_model()

        # Opponent modeling
        self.opponent_placement_model = defaultdict(float)  # Tracks opponent's ship placement tendencies
        self.opponent_attack_patterns = defaultdict(float)  # Tracks opponent's attack patterns
        self.opponent_profile = self.load_opponent_profile()  # Load previously learned opponent profiles

        # Meta-learning state - automatically adapt weights based on game context
        self.meta_weights = {
            'density': 0.6,
            'neural': 0.5,
            'montecarlo': 0.7,
            'information_gain': 0.5,
            'opponent_model': 0.3
        }

        # Graph-based representation of ship configurations
        self.ship_graph = None
        if NETWORKX_AVAILABLE:
            self.ship_graph = self.initialize_ship_graph()

        # Information theory metrics
        self.entropy_grid = np.zeros((self.board_size, self.board_size))

        # Performance analytics
        self.performance_metrics = {
            'games_played': 0,
            'wins': 0,
            'losses': 0,
            'avg_moves_to_win': 0,
            'hit_rate': 0.0
        }

        # Store the last move and result
        self.last_move = None
        self.last_result = None

        logger.info(f"AI Agent initialized: {name}")

    def init_neural_model(self):
        """
        Initialize the neural model using the exact same approach as in AI_agent2.
        This method is separated to make debugging easier.
        """
        if not TF_AVAILABLE:
            logger.warning("TensorFlow not available, neural model will be disabled")
            return

        # Exact model loading approach from AI_agent2
        try:
            # Construct model path the same way AI_agent2 does
            model_path = os.path.join(MODEL_DIR, MODELS_SUBDIR, "battleship_heatmap.h5")

            # Check file existence
            if not os.path.exists(model_path):
                logger.warning(f"Neural model file not found at: {model_path}")
                # Print all files in the models directory to help debug
                models_dir = os.path.join(MODEL_DIR, MODELS_SUBDIR)
                logger.info(f"Files in models directory ({models_dir}):")
                if os.path.exists(models_dir):
                    for file in os.listdir(models_dir):
                        logger.info(f"  - {file}")
                else:
                    logger.warning("Models directory doesn't exist")
                return

            # Load model with exactly the same parameters
            self.nn_model = load_model(model_path, compile=False)
            logger.info(f"Loaded neural heatmap model from {model_path}")

            # Print model summary for debugging
            print(f"[{self.name}] - Loaded neural heatmap model successfully.")
        except Exception as e:
            logger.error(f"Error loading neural model: {e}")
            self.nn_model = None

    def _update_min_max_ship_sizes(self):
        """Update min and max ship size properties based on remaining ships"""
        if self.remaining_ship_sizes:
            self.min_remaining_ship_size = min(self.remaining_ship_sizes)
            self.max_remaining_ship_size = max(self.remaining_ship_sizes)
        else:
            self.min_remaining_ship_size = 0
            self.max_remaining_ship_size = 0

    def initialize_ship_graph(self) -> Any:
        """Initialize a graph representation of potential ship positions"""
        if not NETWORKX_AVAILABLE:
            return None

        G = nx.Graph()

        # Add all board positions as nodes
        for r in range(self.board_size):
            for c in range(self.board_size):
                G.add_node((r, c), status='unknown', ship_id=None)

        # Add edges for potential ship connections (horizontal and vertical)
        for r in range(self.board_size):
            for c in range(self.board_size):
                # Horizontal connections
                if c < self.board_size - 1:
                    G.add_edge((r, c), (r, c + 1), orientation='horizontal', probability=1.0)
                # Vertical connections
                if r < self.board_size - 1:
                    G.add_edge((r, c), (r + 1, c), orientation='vertical', probability=1.0)

        return G

    def update_ship_graph(self, move: Tuple[int, int], result: str):
        """Update the ship graph based on the result of a move"""
        if not NETWORKX_AVAILABLE or self.ship_graph is None:
            return

        r, c = move

        try:
            # Update node status
            if result == "miss":
                self.ship_graph.nodes[(r, c)]['status'] = 'miss'

                # Update edge probabilities - no ship can go through this cell
                neighbors = list(self.ship_graph.neighbors((r, c)))
                for neighbor in neighbors:
                    if self.ship_graph.has_edge((r, c), neighbor):
                        self.ship_graph.remove_edge((r, c), neighbor)

            elif result in ["hit", "sunk"]:
                self.ship_graph.nodes[(r, c)]['status'] = 'hit'

                # If sunk, mark all hit cells as part of the same ship and remove all other connections
                if result == "sunk":
                    ship_cells = set(self.hits)

                    # Assign a unique ship ID to these cells
                    ship_id = len(self.sunk_ships_info)

                    for hit_r, hit_c in ship_cells:
                        self.ship_graph.nodes[(hit_r, hit_c)]['ship_id'] = ship_id

                        # Remove all edges to non-ship cells
                        neighbors = list(self.ship_graph.neighbors((hit_r, hit_c)))
                        for neighbor in neighbors:
                            if (neighbor[0], neighbor[1]) not in ship_cells:
                                if self.ship_graph.has_edge((hit_r, hit_c), neighbor):
                                    self.ship_graph.remove_edge((hit_r, hit_c), neighbor)
        except Exception as e:
            logger.error(f"Error updating ship graph: {e}")

    def load_opponent_profile(self) -> Dict:
        """Load previously saved opponent profiles"""
        profile_path = os.path.join(MODEL_DIR, MODELS_SUBDIR, "opponent_profiles.pkl")
        if os.path.exists(profile_path):
            try:
                with open(profile_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.error(f"Error loading opponent profiles: {e}")

        # Default profile if none exists
        return {
            'placement_tendencies': defaultdict(float),
            'attack_patterns': defaultdict(float),
            'ship_orientations': {'horizontal': 0.5, 'vertical': 0.5},
            'clustering_tendency': 0.5  # 0.0 = spread out, 1.0 = clustered
        }

    def save_opponent_profile(self):
        """Save learned opponent profile for future games"""
        if not self.opponent_modeling:
            return

        profile_path = os.path.join(MODEL_DIR, MODELS_SUBDIR, "opponent_profiles.pkl")
        try:
            with open(profile_path, 'wb') as f:
                pickle.dump(self.opponent_profile, f)
            logger.info("Saved opponent profile successfully")
        except Exception as e:
            logger.error(f"Error saving opponent profile: {e}")

    def take_turn(self, opponent_board) -> str:
        """Choose a move, attack opponent's board, and update internal state"""
        self.turn_count += 1

        # Capture board state for training
        if self.continuous_learning:
            self.board_states_history.append(self.get_feature_state())

        # Select the next move
        move = self.select_move(opponent_board)

        if move is None:
            if self.available_moves:
                move = random.choice(tuple(self.available_moves))
            else:
                return "no_moves"

        self.last_move = move
        if move in self.available_moves:
            self.available_moves.remove(move)
        else:
            if self.available_moves:  # Pick another if somehow the chosen one was already removed
                move = random.choice(tuple(self.available_moves))
                self.available_moves.remove(move)
            else:  # No moves left at all
                return "no_moves"

        # Attack and get result
        result = opponent_board.attack(move)
        self.last_result = result

        # Log the move for analysis
        self.move_log.append((self.name, move[0], move[1], result))

        # Update internal state based on move result
        self.update_state(move, result, opponent_board)

        return result

    def update_state(self, move: Tuple[int, int], result: str, opponent_board):
        """Update the internal state based on a move result"""
        r, c = move
        if result in ("hit", "sunk"):
            if self.result_grid[r, c] == 0:
                self.hit_count += 1
            self.result_grid[r, c] = 1

            if move not in self.hits:
                self.hits.append(move)

            if self.mode == "hunt":
                self.mode = "target"
                self.target_axis_hits = [move]
                self.target_blocked_ends = {"front": False, "back": False}
            else:
                if move not in self.target_axis_hits:
                    self.target_axis_hits.append(move)
                self._determine_target_axis()

            # Handle sunk ships
            if result == "sunk":
                sunk_ship_identified = False
                ship_obj = None

                if hasattr(opponent_board, "ship_lookup"):
                    ship_obj = opponent_board.ship_lookup.get(move)

                if ship_obj and ship_obj.is_sunk():
                    if ship_obj.size in self.remaining_ship_sizes:
                        sunk_ship_coords = set(ship_obj.coordinates)

                        # Record orientation for opponent modeling
                        if self.opponent_modeling:
                            first_coord = ship_obj.coordinates[0]
                            last_coord = ship_obj.coordinates[-1]
                            is_horizontal = first_coord[0] == last_coord[0]
                            orientation = 'horizontal' if is_horizontal else 'vertical'
                            self.opponent_profile['ship_orientations'][orientation] += 1

                        self.sunk_ships_info.append({"coords": sunk_ship_coords, "size": ship_obj.size})
                        if ship_obj.size in self.remaining_ship_sizes:
                            self.remaining_ship_sizes.remove(ship_obj.size)
                        self._update_min_max_ship_sizes()

                        # Update hits list - remove sunk ship hits
                        self.hits = [h for h in self.hits if h not in sunk_ship_coords]
                        self.target_axis_hits = [h for h in self.target_axis_hits if h not in sunk_ship_coords]

                        sunk_ship_identified = True

                # Reset targeting if no hits remain
                if not self.hits:
                    self.mode = "hunt"
                    self.target_axis_hits = []
                    self.target_blocked_ends = {"front": False, "back": False}
                else:
                    self._determine_target_axis()
                    if not self.target_axis_hits:
                        self.mode = "hunt"
                        self.target_blocked_ends = {"front": False, "back": False}

        elif result == "miss":
            self.result_grid[r, c] = -1
            if self.mode == "target":
                self._mark_blocked_if_miss_at_target_end(move)

        # Update graph representation of ship configurations
        self.update_ship_graph(move, result)

        # Update information theory metrics
        self.update_information_metrics()

    def _determine_target_axis(self):
        """Determine axis of orientation based on current hits"""
        if len(self.hits) < 1:
            self.target_axis_hits = []
            return

        sorted_hits = sorted(self.hits, key=lambda x: (x[0], x[1]))

        if len(sorted_hits) == 1:
            self.target_axis_hits = list(sorted_hits)
            return

        is_horiz = all(h[0] == sorted_hits[0][0] for h in sorted_hits)
        is_vert = all(h[1] == sorted_hits[0][1] for h in sorted_hits)

        if is_horiz or is_vert:
            self.target_axis_hits = list(sorted_hits)
        else:
            # Check if there are collinear subsets
            row_groups = defaultdict(list)
            col_groups = defaultdict(list)

            for hit in sorted_hits:
                row_groups[hit[0]].append(hit)
                col_groups[hit[1]].append(hit)

            # Find the longest row or column group
            longest_row = []
            for row, hits in row_groups.items():
                if len(hits) > len(longest_row):
                    longest_row = hits

            longest_col = []
            for col, hits in col_groups.items():
                if len(hits) > len(longest_col):
                    longest_col = hits

            if len(longest_row) >= len(longest_col) and len(longest_row) > 1:
                self.target_axis_hits = sorted(longest_row, key=lambda x: x[1])
            elif len(longest_col) > 1:
                self.target_axis_hits = sorted(longest_col, key=lambda x: x[0])
            else:
                # If no clear axis, focus on most recent hit
                self.target_axis_hits = [sorted_hits[-1]] if sorted_hits else []

    def _mark_blocked_if_miss_at_target_end(self, miss_move: Tuple[int, int]):
        """Mark a targeting direction as blocked if we miss at the end of a line of hits"""
        if not self.target_axis_hits or len(self.target_axis_hits) < 1:
            return

        current_hits = sorted(self.target_axis_hits, key=lambda t: (t[0], t[1]))

        is_horizontal = False
        if len(current_hits) > 1:
            is_horizontal = current_hits[0][0] == current_hits[-1][0]
        elif len(current_hits) == 1:
            return

        front_hit = current_hits[0]
        back_hit = current_hits[-1]
        front_coord_candidate, back_coord_candidate = None, None

        if is_horizontal:
            front_coord_candidate = (front_hit[0], front_hit[1] - 1)
            back_coord_candidate = (back_hit[0], back_hit[1] + 1)
        else:  # Vertical
            front_coord_candidate = (front_hit[0] - 1, front_hit[1])
            back_coord_candidate = (back_hit[0] + 1, back_hit[1])

        if miss_move == front_coord_candidate:
            self.target_blocked_ends["front"] = True
        elif miss_move == back_coord_candidate:
            self.target_blocked_ends["back"] = True

    def select_move(self, opponent_board=None) -> Optional[Tuple[int, int]]:
        """Select the next move to make based on all available strategies"""
        # Special endgame cases
        if self._squares_remaining_on_board() == 1:
            move = self._select_final_square_overall()
            if move and move in self.available_moves:
                return move

        if len(self.remaining_ship_sizes) == 1:
            move = self._target_final_ship_exhaustive()
            if move and move in self.available_moves:
                return move

        if self._squares_remaining_on_board() <= 4 and self._squares_remaining_on_board() > 1:
            move = self._exhaustive_endgame_search()
            if move and move in self.available_moves:
                return move

        # Target mode - focus on completing partially discovered ships
        if self.mode == "target" and self.hits:
            move = self._select_target_mode_move_enhanced()
            if move and move in self.available_moves:
                return move
            else:
                self.mode = "hunt"
                self.target_axis_hits = []
                self.target_blocked_ends = {"front": False, "back": False}

        # Hunt mode - using combined probability grid with multiple information sources
        prob_grid = self.compute_master_probability_grid()

        # Ensure only available moves have non-zero probability
        for r_idx in range(self.board_size):
            for c_idx in range(self.board_size):
                if (r_idx, c_idx) not in self.available_moves:
                    prob_grid[r_idx, c_idx] = -np.inf

        if prob_grid.max() <= -1:
            if self.available_moves:
                return self._fallback_hunt_move()
            return None

        # Select the best move
        max_prob = prob_grid.max()
        best_moves_indices = np.argwhere(prob_grid >= max_prob - 1e-9)
        valid_best_moves = [tuple(m) for m in best_moves_indices if tuple(m) in self.available_moves]

        if valid_best_moves:
            return random.choice(valid_best_moves)
        elif self.available_moves:
            return self._fallback_hunt_move()

        return None

    def _squares_remaining_on_board(self) -> int:
        """Calculate how many ship squares remain undiscovered"""
        return TOTAL_SHIP_SQUARES - self.hit_count

    def _fallback_hunt_move(self) -> Optional[Tuple[int, int]]:
        """Fallback strategy for hunt mode when other methods fail"""
        # Try parity pattern first
        parity_moves = []
        skip_dist = self.min_remaining_ship_size if self.min_remaining_ship_size > 0 else 2

        current_parity_offset = self.turn_count % skip_dist
        for r_idx in range(self.board_size):
            for c_idx in range(self.board_size):
                if (r_idx + c_idx) % skip_dist == current_parity_offset:
                    if (r_idx, c_idx) in self.available_moves:
                        parity_moves.append((r_idx, c_idx))

        if parity_moves:
            return random.choice(parity_moves)

        # Just pick a random available move
        if self.available_moves:
            return random.choice(tuple(self.available_moves))

        return None

    def compute_master_probability_grid(self) -> np.ndarray:
        """Compute a master probability grid that combines multiple information sources"""
        # Start with basic probability grid
        prob_grid = np.zeros((self.board_size, self.board_size), dtype=float)

        # Add classical ship density probability
        density_grid = self._compute_density_grid()
        if density_grid.max() > 0:
            density_weight = self.meta_weights['density']
            prob_grid += density_weight * (density_grid / density_grid.max())

        # Use parity search pattern for efficiency in hunt mode
        if self.mode == "hunt" and not self.hits and self.min_remaining_ship_size >= PARITY_MIN_SHIP_SIZE:
            parity_mask = self._compute_adaptive_parity_mask()
            if prob_grid.max() > 0:
                prob_grid *= parity_mask
            else:
                prob_grid = parity_mask.astype(float)

        # Add neural network prediction if available
        if TF_AVAILABLE and self.nn_model:
            try:
                nn_heatmap_raw = self._get_neural_heatmap()
                if nn_heatmap_raw is not None and nn_heatmap_raw.max() > 0:
                    neural_weight = self.meta_weights['neural']
                    current_max = prob_grid.max()
                    if current_max > 0 and nn_heatmap_raw.max() > 0:
                        prob_grid = (1 - neural_weight) * prob_grid + neural_weight * (
                                    nn_heatmap_raw / nn_heatmap_raw.max()) * current_max
                    elif nn_heatmap_raw.max() > 0:
                        prob_grid = neural_weight * (nn_heatmap_raw / nn_heatmap_raw.max())
            except Exception as e:
                logger.error(f"Error using neural heatmap: {e}")

        # Add Monte Carlo simulation results
        mc_grid_raw = self._run_monte_carlo_simulations()
        if mc_grid_raw is not None and mc_grid_raw.max() > 0:
            mc_weight = self.meta_weights['montecarlo']
            current_max = prob_grid.max()
            mc_norm_factor = mc_grid_raw.max()
            if current_max > 0:
                prob_grid = (1 - mc_weight) * prob_grid + mc_weight * (mc_grid_raw / mc_norm_factor) * current_max
            elif mc_norm_factor > 0:
                prob_grid = mc_weight * (mc_grid_raw / mc_norm_factor)

        # Add information theory-based probability
        info_grid = self._compute_information_gain_grid()
        if info_grid is not None and info_grid.max() > 0:
            info_weight = self.meta_weights['information_gain']
            current_max = prob_grid.max()
            if current_max > 0:
                prob_grid = (1 - info_weight) * prob_grid + info_weight * (info_grid / info_grid.max()) * current_max

        # Add bonus for neighbors of current hits
        if self.hits:
            hit_bonus_grid = np.zeros_like(prob_grid)
            bonus_magnitude = prob_grid.max() * 2.0 if prob_grid.max() > 0 else 1.0

            hit_coords = []
            for r_hit, c_hit in self.hits:
                hit_coords.append((r_hit, c_hit))

            for r_hit, c_hit in self.hits:
                is_part_of_sunk = False
                for ship_info in self.sunk_ships_info:
                    if (r_hit, c_hit) in ship_info['coords']:
                        is_part_of_sunk = True
                        break

                if is_part_of_sunk:
                    continue

                for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nr, nc = r_hit + dr, c_hit + dc
                    if 0 <= nr < self.board_size and 0 <= nc < self.board_size and self.result_grid[nr, nc] == 0:
                        hit_bonus_grid[nr, nc] += bonus_magnitude

            prob_grid += hit_bonus_grid

        return prob_grid

    def _compute_density_grid(self) -> np.ndarray:
        """Compute ship density probabilities using classical ship placement counting"""
        density_grid = np.zeros((self.board_size, self.board_size), dtype=float)

        # Get sets of constraints
        sunk_cells_set = set()
        for ship_info in self.sunk_ships_info:
            for coord in ship_info['coords']:
                sunk_cells_set.add(coord)

        miss_cells_set = set()
        for r in range(self.board_size):
            for c in range(self.board_size):
                if self.result_grid[r, c] == -1:
                    miss_cells_set.add((r, c))

        must_cover_hits = set(self.hits)

        # Combined set of all unavailable cells for placement checks
        fixed_obstacles = sunk_cells_set | miss_cells_set

        # Try placing each remaining ship in all possible positions
        for ship_len in self.remaining_ship_sizes:
            for r in range(self.board_size):
                for c in range(self.board_size - ship_len + 1):  # Horizontal
                    coords = set((r, c + i) for i in range(ship_len))
                    if not coords & fixed_obstacles:  # Check against misses and sunk ships
                        # Check that we can cover active hits - either this placement covers some/all hits,
                        # or there's a valid configuration where remaining hits are covered by other ships
                        if not must_cover_hits or any(h in coords for h in must_cover_hits) or must_cover_hits.issubset(
                                coords):
                            for sr, sc in coords:
                                if self.result_grid[sr, sc] == 0:
                                    density_grid[sr, sc] += 1

            for c_col in range(self.board_size):  # Vertical
                for r_row in range(self.board_size - ship_len + 1):
                    coords = set((r_row + i, c_col) for i in range(ship_len))
                    if not coords & fixed_obstacles:
                        if not must_cover_hits or any(h in coords for h in must_cover_hits) or must_cover_hits.issubset(
                                coords):
                            for sr, sc in coords:
                                if self.result_grid[sr, sc] == 0:
                                    density_grid[sr, sc] += 1

        return density_grid

    def _compute_adaptive_parity_mask(self) -> np.ndarray:
        """Compute a parity mask that adapts based on minimum ship size and game progress"""
        parity_mask = np.zeros((self.board_size, self.board_size), dtype=int)
        skip = self.min_remaining_ship_size

        if skip <= 1:
            return np.ones_like(parity_mask, dtype=int)

        # Rotate starting offset based on turn count for better coverage
        start_offset = (self.turn_count // (self.board_size // skip + 1)) % skip

        for r in range(self.board_size):
            for c in range(self.board_size):
                if (r + c) % skip == start_offset:
                    parity_mask[r, c] = 1

        return parity_mask

    def _get_neural_heatmap(self) -> Optional[np.ndarray]:
        """Get ship probability heatmap from neural network model"""
        # Replicate exactly how AIPlayer2 does it
        if not TF_AVAILABLE or self.nn_model is None:
            return None

        # Prepare input state tensor
        miss_plane = (self.result_grid == -1).astype(np.float32)
        hit_plane = (self.result_grid == 1).astype(np.float32)
        unknown_plane = (self.result_grid == 0).astype(np.float32)
        state_tensor = np.stack([miss_plane, hit_plane, unknown_plane], axis=-1)
        state_tensor = np.expand_dims(state_tensor, axis=0)

        # Get predictions from model using the same approach as AIPlayer2
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # Suppress deprecation warnings
                predictions = self.nn_model.predict(state_tensor, verbose=0)[0]

            # Handle different output shapes
            if len(predictions.shape) == 3 and predictions.shape[2] == 1:
                predictions = predictions.reshape((self.board_size, self.board_size))

            if predictions.shape != (self.board_size, self.board_size):
                logger.warning(
                    f"Neural prediction shape mismatch: expected {(self.board_size, self.board_size)}, got {predictions.shape}")
                return None

            # Mask unavailable moves
            final_preds = np.zeros_like(predictions)
            for r in range(self.board_size):
                for c in range(self.board_size):
                    if (r, c) in self.available_moves:
                        final_preds[r, c] = predictions[r, c]

            return final_preds
        except Exception as e:
            logger.error(f"Error in neural prediction: {e}")
            return None

    def _run_monte_carlo_simulations(self) -> Optional[np.ndarray]:
        """Run Monte Carlo simulations to estimate ship location probabilities"""
        if not self.available_moves or not self.remaining_ship_sizes:
            return None

        occupancy_grid = np.zeros((self.board_size, self.board_size), dtype=int)

        # Convert sets to lists for safety during iteration
        sunk_cells_set = set()
        for si in self.sunk_ships_info:
            for coord in si['coords']:
                sunk_cells_set.add(coord)

        misses_set = set()
        for r in range(self.board_size):
            for c in range(self.board_size):
                if self.result_grid[r, c] == -1:
                    misses_set.add((r, c))

        hits_set = set(self.hits)

        successful_placements = 0
        # Increase samples for endgame to improve accuracy
        num_samples = MC_SAMPLES_BASE * (3 if self._squares_remaining_on_board() <= 4 else 1)

        # For each sample, attempt to place all remaining ships
        for _ in range(num_samples):
            current_board_ship_cells = set()
            temp_remaining_ships = list(self.remaining_ship_sizes)

            # Randomize ship placement order
            random.shuffle(temp_remaining_ships)
            possible_config = True

            # Try to place each remaining ship
            for ship_len in temp_remaining_ships:
                placed_this_ship = False

                # Make multiple attempts to place each ship
                for _attempt in range(20):
                    # Choose random orientation and position
                    is_horiz = random.choice([True, False])
                    r_coord = random.randrange(self.board_size) if is_horiz else random.randrange(
                        self.board_size - ship_len + 1)
                    c_coord = random.randrange(self.board_size - ship_len + 1) if is_horiz else random.randrange(
                        self.board_size)

                    # Generate ship coordinates
                    potential_coords = set()
                    if is_horiz:
                        for i in range(ship_len):
                            potential_coords.add((r_coord, c_coord + i))
                    else:
                        for i in range(ship_len):
                            potential_coords.add((r_coord + i, c_coord))

                    # Check if placement is valid
                    if not potential_coords & misses_set and \
                            not potential_coords & sunk_cells_set and \
                            not potential_coords & current_board_ship_cells:
                        current_board_ship_cells.update(potential_coords)
                        placed_this_ship = True
                        break

                if not placed_this_ship:
                    possible_config = False
                    break

            # If we successfully placed all ships and cover all known hits
            if possible_config and hits_set.issubset(current_board_ship_cells):
                successful_placements += 1
                for r_mc, c_mc in current_board_ship_cells:
                    if (r_mc, c_mc) in self.available_moves:
                        occupancy_grid[r_mc, c_mc] += 1

        if successful_placements == 0 or occupancy_grid.max() == 0:
            return None

        return occupancy_grid.astype(float) / occupancy_grid.max()

    def update_information_metrics(self):
        """Update information theory metrics for all cells"""
        # Skip if no remaining ship squares
        if self._squares_remaining_on_board() == 0:
            return

        # Initialize entropy grid
        self.entropy_grid = np.zeros((self.board_size, self.board_size))

        # Run MC simulations to estimate ship presence probability distribution
        mc_grid = self._run_monte_carlo_simulations()
        if mc_grid is None:
            return

        # Compute entropy at each cell
        for r in range(self.board_size):
            for c in range(self.board_size):
                if (r, c) in self.available_moves:
                    p = mc_grid[r, c]
                    if 0 < p < 1:  # Only compute for meaningful probabilities
                        # Binary entropy: -p*log(p) - (1-p)*log(1-p)
                        if SCIPY_AVAILABLE:
                            self.entropy_grid[r, c] = entropy([p, 1 - p], base=2)
                        else:
                            self.entropy_grid[r, c] = simple_entropy(p)

    def _compute_information_gain_grid(self) -> np.ndarray:
        """Compute expected information gain for each possible move"""
        # Start with current entropy values
        if not hasattr(self, 'entropy_grid') or self.entropy_grid is None:
            self.update_information_metrics()

        if not hasattr(self, 'entropy_grid') or self.entropy_grid is None:
            return None

        # For each cell, compute expected information gain
        info_gain = np.zeros((self.board_size, self.board_size))

        # Run MC simulations to estimate hit probability
        mc_grid = self._run_monte_carlo_simulations()
        if mc_grid is None:
            return self.entropy_grid  # Fallback to entropy if MC fails

        # For each cell, compute expected information gain
        for r in range(self.board_size):
            for c in range(self.board_size):
                if (r, c) in self.available_moves:
                    p_hit = mc_grid[r, c]  # Probability of hit

                    # Expected information gain = current entropy - weighted sum of conditional entropies
                    # We approximate this with the probability of hit and current entropy
                    info_gain[r, c] = self.entropy_grid[r, c] * p_hit

        return info_gain

    def _select_target_mode_move_enhanced(self) -> Optional[Tuple[int, int]]:
        """Enhanced targeting mode move selection"""
        if not self.hits:
            self.mode = "hunt"
            return None

        potential_moves = []
        self._determine_target_axis()

        # If we have aligned hits, extend along the axis
        if self.target_axis_hits:
            current_axis_hits = sorted(self.target_axis_hits, key=lambda t: (t[0], t[1]))

            if len(current_axis_hits) >= 1:
                min_h, max_h = current_axis_hits[0], current_axis_hits[-1]
                is_horiz = len(current_axis_hits) > 1 and min_h[0] == max_h[0]
                is_vert = len(current_axis_hits) > 1 and min_h[1] == max_h[1]

                if is_horiz:
                    if not self.target_blocked_ends["front"] and min_h[1] > 0:
                        potential_moves.append((min_h[0], min_h[1] - 1))
                    if not self.target_blocked_ends["back"] and max_h[1] < self.board_size - 1:
                        potential_moves.append((max_h[0], max_h[1] + 1))
                elif is_vert:
                    if not self.target_blocked_ends["front"] and min_h[0] > 0:
                        potential_moves.append((min_h[0] - 1, min_h[1]))
                    if not self.target_blocked_ends["back"] and max_h[0] < self.board_size - 1:
                        potential_moves.append((max_h[0] + 1, max_h[1]))
                else:
                    # Single hit - try all four directions
                    r_single, c_single = current_axis_hits[0]
                    dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
                    for dr, dc in dirs:
                        nr, nc = r_single + dr, c_single + dc
                        if 0 <= nr < self.board_size and 0 <= nc < self.board_size:
                            potential_moves.append((nr, nc))
        else:
            # Fallback - collect all orthogonal neighbors of all hits
            temp_potential_moves = set()
            for r_h, c_h in self.hits:
                for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nr, nc = r_h + dr, c_h + dc
                    if 0 <= nr < self.board_size and 0 <= nc < self.board_size:
                        temp_potential_moves.add((nr, nc))
            potential_moves.extend(list(temp_potential_moves))

        # Filter to valid moves
        valid_moves = [m for m in potential_moves if m in self.available_moves]

        if not valid_moves:
            self.mode = "hunt"
            self.target_axis_hits = []
            self.target_blocked_ends = {"front": False, "back": False}
            return None

        return self._prioritize_target_moves(valid_moves)

    def _prioritize_target_moves(self, moves: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        """Prioritize targeting moves using available heuristics and models"""
        if not moves:
            return None

        # If we have a neural model, use it for ranking
        if TF_AVAILABLE and self.nn_model:
            heatmap = self._get_neural_heatmap()
            if heatmap is not None:
                scored_moves = []
                for m in moves:
                    scored_moves.append((m, heatmap[m[0], m[1]]))
                scored_moves.sort(key=lambda x: x[1], reverse=True)
                if scored_moves:
                    return scored_moves[0][0]

        # If we have a graph representation, use it
        if NETWORKX_AVAILABLE and self.ship_graph:
            # Calculate connection scores based on the graph
            move_scores = {}
            for move in moves:
                score = 0
                # Check connections to existing hits
                for hit in self.hits:
                    if self.ship_graph.has_edge(move, hit):
                        # Higher score for aligned hits (recognized patterns)
                        if len(self.target_axis_hits) >= 2:
                            is_aligned = False
                            if self.target_axis_hits[0][0] == self.target_axis_hits[1][0]:  # horizontal
                                is_aligned = move[0] == self.target_axis_hits[0][0]
                            else:  # vertical
                                is_aligned = move[1] == self.target_axis_hits[0][1]
                            score += 2 if is_aligned else 1
                        else:
                            score += 1
                move_scores[move] = score

            # Get highest scoring moves
            if move_scores:
                max_score = max(move_scores.values())
                best_moves = [m for m, s in move_scores.items() if s == max_score]

                if best_moves:
                    return random.choice(best_moves)

        # Fallback to random selection from moves
        return random.choice(moves) if moves else None

    def _select_final_square_overall(self) -> Optional[Tuple[int, int]]:
        """Select the final square when only one ship square remains on the board"""
        if self._squares_remaining_on_board() != 1:
            return None
        if not self.available_moves:
            return None

        if self.hits:
            # Prioritize neighbors of current hits
            candidate_final_squares = set()
            for r_hit, c_hit in self.hits:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r_hit + dr, c_hit + dc
                    if 0 <= nr < self.board_size and 0 <= nc < self.board_size and (nr, nc) in self.available_moves:
                        candidate_final_squares.add((nr, nc))

            if candidate_final_squares:
                # If we have aligned hits, prioritize extending the line
                sorted_hits = sorted(self.hits, key=lambda x: (x[0], x[1]))
                if len(sorted_hits) >= 1:
                    min_h, max_h = sorted_hits[0], sorted_hits[-1]
                    is_horiz = len(sorted_hits) > 1 and min_h[0] == max_h[0]
                    is_vert = len(sorted_hits) > 1 and min_h[1] == max_h[1]
                    line_extension_candidates = []

                    if is_horiz:
                        front_coord = (min_h[0], min_h[1] - 1)
                        back_coord = (max_h[0], max_h[1] + 1)
                        if front_coord in candidate_final_squares:
                            line_extension_candidates.append(front_coord)
                        if back_coord in candidate_final_squares:
                            line_extension_candidates.append(back_coord)
                    elif is_vert:
                        front_coord = (min_h[0] - 1, min_h[1])
                        back_coord = (max_h[0] + 1, max_h[1])
                        if front_coord in candidate_final_squares:
                            line_extension_candidates.append(front_coord)
                        if back_coord in candidate_final_squares:
                            line_extension_candidates.append(back_coord)

                    if line_extension_candidates:
                        return random.choice(line_extension_candidates)

                return random.choice(list(candidate_final_squares))

        # Fallback if no clear target - use probabilistic methods
        prob_grid = self.compute_master_probability_grid()
        best_vals = np.argsort(prob_grid.ravel())[-5:]  # Get indices of top 5 probabilities
        best_coords = [divmod(int(idx), self.board_size) for idx in best_vals]
        valid_coords = [(int(r), int(c)) for r, c in best_coords if (int(r), int(c)) in self.available_moves]

        if valid_coords:
            return valid_coords[0]  # Return highest probability

        # Final fallback - random move
        return random.choice(tuple(self.available_moves)) if self.available_moves else None

    def _target_final_ship_exhaustive(self) -> Optional[Tuple[int, int]]:
        """Exhaustive search when only one ship remains"""
        if len(self.remaining_ship_sizes) != 1:
            return None

        ship_len = self.remaining_ship_sizes[0]
        current_hits = list(self.hits)

        if not current_hits and self.min_remaining_ship_size > 0:
            return None
        elif not current_hits and self.min_remaining_ship_size == 0:
            return None

        # Identify constraints
        miss_squares = set()
        for r in range(self.board_size):
            for c in range(self.board_size):
                if self.result_grid[r, c] == -1:
                    miss_squares.add((r, c))

        sunk_ship_coords_set = set()
        for ship_info in self.sunk_ships_info:
            for coord in ship_info['coords']:
                sunk_ship_coords_set.add(coord)

        valid_placements_coords = []

        # Try all possible placements for the final ship
        for r_start in range(self.board_size):
            for c_start in range(self.board_size):
                for dr_orient, dc_orient in [(0, 1), (1, 0)]:  # horizontal and vertical
                    current_placement_cells = set()
                    possible = True

                    # Generate ship coordinates
                    for i in range(ship_len):
                        r_cell, c_cell = r_start + i * dr_orient, c_start + i * dc_orient
                        if not (0 <= r_cell < self.board_size and 0 <= c_cell < self.board_size):
                            possible = False
                            break
                        current_placement_cells.add((r_cell, c_cell))

                    if not possible:
                        continue

                    # Check constraints
                    if not all(h_coord in current_placement_cells for h_coord in current_hits):
                        continue
                    if current_placement_cells & miss_squares:
                        continue
                    if current_placement_cells & sunk_ship_coords_set:
                        continue

                    valid_placements_coords.append(current_placement_cells)

        if not valid_placements_coords:
            return None

        # Count frequency of each cell in valid placements
        freq_map = {}
        for placement in valid_placements_coords:
            for cell in placement:
                if cell in self.available_moves:
                    freq_map[cell] = freq_map.get(cell, 0) + 1

        if not freq_map:
            return self._select_target_mode_move_enhanced() if self.hits else None

        # Rank options by frequency, breaking ties with neural model if available
        if TF_AVAILABLE and self.nn_model:
            heatmap = self._get_neural_heatmap()
            if heatmap is not None:
                # Sort by frequency first, then neural model score
                scored_options = sorted(
                    [(cell, (freq, heatmap[cell[0], cell[1]])) for cell, freq in freq_map.items()],
                    key=lambda x: (x[1][0], x[1][1]),
                    reverse=True
                )
                if scored_options:
                    return scored_options[0][0]

        # Fallback to just frequency ranking
        best_cell = max(freq_map.items(), key=lambda x: x[1])[0] if freq_map else None
        return best_cell

    def _exhaustive_endgame_search(self) -> Optional[Tuple[int, int]]:
        """Exhaustive search for optimal move when few ship squares remain"""
        if not self.remaining_ship_sizes:
            return None

        # Identify constraints
        miss_coords = set()
        for r in range(self.board_size):
            for c in range(self.board_size):
                if self.result_grid[r, c] == -1:
                    miss_coords.add((r, c))

        hit_coords_must_cover = set(self.hits)
        sunk_coords_occupied = set()
        for ship_info in self.sunk_ships_info:
            for coord in ship_info['coords']:
                sunk_coords_occupied.add(coord)

        # Map to track frequency of each cell in valid configurations
        candidate_square_freq = {}

        # Sort ships by size (decreasing) for better pruning
        ships_to_place = sorted(list(self.remaining_ship_sizes), reverse=True)

        # Limit search depth for performance
        if len(ships_to_place) > 3:
            logger.info(f"Too many ships ({len(ships_to_place)}) for exhaustive search, using regular targeting")
            return self._select_target_mode_move_enhanced() if self.hits else self._fallback_hunt_move()

        # Memoization cache for recursive function
        memo_recurse = {}

        def recurse_placements(ship_idx, currently_occupied):
            """Recursive function to try all valid ship placements"""
            # Convert to immutable type for memoization
            state_key = (ship_idx, tuple(sorted(currently_occupied)))
            if state_key in memo_recurse:
                return memo_recurse[state_key]

            # Base case: all ships placed
            if ship_idx == len(ships_to_place):
                # Valid if all hits are covered
                if hit_coords_must_cover.issubset(set(currently_occupied) | sunk_coords_occupied):
                    # Count each cell in this valid configuration
                    for cell_coord in currently_occupied:
                        if cell_coord in self.available_moves:
                            candidate_square_freq[cell_coord] = candidate_square_freq.get(cell_coord, 0) + 1
                    return True
                return False

            # Get current ship length
            current_ship_len = ships_to_place[ship_idx]
            found_any = False

            # Try all possible placements for current ship
            for r_start in range(self.board_size):
                for c_start in range(self.board_size):
                    for dr_orient, dc_orient in [(0, 1), (1, 0)]:  # horizontal and vertical
                        current_ship_coords = []
                        possible_to_place = True

                        # Generate ship coordinates
                        for i in range(current_ship_len):
                            r_cell, c_cell = r_start + i * dr_orient, c_start + i * dc_orient

                            # Check bounds
                            if not (0 <= r_cell < self.board_size and 0 <= c_cell < self.board_size):
                                possible_to_place = False
                                break

                            # Check conflicts
                            if (r_cell, c_cell) in miss_coords or \
                                    (r_cell, c_cell) in sunk_coords_occupied or \
                                    (r_cell, c_cell) in currently_occupied:
                                possible_to_place = False
                                break

                            current_ship_coords.append((r_cell, c_cell))

                        # If placement is valid, continue to next ship
                        if possible_to_place:
                            new_occupied = currently_occupied + current_ship_coords
                            if recurse_placements(ship_idx + 1, new_occupied):
                                found_any = True

            # Cache result before returning
            memo_recurse[state_key] = found_any
            return found_any

        # Start recursion with no ships placed
        recurse_placements(0, [])

        if not candidate_square_freq:
            return None

        # Rank options by frequency
        best_cell = max(candidate_square_freq.items(), key=lambda x: x[1])[0] if candidate_square_freq else None
        return best_cell

    def get_feature_state(self) -> Dict:
        """Create a comprehensive feature state for learning and analysis"""
        state = {
            "turn": self.turn_count,
            "mode": self.mode,
            "hit_count": self.hit_count,
            "board_state": self.result_grid.copy(),
            "remaining_ships": list(self.remaining_ship_sizes),
            "active_hits": list(self.hits),
        }
        return state

    def learn_from_game(self, game_result: str, opponent_moves=None):
        """Update knowledge and models based on game results"""
        if not self.continuous_learning:
            return

        try:
            # Record game result
            self.performance_metrics['games_played'] += 1
            if game_result == "win":
                self.performance_metrics['wins'] += 1
            else:
                self.performance_metrics['losses'] += 1

            # Update opponent modeling if enabled
            if self.opponent_modeling and opponent_moves:
                self._update_opponent_model(opponent_moves)

            # Save updated metrics and models
            self._save_performance_metrics()

            logger.info(f"Learned from game: {game_result}")

        except Exception as e:
            logger.error(f"Error during learning from game: {e}")

    def _update_opponent_model(self, opponent_moves):
        """Update opponent modeling based on observed moves"""
        if not opponent_moves:
            return

        try:
            # Create a new dictionary for attack patterns
            attack_patterns = dict(self.opponent_profile['attack_patterns'])

            # Update attack pattern frequencies
            for move in opponent_moves:
                r, c = move[0], move[1]
                key = f"{r}-{c}"
                if key in attack_patterns:
                    attack_patterns[key] += 1
                else:
                    attack_patterns[key] = 1

            # Normalize patterns
            total = sum(attack_patterns.values())
            if total > 0:
                for k in attack_patterns:
                    attack_patterns[k] /= total

            # Update the profile
            self.opponent_profile['attack_patterns'] = attack_patterns

            # Save updated profile
            self.save_opponent_profile()

        except Exception as e:
            logger.error(f"Error updating opponent model: {e}")

    def _save_performance_metrics(self):
        """Save performance metrics to file"""
        metrics_path = os.path.join(MODEL_DIR, MODELS_SUBDIR, "ai_performance_metrics.pkl")
        try:
            with open(metrics_path, 'wb') as f:
                pickle.dump(self.performance_metrics, f)
        except Exception as e:
            logger.error(f"Error saving performance metrics: {e}")

    def view_display(self) -> str:
        """Generate a string representation of the current board state"""
        mapping = {0: ".", -1: "O", 1: "X"}
        board_str = f"[{self.name}] View (Turn: {self.turn_count}):\n  " + " ".join(
            map(str, range(self.board_size))) + "\n"

        for r in range(self.board_size):
            row_str = f"{r} "
            for c in range(self.board_size):
                is_sunk_part = False
                for ship_info in self.sunk_ships_info:
                    if (r, c) in ship_info['coords']:
                        is_sunk_part = True
                        break

                if is_sunk_part:
                    row_str += "S "
                else:
                    row_str += mapping[self.result_grid[r, c]] + " "
            board_str += row_str.strip() + "\n"

        board_str += f"Mode: {self.mode}, Hits: {self.hits}, Blocked Ends: {self.target_blocked_ends}\n"
        board_str += f"Rem Ships: {self.remaining_ship_sizes} (Min: {self.min_remaining_ship_size}, Max: {self.max_remaining_ship_size})\n"
        board_str += f"Hit Count: {self.hit_count}, Squares Rem: {self._squares_remaining_on_board()}, Moves Left: {len(self.available_moves)}"

        return board_str.strip()