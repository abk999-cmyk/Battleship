import random
from typing import List, Set, Tuple, Optional, Dict
import numpy as np
from player import Player
from board import SHIP_SIZES  # Assuming SHIP_SIZES is accessible from board.py or defined here

# ----------------------------------------------------------------------
# Global constants
# ----------------------------------------------------------------------
BOARD_SIZE: int = 10
TOTAL_SHIP_SQUARES: int = sum(SHIP_SIZES)
MC_SAMPLES_BASE: int = 1500  # Base Monte-Carlo samples
PARITY_MIN_SHIP_SIZE: int = 2  # Min ship size to apply parity checks actively

# Attempt to import TensorFlow for neural network model
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model

    TF_AVAILABLE = True
except ImportError:
    tf = None
    load_model = None
    TF_AVAILABLE = False


class AIPlayer2(Player):
    def __init__(self, name: str):
        super().__init__(name)
        self.board_size = BOARD_SIZE
        self.board.place_ships()  # Player places their own ships

        # Core state variables
        self.available_moves: Set[Tuple[int, int]] = {(r, c) for r in range(self.board_size) for c in
                                                      range(self.board_size)}
        self.result_grid: np.ndarray = np.zeros((self.board_size, self.board_size),
                                                dtype=int)  # 0: unknown, 1: hit, -1: miss

        self.hits: List[Tuple[int, int]] = []  # Current, unconfirmed hits (part of a ship not yet sunk)
        self.sunk_ships_info: List[Dict] = []  # Store info about ships we've sunk (coords, size)

        self.hit_count: int = 0  # Total squares confirmed as part of any ship (hit or sunk)

        self.mode: str = "hunt"  # "hunt" or "target"

        self.remaining_ship_sizes: List[int] = sorted(list(SHIP_SIZES), reverse=True)
        self._update_min_max_ship_sizes()

        # For tracking targeting direction
        self.target_axis_hits: List[Tuple[int, int]] = []  # Hits that form the current target line
        self.target_blocked_ends: Dict[str, bool] = {"front": False, "back": False}  # For line targeting

        # Neural network model
        self.nn_model = None
        if TF_AVAILABLE and load_model:
            try:
                self.nn_model = load_model("models/battleship_heatmap.h5", compile=False)
                print(f"[{self.name}] - Loaded neural heatmap model successfully.")
            except Exception as e:
                print(f"[{self.name}] - Warning: Could not load neural model. Error: {e}")
        else:
            print(f"[{self.name}] - Info: TensorFlow not available or model loading failed. Neural model disabled.")

        self.turn_count = 0  # For adaptive strategies if needed

    def _update_min_max_ship_sizes(self):
        if self.remaining_ship_sizes:
            self.min_remaining_ship_size: int = min(self.remaining_ship_sizes)
            self.max_remaining_ship_size: int = max(self.remaining_ship_sizes)
        else:
            self.min_remaining_ship_size: int = 0
            self.max_remaining_ship_size: int = 0

    def take_turn(self, opponent_board) -> str:
        self.turn_count += 1
        move = self.select_move(opponent_board)

        if move is None:
            if self.available_moves:
                move = random.choice(list(self.available_moves))
            else:
                return "no_moves"

        self.last_move = move
        if move in self.available_moves:
            self.available_moves.remove(move)
        else:
            if self.available_moves:  # Pick another if somehow the chosen one was already removed
                move = random.choice(list(self.available_moves))
                self.available_moves.remove(move)
            else:  # No moves left at all
                return "no_moves"

        result: str = opponent_board.attack(move)
        self.last_result = result

        self.update_state(move, result, opponent_board)
        return result

    def update_state(self, move: Tuple[int, int], result: str, opponent_board):
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

            if result == "sunk":
                sunk_ship_identified = False
                if hasattr(opponent_board, "ship_lookup"):
                    ship_obj = opponent_board.ship_lookup.get(move)
                    if ship_obj and ship_obj.is_sunk():
                        if ship_obj.size in self.remaining_ship_sizes:
                            sunk_ship_coords = set(ship_obj.coordinates)
                            self.sunk_ships_info.append({"coords": sunk_ship_coords, "size": ship_obj.size})
                            self.remaining_ship_sizes.remove(ship_obj.size)
                            self._update_min_max_ship_sizes()

                            self.hits = [h for h in self.hits if h not in sunk_ship_coords]
                            self.target_axis_hits = [h for h in self.target_axis_hits if h not in sunk_ship_coords]

                            sunk_ship_identified = True

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

    def _determine_target_axis(self):
        if len(self.hits) < 1:
            self.target_axis_hits = []
            return

        self.hits.sort(key=lambda x: (x[0], x[1]))

        if len(self.hits) == 1:
            self.target_axis_hits = list(self.hits)
            return

        is_horiz = all(h[0] == self.hits[0][0] for h in self.hits)
        is_vert = all(h[1] == self.hits[0][1] for h in self.hits)

        if is_horiz or is_vert:
            self.target_axis_hits = list(self.hits)
        else:
            # If current self.hits are not collinear, self.target_axis_hits should be
            # the longest collinear subset of self.hits containing the latest hit.
            # For simplicity now, if they are not collinear, we could focus on the latest hit.
            # However, current update_state calls this after adding a hit, so self.hits could grow non-collinear.
            # A more robust way: find the longest collinear sequence within self.hits.
            # For now, if not all hits are collinear, we keep the previous valid axis or reset to latest hit.
            # If self.target_axis_hits itself is not collinear (which shouldn't happen with this func),
            # then reset it.
            current_axis_is_horiz = len(self.target_axis_hits) > 1 and all(
                h[0] == self.target_axis_hits[0][0] for h in self.target_axis_hits)
            current_axis_is_vert = len(self.target_axis_hits) > 1 and all(
                h[1] == self.target_axis_hits[0][1] for h in self.target_axis_hits)

            if not (
                    current_axis_is_horiz or current_axis_is_vert) and self.target_axis_hits:  # if current axis is invalid
                self.target_axis_hits = [self.hits[-1]] if self.hits else []  # Focus on last hit

    def _mark_blocked_if_miss_at_target_end(self, miss_move: Tuple[int, int]):
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
        if self._squares_remaining_on_board() == 1:
            move = self._select_final_square_overall()
            if move and move in self.available_moves: return move

        if len(self.remaining_ship_sizes) == 1:
            move = self._target_final_ship_exhaustive()
            if move and move in self.available_moves: return move

        if self._squares_remaining_on_board() <= 4 and self._squares_remaining_on_board() > 1:
            move = self._exhaustive_endgame_search_from_original()
            if move and move in self.available_moves: return move

        if self.mode == "target" and self.hits:
            move = self._select_target_mode_move_enhanced()
            if move and move in self.available_moves:
                return move
            else:
                self.mode = "hunt"
                self.target_axis_hits = []
                self.target_blocked_ends = {"front": False, "back": False}

        prob_grid = self.compute_master_probability_grid()
        for r_idx in range(self.board_size):
            for c_idx in range(self.board_size):
                if (r_idx, c_idx) not in self.available_moves:
                    prob_grid[r_idx, c_idx] = -np.inf

        if prob_grid.max() <= -1:
            if self.available_moves:
                return self._fallback_hunt_move()
            return None

        max_prob = prob_grid.max()
        best_moves_indices = np.argwhere(prob_grid >= max_prob - 1e-9)
        valid_best_moves = [tuple(m) for m in best_moves_indices if tuple(m) in self.available_moves]

        if valid_best_moves:
            return random.choice(valid_best_moves)
        elif self.available_moves:
            return self._fallback_hunt_move()

        return None

    def _squares_remaining_on_board(self) -> int:
        return TOTAL_SHIP_SQUARES - self.hit_count

    def _fallback_hunt_move(self) -> Optional[Tuple[int, int]]:
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

        if self.available_moves:
            return random.choice(list(self.available_moves))
        return None

    def compute_master_probability_grid(self) -> np.ndarray:
        prob_grid = np.zeros((self.board_size, self.board_size), dtype=float)
        density_grid = self._compute_density_grid()
        if density_grid.max() > 0:
            prob_grid += density_grid / density_grid.max()

        if self.mode == "hunt" and not self.hits and self.min_remaining_ship_size >= PARITY_MIN_SHIP_SIZE:
            parity_mask = self._compute_aggressive_parity_mask()
            if prob_grid.max() > 0:
                prob_grid *= parity_mask
            else:
                prob_grid = parity_mask.astype(float)

        nn_heatmap_raw = None
        if self.nn_model:
            try:
                nn_heatmap_raw = self._get_neural_heatmap()
                if nn_heatmap_raw is not None and nn_heatmap_raw.max() > 0:
                    alpha = 0.6 if self.mode == "hunt" else 0.2
                    if self.hit_count == 0: alpha = 0.7
                    current_max = prob_grid.max()
                    if current_max > 0 and nn_heatmap_raw.max() > 0:  # Check divisor
                        prob_grid = (1 - alpha) * prob_grid + alpha * (
                                    nn_heatmap_raw / nn_heatmap_raw.max()) * current_max
                    elif nn_heatmap_raw.max() > 0:  # If prob_grid is flat zero and nn_heatmap is not
                        prob_grid = alpha * (nn_heatmap_raw / nn_heatmap_raw.max())
            except Exception as e:
                print(f"[{self.name}] Error during NN heatmap usage: {e}")

        mc_grid_raw = self._run_monte_carlo_simulations_enhanced()
        if mc_grid_raw is not None and mc_grid_raw.max() > 0:
            beta = 0.5
            if not self.hits: beta = 0.3
            current_max = prob_grid.max()
            mc_norm_factor = mc_grid_raw.max()
            if current_max > 0:
                prob_grid = (1 - beta) * prob_grid + beta * (mc_grid_raw / mc_norm_factor) * current_max
            elif mc_norm_factor > 0:
                prob_grid = beta * (mc_grid_raw / mc_norm_factor)

        if self.hits:
            hit_bonus_grid = np.zeros_like(prob_grid)
            bonus_magnitude = prob_grid.max() * 2.0 if prob_grid.max() > 0 else 1.0
            for r_hit, c_hit in self.hits:
                is_part_of_sunk = any((r_hit, c_hit) in sunk_info['coords'] for sunk_info in self.sunk_ships_info)
                if is_part_of_sunk: continue
                for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nr, nc = r_hit + dr, c_hit + dc
                    if 0 <= nr < self.board_size and 0 <= nc < self.board_size and self.result_grid[nr, nc] == 0:
                        hit_bonus_grid[nr, nc] += bonus_magnitude
            prob_grid += hit_bonus_grid
        return prob_grid

    def _compute_density_grid(self) -> np.ndarray:
        density_grid = np.zeros((self.board_size, self.board_size), dtype=float)
        sunk_cells_set = set(coord for ship_info in self.sunk_ships_info for coord in ship_info['coords'])
        miss_cells_set = set(tuple(coord) for coord in np.argwhere(self.result_grid == -1))
        must_cover_hits = set(self.hits)
        # Create a combined set of all unavailable cells for placement checks
        fixed_obstacles = sunk_cells_set | miss_cells_set

        for ship_len in self.remaining_ship_sizes:
            for r in range(self.board_size):
                for c in range(self.board_size - ship_len + 1):  # Horizontal
                    coords = {(r, c + i) for i in range(ship_len)}
                    if not (coords & fixed_obstacles):  # Check against misses and sunk ships
                        # If there are active hits, this placement must be capable of covering them.
                        # This means either this placement itself covers some/all hits,
                        # OR the remaining hits can be covered by OTHER future ships.
                        # The original simpler check was: if must_cover_hits.issubset(coords | other_non_obstacle_cells)
                        # For density, we just count valid placements. The MC sim handles hit coverage more directly.
                        # Let's simplify: if a placement is geometrically valid (no overlap with fixed obstacles), count it.
                        # The probabilistic combination later handles conditioning on hits.
                        # However, if must_cover_hits is non-empty, a placement NOT covering ANY of them, while others do, is less likely.
                        # A stricter density would be: count placements that ARE consistent with current hits.
                        if not must_cover_hits or any(h in coords for h in must_cover_hits) or must_cover_hits.issubset(
                                coords):
                            for sr, sc in coords:
                                if self.result_grid[sr, sc] == 0: density_grid[sr, sc] += 1
            for c_col in range(self.board_size):  # Renamed c to c_col to avoid conflict
                for r_row in range(self.board_size - ship_len + 1):  # Vertical, r_row
                    coords = {(r_row + i, c_col) for i in range(ship_len)}
                    if not (coords & fixed_obstacles):
                        if not must_cover_hits or any(h in coords for h in must_cover_hits) or must_cover_hits.issubset(
                                coords):
                            for sr, sc in coords:
                                if self.result_grid[sr, sc] == 0: density_grid[sr, sc] += 1
        return density_grid

    def _compute_aggressive_parity_mask(self) -> np.ndarray:
        parity_mask = np.zeros((self.board_size, self.board_size), dtype=int)
        skip = self.min_remaining_ship_size
        if skip <= 1: return np.ones_like(parity_mask, dtype=int)

        start_offset = (self.turn_count // (self.board_size // skip + 1)) % skip
        for r in range(self.board_size):
            for c in range(self.board_size):
                if (r + c) % skip == start_offset:
                    parity_mask[r, c] = 1
        return parity_mask

    def _get_neural_heatmap(self) -> Optional[np.ndarray]:
        if not self.nn_model: return None
        miss_plane = (self.result_grid == -1).astype(np.float32)
        hit_plane = (self.result_grid == 1).astype(np.float32)
        unknown_plane = (self.result_grid == 0).astype(np.float32)
        state_tensor = np.stack([miss_plane, hit_plane, unknown_plane], axis=-1)
        state_tensor = np.expand_dims(state_tensor, axis=0)
        predictions = self.nn_model.predict(state_tensor, verbose=0)[0]
        if predictions.shape != (self.board_size, self.board_size) and predictions.shape == (
        self.board_size, self.board_size, 1):
            predictions = predictions.reshape((self.board_size, self.board_size))
        elif predictions.shape != (self.board_size, self.board_size):
            # print(f"[{self.name}] NN prediction shape mismatch: {predictions.shape}")
            return None

        final_preds = np.zeros_like(predictions)
        for r_idx in range(self.board_size):
            for c_idx in range(self.board_size):
                if (r_idx, c_idx) in self.available_moves:
                    final_preds[r_idx, c_idx] = predictions[r_idx, c_idx]
        return final_preds

    def _run_monte_carlo_simulations_enhanced(self) -> Optional[np.ndarray]:
        if not self.available_moves or not self.remaining_ship_sizes: return None
        occupancy_grid = np.zeros((self.board_size, self.board_size), dtype=int)
        sunk_cells_set = set(coord for si in self.sunk_ships_info for coord in si['coords'])
        misses_set = set(tuple(c) for c in np.argwhere(self.result_grid == -1))
        hits_set = set(self.hits)
        successful_placements = 0
        num_samples = MC_SAMPLES_BASE * (3 if self._squares_remaining_on_board() <= 4 else 1)

        for _ in range(num_samples):
            current_board_ship_cells = set()
            temp_remaining_ships = list(self.remaining_ship_sizes)
            random.shuffle(temp_remaining_ships)
            possible_config = True
            for ship_len in temp_remaining_ships:
                placed_this_ship = False
                for _attempt in range(20):
                    is_horiz = random.choice([True, False])
                    r_coord = random.randrange(self.board_size) if is_horiz else random.randrange(
                        self.board_size - ship_len + 1)
                    c_coord = random.randrange(self.board_size - ship_len + 1) if is_horiz else random.randrange(
                        self.board_size)
                    potential_coords = set()
                    if is_horiz:
                        potential_coords = {(r_coord, c_coord + i) for i in range(ship_len)}
                    else:
                        potential_coords = {(r_coord + i, c_coord) for i in range(ship_len)}
                    if not (potential_coords & misses_set) and \
                            not (potential_coords & sunk_cells_set) and \
                            not (potential_coords & current_board_ship_cells):
                        current_board_ship_cells.update(potential_coords)
                        placed_this_ship = True;
                        break
                if not placed_this_ship: possible_config = False; break
            if possible_config and hits_set.issubset(current_board_ship_cells):
                successful_placements += 1
                for r_mc, c_mc in current_board_ship_cells:  # Renamed r_coord, c_coord
                    if (r_mc, c_mc) in self.available_moves:
                        occupancy_grid[r_mc, c_mc] += 1

        if successful_placements == 0 or occupancy_grid.max() == 0: return None
        return occupancy_grid.astype(float) / occupancy_grid.max()

    def _select_target_mode_move_enhanced(self) -> Optional[Tuple[int, int]]:
        if not self.hits: self.mode = "hunt"; return None
        potential_moves: List[Tuple[int, int]] = []
        self._determine_target_axis()

        if self.target_axis_hits:
            current_axis_hits = sorted(self.target_axis_hits, key=lambda t: (t[0], t[1]))
            min_h, max_h = current_axis_hits[0], current_axis_hits[-1]
            is_horiz = len(current_axis_hits) > 1 and min_h[0] == max_h[0]
            is_vert = len(current_axis_hits) > 1 and min_h[1] == max_h[1]

            if is_horiz:
                if not self.target_blocked_ends["front"]: potential_moves.append((min_h[0], min_h[1] - 1))
                if not self.target_blocked_ends["back"]: potential_moves.append((max_h[0], max_h[1] + 1))
            elif is_vert:
                if not self.target_blocked_ends["front"]: potential_moves.append((min_h[0] - 1, min_h[1]))
                if not self.target_blocked_ends["back"]: potential_moves.append((max_h[0] + 1, max_h[1]))
            else:
                r_single, c_single = current_axis_hits[0]  # Renamed r,c
                for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    potential_moves.append((r_single + dr, c_single + dc))
        else:
            temp_potential_moves = set()
            for r_h, c_h in self.hits:
                for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    temp_potential_moves.add((r_h + dr, c_h + dc))
            potential_moves.extend(list(temp_potential_moves))

        valid_moves = [m for m in potential_moves if
                       0 <= m[0] < self.board_size and 0 <= m[1] < self.board_size and m in self.available_moves]

        if not valid_moves:
            self.mode = "hunt";
            self.target_axis_hits = [];
            self.target_blocked_ends = {"front": False, "back": False}
            return None
        return self._prioritize_target_moves(valid_moves)

    def _prioritize_target_moves(self, moves: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        if not moves: return None
        if self.nn_model:
            heatmap = self._get_neural_heatmap()
            if heatmap is not None:
                scored_moves = sorted(moves, key=lambda m: heatmap[m[0], m[1]], reverse=True)
                if scored_moves: return scored_moves[0]  # Ensure list is not empty
        return random.choice(moves) if moves else None

    def _select_final_square_overall(self) -> Optional[Tuple[int, int]]:
        if self._squares_remaining_on_board() != 1: return None
        if not self.available_moves: return None

        if self.hits:
            candidate_final_squares = set()
            for r_hit, c_hit in self.hits:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    neighbor = (r_hit + dr, c_hit + dc)
                    if neighbor in self.available_moves:
                        candidate_final_squares.add(neighbor)

            if candidate_final_squares:
                sorted_hits = sorted(self.hits, key=lambda x: (x[0], x[1]))
                if len(sorted_hits) >= 1:
                    min_h, max_h = sorted_hits[0], sorted_hits[-1]
                    is_horiz = len(sorted_hits) > 1 and min_h[0] == max_h[0]
                    is_vert = len(sorted_hits) > 1 and min_h[1] == max_h[1]
                    line_extension_candidates = []
                    if is_horiz:
                        if (min_h[0], min_h[1] - 1) in candidate_final_squares: line_extension_candidates.append(
                            (min_h[0], min_h[1] - 1))
                        if (max_h[0], max_h[1] + 1) in candidate_final_squares: line_extension_candidates.append(
                            (max_h[0], max_h[1] + 1))
                    elif is_vert:
                        if (min_h[0] - 1, min_h[1]) in candidate_final_squares: line_extension_candidates.append(
                            (min_h[0] - 1, min_h[1]))
                        if (max_h[0] + 1, max_h[1]) in candidate_final_squares: line_extension_candidates.append(
                            (max_h[0] + 1, max_h[1]))
                    if line_extension_candidates: return random.choice(line_extension_candidates)
                return random.choice(list(candidate_final_squares))
            else:  # Fallback if hits have no available neighbors (should be rare if logic is correct)
                return random.choice(list(self.available_moves)) if self.available_moves else None
        return random.choice(list(self.available_moves)) if self.available_moves else None

    def _target_final_ship_exhaustive(self) -> Optional[Tuple[int, int]]:
        if len(self.remaining_ship_sizes) != 1: return None
        ship_len = self.remaining_ship_sizes[0]
        current_hits = list(self.hits)
        if not current_hits and self.min_remaining_ship_size > 0:
            return None
        elif not current_hits and self.min_remaining_ship_size == 0:
            return None

        miss_squares = set(tuple(c) for c in np.argwhere(self.result_grid == -1))
        sunk_ship_coords_set = set(coord for ship_info in self.sunk_ships_info for coord in ship_info['coords'])
        valid_placements_coords = []

        for r_start in range(self.board_size):
            for c_start in range(self.board_size):
                for dr_orient, dc_orient in [(0, 1), (1, 0)]:
                    current_placement_cells = set()
                    possible = True
                    for i in range(ship_len):
                        r_cell, c_cell = r_start + i * dr_orient, c_start + i * dc_orient
                        if not (0 <= r_cell < self.board_size and 0 <= c_cell < self.board_size):
                            possible = False;
                            break
                        current_placement_cells.add((r_cell, c_cell))
                    if not possible: continue
                    if not all(h_coord in current_placement_cells for h_coord in current_hits): continue
                    if current_placement_cells & miss_squares: continue
                    if current_placement_cells & sunk_ship_coords_set: continue
                    valid_placements_coords.append(current_placement_cells)

        if not valid_placements_coords: return None
        freq_map: Dict[Tuple[int, int], int] = {}
        for placement in valid_placements_coords:
            for cell in placement:
                if cell in self.available_moves:
                    freq_map[cell] = freq_map.get(cell, 0) + 1

        if not freq_map:
            return self._select_target_mode_move_enhanced() if self.hits else None

        sorted_options = []
        if self.nn_model:
            heatmap = self._get_neural_heatmap()
            if heatmap is not None:
                sorted_options = sorted(freq_map.items(), key=lambda item: (item[1], heatmap[item[0][0], item[0][1]]),
                                        reverse=True)
            else:
                sorted_options = sorted(freq_map.items(), key=lambda item: item[1], reverse=True)
        else:
            sorted_options = sorted(freq_map.items(), key=lambda item: item[1], reverse=True)

        if sorted_options: return sorted_options[0][0]
        return None

    def _exhaustive_endgame_search_from_original(self) -> Optional[Tuple[int, int]]:
        if not self.remaining_ship_sizes: return None
        miss_coords = set(tuple(c) for c in np.argwhere(self.result_grid == -1))
        hit_coords_must_cover = set(self.hits)
        sunk_coords_occupied = set(coord for ship_info in self.sunk_ships_info for coord in ship_info['coords'])
        candidate_square_freq: Dict[Tuple[int, int], int] = {}
        ships_to_place = sorted(list(self.remaining_ship_sizes), reverse=True)
        memo_recurse = {}

        def recurse_placements(ship_idx: int, currently_occupied_by_new_placements: Set[Tuple[int, int]]):
            state_key = (ship_idx, tuple(sorted(list(currently_occupied_by_new_placements))))
            if state_key in memo_recurse: return memo_recurse[state_key]
            if ship_idx == len(ships_to_place):
                if hit_coords_must_cover.issubset(
                        currently_occupied_by_new_placements | sunk_coords_occupied):  # Check sunk_coords_occupied too
                    for cell_coord in currently_occupied_by_new_placements:
                        if cell_coord in self.available_moves:
                            candidate_square_freq[cell_coord] = candidate_square_freq.get(cell_coord, 0) + 1
                    return True
                return False

            current_ship_len = ships_to_place[ship_idx]
            found_any_placement_for_subtree = False
            for r_start in range(self.board_size):
                for c_start in range(self.board_size):
                    for dr_orient, dc_orient in [(0, 1), (1, 0)]:
                        current_ship_potential_coords = set()
                        possible_to_place_here = True
                        for i in range(current_ship_len):
                            r_cell, c_cell = r_start + i * dr_orient, c_start + i * dc_orient
                            if not (0 <= r_cell < self.board_size and 0 <= c_cell < self.board_size):
                                possible_to_place_here = False;
                                break
                            if (r_cell, c_cell) in miss_coords or \
                                    (r_cell, c_cell) in sunk_coords_occupied or \
                                    (r_cell, c_cell) in currently_occupied_by_new_placements:
                                possible_to_place_here = False;
                                break
                            current_ship_potential_coords.add((r_cell, c_cell))
                        if possible_to_place_here:
                            if recurse_placements(ship_idx + 1,
                                                  currently_occupied_by_new_placements | current_ship_potential_coords):
                                found_any_placement_for_subtree = True
            memo_recurse[state_key] = found_any_placement_for_subtree
            return found_any_placement_for_subtree

        recurse_placements(0, set())
        if not candidate_square_freq: return None

        best_options = []
        if self.nn_model:
            heatmap = self._get_neural_heatmap()
            if heatmap is not None:
                best_options = sorted(candidate_square_freq.items(),
                                      key=lambda item: (item[1], heatmap[item[0][0], item[0][1]]), reverse=True)
            else:
                best_options = sorted(candidate_square_freq.items(), key=lambda item: item[1], reverse=True)
        else:
            best_options = sorted(candidate_square_freq.items(), key=lambda item: item[1], reverse=True)

        if best_options: return best_options[0][0]
        return None

    def view_display(self) -> str:
        mapping = {0: ".", -1: "O", 1: "X"}
        board_str = f"[{self.name}] View (Turn: {self.turn_count}):\n  " + " ".join(
            map(str, range(self.board_size))) + "\n"
        for r in range(self.board_size):
            row_str = f"{r} "
            for c in range(self.board_size):
                is_sunk_part = any((r, c) in ship_info['coords'] for ship_info in self.sunk_ships_info)
                if is_sunk_part:
                    row_str += "S "
                else:
                    row_str += mapping[self.result_grid[r, c]] + " "
            board_str += row_str.strip() + "\n"
        board_str += f"Mode: {self.mode}, Hits: {self.hits}, Axis: {self.target_axis_hits}, Blocked: {self.target_blocked_ends}\n"
        board_str += f"Rem Ships: {self.remaining_ship_sizes} (Min: {self.min_remaining_ship_size}, Max: {self.max_remaining_ship_size})\n"
        board_str += f"Hit Count: {self.hit_count}, Squares Rem: {self._squares_remaining_on_board()}, Avail Moves: {len(self.available_moves)}"
        return board_str.strip()