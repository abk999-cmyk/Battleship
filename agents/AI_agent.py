import random
from typing import List, Set, Tuple, Optional
import sys
from pathlib import Path

import numpy as np

# Add the parent directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.player import Player

# ----------------------------------------------------------------------
# Global constants
# ----------------------------------------------------------------------
BOARD_SIZE: int = 10
TOTAL_SHIP_SQUARES: int = 17
MC_SAMPLES: int = 1500      # default Monte‑Carlo samples per turn

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
except ImportError:
    tf = None

SHIP_SIZES = [5, 4, 3, 3, 2]


class AIPlayer(Player):
    def __init__(self, name):
        super().__init__(name)
        self.board_size = BOARD_SIZE
        self.board.place_ships()
        self.available_moves = {(r, c) for r in range(self.board_size) for c in range(self.board_size)}
        self.result_grid = np.zeros((self.board_size, self.board_size))
        self.hits = []
        # Remember coordinates of cells that belong to ships we've already sunk
        self.sunk_cells = set()
        # Track total successful hits so we know how many ship squares remain
        self.hit_count = 0      # 17 total ship squares in classic setup
        # Track whether either end of a located ship has proven impossible
        # (miss or off‑board) so we stop probing that direction.
        self.blocked = {"front": False, "back": False}
        self.mode = "hunt"
        self.remaining_ship_sizes = list(SHIP_SIZES)
        self.min_ship_size = min(self.remaining_ship_sizes)

        # Neural model
        self.nn_model = None
        if tf:
            try:
                self.nn_model = load_model("models/battleship_heatmap.h5", compile=False)
                print("[DBTBA] - loaded neural heat-map model.")
            except Exception:
                print("[DBTBA] - no neural model found.")

    def take_turn(self, opponent_board):
        """
        Perform one full turn: choose a square, fire, update internal state.
        RETURNS
        -------
        move (row, col)   – if a shot was fired
        None              – if the agent had no moves left
        """
        move = self.select_move(opponent_board)

        # If no legal moves remain, notify the caller (game engine).
        if move is None:
            return None

        self.last_move = move
        self.available_moves.discard(move)

        # Execute the attack.
        result = opponent_board.attack(move)
        self.last_result = result

        # Update knowledge grid, ship tracking, etc.
        self.update_result_grid(move, result)

        if result == "sunk":
            ship = getattr(opponent_board, "ship_lookup", {}).get(move)
            if ship and ship.size in self.remaining_ship_sizes:
                self.remaining_ship_sizes.remove(ship.size)
                self.min_ship_size = min(self.remaining_ship_sizes) if self.remaining_ship_sizes else 1

        return move  # game engine only needs to know *something* was played

    def select_move(self, opponent_board=None):
        """
        Decide on the next coordinate to attack.
        """
        # Special case: only one ship square left anywhere on the board.
        if self._squares_remaining() == 1:
            m = self._select_final_square()
            if m is not None:
                return m

        # Special case: only the final ship remains.
        if len(self.remaining_ship_sizes) == 1:
            if self.hits:  # we have already located part of it
                m = self._select_final_ship_move()
                if m is not None:
                    return m
            else:          # no hits on the last ship yet
                m = self._single_ship_hunt_move()
                if m is not None:
                    return m

        # Special case: very few ship squares remain anywhere (≤ 4).
        if self._squares_remaining() <= 4:
            m = self._exhaustive_endgame_move()
            if m is not None:
                return m

        # --- TARGET MODE ---
        if self.mode == "target" and self.hits:
            m = self.select_targeting_move()
            if m:
                return m
            # fall back to hunt mode if no good targeting moves
            self.mode = "hunt"
            self.hits.clear()
            self.blocked = {"front": False, "back": False}

        # --- HUNT MODE ---
        prob = self.compute_probability_grid()
        if prob.max() == 0:
            return random.choice(tuple(self.available_moves))

        best = np.argwhere(prob == prob.max())
        return random.choice([tuple(p) for p in best])
    def _single_ship_hunt_move(self):
        """
        When exactly one ship is afloat and we have *no* hits on it,
        enumerate every legal placement of that ship and fire at the square
        that appears in the most placements (maximum information gain).
        """
        if self.hits or len(self.remaining_ship_sizes) != 1:
            return None

        size = self.board_size
        ship_len = self.remaining_ship_sizes[0]
        miss = {(r, c) for r in range(size) for c in range(size)
                if self.result_grid[r, c] == -1}
        occupied_by_sunk = self.sunk_cells

        freq = {}
        for r in range(size):
            for c in range(size):
                for dr, dc in [(0, 1), (1, 0)]:
                    end_r = r + dr * (ship_len - 1)
                    end_c = c + dc * (ship_len - 1)
                    if not (0 <= end_r < size and 0 <= end_c < size):
                        continue
                    cells = {(r + k * dr, c + k * dc) for k in range(ship_len)}
                    if cells & miss or cells & occupied_by_sunk:
                        continue
                    for cell in cells:
                        if cell in self.available_moves:
                            freq[cell] = freq.get(cell, 0) + 1

        if not freq:
            return None
        max_freq = max(freq.values())
        best = [cell for cell, cnt in freq.items() if cnt == max_freq]
        return random.choice(best)

    def _select_final_square(self):
        """
        When exactly one ship square is left, infer where it must be.
        We assume self.hits contains the partial ship we are chasing.
        """
        if not self.hits:
            # No partial ship info – fall back to highest prob
            return next(iter(self.available_moves))

        hits = sorted(self.hits)
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        # Remaining ship size = size of last unsunk ship minus known hits
        remaining_ship_size = None
        if self.remaining_ship_sizes:
            # Only one unsunk ship should remain
            remaining_ship_size = self.remaining_ship_sizes[0] - len(hits)

        # If orientation known, probe only the two ends (max 2 options)
        if len(hits) >= 2 and (hits[0][0] == hits[1][0] or hits[0][1] == hits[1][1]):
            horizontal = hits[0][0] == hits[1][0]
            if horizontal:
                row = hits[0][0]
                cols = [c for _, c in hits]
                opts = [(row, min(cols) - 1), (row, max(cols) + 1)]
            else:
                col = hits[0][1]
                rows = [r for r, _ in hits]
                opts = [(min(rows) - 1, col), (max(rows) + 1, col)]
            opts = [p for p in opts if p in self.available_moves]
            if opts:
                return random.choice(opts)

        # Otherwise, probe each neighbour of the hit(s) that still fits
        frontier = set()
        for r, c in hits:
            for dr, dc in dirs:
                p = (r + dr, c + dc)
                if p in self.available_moves:
                    frontier.add(p)

        # limit to 4 possibilities max as per requirement
        picks = list(frontier)[:4]
        if picks:
            return random.choice(picks)

        # Fallback – any available move
        return next(iter(self.available_moves))

    def select_targeting_move(self):
        """
        With confirmed hits, extend intelligently to finish the ship.
        """
        if not self.hits:
            return None

        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        hits = sorted(self.hits)

        # Two aligned hits → orientation known
        if len(hits) >= 2 and (hits[0][0] == hits[1][0] or hits[0][1] == hits[1][1]):
            if hits[0][0] == hits[1][0]:  # horizontal
                row = hits[0][0]
                cols = [c for _, c in hits]
                cand = []
                if not self.blocked["front"]:
                    cand.append((row, min(cols) - 1))
                if not self.blocked["back"]:
                    cand.append((row, max(cols) + 1))
                candidates = cand
            else:  # vertical
                col = hits[0][1]
                rows = [r for r, _ in hits]
                cand = []
                if not self.blocked["front"]:
                    cand.append((min(rows) - 1, col))
                if not self.blocked["back"]:
                    cand.append((max(rows) + 1, col))
                candidates = cand
        else:
            # Single hit – probe all neighbours
            r, c = hits[0]
            candidates = [(r + dr, c + dc) for dr, dc in dirs]

        # Filter to available moves
        candidates = [m for m in candidates if m in self.available_moves]
        if candidates:
            return random.choice(candidates)

        # Fallback: any neighbour of any hit
        frontier = []
        for r, c in hits:
            for dr, dc in dirs:
                m = (r + dr, c + dc)
                if m in self.available_moves:
                    frontier.append(m)
        return random.choice(frontier) if frontier else None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _squares_remaining(self) -> int:
        """Return how many ship squares remain un‑hit."""
        return TOTAL_SHIP_SQUARES - self.hit_count

    def _mark_blocked(self, miss_move):
        """
        When in target mode with known orientation, record that the side
        we just probed is blocked so we do not waste another move there.
        """
        if len(self.hits) < 2:
            return  # orientation not yet fixed

        # Determine orientation
        horizontal = self.hits[0][0] == self.hits[1][0]
        # Sort hits along orientation to find 'front' and 'back'
        hits_sorted = sorted(self.hits, key=lambda t: t[1] if horizontal else t[0])
        front_hit = hits_sorted[0]
        back_hit = hits_sorted[-1]

        # Build the coordinate just outside each end
        if horizontal:
            front_coord = (front_hit[0], front_hit[1] - 1)
            back_coord = (back_hit[0], back_hit[1] + 1)
        else:
            front_coord = (front_hit[0] - 1, front_hit[1])
            back_coord = (back_hit[0] + 1, back_hit[1])

        if miss_move == front_coord:
            self.blocked["front"] = True
        elif miss_move == back_coord:
            self.blocked["back"] = True


    def update_result_grid(self, move, result):
        r, c = move
        if result in ("hit", "sunk"):
            self.result_grid[r, c] = 1
            self.hit_count += 1
            self.hits.append((r, c))
            if result == "sunk":
                # Persist this ship’s coordinates so we can reason about them later
                self.sunk_cells.update(self.hits)
                self.hits.clear()
                self.mode = "hunt"
                self.blocked = {"front": False, "back": False}
            else:
                self.mode = "target"
        elif result == "miss":
            self.result_grid[r, c] = -1
            if self.mode == "target":
                self._mark_blocked(move)

    def compute_probability_grid(self):
        """
        Generate a probability heat‑map for each untried square.
        """
        size = self.board_size
        avail = np.zeros((size, size), dtype=bool)
        for r, c in self.available_moves:
            avail[r, c] = True

        grid = np.zeros((size, size), dtype=np.float32)
        # Cache grid.max calls for efficiency
        grid_max: float

        # Classical placement counting, weighted by ship size
        for ship in self.remaining_ship_sizes:
            weight = float(ship)
            # horizontal
            for r in range(size):
                for c in range(size - ship + 1):
                    if avail[r, c:c + ship].all():
                        grid[r, c:c + ship] += weight
            # vertical
            for c in range(size):
                for r in range(size - ship + 1):
                    if avail[r:r + ship, c].all():
                        grid[r:r + ship, c] += weight

        # Parity optimisation — but never mask squares that are direct
        # orthogonal neighbours of any unsunk hit; those are too valuable.
        if self.mode == "hunt" and self.min_ship_size >= 2:
            parity_mask = (np.add.outer(np.arange(size), np.arange(size)) % 2 == 0)

            # Build neighbour mask for all unsunk hits
            neighbour_mask = np.zeros_like(parity_mask, dtype=bool)
            dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            for r, c in [(r, c) for r in range(size) for c in range(size)
                         if self.result_grid[r, c] == 1 and (r, c) not in self.sunk_cells]:
                for dr, dc in dirs:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < size and 0 <= nc < size:
                        neighbour_mask[nr, nc] = True

            keep_mask = parity_mask | neighbour_mask
            grid *= keep_mask

        # Neural heat‑map blend
        if self.nn_model is not None:
            try:
                heat = self.neural_heatmap()
                if heat is not None and heat.max() > 0:
                    ##################################################################
                    alpha = 0.6 if self.mode == "hunt" else 0.0 # adjust RL and prob here 0=prob, 1=rl
                    ##################################################################
                    grid_max = grid.max() or 1.0
                    grid = (1 - alpha) * grid + alpha * heat * grid_max
            except Exception:
                pass

        # Monte‑Carlo posterior blend (only if we have hits)
        mc = self._monte_carlo_grid()
        if mc is not None:
            grid_max = grid.max()
            grid = 0.5 * grid + 0.5 * mc * (grid_max or 1.0)

        # Strong local bonus: any unsunk hit gets its 4-neighbours boosted
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        grid_max = grid.max()
        for r, c in [(r, c) for r in range(size) for c in range(size)
                     if self.result_grid[r, c] == 1 and (r, c) not in self.sunk_cells]:
            for dr, dc in dirs:
                nr, nc = r + dr, c + dc
                if 0 <= nr < size and 0 <= nc < size and (nr, nc) in self.available_moves:
                    grid[nr, nc] += grid_max * 2  # strong encouragement

        # Zero out already‑played squares
        for r, c in set([(r, c) for r in range(size) for c in range(size)]) - self.available_moves:
            grid[r, c] = 0.0

        # Normalize grid before return
        grid_max = grid.max()
        if grid_max > 0:
            grid = grid / grid_max

        return grid

    def _monte_carlo_grid(self):
        """
        Sample MC_SAMPLES full board placements consistent with all evidence
        (hits, misses, sunk ships). Return a normalized heat‑map.
        """
        size = self.board_size
        occupied_count = np.zeros((size, size), dtype=np.int32)

        # Pre‑compute constraints
        miss = {(r, c) for r in range(size) for c in range(size)
                if self.result_grid[r, c] == -1}
        hit  = {(r, c) for r in range(size) for c in range(size)
                if self.result_grid[r, c] == 1}

        # If no hits yet, fall back to classical grid
        if not hit:
            return None

        rng = random.Random()
        samples = MC_SAMPLES * (3 if self._squares_remaining() <= 4 else 1)
        for _ in range(samples):
            placement_cells = set()
            ok = True
            for ship_len in SHIP_SIZES:
                # Skip ships already sunk
                if ship_len not in self.remaining_ship_sizes:
                    continue
                # Try up to 20 random placements for this ship
                for _attempt in range(20):
                    horiz = rng.choice([True, False])
                    if horiz:
                        r = rng.randrange(size)
                        c = rng.randrange(size - ship_len + 1)
                        cells = {(r, c + i) for i in range(ship_len)}
                    else:
                        r = rng.randrange(size - ship_len + 1)
                        c = rng.randrange(size)
                        cells = {(r + i, c) for i in range(ship_len)}

                    # constraints
                    if cells & placement_cells:
                        continue
                    if cells & miss:
                        continue
                    # Ensure we can still cover all hits eventually
                    if not hit.issubset(cells | placement_cells):
                        continue
                    placement_cells |= cells
                    break
                else:
                    ok = False
                    break

            if not ok or not hit.issubset(placement_cells):
                continue

            # Count occupancy
            for r, c in placement_cells:
                if (r, c) in self.available_moves:
                    occupied_count[r, c] += 1

        if occupied_count.sum() == 0:
            return None

        return occupied_count / occupied_count.max()
    def _exhaustive_endgame_move(self):
        """
        Enumerate *all* legal placements for each remaining ship that are
        still compatible with every hit/miss on the board.  Build a
        frequency map of candidate squares and return the most common
        (ties broken randomly).  Runs only when ≤4 unknown squares remain,
        so brute force is cheap and exact.
        """
        size = self.board_size
        miss = {(r, c) for r in range(size) for c in range(size)
                if self.result_grid[r, c] == -1}
        hit  = {(r, c) for r in range(size) for c in range(size)
                if self.result_grid[r, c] == 1}
        candidates = {}

        # Generate every set of placements for remaining ships recursively
        ships_left = list(self.remaining_ship_sizes)

        def recurse(idx, occupied):
            if idx == len(ships_left):
                # Valid full placement – tally unknown squares
                for cell in occupied:
                    if cell in self.available_moves:
                        candidates[cell] = candidates.get(cell, 0) + 1
                return
            ship_len = ships_left[idx]
            for r in range(size):
                for c in range(size):
                    for dr, dc in [(0, 1), (1, 0)]:
                        end_r = r + dr * (ship_len - 1)
                        end_c = c + dc * (ship_len - 1)
                        if not (0 <= end_r < size and 0 <= end_c < size):
                            continue
                        cells = {(r + k * dr, c + k * dc) for k in range(ship_len)}
                        if cells & miss or cells & occupied:
                            continue
                        if not hit.issubset(cells | occupied):
                            continue
                        recurse(idx + 1, occupied | cells)

        recurse(0, set())

        if not candidates:
            return None
        max_freq = max(candidates.values())
        best = [cell for cell, f in candidates.items() if f == max_freq]
        return random.choice(best)

    def neural_heatmap(self):
        miss_plane = (self.result_grid == -1).astype(np.float32)
        hit_plane = (self.result_grid == 1).astype(np.float32)
        unk_plane = (self.result_grid == 0).astype(np.float32)

        x = np.stack([miss_plane, hit_plane, unk_plane], axis=-1)[np.newaxis, ...]
        preds = self.nn_model.predict(x, verbose=0)[0]

        if preds.shape != (self.board_size, self.board_size):
            return None

        # Mask invalid moves
        masked_preds = np.zeros_like(preds)
        for r in range(self.board_size):
            for c in range(self.board_size):
                if (r, c) in self.available_moves:
                    masked_preds[r, c] = preds[r, c]

        # Normalize heatmap
        if np.max(masked_preds) > 0:
            masked_preds = masked_preds / np.max(masked_preds)

        return masked_preds

    def view_display(self):
        mapping = {0: ".", -1: "O", 1: "X"}
        return "\n".join(" ".join(mapping[int(self.result_grid[r, c])] for c in range(self.board_size)) for r in
                         range(self.board_size))

    def _select_final_ship_move(self):
        """
        Robust logic for the last remaining ship:
        • Enumerate every legal placement of the remaining ship size that
          covers all current hits and avoids recorded misses.
        • Collect all still‑unknown squares belonging to those placements.
        • Pick randomly from the ≤4 most frequent candidate squares.
        """
        if len(self.remaining_ship_sizes) != 1:
            return None  # Not the last ship

        ship_len = self.remaining_ship_sizes[0]
        hits = list(self.hits)
        if not hits:
            return None  # No information yet – fall back to normal logic

        size = self.board_size
        # Convenience set for miss squares
        miss_squares = {(r, c) for r in range(size) for c in range(size) if self.result_grid[r, c] == -1}

        valid_placements = []

        def placement_cells(start_r, start_c, dr, dc):
            return [(start_r + i*dr, start_c + i*dc) for i in range(ship_len)]

        # Try every horizontal & vertical placement
        for r in range(size):
            for c in range(size):
                for dr, dc in [(0, 1), (1, 0)]:  # horizontal, vertical
                    cells = placement_cells(r, c, dr, dc)
                    # Ensure within bounds
                    if cells[-1][0] >= size or cells[-1][1] >= size:
                        continue
                    cell_set = set(cells)
                    # Must include all current hits
                    if not all(h in cell_set for h in hits):
                        continue
                    # Cannot overlap a recorded miss
                    if cell_set & miss_squares:
                        continue
                    valid_placements.append(cell_set)

        if not valid_placements:
            return None  # Should rarely happen

        # Count frequency of each yet‑unknown square across valid placements
        freq = {}
        for placement in valid_placements:
            for cell in placement:
                if cell in self.available_moves:  # skip known hits / already played
                    freq[cell] = freq.get(cell, 0) + 1

        if not freq:
            return None  # No candidate (all squares of ship already hit?)

        # Sort by descending frequency, keep up to 4
        top_cells = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)
        candidates = [cell for cell, _ in top_cells[:4]]

        return random.choice(candidates)