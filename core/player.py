import sys
from pathlib import Path

# Add the parent directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.board import Board


class Player:
    def __init__(self, name):
        self.name = name
        self.board = Board()
    # --- helpers for dataset / RL scripts -----------------
    def turn_board_state(self):
        """Return the public result grid the AI keeps (hits/misses/unknown)."""
        # AIPlayer has `result_grid`; if not present, fall back to raw board grid.
        return getattr(self, "result_grid", self.board.grid)

    def true_ship_grid(self):
        """Binary 10×10 grid of this player’s actual ship locations (for labels)."""
        import numpy as np
        g = np.zeros((10, 10), dtype=np.float32)
        if hasattr(self.board, "ship_lookup"):
            for (r, c) in self.board.ship_lookup.keys():
                g[r, c] = 1.0
        return g


class HumanPlayer(Player):
    def __init__(self, name, manual_setup='n'):
        super().__init__(name)
        if manual_setup.lower() == 'y':
            print(f"{self.name} is manually placing ships.")
            self.board.place_ships_manual()
        else:
            print(f"{self.name}'s ships are being placed automatically.")
            self.board.place_ships()

    def take_turn(self, opponent_board):
        """Handles the player's attack turn."""
        while True:
            move = input(f"\n{self.name}, enter your move (row,col): ").strip()

            if "," not in move:
                print("Oops! Enter your move in 'row,col' format.")
                continue

            try:
                row, col = map(int, move.split(","))
            except ValueError:
                print("That didn’t look like numbers. Try again!")
                continue

            result = opponent_board.attack((row, col))

            if result == 'invalid':
                print("Out of bounds! Pick a valid spot.")
            elif result == 'already':
                print("You've already fired there! Try somewhere else.")
            else:
                print(f"{self.name} fires at ({row}, {col}) and it's a {result.upper()}!")
                return result
