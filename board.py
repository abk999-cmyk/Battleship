import numpy as np
import random
from ship import Ship

# Toggle runtime chatter. Set to False for silent automated runs.
VERBOSE = False

SHIP_SIZES = [5, 4, 3, 3, 2]

class Board:
    def __init__(self, size=10):
        self.size = size
        self.grid = np.zeros((size, size), dtype=int)
        self.ships = []
        self.ship_lookup = {}  # Renamed from ship_map to feel more natural

    def place_ships(self, sizes=SHIP_SIZES):
        """Randomly places ships, but not perfectly optimized."""
        for ship_size in sizes:
            placed = False
            while not placed:
                direction = random.choice(["H", "V"])
                row, col = random.randint(0, 9), random.randint(0, 9)

                if direction == "H" and col + ship_size <= 10:
                    coords = [(row, col + i) for i in range(ship_size)]
                elif direction == "V" and row + ship_size <= 10:
                    coords = [(row + i, col) for i in range(ship_size)]
                else:
                    continue  # Pick new random coordinates

                if any(c in self.ship_lookup for c in coords):
                    continue  # Avoid overlapping

                self.ships.append(Ship(ship_size, coords, direction))
                for c in coords:
                    self.ship_lookup[c] = self.ships[-1]
                placed = True  # Exit loop once placed

    def place_ships_manual(self, sizes=SHIP_SIZES):
        """Allows manual ship placement, but not super strict."""
        if VERBOSE:
            print("\nAlright, time to place your ships!")

        for s in sizes:
            placed = False
            while not placed:
                if VERBOSE:
                    print(f"\nYou're placing a ship of size {s}.")
                direction = input("Enter H (horizontal) or V (vertical): ").strip().upper()

                if direction not in ["H", "V"]:
                    if VERBOSE:
                        print("C'mon, that's not an option. Try H or V.")
                    continue

                try:
                    row, col = map(int, input("Row, Col (0-9): ").split(","))
                except ValueError:
                    if VERBOSE:
                        print("Invalid. Try again!")
                    continue

                if not (0 <= row < 10 and 0 <= col < 10):
                    if VERBOSE:
                        print("Pick something within 0-9.")
                    continue

                if direction == "H" and col + s > 10:
                    if VERBOSE:
                        print("Uh-oh, no space to fit it horizontally.")
                    continue
                if direction == "V" and row + s > 10:
                    if VERBOSE:
                        print("Oops! No room to place it vertically.")
                    continue

                coords = [(row, col + i) for i in range(s)] if direction == "H" else [(row + i, col) for i in range(s)]

                if any(c in self.ship_lookup for c in coords):
                    if VERBOSE:
                        print("You've already placed something there. Pick another spot!")
                    continue

                self.ships.append(Ship(s, coords, direction))
                for c in coords:
                    self.ship_lookup[c] = self.ships[-1]
                if VERBOSE:
                    print("Nice! Ship placed.")
                placed = True

    def attack(self, coord):
        """Handles attacks. If it's invalid, too bad!"""
        # Ensure coord is a tuple of Python ints
        if isinstance(coord, (list, tuple)):
            row, col = int(coord[0]), int(coord[1])
        elif isinstance(coord, np.ndarray):
            flat = coord.ravel()
            if flat.size < 2:
                return "invalid"
            row, col = int(flat[0]), int(flat[1])
        else:  # single int or unexpected type
            idx = int(coord)
            row, col = divmod(idx, 10)

        if row not in range(10) or col not in range(10):
            if VERBOSE:
                print("That shot went out of bounds. Try again!")
            return "invalid"

        if self.grid[row, col] != 0:
            if VERBOSE:
                print("You've already hit this spot. Pick somewhere new.")
            return "already"

        if coord in self.ship_lookup:
            self.grid[row, col] = 2  # Hit
            ship = self.ship_lookup[coord]
            ship.check_hit(coord)
            if VERBOSE:
                print(f"Direct hit on a {ship.size}-unit ship!")
            return "sunk" if ship.is_sunk() else "hit"

        self.grid[row, col] = 1  # Miss
        if VERBOSE:
            print("Splash! That was a miss.")
        return "miss"

    def all_ships_sunk(self):
        """Returns True if all ships are destroyed."""
        return bool(self.ships) and all(ship.is_sunk() for ship in self.ships)

    def display(self, reveal=False):
        """Show the board, but keep it simple."""
        symbols = {0: ".", 1: "O", 2: "X"}
        board_rep = []
        for r in range(10):
            row_display = []
            for c in range(10):
                if reveal and (r, c) in self.ship_lookup:
                    row_display.append("S")  # Show ships if reveal is True
                else:
                    row_display.append(symbols[self.grid[r, c]])
            board_rep.append(" ".join(row_display))
        return "\n".join(board_rep)



'''
    def deep_copy(self):
        """
        Creates a deep copy of the board, including a new grid, new Ship objects,
        and a rebuilt ship_positions mapping.
        """
        new_board = Board(10)
        new_board.grid = self.grid.copy()
        new_board.ships = []
        for ship in self.ships:
            new_ship = Ship(ship.size, list(ship.coordinates), ship.orientation)
            new_ship.hits = set(ship.hits)
            new_board.ships.append(new_ship)
        new_board.ship_positions = {}
        for coord, ship in self.ship_head_map.items():
            # Find the corresponding new_ship (matching coordinates)
            for new_ship in new_board.ships:
                if new_ship.coordinates == ship.coordinates:
                    new_board.ship_positions[coord] = new_ship
                    break
        return new_board
'''