"""Board representation for Battleship web game."""

import numpy as np
import random
from .ship import Ship

SHIP_SIZES = [5, 4, 3, 3, 2]

class Board:
    def __init__(self, size=10):
        self.size = size
        self.grid = np.zeros((size, size), dtype=int)
        self.ships = []
        self.ship_lookup = {}  # Maps coordinates to ship objects
        
    def place_ships(self, sizes=SHIP_SIZES):
        """Randomly places ships on the board."""
        for ship_size in sizes:
            placed = False
            attempts = 0
            while not placed and attempts < 1000:  # Prevent infinite loops
                attempts += 1
                direction = random.choice(["H", "V"])
                row, col = random.randint(0, 9), random.randint(0, 9)
                
                if direction == "H" and col + ship_size <= 10:
                    coords = [(row, col + i) for i in range(ship_size)]
                elif direction == "V" and row + ship_size <= 10:
                    coords = [(row + i, col) for i in range(ship_size)]
                else:
                    continue
                
                # Check for overlapping ships
                if any(c in self.ship_lookup for c in coords):
                    continue
                
                # Place the ship
                ship = Ship(ship_size, coords, direction)
                self.ships.append(ship)
                for c in coords:
                    self.ship_lookup[c] = ship
                placed = True
    
    def place_ships_manual(self, ship_placements):
        """Place ships based on manual placement data."""
        self.ships = []
        self.ship_lookup = {}
        
        for placement in ship_placements:
            coords = placement['coordinates']
            direction = placement['orientation']
            size = placement['size']
            
            # Validate placement
            if not self._is_valid_placement(coords):
                raise ValueError(f"Invalid ship placement: {coords}")
            
            ship = Ship(size, coords, direction)
            self.ships.append(ship)
            for c in coords:
                self.ship_lookup[tuple(c)] = ship
    
    def _is_valid_placement(self, coords):
        """Check if ship placement is valid."""
        for coord in coords:
            row, col = coord
            if row < 0 or row >= 10 or col < 0 or col >= 10:
                return False
            if tuple(coord) in self.ship_lookup:
                return False
        return True
    
    def attack(self, coord):
        """Handle an attack on the board."""
        row, col = coord
        
        # Check bounds
        if row < 0 or row >= 10 or col < 0 or col >= 10:
            return "invalid"
        
        # Check if already attacked
        if self.grid[row, col] != 0:
            return "already"
        
        # Check for hit
        coord_tuple = (row, col)
        if coord_tuple in self.ship_lookup:
            self.grid[row, col] = 2  # Hit
            ship = self.ship_lookup[coord_tuple]
            ship.check_hit(coord_tuple)
            return "sunk" if ship.is_sunk() else "hit"
        else:
            self.grid[row, col] = 1  # Miss
            return "miss"
    
    def all_ships_sunk(self):
        """Check if all ships are sunk."""
        return bool(self.ships) and all(ship.is_sunk() for ship in self.ships)
    
    def get_grid_display(self, reveal=False):
        """Get board display as 2D array for web frontend."""
        display_grid = []
        for r in range(10):
            row = []
            for c in range(10):
                if reveal and (r, c) in self.ship_lookup:
                    row.append("ship")
                elif self.grid[r, c] == 0:
                    row.append("water")
                elif self.grid[r, c] == 1:
                    row.append("miss")
                elif self.grid[r, c] == 2:
                    row.append("hit")
                else:
                    row.append("water")
            display_grid.append(row)
        return display_grid
    
    def to_dict(self, reveal=False):
        """Convert board to dictionary for JSON serialization."""
        return {
            'grid': self.get_grid_display(reveal=reveal),
            'ships': [ship.to_dict() for ship in self.ships] if reveal else [],
            'all_ships_sunk': self.all_ships_sunk()
        }
