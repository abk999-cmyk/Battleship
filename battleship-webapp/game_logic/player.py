"""Player classes for Battleship web game."""

from .board import Board
import random

class Player:
    def __init__(self, name, player_type="human"):
        self.name = name
        self.player_type = player_type
        self.board = Board()
        self.target_grid = [[0 for _ in range(10)] for _ in range(10)]  # Track attacks on opponent
        
    def place_ships_auto(self):
        """Automatically place ships on board."""
        self.board.place_ships()
        
    def place_ships_manual(self, ship_placements):
        """Place ships based on manual placement."""
        self.board.place_ships_manual(ship_placements)
        
    def attack(self, coord):
        """Make an attack (for AI players)."""
        # This will be overridden by AI players
        return random.choice([(r, c) for r in range(10) for c in range(10) 
                            if self.target_grid[r][c] == 0])
    
    def update_attack_result(self, coord, result):
        """Update the target grid with attack result."""
        row, col = coord
        if result == "hit":
            self.target_grid[row][col] = 2
        elif result == "miss":
            self.target_grid[row][col] = 1
        elif result == "sunk":
            self.target_grid[row][col] = 3
    
    def to_dict(self, reveal_ships=False):
        """Convert player to dictionary for JSON serialization."""
        return {
            'name': self.name,
            'player_type': self.player_type,
            'board': self.board.to_dict(reveal=reveal_ships),
            'target_grid': self.target_grid
        }

class HumanPlayer(Player):
    def __init__(self, name):
        super().__init__(name, "human")

class AIPlayer(Player):
    def __init__(self, name, difficulty="medium"):
        super().__init__(name, "ai")
        self.difficulty = difficulty
        self.last_hit = None
        self.hunting_direction = None
        self.hunting_coords = []
        
    def attack(self, opponent_board):
        """AI attack logic with different difficulty levels."""
        if self.difficulty == "easy":
            return self._random_attack()
        elif self.difficulty == "medium":
            return self._smart_attack()
        else:  # hard
            return self._advanced_attack()
    
    def _random_attack(self):
        """Random attack strategy."""
        available_coords = [(r, c) for r in range(10) for c in range(10) 
                          if self.target_grid[r][c] == 0]
        return random.choice(available_coords) if available_coords else None
    
    def _smart_attack(self):
        """Smart attack strategy with hunting mode."""
        # If we have a recent hit, try adjacent squares
        if self.last_hit and not self.hunting_coords:
            self._find_hunting_coords()
        
        if self.hunting_coords:
            return self.hunting_coords.pop(0)
        
        # Use probability-based targeting
        return self._probability_attack()
    
    def _advanced_attack(self):
        """Advanced attack strategy."""
        # More sophisticated AI logic can be added here
        return self._smart_attack()
    
    def _find_hunting_coords(self):
        """Find coordinates adjacent to last hit."""
        if not self.last_hit:
            return
            
        row, col = self.last_hit
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # right, left, down, up
        
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if (0 <= new_row < 10 and 0 <= new_col < 10 and 
                self.target_grid[new_row][new_col] == 0):
                self.hunting_coords.append((new_row, new_col))
    
    def _probability_attack(self):
        """Attack based on probability of ship placement."""
        # Simple probability: prefer center squares and checkerboard pattern
        available_coords = [(r, c) for r in range(10) for c in range(10) 
                          if self.target_grid[r][c] == 0]
        
        if not available_coords:
            return None
        
        # Prefer checkerboard pattern for initial hunting
        checkerboard_coords = [coord for coord in available_coords 
                             if (coord[0] + coord[1]) % 2 == 0]
        
        if checkerboard_coords:
            return random.choice(checkerboard_coords)
        else:
            return random.choice(available_coords)
    
    def update_attack_result(self, coord, result):
        """Update AI state based on attack result."""
        super().update_attack_result(coord, result)
        
        if result == "hit":
            self.last_hit = coord
            if not self.hunting_coords:
                self._find_hunting_coords()
        elif result == "sunk":
            # Reset hunting state when ship is sunk
            self.last_hit = None
            self.hunting_coords = []
            self.hunting_direction = None
        elif result == "miss" and self.hunting_coords:
            # Continue hunting if we're in hunting mode
            pass
