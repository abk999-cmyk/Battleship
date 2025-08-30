"""Game engine for Battleship web game."""

import uuid
from datetime import datetime

class BattleshipGame:
    def __init__(self, player1, player2):
        self.id = str(uuid.uuid4())
        self.player1 = player1
        self.player2 = player2
        self.current_player = player1
        self.other_player = player2
        self.move_count = 0
        self.turn_count = 0
        self.winner = None
        self.game_over = False
        self.created_at = datetime.now()
        self.last_move = None
        self.last_result = None
        
    def is_over(self):
        """Check if game is over."""
        return self.game_over or self.winner is not None
    
    def make_move(self, coord):
        """Process a move in the game."""
        if self.is_over():
            return {"error": "Game is already over"}
        
        row, col = coord
        
        # Attack the other player's board
        result = self.other_player.board.attack((row, col))
        
        if result in ["invalid", "already"]:
            return {"error": f"Invalid move: {result}"}
        
        # Update game state
        self.last_move = coord
        self.last_result = result
        self.move_count += 1
        
        # Update current player's knowledge
        self.current_player.update_attack_result(coord, result)
        
        # Check for win condition
        if self.other_player.board.all_ships_sunk():
            self.winner = self.current_player
            self.game_over = True
        else:
            # Switch turns
            self.current_player, self.other_player = self.other_player, self.current_player
            self.turn_count += 1
        
        return {
            "success": True,
            "result": result,
            "coord": coord,
            "game_over": self.is_over(),
            "winner": self.winner.name if self.winner else None,
            "current_player": self.current_player.name,
            "move_count": self.move_count,
            "turn_count": self.turn_count
        }
    
    def make_ai_move(self):
        """Make a move for AI player."""
        if self.is_over() or self.current_player.player_type != "ai":
            return None
        
        # Get AI's chosen attack coordinates
        coord = self.current_player.attack(self.other_player.board)
        
        if coord is None:
            # No valid moves left
            self.game_over = True
            self.winner = self.other_player
            return {"error": "No valid moves left"}
        
        return self.make_move(coord)
    
    def get_player_view(self, player_name):
        """Get game state from a specific player's perspective."""
        if player_name == self.player1.name:
            viewing_player = self.player1
            opponent = self.player2
        elif player_name == self.player2.name:
            viewing_player = self.player2
            opponent = self.player1
        else:
            return None
        
        return {
            "game_id": self.id,
            "player": viewing_player.to_dict(reveal_ships=True),
            "opponent": {
                "name": opponent.name,
                "player_type": opponent.player_type,
                "board": opponent.board.to_dict(reveal=False),  # Don't reveal opponent ships
                "target_grid": viewing_player.target_grid
            },
            "current_player": self.current_player.name,
            "game_over": self.is_over(),
            "winner": self.winner.name if self.winner else None,
            "move_count": self.move_count,
            "turn_count": self.turn_count,
            "last_move": self.last_move,
            "last_result": self.last_result
        }
    
    def to_dict(self):
        """Convert game to dictionary for JSON serialization."""
        return {
            "game_id": self.id,
            "player1": self.player1.to_dict(),
            "player2": self.player2.to_dict(),
            "current_player": self.current_player.name,
            "game_over": self.is_over(),
            "winner": self.winner.name if self.winner else None,
            "move_count": self.move_count,
            "turn_count": self.turn_count,
            "created_at": self.created_at.isoformat(),
            "last_move": self.last_move,
            "last_result": self.last_result
        }
