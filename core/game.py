import sys
from pathlib import Path

# Add the parent directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.player import HumanPlayer          # used elsewhere in the UI
from agents.AI_agent import AIPlayer           # legacy agent still referenced

class BattleshipGame:
    """Minimal, head-less game engine for AI-versus-AI training & GA runs."""

    def __init__(self, player1, player2):
        self.p1, self.p2 = player1, player2
        self.current, self.opponent = self.p1, self.p2
        self.move_count: int = 0
        self.turns: int = 0
        self.winner = None              # becomes a Player object when the game ends

    # ------------------------------------------------------------------ #
    # Core loop helpers
    # ------------------------------------------------------------------ #
    def is_over(self) -> bool:
        return self.winner is not None

    def play(self):
        """Run until someone wins (or forfeits because no moves remain)."""
        while not self.is_over():
            self.step()
        return self.winner

    def step(self):
        """Let the *current* player act, then hand control to the opponent."""
        move_played = self.current.take_turn(self.opponent.board)  # may be None
        self.register_move(move_played)

    # ----------------------------- utilities --------------------------- #
    def step_manual(self, move):
        """External driver supplies a move (used by RL scripts)."""
        if hasattr(self.current, "available_moves"):
            self.current.available_moves.discard(move)
        result = self.opponent.board.attack(move)
        if hasattr(self.current, "update_result_grid"):
            self.current.update_result_grid(move, result)
        self.post_move()

    def register_move(self, move):
        """Common bookkeeping after *automatic* moves."""
        # Guard: move can be None when the player had no legal squares left.
        if move is not None and hasattr(self.current, "available_moves"):
            self.current.available_moves.discard(move)
        self.post_move()

    def post_move(self):
        """Determine whether the game is over and swap turn order."""
        self.move_count += 1
        self.turns += 1

        # 1️⃣  Check if the defending fleet is gone.
        if self.opponent.board.all_ships_sunk():
            self.winner = self.current
            return

        # 2️⃣  Normal turn swap.
        self.current, self.opponent = self.opponent, self.current

        # 3️⃣  NEW: immediate forfeit if the next player has no legal moves.
        if hasattr(self.current, "available_moves") and not self.current.available_moves:
            # They are about to play but the board is exhausted → they lose.
            self.winner = self.opponent
