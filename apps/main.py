# main.py
import argparse
import os
import time
import csv
import sys
from pathlib import Path
from tqdm import tqdm

# Add the parent directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.AI_agent2 import AIPlayer2
from agents.AI_agent4 import AIAgent4
from agents.AI_testing_agents import (
    NaiveAgent1, NaiveAgent2, NaiveAgent3, NaiveAgent4, NaiveAgent5,
    NaiveAgent6, NaiveAgent7, NaiveAgent8, NaiveAgent9, NaiveAgent10,
    UltimateBattleshipAgent
)
from core.game import BattleshipGame

NUM_GAMES_PER_AGENT = 1000

# ----------------------------------------------------------------------
def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")

def run_batch(num_games=NUM_GAMES_PER_AGENT):
    agent_classes = [
        NaiveAgent1, NaiveAgent2, NaiveAgent3, NaiveAgent4, NaiveAgent5,
        NaiveAgent6, NaiveAgent7, NaiveAgent8, NaiveAgent9, NaiveAgent10,
        UltimateBattleshipAgent
    ]
    results = []
    for opp_cls in agent_classes:
        wins = losses = 0
        total_moves_to_win = 0
        desc = f"Testing {opp_cls.__name__}"
        for _ in tqdm(range(num_games), desc=desc, ncols=80):
            ai = AIAgent4("MainAI") #########################################################
            opp = opp_cls("Opponent")
            game = BattleshipGame(ai, opp)
            game.play()
            moves = getattr(game, "move_count", None)
            if moves is None:
                # Fallback: estimate moves as half the step count rounded up
                moves = getattr(game, "turns", 0)
            if game.winner is ai:
                wins += 1
                total_moves_to_win += moves
            else:
                losses += 1
        avg_moves = total_moves_to_win / wins if wins else 0
        results.append({
            "opponent": opp_cls.__name__,
            "main_ai_wins": wins,
            "main_ai_losses": losses,
            "avg_moves_to_win": round(avg_moves, 2)/2
        })

    with open("testing_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["opponent", "main_ai_wins", "main_ai_losses", "avg_moves_to_win"]
        )
        writer.writeheader()
        writer.writerows(results)

    print("\n✅ Batch complete. Results saved to testing_results.csv")

# ----------------------------------------------------------------------
def watch_game(opp_cls, delay=0.3):
    ai = AIAgent4("MainAI")
    opp = opp_cls("Opponent")
    game = BattleshipGame(ai, opp)

    turn = 1
    while not game.is_over():
        clear_screen()

        previous_player = game.current
        game.step()
        move = getattr(previous_player, "last_move", None)
        result = getattr(previous_player, "last_result", None)

        print(f"Turn {turn}: {previous_player.name} attacked {move} → {result}\n")

        print("MainAI's board (ships revealed):")
        print(ai.board.display(reveal=True))
        print("\nMainAI's view of Opponent (hits and misses):")
        print(ai.view_display())

        print("\nOpponent's actual board (ships hidden):")
        print(opp.board.display(reveal=False))

        time.sleep(delay)
        turn += 1

    clear_screen()
    print(f"\n🏁 Game over in {(turn/2)-1} turns! Winner: {game.winner.name}\n")
    print("MainAI full board (ships revealed):")
    print(ai.board.display(reveal=True))
    print("\nOpponent full board (ships revealed):")
    print(opp.board.display(reveal=True))

# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true", help="Watch a single game against an opponent")
    parser.add_argument("--opp", type=str, default="ultimate", help="Opponent: ultimate or naive1..naive10")
    parser.add_argument("--delay", type=float, default=0.3, help="Seconds between turns when watching")
    parser.add_argument("--games", type=int, default=NUM_GAMES_PER_AGENT, help="Number of games per opponent for batch")

    args = parser.parse_args()

    if args.watch:
        mapping = {
            "ultimate": UltimateBattleshipAgent,
            "naive1": NaiveAgent1, "naive2": NaiveAgent2, "naive3": NaiveAgent3,
            "naive4": NaiveAgent4, "naive5": NaiveAgent5, "naive6": NaiveAgent6,
            "naive7": NaiveAgent7, "naive8": NaiveAgent8, "naive9": NaiveAgent9,
            "naive10": NaiveAgent10
        }
        opp_cls = mapping.get(args.opp.lower(), UltimateBattleshipAgent)
        watch_game(opp_cls, args.delay)
    else:
        run_batch(args.games)