import argparse
import csv
import os
from tqdm import tqdm

from AI_agent4 import AIAgent4
from AI_agent2 import AIPlayer2
from AI_testing_agents import UltimateBattleshipAgent, NaiveAgent6
from game import BattleshipGame


def run_eval(num_games: int, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    opponents = [
        (UltimateBattleshipAgent, "UltimateBattleshipAgent"),
        (NaiveAgent6, "NaiveAgent6"),
        (AIPlayer2, "AIPlayer2"),
    ]

    rows = []
    for OppCls, opp_name in opponents:
        wins = 0
        losses = 0
        sum_moves_to_win = 0
        for _ in tqdm(range(num_games), desc=f"AIAgent4 vs {opp_name}", ncols=80):
            ai = AIAgent4("MainAI")
            opp = OppCls("Opponent")
            game = BattleshipGame(ai, opp)
            game.play()
            # move_count is incremented per step (two half-moves). Convert to turns per player.
            turns = getattr(game, "move_count", 0) // 2
            if game.winner is ai:
                wins += 1
                sum_moves_to_win += turns
            else:
                losses += 1
        avg_moves = (sum_moves_to_win / wins) if wins else 0.0
        rows.append({
            "opponent": opp_name,
            "games": num_games,
            "main_ai_wins": wins,
            "main_ai_losses": losses,
            "avg_moves_to_win": round(avg_moves, 2),
        })

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✅ Quick evaluation complete. Results → {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--out", type=str, default="reports/quick_results.csv")
    args = ap.parse_args()
    run_eval(args.games, args.out)
