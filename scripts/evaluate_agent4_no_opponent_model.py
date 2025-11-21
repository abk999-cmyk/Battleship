#!/usr/bin/env python3
"""
evaluate_agent4_no_opponent_model.py - Evaluate AIAgent4 with opponent modeling disabled
========================================================================================

Tests AIAgent4 (with opponent_model=0.0) against all 10 naive agents to isolate
the contribution of opponent modeling by comparing with the full AIAgent4.

Usage:
    python scripts/evaluate_agent4_no_opponent_model.py --games 100
"""

import sys
import os
import csv
import random
import logging
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set seed for reproducibility
random.seed(1337)
np.random.seed(1337)

# Suppress agent logging
logging.getLogger('AI_agent4').setLevel(logging.WARNING)

from game import BattleshipGame
from AI_agent4 import AIAgent4
from AI_testing_agents import (
    NaiveAgent1, NaiveAgent2, NaiveAgent3, NaiveAgent4, NaiveAgent5,
    NaiveAgent6, NaiveAgent7, NaiveAgent8, NaiveAgent9, NaiveAgent10
)

# Configuration: AIAgent4 WITHOUT opponent modeling
AGENT_CONFIG = {
    'name': 'AIAgent4-NoOpponentModel',
    'weights': {
        'density': 0.4904895891191308,
        'neural': 0.6021993763688016,
        'montecarlo': 0.9711647059426145,
        'information_gain': 0.0,
        'opponent_model': 0.0  # DISABLED
    }
}

NAIVE_AGENTS = [
    NaiveAgent1, NaiveAgent2, NaiveAgent3, NaiveAgent4, NaiveAgent5,
    NaiveAgent6, NaiveAgent7, NaiveAgent8, NaiveAgent9, NaiveAgent10
]


def run_single_game(opponent_class, game_num):
    """Run a single game between AIAgent4 (no opponent model) and a naive agent."""
    # Create agent with disabled opponent modeling
    agent = AIAgent4(AGENT_CONFIG['name'])
    agent.meta_weights.update(AGENT_CONFIG['weights'])
    
    # Create opponent
    opponent = opponent_class(f"{opponent_class.__name__}")
    
    # Run game
    game = BattleshipGame(agent, opponent)
    game.play()
    
    # Determine winner and moves (corrected to show winner's moves only)
    winner_name = game.winner.name if game.winner else "Draw"
    moves_to_win = (game.move_count // 2) if hasattr(game, 'move_count') else (game.turns // 2)
    
    return {
        'opponent_name': opponent_class.__name__,
        'game_num': game_num,
        'winner': winner_name,
        'moves_to_win': moves_to_win,
        'agent_won': game.winner is agent
    }


def run_evaluation(games_per_opponent=100):
    """Run the evaluation against all naive agents."""
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"results/agent4_no_opponent_model_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Evaluating AIAgent4 (opponent_model=0.0) with {games_per_opponent} games per opponent")
    print(f"Output directory: {output_dir}")
    print(f"Total games: {len(NAIVE_AGENTS)} opponents × {games_per_opponent} = {len(NAIVE_AGENTS) * games_per_opponent}")
    print("=" * 80)
    
    # Store all results
    all_results = []
    
    # Calculate total games for progress bar
    total_games = len(NAIVE_AGENTS) * games_per_opponent
    
    # Run all games with progress bar
    with tqdm(total=total_games, desc="Running games", ncols=100) as pbar:
        for opponent_class in NAIVE_AGENTS:
            for game_num in range(1, games_per_opponent + 1):
                result = run_single_game(opponent_class, game_num)
                all_results.append(result)
                pbar.update(1)
    
    # Save detailed CSV
    print("\nSaving detailed results CSV...")
    csv_path = output_dir / "detailed_results.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['opponent_name', 'game_num', 'winner', 'moves_to_win'])
        writer.writeheader()
        for result in all_results:
            writer.writerow({
                'opponent_name': result['opponent_name'],
                'game_num': result['game_num'],
                'winner': result['winner'],
                'moves_to_win': result['moves_to_win']
            })
    
    # Generate summary report
    print("Generating summary report...")
    generate_summary_report(all_results, output_dir, games_per_opponent)
    
    print(f"\n{'=' * 80}")
    print(f"Evaluation complete!")
    print(f"Results saved to: {output_dir}")
    print(f"{'=' * 80}")


def generate_summary_report(all_results, output_dir, games_per_opponent):
    """Generate a text summary report."""
    
    report_path = output_dir / "summary_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("AIAgent4 WITHOUT OPPONENT MODELING - EVALUATION REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Games per opponent: {games_per_opponent}\n")
        f.write(f"Random seed: 1337\n")
        f.write(f"Configuration: opponent_model = 0.0 (DISABLED)\n")
        f.write(f"Other weights: density={AGENT_CONFIG['weights']['density']:.2f}, ")
        f.write(f"neural={AGENT_CONFIG['weights']['neural']:.2f}, ")
        f.write(f"montecarlo={AGENT_CONFIG['weights']['montecarlo']:.2f}, ")
        f.write(f"information_gain={AGENT_CONFIG['weights']['information_gain']:.2f}\n")
        f.write("\n")
        
        # Overall statistics
        total_games = len(all_results)
        total_wins = sum(1 for r in all_results if r['agent_won'])
        win_rate = (total_wins / total_games * 100) if total_games > 0 else 0
        
        wins_only = [r for r in all_results if r['agent_won']]
        avg_moves_to_win = np.mean([r['moves_to_win'] for r in wins_only]) if wins_only else 0
        
        f.write("=" * 80 + "\n")
        f.write("OVERALL PERFORMANCE\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total Games:          {total_games}\n")
        f.write(f"Total Wins:           {total_wins}\n")
        f.write(f"Total Losses:         {total_games - total_wins}\n")
        f.write(f"Win Rate:             {win_rate:.2f}%\n")
        f.write(f"Avg Moves to Win:     {avg_moves_to_win:.2f}\n")
        f.write("\n")
        
        # Per-opponent breakdown
        f.write("=" * 80 + "\n")
        f.write("PERFORMANCE VS EACH OPPONENT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"{'Opponent':<20} {'Wins':>6} {'Losses':>7} {'Win Rate':>10} {'Avg Moves':>12}\n")
        f.write("-" * 80 + "\n")
        
        for opponent_class in NAIVE_AGENTS:
            opponent_name = opponent_class.__name__
            opp_results = [r for r in all_results if r['opponent_name'] == opponent_name]
            
            opp_wins = sum(1 for r in opp_results if r['agent_won'])
            opp_total = len(opp_results)
            opp_win_rate = (opp_wins / opp_total * 100) if opp_total > 0 else 0
            
            opp_wins_only = [r for r in opp_results if r['agent_won']]
            opp_avg_moves = np.mean([r['moves_to_win'] for r in opp_wins_only]) if opp_wins_only else 0
            
            f.write(f"{opponent_name:<20} {opp_wins:>6} {opp_total - opp_wins:>7} "
                   f"{opp_win_rate:>9.1f}% {opp_avg_moves:>11.2f}\n")
        
        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write("NOTES\n")
        f.write("=" * 80 + "\n\n")
        f.write("This evaluation tests AIAgent4 with opponent modeling DISABLED to isolate\n")
        f.write("its contribution. Compare these results with the full AIAgent4 evaluation\n")
        f.write("(opponent_model=0.51) to measure the impact of opponent modeling.\n")
    
    print(f"Summary report saved to: {report_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Evaluate AIAgent4 without opponent modeling"
    )
    parser.add_argument('--games', type=int, default=100,
                        help='Number of games per opponent (default: 100)')
    
    args = parser.parse_args()
    
    run_evaluation(games_per_opponent=args.games)


if __name__ == "__main__":
    main()

