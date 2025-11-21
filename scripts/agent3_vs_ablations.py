#!/usr/bin/env python3
"""
agent3_vs_ablations.py - Head-to-head comparison of AIAgent3 vs AIAgent4 ablations
================================================================================

Runs 100 games of AIAgent3 against each ablated AIAgent4 variant to compare
performance in direct competition.

Usage:
    python scripts/agent3_vs_ablations.py --games 10   # Test run
    python scripts/agent3_vs_ablations.py --games 100  # Full evaluation
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

# Add parent directory to path to import game modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set seed for reproducibility
random.seed(1337)
np.random.seed(1337)

# Suppress agent logging to keep progress bar clean
logging.getLogger('AI_Agent3').setLevel(logging.WARNING)
logging.getLogger('AI_agent4').setLevel(logging.WARNING)

from game import BattleshipGame
from AI_agent3 import AIAgent3
from AI_agent4 import AIAgent4

# Agent configurations
AGENT3_CONFIG = {
    'name': 'AIAgent3',
    'class': AIAgent3,
    'weights': {
        'density': 0.6,
        'neural': 0.5,
        'montecarlo': 0.7,
        'information_gain': 0.5,
        'opponent_model': 0.0
    }
}

ABLATION_CONFIGS = [
    {
        'name': 'AIAgent4-Neural-Only',
        'class': AIAgent4,
        'weights': {
            'density': 0.0,
            'neural': 1.0,
            'montecarlo': 0.0,
            'information_gain': 0.0,
            'opponent_model': 0.0
        }
    },
    {
        'name': 'AIAgent4-Density-Only',
        'class': AIAgent4,
        'weights': {
            'density': 1.0,
            'neural': 0.0,
            'montecarlo': 0.0,
            'information_gain': 0.0,
            'opponent_model': 0.0
        }
    },
    {
        'name': 'AIAgent4-MonteCarlo-Only',
        'class': AIAgent4,
        'weights': {
            'density': 0.0,
            'neural': 0.0,
            'montecarlo': 1.0,
            'information_gain': 0.0,
            'opponent_model': 0.0
        }
    },
    {
        'name': 'AIAgent4-InfoGain-Only',
        'class': AIAgent4,
        'weights': {
            'density': 0.0,
            'neural': 0.0,
            'montecarlo': 0.0,
            'information_gain': 1.0,
            'opponent_model': 0.0
        }
    }
]


def run_head_to_head(agent1_config, agent2_config, game_num):
    """
    Run a single head-to-head game between two agent configurations.
    
    Returns:
        dict: Game results
    """
    # Create agent 1 (AIAgent3)
    if agent1_config['class'] == AIAgent3:
        agent1 = AIAgent3(agent1_config['name'], continuous_learning=False, opponent_modeling=False)
    else:
        agent1 = AIAgent4(agent1_config['name'])
    agent1.meta_weights.update(agent1_config['weights'])
    
    # Create agent 2 (AIAgent4 ablation)
    if agent2_config['class'] == AIAgent3:
        agent2 = AIAgent3(agent2_config['name'], continuous_learning=False, opponent_modeling=False)
    else:
        agent2 = AIAgent4(agent2_config['name'])
    agent2.meta_weights.update(agent2_config['weights'])
    
    # Run game
    game = BattleshipGame(agent1, agent2)
    game.play()
    
    # Determine winner and moves
    winner_name = game.winner.name if game.winner else "Draw"
    moves_to_win = (game.move_count // 2) if hasattr(game, 'move_count') else (game.turns // 2)
    
    return {
        'agent1_name': agent1_config['name'],
        'agent2_name': agent2_config['name'],
        'game_num': game_num,
        'winner': winner_name,
        'moves_to_win': moves_to_win,
        'agent1_won': game.winner is agent1,
        'agent2_won': game.winner is agent2
    }


def run_comparison_study(games_per_matchup=100):
    """
    Run AIAgent3 vs all ablated AIAgent4 variants.
    
    Args:
        games_per_matchup: Number of games to play for each matchup
    """
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"results/agent3_vs_ablations_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate total games
    total_games = len(ABLATION_CONFIGS) * games_per_matchup
    
    print(f"Starting AIAgent3 vs Ablations Comparison Study")
    print(f"Output directory: {output_dir}")
    print(f"Games per matchup: {games_per_matchup}")
    print(f"Total matchups: {len(ABLATION_CONFIGS)}")
    print(f"Total games: {total_games}")
    print("=" * 80)
    
    # Store all results
    h2h_results = []
    
    # Run all games with progress bar
    with tqdm(total=total_games, desc="Running games", ncols=100) as pbar:
        for opponent_config in ABLATION_CONFIGS:
            for game_num in range(1, games_per_matchup + 1):
                result = run_head_to_head(AGENT3_CONFIG, opponent_config, game_num)
                h2h_results.append(result)
                pbar.update(1)
    
    # Save detailed results CSV
    print("\nSaving detailed results...")
    csv_path = output_dir / "agent3_vs_ablations_detailed.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['agent1_name', 'agent2_name', 'game_num', 'winner', 'moves_to_win'])
        writer.writeheader()
        for result in h2h_results:
            writer.writerow({
                'agent1_name': result['agent1_name'],
                'agent2_name': result['agent2_name'],
                'game_num': result['game_num'],
                'winner': result['winner'],
                'moves_to_win': result['moves_to_win']
            })
    
    # Generate summary report
    print("Generating summary report...")
    generate_summary_report(h2h_results, output_dir, games_per_matchup)
    
    print(f"\n{'=' * 80}")
    print(f"Comparison study complete!")
    print(f"Results saved to: {output_dir}")
    print(f"{'=' * 80}")


def generate_summary_report(h2h_results, output_dir, games_per_matchup):
    """Generate a comprehensive summary report."""
    
    report_path = output_dir / "summary_report_agent3.txt"
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("AIAGENT3 VS ABLATED AIAGENT4 VARIANTS - COMPARISON REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Games per matchup: {games_per_matchup}\n")
        f.write(f"Random seed: 1337\n")
        f.write(f"Total games: {len(h2h_results)}\n")
        f.write("\n")
        
        # Head-to-head results
        f.write("=" * 80 + "\n")
        f.write("HEAD-TO-HEAD RESULTS: AIAgent3 vs AIAgent4 Ablations\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"{'Opponent':<30} {'Agent3 Wins':>12} {'Opp Wins':>10} {'Win Rate':>10} {'Avg Moves':>12}\n")
        f.write("-" * 80 + "\n")
        
        # Group by opponent and calculate statistics
        opponents_set = set([r['agent2_name'] for r in h2h_results])
        summary_stats = []
        
        for opponent_name in sorted(opponents_set):
            opp_games = [r for r in h2h_results if r['agent2_name'] == opponent_name]
            agent3_wins = sum(1 for r in opp_games if r['agent1_won'])
            total = len(opp_games)
            win_rate = (agent3_wins / total * 100) if total > 0 else 0
            
            # Calculate average moves across all games (not just wins)
            avg_moves = np.mean([r['moves_to_win'] for r in opp_games]) if opp_games else 0
            
            f.write(f"{opponent_name:<30} {agent3_wins:>12} {total - agent3_wins:>10} {win_rate:>9.2f}% {avg_moves:>11.2f}\n")
            
            summary_stats.append({
                'opponent': opponent_name,
                'agent3_wins': agent3_wins,
                'opponent_wins': total - agent3_wins,
                'win_rate': win_rate,
                'avg_moves': avg_moves
            })
        
        # Overall summary
        f.write("\n" + "=" * 80 + "\n")
        f.write("OVERALL SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        total_agent3_wins = sum(s['agent3_wins'] for s in summary_stats)
        total_games_played = len(h2h_results)
        overall_win_rate = (total_agent3_wins / total_games_played * 100) if total_games_played > 0 else 0
        overall_avg_moves = np.mean([r['moves_to_win'] for r in h2h_results])
        
        f.write(f"AIAgent3 Overall Performance:\n")
        f.write(f"  Total Games:          {total_games_played}\n")
        f.write(f"  Total Wins:           {total_agent3_wins}\n")
        f.write(f"  Total Losses:         {total_games_played - total_agent3_wins}\n")
        f.write(f"  Overall Win Rate:     {overall_win_rate:.2f}%\n")
        f.write(f"  Overall Avg Moves:    {overall_avg_moves:.2f}\n")
        f.write("\n")
        
        # Performance analysis
        f.write("=" * 80 + "\n")
        f.write("PERFORMANCE ANALYSIS\n")
        f.write("=" * 80 + "\n\n")
        
        # Best and worst matchups
        best_matchup = max(summary_stats, key=lambda x: x['win_rate'])
        worst_matchup = min(summary_stats, key=lambda x: x['win_rate'])
        fastest_matchup = min(summary_stats, key=lambda x: x['avg_moves'])
        slowest_matchup = max(summary_stats, key=lambda x: x['avg_moves'])
        
        f.write(f"Best Matchup (highest win rate):\n")
        f.write(f"  vs {best_matchup['opponent']}: {best_matchup['win_rate']:.2f}% win rate\n\n")
        
        f.write(f"Worst Matchup (lowest win rate):\n")
        f.write(f"  vs {worst_matchup['opponent']}: {worst_matchup['win_rate']:.2f}% win rate\n\n")
        
        f.write(f"Fastest Games (fewest moves):\n")
        f.write(f"  vs {fastest_matchup['opponent']}: {fastest_matchup['avg_moves']:.2f} avg moves\n\n")
        
        f.write(f"Slowest Games (most moves):\n")
        f.write(f"  vs {slowest_matchup['opponent']}: {slowest_matchup['avg_moves']:.2f} avg moves\n\n")
    
    print(f"Summary report saved to: {report_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Run AIAgent3 vs AIAgent4 ablations comparison")
    parser.add_argument('--games', type=int, default=100,
                        help='Number of games per matchup (default: 100)')
    
    args = parser.parse_args()
    
    run_comparison_study(games_per_matchup=args.games)


if __name__ == "__main__":
    main()




