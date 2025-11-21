#!/usr/bin/env python3
"""
ablation_study.py - Comprehensive ablation study of AIAgent4 strategy components
================================================================================

Tests AIAgent3 and AIAgent4 with different strategy configurations against
10 naive opponent agents to understand the contribution of each strategy component.

Usage:
    python scripts/ablation_study.py --games 1     # Test run
    python scripts/ablation_study.py --games 100   # Full evaluation
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
from AI_testing_agents import (
    NaiveAgent1, NaiveAgent2, NaiveAgent3, NaiveAgent4, NaiveAgent5,
    NaiveAgent6, NaiveAgent7, NaiveAgent8, NaiveAgent9, NaiveAgent10
)

# Agent configurations for ablation study
AGENT_CONFIGS = [
    {
        'name': 'AIAgent3',
        'class': AIAgent3,
        'weights': {
            'density': 0.6,
            'neural': 0.5,
            'montecarlo': 0.7,
            'information_gain': 0.5,
            'opponent_model': 0.0
        }
    },
    {
        'name': 'AIAgent4-Full',
        'class': AIAgent4,
        'weights': {
            'density': 0.4904895891191308,
            'neural': 0.6021993763688016,
            'montecarlo': 0.9711647059426145,
            'information_gain': 0.0,
            'opponent_model': 0.5145790072743006
        }
    },
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

# Naive opponent classes
NAIVE_AGENTS = [
    NaiveAgent1, NaiveAgent2, NaiveAgent3, NaiveAgent4, NaiveAgent5,
    NaiveAgent6, NaiveAgent7, NaiveAgent8, NaiveAgent9, NaiveAgent10
]


def run_single_game(agent_config, opponent_class, game_num):
    """
    Run a single game between an agent configuration and a naive opponent.
    
    Returns:
        dict: Game results with agent_name, opponent_name, game_num, winner, moves_to_win
    """
    # Create agent with specified configuration
    if agent_config['class'] == AIAgent3:
        agent = AIAgent3(agent_config['name'], continuous_learning=False, opponent_modeling=False)
    else:
        agent = AIAgent4(agent_config['name'])
    
    # Override meta_weights
    agent.meta_weights.update(agent_config['weights'])
    
    # Create opponent
    opponent = opponent_class(f"{opponent_class.__name__}")
    
    # Run game
    game = BattleshipGame(agent, opponent)
    game.play()
    
    # Determine winner and moves
    winner_name = game.winner.name if game.winner else "Draw"
    moves_to_win = (game.move_count // 2) if hasattr(game, 'move_count') else (game.turns // 2)
    
    return {
        'agent_name': agent_config['name'],
        'opponent_name': opponent_class.__name__,
        'game_num': game_num,
        'winner': winner_name,
        'moves_to_win': moves_to_win,
        'agent_won': game.winner is agent
    }


def run_head_to_head(agent1_config, agent2_config, game_num):
    """
    Run a single head-to-head game between two agent configurations.
    
    Returns:
        dict: Game results
    """
    # Create agent 1
    if agent1_config['class'] == AIAgent3:
        agent1 = AIAgent3(agent1_config['name'], continuous_learning=False, opponent_modeling=False)
    else:
        agent1 = AIAgent4(agent1_config['name'])
    agent1.meta_weights.update(agent1_config['weights'])
    
    # Create agent 2
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
        'agent1_won': game.winner is agent1
    }


def run_ablation_study(games_per_matchup=1):
    """
    Run the complete ablation study.
    
    Args:
        games_per_matchup: Number of games to play for each agent-opponent pair
    """
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"results/ablation_study_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate total games
    naive_games = len(AGENT_CONFIGS) * len(NAIVE_AGENTS) * games_per_matchup
    # AIAgent4-Full vs all other agents (5 matchups)
    head_to_head_games = 5 * games_per_matchup
    total_games = naive_games + head_to_head_games
    
    print(f"Starting ablation study with {games_per_matchup} game(s) per matchup")
    print(f"Output directory: {output_dir}")
    print(f"Naive agent games: {naive_games} ({len(AGENT_CONFIGS)} agents × {len(NAIVE_AGENTS)} opponents × {games_per_matchup})")
    print(f"Head-to-head games: {head_to_head_games} (AIAgent4-Full vs 5 other agents × {games_per_matchup})")
    print(f"Total games: {total_games}")
    print("=" * 80)
    
    # Store all results
    all_results = []
    h2h_results = []
    
    # Run all games vs naive agents with progress bar
    with tqdm(total=total_games, desc="Running games", ncols=100) as pbar:
        # Phase 1: All agents vs naive opponents
        for agent_config in AGENT_CONFIGS:
            for opponent_class in NAIVE_AGENTS:
                for game_num in range(1, games_per_matchup + 1):
                    result = run_single_game(agent_config, opponent_class, game_num)
                    all_results.append(result)
                    pbar.update(1)
        
        # Phase 2: AIAgent4-Full vs other agents
        agent4_full = AGENT_CONFIGS[1]  # AIAgent4-Full is at index 1
        other_agents = [AGENT_CONFIGS[0]] + AGENT_CONFIGS[2:]  # AIAgent3 + other ablations
        
        for opponent_config in other_agents:
            for game_num in range(1, games_per_matchup + 1):
                result = run_head_to_head(agent4_full, opponent_config, game_num)
                h2h_results.append(result)
                pbar.update(1)
    
    # Save individual CSV files per opponent
    print("\nSaving detailed CSV files per opponent...")
    for opponent_class in NAIVE_AGENTS:
        opponent_name = opponent_class.__name__
        opponent_results = [r for r in all_results if r['opponent_name'] == opponent_name]
        
        csv_path = output_dir / f"agent_vs_{opponent_name.lower()}.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['agent_name', 'game_num', 'winner', 'moves_to_win'])
            writer.writeheader()
            for result in opponent_results:
                writer.writerow({
                    'agent_name': result['agent_name'],
                    'game_num': result['game_num'],
                    'winner': result['winner'],
                    'moves_to_win': result['moves_to_win']
                })
    
    # Save head-to-head results
    if h2h_results:
        print("Saving head-to-head results...")
        h2h_path = output_dir / "agent4_full_vs_others.csv"
        with open(h2h_path, 'w', newline='') as f:
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
    generate_summary_report(all_results, h2h_results, output_dir, games_per_matchup)
    
    print(f"\n{'=' * 80}")
    print(f"Ablation study complete!")
    print(f"Results saved to: {output_dir}")
    print(f"{'=' * 80}")


def generate_summary_report(all_results, h2h_results, output_dir, games_per_matchup):
    """Generate a comprehensive summary report."""
    
    report_path = output_dir / "summary_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("ABLATION STUDY SUMMARY REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Games per matchup: {games_per_matchup}\n")
        f.write(f"Random seed: 1337\n")
        f.write(f"Total naive agent games: {len(all_results)}\n")
        f.write(f"Total head-to-head games: {len(h2h_results)}\n")
        f.write("\n")
        
        # Per-agent summary
        f.write("=" * 80 + "\n")
        f.write("AGENT PERFORMANCE SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        for agent_config in AGENT_CONFIGS:
            agent_name = agent_config['name']
            agent_results = [r for r in all_results if r['agent_name'] == agent_name]
            
            f.write(f"\n{agent_name}\n")
            f.write("-" * 80 + "\n")
            f.write(f"Configuration: {agent_config['weights']}\n\n")
            
            # Overall statistics
            total_games = len(agent_results)
            total_wins = sum(1 for r in agent_results if r['agent_won'])
            win_rate = (total_wins / total_games * 100) if total_games > 0 else 0
            
            wins_only = [r for r in agent_results if r['agent_won']]
            avg_moves_to_win = np.mean([r['moves_to_win'] for r in wins_only]) if wins_only else 0
            
            f.write(f"Overall Statistics:\n")
            f.write(f"  Total Games:          {total_games}\n")
            f.write(f"  Total Wins:           {total_wins}\n")
            f.write(f"  Total Losses:         {total_games - total_wins}\n")
            f.write(f"  Win Rate:             {win_rate:.2f}%\n")
            f.write(f"  Avg Moves to Win:     {avg_moves_to_win:.2f}\n")
            f.write("\n")
            
            # Per-opponent breakdown
            f.write(f"Performance vs Each Opponent:\n")
            f.write(f"{'Opponent':<20} {'Wins':>6} {'Losses':>7} {'Win Rate':>10} {'Avg Moves':>12}\n")
            f.write("-" * 80 + "\n")
            
            for opponent_class in NAIVE_AGENTS:
                opponent_name = opponent_class.__name__
                opp_results = [r for r in agent_results if r['opponent_name'] == opponent_name]
                
                opp_wins = sum(1 for r in opp_results if r['agent_won'])
                opp_total = len(opp_results)
                opp_win_rate = (opp_wins / opp_total * 100) if opp_total > 0 else 0
                
                opp_wins_only = [r for r in opp_results if r['agent_won']]
                opp_avg_moves = np.mean([r['moves_to_win'] for r in opp_wins_only]) if opp_wins_only else 0
                
                f.write(f"{opponent_name:<20} {opp_wins:>6} {opp_total - opp_wins:>7} {opp_win_rate:>9.1f}% {opp_avg_moves:>11.2f}\n")
            
            f.write("\n")
        
        # Comparison table
        f.write("\n" + "=" * 80 + "\n")
        f.write("AGENT COMPARISON TABLE\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"{'Agent Name':<30} {'Total Wins':>12} {'Win Rate':>10} {'Avg Moves':>12}\n")
        f.write("-" * 80 + "\n")
        
        for agent_config in AGENT_CONFIGS:
            agent_name = agent_config['name']
            agent_results = [r for r in all_results if r['agent_name'] == agent_name]
            
            total_wins = sum(1 for r in agent_results if r['agent_won'])
            total_games = len(agent_results)
            win_rate = (total_wins / total_games * 100) if total_games > 0 else 0
            
            wins_only = [r for r in agent_results if r['agent_won']]
            avg_moves = np.mean([r['moves_to_win'] for r in wins_only]) if wins_only else 0
            
            f.write(f"{agent_name:<30} {total_wins:>12} {win_rate:>9.2f}% {avg_moves:>11.2f}\n")
        
        # Head-to-head results
        if h2h_results:
            f.write("\n" + "=" * 80 + "\n")
            f.write("HEAD-TO-HEAD RESULTS: AIAgent4-Full vs Other Agents\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"{'Opponent':<30} {'Agent4 Wins':>12} {'Opp Wins':>10} {'Win Rate':>10} {'Avg Moves':>12}\n")
            f.write("-" * 80 + "\n")
            
            # Group by opponent
            opponents_set = set([r['agent2_name'] for r in h2h_results])
            for opponent_name in sorted(opponents_set):
                opp_games = [r for r in h2h_results if r['agent2_name'] == opponent_name]
                agent4_wins = sum(1 for r in opp_games if r['agent1_won'])
                total = len(opp_games)
                win_rate = (agent4_wins / total * 100) if total > 0 else 0
                
                wins_only = [r for r in opp_games if r['agent1_won']]
                avg_moves = np.mean([r['moves_to_win'] for r in wins_only]) if wins_only else 0
                
                f.write(f"{opponent_name:<30} {agent4_wins:>12} {total - agent4_wins:>10} {win_rate:>9.2f}% {avg_moves:>11.2f}\n")
    
    print(f"Summary report saved to: {report_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Run ablation study for AIAgent4")
    parser.add_argument('--games', type=int, default=1,
                        help='Number of games per agent-opponent matchup (default: 1)')
    
    args = parser.parse_args()
    
    run_ablation_study(games_per_matchup=args.games)


if __name__ == "__main__":
    main()

