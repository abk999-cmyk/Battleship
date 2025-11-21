#!/usr/bin/env python3
"""
benchmark_agent4.py - Comprehensive Benchmark and Ablation Study for AIAgent4
=============================================================================

This script performs a rigorous evaluation of AIAgent4 against:
1. AIAgent3 (100 games)
2. AIPlayer2 (100 games)
3. Ablated versions of itself (5 strategies × 100 games = 500 games)

The ablation study isolates each strategy component to measure its individual
contribution to overall performance.

Results include:
- Win rates and confidence intervals
- Average moves per game (efficiency metrics)
- Statistical significance testing
- Publication-quality IEEE-formatted visualizations

Author: Research Team
Date: October 2025
"""

import os
import sys
import json
import time
import random
import gc
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tqdm import tqdm

# Import game engine and agents
from game import BattleshipGame
from AI_agent4 import AIAgent4
from AI_agent3 import AIAgent3
from AI_agent2 import AIPlayer2

# Configure logging
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / f'benchmark_agent4_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('BenchmarkAgent4')

# Configuration
GAMES_PER_MATCHUP = 100
RANDOM_SEED = 1337
STRATEGY_KEYS = ['density', 'neural', 'montecarlo', 'information_gain', 'opponent_model']

# IEEE Publication settings
IEEE_DPI = 300
IEEE_FIGSIZE = (10, 6)
IEEE_FONT_SIZE = 11
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = IEEE_FONT_SIZE
plt.rcParams['axes.labelsize'] = IEEE_FONT_SIZE
plt.rcParams['xtick.labelsize'] = IEEE_FONT_SIZE - 1
plt.rcParams['ytick.labelsize'] = IEEE_FONT_SIZE - 1
plt.rcParams['legend.fontsize'] = IEEE_FONT_SIZE - 1


class BenchmarkRunner:
    """Main benchmark orchestrator for Agent4 evaluation."""
    
    def __init__(self, output_dir: str = None):
        """Initialize benchmark runner."""
        self.random_seed = RANDOM_SEED
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        
        # Create output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(output_dir or f"results/agent4_benchmark_{timestamp}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Benchmark output directory: {self.output_dir}")
        
        # Load GA weights for Agent4
        self.ga_weights = self._load_ga_weights()
        logger.info(f"Loaded GA weights: {self.ga_weights}")
        
        # Results storage
        self.detailed_results = []
        self.summary_stats = {}
        
    def _load_ga_weights(self) -> Dict[str, float]:
        """Load GA-optimized weights from models/ga_weights.json."""
        ga_path = Path("models/ga_weights.json")
        if not ga_path.exists():
            logger.warning(f"GA weights file not found at {ga_path}, using defaults")
            return {key: 0.5 for key in STRATEGY_KEYS}
        
        with open(ga_path, 'r') as f:
            weights = json.load(f)
        
        # Ensure all required keys are present
        for key in STRATEGY_KEYS:
            if key not in weights:
                logger.warning(f"Missing weight for '{key}', setting to 0.5")
                weights[key] = 0.5
        
        return weights
    
    def run_game(self, player1, player2, game_id: str) -> Dict[str, Any]:
        """
        Run a single game and return results.
        
        Returns:
            Dictionary with game_id, winner, moves, duration, and other metrics
        """
        start_time = time.time()
        
        try:
            game = BattleshipGame(player1, player2)
            winner = game.play()
            duration = time.time() - start_time
            
            # Determine winner name and side
            winner_name = winner.name if winner else "None"
            player1_won = (winner == player1)
            
            # Move count per player (game.move_count is total for both players)
            total_moves = game.move_count
            moves_per_player = total_moves // 2
            
            result = {
                'game_id': game_id,
                'player1_name': player1.name,
                'player2_name': player2.name,
                'winner': winner_name,
                'player1_won': player1_won,
                'total_moves': total_moves,
                'moves_per_player': moves_per_player,
                'duration_sec': duration
            }
            
            # Cleanup
            del game
            gc.collect()
            
            return result
            
        except Exception as e:
            logger.error(f"Error in game {game_id}: {e}", exc_info=True)
            return {
                'game_id': game_id,
                'player1_name': player1.name,
                'player2_name': player2.name,
                'winner': 'ERROR',
                'player1_won': False,
                'total_moves': 0,
                'moves_per_player': 0,
                'duration_sec': time.time() - start_time
            }
    
    def run_matchup(self, matchup_name: str, agent4_weights: Dict[str, float], 
                    opponent_class, opponent_name: str, num_games: int = GAMES_PER_MATCHUP) -> List[Dict]:
        """
        Run a complete matchup between Agent4 and an opponent.
        
        Args:
            matchup_name: Descriptive name for this matchup
            agent4_weights: Weight configuration for Agent4
            opponent_class: Class of opponent agent
            opponent_name: Name for opponent instances
            num_games: Number of games to play
            
        Returns:
            List of game result dictionaries
        """
        logger.info(f"Starting matchup: {matchup_name} ({num_games} games)")
        results = []
        
        for game_idx in tqdm(range(num_games), desc=f"  {matchup_name}", leave=False):
            # Create fresh agent instances
            agent4 = AIAgent4(name=f"Agent4", continuous_learning=False, opponent_modeling=False)
            agent4.meta_weights.update(agent4_weights)
            
            if opponent_class == AIAgent4:
                # For ablation studies, create another Agent4 with different weights
                opponent = opponent_class(name=opponent_name, continuous_learning=False, opponent_modeling=False)
                # opponent_name contains the ablation config info
            elif opponent_class == AIAgent3:
                opponent = opponent_class(name=opponent_name, continuous_learning=False, opponent_modeling=False)
            else:
                opponent = opponent_class(name=opponent_name)
            
            game_id = f"{matchup_name.replace(' ', '_')}_{game_idx:03d}"
            
            # Run game with Agent4 as player1
            result = self.run_game(agent4, opponent, game_id)
            result['matchup'] = matchup_name
            result['agent4_weights'] = agent4_weights.copy()
            
            results.append(result)
            
            # Cleanup
            del agent4, opponent
            gc.collect()
        
        logger.info(f"Completed matchup: {matchup_name}")
        return results
    
    def run_all_benchmarks(self):
        """Execute all benchmark tests."""
        logger.info("=" * 80)
        logger.info("Starting comprehensive Agent4 benchmark suite")
        logger.info("=" * 80)
        
        all_results = []
        
        # 1. Agent4 (full GA weights) vs Agent3
        logger.info("\n[1/7] Agent4 vs Agent3")
        results = self.run_matchup(
            matchup_name="Agent4 vs Agent3",
            agent4_weights=self.ga_weights,
            opponent_class=AIAgent3,
            opponent_name="Agent3"
        )
        all_results.extend(results)
        
        # 2. Agent4 (full GA weights) vs Agent2
        logger.info("\n[2/7] Agent4 vs Agent2")
        results = self.run_matchup(
            matchup_name="Agent4 vs Agent2",
            agent4_weights=self.ga_weights,
            opponent_class=AIPlayer2,
            opponent_name="Agent2"
        )
        all_results.extend(results)
        
        # 3-7. Ablation studies - each strategy isolated
        for idx, strategy in enumerate(STRATEGY_KEYS, start=3):
            logger.info(f"\n[{idx}/7] Ablation: {strategy} only")
            
            # Create ablated weights (all 0 except target strategy)
            ablated_weights = {key: 0.0 for key in STRATEGY_KEYS}
            ablated_weights[strategy] = self.ga_weights[strategy]
            
            # Agent4 (full weights) vs Agent4 (ablated weights)
            matchup_name = f"Agent4 vs Ablated-{strategy}"
            
            # Run games with custom opponent creation
            results = []
            for game_idx in tqdm(range(GAMES_PER_MATCHUP), desc=f"  {matchup_name}", leave=False):
                # Agent4 with full GA weights
                agent4_full = AIAgent4(name="Agent4-Full", continuous_learning=False, opponent_modeling=False)
                agent4_full.meta_weights.update(self.ga_weights)
                
                # Agent4 with ablated weights (only one strategy active)
                agent4_ablated = AIAgent4(name=f"Agent4-{strategy}", continuous_learning=False, opponent_modeling=False)
                agent4_ablated.meta_weights.update(ablated_weights)
                
                game_id = f"{matchup_name.replace(' ', '_')}_{game_idx:03d}"
                result = self.run_game(agent4_full, agent4_ablated, game_id)
                result['matchup'] = matchup_name
                result['agent4_weights'] = self.ga_weights.copy()
                result['ablated_strategy'] = strategy
                result['ablated_weights'] = ablated_weights.copy()
                
                results.append(result)
                
                # Cleanup
                del agent4_full, agent4_ablated
                gc.collect()
            
            all_results.extend(results)
        
        # Store all detailed results
        self.detailed_results = all_results
        logger.info("\n" + "=" * 80)
        logger.info("All benchmarks completed!")
        logger.info("=" * 80)
    
    def compute_statistics(self):
        """Compute summary statistics for all matchups."""
        logger.info("\nComputing summary statistics...")
        
        df = pd.DataFrame(self.detailed_results)
        
        # Group by matchup
        grouped = df.groupby('matchup')
        
        summary = {}
        
        for matchup_name, group in grouped:
            # Agent4 is always player1 in our setup
            wins = group['player1_won'].sum()
            total_games = len(group)
            win_rate = (wins / total_games) * 100
            
            # Move statistics for Agent4 (player1)
            moves_all_games = group['moves_per_player']
            avg_moves_all = moves_all_games.mean()
            std_moves_all = moves_all_games.std()
            
            # Moves when Agent4 wins
            wins_df = group[group['player1_won'] == True]
            if len(wins_df) > 0:
                avg_moves_wins = wins_df['moves_per_player'].mean()
                std_moves_wins = wins_df['moves_per_player'].std()
            else:
                avg_moves_wins = 0
                std_moves_wins = 0
            
            # Confidence intervals (95%)
            ci_95 = 1.96 * (std_moves_all / np.sqrt(total_games))
            
            # Standard error
            se_win_rate = np.sqrt((win_rate * (100 - win_rate)) / total_games)
            
            summary[matchup_name] = {
                'total_games': total_games,
                'wins': int(wins),
                'losses': int(total_games - wins),
                'win_rate': win_rate,
                'win_rate_se': se_win_rate,
                'avg_moves_all_games': avg_moves_all,
                'std_moves_all_games': std_moves_all,
                'ci_95_moves': ci_95,
                'avg_moves_when_winning': avg_moves_wins,
                'std_moves_when_winning': std_moves_wins,
                'avg_duration_sec': group['duration_sec'].mean()
            }
        
        self.summary_stats = summary
        
        logger.info("Summary statistics computed successfully")
        return summary
    
    def save_results(self):
        """Save all results to files."""
        logger.info("\nSaving results...")
        
        # 1. Detailed results CSV
        df = pd.DataFrame(self.detailed_results)
        csv_path = self.output_dir / "detailed_results.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved detailed results: {csv_path}")
        
        # 2. Summary statistics JSON
        json_path = self.output_dir / "summary_statistics.json"
        with open(json_path, 'w') as f:
            json.dump(self.summary_stats, f, indent=2)
        logger.info(f"Saved summary statistics: {json_path}")
        
        # 3. Human-readable report
        report_path = self.output_dir / "benchmark_report.txt"
        self._generate_text_report(report_path)
        logger.info(f"Saved benchmark report: {report_path}")
    
    def _generate_text_report(self, output_path: Path):
        """Generate a human-readable text report."""
        with open(output_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("Agent4 Comprehensive Benchmark Report\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Random Seed: {self.random_seed}\n")
            f.write(f"Games per Matchup: {GAMES_PER_MATCHUP}\n\n")
            
            f.write("GA Weights Used:\n")
            for key, value in self.ga_weights.items():
                f.write(f"  {key:20s}: {value:.6f}\n")
            f.write("\n" + "=" * 80 + "\n\n")
            
            # Summary for each matchup
            for matchup_name, stats in self.summary_stats.items():
                f.write(f"{matchup_name}\n")
                f.write("-" * len(matchup_name) + "\n")
                f.write(f"  Total Games:              {stats['total_games']}\n")
                f.write(f"  Wins:                     {stats['wins']}\n")
                f.write(f"  Losses:                   {stats['losses']}\n")
                f.write(f"  Win Rate:                 {stats['win_rate']:.2f}% ± {stats['win_rate_se']:.2f}%\n")
                f.write(f"  Avg Moves (all games):    {stats['avg_moves_all_games']:.2f} ± {stats['std_moves_all_games']:.2f}\n")
                f.write(f"  95% CI (moves):           ±{stats['ci_95_moves']:.2f}\n")
                f.write(f"  Avg Moves (wins only):    {stats['avg_moves_when_winning']:.2f} ± {stats['std_moves_when_winning']:.2f}\n")
                f.write(f"  Avg Duration:             {stats['avg_duration_sec']:.2f} sec\n")
                f.write("\n")
            
            f.write("=" * 80 + "\n")
    
    def generate_visualizations(self):
        """Generate all publication-quality visualizations."""
        logger.info("\nGenerating IEEE-quality visualizations...")
        
        sns.set_palette("colorblind")
        
        # Extract data for plotting
        matchups = list(self.summary_stats.keys())
        win_rates = [self.summary_stats[m]['win_rate'] for m in matchups]
        win_rate_errors = [self.summary_stats[m]['win_rate_se'] for m in matchups]
        avg_moves = [self.summary_stats[m]['avg_moves_all_games'] for m in matchups]
        move_errors = [self.summary_stats[m]['ci_95_moves'] for m in matchups]
        
        # 1. Win Rate Comparison
        self._plot_win_rates(matchups, win_rates, win_rate_errors)
        
        # 2. Average Moves per Game
        self._plot_avg_moves(matchups, avg_moves, move_errors)
        
        # 3. Ablation Study Impact
        self._plot_ablation_impact()
        
        # 4. Scatter Plot: Efficiency vs Win Rate
        self._plot_efficiency_scatter(matchups, win_rates, avg_moves)
        
        # 5. Performance Heatmap
        self._plot_performance_heatmap()
        
        logger.info("All visualizations generated successfully")
    
    def _plot_win_rates(self, matchups: List[str], win_rates: List[float], errors: List[float]):
        """Generate win rate comparison bar chart."""
        fig, ax = plt.subplots(figsize=IEEE_FIGSIZE, dpi=IEEE_DPI)
        
        # Shorten matchup names for display
        display_names = [m.replace("Agent4 vs ", "vs ").replace("Ablated-", "Only-") for m in matchups]
        
        x_pos = np.arange(len(matchups))
        colors = ['#1f77b4' if 'Ablated' not in m else '#ff7f0e' for m in matchups]
        
        bars = ax.bar(x_pos, win_rates, yerr=errors, capsize=5, color=colors, 
                      edgecolor='black', linewidth=1.2, alpha=0.8)
        
        ax.set_xlabel('Opponent/Configuration', fontweight='bold')
        ax.set_ylabel('Win Rate (%)', fontweight='bold')
        ax.set_title('Agent4 Win Rate Against Various Opponents', fontweight='bold', pad=20)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(display_names, rotation=45, ha='right')
        ax.set_ylim(0, 105)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.axhline(y=50, color='red', linestyle='--', linewidth=1, alpha=0.5, label='50% baseline')
        
        # Add value labels on bars
        for bar, rate in zip(bars, win_rates):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{rate:.1f}%', ha='center', va='bottom', fontsize=9)
        
        ax.legend(loc='lower right')
        plt.tight_layout()
        
        # Save in multiple formats
        fig.savefig(self.output_dir / "fig_win_rates.png", dpi=IEEE_DPI, bbox_inches='tight')
        fig.savefig(self.output_dir / "fig_win_rates.pdf", bbox_inches='tight')
        plt.close(fig)
        logger.info("  ✓ Win rate comparison saved")
    
    def _plot_avg_moves(self, matchups: List[str], avg_moves: List[float], errors: List[float]):
        """Generate average moves comparison bar chart."""
        fig, ax = plt.subplots(figsize=IEEE_FIGSIZE, dpi=IEEE_DPI)
        
        display_names = [m.replace("Agent4 vs ", "vs ").replace("Ablated-", "Only-") for m in matchups]
        
        x_pos = np.arange(len(matchups))
        colors = ['#2ca02c' if 'Ablated' not in m else '#d62728' for m in matchups]
        
        bars = ax.bar(x_pos, avg_moves, yerr=errors, capsize=5, color=colors,
                      edgecolor='black', linewidth=1.2, alpha=0.8)
        
        ax.set_xlabel('Opponent/Configuration', fontweight='bold')
        ax.set_ylabel('Average Moves per Game', fontweight='bold')
        ax.set_title('Agent4 Efficiency: Average Moves per Game', fontweight='bold', pad=20)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(display_names, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels on bars
        for bar, moves in zip(bars, avg_moves):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{moves:.1f}', ha='center', va='bottom', fontsize=9)
        
        # Add note about lower being better
        ax.text(0.02, 0.98, 'Note: Lower is better (more efficient)', 
               transform=ax.transAxes, fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        fig.savefig(self.output_dir / "fig_avg_moves.png", dpi=IEEE_DPI, bbox_inches='tight')
        fig.savefig(self.output_dir / "fig_avg_moves.pdf", bbox_inches='tight')
        plt.close(fig)
        logger.info("  ✓ Average moves comparison saved")
    
    def _plot_ablation_impact(self):
        """Generate ablation study impact visualization."""
        # Extract ablation-specific data
        ablation_data = {}
        for matchup_name, stats in self.summary_stats.items():
            if 'Ablated-' in matchup_name:
                strategy = matchup_name.replace('Agent4 vs Ablated-', '')
                ablation_data[strategy] = stats['win_rate']
        
        if not ablation_data:
            logger.warning("  ! No ablation data found, skipping ablation impact plot")
            return
        
        fig, ax = plt.subplots(figsize=(8, 6), dpi=IEEE_DPI)
        
        strategies = list(ablation_data.keys())
        win_rates = list(ablation_data.values())
        
        y_pos = np.arange(len(strategies))
        
        bars = ax.barh(y_pos, win_rates, color='#9467bd', edgecolor='black', 
                       linewidth=1.2, alpha=0.8)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(strategies)
        ax.set_xlabel('Win Rate (%)', fontweight='bold')
        ax.set_ylabel('Isolated Strategy', fontweight='bold')
        ax.set_title('Ablation Study: Individual Strategy Contributions', fontweight='bold', pad=20)
        ax.set_xlim(0, 105)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.axvline(x=50, color='red', linestyle='--', linewidth=1, alpha=0.5, label='50% baseline')
        
        # Add value labels
        for bar, rate in zip(bars, win_rates):
            width = bar.get_width()
            ax.text(width + 1, bar.get_y() + bar.get_height()/2.,
                   f'{rate:.1f}%', ha='left', va='center', fontsize=10)
        
        ax.legend(loc='lower right')
        plt.tight_layout()
        
        fig.savefig(self.output_dir / "fig_ablation_impact.png", dpi=IEEE_DPI, bbox_inches='tight')
        fig.savefig(self.output_dir / "fig_ablation_impact.pdf", bbox_inches='tight')
        plt.close(fig)
        logger.info("  ✓ Ablation impact visualization saved")
    
    def _plot_efficiency_scatter(self, matchups: List[str], win_rates: List[float], 
                                  avg_moves: List[float]):
        """Generate scatter plot of efficiency vs win rate."""
        fig, ax = plt.subplots(figsize=IEEE_FIGSIZE, dpi=IEEE_DPI)
        
        # Categorize matchups
        colors = []
        markers = []
        for m in matchups:
            if 'Agent3' in m:
                colors.append('#1f77b4')
                markers.append('o')
            elif 'Agent2' in m:
                colors.append('#2ca02c')
                markers.append('s')
            else:  # Ablated
                colors.append('#ff7f0e')
                markers.append('^')
        
        # Plot each point individually for different markers
        for i, (x, y, c, mark) in enumerate(zip(win_rates, avg_moves, colors, markers)):
            label = matchups[i].replace("Agent4 vs ", "").replace("Ablated-", "Only-")
            ax.scatter(x, y, c=c, marker=mark, s=150, alpha=0.7, 
                      edgecolors='black', linewidth=1.5, label=label)
        
        ax.set_xlabel('Win Rate (%)', fontweight='bold')
        ax.set_ylabel('Average Moves per Game', fontweight='bold')
        ax.set_title('Agent4 Performance: Win Rate vs Efficiency', fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add quadrant lines
        ax.axvline(x=50, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        median_moves = np.median(avg_moves)
        ax.axhline(y=median_moves, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        
        # Add quadrant labels
        ax.text(75, ax.get_ylim()[1] * 0.95, 'High Win Rate\nHigh Moves', 
               ha='center', va='top', fontsize=9, style='italic', alpha=0.6)
        ax.text(75, ax.get_ylim()[0] * 1.05, 'High Win Rate\nLow Moves (Optimal)', 
               ha='center', va='bottom', fontsize=9, style='italic', alpha=0.6,
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
        
        ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
        plt.tight_layout()
        
        fig.savefig(self.output_dir / "fig_scatter_efficiency.png", dpi=IEEE_DPI, bbox_inches='tight')
        fig.savefig(self.output_dir / "fig_scatter_efficiency.pdf", bbox_inches='tight')
        plt.close(fig)
        logger.info("  ✓ Efficiency scatter plot saved")
    
    def _plot_performance_heatmap(self):
        """Generate combined performance heatmap."""
        # Prepare data matrix
        matchups = list(self.summary_stats.keys())
        display_names = [m.replace("Agent4 vs ", "").replace("Ablated-", "Only-") 
                        for m in matchups]
        
        metrics = ['Win Rate (%)', 'Avg Moves', 'Std Dev Moves']
        data_matrix = np.zeros((len(matchups), len(metrics)))
        
        for i, matchup in enumerate(matchups):
            stats = self.summary_stats[matchup]
            data_matrix[i, 0] = stats['win_rate']
            data_matrix[i, 1] = stats['avg_moves_all_games']
            data_matrix[i, 2] = stats['std_moves_all_games']
        
        # Normalize each column to [0, 1] for better visualization
        # For moves and std dev, lower is better, so invert
        normalized_matrix = np.zeros_like(data_matrix)
        normalized_matrix[:, 0] = data_matrix[:, 0] / 100  # Win rate
        normalized_matrix[:, 1] = 1 - (data_matrix[:, 1] / data_matrix[:, 1].max())  # Inverted moves
        normalized_matrix[:, 2] = 1 - (data_matrix[:, 2] / data_matrix[:, 2].max())  # Inverted std
        
        fig, ax = plt.subplots(figsize=(8, 10), dpi=IEEE_DPI)
        
        im = ax.imshow(normalized_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        
        # Set ticks
        ax.set_xticks(np.arange(len(metrics)))
        ax.set_yticks(np.arange(len(matchups)))
        ax.set_xticklabels(metrics)
        ax.set_yticklabels(display_names)
        
        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Normalized Performance (Higher is Better)', rotation=-90, va="bottom", fontweight='bold')
        
        # Add text annotations with actual values
        for i in range(len(matchups)):
            for j in range(len(metrics)):
                text = ax.text(j, i, f'{data_matrix[i, j]:.1f}',
                             ha="center", va="center", color="black", fontsize=9,
                             weight='bold')
        
        ax.set_title('Agent4 Performance Heatmap\n(Green = Better Performance)', 
                    fontweight='bold', pad=20)
        fig.tight_layout()
        
        fig.savefig(self.output_dir / "fig_heatmap_performance.png", dpi=IEEE_DPI, bbox_inches='tight')
        fig.savefig(self.output_dir / "fig_heatmap_performance.pdf", bbox_inches='tight')
        plt.close(fig)
        logger.info("  ✓ Performance heatmap saved")
    
    def perform_statistical_tests(self):
        """Perform statistical significance tests between matchups."""
        logger.info("\nPerforming statistical significance tests...")
        
        df = pd.DataFrame(self.detailed_results)
        
        # Get move data for each matchup
        matchup_moves = {}
        for matchup in df['matchup'].unique():
            matchup_data = df[df['matchup'] == matchup]
            matchup_moves[matchup] = matchup_data['moves_per_player'].values
        
        # Perform pairwise t-tests
        matchup_list = list(matchup_moves.keys())
        significance_results = {}
        
        for i in range(len(matchup_list)):
            for j in range(i + 1, len(matchup_list)):
                matchup1 = matchup_list[i]
                matchup2 = matchup_list[j]
                
                moves1 = matchup_moves[matchup1]
                moves2 = matchup_moves[matchup2]
                
                # Two-sample t-test
                t_stat, p_value = stats.ttest_ind(moves1, moves2)
                
                comparison_key = f"{matchup1} vs {matchup2}"
                significance_results[comparison_key] = {
                    't_statistic': float(t_stat),
                    'p_value': float(p_value),
                    'significant_at_0.05': bool(p_value < 0.05),
                    'significant_at_0.01': bool(p_value < 0.01)
                }
        
        # Save significance test results
        sig_path = self.output_dir / "statistical_significance.json"
        with open(sig_path, 'w') as f:
            json.dump(significance_results, f, indent=2)
        
        logger.info(f"Statistical tests saved to: {sig_path}")
        
        return significance_results


def main():
    """Main execution function."""
    logger.info("╔" + "═" * 78 + "╗")
    logger.info("║" + " " * 20 + "Agent4 Comprehensive Benchmark" + " " * 28 + "║")
    logger.info("╚" + "═" * 78 + "╝")
    
    # Initialize benchmark runner
    runner = BenchmarkRunner()
    
    # Run all benchmarks
    runner.run_all_benchmarks()
    
    # Compute statistics
    runner.compute_statistics()
    
    # Perform statistical tests
    runner.perform_statistical_tests()
    
    # Save results
    runner.save_results()
    
    # Generate visualizations
    runner.generate_visualizations()
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ Benchmark complete!")
    logger.info(f"📁 Results saved to: {runner.output_dir}")
    logger.info("=" * 80)
    
    # Print summary to console
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    for matchup_name, stats in runner.summary_stats.items():
        print(f"\n{matchup_name}:")
        print(f"  Win Rate: {stats['win_rate']:.2f}% (±{stats['win_rate_se']:.2f}%)")
        print(f"  Avg Moves: {stats['avg_moves_all_games']:.2f} (±{stats['ci_95_moves']:.2f})")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

