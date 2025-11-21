#!/usr/bin/env python3
"""
P-value Analysis Script for Battleship AI Agents

Analyzes Agent4 vs Agent3 performance data:
1. Parses game logs to extract results
2. Performs Wilcoxon signed-rank test at intervals
3. Creates p-value convergence plots
4. Creates box/whisker plots for performance comparison

Author: Automated Analysis
Date: November 2025
"""

import re
import os
import sys
import logging
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/p_value_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


class GameDataParser:
    """Parser for extracting game results from log files."""
    
    def __init__(self, log_path: str):
        self.log_path = log_path
        self.games = []
        
    def parse_log(self) -> pd.DataFrame:
        """
        Parse the log file to extract game results.
        
        Returns:
            DataFrame with columns: game_num, winner, agent4_wins, moves
        """
        logger.info(f"Parsing log file: {self.log_path}")
        
        # Pattern to match game over lines
        # Example: "2025-10-08 16:45:23,355 - BattleshipDashboard - INFO - Game over: AI_agent4-0 wins in 75 moves, 8.9s"
        pattern = re.compile(
            r'Game over: (AI_agent4-\d+|AI_Agent3-\d+) wins in (\d+) moves'
        )
        
        with open(self.log_path, 'r') as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    winner_raw = match.group(1)
                    moves = int(match.group(2))
                    
                    # Determine winner
                    if 'AI_agent4' in winner_raw:
                        winner = 'Agent4'
                        agent4_wins = 1
                    else:
                        winner = 'Agent3'
                        agent4_wins = 0
                    
                    self.games.append({
                        'game_num': len(self.games) + 1,
                        'winner': winner,
                        'agent4_wins': agent4_wins,
                        'moves': moves
                    })
        
        df = pd.DataFrame(self.games)
        logger.info(f"Parsed {len(df)} games from log file")
        logger.info(f"Agent4 wins: {df['agent4_wins'].sum()}, Agent3 wins: {len(df) - df['agent4_wins'].sum()}")
        
        return df


class WilcoxonAnalyzer:
    """Performs Wilcoxon signed-rank test analysis at intervals."""
    
    def __init__(self, game_data: pd.DataFrame):
        self.game_data = game_data
        self.results = []
        
    def analyze_at_intervals(self, interval: int = 50, max_games: int = 1000) -> pd.DataFrame:
        """
        Perform Wilcoxon test at regular intervals.
        
        For Wilcoxon signed-rank test, we need paired data. Since each game has one winner,
        we'll compare the distribution of moves for Agent4 wins vs Agent3 wins.
        
        Args:
            interval: Number of games between each analysis point
            max_games: Maximum number of games to analyze (default: 1000)
            
        Returns:
            DataFrame with columns: n_games, agent4_wins, agent3_wins, 
                                   agent4_avg_moves, agent3_avg_moves, 
                                   p_value, significant
        """
        logger.info(f"Performing Wilcoxon analysis at {interval}-game intervals (max: {max_games} games)")
        
        # Limit to max_games if specified
        max_games = min(max_games, len(self.game_data))
        
        for n in range(interval, max_games + 1, interval):
            subset = self.game_data.iloc[:n]
            
            # Separate moves by winner
            agent4_moves = subset[subset['winner'] == 'Agent4']['moves'].values
            agent3_moves = subset[subset['winner'] == 'Agent3']['moves'].values
            
            # Statistics
            agent4_win_count = len(agent4_moves)
            agent3_win_count = len(agent3_moves)
            agent4_avg = agent4_moves.mean() if len(agent4_moves) > 0 else np.nan
            agent3_avg = agent3_moves.mean() if len(agent3_moves) > 0 else np.nan
            
            # Wilcoxon-Mann-Whitney U test (for independent samples)
            # Testing if Agent4 wins faster (fewer moves)
            if len(agent4_moves) > 0 and len(agent3_moves) > 0:
                try:
                    # Mann-Whitney U test (one-sided: Agent4 < Agent3)
                    statistic, p_value = stats.mannwhitneyu(
                        agent4_moves, 
                        agent3_moves, 
                        alternative='less'
                    )
                except Exception as e:
                    logger.warning(f"Could not compute test for n={n}: {e}")
                    p_value = np.nan
            else:
                p_value = np.nan
            
            significant = p_value < 0.05 if not np.isnan(p_value) else False
            
            self.results.append({
                'n_games': n,
                'agent4_wins': agent4_win_count,
                'agent3_wins': agent3_win_count,
                'agent4_avg_moves': agent4_avg,
                'agent3_avg_moves': agent3_avg,
                'p_value': p_value,
                'significant': significant
            })
            
            agent4_avg_str = f"{agent4_avg:.2f}" if not np.isnan(agent4_avg) else "N/A"
            agent3_avg_str = f"{agent3_avg:.2f}" if not np.isnan(agent3_avg) else "N/A"
            p_value_str = f"{p_value:.6f}" if not np.isnan(p_value) else "N/A"
            
            logger.info(f"n={n}: Agent4={agent4_win_count} wins ({agent4_avg_str} moves), "
                       f"Agent3={agent3_win_count} wins ({agent3_avg_str} moves), "
                       f"p={p_value_str}")
        
        results_df = pd.DataFrame(self.results)
        return results_df


class VisualizationGenerator:
    """Generates all required visualizations."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def plot_pvalue_convergence(self, wilcoxon_results: pd.DataFrame, 
                                output_file: str = 'p_value_convergence.png'):
        """
        Create p-value convergence plot.
        
        Args:
            wilcoxon_results: DataFrame with n_games and p_value columns
            output_file: Output filename
        """
        logger.info("Creating p-value convergence plot")
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Plot p-values
        n_games = wilcoxon_results['n_games'].values
        p_values = wilcoxon_results['p_value'].values
        
        # Remove NaN values for plotting
        valid_mask = ~np.isnan(p_values)
        n_games_valid = n_games[valid_mask]
        p_values_valid = p_values[valid_mask]
        
        ax.plot(n_games_valid, p_values_valid, 'o-', linewidth=2, 
                markersize=8, color='#2E86AB', label='P-value')
        
        # Add significance threshold
        ax.axhline(y=0.05, color='red', linestyle='--', linewidth=2, 
                  label='Significance threshold (p=0.05)', alpha=0.7)
        
        # Find first significant point
        sig_points = wilcoxon_results[wilcoxon_results['significant'] == True]
        if len(sig_points) > 0:
            first_sig = sig_points.iloc[0]
            ax.axvline(x=first_sig['n_games'], color='green', 
                      linestyle=':', linewidth=2, alpha=0.5)
            
            # Position annotation appropriately for linear scale
            y_range = max(p_values_valid) - min(p_values_valid)
            annotation_y = first_sig['p_value'] + y_range * 0.15
            
            ax.annotate(f"Significance achieved\nat {first_sig['n_games']} games\n(p={first_sig['p_value']:.4f})",
                       xy=(first_sig['n_games'], first_sig['p_value']),
                       xytext=(first_sig['n_games'] - 150, annotation_y),
                       arrowprops=dict(arrowstyle='->', color='green', lw=2),
                       fontsize=11, color='green', weight='bold',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                                edgecolor='green', alpha=0.8))
        
        # Linear y-axis with regular intervals
        ax.set_yscale('linear')
        
        # Labels and title
        ax.set_xlabel('Number of Games', fontsize=14, weight='bold')
        ax.set_ylabel('P-value', fontsize=14, weight='bold')
        ax.set_title('Statistical Significance Convergence: Agent4 vs Agent3\n(Mann-Whitney U Test)',
                    fontsize=16, weight='bold', pad=20)
        
        # Set y-axis limits and ticks for better readability
        ax.set_ylim(0, max(0.6, max(p_values_valid) * 1.1))
        ax.set_yticks([0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6])
        
        # Grid
        ax.grid(True, alpha=0.3, which='both')
        ax.legend(fontsize=12, loc='upper right')
        
        # Set x-axis limits
        ax.set_xlim(0, max(n_games) + 50)
        
        plt.tight_layout()
        output_path = self.output_dir / output_file
        plt.savefig(output_path, bbox_inches='tight')
        logger.info(f"Saved p-value convergence plot to {output_path}")
        plt.close()
        
    def plot_agent4_vs_agent3_boxplots(self, game_data: pd.DataFrame,
                                        output_file: str = 'agent4_vs_agent3_boxplots.png'):
        """
        Create box plots for Agent4 vs Agent3 comparison.
        
        Args:
            game_data: DataFrame with game results
            output_file: Output filename
        """
        logger.info("Creating Agent4 vs Agent3 box plots")
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Plot 1: Moves distribution
        ax1 = axes[0]
        
        # Prepare data for box plot
        agent4_moves = game_data[game_data['winner'] == 'Agent4']['moves'].values
        agent3_moves = game_data[game_data['winner'] == 'Agent3']['moves'].values
        
        box_data = [agent4_moves, agent3_moves]
        positions = [1, 2]
        labels = ['Agent4', 'Agent3']
        colors = ['#2E86AB', '#A23B72']
        
        bp = ax1.boxplot(box_data, positions=positions, tick_labels=labels,
                        patch_artist=True, widths=0.6,
                        showmeans=False,
                        boxprops=dict(linewidth=2),
                        whiskerprops=dict(linewidth=2),
                        capprops=dict(linewidth=2),
                        medianprops=dict(linewidth=2.5, color='black'))
        
        # Color the boxes
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax1.set_ylabel('Moves to Win', fontsize=13, weight='bold')
        ax1.set_title('Moves Distribution per Agent\n(When Agent Wins)', 
                     fontsize=14, weight='bold', pad=15)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add statistics text
        stats_text = f"Agent4: μ={agent4_moves.mean():.1f}, σ={agent4_moves.std():.1f}, n={len(agent4_moves)}\n"
        stats_text += f"Agent3: μ={agent3_moves.mean():.1f}, σ={agent3_moves.std():.1f}, n={len(agent3_moves)}"
        ax1.text(0.5, 0.02, stats_text, transform=ax1.transAxes,
                fontsize=10, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Plot 2: Win counts
        ax2 = axes[1]
        
        agent4_wins = game_data['agent4_wins'].sum()
        agent3_wins = len(game_data) - agent4_wins
        total_games = len(game_data)
        
        wins = [agent4_wins, agent3_wins]
        win_percentages = [w / total_games * 100 for w in wins]
        
        bars = ax2.bar(labels, wins, color=colors, alpha=0.7, 
                      edgecolor='black', linewidth=2, width=0.6)
        
        # Add percentage labels on bars
        for bar, pct in zip(bars, win_percentages):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}\n({pct:.1f}%)',
                    ha='center', va='bottom', fontsize=12, weight='bold')
        
        ax2.set_ylabel('Number of Wins', fontsize=13, weight='bold')
        ax2.set_title(f'Win Comparison\n(Total: {total_games} games)', 
                     fontsize=14, weight='bold', pad=15)
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle(f'Agent4 vs Agent3 Head-to-Head Performance ({total_games} Games)',
                    fontsize=16, weight='bold', y=1.02)
        
        plt.tight_layout()
        output_path = self.output_dir / output_file
        plt.savefig(output_path, bbox_inches='tight')
        logger.info(f"Saved Agent4 vs Agent3 box plots to {output_path}")
        plt.close()
        
    def plot_agent4_vs_variants_boxplots(self, csv_path: str,
                                         output_file: str = 'agent4_vs_variants_boxplots.png'):
        """
        Create box plots for Agent4-Full vs its variants.
        
        Args:
            csv_path: Path to CSV file with variant comparison data
            output_file: Output filename
        """
        logger.info("Creating Agent4-Full vs variants box plots")
        
        # Load data
        df = pd.read_csv(csv_path)
        
        # Get unique opponents (excluding Agent3 since we have separate analysis)
        opponents = df['agent2_name'].unique()
        
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        
        # Plot 1: Moves distribution
        ax1 = axes[0]
        
        moves_data = []
        labels = []
        colors_palette = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
        
        for i, opp in enumerate(opponents):
            subset = df[df['agent2_name'] == opp]
            moves = subset['moves_to_win'].values
            moves_data.append(moves)
            
            # Simplify labels
            if 'Agent3' in opp:
                labels.append('Agent3')
            elif 'Neural' in opp:
                labels.append('Neural-Only')
            elif 'Density' in opp:
                labels.append('Density-Only')
            elif 'MonteCarlo' in opp:
                labels.append('Monte Carlo-Only')
            elif 'InfoGain' in opp:
                labels.append('InfoGain-Only')
            else:
                labels.append(opp.replace('AIAgent4-', ''))
        
        positions = list(range(1, len(moves_data) + 1))
        
        bp = ax1.boxplot(moves_data, positions=positions, tick_labels=labels,
                        patch_artist=True, widths=0.6,
                        showmeans=False,
                        boxprops=dict(linewidth=1.5),
                        whiskerprops=dict(linewidth=1.5),
                        capprops=dict(linewidth=1.5),
                        medianprops=dict(linewidth=2, color='black'))
        
        # Color the boxes
        for patch, color in zip(bp['boxes'], colors_palette[:len(moves_data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax1.set_xlabel('Opponent', fontsize=13, weight='bold')
        ax1.set_ylabel('Moves to Win', fontsize=13, weight='bold')
        ax1.set_title('Moves Distribution: Agent4-Full vs Opponents', 
                     fontsize=14, weight='bold', pad=15)
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.tick_params(axis='x', rotation=30)
        
        # Plot 2: Win rates
        ax2 = axes[1]
        
        win_rates = []
        for opp in opponents:
            subset = df[df['agent2_name'] == opp]
            agent4_wins = (subset['winner'] == 'AIAgent4-Full').sum()
            total = len(subset)
            win_rate = (agent4_wins / total) * 100
            win_rates.append(win_rate)
        
        bars = ax2.bar(labels, win_rates, 
                      color=colors_palette[:len(win_rates)], 
                      alpha=0.7, edgecolor='black', linewidth=2, width=0.6)
        
        # Add percentage labels on bars
        for bar, rate in zip(bars, win_rates):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{rate:.1f}%',
                    ha='center', va='bottom', fontsize=11, weight='bold')
        
        # Add 50% reference line
        ax2.axhline(y=50, color='red', linestyle='--', linewidth=2, 
                   alpha=0.5, label='50% (even)')
        
        ax2.set_xlabel('Opponent', fontsize=13, weight='bold')
        ax2.set_ylabel('Agent4-Full Win Rate (%)', fontsize=13, weight='bold')
        ax2.set_title('Win Rates: Agent4-Full vs Opponents', 
                     fontsize=14, weight='bold', pad=15)
        ax2.set_ylim(0, 105)
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.tick_params(axis='x', rotation=30)
        ax2.legend(fontsize=10)
        
        plt.suptitle('Agent4-Full vs Variants Performance Comparison (100 games each)',
                    fontsize=16, weight='bold', y=1.00)
        
        plt.tight_layout()
        output_path = self.output_dir / output_file
        plt.savefig(output_path, bbox_inches='tight')
        logger.info(f"Saved Agent4 vs variants box plots to {output_path}")
        plt.close()


def main():
    """Main execution function."""
    logger.info("=" * 80)
    logger.info("Starting P-value Analysis for Battleship AI Agents")
    logger.info("=" * 80)
    
    # Define paths
    base_dir = Path(__file__).parent.parent
    log_path = base_dir / 'logs' / 'ai_agent3.log'
    csv_path = base_dir / 'results' / 'ablation_study_20251020_171708' / 'agent4_full_vs_others.csv'
    output_dir = base_dir / 'results' / 'p_value'
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Step 1: Parse log file
    logger.info("\nSTEP 1: Parsing game log...")
    parser = GameDataParser(str(log_path))
    game_data = parser.parse_log()
    
    # Step 2: Wilcoxon analysis (limit to 1000 games)
    logger.info("\nSTEP 2: Performing Wilcoxon analysis at intervals...")
    analyzer = WilcoxonAnalyzer(game_data)
    wilcoxon_results = analyzer.analyze_at_intervals(interval=50, max_games=1000)
    
    # Save Wilcoxon results
    wilcoxon_csv = output_dir / 'wilcoxon_results.csv'
    wilcoxon_results.to_csv(wilcoxon_csv, index=False)
    logger.info(f"Saved Wilcoxon results to {wilcoxon_csv}")
    
    # Step 3: Generate visualizations
    logger.info("\nSTEP 3: Generating visualizations...")
    viz = VisualizationGenerator(str(output_dir))
    
    # P-value convergence plot
    viz.plot_pvalue_convergence(wilcoxon_results)
    
    # Agent4 vs Agent3 box plots (limit to 1000 games)
    viz.plot_agent4_vs_agent3_boxplots(game_data.iloc[:1000])
    
    # Agent4 vs variants box plots
    viz.plot_agent4_vs_variants_boxplots(str(csv_path))
    
    # Step 4: Generate summary report
    logger.info("\nSTEP 4: Generating summary report...")
    summary_path = output_dir / 'analysis_summary.txt'
    
    with open(summary_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("P-VALUE ANALYSIS SUMMARY REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        # Use only first 1000 games for summary consistency
        game_data_1000 = game_data.iloc[:1000]
        
        f.write("1. DATA OVERVIEW\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total games analyzed: {len(game_data_1000)}\n")
        f.write(f"Agent4 wins: {game_data_1000['agent4_wins'].sum()} ({game_data_1000['agent4_wins'].sum()/len(game_data_1000)*100:.2f}%)\n")
        f.write(f"Agent3 wins: {len(game_data_1000) - game_data_1000['agent4_wins'].sum()} "
                f"({(len(game_data_1000) - game_data_1000['agent4_wins'].sum())/len(game_data_1000)*100:.2f}%)\n\n")
        
        agent4_moves = game_data_1000[game_data_1000['winner'] == 'Agent4']['moves']
        agent3_moves = game_data_1000[game_data_1000['winner'] == 'Agent3']['moves']
        
        f.write(f"Agent4 moves (when winning): μ={agent4_moves.mean():.2f}, σ={agent4_moves.std():.2f}\n")
        f.write(f"Agent3 moves (when winning): μ={agent3_moves.mean():.2f}, σ={agent3_moves.std():.2f}\n\n")
        
        f.write("2. STATISTICAL SIGNIFICANCE ANALYSIS\n")
        f.write("-" * 80 + "\n")
        
        sig_results = wilcoxon_results[wilcoxon_results['significant'] == True]
        if len(sig_results) > 0:
            first_sig = sig_results.iloc[0]
            f.write(f"Statistical significance (p < 0.05) first achieved at: {first_sig['n_games']} games\n")
            f.write(f"P-value at first significance: {first_sig['p_value']:.6f}\n\n")
        else:
            f.write("Statistical significance (p < 0.05) was not achieved.\n\n")
        
        f.write("P-value progression:\n")
        for _, row in wilcoxon_results.iterrows():
            sig_marker = " ***" if row['significant'] else ""
            f.write(f"  {row['n_games']:4d} games: p = {row['p_value']:.6f}{sig_marker}\n")
        
        f.write("\n3. KEY FINDINGS\n")
        f.write("-" * 80 + "\n")
        
        # Determine findings
        total_agent4_wins = game_data_1000['agent4_wins'].sum()
        total_games = len(game_data_1000)
        win_rate_diff = (total_agent4_wins / total_games) - 0.5
        
        if abs(win_rate_diff) < 0.05:
            f.write("- Agent4 and Agent3 show relatively balanced win rates (within 5%).\n")
        elif win_rate_diff > 0:
            f.write(f"- Agent4 shows superior win rate: {total_agent4_wins/total_games*100:.2f}% vs {(1-total_agent4_wins/total_games)*100:.2f}%\n")
        else:
            f.write(f"- Agent3 shows superior win rate: {(1-total_agent4_wins/total_games)*100:.2f}% vs {total_agent4_wins/total_games*100:.2f}%\n")
        
        if len(sig_results) > 0:
            first_sig_n = sig_results.iloc[0]['n_games']
            f.write(f"- Statistical significance emerges at {first_sig_n} games, suggesting this is the minimum\n")
            f.write(f"  sample size needed to detect performance differences.\n")
        
        f.write(f"- Agent4 average moves when winning: {agent4_moves.mean():.2f}\n")
        f.write(f"- Agent3 average moves when winning: {agent3_moves.mean():.2f}\n")
        
        f.write("\n4. FILES GENERATED\n")
        f.write("-" * 80 + "\n")
        f.write("- p_value_convergence.png: P-value vs number of games\n")
        f.write("- agent4_vs_agent3_boxplots.png: Distribution and win comparison\n")
        f.write("- agent4_vs_variants_boxplots.png: Comparison with ablation variants\n")
        f.write("- wilcoxon_results.csv: Detailed statistical test results\n")
        f.write("- analysis_summary.txt: This summary report\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("Analysis completed successfully.\n")
        f.write("=" * 80 + "\n")
    
    logger.info(f"Saved summary report to {summary_path}")
    
    logger.info("\n" + "=" * 80)
    logger.info("P-value analysis completed successfully!")
    logger.info(f"All results saved to: {output_dir}")
    logger.info("=" * 80)
    
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)

