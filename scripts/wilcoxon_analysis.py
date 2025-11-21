#!/usr/bin/env python3
"""
Wilcoxon Signed-Rank Test Analysis for Agent4 Full vs Ablations
Analyzes statistical significance of performance differences.
"""

import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
from pathlib import Path
import logging
from datetime import datetime

# Setup logging
log_dir = Path(__file__).parent.parent / "logs"
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"wilcoxon_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def analyze_matchup(df, agent1_name, agent2_name):
    """
    Perform Wilcoxon signed-rank test for a specific matchup.
    
    Args:
        df: DataFrame with game results
        agent1_name: Name of first agent (AIAgent4-Full)
        agent2_name: Name of second agent to compare against
        
    Returns:
        dict: Statistics and test results
    """
    try:
        # Filter data for this specific matchup
        matchup_df = df[df['agent2_name'] == agent2_name].copy()
        
        if len(matchup_df) == 0:
            logger.warning(f"No data found for matchup: {agent1_name} vs {agent2_name}")
            return None
        
        # Calculate wins for each agent
        agent1_wins = (matchup_df['winner'] == agent1_name).sum()
        agent2_wins = (matchup_df['winner'] == agent2_name).sum()
        total_games = len(matchup_df)
        
        # Win rates
        agent1_win_rate = agent1_wins / total_games * 100
        agent2_win_rate = agent2_wins / total_games * 100
        
        # Create paired data: 1 for agent1 win, 0 for agent2 win
        # This allows us to use Wilcoxon signed-rank test
        agent1_outcomes = (matchup_df['winner'] == agent1_name).astype(int)
        agent2_outcomes = (matchup_df['winner'] == agent2_name).astype(int)
        
        # Wilcoxon signed-rank test on win outcomes
        # We're testing if the median difference is zero
        differences = agent1_outcomes - agent2_outcomes
        
        # Remove zero differences (ties) for the test
        non_zero_diffs = differences[differences != 0]
        
        if len(non_zero_diffs) < 5:
            logger.warning(f"Too few non-zero differences ({len(non_zero_diffs)}) for {agent2_name}")
            wilcoxon_stat = None
            p_value = None
        else:
            try:
                # Perform the test
                wilcoxon_result = wilcoxon(non_zero_diffs, alternative='two-sided')
                wilcoxon_stat = wilcoxon_result.statistic
                p_value = wilcoxon_result.pvalue
            except Exception as e:
                logger.error(f"Error in Wilcoxon test for {agent2_name}: {str(e)}")
                wilcoxon_stat = None
                p_value = None
        
        # Also analyze moves to win (efficiency)
        agent1_moves = matchup_df[matchup_df['winner'] == agent1_name]['moves_to_win']
        agent2_moves = matchup_df[matchup_df['winner'] == agent2_name]['moves_to_win']
        
        # Average moves to win
        avg_agent1_moves = agent1_moves.mean() if len(agent1_moves) > 0 else None
        avg_agent2_moves = agent2_moves.mean() if len(agent2_moves) > 0 else None
        
        results = {
            'opponent': agent2_name,
            'total_games': total_games,
            'agent1_wins': agent1_wins,
            'agent2_wins': agent2_wins,
            'agent1_win_rate': agent1_win_rate,
            'agent2_win_rate': agent2_win_rate,
            'wilcoxon_statistic': wilcoxon_stat,
            'p_value': p_value,
            'significant_at_0.05': p_value < 0.05 if p_value is not None else None,
            'significant_at_0.01': p_value < 0.01 if p_value is not None else None,
            'avg_moves_agent1': avg_agent1_moves,
            'avg_moves_agent2': avg_agent2_moves,
            'non_zero_differences': len(non_zero_diffs)
        }
        
        logger.info(f"Completed analysis for {agent1_name} vs {agent2_name}")
        return results
        
    except Exception as e:
        logger.error(f"Error analyzing matchup {agent1_name} vs {agent2_name}: {str(e)}")
        return None


def main():
    """Main analysis function."""
    try:
        logger.info("=" * 80)
        logger.info("Starting Wilcoxon Signed-Rank Test Analysis")
        logger.info("=" * 80)
        
        # Load data
        data_path = Path(__file__).parent.parent / "results" / "ablation_study_20251020_171708" / "agent4_full_vs_others.csv"
        
        if not data_path.exists():
            logger.error(f"Data file not found: {data_path}")
            return
        
        logger.info(f"Loading data from: {data_path}")
        df = pd.read_csv(data_path)
        logger.info(f"Loaded {len(df)} game records")
        
        # Get unique agent names
        agent1_name = df['agent1_name'].iloc[0]  # Should be AIAgent4-Full
        opponents = df['agent2_name'].unique()
        
        logger.info(f"Agent 1: {agent1_name}")
        logger.info(f"Number of opponents: {len(opponents)}")
        logger.info(f"Opponents: {', '.join(opponents)}")
        
        # Analyze each matchup
        results = []
        for opponent in opponents:
            logger.info(f"\nAnalyzing: {agent1_name} vs {opponent}")
            result = analyze_matchup(df, agent1_name, opponent)
            if result:
                results.append(result)
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Sort by p-value
        results_df = results_df.sort_values('p_value', na_position='last')
        
        # Save results
        output_path = Path(__file__).parent.parent / "results" / "ablation_study_20251020_171708" / "wilcoxon_test_results.csv"
        results_df.to_csv(output_path, index=False)
        logger.info(f"\nResults saved to: {output_path}")
        
        # Print summary
        print("\n" + "=" * 100)
        print("WILCOXON SIGNED-RANK TEST RESULTS")
        print("=" * 100)
        print(f"\nTesting: {agent1_name} vs Various Ablations")
        print(f"Number of matchups analyzed: {len(results_df)}")
        print(f"\nNull Hypothesis (H0): No difference in performance between agents")
        print(f"Alternative Hypothesis (H1): Significant difference in performance\n")
        
        print("-" * 100)
        print(f"{'Opponent':<30} {'Win Rate':<15} {'P-Value':<12} {'Significant':<15} {'Effect':<15}")
        print("-" * 100)
        
        for _, row in results_df.iterrows():
            opponent_short = row['opponent'].replace('AIAgent4-', '')
            win_rate_str = f"{row['agent1_win_rate']:.1f}% vs {row['agent2_win_rate']:.1f}%"
            
            if row['p_value'] is not None:
                p_val_str = f"{row['p_value']:.6f}"
                
                if row['significant_at_0.01']:
                    sig_str = "Yes (p<0.01)"
                elif row['significant_at_0.05']:
                    sig_str = "Yes (p<0.05)"
                else:
                    sig_str = "No"
                
                # Determine effect direction
                if row['agent1_win_rate'] > row['agent2_win_rate']:
                    effect = "Full > Ablation"
                elif row['agent1_win_rate'] < row['agent2_win_rate']:
                    effect = "Full < Ablation"
                else:
                    effect = "Tied"
            else:
                p_val_str = "N/A"
                sig_str = "N/A"
                effect = "N/A"
            
            print(f"{opponent_short:<30} {win_rate_str:<15} {p_val_str:<12} {sig_str:<15} {effect:<15}")
        
        print("-" * 100)
        
        # Summary statistics
        significant_05 = results_df['significant_at_0.05'].sum()
        significant_01 = results_df['significant_at_0.01'].sum()
        
        print(f"\nSummary:")
        print(f"  - Matchups significant at α=0.05: {significant_05}/{len(results_df)}")
        print(f"  - Matchups significant at α=0.01: {significant_01}/{len(results_df)}")
        print(f"  - Average win rate for {agent1_name}: {results_df['agent1_win_rate'].mean():.2f}%")
        
        # Efficiency analysis
        print("\n" + "=" * 100)
        print("EFFICIENCY ANALYSIS (Average Moves to Win)")
        print("=" * 100)
        print(f"{'Opponent':<30} {'Full Avg Moves':<20} {'Ablation Avg Moves':<20} {'Difference':<15}")
        print("-" * 100)
        
        for _, row in results_df.iterrows():
            opponent_short = row['opponent'].replace('AIAgent4-', '')
            
            if row['avg_moves_agent1'] is not None and row['avg_moves_agent2'] is not None:
                diff = row['avg_moves_agent1'] - row['avg_moves_agent2']
                diff_str = f"{diff:+.2f}"
                print(f"{opponent_short:<30} {row['avg_moves_agent1']:<20.2f} {row['avg_moves_agent2']:<20.2f} {diff_str:<15}")
            else:
                print(f"{opponent_short:<30} {'N/A':<20} {'N/A':<20} {'N/A':<15}")
        
        print("-" * 100)
        print("\nNote: Negative difference means Full agent wins faster (more efficient)")
        
        # Save detailed report
        report_path = Path(__file__).parent.parent / "results" / "ablation_study_20251020_171708" / "wilcoxon_test_report.txt"
        with open(report_path, 'w') as f:
            f.write("=" * 100 + "\n")
            f.write("WILCOXON SIGNED-RANK TEST RESULTS\n")
            f.write("=" * 100 + "\n\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Data File: {data_path}\n")
            f.write(f"Total Games Analyzed: {len(df)}\n\n")
            
            f.write("TEST DESCRIPTION:\n")
            f.write("The Wilcoxon signed-rank test is a non-parametric statistical test used to\n")
            f.write("compare paired samples. It tests whether the median difference between pairs\n")
            f.write("is zero, making it suitable for comparing win rates between agents.\n\n")
            
            f.write("INTERPRETATION:\n")
            f.write("- p-value < 0.01: Very strong evidence of difference (highly significant)\n")
            f.write("- p-value < 0.05: Strong evidence of difference (significant)\n")
            f.write("- p-value >= 0.05: Insufficient evidence to reject null hypothesis\n\n")
            
            f.write("RESULTS:\n")
            f.write("-" * 100 + "\n")
            
            for _, row in results_df.iterrows():
                f.write(f"\nOpponent: {row['opponent']}\n")
                f.write(f"  Total Games: {row['total_games']}\n")
                f.write(f"  {agent1_name} Wins: {row['agent1_wins']} ({row['agent1_win_rate']:.2f}%)\n")
                f.write(f"  {row['opponent']} Wins: {row['agent2_wins']} ({row['agent2_win_rate']:.2f}%)\n")
                f.write(f"  Wilcoxon Statistic: {row['wilcoxon_statistic']}\n")
                p_val_str = f"{row['p_value']:.6f}" if row['p_value'] is not None else 'N/A'
                f.write(f"  P-Value: {p_val_str}\n")
                f.write(f"  Significant at α=0.05: {row['significant_at_0.05']}\n")
                f.write(f"  Significant at α=0.01: {row['significant_at_0.01']}\n")
                
                if row['avg_moves_agent1'] is not None:
                    f.write(f"  Average Moves to Win ({agent1_name}): {row['avg_moves_agent1']:.2f}\n")
                if row['avg_moves_agent2'] is not None:
                    f.write(f"  Average Moves to Win ({row['opponent']}): {row['avg_moves_agent2']:.2f}\n")
                
                f.write("-" * 100 + "\n")
        
        logger.info(f"Detailed report saved to: {report_path}")
        
        logger.info("\n" + "=" * 80)
        logger.info("Analysis Complete!")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Fatal error in main: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

