#!/usr/bin/env python3
"""
Wilcoxon Signed-Rank Test for AI_agent4 vs AI_Agent3
Analyzes the 1000 game matchup from the batch simulation.
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
log_file = log_dir / f"wilcoxon_agent4_vs_agent3_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def analyze_recent_matchup():
    """
    Analyze the most recent AI_agent4 vs AI_Agent3 matchup from game data.
    """
    try:
        logger.info("=" * 80)
        logger.info("Wilcoxon Signed-Rank Test: AI_agent4 vs AI_Agent3")
        logger.info("=" * 80)
        
        # Load all moves data
        data_dir = Path(__file__).parent.parent / "data"
        all_moves_file = data_dir / "all_moves.csv"
        
        if not all_moves_file.exists():
            logger.error(f"Data file not found: {all_moves_file}")
            return
        
        logger.info(f"Loading data from: {all_moves_file}")
        moves_df = pd.read_csv(all_moves_file)
        logger.info(f"Loaded {len(moves_df)} move records")
        
        # Group by game to identify AIAgent4 vs AIAgent3 games
        games_list = []
        for game_id in moves_df['game_id'].unique():
            game_moves = moves_df[moves_df['game_id'] == game_id]
            player_types = game_moves['player_type'].unique()
            
            # Check if this is an AIAgent4 vs AIAgent3 game
            if 'AIAgent4' in player_types and 'AIAgent3' in player_types:
                # Determine winner (last move's player is the winner)
                last_move = game_moves.iloc[-1]
                winner_type = last_move['player_type']
                
                # Count moves (divide by 2 since each turn has 2 moves)
                total_moves = len(game_moves)
                
                games_list.append({
                    'game_id': game_id,
                    'winner': winner_type,
                    'total_moves': total_moves,
                    'moves_to_win': total_moves
                })
        
        if not games_list:
            logger.error("No AIAgent4 vs AIAgent3 games found in data")
            return
        
        games_df = pd.DataFrame(games_list)
        logger.info(f"Found {len(games_df)} AIAgent4 vs AIAgent3 games")
        
        # Calculate statistics
        agent4_wins = (games_df['winner'] == 'AIAgent4').sum()
        agent3_wins = (games_df['winner'] == 'AIAgent3').sum()
        total_games = len(games_df)
        
        agent4_win_rate = agent4_wins / total_games * 100
        agent3_win_rate = agent3_wins / total_games * 100
        
        logger.info(f"AIAgent4 wins: {agent4_wins} ({agent4_win_rate:.2f}%)")
        logger.info(f"AIAgent3 wins: {agent3_wins} ({agent3_win_rate:.2f}%)")
        
        # Create binary outcome arrays for Wilcoxon test
        # 1 for AIAgent4 win, -1 for AIAgent3 win
        outcomes = games_df['winner'].apply(lambda x: 1 if x == 'AIAgent4' else -1).values
        
        # Wilcoxon signed-rank test
        # H0: median difference is 0 (no difference in performance)
        # H1: median difference is not 0 (significant difference)
        logger.info("\nPerforming Wilcoxon signed-rank test...")
        
        # We compare against 0 (no preference)
        wilcoxon_result = wilcoxon(outcomes, alternative='two-sided')
        
        test_statistic = wilcoxon_result.statistic
        p_value = wilcoxon_result.pvalue
        
        # Calculate effect size (r = Z / sqrt(N))
        from scipy.stats import norm
        z_score = norm.ppf(1 - p_value/2)  # Two-tailed
        effect_size = z_score / np.sqrt(total_games)
        
        # Average moves analysis
        agent4_moves = games_df[games_df['winner'] == 'AIAgent4']['moves_to_win']
        agent3_moves = games_df[games_df['winner'] == 'AIAgent3']['moves_to_win']
        
        avg_agent4_moves = agent4_moves.mean() if len(agent4_moves) > 0 else 0
        avg_agent3_moves = agent3_moves.mean() if len(agent3_moves) > 0 else 0
        
        # Print results
        print("\n" + "=" * 80)
        print("WILCOXON SIGNED-RANK TEST RESULTS")
        print("=" * 80)
        print(f"\nMatchup: AIAgent4 vs AIAgent3")
        print(f"Total Games: {total_games}")
        print(f"\nPerformance Statistics:")
        print(f"  AIAgent4: {agent4_wins} wins ({agent4_win_rate:.2f}%)")
        print(f"  AIAgent3: {agent3_wins} wins ({agent3_win_rate:.2f}%)")
        print(f"  Win Rate Difference: {agent4_win_rate - agent3_win_rate:.2f}%")
        
        print(f"\nEfficiency (Average Moves to Win):")
        print(f"  AIAgent4: {avg_agent4_moves:.2f} moves")
        print(f"  AIAgent3: {avg_agent3_moves:.2f} moves")
        
        print(f"\nWilcoxon Signed-Rank Test:")
        print(f"  Test Statistic (W): {test_statistic:.2f}")
        print(f"  P-Value: {p_value:.6f}")
        print(f"  Effect Size (r): {abs(effect_size):.4f}")
        
        print(f"\nStatistical Significance:")
        if p_value < 0.001:
            significance = "*** HIGHLY SIGNIFICANT (p < 0.001)"
            interpretation = "Extremely strong evidence of performance difference"
        elif p_value < 0.01:
            significance = "** VERY SIGNIFICANT (p < 0.01)"
            interpretation = "Very strong evidence of performance difference"
        elif p_value < 0.05:
            significance = "* SIGNIFICANT (p < 0.05)"
            interpretation = "Strong evidence of performance difference"
        else:
            significance = "NOT SIGNIFICANT (p >= 0.05)"
            interpretation = "Insufficient evidence to conclude performance difference"
        
        print(f"  Result: {significance}")
        print(f"  Interpretation: {interpretation}")
        
        # Effect size interpretation
        print(f"\nEffect Size Interpretation:")
        if abs(effect_size) < 0.1:
            effect_desc = "Negligible"
        elif abs(effect_size) < 0.3:
            effect_desc = "Small"
        elif abs(effect_size) < 0.5:
            effect_desc = "Medium"
        else:
            effect_desc = "Large"
        print(f"  Effect Size: {effect_desc} (r = {abs(effect_size):.4f})")
        
        print("\n" + "=" * 80)
        
        # Save detailed report
        report_path = Path(__file__).parent.parent / "logs" / f"wilcoxon_agent4_vs_agent3_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("WILCOXON SIGNED-RANK TEST RESULTS\n")
            f.write("AIAgent4 vs AIAgent3 Matchup Analysis\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Data Source: {all_moves_file}\n")
            f.write(f"Total Games Analyzed: {total_games}\n\n")
            
            f.write("PERFORMANCE STATISTICS:\n")
            f.write(f"  AIAgent4 Wins: {agent4_wins} ({agent4_win_rate:.2f}%)\n")
            f.write(f"  AIAgent3 Wins: {agent3_wins} ({agent3_win_rate:.2f}%)\n")
            f.write(f"  Win Rate Difference: {agent4_win_rate - agent3_win_rate:.2f}%\n\n")
            
            f.write("EFFICIENCY METRICS:\n")
            f.write(f"  AIAgent4 Avg Moves to Win: {avg_agent4_moves:.2f}\n")
            f.write(f"  AIAgent3 Avg Moves to Win: {avg_agent3_moves:.2f}\n\n")
            
            f.write("WILCOXON SIGNED-RANK TEST:\n")
            f.write(f"  Test Statistic (W): {test_statistic:.2f}\n")
            f.write(f"  P-Value: {p_value:.6f}\n")
            f.write(f"  Significance: {significance}\n")
            f.write(f"  Effect Size (r): {abs(effect_size):.4f} ({effect_desc})\n\n")
            
            f.write("INTERPRETATION:\n")
            f.write(f"  {interpretation}\n\n")
            
            f.write("TEST DESCRIPTION:\n")
            f.write("The Wilcoxon signed-rank test is a non-parametric statistical test that\n")
            f.write("assesses whether the median of differences differs from zero. It is used here\n")
            f.write("to determine if there is a statistically significant difference in performance\n")
            f.write("between AIAgent4 and AIAgent3.\n\n")
            
            f.write("SIGNIFICANCE LEVELS:\n")
            f.write("  p < 0.001: Highly significant (***)\n")
            f.write("  p < 0.01:  Very significant (**)\n")
            f.write("  p < 0.05:  Significant (*)\n")
            f.write("  p >= 0.05: Not significant\n\n")
            
            f.write("EFFECT SIZE INTERPRETATION:\n")
            f.write("  |r| < 0.1:  Negligible effect\n")
            f.write("  |r| < 0.3:  Small effect\n")
            f.write("  |r| < 0.5:  Medium effect\n")
            f.write("  |r| >= 0.5: Large effect\n")
        
        logger.info(f"\nDetailed report saved to: {report_path}")
        
        # Also print the key result prominently
        print("\n" + "=" * 80)
        print("KEY FINDING")
        print("=" * 80)
        print(f"P-Value: {p_value:.6f}")
        print(f"Conclusion: AIAgent4 {'SIGNIFICANTLY OUTPERFORMS' if p_value < 0.05 and agent4_win_rate > agent3_win_rate else 'DOES NOT SIGNIFICANTLY OUTPERFORM'} AIAgent3")
        print("=" * 80 + "\n")
        
        logger.info("Analysis complete!")
        
    except Exception as e:
        logger.error(f"Error in analysis: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    analyze_recent_matchup()



