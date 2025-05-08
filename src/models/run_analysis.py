"""
Script to demonstrate Bayesian player performance analysis.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from src.models.player_performance import PlayerPerformance, HierarchicalPlayerPerformance
from src.visualization.plot_performance import create_performance_report

def run_basic_analysis(player_data: pd.DataFrame) -> None:
    """
    Run basic Bayesian analysis using conjugate prior.
    
    Args:
        player_data: DataFrame with player performance data
    """
    # Initialize model with uniform prior
    model = PlayerPerformance(prior_alpha=1.0, prior_beta=1.0)
    
    # Update with player data
    model.update_posterior(
        goals=player_data['goals'].sum(),
        minutes_played=player_data['minutes_played'].sum()
    )
    
    # Get estimates
    mean = model.get_posterior_mean()
    var = model.get_posterior_variance()
    ci_lower, ci_upper = model.get_credible_interval()
    
    print("\nBasic Bayesian Analysis Results:")
    print(f"Posterior mean (goals per match): {mean:.4f}")
    print(f"Posterior variance: {var:.4f}")
    print(f"95% Credible Interval: [{ci_lower:.4f}, {ci_upper:.4f}]")

def run_hierarchical_analysis(player_data: pd.DataFrame,
                            output_dir: str = "results") -> None:
    """
    Run hierarchical Bayesian analysis using MCMC.
    
    Args:
        player_data: DataFrame with player performance data
        output_dir: Directory to save results
    """
    # Prepare data for hierarchical model
    model_data = {
        'player_ids': player_data['player_api_id'].values,
        'goals': player_data['goals'].values,
        'minutes': player_data['minutes_played'].values
    }
    
    # Calculate league statistics for prior
    league_mean = player_data['goals_per_90'].mean()
    league_std = player_data['goals_per_90'].std()
    
    # Initialize and fit model
    model = HierarchicalPlayerPerformance(league_mean, league_std)
    model.fit(model_data)
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get qualified players (those with sufficient minutes)
    qualified_players = model.get_qualified_players()
    print(f"\nPlayers with sufficient minutes (≥{model.min_minutes}): {len(qualified_players)}")
    print(f"Total players in dataset: {len(player_data['player_api_id'].unique())}")
    
    # Get estimates for each qualified player
    results = []
    for player_id in qualified_players:
        try:
            estimates = model.get_player_estimates(player_id)
            player_name = player_data[player_data['player_api_id'] == player_id]['player_name'].iloc[0]
            minutes_played = player_data[player_data['player_api_id'] == player_id]['minutes_played'].sum()
            
            results.append({
                'player_id': player_id,
                'player_name': player_name,
                'minutes_played': minutes_played,
                'mean_goals_per_90': estimates['mean'],
                'std_goals_per_90': estimates['std'],
                'ci_lower': estimates['ci_lower'],
                'ci_upper': estimates['ci_upper']
            })
        except Exception as e:
            print(f"Error processing player {player_id}: {str(e)}")
            continue
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{output_dir}/player_estimates.csv", index=False)
    
    print("\nHierarchical Bayesian Analysis Results:")
    print(f"Results saved to {output_dir}/")
    print("\nTop 5 players by expected goals per 90:")
    print(results_df.sort_values('mean_goals_per_90', ascending=False).head())
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Mean goals per 90: {results_df['mean_goals_per_90'].mean():.4f}")
    print(f"Median goals per 90: {results_df['mean_goals_per_90'].median():.4f}")
    print(f"Standard deviation: {results_df['mean_goals_per_90'].std():.4f}")
    
    # Create performance report visualizations
    create_performance_report(results_df, output_dir)

def main():
    """Run the complete analysis."""
    # Load processed data
    data_path = "data/processed/training_data.csv"
    player_data = pd.read_csv(data_path)
    
    # Run analyses
    run_basic_analysis(player_data)
    run_hierarchical_analysis(player_data)

if __name__ == "__main__":
    main() 