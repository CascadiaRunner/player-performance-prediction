"""
Script to demonstrate Bayesian player performance analysis.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from .player_performance import PlayerPerformance, HierarchicalPlayerPerformance
from src.visualization.plot_performance import create_performance_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

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

def run_hierarchical_analysis(player_data: pd.DataFrame, output_dir: str = "results") -> None:
    """
    Run hierarchical Bayesian analysis using MCMC, grouped by league.
    
    Args:
        player_data: DataFrame with player performance data
        output_dir: Directory to save results
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Group data by league
    leagues = player_data['league_name'].unique()
    league_results = {}
    
    for league in leagues:
        league_data = player_data[player_data['league_name'] == league]
        print(f"\nProcessing league: {league}")
        
        # Prepare data for hierarchical model
        model_data = {
            'player_ids': league_data['player_api_id'].values,
            'goals': league_data['goals'].values,
            'minutes': league_data['minutes_played'].values
        }
        
        # Calculate league statistics for prior
        league_mean = league_data['goals_per_90'].mean()
        league_std = league_data['goals_per_90'].std()
        
        # Initialize and fit model
        model = HierarchicalPlayerPerformance(league_mean, league_std)
        model.fit(model_data)
        
        # Get qualified players (those with sufficient minutes)
        qualified_players = model.get_qualified_players()
        print(f"Players with sufficient minutes (≥{model.min_minutes}): {len(qualified_players)}")
        print(f"Total players in dataset: {len(league_data['player_api_id'].unique())}")
        
        # Get estimates for each qualified player
        results = []
        for player_id in qualified_players:
            try:
                estimates = model.get_player_estimates(player_id)
                player_name = league_data[league_data['player_api_id'] == player_id]['player_name'].iloc[0]
                minutes_played = league_data[league_data['player_api_id'] == player_id]['minutes_played'].sum()
                
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
        
        # Save league-specific results
        results_df = pd.DataFrame(results)
        league_results[league] = results_df
        results_df.to_csv(f"{output_dir}/{league}_player_estimates.csv", index=False)
        
        print(f"\nHierarchical Bayesian Analysis Results for {league}:")
        print(f"Results saved to {output_dir}/{league}_player_estimates.csv")
        print("\nTop 5 players by expected goals per 90:")
        print(results_df.sort_values('mean_goals_per_90', ascending=False).head())
        
        # Print summary statistics
        print("\nSummary Statistics:")
        print(f"Mean goals per 90: {results_df['mean_goals_per_90'].mean():.4f}")
        print(f"Median goals per 90: {results_df['mean_goals_per_90'].median():.4f}")
        print(f"Standard deviation: {results_df['mean_goals_per_90'].std():.4f}")
        
        # Create performance report visualizations
        create_performance_report(results_df, f"{output_dir}/{league}_report")
    
    # Compare leagues
    league_comparison = pd.DataFrame({
        'league': leagues,
        'mean_goals_per_90': [league_results[league]['mean_goals_per_90'].mean() for league in leagues],
        'median_goals_per_90': [league_results[league]['mean_goals_per_90'].median() for league in leagues],
        'std_goals_per_90': [league_results[league]['mean_goals_per_90'].std() for league in leagues]
    })
    league_comparison.to_csv(f"{output_dir}/league_comparison.csv", index=False)
    print("\nLeague Comparison:")
    print(league_comparison)

def test_model(player_data: pd.DataFrame, output_dir: str = "results") -> None:
    """
    Test the model by splitting data into training and testing sets, fitting the model on training data,
    and evaluating it on testing data, separately for each league.
    
    Args:
        player_data: DataFrame with player performance data
        output_dir: Directory to save results
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get unique leagues
    leagues = player_data['league_name'].unique()
    league_metrics = []
    
    for league in leagues:
        print(f"\nTesting model for {league}")
        league_data = player_data[player_data['league_name'] == league]
        
        # Skip leagues with too few players
        if len(league_data) < 100:
            print(f"Skipping {league} - insufficient data")
            continue
        
        # Split data into training and testing sets
        train_data, test_data = train_test_split(league_data, test_size=0.2, random_state=42)
        
        # Prepare training data
        model_data = {
            'player_ids': train_data['player_api_id'].values,
            'goals': train_data['goals'].values,
            'minutes': train_data['minutes_played'].values
        }
        
        # Calculate league statistics for prior
        league_mean = train_data['goals_per_90'].mean()
        league_std = train_data['goals_per_90'].std()
        
        # Initialize and fit model
        model = HierarchicalPlayerPerformance(league_mean, league_std)
        model.fit(model_data)
        
        # Evaluate on test data
        test_results = []
        for _, row in test_data.iterrows():
            player_id = row['player_api_id']
            if player_id in model.get_qualified_players():
                estimates = model.get_player_estimates(player_id)
                test_results.append({
                    'player_id': player_id,
                    'player_name': row['player_name'],
                    'actual_goals_per_90': (row['goals'] * 90) / row['minutes_played'],
                    'predicted_goals_per_90': estimates['mean']
                })
        
        if not test_results:
            print(f"No qualified players in test set for {league}")
            continue
            
        test_results_df = pd.DataFrame(test_results)
        
        # Calculate metrics
        mse = mean_squared_error(test_results_df['actual_goals_per_90'], test_results_df['predicted_goals_per_90'])
        r2 = r2_score(test_results_df['actual_goals_per_90'], test_results_df['predicted_goals_per_90'])
        
        # Calculate additional metrics
        mean_abs_error = np.mean(np.abs(test_results_df['actual_goals_per_90'] - test_results_df['predicted_goals_per_90']))
        median_abs_error = np.median(np.abs(test_results_df['actual_goals_per_90'] - test_results_df['predicted_goals_per_90']))
        
        # Store league metrics
        league_metrics.append({
            'league': league,
            'mse': mse,
            'r2': r2,
            'mean_abs_error': mean_abs_error,
            'median_abs_error': median_abs_error,
            'num_test_players': len(test_results_df),
            'mean_actual_goals': test_results_df['actual_goals_per_90'].mean(),
            'std_actual_goals': test_results_df['actual_goals_per_90'].std()
        })
        
        print(f"Number of test players: {len(test_results_df)}")
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"R² Score: {r2:.4f}")
        print(f"Mean Absolute Error: {mean_abs_error:.4f}")
        print(f"Median Absolute Error: {median_abs_error:.4f}")
        
        # Save league-specific test results
        test_results_df.to_csv(f"{output_dir}/{league}_test_results.csv", index=False)
    
    # Create and save league comparison metrics
    metrics_df = pd.DataFrame(league_metrics)
    metrics_df = metrics_df.sort_values('r2', ascending=False)
    metrics_df.to_csv(f"{output_dir}/league_test_metrics.csv", index=False)
    
    print("\nLeague Performance Comparison (sorted by R²):")
    print(metrics_df[['league', 'r2', 'mse', 'mean_abs_error', 'num_test_players']].to_string(index=False))

def main():
    """Run the complete analysis."""
    # Load processed data
    data_path = "data/processed/training_data.csv"
    player_data = pd.read_csv(data_path)
    
    # Run analyses
    run_basic_analysis(player_data)
    run_hierarchical_analysis(player_data)
    test_model(player_data)

if __name__ == "__main__":
    main() 