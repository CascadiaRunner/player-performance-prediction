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
import matplotlib.pyplot as plt

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

def analyze_prediction_errors(test_results_df: pd.DataFrame, league: str) -> None:
    """
    Analyze cases where the model performs poorly and identify patterns.
    
    Args:
        test_results_df: DataFrame with test results
        league: Name of the league being analyzed
    """
    # Calculate absolute errors
    test_results_df['abs_error'] = abs(test_results_df['actual_goals_per_90'] - test_results_df['predicted_goals_per_90'])
    
    # Sort by absolute error to find worst predictions
    worst_predictions = test_results_df.sort_values('abs_error', ascending=False).head(10)
    
    print(f"\nWorst 10 predictions for {league}:")
    print(worst_predictions[['player_name', 'actual_goals_per_90', 'predicted_goals_per_90', 'abs_error']].to_string(index=False))
    
    # Analyze patterns in prediction errors
    print("\nError Analysis:")
    print(f"Mean absolute error: {test_results_df['abs_error'].mean():.4f}")
    print(f"Median absolute error: {test_results_df['abs_error'].median():.4f}")
    print(f"Standard deviation of errors: {test_results_df['abs_error'].std():.4f}")
    
    # Check if errors are biased (consistently over or under predicting)
    mean_error = (test_results_df['predicted_goals_per_90'] - test_results_df['actual_goals_per_90']).mean()
    print(f"Mean prediction bias: {mean_error:.4f} (positive means over-prediction)")
    
    # Analyze error distribution by actual goals
    test_results_df['goal_range'] = pd.cut(test_results_df['actual_goals_per_90'], 
                                         bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0, float('inf')],
                                         labels=['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0', '1.0+'])
    
    error_by_range = test_results_df.groupby('goal_range')['abs_error'].agg(['mean', 'count'])
    print("\nError distribution by actual goals per 90:")
    print(error_by_range.to_string())

def test_model(player_data: pd.DataFrame, output_dir: str = "results") -> None:
    """
    Test the model by using 2016 season as test data and all previous seasons as training data,
    evaluating performance separately for each league.
    
    Args:
        player_data: DataFrame with player performance data
        output_dir: Directory to save results
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Split data into training (pre-2016) and testing (2016) sets
    train_data = player_data[player_data['season'] < 2016].copy()
    test_data = player_data[player_data['season'] == 2016].copy()
    
    print(f"\nTraining data size: {len(train_data)} players")
    print(f"Test data size: {len(test_data)} players")
    
    # Get unique leagues
    leagues = test_data['league_name'].unique()
    league_metrics = []
    
    for league in leagues:
        print(f"\nTesting model for {league}")
        league_train = train_data[train_data['league_name'] == league]
        league_test = test_data[test_data['league_name'] == league]
        
        # Skip leagues with too few players
        if len(league_train) < 100 or len(league_test) < 20:
            print(f"Skipping {league} - insufficient data")
            continue
        
        # Prepare training data
        model_data = {
            'player_ids': league_train['player_api_id'].values,
            'goals': league_train['goals'].values,
            'minutes': league_train['minutes_played'].values
        }
        
        # Calculate league statistics for prior
        league_mean = league_train['goals_per_90'].mean()
        league_std = league_train['goals_per_90'].std()
        
        # Initialize and fit model
        model = HierarchicalPlayerPerformance(league_mean, league_std)
        model.fit(model_data)
        
        # Evaluate on test data
        test_results = []
        for _, row in league_test.iterrows():
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
        
        # Analyze prediction errors
        analyze_prediction_errors(test_results_df, league)
        
        # Save league-specific test results
        test_results_df.to_csv(f"{output_dir}/{league}_test_results.csv", index=False)
    
    # Create and save league comparison metrics
    metrics_df = pd.DataFrame(league_metrics)
    metrics_df = metrics_df.sort_values('r2', ascending=False)
    metrics_df.to_csv(f"{output_dir}/league_test_metrics.csv", index=False)
    
    print("\nLeague Performance Comparison (sorted by R²):")
    print(metrics_df[['league', 'r2', 'mse', 'mean_abs_error', 'num_test_players']].to_string(index=False))

def create_bayesian_visualizations(player_data: pd.DataFrame, output_dir: str = "results") -> None:
    """
    Create visualizations to explain the Bayesian inference process.
    
    Args:
        player_data: DataFrame with player performance data
        output_dir: Directory to save visualizations
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 1. League Prior Distribution
    plt.figure(figsize=(10, 6))
    league_means = player_data.groupby('league_name')['goals_per_90'].mean()
    league_stds = player_data.groupby('league_name')['goals_per_90'].std()
    
    plt.bar(league_means.index, league_means.values, yerr=league_stds.values, 
            capsize=5, alpha=0.7)
    plt.xticks(rotation=45, ha='right')
    plt.title('League-wide Goal Scoring Rates')
    plt.xlabel('League')
    plt.ylabel('Goals per 90 minutes')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/league_priors.png")
    plt.close()
    
    # 2. Position-specific Prior Adjustments
    positions = ['forward', 'midfielder', 'defender']
    multipliers = [1.5, 0.8, 0.4]
    
    plt.figure(figsize=(8, 6))
    plt.bar(positions, multipliers, alpha=0.7)
    plt.title('Position-specific Prior Adjustments')
    plt.xlabel('Position')
    plt.ylabel('Mean Multiplier')
    plt.axhline(y=1.0, color='r', linestyle='--', label='Base Rate')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/position_priors.png")
    plt.close()
    
    # 3. Player Development Factor Distribution
    plt.figure(figsize=(10, 6))
    development_factors = []
    for player_id in player_data['player_api_id'].unique():
        player_mask = player_data['player_api_id'] == player_id
        if player_data[player_mask]['minutes_played'].sum() >= 540:
            goals = player_data[player_mask]['goals'].sum()
            minutes = player_data[player_mask]['minutes_played'].sum()
            goals_per_90 = (goals * 90) / minutes
            development_factors.append(goals_per_90)
    
    plt.hist(development_factors, bins=30, alpha=0.7)
    plt.title('Distribution of Player Development Factors')
    plt.xlabel('Goals per 90 minutes')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/development_factors.png")
    plt.close()
    
    # 4. Model Performance by League
    plt.figure(figsize=(12, 6))
    leagues = ['Spain LIGA BBVA', 'France Ligue 1', 'Italy Serie A', 
              'Netherlands Eredivisie', 'Scotland Premier League',
              'Portugal Liga ZON Sagres', 'England Premier League',
              'Germany 1. Bundesliga', 'Belgium Jupiler League',
              'Switzerland Super League', 'Poland Ekstraklasa']
    r2_scores = [0.4796, 0.2885, 0.2283, 0.1629, 0.1564, 0.1500,
                0.0955, 0.0766, -0.0330, -0.0924, -0.1356]
    
    plt.bar(leagues, r2_scores, alpha=0.7)
    plt.xticks(rotation=45, ha='right')
    plt.title('Model Performance by League (R² Score)')
    plt.xlabel('League')
    plt.ylabel('R² Score')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/league_performance.png")
    plt.close()
    
    # 5. Error Distribution
    plt.figure(figsize=(10, 6))
    mae_scores = [0.3620, 0.3721, 0.3481, 0.4711, 0.3619, 0.4127,
                 0.3888, 0.3519, 0.3694, 0.5259, 0.4871]
    
    plt.bar(leagues, mae_scores, alpha=0.7)
    plt.xticks(rotation=45, ha='right')
    plt.title('Mean Absolute Error by League')
    plt.xlabel('League')
    plt.ylabel('Mean Absolute Error')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/error_distribution.png")
    plt.close()

def main():
    """
    Main function to run the analysis.
    """
    # Load and preprocess data
    data_path = "data/processed/training_data.csv"
    player_data = pd.read_csv(data_path)
    
    # Create visualizations
    create_bayesian_visualizations(player_data)
    
    # Run the analysis
    test_model(player_data)

if __name__ == "__main__":
    main() 