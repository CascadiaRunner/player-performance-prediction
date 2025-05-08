"""
Visualization functions for player performance analysis.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Optional, List, Dict
from pathlib import Path

def plot_player_ranking(results_df: pd.DataFrame, 
                       top_n: int = 20,
                       save_path: Optional[str] = None) -> None:
    """
    Plot top N players ranked by their expected goals per 90, with credible intervals.
    
    Args:
        results_df: DataFrame with player estimates
        top_n: Number of top players to show
        save_path: Optional path to save the plot
    """
    # Sort and get top N players
    top_players = results_df.sort_values('mean_goals_per_90', ascending=False).head(top_n)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Calculate error bars, ensuring they're non-negative
    lower_error = np.maximum(0, top_players['mean_goals_per_90'] - top_players['ci_lower'])
    upper_error = np.maximum(0, top_players['ci_upper'] - top_players['mean_goals_per_90'])
    
    # Plot error bars (credible intervals)
    plt.errorbar(top_players['mean_goals_per_90'], 
                range(len(top_players)),
                xerr=[lower_error, upper_error],
                fmt='o',
                capsize=5,
                label='95% Credible Interval')
    
    # Add player names
    plt.yticks(range(len(top_players)), top_players['player_name'])
    
    # Customize plot
    plt.xlabel('Expected Goals per 90 Minutes')
    plt.title(f'Top {top_n} Players by Expected Goals per 90')
    plt.grid(True, axis='x', linestyle='--', alpha=0.7)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def plot_minutes_vs_performance(results_df: pd.DataFrame,
                              save_path: Optional[str] = None) -> None:
    """
    Plot relationship between minutes played and performance estimates.
    
    Args:
        results_df: DataFrame with player estimates
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    # Create scatter plot
    scatter = plt.scatter(results_df['minutes_played'],
                         results_df['mean_goals_per_90'],
                         alpha=0.6,
                         c=results_df['std_goals_per_90'],
                         cmap='viridis')
    
    # Add colorbar
    plt.colorbar(scatter, label='Uncertainty (Std Dev)')
    
    # Add trend line
    z = np.polyfit(results_df['minutes_played'], results_df['mean_goals_per_90'], 1)
    p = np.poly1d(z)
    plt.plot(results_df['minutes_played'], 
             p(results_df['minutes_played']), 
             "r--", 
             alpha=0.8,
             label='Trend Line')
    
    # Customize plot
    plt.xlabel('Minutes Played')
    plt.ylabel('Expected Goals per 90')
    plt.title('Relationship Between Playing Time and Performance')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def plot_uncertainty_distribution(results_df: pd.DataFrame,
                                save_path: Optional[str] = None) -> None:
    """
    Plot distribution of uncertainty in player estimates.
    
    Args:
        results_df: DataFrame with player estimates
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    # Create histogram of standard deviations
    sns.histplot(data=results_df, x='std_goals_per_90', bins=30)
    
    # Add vertical line for mean uncertainty
    mean_uncertainty = results_df['std_goals_per_90'].mean()
    plt.axvline(mean_uncertainty, color='r', linestyle='--', 
                label=f'Mean Uncertainty: {mean_uncertainty:.4f}')
    
    # Customize plot
    plt.xlabel('Uncertainty in Estimates (Standard Deviation)')
    plt.ylabel('Number of Players')
    plt.title('Distribution of Uncertainty in Player Performance Estimates')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def plot_performance_comparison(results_df: pd.DataFrame,
                              player_ids: List[int],
                              save_path: Optional[str] = None) -> None:
    """
    Compare performance estimates for specific players.
    
    Args:
        results_df: DataFrame with player estimates
        player_ids: List of player IDs to compare
        save_path: Optional path to save the plot
    """
    # Filter for selected players
    selected_players = results_df[results_df['player_id'].isin(player_ids)]
    
    plt.figure(figsize=(12, 6))
    
    # Create grouped bar plot
    x = np.arange(len(selected_players))
    width = 0.35
    
    plt.bar(x, selected_players['mean_goals_per_90'], width,
            yerr=selected_players['std_goals_per_90'],
            capsize=5,
            label='Expected Goals per 90')
    
    # Customize plot
    plt.xlabel('Players')
    plt.ylabel('Expected Goals per 90')
    plt.title('Performance Comparison of Selected Players')
    plt.xticks(x, selected_players['player_name'], rotation=45, ha='right')
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def create_performance_report(results_df: pd.DataFrame,
                            output_dir: str = "results") -> None:
    """
    Create a comprehensive performance report with multiple visualizations.
    
    Args:
        results_df: DataFrame with player estimates
        output_dir: Directory to save visualizations
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate all plots
    plot_player_ranking(results_df, save_path=f"{output_dir}/top_players.png")
    plot_minutes_vs_performance(results_df, save_path=f"{output_dir}/minutes_vs_performance.png")
    plot_uncertainty_distribution(results_df, save_path=f"{output_dir}/uncertainty_distribution.png")
    
    # If we have enough players, create a comparison plot for top 5
    if len(results_df) >= 5:
        top_5_ids = results_df.nlargest(5, 'mean_goals_per_90')['player_id'].tolist()
        plot_performance_comparison(results_df, top_5_ids, 
                                  save_path=f"{output_dir}/top_5_comparison.png")
    
    print(f"\nPerformance report visualizations saved to {output_dir}/")

def plot_prior_posterior(prior_alpha: float, prior_beta: float,
                        posterior_alpha: float, posterior_beta: float,
                        save_path: str = None) -> None:
    """
    Plot prior and posterior distributions.
    
    Args:
        prior_alpha: Alpha parameter for prior
        prior_beta: Beta parameter for prior
        posterior_alpha: Alpha parameter for posterior
        posterior_beta: Beta parameter for posterior
        save_path: Optional path to save the plot
    """
    # TODO: Implement prior-posterior plot
    pass

def plot_performance_trajectory(player_data: dict, save_path: str = None) -> None:
    """
    Plot player performance trajectory over time.
    
    Args:
        player_data: Dictionary containing player performance data
        save_path: Optional path to save the plot
    """
    # TODO: Implement performance trajectory plot
    pass 